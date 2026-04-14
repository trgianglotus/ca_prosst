"""CALM: Confidence-Aware Log-likelihood Mixture for ProSST zero-shot scoring.

For each protein we run two forward passes:
    1. Full ProSST (structure on): yields per-residue log-probs L_struct
    2. Structure-ablated ProSST (ss_embeddings zeroed): yields L_seq

Per-substitution score is a per-position pLDDT-gated mix:
    w_i  = sigmoid((pLDDT_i - p0) / T)
    s_i  = w_i * (L_struct[i, mt] - L_struct[i, wt])
         + (1 - w_i) * (L_seq[i, mt] - L_seq[i, wt])

Multi-substitution mutants sum over i. The two scalars (p0, T) are tuned on
a held-out subset of assays (5-fold CV by default) and evaluated on the
complement. Also reports the "oracle" (p0, T) tuned on all 217 assays for
ceiling analysis.

Output is fully compatible with proteingym_eval.py summary.json layout, plus
a calm_hparams.json with the tuned (p0, T), per-fold choices, and grid
search results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from scipy.stats import spearmanr
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ca_prosst.models import CAProSSTConfig, CAProSSTForMaskedLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def read_seq(path: Path) -> str:
    for rec in SeqIO.parse(str(path), "fasta"):
        return str(rec.seq)
    raise ValueError(f"empty fasta: {path}")


def tokenize_structure_sequence(structure_sequence: list[int]) -> torch.Tensor:
    shifted = [i + 3 for i in structure_sequence]
    shifted = [1, *shifted, 2]
    return torch.tensor([shifted], dtype=torch.long)


def build_plddt_tensor(
    plddt_path: Optional[Path], seq_len: int, device: torch.device
) -> Optional[torch.Tensor]:
    if plddt_path is None or not plddt_path.exists():
        return None
    arr = np.load(plddt_path).astype(np.float32)
    if arr.shape[0] != seq_len:
        if arr.shape[0] < seq_len:
            pad = np.zeros(seq_len - arr.shape[0], dtype=np.float32)
            arr = np.concatenate([arr, pad])
        else:
            arr = arr[:seq_len]
    padded = np.concatenate([[100.0], arr, [100.0]]).astype(np.float32)
    return torch.from_numpy(padded).unsqueeze(0).to(device)


def ndcg_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order]
    y = y - y.min() + 1e-9
    gains = (2 ** y - 1) / np.log2(np.arange(2, len(y) + 2))
    dcg = gains.sum()
    ideal = np.sort(y_true)[::-1]
    ideal = ideal - ideal.min() + 1e-9
    ig = (2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))
    idcg = ig.sum()
    return float(dcg / idcg) if idcg > 0 else float("nan")


def top_recall(y_true: np.ndarray, y_score: np.ndarray, frac: float = 0.1) -> float:
    n = len(y_true)
    k = max(1, int(np.ceil(n * frac)))
    true_top = set(np.argsort(-y_true)[:k].tolist())
    pred_top = set(np.argsort(-y_score)[:k].tolist())
    return len(true_top & pred_top) / k


def bootstrap_ci(values: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stratified(df: pd.DataFrame, col: str = "mean_plddt") -> dict:
    d = df.dropna(subset=[col])
    if d.empty:
        return {}
    quartiles = np.quantile(d[col], [0.0, 0.25, 0.5, 0.75, 1.0])
    out = {}
    for i in range(4):
        lo, hi = quartiles[i], quartiles[i + 1]
        if i < 3:
            q = d[(d[col] >= lo) & (d[col] < hi)]
        else:
            q = d[(d[col] >= lo) & (d[col] <= hi)]
        out[f"Q{i + 1}"] = {
            "lo": float(lo),
            "hi": float(hi),
            "n": int(len(q)),
            "mean_spearman": float(q["spearman"].mean()) if len(q) else float("nan"),
        }
    return out


@torch.no_grad()
def compute_protein_cache(
    model: CAProSSTForMaskedLM,
    tokenizer,
    name: str,
    residue_dir: Path,
    structure_dir: Path,
    mutant_dir: Path,
    plddt_dir: Path,
    device: torch.device,
) -> Optional[dict]:
    """Run both forward passes and cache everything needed for CALM scoring.

    Returns a dict with per-substitution arrays (positions, wt, mt, seg_id,
    plddt_at_pos), struct_deltas, seq_deltas, and DMS scores.
    """
    residue_fa = residue_dir / f"{name}.fasta"
    structure_fa = structure_dir / f"{name}.fasta"
    mutant_csv = mutant_dir / f"{name}.csv"
    plddt_path = plddt_dir / f"{name}.npy"
    if not (residue_fa.exists() and structure_fa.exists() and mutant_csv.exists()
            and plddt_path.exists()):
        return None

    sequence = read_seq(residue_fa)
    structure_sequence = [int(i) for i in read_seq(structure_fa).split(",")]
    ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)

    tokenized = tokenizer([sequence], return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    plddt_tensor = build_plddt_tensor(plddt_path, len(sequence), device)
    plddt_arr = plddt_tensor[0, 1:-1].detach().cpu().numpy()  # [L], residue-aligned
    nonzero = plddt_arr[plddt_arr > 0]
    mean_plddt = float(nonzero.mean()) if nonzero.size > 0 else 0.0

    # Pass 1: structure ON
    model.ca_mode = "none"
    out_struct = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=ss_input_ids,
        plddt=None,
    )
    L_struct = torch.log_softmax(out_struct.logits[:, 1:-1, :], dim=-1)[0]  # [L, V]

    # Pass 2: structure OFF (sequence-only ablation)
    model.ca_mode = "zero"
    out_zero = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=ss_input_ids,
        plddt=None,
    )
    L_seq = torch.log_softmax(out_zero.logits[:, 1:-1, :], dim=-1)[0]

    # Parse all substitutions vectorized
    df = pd.read_csv(mutant_csv)
    vocab = tokenizer.get_vocab()
    positions: list[int] = []
    wt_ids: list[int] = []
    mt_ids: list[int] = []
    segment_ids: list[int] = []
    for i, mutant in enumerate(df["mutant"].tolist()):
        for sub in mutant.split(":"):
            positions.append(int(sub[1:-1]) - 1)
            wt_ids.append(vocab[sub[0]])
            mt_ids.append(vocab[sub[-1]])
            segment_ids.append(i)

    pos = torch.tensor(positions, dtype=torch.long, device=device)
    wt = torch.tensor(wt_ids, dtype=torch.long, device=device)
    mt = torch.tensor(mt_ids, dtype=torch.long, device=device)

    d_struct = (L_struct[pos, mt] - L_struct[pos, wt]).detach().cpu().numpy()
    d_seq = (L_seq[pos, mt] - L_seq[pos, wt]).detach().cpu().numpy()
    plddt_at_pos = plddt_arr[np.minimum(np.array(positions), len(plddt_arr) - 1)]

    return {
        "name": name,
        "n_mutants": len(df),
        "mean_plddt": mean_plddt,
        "segment_ids": np.array(segment_ids, dtype=np.int64),
        "plddt_at_pos": plddt_at_pos.astype(np.float32),
        "d_struct": d_struct.astype(np.float32),
        "d_seq": d_seq.astype(np.float32),
        "y_true": df["DMS_score"].to_numpy().astype(np.float32),
        "mutants": df["mutant"].tolist(),
    }


def score_with_hparams(cache: dict, p0: float, T: float) -> tuple[float, np.ndarray]:
    """Mix structure/seq scores per substitution, sum per mutant, compute Spearman."""
    w = 1.0 / (1.0 + np.exp(-(cache["plddt_at_pos"] - p0) / T))
    mixed = w * cache["d_struct"] + (1.0 - w) * cache["d_seq"]
    n = int(cache["segment_ids"].max()) + 1
    scores = np.zeros(n, dtype=np.float32)
    np.add.at(scores, cache["segment_ids"], mixed)
    y_true = cache["y_true"]
    rho = spearmanr(y_true, scores).correlation
    return float(rho if rho is not None else float("nan")), scores


def tune_hparams(caches: list[dict], p0_grid, T_grid) -> tuple[float, float, np.ndarray]:
    """Grid search (p0, T) to maximize mean Spearman. Returns (p0*, T*, grid)."""
    grid = np.zeros((len(p0_grid), len(T_grid)), dtype=np.float32)
    for i, p0 in enumerate(p0_grid):
        for j, T in enumerate(T_grid):
            rhos = []
            for c in caches:
                r, _ = score_with_hparams(c, p0, T)
                if not np.isnan(r):
                    rhos.append(r)
            grid[i, j] = np.mean(rhos) if rhos else float("nan")
    i, j = np.unravel_index(np.nanargmax(grid), grid.shape)
    return float(p0_grid[i]), float(T_grid[j]), grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="AI4Protein/ProSST-2048")
    ap.add_argument("--residue_dir", required=True, type=Path)
    ap.add_argument("--structure_dir", required=True, type=Path)
    ap.add_argument("--mutant_dir", required=True, type=Path)
    ap.add_argument("--plddt_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--n_folds", type=int, default=5, help="CV folds for (p0, T) tuning")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model_path} (CALM)")
    config = CAProSSTConfig.from_pretrained(args.model_path)
    config.ca_mode = "none"  # will be swapped per forward pass
    model = CAProSSTForMaskedLM.from_pretrained(
        args.model_path, config=config
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    names = sorted(p.stem for p in args.residue_dir.glob("*.fasta"))
    if args.limit:
        names = names[: args.limit]

    # --- 1. Compute per-protein caches (two forward passes each) ---
    cache_dir = args.out_dir / "caches"
    cache_dir.mkdir(exist_ok=True)
    caches: list[dict] = []
    for name in tqdm(names, desc="computing caches"):
        cache_path = cache_dir / f"{name}.npz"
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=True)
            cache = {k: z[k].item() if z[k].ndim == 0 else z[k] for k in z.files}
            # also pull mutants list back
            cache["mutants"] = list(z["mutants"])
            caches.append(cache)
            continue
        c = compute_protein_cache(
            model, tokenizer, name,
            args.residue_dir, args.structure_dir, args.mutant_dir, args.plddt_dir, device,
        )
        if c is None:
            continue
        np.savez_compressed(
            cache_path,
            name=c["name"], n_mutants=c["n_mutants"], mean_plddt=c["mean_plddt"],
            segment_ids=c["segment_ids"], plddt_at_pos=c["plddt_at_pos"],
            d_struct=c["d_struct"], d_seq=c["d_seq"], y_true=c["y_true"],
            mutants=np.array(c["mutants"]),
        )
        caches.append(c)
    print(f"cached {len(caches)} proteins")

    # --- 2. Hyperparameter grid ---
    p0_grid = np.arange(40.0, 96.0, 2.0)   # 28 values
    T_grid = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])

    # --- 3. Oracle (tune on all, ceiling) ---
    p0_o, T_o, grid_o = tune_hparams(caches, p0_grid, T_grid)
    oracle_rhos = []
    for c in caches:
        r, _ = score_with_hparams(c, p0_o, T_o)
        oracle_rhos.append(r)
    oracle_mean = float(np.nanmean(oracle_rhos))
    print(f"[oracle] p0={p0_o:.1f} T={T_o:.1f}  mean Spearman={oracle_mean:.4f}")

    # --- 4. K-fold CV (honest eval) ---
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(caches))
    folds = np.array_split(idx, args.n_folds)
    per_protein_rows = [None] * len(caches)
    fold_choices = []
    for k in range(args.n_folds):
        test_idx = set(folds[k].tolist())
        train_caches = [caches[i] for i in range(len(caches)) if i not in test_idx]
        p0_k, T_k, _ = tune_hparams(train_caches, p0_grid, T_grid)
        fold_choices.append({"fold": k, "p0": p0_k, "T": T_k, "n_train": len(train_caches), "n_test": len(folds[k])})
        for i in folds[k]:
            c = caches[i]
            rho, scores = score_with_hparams(c, p0_k, T_k)
            per_protein_rows[i] = {
                "name": c["name"],
                "n_mutants": int(c["n_mutants"]),
                "spearman": rho,
                "ndcg": ndcg_score(c["y_true"], scores),
                "top_recall_10pct": top_recall(c["y_true"], scores, 0.1),
                "mean_plddt": float(c["mean_plddt"]),
                "fold": k,
                "p0": p0_k,
                "T": T_k,
            }

    df = pd.DataFrame(per_protein_rows)
    df.to_csv(args.out_dir / "per_protein.csv", index=False)
    sp = df["spearman"].to_numpy()
    lo, hi = bootstrap_ci(sp, n=args.bootstrap)

    summary = {
        "method": "CALM",
        "n_proteins": int(len(df)),
        "cv_mean_spearman": float(np.nanmean(sp)),
        "cv_median_spearman": float(np.nanmedian(sp)),
        "cv_spearman_95ci": [lo, hi],
        "cv_mean_ndcg": float(np.nanmean(df["ndcg"])),
        "cv_mean_top_recall_10pct": float(np.nanmean(df["top_recall_10pct"])),
        "stratified_by_plddt": stratified(df),
        "oracle_p0": p0_o,
        "oracle_T": T_o,
        "oracle_mean_spearman": oracle_mean,
        "fold_choices": fold_choices,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Also dump the oracle grid for figures
    np.save(args.out_dir / "oracle_grid.npy", grid_o)
    np.save(args.out_dir / "p0_grid.npy", p0_grid)
    np.save(args.out_dir / "T_grid.npy", T_grid)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
