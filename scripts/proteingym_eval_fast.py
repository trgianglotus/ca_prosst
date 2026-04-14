"""Vectorized + resumable CA-ProSST ProteinGym evaluation.

Two improvements over proteingym_eval.py:
    1. Vectorized mutant scoring: ~500k .item() calls → one GPU op.
       On large assays like HIS7_YEAST (~496k mutants) this is 10-100x faster.
    2. Resume support: on restart, skip proteins whose prediction CSV already
       exists and reload their cached metrics from per_protein.csv. Safe to
       kill and re-run at any time — the only work lost is the in-flight
       protein.

Output format is identical to proteingym_eval.py (same summary.json schema).
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


def ndcg_score(y_true: np.ndarray, y_score: np.ndarray, k: Optional[int] = None) -> float:
    order = np.argsort(-y_score)
    y = y_true[order]
    if k is not None:
        y = y[:k]
    y = y - y.min() + 1e-9
    gains = (2 ** y - 1) / np.log2(np.arange(2, len(y) + 2))
    dcg = gains.sum()
    ideal = np.sort(y_true)[::-1]
    if k is not None:
        ideal = ideal[:k]
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


def vectorized_mutant_scores(
    mutants: list[str], logits: torch.Tensor, vocab: dict, device: torch.device
) -> np.ndarray:
    """Score N mutants in a few tensor ops.

    logits: [L, V] (already log-softmaxed, with CLS/EOS stripped)
    Returns: np.ndarray of length N with per-mutant summed delta log-probs.
    """
    positions: list[int] = []
    wt_ids: list[int] = []
    mt_ids: list[int] = []
    segment_ids: list[int] = []
    for i, mutant in enumerate(mutants):
        for sub in mutant.split(":"):
            positions.append(int(sub[1:-1]) - 1)
            wt_ids.append(vocab[sub[0]])
            mt_ids.append(vocab[sub[-1]])
            segment_ids.append(i)

    pos = torch.tensor(positions, dtype=torch.long, device=device)
    wt = torch.tensor(wt_ids, dtype=torch.long, device=device)
    mt = torch.tensor(mt_ids, dtype=torch.long, device=device)
    seg = torch.tensor(segment_ids, dtype=torch.long, device=device)

    # logits: [L, V]. Index with fancy indexing.
    deltas = logits[pos, mt] - logits[pos, wt]  # [total_subs]
    scores = torch.zeros(len(mutants), dtype=deltas.dtype, device=device)
    scores.index_add_(0, seg, deltas)
    return scores.detach().cpu().numpy()


@torch.no_grad()
def score_protein(
    model: CAProSSTForMaskedLM,
    tokenizer,
    name: str,
    residue_dir: Path,
    structure_dir: Path,
    mutant_dir: Path,
    plddt_dir: Optional[Path],
    device: torch.device,
) -> Optional[tuple]:
    residue_fa = residue_dir / f"{name}.fasta"
    structure_fa = structure_dir / f"{name}.fasta"
    mutant_csv = mutant_dir / f"{name}.csv"
    if not (residue_fa.exists() and structure_fa.exists() and mutant_csv.exists()):
        return None

    sequence = read_seq(residue_fa)
    structure_sequence = [int(i) for i in read_seq(structure_fa).split(",")]
    ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)

    tokenized = tokenizer([sequence], return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    plddt = None
    mean_plddt: Optional[float] = None
    if plddt_dir is not None:
        p_path = plddt_dir / f"{name}.npy"
        plddt = build_plddt_tensor(p_path, len(sequence), device)
        if plddt is not None:
            interior = plddt[0, 1:-1].detach().cpu().numpy()
            nonzero = interior[interior > 0]
            mean_plddt = float(nonzero.mean()) if nonzero.size > 0 else 0.0

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=ss_input_ids,
        plddt=plddt,
        labels=input_ids,
    )
    # Strip CLS/EOS -> [L, V]
    logits = torch.log_softmax(outputs.logits[:, 1:-1, :], dim=-1)[0]

    df = pd.read_csv(mutant_csv)
    vocab = tokenizer.get_vocab()
    scores = vectorized_mutant_scores(df["mutant"].tolist(), logits, vocab, device)
    df["ca_prosst_score"] = scores

    y_true = df["DMS_score"].to_numpy()
    y_pred = scores
    result = {
        "name": name,
        "n_mutants": len(df),
        "spearman": float(spearmanr(y_true, y_pred).correlation),
        "ndcg": ndcg_score(y_true, y_pred),
        "top_recall_10pct": top_recall(y_true, y_pred, 0.1),
        "mean_plddt": mean_plddt,
    }
    return result, df[["mutant", "DMS_score", "ca_prosst_score"]]


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="AI4Protein/ProSST-2048")
    ap.add_argument("--residue_dir", required=True, type=Path)
    ap.add_argument("--structure_dir", required=True, type=Path)
    ap.add_argument("--mutant_dir", required=True, type=Path)
    ap.add_argument("--plddt_dir", type=Path, default=None)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--ca_mode", choices=["none", "hard", "soft", "gate"], default="none")
    ap.add_argument("--ca_threshold", type=float, default=70.0)
    ap.add_argument("--gate_checkpoint", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    if args.ca_mode != "none" and args.plddt_dir is None:
        print("--plddt_dir required for non-'none' ca_mode", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model_path} (ca_mode={args.ca_mode})  [FAST]")
    config = CAProSSTConfig.from_pretrained(args.model_path)
    config.ca_mode = args.ca_mode
    config.ca_threshold = args.ca_threshold
    model = CAProSSTForMaskedLM.from_pretrained(
        args.model_path, config=config
    ).to(device).eval()

    if args.gate_checkpoint is not None and model.ca_gate is not None:
        sd = torch.load(args.gate_checkpoint, map_location=device)
        model.ca_gate.load_state_dict(sd)
        print(f"loaded gate checkpoint: {args.gate_checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    names = sorted(p.stem for p in args.residue_dir.glob("*.fasta"))
    if args.limit:
        names = names[: args.limit]

    pred_dir = args.out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    # --- resume: load cached rows for already-scored proteins ---
    done = {p.stem for p in pred_dir.glob("*.csv")}
    rows: list[dict] = []
    existing_csv = args.out_dir / "per_protein.csv"
    cached_from_csv: set = set()
    if existing_csv.exists() and done:
        old = pd.read_csv(existing_csv)
        for _, r in old.iterrows():
            if r["name"] in done:
                rows.append(r.to_dict())
                cached_from_csv.add(r["name"])
    # For prediction CSVs with no per_protein.csv entry, recompute metrics.
    missing = done - cached_from_csv
    for name in sorted(missing):
        try:
            pdf = pd.read_csv(pred_dir / f"{name}.csv")
            if "DMS_score" not in pdf or "ca_prosst_score" not in pdf:
                continue
            y_true = pdf["DMS_score"].to_numpy()
            y_pred = pdf["ca_prosst_score"].to_numpy()
            p_path = (args.plddt_dir / f"{name}.npy") if args.plddt_dir else None
            mean_plddt = None
            if p_path is not None and p_path.exists():
                arr = np.load(p_path).astype(np.float32)
                nonzero = arr[arr > 0]
                mean_plddt = float(nonzero.mean()) if nonzero.size > 0 else 0.0
            rows.append({
                "name": name,
                "n_mutants": len(pdf),
                "spearman": float(spearmanr(y_true, y_pred).correlation),
                "ndcg": ndcg_score(y_true, y_pred),
                "top_recall_10pct": top_recall(y_true, y_pred, 0.1),
                "mean_plddt": mean_plddt,
            })
        except Exception as e:
            print(f"[resume] failed to recompute {name}: {e}")
    cached_names = {r["name"] for r in rows}
    if rows:
        print(f"[resume] {len(rows)} proteins already scored — skipping")

    for name in tqdm(names, desc="scoring"):
        if name in cached_names:
            continue
        res = score_protein(
            model, tokenizer, name,
            args.residue_dir, args.structure_dir, args.mutant_dir, args.plddt_dir, device,
        )
        if res is None:
            continue
        row, pred_df = res
        rows.append(row)
        pred_df.to_csv(pred_dir / f"{name}.csv", index=False)
        # Write incremental per_protein.csv so we can resume after crash.
        pd.DataFrame(rows).to_csv(args.out_dir / "per_protein.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "per_protein.csv", index=False)

    sp = df["spearman"].to_numpy()
    lo, hi = bootstrap_ci(sp, n=args.bootstrap)
    summary = {
        "ca_mode": args.ca_mode,
        "ca_threshold": args.ca_threshold,
        "n_proteins": int(len(df)),
        "mean_spearman": float(np.nanmean(sp)),
        "median_spearman": float(np.nanmedian(sp)),
        "spearman_95ci": [lo, hi],
        "mean_ndcg": float(np.nanmean(df["ndcg"])),
        "mean_top_recall_10pct": float(np.nanmean(df["top_recall_10pct"])),
        "stratified_by_plddt": stratified(df),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
