"""CA-ProSST zero-shot ProteinGym evaluation harness.

Differences from ProSST's `zero_shot/proteingym_benchmark.py`:
    - Uses the local vendored CAProSSTForMaskedLM (not HF trust_remote_code).
    - Accepts `--ca_mode {none,hard,soft,gate}` and loads pLDDT per protein.
    - Writes scores to a separate output dir instead of mutating the input
      substitutions CSVs — safe to rerun with different configs.
    - Reports aggregated metrics (mean Spearman, NDCG, top-recall) across all
      assays, pLDDT-quartile stratified Spearman, and bootstrap 95% CI.

Expected layout:
    residue_dir/<name>.fasta        # wildtype sequence
    structure_dir/<name>.fasta      # comma-separated quantized structure ids
    mutant_dir/<name>.csv           # must contain columns: mutant, DMS_score
    plddt_dir/<name>.npy            # optional; required for non-'none' ca_mode
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

# Allow `python ca_prosst/scripts/proteingym_eval.py` from the repo root.
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
    """Return [1, seq_len+2] tensor with CLS/EOS positions set to 100."""
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
    """Standard NDCG where gains are rescaled non-negative DMS scores."""
    order = np.argsort(-y_score)
    y = y_true[order]
    if k is not None:
        y = y[:k]
    y = y - y.min() + 1e-9  # shift to non-negative gains
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
) -> Optional[dict]:
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
    logits = torch.log_softmax(outputs.logits[:, 1:-1, :], dim=-1)

    df = pd.read_csv(mutant_csv)
    vocab = tokenizer.get_vocab()
    scores = []
    for mutant in df["mutant"].tolist():
        s = 0.0
        for sub in mutant.split(":"):
            wt, idx, mt = sub[0], int(sub[1:-1]) - 1, sub[-1]
            s += (logits[0, idx, vocab[mt]] - logits[0, idx, vocab[wt]]).item()
        scores.append(s)
    df["ca_prosst_score"] = scores

    y_true = df["DMS_score"].to_numpy()
    y_pred = np.array(scores)
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
    ap.add_argument("--gate_checkpoint", type=Path, default=None,
                    help="Optional .pt with learned ca_gate state dict")
    ap.add_argument("--limit", type=int, default=None, help="only score first N proteins (debug)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    if args.ca_mode != "none" and args.plddt_dir is None:
        print("--plddt_dir required for non-'none' ca_mode", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model_path} (ca_mode={args.ca_mode})")
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

    rows: list[dict] = []
    pred_dir = args.out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for name in tqdm(names, desc="scoring"):
        res = score_protein(
            model, tokenizer, name,
            args.residue_dir, args.structure_dir, args.mutant_dir, args.plddt_dir, device,
        )
        if res is None:
            continue
        row, pred_df = res
        rows.append(row)
        pred_df.to_csv(pred_dir / f"{name}.csv", index=False)

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
