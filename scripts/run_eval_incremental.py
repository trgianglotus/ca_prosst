"""Incremental wrapper around proteingym_eval that scores one protein at a time.

Skips proteins already scored (prediction CSV exists). Writes per_protein.csv
and summary.json after each protein so partial results are always available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from scipy.stats import spearmanr
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ca_prosst.models import CAProSSTConfig, CAProSSTForMaskedLM
from ca_prosst.scripts.proteingym_eval import (
    build_plddt_tensor, ndcg_score, top_recall, bootstrap_ci, stratified,
    tokenize_structure_sequence, read_seq,
)
from transformers import AutoTokenizer


def score_protein_chunked(
    model, tokenizer, name, residue_dir, structure_dir, mutant_dir, plddt_dir, device,
):
    """Score a single protein, handling large mutant sets without OOM."""
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
    mean_plddt = None
    if plddt_dir is not None:
        p_path = plddt_dir / f"{name}.npy"
        plddt = build_plddt_tensor(p_path, len(sequence), device)
        if plddt is not None:
            interior = plddt[0, 1:-1].detach().cpu().numpy()
            nonzero = interior[interior > 0]
            mean_plddt = float(nonzero.mean()) if nonzero.size > 0 else 0.0

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ss_input_ids=ss_input_ids,
            plddt=plddt,
            labels=input_ids,
        )
        logits = torch.log_softmax(outputs.logits[:, 1:-1, :], dim=-1).cpu()

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


def write_summary(rows, out_dir, ca_mode, ca_threshold, n_bootstrap=1000):
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_protein.csv", index=False)
    sp = df["spearman"].to_numpy()
    lo, hi = bootstrap_ci(sp, n=n_bootstrap)
    summary = {
        "ca_mode": ca_mode,
        "ca_threshold": ca_threshold,
        "n_proteins": int(len(df)),
        "mean_spearman": float(np.nanmean(sp)),
        "median_spearman": float(np.nanmedian(sp)),
        "spearman_95ci": [lo, hi],
        "mean_ndcg": float(np.nanmean(df["ndcg"])),
        "mean_top_recall_10pct": float(np.nanmean(df["top_recall_10pct"])),
        "stratified_by_plddt": stratified(df),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
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
    ap.add_argument("--max_mutants", type=int, default=200000,
                    help="skip proteins with more mutants than this")
    args = ap.parse_args()

    if args.ca_mode != "none" and args.plddt_dir is None:
        print("--plddt_dir required for non-'none' ca_mode", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
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

    # Collect already-scored proteins
    done = {p.stem for p in pred_dir.glob("*.csv")}
    rows = []
    # Load existing results
    existing_csv = args.out_dir / "per_protein.csv"
    if existing_csv.exists() and done:
        old_df = pd.read_csv(existing_csv)
        for _, r in old_df.iterrows():
            if r["name"] in done:
                rows.append(r.to_dict())

    skipped = []
    for name in tqdm(names, desc="scoring"):
        if name in done:
            continue
        # Check mutant count
        mutant_csv = args.mutant_dir / f"{name}.csv"
        if mutant_csv.exists():
            n_lines = sum(1 for _ in open(mutant_csv)) - 1
            if n_lines > args.max_mutants:
                print(f"  SKIP {name}: {n_lines} mutants > {args.max_mutants}")
                skipped.append((name, n_lines))
                continue

        try:
            res = score_protein_chunked(
                model, tokenizer, name,
                args.residue_dir, args.structure_dir, args.mutant_dir,
                args.plddt_dir, device,
            )
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            skipped.append((name, str(e)))
            continue

        if res is None:
            continue
        row, pred_df = res
        rows.append(row)
        pred_df.to_csv(pred_dir / f"{name}.csv", index=False)

        # Write incremental summary
        write_summary(rows, args.out_dir, args.ca_mode, args.ca_threshold)
        print(f"  {name}: spearman={row['spearman']:.4f} ({len(rows)} done)")

    # Final summary
    summary = write_summary(rows, args.out_dir, args.ca_mode, args.ca_threshold)
    if skipped:
        (args.out_dir / "skipped.json").write_text(
            json.dumps(skipped, indent=2)
        )
    print(f"\nFINAL: {len(rows)} proteins scored, {len(skipped)} skipped")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
