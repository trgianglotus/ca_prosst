"""Cache ESM-2 zero-shot log-likelihood deltas for ProteinGym substitutions.

One forward pass per protein with the wild-type sequence; extract per-residue
log-softmax and compute per-substitution log p(mt) - log p(wt) deltas.
Output schema matches the CALM caches (segment_ids, d_esm, y_true, mutants)
for drop-in use in the ensemble script.

Default model: facebook/esm2_t33_650M_UR50D (max input 1024).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM


def read_seq(path: Path) -> str:
    for rec in SeqIO.parse(str(path), "fasta"):
        return str(rec.seq)
    raise ValueError(f"empty fasta: {path}")


@torch.no_grad()
def compute_cache(
    model: EsmForMaskedLM,
    tokenizer,
    name: str,
    residue_dir: Path,
    mutant_dir: Path,
    device: torch.device,
    max_len: int,
):
    residue_fa = residue_dir / f"{name}.fasta"
    mutant_csv = mutant_dir / f"{name}.csv"
    if not (residue_fa.exists() and mutant_csv.exists()):
        return None

    sequence = read_seq(residue_fa)
    # Truncate if needed (ESM-2 max input 1024 tokens incl. specials)
    seq_for_model = sequence[: max_len - 2]
    tok = tokenizer([seq_for_model], return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = torch.log_softmax(out.logits[:, 1:-1, :], dim=-1)[0]  # [L', V]
    L_cap = logits.shape[0]

    df = pd.read_csv(mutant_csv)
    vocab = tokenizer.get_vocab()

    positions, wt_ids, mt_ids, segments, keep = [], [], [], [], []
    skipped_oob = 0
    for i, mutant in enumerate(df["mutant"].tolist()):
        subs = mutant.split(":")
        out_of_bounds = False
        parsed = []
        for sub in subs:
            idx = int(sub[1:-1]) - 1
            if idx >= L_cap:
                out_of_bounds = True
                break
            wt, mt = sub[0], sub[-1]
            if wt not in vocab or mt not in vocab:
                out_of_bounds = True
                break
            parsed.append((idx, vocab[wt], vocab[mt]))
        if out_of_bounds:
            skipped_oob += 1
            continue
        for (idx, w, m) in parsed:
            positions.append(idx)
            wt_ids.append(w)
            mt_ids.append(m)
            segments.append(i)
        keep.append(i)

    if not positions:
        return None

    pos = torch.tensor(positions, dtype=torch.long, device=device)
    wt = torch.tensor(wt_ids, dtype=torch.long, device=device)
    mt = torch.tensor(mt_ids, dtype=torch.long, device=device)
    d_esm = (logits[pos, mt] - logits[pos, wt]).detach().cpu().numpy()

    # Remap segment ids to contiguous over kept mutants
    idx_map = {orig: new for new, orig in enumerate(keep)}
    new_segments = np.array([idx_map[s] for s in segments], dtype=np.int64)

    return {
        "name": name,
        "n_mutants_total": len(df),
        "n_mutants_kept": len(keep),
        "n_oob_skipped": skipped_oob,
        "segment_ids": new_segments,
        "d_esm": d_esm.astype(np.float32),
        "y_true": df["DMS_score"].to_numpy().astype(np.float32)[keep],
        "mutants": np.array([df["mutant"].iloc[k] for k in keep]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--residue_dir", required=True, type=Path)
    ap.add_argument("--mutant_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "caches"
    cache_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"loading {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = EsmForMaskedLM.from_pretrained(args.model_path).to(device).eval()

    names = sorted(p.stem for p in args.residue_dir.glob("*.fasta"))
    if args.limit:
        names = names[: args.limit]

    skipped, n_done = [], 0
    for name in tqdm(names, desc="ESM-2 caching"):
        cache_path = cache_dir / f"{name}.npz"
        if cache_path.exists():
            n_done += 1
            continue
        try:
            c = compute_cache(model, tokenizer, name,
                              args.residue_dir, args.mutant_dir,
                              device, args.max_len)
        except Exception as e:
            print(f"  ERR {name}: {e}")
            skipped.append((name, str(e)))
            continue
        if c is None:
            skipped.append((name, "no data"))
            continue
        np.savez_compressed(cache_path, **{k: v for k, v in c.items() if k != "mutants"},
                            mutants=c["mutants"])
        n_done += 1

    print(f"cached {n_done}/{len(names)}; skipped={len(skipped)}")
    if skipped:
        import json
        (args.out_dir / "skipped.json").write_text(json.dumps(skipped, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
