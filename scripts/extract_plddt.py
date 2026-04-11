"""Extract per-residue pLDDT from AlphaFold2 PDB files.

For AF2 structures, the per-atom B-factor column stores pLDDT. We read the CA
atom of each residue from chain A, verify the sequence matches the reference
residue FASTA, and save a float32 numpy array to <out>/<name>.npy.

Alignment guarantee: the output array length equals the reference sequence
length from `residue_sequence/<name>.fasta`. Residues missing from the PDB are
filled with 0.0 (treated as zero confidence downstream).

Usage:
    python scripts/extract_plddt.py \
        --pdb_dir /path/to/AlphaFold2_PDB \
        --residue_dir ProSST/example_data/residue_sequence \
        --out_dir ca_prosst/data/plddt

PDB naming: expects <name>.pdb matching the fasta stem. A --pdb_suffix flag
lets you override if the dump uses different extensions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from tqdm import tqdm

THREE_TO_ONE = {k.upper(): v for k, v in protein_letters_3to1.items()}


def read_fasta(path: Path) -> str:
    for rec in SeqIO.parse(str(path), "fasta"):
        return str(rec.seq)
    raise ValueError(f"empty fasta: {path}")


def extract_one(pdb_path: Path, ref_seq: str) -> tuple[np.ndarray, dict]:
    """Return (plddt[len(ref_seq)], stats_dict).

    Strategy: walk the first model, first chain, pick one CA per residue in
    resseq order, build (aa_letter, bfactor) list, then left-align against the
    reference by a simple exact-match scan. Missing residues → 0 pLDDT.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    model = next(structure.get_models())

    pdb_seq_chars: list[str] = []
    pdb_plddt: list[float] = []
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":  # skip HETATM / water
                continue
            resname = residue.get_resname().upper()
            letter = THREE_TO_ONE.get(resname)
            if letter is None:
                continue
            if "CA" not in residue:
                continue
            pdb_seq_chars.append(letter)
            pdb_plddt.append(float(residue["CA"].get_bfactor()))
        break  # first chain only

    pdb_seq = "".join(pdb_seq_chars)
    out = np.zeros(len(ref_seq), dtype=np.float32)

    if not pdb_seq:
        return out, {"pdb_len": 0, "ref_len": len(ref_seq), "aligned": 0, "mode": "empty"}

    # Try exact substring alignment first (cheap and typical for AF2 dumps).
    start = ref_seq.find(pdb_seq)
    if start >= 0:
        out[start : start + len(pdb_seq)] = pdb_plddt
        return out, {
            "pdb_len": len(pdb_seq),
            "ref_len": len(ref_seq),
            "aligned": len(pdb_seq),
            "mode": "substring",
            "offset": start,
        }

    # Fall back: per-position exact match where lengths agree.
    if len(pdb_seq) == len(ref_seq):
        matches = sum(a == b for a, b in zip(pdb_seq, ref_seq))
        if matches / len(ref_seq) >= 0.95:
            out[:] = pdb_plddt
            return out, {
                "pdb_len": len(pdb_seq),
                "ref_len": len(ref_seq),
                "aligned": len(ref_seq),
                "mode": "lenmatch",
                "identity": matches / len(ref_seq),
            }

    # Last resort: truncate/pad to min length, warn via stats.
    n = min(len(pdb_seq), len(ref_seq))
    out[:n] = pdb_plddt[:n]
    return out, {
        "pdb_len": len(pdb_seq),
        "ref_len": len(ref_seq),
        "aligned": n,
        "mode": "mismatch",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_dir", required=True, type=Path)
    ap.add_argument("--residue_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--pdb_suffix", default=".pdb")
    ap.add_argument(
        "--missing_ok",
        action="store_true",
        help="Skip proteins without a PDB instead of failing",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fastas = sorted(args.residue_dir.glob("*.fasta"))
    if not fastas:
        print(f"no FASTAs in {args.residue_dir}", file=sys.stderr)
        return 2

    missing, mismatches, ok = [], [], 0
    for fa in tqdm(fastas, desc="plddt"):
        name = fa.stem
        pdb_path = args.pdb_dir / f"{name}{args.pdb_suffix}"
        if not pdb_path.exists():
            missing.append(name)
            if args.missing_ok:
                continue
            print(f"missing pdb: {pdb_path}", file=sys.stderr)
            return 3
        ref = read_fasta(fa)
        plddt, stats = extract_one(pdb_path, ref)
        np.save(args.out_dir / f"{name}.npy", plddt)
        if stats["mode"] in ("mismatch", "empty"):
            mismatches.append((name, stats))
        ok += 1

    print(f"ok={ok}, missing={len(missing)}, mismatches={len(mismatches)}")
    if missing:
        (args.out_dir / "_missing.txt").write_text("\n".join(missing))
    if mismatches:
        lines = [f"{n}\t{s}" for n, s in mismatches]
        (args.out_dir / "_mismatches.txt").write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
