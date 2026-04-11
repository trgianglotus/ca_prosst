# CA-ProSST: Confidence-Aware ProSST

Extension of ProSST (NeurIPS 2024) that modulates structure-token contribution
in disentangled attention based on residue-level structural confidence (pLDDT).

## Layout

```
ca_prosst/
├── models/
│   ├── modeling_prosst.py        # vendored from AI4Protein/ProSST-2048 (HF)
│   ├── configuration_prosst.py   # vendored
│   ├── ca_modeling.py            # V1/V2/V3 on top of ProSST
│   └── __init__.py
├── scripts/
│   ├── setup_env.sh              # conda env + pinned deps
│   ├── download_pdbs.sh          # ProteinGym AF2 PDB dump
│   ├── extract_plddt.py          # PDB → per-residue pLDDT .npy
│   ├── proteingym_eval.py        # zero-shot eval with stratified metrics
│   └── train_gate.py             # V3 learned-gate fine-tune
├── data/                          # populated by scripts (pdb/, plddt/, train/)
├── outputs/                       # eval summaries + gate checkpoints
└── README.md
```

## Variants (`--ca_mode`)

| Mode   | What it does                                                     | Adds params |
|--------|------------------------------------------------------------------|-------------|
| `none` | Pass-through — equivalent to vanilla ProSST                      | 0           |
| `hard` | Zero `ss_embeddings` for residues with pLDDT < `ca_threshold`    | 0           |
| `soft` | Multiply `ss_embeddings` by `pLDDT / 100`                        | 0           |
| `gate` | `sigmoid(MLP([aa_hidden, conf])) * ss_embeddings` (learned)      | ~200k       |

All transforms are applied once to `ss_embeddings` before the encoder — the
ProSST disentangled-attention `aa2ss` / `ss2aa` terms see the modulated values
in every layer ([models/ca_modeling.py:1](models/ca_modeling.py)).

## End-to-end recipe

### 1. Environment

```bash
bash ca_prosst/scripts/setup_env.sh prosst
conda activate prosst
```

### 2. Download ProteinGym AF2 PDBs (~3.5 GB)

```bash
bash ca_prosst/scripts/download_pdbs.sh
```

### 3. Extract pLDDT aligned to residue FASTAs

```bash
python ca_prosst/scripts/extract_plddt.py \
    --pdb_dir ca_prosst/data/pdb \
    --residue_dir ProSST/example_data/residue_sequence \
    --out_dir ca_prosst/data/plddt
```

Check `ca_prosst/data/plddt/_mismatches.txt` — any entries there need manual
inspection (sequence ↔ PDB alignment failed).

### 4. Baseline and CA variants on ProteinGym

```bash
# ProSST baseline — no pLDDT needed
python ca_prosst/scripts/proteingym_eval.py \
    --residue_dir ProSST/example_data/residue_sequence \
    --structure_dir ProSST/example_data/structure_sequence/2048 \
    --mutant_dir ProSST/example_data/substitutions \
    --out_dir ca_prosst/outputs/baseline \
    --ca_mode none

# V1 hard mask
python ca_prosst/scripts/proteingym_eval.py \
    ... \
    --plddt_dir ca_prosst/data/plddt \
    --out_dir ca_prosst/outputs/v1_hard70 \
    --ca_mode hard --ca_threshold 70

# V2 soft weight
python ca_prosst/scripts/proteingym_eval.py \
    ... \
    --plddt_dir ca_prosst/data/plddt \
    --out_dir ca_prosst/outputs/v2_soft \
    --ca_mode soft

# V3 learned gate (requires training first — see step 5)
python ca_prosst/scripts/proteingym_eval.py \
    ... \
    --plddt_dir ca_prosst/data/plddt \
    --out_dir ca_prosst/outputs/v3_gate \
    --ca_mode gate \
    --gate_checkpoint ca_prosst/outputs/gate_v3/gate_best.pt
```

Each run writes `per_protein.csv`, `predictions/<name>.csv`, and
`summary.json` (mean Spearman, 95% CI, NDCG, top-10% recall, pLDDT-quartile
stratification).

### 5. Train V3 learned gate

The gate needs a *held-out* protein set (not ProteinGym assay targets). Build
a train dir with the same layout as ProteinGym:

```
<train_dir>/
    residue_sequence/<name>.fasta
    structure_sequence/2048/<name>.fasta
    plddt/<name>.npy
```

Then:

```bash
python ca_prosst/scripts/train_gate.py \
    --train_dir ca_prosst/data/train \
    --val_dir   ca_prosst/data/val \
    --out_dir   ca_prosst/outputs/gate_v3 \
    --epochs 3 --batch_size 1 --lr 1e-3 --fp16
```

Only `ca_gate.*` (~200k params) is trainable; the base ProSST backbone is
frozen via `freeze_base()`.

## Smoke test (no PDB download, no training)

The upstream `ProSST/example_data/smoke/` subset has 1 protein. To quickly
verify the eval harness runs on that with `--ca_mode none`:

```bash
python ca_prosst/scripts/proteingym_eval.py \
    --residue_dir ProSST/example_data/smoke/residue_sequence \
    --structure_dir ProSST/example_data/smoke/structure_sequence/2048 \
    --mutant_dir ProSST/example_data/smoke/substitutions \
    --out_dir ca_prosst/outputs/smoke \
    --ca_mode none
```

For CA modes, create a dummy pLDDT with `numpy.full(L, 90.0)` to confirm
wiring before real PDBs are in place.

## Design notes

- **Why modulate at embeddings, not per-layer attention?** ProSST's encoder
  reuses the same `ss_hidden_states` across all 12 attention layers (a
  constant argument, not per-layer state). One transform on the single tensor
  propagates to every layer's `aa2ss`/`ss2aa` term. Simpler and faithful.
- **Why subclass `ProSSTForMaskedLM` instead of patching attention?** The HF
  checkpoint loads directly into the subclass; the only new parameters are
  `ca_gate.*`, cleanly marked as missing on load so HF initializes them.
- **Gate bias init.** The final linear bias is set to +2.0 so `sigmoid(+2) ≈
  0.88`, keeping the model near baseline behavior at the start of training.
  This avoids the gate collapsing structure contribution to zero before it
  learns anything useful.
