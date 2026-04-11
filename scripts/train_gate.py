"""Fine-tune the CA-ProSST learned gate (V3) with a masked-LM objective.

Base model and all ProSST weights are frozen — only `ca_gate.*` trains. The
gate learns when to trust structure based on per-residue pLDDT by watching how
well residue reconstruction works under different confidence profiles.

Expected training directory layout:
    train_dir/
        residue_sequence/<name>.fasta      # wildtype AA sequence
        structure_sequence/2048/<name>.fasta  # comma-separated structure ids
        plddt/<name>.npy                   # per-residue pLDDT (aligned)

The user picks this dir — it should be a *held-out* protein set, NOT the
ProteinGym assay proteins, to keep the final benchmark zero-shot.

Usage:
    python ca_prosst/scripts/train_gate.py \
        --base_model AI4Protein/ProSST-2048 \
        --train_dir ca_prosst/data/train \
        --val_dir   ca_prosst/data/val \
        --out_dir   ca_prosst/outputs/gate_v3 \
        --epochs 3 --batch_size 1 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ca_prosst.models import CAProSSTConfig, CAProSSTForMaskedLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def read_seq(path: Path) -> str:
    for rec in SeqIO.parse(str(path), "fasta"):
        return str(rec.seq)
    raise ValueError(f"empty fasta: {path}")


@dataclass
class Sample:
    name: str
    residue_seq: str
    structure_ids: list[int]
    plddt: np.ndarray  # shape (L,) in 0..100


class ProSSTDataset(Dataset):
    def __init__(self, root: Path, structure_vocab: str = "2048") -> None:
        self.residue_dir = root / "residue_sequence"
        self.structure_dir = root / "structure_sequence" / structure_vocab
        self.plddt_dir = root / "plddt"
        names = []
        for fa in sorted(self.residue_dir.glob("*.fasta")):
            n = fa.stem
            if (self.structure_dir / f"{n}.fasta").exists() and (
                self.plddt_dir / f"{n}.npy"
            ).exists():
                names.append(n)
        if not names:
            raise RuntimeError(f"no complete samples under {root}")
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> Sample:
        n = self.names[idx]
        seq = read_seq(self.residue_dir / f"{n}.fasta")
        ss = [int(i) for i in read_seq(self.structure_dir / f"{n}.fasta").split(",")]
        plddt = np.load(self.plddt_dir / f"{n}.npy").astype(np.float32)
        L = min(len(seq), len(ss), plddt.shape[0])
        return Sample(n, seq[:L], ss[:L], plddt[:L])


def collate(batch: list[Sample], tokenizer, mask_token_id: int,
            mlm_prob: float, max_len: int, device: torch.device):
    """Single-sequence batch with random MLM masking + pLDDT tensor."""
    # The ProSST tokenizer tokenizes char-by-char, so we just clamp length.
    texts = [b.residue_seq[:max_len] for b in batch]
    tok = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    # Build ss_input_ids (same length as input_ids, with special tokens at ends).
    B, T = input_ids.shape
    ss_ids = torch.zeros((B, T), dtype=torch.long, device=device)
    plddt = torch.zeros((B, T), dtype=torch.float32, device=device)
    for i, b in enumerate(batch):
        L = min(len(b.structure_ids), T - 2, len(b.residue_seq), max_len)
        # cls/eos pads are 1 / 2 after shift; shift existing ids by +3
        ss_ids[i, 0] = 1
        ss_ids[i, 1 : 1 + L] = torch.tensor(
            [s + 3 for s in b.structure_ids[:L]], dtype=torch.long, device=device
        )
        ss_ids[i, 1 + L] = 2
        plddt[i, 0] = 100.0
        plddt[i, 1 + L] = 100.0
        plddt[i, 1 : 1 + L] = torch.from_numpy(b.plddt[:L]).to(device)

    # MLM masking on residues only (exclude CLS/EOS/PAD).
    labels = input_ids.clone()
    rand = torch.rand(input_ids.shape, device=device)
    special = (attention_mask == 0) | (
        torch.arange(T, device=device)[None, :] == 0
    ) | (
        torch.arange(T, device=device)[None, :] == (attention_mask.sum(-1, keepdim=True) - 1)
    )
    mask = (rand < mlm_prob) & (~special)
    labels[~mask] = -100
    masked_input = input_ids.clone()
    masked_input[mask] = mask_token_id
    return {
        "input_ids": masked_input,
        "ss_input_ids": ss_ids,
        "attention_mask": attention_mask,
        "plddt": plddt,
        "labels": labels,
    }


def run_epoch(model, loader, optimizer, scaler, device, train: bool, log_every: int = 20):
    model.train(train)
    losses = []
    pbar = tqdm(loader, desc="train" if train else "val", leave=False)
    for i, batch in enumerate(pbar):
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out = model(**batch)
            loss = out.loss
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        if i % log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return float(np.mean(losses)) if losses else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="AI4Protein/ProSST-2048")
    ap.add_argument("--train_dir", required=True, type=Path)
    ap.add_argument("--val_dir", type=Path, default=None)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--structure_vocab", default="2048")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--mlm_prob", type=float, default=0.15)
    ap.add_argument("--max_len", type=int, default=1022)
    ap.add_argument("--ca_gate_hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"loading {args.base_model}")
    config = CAProSSTConfig.from_pretrained(args.base_model)
    config.ca_mode = "gate"
    config.ca_gate_hidden = args.ca_gate_hidden
    model = CAProSSTForMaskedLM.from_pretrained(args.base_model, config=config).to(device)
    model.freeze_base()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    mask_id = tokenizer.mask_token_id

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_trainable}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    def mk_loader(root: Path) -> DataLoader:
        ds = ProSSTDataset(root, args.structure_vocab)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=lambda b: collate(
                b, tokenizer, mask_id, args.mlm_prob, args.max_len, device
            ),
        )

    train_loader = mk_loader(args.train_dir)
    val_loader = mk_loader(args.val_dir) if args.val_dir is not None else None

    best_val = math.inf
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device, train=True)
        val_loss = (
            run_epoch(model, val_loader, None, None, device, train=False)
            if val_loader is not None else float("nan")
        )
        dt = time.time() - t0
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss, "sec": dt})
        print(f"epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} ({dt:.0f}s)")

        ckpt_path = args.out_dir / f"gate_epoch{epoch}.pt"
        torch.save(model.ca_gate.state_dict(), ckpt_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.ca_gate.state_dict(), args.out_dir / "gate_best.pt")

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
