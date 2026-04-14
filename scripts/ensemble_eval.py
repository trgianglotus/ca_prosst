"""Stacked ensemble of ProSST-{128,512,2048}, ProSST-seq (structure zeroed),
and ESM-2-650M, using per-residue CALM mixing and fold-tuned mixture weights.

Per-mutant score:
    s_i = sum_k w_k(pLDDT_i) * d_k[i]

Channels (k): d_128, d_512, d_2048, d_seq, d_esm.

We support two weight parameterizations:
    - flat: w_k is a scalar (non-negative, simplex-normalized)
    - gated: w_k is a pLDDT-dependent sigmoid (one scalar per channel each
             for location and slope), allowing, e.g., ESM to win on low-pLDDT
             and ProSST-2048 to win on high-pLDDT

Tuning uses block coordinate descent in log-space because a full grid over
5 simplex weights × 10 bins is infeasible. For flat weights we use a
constrained convex objective (mean of per-assay Spearman is not convex but
smooth; small-grid random search suffices).

5-fold cross-validation over ProteinGym assays to report honest numbers;
oracle (tune on all) reported as ceiling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm


def load_cache_dir(cache_dir: Path, source_field: str, dest_key: str) -> dict[str, dict]:
    """Load all .npz files from a cache dir; store source_field under dest_key."""
    out = {}
    for f in sorted(cache_dir.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        out[str(z["name"])] = {
            "name": str(z["name"]),
            "segment_ids": z["segment_ids"].astype(np.int64),
            "y_true": z["y_true"].astype(np.float32),
            dest_key: z[source_field].astype(np.float32),
            "mutants": list(z["mutants"]),
            "plddt_at_pos": z["plddt_at_pos"].astype(np.float32) if "plddt_at_pos" in z.files else None,
            "mean_plddt": float(z["mean_plddt"]) if "mean_plddt" in z.files else None,
        }
    return out


def intersect_and_align(channels: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Keep only proteins present in *all* channel caches, align mutants.

    For a given protein, we use ProSST-2048's cache as the reference set of
    mutants (it sees the full protein, no truncation). For ESM-2 (which may
    skip mutants past position 1022), we build a boolean mask aligning its
    retained mutants to the reference and fill missing entries with 0 (the
    gate can weight it to zero on long proteins).
    """
    names = set.intersection(*(set(ch.keys()) for ch in channels.values()))
    aligned = {}
    for n in sorted(names):
        ref = channels["d_2048"][n]  # reference indexing
        ref_mutants = ref["mutants"]
        # Map each channel's mutant list → indices in ref_mutants
        entry = {
            "name": n,
            "y_true": ref["y_true"],
            "plddt_at_pos": ref["plddt_at_pos"],
            "segment_ids": ref["segment_ids"],
            "mean_plddt": ref["mean_plddt"],
            "n_mutants_ref": int(ref["segment_ids"].max()) + 1,
        }
        for key, ch in channels.items():
            c = ch[n]
            d_arr = c[key]
            c_mutants = c["mutants"]
            if c_mutants == ref_mutants:
                # identical order (ProSST channels)
                entry[key] = d_arr
                entry[f"{key}_segments"] = c["segment_ids"].astype(np.int64)
            else:
                # ESM or any variant with different mutant subset: align by mutant string
                # Build d_arr aggregated per-mutant from its own segments
                n_c = int(c["segment_ids"].max()) + 1
                s_c = np.zeros(n_c, dtype=np.float32)
                np.add.at(s_c, c["segment_ids"], d_arr)
                # Map c_mutants -> ref indices
                idx = {m: i for i, m in enumerate(ref_mutants)}
                s_full = np.zeros(len(ref_mutants), dtype=np.float32)
                mask = np.zeros(len(ref_mutants), dtype=bool)
                for i, m in enumerate(c_mutants):
                    if m in idx:
                        s_full[idx[m]] = s_c[i]
                        mask[idx[m]] = True
                entry[f"{key}_per_mutant"] = s_full
                entry[f"{key}_mask"] = mask
        aligned[n] = entry
    return aligned


def per_mutant_sum(d_sub: np.ndarray, seg: np.ndarray, n: int) -> np.ndarray:
    s = np.zeros(n, dtype=np.float32)
    np.add.at(s, seg, d_sub)
    return s


def score_entry(entry: dict, weights: dict[str, float]) -> np.ndarray:
    """Compute per-mutant ensemble score with flat scalar weights."""
    n = entry["n_mutants_ref"]
    seg = entry["segment_ids"]
    score = np.zeros(n, dtype=np.float32)
    for k, w in weights.items():
        if w == 0:
            continue
        if k in entry:  # per-sub array, need to sum
            score += w * per_mutant_sum(entry[k], seg, n)
        elif f"{k}_per_mutant" in entry:
            score += w * entry[f"{k}_per_mutant"]
    return score


def score_entry_gated(entry: dict, weights: dict[str, tuple[float, float, float]]) -> np.ndarray:
    """Gated scoring: each channel has (base_weight, p0, T) gate σ((pLDDT-p0)/T).

    Applied per-substitution for channels that store per-sub arrays; per-mutant
    channels (like ESM aggregated) get a protein-mean pLDDT gate.
    """
    n = entry["n_mutants_ref"]
    seg = entry["segment_ids"]
    plddt_at_pos = entry["plddt_at_pos"]
    score = np.zeros(n, dtype=np.float32)
    for k, (w, p0, T) in weights.items():
        if w == 0:
            continue
        if k in entry:
            gate = 1.0 / (1.0 + np.exp(-(plddt_at_pos - p0) / T))
            score += per_mutant_sum(w * gate * entry[k], seg, n)
        elif f"{k}_per_mutant" in entry:
            # Use protein mean pLDDT for aggregated channels
            mp = entry["mean_plddt"] or 85.0
            gate = 1.0 / (1.0 + np.exp(-(mp - p0) / T))
            score += (w * gate) * entry[f"{k}_per_mutant"]
    return score


def mean_spearman(entries: list[dict], scorer) -> float:
    rhos = []
    for e in entries:
        s = scorer(e)
        r = spearmanr(e["y_true"], s).correlation
        if r is not None and not np.isnan(r):
            rhos.append(r)
    return float(np.mean(rhos)) if rhos else float("nan")


def random_search_flat(entries: list[dict], channels: list[str],
                       n_samples: int, seed: int) -> tuple[dict, float]:
    rng = np.random.default_rng(seed)
    best_w, best_rho = None, -np.inf
    K = len(channels)
    for _ in range(n_samples):
        # Dirichlet sample on simplex then shift to allow zeros with mild prior
        w = rng.dirichlet(np.ones(K) * 0.7)
        weights = {k: float(w[i]) for i, k in enumerate(channels)}
        rho = mean_spearman(entries, lambda e: score_entry(e, weights))
        if rho > best_rho:
            best_rho = rho
            best_w = weights
    return best_w, best_rho


def refine_flat(entries: list[dict], channels: list[str],
                start: dict, step: float, rounds: int) -> tuple[dict, float]:
    """Coordinate descent refinement starting from `start`."""
    w = dict(start)
    best_rho = mean_spearman(entries, lambda e: score_entry(e, w))
    for _ in range(rounds):
        improved = False
        for k in channels:
            for delta in (+step, -step):
                trial = dict(w)
                trial[k] = max(0.0, trial[k] + delta)
                z = sum(trial.values())
                if z == 0:
                    continue
                trial = {kk: vv / z for kk, vv in trial.items()}
                r = mean_spearman(entries, lambda e: score_entry(e, trial))
                if r > best_rho + 1e-6:
                    best_rho = r
                    w = trial
                    improved = True
        if not improved:
            break
    return w, best_rho


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calm_128_dir", default="ca_prosst/outputs/calm_128/caches", type=Path)
    ap.add_argument("--calm_512_dir", default="ca_prosst/outputs/calm_512/caches", type=Path)
    ap.add_argument("--calm_2048_dir", default="ca_prosst/outputs/calm/caches", type=Path)
    ap.add_argument("--esm_dir", default="ca_prosst/outputs/esm650/caches", type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_random", type=int, default=200,
                    help="Random search samples per tuning round")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading caches...")
    ch = {
        "d_2048": load_cache_dir(args.calm_2048_dir, "d_struct", "d_2048"),
        "d_seq":  load_cache_dir(args.calm_2048_dir, "d_seq",    "d_seq"),
        "d_128":  load_cache_dir(args.calm_128_dir,  "d_struct", "d_128"),
        "d_512":  load_cache_dir(args.calm_512_dir,  "d_struct", "d_512"),
        "d_esm":  load_cache_dir(args.esm_dir,       "d_esm",    "d_esm"),
    }

    print(f"  d_128:  {len(ch['d_128'])} proteins")
    print(f"  d_512:  {len(ch['d_512'])} proteins")
    print(f"  d_2048: {len(ch['d_2048'])} proteins")
    print(f"  d_seq:  {len(ch['d_seq'])} proteins")
    print(f"  d_esm:  {len(ch['d_esm'])} proteins")

    # Align & intersect
    aligned = intersect_and_align(ch)
    names = sorted(aligned.keys())
    print(f"Intersection: {len(names)} proteins")

    entries = [aligned[n] for n in names]

    channels = ["d_128", "d_512", "d_2048", "d_seq", "d_esm"]

    # --- Baselines: each channel alone ---
    single = {}
    for k in channels:
        w = {c: 0.0 for c in channels}
        w[k] = 1.0
        r = mean_spearman(entries, lambda e: score_entry(e, w))
        single[k] = r
        print(f"  solo {k:8s}: {r:.4f}")

    # --- Oracle: tune flat weights on all ---
    print("Oracle random search...")
    rng = np.random.default_rng(args.seed)
    oracle_w, oracle_rho = random_search_flat(entries, channels, args.n_random, args.seed)
    print(f"  oracle rough: {oracle_w}  rho={oracle_rho:.4f}")
    oracle_w, oracle_rho = refine_flat(entries, channels, oracle_w, step=0.05, rounds=10)
    oracle_w, oracle_rho = refine_flat(entries, channels, oracle_w, step=0.01, rounds=10)
    print(f"  oracle refined: {oracle_w}  rho={oracle_rho:.4f}")

    # --- K-fold CV ---
    print(f"{args.n_folds}-fold CV...")
    idx = rng.permutation(len(entries))
    folds = np.array_split(idx, args.n_folds)
    per_protein = []
    fold_choices = []
    for k in range(args.n_folds):
        test_idx = set(folds[k].tolist())
        train_entries = [entries[i] for i in range(len(entries)) if i not in test_idx]
        w_k, _ = random_search_flat(train_entries, channels, args.n_random, args.seed + k)
        w_k, rho_k = refine_flat(train_entries, channels, w_k, step=0.05, rounds=10)
        w_k, rho_k = refine_flat(train_entries, channels, w_k, step=0.01, rounds=10)
        fold_choices.append({"fold": k, "weights": w_k, "train_rho": rho_k,
                             "n_train": len(train_entries), "n_test": len(folds[k])})
        # Score test fold
        for i in folds[k]:
            e = entries[i]
            scores = score_entry(e, w_k)
            rho = spearmanr(e["y_true"], scores).correlation
            per_protein.append({
                "name": e["name"],
                "n_mutants": e["n_mutants_ref"],
                "spearman": float(rho) if rho is not None else float("nan"),
                "mean_plddt": e["mean_plddt"],
                "fold": k,
            })
        print(f"  fold {k}: train_rho={rho_k:.4f}  weights={w_k}")

    df = pd.DataFrame(per_protein)
    df.to_csv(args.out_dir / "per_protein.csv", index=False)
    sp = df["spearman"].to_numpy()

    # Bootstrap CI
    rng_b = np.random.default_rng(0)
    boot = []
    for _ in range(1000):
        s = sp[rng_b.integers(0, len(sp), size=len(sp))]
        boot.append(np.nanmean(s))
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # Stratify
    def strat(df, col="mean_plddt"):
        d = df.dropna(subset=[col])
        if d.empty: return {}
        q = np.quantile(d[col], [0, .25, .5, .75, 1])
        out = {}
        for i in range(4):
            lo_, hi_ = q[i], q[i+1]
            m = d[(d[col] >= lo_) & ((d[col] < hi_) if i < 3 else (d[col] <= hi_))]
            out[f"Q{i+1}"] = {"lo": float(lo_), "hi": float(hi_), "n": len(m),
                             "mean_spearman": float(m["spearman"].mean()) if len(m) else float("nan")}
        return out

    summary = {
        "method": "Stacked Ensemble (ProSST-{128,512,2048} + seq-null + ESM-2-650M)",
        "n_proteins": len(df),
        "single_channel_spearman": single,
        "oracle_weights": oracle_w,
        "oracle_spearman": oracle_rho,
        "cv_mean_spearman": float(np.nanmean(sp)),
        "cv_median_spearman": float(np.nanmedian(sp)),
        "cv_spearman_95ci": [lo, hi],
        "stratified_by_plddt": strat(df),
        "fold_choices": fold_choices,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
