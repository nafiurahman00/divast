"""Download the NLBSE'27 dataset, run EDA, and build a jointly-stratified
train/validation split (the public `test` split is left untouched for final
reporting, mirroring the organisers' hidden test).

Outputs (under data/):
  * stats.json            label distribution, token-length percentiles
  * val_indices.json      indices (into the HF `train` split) used for validation
Usage:
  python src/prepare_data.py            # full EDA + split
  python src/prepare_data.py --quick    # skip token-length histogram (fast)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import (HF_DATASET, LABEL_COLUMN, LABEL_NAMES, PATHS, TEXT_COLUMN,
                    set_seed)


def joint_class(label):
    # label is [vuln, mat]; map to 0..3 for stratification
    return int(label[0]) * 2 + int(label[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tokenizer", default="microsoft/codebert-base",
                    help="tokenizer used only for length statistics")
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(PATHS.data, exist_ok=True)

    from datasets import load_dataset

    print(f"Loading {HF_DATASET} ...")
    ds = load_dataset(HF_DATASET)
    print(ds)

    stats = {"splits": {}}
    for split in ds:
        labels = np.array(ds[split][LABEL_COLUMN])
        n = len(labels)
        vuln_pos = int(labels[:, 0].sum())
        mat_pos = int(labels[:, 1].sum())
        joint = np.array([joint_class(l) for l in labels])
        dist = {f"[{b//2},{b%2}]": int((joint == b).sum()) for b in range(4)}
        stats["splits"][split] = {
            "n": n,
            "vuln_positive": vuln_pos,
            "vuln_positive_rate": vuln_pos / n,
            "mat_positive": mat_pos,
            "mat_positive_rate": mat_pos / n,
            "joint_distribution": dist,
        }
        print(f"\n[{split}] n={n}")
        print(f"  Vulnerability positive: {vuln_pos} ({vuln_pos/n:.4f})")
        print(f"  MAT positive:           {mat_pos} ({mat_pos/n:.4f})")
        print(f"  Joint [v,mat] dist:     {dist}")

    # pos_weight for BCE loss (neg/pos), computed on train only
    train_labels = np.array(ds["train"][LABEL_COLUMN])
    pos = train_labels.sum(axis=0)
    neg = len(train_labels) - pos
    pos_weight = (neg / np.clip(pos, 1, None)).tolist()
    stats["pos_weight"] = {LABEL_NAMES[i]: pos_weight[i] for i in range(len(LABEL_NAMES))}
    print(f"\npos_weight (neg/pos) for BCE: {stats['pos_weight']}")

    # token-length statistics (optional)
    if not args.quick:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        sample = ds["train"][TEXT_COLUMN]
        lengths = []
        BATCH = 2000
        for i in range(0, len(sample), BATCH):
            enc = tok(sample[i:i + BATCH], add_special_tokens=True,
                      truncation=False)["input_ids"]
            lengths.extend(len(x) for x in enc)
        lengths = np.array(lengths)
        pct = {f"p{p}": int(np.percentile(lengths, p)) for p in (50, 75, 90, 95, 99)}
        stats["train_token_lengths"] = {
            "tokenizer": args.tokenizer,
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
            "percentiles": pct,
            "frac_over_256": float((lengths > 256).mean()),
            "frac_over_512": float((lengths > 512).mean()),
        }
        print(f"\nToken lengths ({args.tokenizer}): mean={lengths.mean():.1f} "
              f"max={lengths.max()} p95={pct['p95']} "
              f">256={(lengths>256).mean():.3f} >512={(lengths>512).mean():.3f}")

    # jointly-stratified validation split out of train
    rng = np.random.default_rng(args.seed)
    joint = np.array([joint_class(l) for l in train_labels])
    val_idx = []
    for b in range(4):
        idx = np.where(joint == b)[0]
        k = int(round(len(idx) * args.val_frac))
        val_idx.extend(rng.choice(idx, size=k, replace=False).tolist())
    val_idx = sorted(int(i) for i in val_idx)
    with open(os.path.join(PATHS.data, "val_indices.json"), "w") as f:
        json.dump({"val_frac": args.val_frac, "seed": args.seed,
                   "n_val": len(val_idx), "indices": val_idx}, f)
    print(f"\nValidation split: {len(val_idx)} instances held out of train.")

    with open(os.path.join(PATHS.data, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {os.path.join(PATHS.data, 'stats.json')}")


if __name__ == "__main__":
    main()
