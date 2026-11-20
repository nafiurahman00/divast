"""Reproduce the analysis of task tags and truncation in the paper (Sections III-B and V-b/c/f).

For the training and the public test set, it reports:
  * how many SATD-positive functions contain a task tag (TODO, FIXME, XXX, HACK; any case),
  * how many of these tags stay inside the 512-token input (510 content tokens) when the
    tokenizer keeps the start (default) or the end of long functions,
  * the input lengths and the share of truncated functions for CodeBERT and CodeBERTa,
  * the SATD F1 of a rule that predicts SATD when a function contains a tag as a whole word,
  * for each trained model: the SATD-positive test functions it detects, and whether their
    tag lies inside its input.

Usage (from the repository root):
  python src/analyze_tags.py                         # trained models found in models/
  python src/analyze_tags.py --model NafiuRahman/divast   # use the released student instead
  python src/analyze_tags.py --out results/tag_analysis.txt
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re

import numpy as np

from common import BASELINE_MODEL, HF_DATASET, LABEL_COLUMN, PATHS, TEXT_COLUMN

TAG = re.compile(r"TODO|FIXME|XXX|HACK", re.IGNORECASE)
TAG_WORD = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)  # keyword rule: whole words
KEEP = 510  # content tokens kept besides <s> and </s>
CODEBERTA = "huggingface/CodeBERTa-small-v1"


def n_tokens(tok, text):
    return len(tok(text, add_special_tokens=False, verbose=False)["input_ids"])


def tag_inside(tok, text, side):
    """True when a task tag of `text` lies inside the input after truncation on `side`."""
    tags = list(TAG.finditer(text))
    if not tags:
        return False
    if side == "right":  # the first KEEP tokens are kept
        return n_tokens(tok, text[:tags[0].start()]) < KEEP
    return n_tokens(tok, text[:tags[-1].start()]) >= n_tokens(tok, text) - KEEP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "--vulcast", dest="model", default=None,
                    help="model id or folder of the student (default: models/kd_L3_135)")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    report = io.StringIO()

    def say(*a):
        print(*a)
        print(*a, file=report)

    ds = load_dataset(HF_DATASET)
    cb = AutoTokenizer.from_pretrained(BASELINE_MODEL)   # CodeBERT tokenizer
    ca = AutoTokenizer.from_pretrained(CODEBERTA)        # CodeBERTa tokenizer

    # 1. Training set: tags inside the input (Section III-B)
    train = ds["train"].to_pandas()
    y_tr = np.array(train[LABEL_COLUMN].tolist())
    pos = [f for f, l in zip(train[TEXT_COLUMN], y_tr[:, 1]) if l == 1]
    tagged = [f for f in pos if TAG.search(f)]
    say(f"TRAINING SET: {len(pos):,} SATD-positive functions, {len(tagged):,} with a task tag")
    for name, tok, side in [("CodeBERT, keep start", cb, "right"),
                            ("CodeBERTa, keep start", ca, "right"),
                            ("CodeBERTa, keep end", ca, "left")]:
        inside = sum(tag_inside(tok, f, side) for f in tagged)
        say(f"  {name:<22} tag inside: {inside:,} | hidden: {len(tagged) - inside}")

    # 2. Test set: lengths and truncation (Sections III-B and V-b)
    test = ds["test"].to_pandas()
    y = np.array(test[LABEL_COLUMN].tolist())
    funcs = test[TEXT_COLUMN].tolist()
    say(f"\nTEST SET: {len(funcs):,} functions")
    for name, tok in [("CodeBERT", cb), ("CodeBERTa", ca)]:
        n = np.array([len(x) for x in tok(funcs, add_special_tokens=True, verbose=False)["input_ids"]])
        cut = n > KEEP + 2
        say(f"  {name:<10} median length {np.median(n):.0f} tokens | truncated: all {cut.mean():.1%}, "
            f"SATD-positive {cut[y[:, 1] == 1].mean():.2%}, vulnerable {cut[y[:, 0] == 1].mean():.1%}")
        if tok is ca:
            excess = n[cut] - (KEEP + 2)
            say(f"  {'':<10} tokens removed from truncated functions: median {np.median(excess):.0f}, "
                f"p90 {np.percentile(excess, 90):.0f}, max {excess.max()}")

    tpos = [f for f, l in zip(funcs, y[:, 1]) if l == 1]
    n_tag = sum(TAG.search(f) is not None for f in tpos)
    say(f"  {len(tpos)} SATD-positive functions, {n_tag} with a task tag, {len(tpos) - n_tag} without")
    for name, tok, side in [("CodeBERT, keep start", cb, "right"),
                            ("CodeBERTa, keep start", ca, "right"),
                            ("CodeBERTa, keep end", ca, "left")]:
        say(f"  {name:<22} tag inside: {sum(tag_inside(tok, f, side) for f in tpos)}")

    # 3. Keyword rule (Section V-b)
    rule = np.array([TAG_WORD.search(f) is not None for f in funcs])
    gold = y[:, 1] == 1
    tp, fp, fn = np.sum(rule & gold), np.sum(rule & ~gold), np.sum(~rule & gold)
    p, r = tp / (tp + fp), tp / (tp + fn)
    say(f"\nKEYWORD RULE (SATD if a tag occurs as a word, any case): P {p:.4f} | R {r:.4f} | F1 {2 * p * r / (p + r):.4f}")

    # 4. Detected SATD functions per model (Sections V-b, V-c, V-f)
    models = [("baseline", BASELINE_MODEL)]
    models.append(("teacher", os.path.join(PATHS.models, "teacher")))
    models.append(("DiVaST", args.model or os.path.join(PATHS.models, "kd_L3_135")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    say("\nSATD-POSITIVE TEST FUNCTIONS DETECTED (threshold 0.5, fp16 on GPU)")
    def available(mid):
        """A local folder that exists, or a Hugging Face id such as org/name."""
        return os.path.isdir(mid) or (not os.path.isabs(mid) and mid.count("/") == 1)

    for name, mid in models:
        if not available(mid):
            say(f"  {name:<20} skipped: {mid} not found")
            continue
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForSequenceClassification.from_pretrained(mid).to(dev).eval()
        inside = np.array([tag_inside(tok, f, tok.truncation_side) for f in tpos])
        found = []
        amp = torch.autocast("cuda", dtype=torch.float16) if dev == "cuda" else contextlib.nullcontext()
        with torch.no_grad(), amp:
            for i in range(0, len(tpos), args.bs):
                enc = tok(tpos[i:i + args.bs], return_tensors="pt", padding=True, truncation=True,
                          max_length=512).to(dev)
                found += (torch.sigmoid(model(**enc).logits.float())[:, 1] > 0.5).tolist()
        found = np.array(found)
        no_tag = np.array([TAG.search(f) is None for f in tpos])
        say(f"  {name:<20} keeps the {'end' if tok.truncation_side == 'left' else 'start'} | "
            f"tag inside {inside.sum()} | detected {found.sum()} | tag inside and detected "
            f"{np.sum(inside & found)} | detected without a tag inside {np.sum(~inside & found)} | "
            f"missed {np.sum(~found)}, of which without any tag {np.sum(~found & no_tag)}")
        del model

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(report.getvalue())
        print("wrote", args.out)


if __name__ == "__main__":
    main()
