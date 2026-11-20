"""Evaluate a model with the EXACT organisers' protocol and compute the
competition score. Faithful re-implementation of CodeBERT_baseline.ipynb in
https://github.com/nlbse2027/code-comment-classification
(cells 3-4): same batching, repetitions, fixed 0.5 threshold, torch.profiler
FLOP counting, and scoring formula.

Usage:
  # Official protocol on the public test split:
  python src/evaluate_submission.py --model models/codeberta_small_cal
  # The organisers' published baseline, for reference:
  python src/evaluate_submission.py --model NLBSE/nlbse27-code-comment-classification-baseline
  # Fast smoke test (CPU, tiny):
  python src/evaluate_submission.py --model models/smoke --limit 64 \
      --batch-size 16 --repetitions 2 --no-amp
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time

import numpy as np

from common import (HF_DATASET, LABEL_COLUMN, PATHS, TEXT_COLUMN,
                    per_task_prf1, submission_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local dir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=600)
    ap.add_argument("--repetitions", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="cap #instances (smoke)")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 autocast")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()

    ds = load_dataset(HF_DATASET)
    test = ds[args.split].to_pandas()
    if args.limit > 0:
        test = test.iloc[:args.limit].reset_index(drop=True)
    print(f"Evaluating on {len(test)} instances of split '{args.split}'")

    tokenized = tokenizer(
        test[TEXT_COLUMN].tolist(), return_tensors="pt",
        padding=True, truncation=True, max_length=args.max_length,
    )
    dataset = TensorDataset(tokenized["input_ids"], tokenized["attention_mask"])
    loader = DataLoader(dataset, batch_size=args.batch_size)

    use_amp = (device.type == "cuda") and (not args.no_amp)
    predictions, total_time, total_flops = [], 0.0, 0.0

    for batch in loader:
        input_ids, attention_mask = (b.to(device, non_blocking=True) for b in batch)
        amp_ctx = torch.autocast(device_type="cuda") if use_amp else contextlib.nullcontext()
        begin = time.time()
        with torch.no_grad(), amp_ctx:
            with torch.profiler.profile(with_flops=True) as prof:
                for _ in range(args.repetitions):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - begin
        total_flops += sum(k.flops for k in prof.key_averages()) / 1e9

        probs = torch.sigmoid(outputs.logits)
        predictions.extend((probs > 0.5).int().cpu().numpy().tolist())
        del input_ids, attention_mask, outputs, probs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    labels = np.array(test[LABEL_COLUMN].tolist())
    predictions = np.array(predictions)
    rows = per_task_prf1(predictions, labels)

    avg_runtime = total_time / args.repetitions
    avg_flops = total_flops / args.repetitions
    avg_f1 = float(np.mean([r["f1"] for r in rows]))
    avg_p = float(np.mean([r["precision"] for r in rows]))
    avg_r = float(np.mean([r["recall"] for r in rows]))
    score = submission_score(avg_f1, avg_runtime, avg_flops)

    print("\n---------------- RESULTS ----------------")
    for r in rows:
        print(f"{r['cat']:>14}:  P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1']:.4f}")
    print(f"{'mean':>14}:  P={avg_p:.4f}  R={avg_r:.4f}  F1={avg_f1:.4f}")
    print(f"Avg runtime (s):   {avg_runtime:.4f}")
    print(f"Avg compute GFLOPs:{avg_flops:.1f}")
    print(f"COMPETITION SCORE: {score:.4f}")

    result = {
        "model": args.model, "split": args.split, "n": int(len(test)),
        "batch_size": args.batch_size, "repetitions": args.repetitions,
        "max_length": args.max_length, "amp": use_amp,
        "per_task": rows, "mean_f1": avg_f1, "mean_precision": avg_p,
        "mean_recall": avg_r, "avg_runtime_s": avg_runtime,
        "avg_gflops": avg_flops, "score": score,
    }
    out = args.out or os.path.join(
        PATHS.results, "eval_" + os.path.basename(args.model.rstrip("/")) + ".json")
    os.makedirs(PATHS.results, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
