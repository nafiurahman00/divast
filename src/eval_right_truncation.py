"""Truncation row of the ablation (Table III): evaluate a trained student with the
tokenizer keeping the START of long functions instead of the end.

Copies the model to <model>_right, sets truncation_side="right" in its
tokenizer_config.json, and runs the official evaluation on it. Only inference
changes; the weights are the same.

Usage (from the repository root):
  python src/eval_right_truncation.py models/kd_L3_135 models/kd_L3_135_s1 models/kd_L3_135_s2
  # -> results/eval_kd_L3_135_right.json, ...
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys


def main():
    for md in sys.argv[1:]:
        md = md.rstrip("/")
        right = md + "_right"
        if not os.path.isdir(right):
            shutil.copytree(md, right)
            path = os.path.join(right, "tokenizer_config.json")
            with open(path) as f:
                cfg = json.load(f)
            cfg["truncation_side"] = "right"
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
        subprocess.run([sys.executable, "src/evaluate_submission.py", "--model", right,
                        "--batch-size", "64", "--repetitions", "1"], check=True)


if __name__ == "__main__":
    main()
