"""Shared constants and helpers for the NLBSE'27 multitask code classification pipeline.

The task: one function string -> two binary labels in order [Vulnerability, MAT/SATD].
Reference: the organisers' repository https://github.com/nlbse2027/code-comment-classification
(README.md: data description; CodeBERT_baseline.ipynb: evaluation loop and score).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Dataset / label constants
# ---------------------------------------------------------------------------
HF_DATASET = "NLBSE/nlbse27-code-comment-classification"
BASELINE_MODEL = "NLBSE/nlbse27-code-comment-classification-baseline"

# Label order is fixed by the organisers' evaluation notebook:
#   index 0 -> Vulnerability / weakness, index 1 -> MAT (SATD proxy).
LABEL_NAMES = ["Vulnerability", "MAT"]
TEXT_COLUMN = "Function"
LABEL_COLUMN = "Label"
NUM_LABELS = 2

# ---------------------------------------------------------------------------
# Official scoring formula (verbatim from CodeBERT_baseline.ipynb, cell 4)
# ---------------------------------------------------------------------------
MAX_AVG_RUNTIME = 1000       # seconds, one full pass over the test split
MAX_AVG_FLOPS = 3_000_000    # GFLOPs, one full pass over the test split


def submission_score(avg_f1: float, avg_runtime: float, avg_flops: float) -> float:
    """Composite competition score. Higher is better; max is 1.0.

    score = 0.6 * mean_F1
          + 0.2 * max((MAX_AVG_RUNTIME - avg_runtime) / MAX_AVG_RUNTIME, 0)
          + 0.2 * max((MAX_AVG_FLOPS  - avg_flops)   / MAX_AVG_FLOPS,   0)
    """
    return (
        0.6 * avg_f1
        + 0.2 * max((MAX_AVG_RUNTIME - avg_runtime) / MAX_AVG_RUNTIME, 0.0)
        + 0.2 * max((MAX_AVG_FLOPS - avg_flops) / MAX_AVG_FLOPS, 0.0)
    )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def per_task_prf1(predictions: np.ndarray, labels: np.ndarray):
    """Per-task precision/recall/F1 computed exactly like the organisers' loop."""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    out = []
    for idx, name in enumerate(LABEL_NAMES):
        p = predictions[:, idx]
        y = labels[:, idx]
        tp = int(np.sum((p == 1) & (y == 1)))
        fp = int(np.sum((p == 1) & (y == 0)))
        fn = int(np.sum((p == 0) & (y == 1)))
        tn = int(np.sum((p == 0) & (y == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out.append({
            "cat": name, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
        })
    return out


@dataclass
class Paths:
    root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @property
    def data(self):
        return os.path.join(self.root, "data")

    @property
    def models(self):
        return os.path.join(self.root, "models")

    @property
    def results(self):
        return os.path.join(self.root, "results")


PATHS = Paths()

# Keep the HF cache inside the project so everything is self-contained / documented.
os.environ.setdefault("HF_HOME", os.path.join(PATHS.root, ".hf_cache"))
