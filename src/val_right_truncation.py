"""Validation F1 of the Table III truncation row: kd_L3_135 (seeds 42, 1, 2) with the
tokenizer keeping the end (as trained; reproduces val_metrics.json) or the start.

Usage (from the repository root):  python src/val_right_truncation.py
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "src")
from common import HF_DATASET, TEXT_COLUMN, LABEL_COLUMN, per_task_prf1
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ds = load_dataset(HF_DATASET)["train"]
idx = sorted(json.load(open("data/val_indices.json"))["indices"])
val = ds.select(idx)
texts, y = val[TEXT_COLUMN], np.array(val[LABEL_COLUMN])
for run in ["kd_L3_135", "kd_L3_135_s1", "kd_L3_135_s2"]:
    model = AutoModelForSequenceClassification.from_pretrained(f"models/{run}").cuda().eval()
    rec = json.load(open(f"models/{run}/val_metrics.json"))["eval_mean_f1"]
    for side in ["left", "right"]:
        tok = AutoTokenizer.from_pretrained(f"models/{run}"); tok.truncation_side = side
        preds = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(texts), 128):
                enc = tok(texts[i:i+128], truncation=True, max_length=400, padding=True, return_tensors="pt").to("cuda")
                preds.append((torch.sigmoid(model(**enc).logits.float()) > 0.5).int().cpu().numpy())
        rows = per_task_prf1(np.concatenate(preds), y)
        print(run, side, "mean %.4f" % np.mean([r["f1"] for r in rows]), " ".join("%s %.4f" % (r["cat"], r["f1"]) for r in rows), "(recorded %.4f)" % rec, flush=True)
