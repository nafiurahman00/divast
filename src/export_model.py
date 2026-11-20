"""Prepare the Hugging Face release of DiVaST.

The repository root holds the submitted student, DiVaST-3L-768H, so it loads by name
exactly like the challenge baseline; the subfolder teacher/ holds the fine-tuned
CodeBERTa-small teacher, so the distillation can be reproduced without retraining it.
Adds vocab.json and merges.txt (RobertaTokenizer needs them on transformers 4.x),
checks that both tokenizers keep the end of long functions, and writes the model card.

Usage (from the repository root):
  python src/export_model.py --student-dir models/kd_L3_135 --teacher-dir models/teacher \
      --repo NafiuRahman/divast --github https://github.com/nafiurahman00/divast
  hf upload NafiuRahman/divast release/divast .      # after `hf auth login`; '.' = repo root
"""
from __future__ import annotations

import argparse
import os
import shutil

CARD = """---
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: huggingface/CodeBERTa-small-v1
datasets:
- NLBSE/nlbse27-code-comment-classification
metrics:
- f1
tags:
- code
- vulnerability-detection
- self-admitted-technical-debt
- knowledge-distillation
- nlbse
---

# DiVaST-3L-768H

DiVaST (Distilled Vulnerability and SATD Transformer) predicts two labels for a C or
C++ function, in the order `[Vulnerability, SATD]` (the SATD label is called MAT in the
dataset). This model is the three-layer student: it keeps layers 2, 4, and 6 of a
fine-tuned CodeBERTa-small teacher (hidden size 768, 62.2M parameters) and was trained
on the labels and on the teacher's per-label probabilities. For functions longer than
512 tokens, its tokenizer keeps the **end** instead of the start
(`truncation_side: "left"` in `tokenizer_config.json`), so fewer `TODO` and `FIXME`
tags are cut off.

Code, replication notebook, and all results: {github}

## Usage

The model loads like the challenge baseline, and a label is predicted when its sigmoid
probability exceeds 0.5:

```python
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("{repo}")
model = RobertaForSequenceClassification.from_pretrained("{repo}").eval()
code = "int f(char *s) {{ /* TODO: check the length */ char b[8]; strcpy(b, s); return 0; }}"
enc = tokenizer([code], return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    vulnerability, satd = (torch.sigmoid(model(**enc).logits)[0] > 0.5).tolist()
```

The six-layer teacher is in the subfolder `teacher/`:
`RobertaForSequenceClassification.from_pretrained("{repo}", subfolder="teacher")`.

## Results

Public test set, official protocol of the challenge (one Colab T4 session, batch 600,
10 repetitions, threshold 0.5):

| Model | F1 vulnerability | F1 SATD | Average F1 | Runtime (s) | GFLOPs | Score |
|---|---|---|---|---|---|---|
| CodeBERT baseline | 0.9390 | 0.8814 | 0.9102 | 242.61 | 2,666,510 | 0.7198 |
| Teacher (`teacher/`) | 0.9419 | 0.9283 | 0.9351 | 120.73 | 1,333,285 | 0.8480 |
| **DiVaST-3L-768H** | 0.9323 | 0.9274 | 0.9299 | 60.53 | 666,673 | **0.9014** |

## Training

Teacher: CodeBERTa-small fine-tuned with binary cross-entropy, 3 epochs, AdamW,
learning rate 3e-5, weight decay 0.01, warm-up over 6% of the steps, batch size 32,
inputs of 400 tokens, fp16, seed 42, on 95% of the training set; the checkpoint with
the best validation average F1 (remaining 5%) is kept.

Student: initialised from the teacher's embeddings, layers 2, 4, and 6, and
classification head, then trained with the same settings on
`(1 - a) * BCE(sigmoid(z_s), y) + a * T^2 * BCE(sigmoid(z_s / T), sigmoid(z_t / T))`,
with `a = 0.5` and `T = 2`; the teacher is frozen. Thresholds tuned on validation
change the average F1 by at most 0.0002, so the model uses 0.5.
"""

FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")


def copy_model(src, dst):
    from huggingface_hub import hf_hub_download

    os.makedirs(dst)
    for f in FILES:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    for f in ("vocab.json", "merges.txt"):
        path = os.path.join(src, f)
        if not os.path.exists(path):
            path = hf_hub_download("huggingface/CodeBERTa-small-v1", f)
        shutil.copy2(path, os.path.join(dst, f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-dir", default="models/kd_L3_135")
    ap.add_argument("--teacher-dir", default="models/teacher")
    ap.add_argument("--repo", required=True, help="Hugging Face id, e.g. NafiuRahman/divast")
    ap.add_argument("--github", required=True, help="URL of the code repository")
    ap.add_argument("--out", default="release/divast")
    args = ap.parse_args()

    from transformers import RobertaForSequenceClassification, RobertaTokenizer

    shutil.rmtree(args.out, ignore_errors=True)
    copy_model(args.student_dir, args.out)
    copy_model(args.teacher_dir, os.path.join(args.out, "teacher"))
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(CARD.format(repo=args.repo, github=args.github))

    for sub, layers in (("", 3), ("teacher", 6)):
        path = os.path.join(args.out, sub)
        tok = RobertaTokenizer.from_pretrained(path)
        assert tok.truncation_side == "left", (sub, tok.truncation_side)
        model = RobertaForSequenceClassification.from_pretrained(path)
        assert model.config.num_labels == 2 and model.config.num_hidden_layers == layers
    print(f"Ready: {args.out}")
    print(f"Upload with:  hf upload {args.repo} {args.out} .")


if __name__ == "__main__":
    main()
