# DiVaST: Distilling a Small Transformer for Joint Vulnerability and SATD Detection

Replication package of our NLBSE'27 Tool Competition paper (multitask code
classification). DiVaST distills a fine-tuned CodeBERTa-small teacher into a
three-layer student, **DiVaST-3L-768H**, built from teacher layers 2, 4, and 6. It
predicts two labels for a C or C++ function, `[Vulnerability, SATD]`.

- Model: [huggingface.co/NafiuRahman/divast](https://huggingface.co/NafiuRahman/divast)
  (the student at the root, the teacher in `teacher/`)
- Notebook: [`DiVaST.ipynb`](DiVaST.ipynb)
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nafiurahman00/divast/blob/main/DiVaST.ipynb)

## Results

Public test set (30,655 functions), official protocol of the challenge (batch 600, fp16,
10 repetitions, threshold 0.5). All models ran in **one** Colab T4 session, as runtime
varied by up to 17% across sessions. Files: [`results/t4/`](results/t4).

| Model | Layers | Hidden | F1 vuln | F1 SATD | Avg F1 | Runtime (s) | GFLOPs | Score |
|---|---|---|---|---|---|---|---|---|
| CodeBERT baseline | 12 | 768 | 0.9390 | 0.8814 | 0.9102 | 242.61 | 2,666,510 | 0.7198 |
| Teacher (CodeBERTa-small) | 6 | 768 | 0.9419 | 0.9283 | 0.9351 | 120.73 | 1,333,285 | 0.8480 |
| DiVaST-4L-768H | 4 | 768 | 0.9352 | 0.9274 | 0.9313 | 84.29 | 888,877 | 0.8827 |
| **DiVaST-3L-768H** (submitted) | 3 | 768 | 0.9323 | 0.9274 | 0.9299 | 60.53 | 666,673 | **0.9014** |
| DiVaST-2L-768H | 2 | 768 | 0.9096 | 0.9223 | 0.9159 | 41.23 | 444,469 | 0.9117 |
| DiVaST-1L-768H | 1 | 768 | 0.8023 | 0.9124 | 0.8573 | 23.12 | 222,265 | 0.8950 |
| DiVaST-6L-512H | 6 | 512 | 0.8474 | 0.9200 | 0.8837 | 67.05 | 592,609 | 0.8773 |
| DiVaST-6L-384H | 6 | 384 | 0.8350 | 0.9218 | 0.8784 | 45.58 | 333,364 | 0.8957 |
| DiVaST-3L-384H | 3 | 384 | 0.8300 | 0.9171 | 0.8736 | 25.29 | 166,692 | 0.9080 |

DiVaST-2L-768H scores 0.010 higher than DiVaST-3L-768H, but misses 250 more vulnerable
functions (+35%). We submit DiVaST-3L-768H for its vulnerability F1 (paper, Sec. V-B).

## Quick start

```python
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("NafiuRahman/divast")
model = RobertaForSequenceClassification.from_pretrained("NafiuRahman/divast").eval()
code = "int f(char *s) { /* TODO: check the length */ char b[8]; strcpy(b, s); return 0; }"
enc = tokenizer([code], return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    vulnerability, satd = (torch.sigmoid(model(**enc).logits)[0] > 0.5).tolist()
```

The tokenizer keeps the **end** of functions longer than 512 tokens
(`truncation_side: "left"` in `tokenizer_config.json`), so fewer task tags are cut off.

## Reproducing the paper

```bash
pip install -r requirements.txt          # we used PyTorch 2.9 and Transformers 5.22
python src/run_experiments.py --only teacher kd_L3_135     # teacher, then DiVaST-3L-768H
python src/run_experiments.py --only kd_L3_135 --seed 1    # seeds 1 and 2 of Table III
```

`run_experiments.py` trains each configuration (`src/train.py`), evaluates it on the test
set (`src/evaluate_submission.py`), tunes per-label thresholds on validation
(`src/calibrate.py`), and evaluates again. It skips any step whose output exists, so an
interrupted run resumes. Models go to `models/<config>`, results to
`results/eval_<config>.json` (`_cal`: tuned thresholds). Every student reads the teacher
from `models/teacher`, so train `teacher` first, or download it from the Hub (step 2 of
the notebook).

Layer indices are 0-based in the code and 1-based in the paper: `--student-layers 1,3,5`
keeps teacher layers 2, 4, and 6.

| Paper | Config |
|---|---|
| Teacher (Tables I, II) | `teacher` |
| DiVaST-3L-768H (Tables I to IV) | `kd_L3_135` (seeds: `--seed 1`, `--seed 2`) |
| DiVaST-4L/2L/1L-768H (Table II) | `kd_L4_1235`, `kd_L2_35`, `kd_L1_5` |
| DiVaST-6L-512H, 6L-384H, 3L-384H (Table II) | `kdn_L6_W512`, `kdn_L6_W384`, `kdn_L3_W384` |
| Table III, Teacher (alpha=0) | `nokd_L3_135` (seeds 42, 1, 2) |
| Table III, Init. from teacher | `kd_L3_135_init_pt` |
| Table III, Labels (alpha=1) | `kd_L3_135_a10` |
| Table III, Keep first 510 tokens | test: `python src/eval_right_truncation.py models/kd_L3_135 models/kd_L3_135_s1 models/kd_L3_135_s2`; validation: `python src/val_right_truncation.py` |
| Table III, Layers 1, 3, 5 / 1, 3, 6 | `kd_L3`, `kd_L3_025` |
| Table III, T=1/4/8, alpha=0.8 | `kd_L3_135_T1`, `kd_L3_135_T4`, `kd_L3_135_T8`, `kd_L3_135_a08` |
| Table III, Logit MSE, + hidden-state loss, 5 epochs | `kd_L3_135_mse`, `kd_L3_135_hid`, `kd_L3_135_ep5` |
| Table IV, task tags | `python src/analyze_tags.py` |
| Teacher selection (Sec. IV) | `codeberta_small`, `codeberta_bce`, `codeberta_focal`, `codeberta_bce_left`, `codebert_base`, `graphcodebert`, ... |

The official runtime and GFLOPs come from `notebooks/divast_t4_eval.ipynb`, run on a
Colab T4 with the trained models on Google Drive (`src/collect_for_drive.py --kd`
packs them; `src/make_t4_notebook.py --kd` builds the notebook).

## Repository layout

```
DiVaST.ipynb                  replication notebook (Colab)
src/
  common.py                   dataset constants, official score formula, paths
  prepare_data.py             validation split -> data/val_indices.json
  make_left_base.py           CodeBERTa-small whose tokenizer keeps the end of functions
  train.py                    fine-tuning, layer selection, width reduction, distillation
  narrow.py                   width reduction (SVD projection of the teacher)
  calibrate.py                per-label thresholds on validation
  evaluate_submission.py      official evaluation loop and score
  eval_right_truncation.py    truncation row of Table III (test)
  val_right_truncation.py     truncation row of Table III (validation)
  analyze_tags.py             task-tag visibility and detected SATD functions
  run_experiments.py          all configurations of the paper, resumable
  export_model.py             Hugging Face release
  collect_for_drive.py, make_t4_notebook.py   T4 evaluation on Colab
notebooks/divast_t4_eval.ipynb  official T4 evaluation of all models
data/val_indices.json         validation split (5% of train, stratified, seed 42)
results/
  t4/                         official T4 results (Tables I and II)
  rtx4090/eval/               test results of every run on the RTX 4090 (one timed
                              repetition: use for F1 only)
  rtx4090/val/                validation F1 of every run (Table III, "Val.")
  rtx4090/val_keep_start.txt  validation F1 of the truncation row of Table III
  rtx4090/train_config/       exact training arguments of every run
  rtx4090/tag_*.txt           task-tag analysis (Table IV, Sec. III-A)
```

## Notes

- The teacher was trained on the RTX 4090 with the `codeberta_bce_left` recipe, first
  under the name `aug_ctl` (the same command, without extra data), and then copied to
  `models/teacher`; `results/rtx4090/train_config/teacher.json` shows its arguments.
- In `results/rtx4090/tag_*.txt`, the row labelled `VulCaST` is the model passed with
  `--model` (an earlier name of the option): `kd_L3_135` or `teacher`.
- F1 varies across training runs (seeds 42, 1, and 2: average F1 0.9285 to 0.9299;
  standard deviation 0.0008), and fp16 training is not bit-exact across GPUs.

## License

MIT, see [LICENSE](LICENSE).
