"""Run the paper's experiment matrix and build a leaderboard ranked by the
competition score. Reuses train.py / calibrate.py / evaluate_submission.py via
subprocess, so there is one source of truth for each step.

For every configuration it:
  1. trains            -> models/<name>
  2. evaluates (raw)   -> results/eval_<name>.json         (calibration OFF)
  3. calibrates        -> models/<name>_cal
  4. evaluates (cal)   -> results/eval_<name>_cal.json      (calibration ON)
then aggregates all results/eval_*.json into results/leaderboard.csv.

It is RESUMABLE: any step whose output already exists is skipped, so you can
stop and restart (or run on a free GPU overnight) without repeating work.

Usage:
  python src/run_experiments.py                 # run the full matrix
  python src/run_experiments.py --only codeberta_small distilroberta
  python src/run_experiments.py --dry-run       # print the plan only
  python src/run_experiments.py --leaderboard   # just rebuild the leaderboard
  python src/run_experiments.py --eval-bs 256   # smaller eval batch for <16GB GPUs
  python src/run_experiments.py --only teacher kd_L3_135      # DiVaST: teacher, then student
  python src/run_experiments.py --only kd_L3_135 --seed 1     # extra seed
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

from common import PATHS

# --------------------------------------------------------------------------
# Experiment matrix. Edit freely; comment rows out to shrink the budget.
# Phase A: backbone comparison (weighted loss, matched recipe).
# Phase B: loss ablation on the efficient submission backbone.
# Phase C: small hyper-parameter sweep on the submission backbone.
# Calibration on/off is produced automatically for every row.
# --------------------------------------------------------------------------
LEFT_BASE = os.path.join(PATHS.models, "_base_codeberta_left")

MATRIX = [
    # Ordered by priority: the submission model and the fast 6-layer runs first,
    # the slow 12-layer runs last, so the most useful results arrive soonest.
    # name                 model                               loss       epochs lr      max_len bs
    ("codeberta_small",    "huggingface/CodeBERTa-small-v1",   "weighted", 3,    3e-5,   400,    32),
    ("codeberta_bce",      "huggingface/CodeBERTa-small-v1",   "bce",      3,    3e-5,   400,    32),
    ("codeberta_focal",    "huggingface/CodeBERTa-small-v1",   "focal",    3,    3e-5,   400,    32),
    ("distilroberta",      "distilroberta-base",               "weighted", 3,    3e-5,   400,    32),
    ("codeberta_lr5e5",    "huggingface/CodeBERTa-small-v1",   "weighted", 3,    5e-5,   400,    32),
    ("codeberta_ep4",      "huggingface/CodeBERTa-small-v1",   "weighted", 4,    3e-5,   400,    32),
    # Same as codeberta_bce, but long functions keep their END instead of their
    # START (truncation_side="left", stored in tokenizer_config.json; built by
    # src/make_left_base.py, run automatically). More SATD tags survive the 512-token cut.
    ("codeberta_bce_left", LEFT_BASE,                          "bce",      3,    3e-5,   400,    32),
    ("codebert_base",      "microsoft/codebert-base",          "weighted", 3,    2e-5,   400,    16),
    ("graphcodebert",      "microsoft/graphcodebert-base",     "weighted", 3,    2e-5,   400,    16),
]

# Distillation study (DiVaST). Students keep some encoder layers of the fine-tuned
# teacher (DistilBERT-style layer selection) and learn from the labels and the
# teacher's probabilities. Layer indices are 0-based: "1,3,5" keeps teacher layers
# 2, 4, and 6 (1-based), as written in the paper.
TEACHER = os.path.join(PATHS.models, "teacher")


def _kd_row(name, layers, teacher=True, lr=3e-5, epochs=3, extra=(), max_len=400, init=TEACHER):
    flags = (["--student-layers", layers] if layers else []) + (["--teacher", TEACHER] if teacher else [])
    return (name, init, "bce", epochs, lr, max_len, 32, flags + list(extra))


HID = ["--kd-hidden", "1.0"]


KD_MATRIX = [
    # The teacher: CodeBERTa-small, tag-preserving (left) truncation, BCE, 3 epochs,
    # seed 42 (the codeberta_bce_left recipe). Train it first: every student reads
    # models/teacher.
    ("teacher", LEFT_BASE, "bce", 3, 3e-5, 400, 32, []),
    # DiVaST-3L-768H, the submitted student (Table I). Seeds 1 and 2: --seed 1 / --seed 2.
    _kd_row("kd_L3_135", "1,3,5"),
    # Table II, layer selection (DiVaST-4L/2L/1L-768H)
    _kd_row("kd_L4_1235", "1,2,3,5"),
    _kd_row("kd_L2_35", "3,5"),
    _kd_row("kd_L1_5", "5"),
    # Table II, width reduction (src/narrow.py): hidden 768 -> W, FFN 4W, initialised
    # by projecting the teacher; they start further from it, hence more epochs and a
    # larger learning rate.
    _kd_row("kdn_L6_W512", "", lr=1e-4, epochs=6, extra=["--student-width", "512"]),
    _kd_row("kdn_L6_W384", "", lr=1e-4, epochs=6, extra=["--student-width", "384"]),
    _kd_row("kdn_L3_W384", "1,3,5", lr=1e-4, epochs=6, extra=["--student-width", "384"]),
    # Table III, ablations of kd_L3_135; each row changes one thing.
    _kd_row("nokd_L3_135", "1,3,5", teacher=False),            # no teacher (alpha=0); seeds 42, 1, 2
    _kd_row("kd_L3_135_init_pt", "1,3,5", init=LEFT_BASE),     # pre-trained init, not the teacher
    _kd_row("kd_L3_135_a10", "1,3,5", extra=["--kd-alpha", "1.0"]),  # no labels (alpha=1)
    _kd_row("kd_L3", "0,2,4"),                                 # layers 1, 3, 5 (1-based)
    _kd_row("kd_L3_025", "0,2,5"),                             # layers 1, 3, 6 (1-based)
    _kd_row("kd_L3_135_T1", "1,3,5", extra=["--kd-temp", "1"]),
    _kd_row("kd_L3_135_T4", "1,3,5", extra=["--kd-temp", "4"]),
    _kd_row("kd_L3_135_T8", "1,3,5", extra=["--kd-temp", "8"]),
    _kd_row("kd_L3_135_a08", "1,3,5", extra=["--kd-alpha", "0.8"]),
    _kd_row("kd_L3_135_mse", "1,3,5", extra=["--kd-loss", "mse"]),  # logit MSE
    _kd_row("kd_L3_135_T8_hid_ep5", "1,3,5", epochs=5, extra=["--kd-temp", "8"] + HID),  # 3-seed check
    _kd_row("kd_L3_135_hid", "1,3,5", extra=HID),                    # + hidden-state loss
    _kd_row("kd_L3_135_ep5", "1,3,5", epochs=5),
    # The truncation row of Table III re-evaluates kd_L3_135 (seeds 42, 1, 2) with
    # right truncation: src/eval_right_truncation.py.
]

PY = sys.executable


def run(cmd):
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def model_dir(name):
    return os.path.join(PATHS.models, name)


def eval_json(name):
    return os.path.join(PATHS.results, f"eval_{name}.json")


def do_config(name, model, loss, epochs, lr, max_len, bs, eval_bs, reps, fp16, save_steps=0,
              seed=None, aug=()):
    md = model_dir(name)
    # 0. the base model that keeps the end of long functions (src/make_left_base.py)
    if model == LEFT_BASE and not os.path.exists(os.path.join(LEFT_BASE, "config.json")):
        run([PY, "src/make_left_base.py"])
    # 1. train
    if not os.path.exists(os.path.join(md, "config.json")):
        cmd = [PY, "src/train.py", "--model", model, "--output-dir", md,
               "--loss", loss, "--epochs", str(epochs), "--lr", str(lr),
               "--max-len", str(max_len), "--bs", str(bs)]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        for a in aug:  # parquet files, or raw train.py flags such as --aug-mode replace
            cmd += ["--aug-file", a] if a.endswith(".parquet") else [a]
        if fp16:
            cmd.append("--fp16")
        if save_steps > 0:
            cmd += ["--save-steps", str(save_steps)]
        run(cmd)
    else:
        print(f"[skip train] {md} exists")
    # 2. evaluate raw (calibration OFF)
    if not os.path.exists(eval_json(name)):
        # Uncalibrated model: only its F1 is reported (runtime and GFLOPS are identical
        # to the calibrated model), so one timed pass is enough.
        run([PY, "src/evaluate_submission.py", "--model", md,
             "--batch-size", str(eval_bs), "--repetitions", "1"])
    else:
        print(f"[skip eval] {eval_json(name)} exists")
    # 3. calibrate
    cal = md + "_cal"
    if not os.path.exists(os.path.join(cal, "config.json")):
        run([PY, "src/calibrate.py", "--model-dir", md, "--out-dir", cal])
    else:
        print(f"[skip calibrate] {cal} exists")
    # 4. evaluate calibrated (calibration ON)
    if not os.path.exists(eval_json(name + "_cal")):
        run([PY, "src/evaluate_submission.py", "--model", cal,
             "--batch-size", str(eval_bs), "--repetitions", str(reps)])
    else:
        print(f"[skip eval] {eval_json(name+'_cal')} exists")


def build_leaderboard():
    rows = []
    for path in sorted(glob.glob(os.path.join(PATHS.results, "eval_*.json"))):
        with open(path) as f:
            d = json.load(f)
        pt = {r["cat"]: r for r in d["per_task"]}
        rows.append({
            "model": os.path.basename(path)[len("eval_"):-len(".json")],
            "mean_f1": round(d["mean_f1"], 4),
            "f1_vuln": round(pt.get("Vulnerability", {}).get("f1", 0), 4),
            "f1_mat": round(pt.get("MAT", {}).get("f1", 0), 4),
            "runtime_s": round(d["avg_runtime_s"], 3),
            "gflops": round(d["avg_gflops"], 1),
            "score": round(d["score"], 4),
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    out = os.path.join(PATHS.results, "leaderboard.csv")
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n{'model':28} {'meanF1':>7} {'vuln':>6} {'mat':>6} {'runtime':>8} {'gflops':>10} {'score':>7}")
    for r in rows:
        print(f"{r['model']:28} {r['mean_f1']:>7} {r['f1_vuln']:>6} {r['f1_mat']:>6} "
              f"{r['runtime_s']:>8} {r['gflops']:>10} {r['score']:>7}")
    print(f"\nWrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="run only these config names")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--leaderboard", action="store_true", help="only rebuild the leaderboard")
    ap.add_argument("--eval-bs", type=int, default=256,
                    help="local eval batch (256 is safe on a 12GB GPU; T4 notebook uses 600)")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--save-steps", type=int, default=2000,
                    help="checkpoint every N training steps so interrupted runs resume (0 = per epoch)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip evaluating the organisers' published baseline")
    ap.add_argument("--seed", type=int, default=None,
                    help="train with this seed instead of 42; outputs get the suffix _s<seed>")
    args = ap.parse_args()

    if args.leaderboard:
        build_leaderboard()
        return

    rows = [c + ([],) for c in MATRIX] + KD_MATRIX
    plan = [c for c in (rows if args.only else [c + ([],) for c in MATRIX])
            if (args.only is None or c[0] in args.only)]
    print("Planned configurations:")
    for c in plan:
        print(f"  {c[0]:20} model={c[1]} loss={c[2]} ep={c[3]} lr={c[4]} ml={c[5]} bs={c[6]} aug={c[7]}")
    if args.dry_run:
        return

    BASELINE = "NLBSE/nlbse27-code-comment-classification-baseline"
    base_json = eval_json(BASELINE.split("/")[-1])
    if not args.no_baseline and args.only is None and not os.path.exists(base_json):
        try:
            run([PY, "src/evaluate_submission.py", "--model", BASELINE,
                 "--batch-size", str(args.eval_bs), "--repetitions", str(args.reps)])
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] baseline evaluation failed: {e}. Continuing.", flush=True)

    for (name, model, loss, epochs, lr, max_len, bs, aug) in plan:
        if args.seed is not None:
            name = f"{name}_s{args.seed}"
        try:
            do_config(name, model, loss, epochs, lr, max_len, bs,
                      eval_bs=args.eval_bs, reps=args.reps, fp16=not args.no_fp16,
                      save_steps=args.save_steps, seed=args.seed, aug=aug)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] config {name} failed: {e}. Continuing.", flush=True)
        build_leaderboard()  # refresh after each config so partial progress is visible

    build_leaderboard()


if __name__ == "__main__":
    main()
