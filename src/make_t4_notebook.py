"""Generate notebooks/t4_eval.ipynb: one Colab notebook that runs the official
evaluation protocol (batch 600, 10 timed repetitions, fixed 0.5 threshold,
torch.profiler FLOPs) on a T4 for every trained model uploaded to Google Drive,
plus the organisers' published baseline. Inference only, no training.

Drive layout expected by the notebook (built by src/collect_for_drive.py):
    My Drive/nlbse27_t4/
        t4_eval.ipynb
        models/<name>_cal/      config, weights, tokenizer, _manifest.json
        results/                created by the notebook; one JSON per model

Each finished model is written to Drive at once, so a disconnect loses at most
the model in progress. Running all cells again skips finished models.

Run with:  python src/make_t4_notebook.py
           python src/make_t4_notebook.py --kd    # DiVaST models, Drive folder divast_t4
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
from run_experiments import MATRIX  # noqa: E402  (single source of config names)

OUT = os.path.join(ROOT, "notebooks", "t4_eval.ipynb")
BASELINE_ID = "NLBSE/nlbse27-code-comment-classification-baseline"
# Run order: the leading model, the baseline and the runner-up first, so the
# comparison the paper rests on comes from one session even if Colab disconnects
# later. Then the other 6-layer models in MATRIX order, then the 12-layer ones.
FIRST = ["codeberta_bce_left_cal", "baseline", "codeberta_bce_cal"]
PRIORITY = [name + "_cal" for (name, *_rest) in MATRIX]


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


INTRO = """# NLBSE'27: official evaluation of all models on a Colab T4

Inference only. Evaluates every model in `My Drive/nlbse27_t4/models/` and the
organisers' baseline with the official protocol (batch 600, 10 timed repetitions,
threshold 0.5, `torch.profiler` FLOPs).

**Run**
1. Runtime > Change runtime type > **T4 GPU** > Save.
2. Runtime > **Run all**, and allow Google Drive access.

**Disconnects.** Every finished model is saved to `My Drive/nlbse27_t4/results/`
immediately. If the session drops, reconnect and Run all again: finished models
are skipped, so only the model in progress is repeated.

**Time.** About 18 min per 6-layer model and 35 min per 12-layer model,
about 4 h in total. The leading model, the baseline and the runner-up run
first (about 1 h 15 min). Keep this tab open and check it about once an hour,
because Colab disconnects sessions it considers idle."""

SETUP = """# 1. Check the GPU and install packages.
import subprocess, importlib.metadata as md

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print("GPU:", gpu or "none")
assert "T4" in gpu, "Not a T4. Runtime > Change runtime type > T4 GPU, then Run all again."

# Same as the organisers' notebook: Colab's own torch and transformers, plus datasets.
# The models were saved with transformers 5.x, so upgrade only if Colab ships 4.x.
try:
    tv = md.version("transformers")
except md.PackageNotFoundError:
    tv = "0"
pkgs = ["datasets"]
if int(tv.split(".")[0]) < 5:
    pkgs.append("transformers==5.22.1")
subprocess.run(["pip", "-q", "install", *pkgs], check=True)
print("torch", md.version("torch"), "| transformers", md.version("transformers"),
      "| datasets", md.version("datasets"))"""

PATHS = """# 2. Mount Google Drive. Inputs and results live there.
from google.colab import drive
drive.mount('/content/drive')

import os
WORK = '/content/drive/MyDrive/nlbse27_t4'
MODELS = os.path.join(WORK, 'models')
RESULTS = os.path.join(WORK, 'results')
assert os.path.isdir(MODELS), f"{MODELS} not found. Upload the nlbse27_t4 folder to the top level of My Drive."
os.makedirs(RESULTS, exist_ok=True)

SRC_DIR = '/content/src'                       # evaluation code (rewritten each session)
LOCAL = '/content/models_local'                # models are copied here before loading
os.makedirs(SRC_DIR, exist_ok=True)
os.environ['HF_HOME'] = '/content/hf_cache'    # dataset and baseline cache on local disk

# Official protocol. Do not change.
BATCH, REPS, LIMIT = 600, 10, 0"""

QUEUE = """# 4. Find the uploaded models and check that each upload is complete.
import json

BASELINE = "%(baseline)s"
PRIORITY = %(priority)s
FIRST = %(first)s

def upload_problem(d):
    \"\"\"Compare the uploaded files with the sizes recorded before upload.\"\"\"
    mf = os.path.join(d, '_manifest.json')
    if not os.path.exists(mf):
        return 'no _manifest.json (folder not made by collect_for_drive.py?)'
    with open(mf) as f:
        expected = json.load(f)
    for name, size in expected.items():
        p = os.path.join(d, name)
        if not os.path.exists(p):
            return f'missing {name}: upload not finished, upload this folder again'
        if os.path.getsize(p) != size:
            return f'{name} is {os.path.getsize(p)} bytes, expected {size}: upload this folder again'
    return None

def result_path(name):
    return os.path.join(RESULTS, f'eval_{name}.json')

queue = []
for name in sorted(os.listdir(MODELS)):
    d = os.path.join(MODELS, name)
    if not os.path.isdir(d):
        continue
    problem = upload_problem(d)
    if problem:
        print(f'SKIPPING {name}: {problem}')
        continue
    with open(os.path.join(d, 'config.json')) as f:
        layers = json.load(f).get('num_hidden_layers', 12)
    rank = PRIORITY.index(name) if name in PRIORITY else len(PRIORITY)
    queue.append((layers, rank, name, d))
queue.append((12, len(PRIORITY), 'baseline', BASELINE))
# FIRST in its order, then the remaining 6-layer models, then the 12-layer ones.
queue.sort(key=lambda q: (0, FIRST.index(q[2])) if q[2] in FIRST else (1, q[0], q[1]))

print(f"{'model':<24}{'layers':>7}  status")
for layers, _, name, _src in queue:
    status = 'done (saved on Drive)' if os.path.exists(result_path(name)) else 'to do'
    print(f'{name:<24}{layers:>7}  {status}')""" % {
    "baseline": BASELINE_ID, "priority": json.dumps(PRIORITY), "first": json.dumps(FIRST)}

RUN = """# 5. Evaluate each model. Each result is saved to Drive as soon as it is done.
import csv, shutil, sys, time, datetime

ENV = {'gpu': gpu, 'torch': md.version('torch'),
       'transformers': md.version('transformers'), 'datasets': md.version('datasets')}

def write_leaderboard():
    rows = []
    for f in sorted(os.listdir(RESULTS)):
        if f.startswith('eval_') and f.endswith('.json'):
            with open(os.path.join(RESULTS, f)) as fh:
                r = json.load(fh)
            f1 = {t['cat']: t['f1'] for t in r['per_task']}
            rows.append({'model': r['model'], 'f1_vuln': round(f1['Vulnerability'], 4),
                         'f1_mat': round(f1['MAT'], 4), 'mean_f1': round(r['mean_f1'], 4),
                         'runtime_s': round(r['avg_runtime_s'], 2),
                         'gflops': round(r['avg_gflops']), 'score': round(r['score'], 4)})
    rows.sort(key=lambda r: -r['score'])
    if rows:
        tmp = os.path.join(RESULTS, 'leaderboard.csv.tmp')
        with open(tmp, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, os.path.join(RESULTS, 'leaderboard.csv'))
    return rows

local_results = os.path.join(os.path.dirname(SRC_DIR), 'results')
for layers, _, name, src in queue:
    if os.path.exists(result_path(name)):
        print(f'[skip] {name}: already saved on Drive')
        continue
    model_arg = src
    if os.path.isdir(src):                  # copy from Drive to local disk first
        model_arg = os.path.join(LOCAL, name)
        shutil.rmtree(model_arg, ignore_errors=True)
        shutil.copytree(src, model_arg)
    tmp_out = os.path.join(local_results, f'eval_{name}.json')
    cmd = [sys.executable, '-u', os.path.join(SRC_DIR, 'evaluate_submission.py'),
           '--model', model_arg, '--batch-size', str(BATCH),
           '--repetitions', str(REPS), '--out', tmp_out]
    if LIMIT:
        cmd += ['--limit', str(LIMIT)]
    t0 = time.time()
    print(f"\\n=== {name} ({layers} layers), started {time.strftime('%H:%M:%S')} ===", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end='', flush=True)
    if proc.wait() != 0:
        print(f'[FAILED] {name}: see the error above. Continuing with the next model.')
        continue
    with open(tmp_out) as f:
        res = json.load(f)
    res['model'] = name
    res['source'] = src
    res['env'] = {**ENV, 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                  'wall_clock_min': round((time.time() - t0) / 60, 1)}
    tmp = result_path(name) + '.tmp'        # write then rename: never a half-written file
    with open(tmp, 'w') as f:
        json.dump(res, f, indent=2)
    os.replace(tmp, result_path(name))
    write_leaderboard()
    if os.path.isdir(src):
        shutil.rmtree(model_arg, ignore_errors=True)
    print(f"SAVED to Drive: {result_path(name)}  (score {res['score']:.4f})", flush=True)

print('\\nAll models processed.')"""

BOARD = """# 6. Leaderboard (also saved as My Drive/nlbse27_t4/results/leaderboard.csv).
rows = write_leaderboard()
print(f"{'model':<24}{'F1 vuln':>9}{'F1 MAT':>9}{'mean F1':>9}{'runtime s':>11}{'GFLOPS':>11}{'score':>8}")
for r in rows:
    print(f"{r['model']:<24}{r['f1_vuln']:>9.4f}{r['f1_mat']:>9.4f}{r['mean_f1']:>9.4f}"
          f"{r['runtime_s']:>11.2f}{r['gflops']:>11,}{r['score']:>8.4f}")
missing = [n for _, _, n, _ in queue if not os.path.exists(result_path(n))]
print('\\nStill missing:', ', '.join(missing) if missing else 'none. All done.')"""

FLUSH = """# 7. Optional, at the very end: force every pending write to reach Drive.
# Results are already synced in the background; this only makes certain of it.
drive.flush_and_unmount()
print('Drive flushed. Results are in My Drive/nlbse27_t4/results/.')"""


KD_INTRO = """# DiVaST (NLBSE'27): official evaluation of the distillation models on a Colab T4

Inference only. Evaluates every model in `My Drive/divast_t4/models/` (the teacher
and seven DiVaST students) and the organisers' baseline with the official protocol
(batch 600, 10 timed repetitions, threshold 0.5, `torch.profiler` FLOPs). All
models run in this one session, so their runtimes are comparable.

**Run**
1. Runtime > Change runtime type > **T4 GPU** > Save.
2. Runtime > **Run all**, and allow Google Drive access.

**Disconnects.** Every finished model is saved to `My Drive/divast_t4/results/`
immediately. If the session drops, reconnect and Run all again: finished models
are skipped, so only the model in progress is repeated.

**Time.** About 2 h in total: the submitted student, the baseline (about 40 min)
and the teacher (about 20 min) run first, then the other students (3 to 15 min
each). Keep this tab open and check it about every 30 min, because Colab
disconnects sessions it considers idle."""
KD_FIRST = ["divast_3L_768H", "divast_2L_768H", "baseline", "teacher"]
KD_PRIORITY = ["divast_3L_768H", "teacher", "divast_4L_768H", "divast_2L_768H", "divast_1L_768H",
               "divast_6L_512H", "divast_6L_384H", "divast_3L_384H"]


def main():
    global OUT
    intro, paths, queue = INTRO, PATHS, QUEUE
    if "--kd" in sys.argv:
        OUT = os.path.join(ROOT, "notebooks", "divast_t4_eval.ipynb")
        intro = KD_INTRO
        paths = PATHS.replace("nlbse27_t4", "divast_t4")
        queue = (QUEUE.replace(json.dumps(PRIORITY), json.dumps(KD_PRIORITY))
                      .replace(json.dumps(FIRST), json.dumps(KD_FIRST))
                      .replace("FIRST in its order, then the remaining 6-layer models, then the 12-layer ones.",
                               "FIRST in its order, then the other students."))
        assert "divast_t4" in paths and json.dumps(KD_FIRST) in queue
    cells = [md(intro), code(SETUP), code(paths)]
    cells.append(md("## 3. Evaluation code\nThe same tested scripts as the local pipeline."))
    for m in ["common.py", "evaluate_submission.py"]:
        with open(os.path.join(SRC, m)) as f:
            cells.append(code(f"%%writefile /content/src/{m}\n{f.read()}"))
    cells += [code(queue), code(RUN), code(BOARD),
              code(FLUSH.replace("nlbse27_t4", "divast_t4") if "--kd" in sys.argv else FLUSH)]
    if "--kd" in sys.argv:                  # every remaining mention of the Drive folder
        for c in cells:
            c["source"] = [line.replace("nlbse27_t4", "divast_t4") for line in c["source"]]
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
