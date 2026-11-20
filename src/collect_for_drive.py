"""Pack the calibrated models and the T4 notebook into one folder for a single
Google Drive upload:

    upload/nlbse27_t4/
        t4_eval.ipynb
        models/<name>_cal/   every file of the model + _manifest.json (file sizes)

Files are hard-linked, so the folder uses no extra disk space. A model is
included only when its calibrated evaluation exists in results/, which means
training and calibration finished. The notebook compares the uploaded files
against _manifest.json, so an incomplete upload is reported instead of failing.

Run with:  python src/collect_for_drive.py
           python src/collect_for_drive.py --kd   # DiVaST distillation models (paper_kd)

--kd packs the nine models of the distillation paper from models_kd/ (copied
from the 4090 box, plain folders at threshold 0.5, no calibration) into
upload/divast_t4/, a separate Drive folder, so results of the earlier session
in nlbse27_t4/ are never mistaken for these.
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from run_experiments import MATRIX  # noqa: E402

MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "upload", "nlbse27_t4")
NOTEBOOK = os.path.join(ROOT, "notebooks", "t4_eval.ipynb")
BASE = {name + "_cal": model_id for (name, model_id, *_rest) in MATRIX}

# --kd: Drive folder name -> folder on the 4090 box (models/<folder>). The paper
# names the students by layers and hidden size (DiVaST-3L-768H, ...).
KD_MODELS = {
    "divast_3L_768H": "kd_L3_135",      # submitted student (seed 42)
    "teacher": "teacher",               # CodeBERTa-small 6L-768H
    "divast_4L_768H": "kd_L4_1235",
    "divast_2L_768H": "kd_L2_35",
    "divast_1L_768H": "kd_L1_5",
    "divast_6L_512H": "kdn_L6_W512",
    "divast_6L_384H": "kdn_L6_W384",
    "divast_3L_384H": "kdn_L3_W384",
}
KD_SRC = os.path.join(ROOT, "models_kd")
KD_OUT = os.path.join(ROOT, "upload", "divast_t4")
KD_NOTEBOOK = os.path.join(ROOT, "notebooks", "divast_t4_eval.ipynb")


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def add_slow_tokenizer_files(name, src, dst):
    """transformers 5.x saves only tokenizer.json, but the organisers' notebook
    loads RobertaTokenizer, which on transformers 4.x needs vocab.json and
    merges.txt (their baseline ships both). Fine-tuning never changes the
    tokenizer, so take both files from the base model and check that the
    vocabulary is identical to the one in our tokenizer.json."""
    from huggingface_hub import hf_hub_download
    added = []
    for fname in ("vocab.json", "merges.txt"):
        if os.path.exists(os.path.join(src, fname)):
            continue
        base = BASE[name]
        path = os.path.join(base, fname) if os.path.isdir(base) else hf_hub_download(base, fname)
        shutil.copy2(path, os.path.join(dst, fname))
        added.append(fname)
    if "vocab.json" in added:
        with open(os.path.join(src, "tokenizer.json")) as f:
            ours = json.load(f)["model"]["vocab"]
        with open(os.path.join(dst, "vocab.json")) as f:
            base = json.load(f)
        if ours != base:
            sys.exit(f"vocab.json of {BASE[name]} differs from {name}/tokenizer.json")
    return added


def main_kd():
    """Pack the distillation models; every student shares the teacher's tokenizer."""
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "make_t4_notebook.py"), "--kd"], check=True)
    shutil.rmtree(KD_OUT, ignore_errors=True)
    os.makedirs(os.path.join(KD_OUT, "models"))
    link_or_copy(KD_NOTEBOOK, os.path.join(KD_OUT, "divast_t4_eval.ipynb"))
    teacher = os.path.join(KD_SRC, "teacher")
    packed, total = [], 0
    for name in KD_MODELS:
        src = os.path.join(KD_SRC, name)
        if not os.path.exists(os.path.join(src, "model.safetensors")):
            print(f"  not ready, skipped: {name}")
            continue
        with open(os.path.join(src, "tokenizer_config.json")) as f:
            side = json.load(f).get("truncation_side")
        if side != "left":
            sys.exit(f"{name}: truncation_side is {side!r}, expected 'left' (tag-preserving truncation)")
        dst = os.path.join(KD_OUT, "models", name)
        os.makedirs(dst)
        manifest = {}
        for f in sorted(os.listdir(src)):
            p = os.path.join(src, f)
            if os.path.isfile(p) and f not in ("training_args.bin", "PROVENANCE.txt"):
                link_or_copy(p, os.path.join(dst, f))
                manifest[f] = os.path.getsize(p)
        for fname in ("vocab.json", "merges.txt"):          # slow-tokenizer files for RobertaTokenizer
            if fname not in manifest:
                shutil.copy2(os.path.join(teacher, fname), os.path.join(dst, fname))
                manifest[fname] = os.path.getsize(os.path.join(dst, fname))
        with open(os.path.join(src, "tokenizer.json")) as f:
            ours = json.load(f)["model"]["vocab"]
        with open(os.path.join(dst, "vocab.json")) as f:
            if json.load(f) != ours:
                sys.exit(f"{name}: vocab.json differs from its tokenizer.json")
        with open(os.path.join(dst, "_manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
        size = sum(manifest.values())
        total += size
        packed.append(name)
        print(f"  packed {name:<18} {size / 1e6:8.0f} MB")
    missing = [n for n in KD_MODELS if n not in packed]
    print(f"\n{len(packed)} models (+ the baseline, downloaded by the notebook), {total / 1e9:.2f} GB in {KD_OUT}")
    if missing:
        print("Not packed yet:", ", ".join(missing))
    print("\nNext: drive.google.com > New > Folder upload > select this folder:\n  " + KD_OUT)


def main():
    if "--kd" in sys.argv:
        return main_kd()
    # Regenerate the notebook so it always carries the current evaluation code.
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "make_t4_notebook.py")], check=True)

    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "models"))
    link_or_copy(NOTEBOOK, os.path.join(OUT, "t4_eval.ipynb"))

    # Only the MATRIX configs; extra-seed runs (<name>_s<seed>_cal) stay local.
    names = sorted(n for n in os.listdir(MODELS) if n in BASE) if os.path.isdir(MODELS) else []
    packed, total = [], 0
    for name in names:
        src = os.path.join(MODELS, name)
        done = os.path.exists(os.path.join(RESULTS, f"eval_{name}.json"))
        if not (done and os.path.exists(os.path.join(src, "model.safetensors"))):
            print(f"  not ready, skipped: {name}")
            continue
        dst = os.path.join(OUT, "models", name)
        os.makedirs(dst)
        manifest = {}
        for f in sorted(os.listdir(src)):
            p = os.path.join(src, f)
            if os.path.isfile(p):
                link_or_copy(p, os.path.join(dst, f))
                manifest[f] = os.path.getsize(p)
        for f in add_slow_tokenizer_files(name, src, dst):
            manifest[f] = os.path.getsize(os.path.join(dst, f))
        with open(os.path.join(dst, "_manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
        size = sum(manifest.values())
        total += size
        packed.append(name)
        print(f"  packed {name:<24} {size / 1e6:8.0f} MB")

    expected = [c[0] + "_cal" for c in MATRIX]
    missing = [n for n in expected if n not in packed]
    print(f"\n{len(packed)} models, {total / 1e9:.2f} GB in {OUT}")
    if missing:
        print("Not packed yet (training not finished):", ", ".join(missing))
    print("\nNext: drive.google.com > New > Folder upload > select this folder:\n  " + OUT)


if __name__ == "__main__":
    main()
