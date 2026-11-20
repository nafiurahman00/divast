"""Create models/_base_codeberta_left: CodeBERTa-small-v1 with truncation_side="left".

The organisers' evaluation truncates every function to 512 tokens. For long
functions, keeping the END instead of the START leaves more SATD tags visible
(441 vs 432 of 505 SATD-positive test functions). The setting is stored in
tokenizer_config.json, so it travels with the model and needs no change to the
organisers' notebook. Training from this folder (instead of the Hub id) gives the
same initial weights; only the truncation side differs.

Run with:  python src/make_left_base.py
"""
import json
import os
import shutil

from huggingface_hub import hf_hub_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

from common import PATHS

BASE_ID = "huggingface/CodeBERTa-small-v1"
OUT = os.path.join(PATHS.models, "_base_codeberta_left")


def main():
    os.makedirs(OUT, exist_ok=True)
    AutoTokenizer.from_pretrained(BASE_ID).save_pretrained(OUT)
    # transformers 5.x does not save a truncation_side set as an attribute, but it
    # reads the key from tokenizer_config.json (as 4.x does) and keeps it on save.
    cfg_path = os.path.join(OUT, "tokenizer_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["truncation_side"] = "left"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    AutoModelForMaskedLM.from_pretrained(BASE_ID).save_pretrained(OUT)
    # The slow-tokenizer files the organisers' RobertaTokenizer needs on transformers 4.x.
    for fname in ("vocab.json", "merges.txt"):
        shutil.copy2(hf_hub_download(BASE_ID, fname), os.path.join(OUT, fname))
    print("wrote", OUT, sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
