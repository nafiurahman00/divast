"""Fine-tune a shared-encoder multitask classifier for the NLBSE'27 competition.

One RoBERTa-family encoder + a 2-logit multi-label head (sigmoid / BCE).
The same architecture (RobertaForSequenceClassification, num_labels=2) that the
organisers' evaluation loop expects, so the saved directory is a drop-in model.

Key knobs for the competition trade-off (predictive F1 vs runtime/GFLOPs):
  --model      microsoft/codebert-base  (12-layer, baseline)
               huggingface/CodeBERTa-small-v1 (6-layer, ~half the FLOPs)
               distilroberta-base, microsoft/graphcodebert-base, ...
  --loss       bce | weighted (neg/pos BCE) | focal, to fight MAT imbalance
  --max-len    training truncation length

Example (full run, GPU):
  python src/train.py --model huggingface/CodeBERTa-small-v1 \
      --epochs 3 --bs 32 --lr 3e-5 --max-len 400 --loss weighted --fp16 \
      --output-dir models/codeberta_small
Smoke test (CPU, tiny):
  python src/train.py --model huggingface/CodeBERTa-small-v1 \
      --subset 200 --epochs 1 --bs 8 --max-len 128 --output-dir models/smoke
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil

import numpy as np

from common import (HF_DATASET, LABEL_COLUMN, LABEL_NAMES, NUM_LABELS, PATHS,
                    TEXT_COLUMN, per_task_prf1, set_seed)


def load_val_indices():
    path = os.path.join(PATHS.data, "val_indices.json")
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f)["indices"])
    return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="huggingface/CodeBERTa-small-v1")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--eval-bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--loss", choices=["bce", "weighted", "focal"], default="weighted",
                    help="bce=plain, weighted=neg/pos BCE weighting, focal=focal loss")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--subset", type=int, default=0,
                    help="if >0, use only this many train rows (smoke test)")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-steps", type=int, default=0,
                    help="evaluate+checkpoint every N steps (0 = every epoch); "
                         "a crashed or disconnected run resumes from the last checkpoint")
    ap.add_argument("--aug-file", action="append", default=[],
                    help="parquet of augmented training rows (src/augment_bt.py); repeatable")
    ap.add_argument("--aug-mode", choices=["add", "replace"], default="add",
                    help="add = append the augmented copies (NLBSE'25 recipe); replace = per epoch, "
                         "show a function as one of its variants with probability --aug-p")
    ap.add_argument("--aug-p", type=float, default=0.5)
    ap.add_argument("--student-layers", default="",
                    help="comma list of encoder layers of --model to keep, e.g. 0,2,4")
    ap.add_argument("--student-width", type=int, default=0,
                    help="if >0, narrow the student to this hidden size (FFN = 4x), see src/narrow.py")
    ap.add_argument("--teacher", default="", help="model dir/id for knowledge distillation")
    ap.add_argument("--kd-alpha", type=float, default=0.5)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--kd-loss", choices=["bce", "mse"], default="bce",
                    help="bce = BCE toward the teacher's tempered sigmoid; mse = plain logit MSE")
    ap.add_argument("--kd-hidden", type=float, default=0.0,
                    help="weight of the hidden-state loss: student layer j learns teacher "
                         "layer j*Lt/Ls (1 - cosine per token, padding masked); 0 = off")
    ap.add_argument("--keep-ckpt", action="store_true",
                    help="keep intermediate checkpoints after a successful run")
    args = ap.parse_args()
    set_seed(args.seed)

    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset {HF_DATASET} ...")
    ds = load_dataset(HF_DATASET)
    val_idx = load_val_indices()

    train_full = ds["train"]
    if val_idx:
        all_idx = np.arange(len(train_full))
        is_val = np.isin(all_idx, list(val_idx))
        train_split = train_full.select(np.where(~is_val)[0])
        val_split = train_full.select(np.where(is_val)[0])
    else:  # fallback: 5% tail
        n = len(train_full)
        cut = int(n * 0.95)
        train_split = train_full.select(range(cut))
        val_split = train_full.select(range(cut, n))

    if args.subset > 0:
        train_split = train_split.select(range(min(args.subset, len(train_split))))
        val_split = val_split.select(range(min(max(args.subset // 4, 20), len(val_split))))
    if args.aug_file and args.aug_mode == "replace":
        # Substitution: one version of the training part per epoch, in which a function
        # with back-translated variants is shown as a random variant with probability
        # --aug-p. Dataset size and label mix stay those of the original training part.
        import math

        import pandas as pd
        from datasets import concatenate_datasets
        assert val_idx and args.subset == 0
        train_rows = np.where(~is_val)[0]
        variants = {}
        for aug_path in args.aug_file:
            aug = pd.read_parquet(aug_path)
            assert not set(aug["src_idx"]) & val_idx, "augmented rows must come from the training part"
            for i, t in zip(aug["src_idx"], aug[TEXT_COLUMN]):
                variants.setdefault(int(i), []).append(t)
        base = train_split.select_columns([TEXT_COLUMN, LABEL_COLUMN])
        texts = base[TEXT_COLUMN]
        n_ep = math.ceil(args.epochs)
        rng = np.random.default_rng(args.seed)
        copies, swapped = [], 0
        for e in range(n_ep):
            t_e = list(texts)
            for pos, i in enumerate(train_rows):
                v = variants.get(int(i))
                if v and rng.random() < args.aug_p:
                    t_e[pos] = v[rng.integers(len(v))]
                    swapped += 1
            copies.append(base.remove_columns([TEXT_COLUMN]).add_column(TEXT_COLUMN, t_e))
        train_split = concatenate_datasets(copies)
        print(f"replace mode: {len(variants)} functions have variants; {n_ep} epoch versions, "
              f"{swapped} swaps in total; training 1 pass over {len(train_split)} rows")
        args.epochs = args.epochs / n_ep
    for aug_path in (args.aug_file if args.aug_mode == "add" else []):
        # Back-translated copies of training-part functions (src/augment_bt.py).
        # Added to the training part only; validation and test stay untouched.
        import pandas as pd
        from datasets import Dataset, Features, Sequence, Value, concatenate_datasets
        aug = pd.read_parquet(aug_path)
        assert not set(aug["src_idx"]) & val_idx, "augmented rows must come from the training part"
        feats = Features({TEXT_COLUMN: Value("string"), LABEL_COLUMN: Sequence(Value("int64"))})
        base = train_split.select_columns([TEXT_COLUMN, LABEL_COLUMN]).cast(feats)
        extra = Dataset.from_dict({TEXT_COLUMN: aug[TEXT_COLUMN].tolist(),
                                   LABEL_COLUMN: [list(map(int, l)) for l in aug[LABEL_COLUMN]]},
                                  features=feats)
        train_split = concatenate_datasets([base, extra])
        print(f"+ {len(extra)} augmented rows from {aug_path}")
    print(f"Train: {len(train_split)}  Val: {len(val_split)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        enc = tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=args.max_len)
        enc["labels"] = [[float(x) for x in lab] for lab in batch[LABEL_COLUMN]]
        return enc

    keep_cols = [TEXT_COLUMN, LABEL_COLUMN]
    train_tok = train_split.map(tokenize, batched=True,
                                remove_columns=train_split.column_names)
    val_tok = val_split.map(tokenize, batched=True,
                            remove_columns=val_split.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label={i: n for i, n in enumerate(LABEL_NAMES)},
        label2id={n: i for i, n in enumerate(LABEL_NAMES)},
    )
    teacher = None
    if args.student_layers:
        # Layer-dropping student: keep these encoder layers of --model (e.g. the
        # fine-tuned teacher), DistilBERT-style; the classifier head is kept too.
        keep = [int(x) for x in args.student_layers.split(",")]
        layers = model.roberta.encoder.layer
        model.roberta.encoder.layer = torch.nn.ModuleList([layers[i] for i in keep])
        model.config.num_hidden_layers = len(keep)
        print(f"student: kept layers {keep} of {len(layers)}")
    if args.student_width and args.kd_hidden > 0:
        raise SystemExit("--kd-hidden needs equal hidden sizes; it is not supported with --student-width")
    if args.student_width:
        # Narrow student: hidden size 768 -> W, FFN -> 4W, initialised from the
        # (layer-dropped) model by projection and head/neuron selection (src/narrow.py).
        from narrow import narrow_roberta
        model = narrow_roberta(model, args.student_width, 4 * args.student_width)
        print(f"student: hidden {args.student_width}, heads {model.config.num_attention_heads}, "
              f"ffn {model.config.intermediate_size}, params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    if args.teacher:
        teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher)
        teacher.eval().half().to("cuda" if torch.cuda.is_available() else "cpu")
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        print(f"distillation: teacher={args.teacher} alpha={args.kd_alpha} T={args.kd_temp}")

    tl = np.array(train_split[LABEL_COLUMN])
    pos = tl.sum(axis=0)
    neg = len(tl) - pos
    pos_weight = torch.tensor((neg / np.clip(pos, 1, None)), dtype=torch.float)
    print(f"loss={args.loss}  pos_weight={pos_weight.tolist()}")

    def focal_loss(logits, targets, gamma, alpha):
        """Multi-label focal loss built on BCE; alpha up-weights positives."""
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        mod = (1 - p_t) ** gamma
        if alpha is not None:
            a_t = alpha * targets + (1 - targets)
            bce = a_t * bce
        return (mod * bce).mean()

    class LossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels").float()
            use_hid = teacher is not None and model.training and args.kd_hidden > 0
            outputs = model(**inputs, output_hidden_states=use_hid)
            logits = outputs.logits
            if args.loss == "bce":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            elif args.loss == "weighted":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight.to(logits.device))
            else:  # focal: moderate per-class alpha = sqrt(neg/pos) to avoid over-weighting
                alpha = torch.sqrt(pos_weight).to(logits.device)
                loss = focal_loss(logits, labels, args.focal_gamma, alpha)
            if teacher is not None and model.training:
                # Soft-target BCE toward the teacher's tempered probabilities.
                with torch.no_grad():
                    t_out = teacher(**inputs, output_hidden_states=use_hid)
                    t_logits = t_out.logits.float()
                T = args.kd_temp
                if args.kd_loss == "mse":
                    soft = torch.nn.functional.mse_loss(logits.float(), t_logits)
                else:
                    soft = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits / T, torch.sigmoid(t_logits / T)) * T * T
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * soft
                if use_hid:
                    # hidden_states[0] is the embedding output; layer j of the student
                    # (1..Ls) is matched to teacher layer j*Lt/Ls, e.g. 1,2,3 -> 2,4,6.
                    s_h, t_h = outputs.hidden_states, t_out.hidden_states
                    ls, lt = len(s_h) - 1, len(t_h) - 1
                    mask = inputs["attention_mask"].float()
                    hid = 0.0
                    for j in range(1, ls + 1):
                        cos = torch.nn.functional.cosine_similarity(
                            s_h[j].float(), t_h[j * lt // ls].float(), dim=-1)
                        hid = hid + ((1 - cos) * mask).sum() / mask.sum()
                    loss = loss + args.kd_hidden * hid / ls
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)
        rows = per_task_prf1(preds, labels.astype(int))
        out = {f"f1_{r['cat']}": r["f1"] for r in rows}
        out["mean_f1"] = float(np.mean([r["f1"] for r in rows]))
        return out

    # Build TrainingArguments robustly across transformers 4.x / 5.x.
    total_steps = max(1, int(len(train_tok) / max(1, args.bs * args.grad_accum) * args.epochs))
    desired = dict(
        output_dir=os.path.join(args.output_dir, "_ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        eval_strategy="steps" if args.save_steps > 0 else "epoch",
        save_strategy="steps" if args.save_steps > 0 else "epoch",
        eval_steps=args.save_steps if args.save_steps > 0 else None,
        save_steps=args.save_steps if args.save_steps > 0 else None,
        save_total_limit=2,  # best + latest, so resuming is always possible
        load_best_model_at_end=True,
        metric_for_best_model="mean_f1",
        greater_is_better=True,
        fp16=args.fp16,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
    )
    sig = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "warmup_ratio" not in sig and "warmup_steps" in sig:
        desired["warmup_steps"] = int(args.warmup_ratio * total_steps)
    if "eval_strategy" not in sig and "evaluation_strategy" in sig:
        desired["evaluation_strategy"] = desired.pop("eval_strategy")
    targs = TrainingArguments(**{k: v for k, v in desired.items() if k in sig and v is not None})

    trainer_cls = LossTrainer
    trainer_kwargs = dict(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    # transformers >=5 renamed `tokenizer` to `processing_class`.
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = trainer_cls(**trainer_kwargs)

    from transformers.trainer_utils import get_last_checkpoint
    ckpt_dir = os.path.join(args.output_dir, "_ckpt")
    last = get_last_checkpoint(ckpt_dir) if os.path.isdir(ckpt_dir) else None
    if last:
        print(f"Resuming from checkpoint {last}")
    trainer.train(resume_from_checkpoint=last)
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    # Save the final model as a clean, self-contained directory.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    if not args.keep_ckpt:
        shutil.rmtree(ckpt_dir, ignore_errors=True)  # final model is saved; free disk/Drive
    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
