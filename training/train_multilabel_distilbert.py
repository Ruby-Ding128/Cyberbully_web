"""使用轻量化 DistilBERT 训练网络欺凌多标签分类模型。

python training/train_multilabel_distilbert.py \
  --data-path data/combined_selected_categories.csv \
  --output-dir outputs/distilbert_multilabel \
  --epochs 3 \
  --train-batch-size 16 \
  --gradient-accumulation-steps 2 \
  --max-length 512


"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


BASE_LABELS = [
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit",
]
NEW_LABELS = ["age", "ethnicity", "gender", "religion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DistilBERT 多标签分类训练")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/combined_selected_categories.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/distilbert_multilabel"))
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--include-target", action="store_true", help="把总体毒性 target 也作为标签")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None, help="快速试跑时随机抽样")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_text_hash(text: str) -> str:
    normalized = " ".join(str(text).lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def load_and_split(
    data_path: Path,
    label_columns: list[str],
    seed: int,
    sample_size: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = ["comment_text", *label_columns]
    frame = pd.read_csv(data_path, usecols=required, low_memory=False)
    frame = frame.dropna(subset=["comment_text"]).copy()
    frame["comment_text"] = frame["comment_text"].astype(str).str.strip()
    frame = frame[frame["comment_text"].ne("")].reset_index(drop=True)

    for column in label_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].dropna().loc[lambda x: (x < 0) | (x > 1)]
        if not invalid.empty:
            raise ValueError(f"标签 {column} 存在不在 [0, 1] 范围内的值")

    if sample_size and sample_size < len(frame):
        frame = frame.sample(sample_size, random_state=seed).reset_index(drop=True)

    frame["text_group"] = frame["comment_text"].map(normalized_text_hash)
    first_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, holdout_idx = next(first_split.split(frame, groups=frame["text_group"]))
    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    holdout = frame.iloc[holdout_idx].reset_index(drop=True)

    second_split = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    val_idx, test_idx = next(second_split.split(holdout, groups=holdout["text_group"]))
    val_frame = holdout.iloc[val_idx].reset_index(drop=True)
    test_frame = holdout.iloc[test_idx].reset_index(drop=True)
    return train_frame, val_frame, test_frame


def to_hf_dataset(
    frame: pd.DataFrame,
    tokenizer,
    label_columns: list[str],
    max_length: int,
) -> Dataset:
    targets = frame[label_columns].fillna(0.0).to_numpy(dtype=np.float32)
    masks = frame[label_columns].notna().to_numpy(dtype=np.float32)
    packed_labels = np.concatenate([targets, masks], axis=1)
    dataset = Dataset.from_dict(
        {
            "text": frame["comment_text"].tolist(),
            "labels": packed_labels.tolist(),
        }
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False,
        )

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def sigmoid(array: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(array, -30, 30)))


def masked_metrics(
    logits: np.ndarray,
    packed_labels: np.ndarray,
    num_labels: int,
    thresholds: np.ndarray,
) -> dict[str, float]:
    targets = packed_labels[:, :num_labels]
    masks = packed_labels[:, num_labels:].astype(bool)
    probabilities = sigmoid(logits)
    predictions = probabilities >= thresholds.reshape(1, -1)
    binary_targets = targets >= 0.5

    per_label_f1 = []
    aucs = []
    for index in range(num_labels):
        valid = masks[:, index]
        if not valid.any():
            continue
        truth = binary_targets[valid, index]
        pred = predictions[valid, index]
        per_label_f1.append(f1_score(truth, pred, zero_division=0))
        if np.unique(truth).size == 2:
            aucs.append(roc_auc_score(truth, probabilities[valid, index]))

    flat_valid = masks.ravel()
    flat_truth = binary_targets.ravel()[flat_valid]
    flat_pred = predictions.ravel()[flat_valid]
    return {
        "f1_micro": float(f1_score(flat_truth, flat_pred, zero_division=0)),
        "f1_macro": float(np.mean(per_label_f1)) if per_label_f1 else 0.0,
        "precision_micro": float(precision_score(flat_truth, flat_pred, zero_division=0)),
        "recall_micro": float(recall_score(flat_truth, flat_pred, zero_division=0)),
        "roc_auc_macro": float(np.mean(aucs)) if aucs else 0.0,
    }


def optimize_thresholds(
    logits: np.ndarray,
    packed_labels: np.ndarray,
    num_labels: int,
) -> np.ndarray:
    targets = packed_labels[:, :num_labels] >= 0.5
    masks = packed_labels[:, num_labels:].astype(bool)
    probabilities = sigmoid(logits)
    candidates = np.arange(0.10, 0.91, 0.05)
    thresholds = np.full(num_labels, 0.5, dtype=np.float32)
    for index in range(num_labels):
        valid = masks[:, index]
        if not valid.any() or np.unique(targets[valid, index]).size < 2:
            continue
        scores = [
            f1_score(targets[valid, index], probabilities[valid, index] >= value, zero_division=0)
            for value in candidates
        ]
        thresholds[index] = candidates[int(np.argmax(scores))]
    return thresholds


class MaskedMultilabelTrainer(Trainer):
    def __init__(self, *args, num_labels: int, pos_weight: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_task_labels = num_labels
        self.pos_weight = pos_weight
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        packed = inputs.pop("labels")
        targets = packed[:, : self.num_task_labels]
        masks = packed[:, self.num_task_labels :]
        outputs = model(**inputs)
        element_loss = F.binary_cross_entropy_with_logits(
            outputs.logits,
            targets,
            pos_weight=self.pos_weight.to(outputs.logits.device),
            reduction="none",
        )
        loss = (element_loss * masks).sum() / masks.sum().clamp_min(1.0)
        return (loss, outputs) if return_outputs else loss


def calculate_pos_weight(frame: pd.DataFrame, labels: list[str]) -> torch.Tensor:
    values = frame[labels]
    observed = values.notna().sum(axis=0).to_numpy(dtype=np.float32)
    positives = (values >= 0.5).sum(axis=0).to_numpy(dtype=np.float32)
    negatives = observed - positives
    weights = negatives / np.maximum(positives, 1.0)
    return torch.tensor(np.clip(weights, 1.0, 20.0), dtype=torch.float32)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    label_columns = (["target"] if args.include_target else []) + BASE_LABELS + NEW_LABELS
    train_frame, val_frame, test_frame = load_and_split(
        args.data_path, label_columns, args.seed, args.sample_size
    )

    split_manifest = {
        "train_rows": len(train_frame),
        "validation_rows": len(val_frame),
        "test_rows": len(test_frame),
        "labels": label_columns,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_dataset = to_hf_dataset(train_frame, tokenizer, label_columns, args.max_length)
    val_dataset = to_hf_dataset(val_frame, tokenizer, label_columns, args.max_length)
    test_dataset = to_hf_dataset(test_frame, tokenizer, label_columns, args.max_length)

    id2label = {index: label for index, label in enumerate(label_columns)}
    label2id = {label: index for index, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_columns),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    default_thresholds = np.full(len(label_columns), args.threshold, dtype=np.float32)

    def compute_metrics(evaluation):
        return masked_metrics(
            evaluation.predictions,
            evaluation.label_ids,
            len(label_columns),
            default_thresholds,
        )

    fp16 = torch.cuda.is_available() and not args.no_fp16
    # transformers 不同版本曾使用 evaluation_strategy / eval_strategy 两种参数名，
    # 部分旧版也不支持 warmup_ratio。这里根据运行环境自动选择兼容参数。
    training_signature = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs = {
        "output_dir": str(args.output_dir / "checkpoints"),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 100,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "save_total_limit": 2,
        "fp16": fp16,
        "dataloader_num_workers": args.num_workers,
        "report_to": "none",
        "seed": args.seed,
    }
    if "eval_strategy" in training_signature:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in training_signature:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        raise RuntimeError("当前 transformers 版本过旧，不支持训练期间验证策略")

    if "warmup_ratio" in training_signature:
        training_kwargs["warmup_ratio"] = args.warmup_ratio
    else:
        batches_per_epoch = math.ceil(len(train_dataset) / args.train_batch_size)
        update_steps_per_epoch = math.ceil(
            batches_per_epoch / args.gradient_accumulation_steps
        )
        total_update_steps = math.ceil(update_steps_per_epoch * args.epochs)
        training_kwargs["warmup_steps"] = round(total_update_steps * args.warmup_ratio)
        print(
            "当前 transformers 不支持 warmup_ratio，"
            f"已自动换算为 warmup_steps={training_kwargs['warmup_steps']}"
        )

    unsupported = sorted(set(training_kwargs) - set(training_signature))
    if unsupported:
        raise RuntimeError(
            "当前 transformers 版本还不支持这些 TrainingArguments 参数："
            + ", ".join(unsupported)
        )
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
        "num_labels": len(label_columns),
        "pos_weight": calculate_pos_weight(train_frame, label_columns),
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = MaskedMultilabelTrainer(
        **trainer_kwargs,
    )

    trainer.train()
    validation_output = trainer.predict(val_dataset, metric_key_prefix="validation")
    thresholds = optimize_thresholds(
        validation_output.predictions, validation_output.label_ids, len(label_columns)
    )
    test_output = trainer.predict(test_dataset, metric_key_prefix="test")
    final_metrics = masked_metrics(
        test_output.predictions,
        test_output.label_ids,
        len(label_columns),
        thresholds,
    )

    final_model_dir = args.output_dir / "final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    threshold_map = {label: float(value) for label, value in zip(label_columns, thresholds)}
    (final_model_dir / "thresholds.json").write_text(
        json.dumps(threshold_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"thresholds": threshold_map, "test_metrics": final_metrics}, indent=2))


if __name__ == "__main__":
    main()
