"""
tune_phobert.py
===============
Tuning siêu tham số cho PhoBERT bằng Optuna.
Lưu kết quả tốt nhất ra file JSON.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import optuna
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    EarlyStoppingCallback,
)

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ai_engine.data.phobert_dataset import LABEL_MAP, ID_TO_LABEL, PhoBertReviewDataset
from ai_engine.models.phobert_trainer import FocalLossTrainer
from scripts.train_phobert import (
    MODEL_CHECKPOINT,
    NUM_LABELS,
    set_seed,
    load_datasets,
    compute_class_counts,
    make_compute_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Tuning siêu tham số PhoBERT bằng Optuna.")
    parser.add_argument("--n_trials", type=int, default=10, help="Số lần thử nghiệm (trials) của Optuna.")
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch tối đa mỗi lần thử.")
    parser.add_argument("--max_length", type=int, default=256, help="Độ dài token tối đa (phải khớp với lúc train).")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Giới hạn số mẫu train để tuning nhanh. 0 = dùng toàn bộ (khuyến nghị với dữ liệu lệch lớp nặng).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    output_dir = REPO_ROOT / "artifacts" / "model" / "tuned" / "phobert_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data — load_datasets returns (train, val, test); tuning only requires train + val.
    train_df, val_df, _test_df = load_datasets(
        text_column="cleaned_text", label_column="sentiment_label"
    )

    # Stratified subsampling for fast tuning. sample_size <= 0 means use all data.
    # Stratification is mandatory here: with a ~94/5/1 class distribution, random subsampling
    # could drop minority classes completely, rendering eval_f1_macro noisy and leading to wrong hyperparameters.
    if args.sample_size and len(train_df) > args.sample_size:
        train_df, _ = train_test_split(
            train_df,
            train_size=args.sample_size,
            stratify=train_df["sentiment_label"],
            random_state=42,
        )
        train_df = train_df.reset_index(drop=True)

        val_cap = max(args.sample_size // 4, NUM_LABELS)
        if len(val_df) > val_cap:
            val_df, _ = train_test_split(
                val_df,
                train_size=val_cap,
                stratify=val_df["sentiment_label"],
                random_state=42,
            )
            val_df = val_df.reset_index(drop=True)
    logger.info("Tuning trên → train: %d, val: %d", len(train_df), len(val_df))
    train_int_labels = [LABEL_MAP[lbl] for lbl in train_df["sentiment_label"]]
    class_counts = compute_class_counts(train_int_labels)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    train_dataset = PhoBertReviewDataset.from_dataframe(
        df=train_df,
        text_column="cleaned_text",
        label_column="sentiment_label",
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_max_length=False,
    )
    val_dataset = PhoBertReviewDataset.from_dataframe(
        df=val_df,
        text_column="cleaned_text",
        label_column="sentiment_label",
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_max_length=False,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_CHECKPOINT,
            num_labels=NUM_LABELS,
            id2label=ID_TO_LABEL,
            label2id=LABEL_MAP,
        )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="no",          # Do not save checkpoints during tuning to avoid running out of disk space
        logging_strategy="epoch",
        load_best_model_at_end=False,  # Must be False when save_strategy="no"
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        disable_tqdm=False,
    )

    # gamma will be overridden by Optuna directly on trainer.gamma in each trial (see optuna_hp_space)
    trainer = FocalLossTrainer(
        class_counts=class_counts,
        gamma=2.0,  # initialization value, overridden in each trial
        num_classes=NUM_LABELS,
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(ID_TO_LABEL),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    def optuna_hp_space(trial):
        # Directly override self.gamma of trainer. The trainer object is reused
        # across trials (only the model is re-initialized by model_init), so FocalLossTrainer.compute_loss
        # will read the updated gamma for each trial. gamma is still tracked by Optuna in best_params.
        trainer.gamma = trial.suggest_float("gamma", 1.0, 3.0)
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
        }

    # Optimize directly on macro-F1 (default in HF is sum of metrics, which is less clear).
    def compute_objective(metrics):
        return metrics["eval_f1_macro"]

    logger.info("Bắt đầu quá trình Tuning với %d trials...", args.n_trials)
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        compute_objective=compute_objective,
        n_trials=args.n_trials,
    )

    logger.info("Hoàn thành Tuning! Best Run: %s", best_run)

    # Save to JSON
    best_params = best_run.hyperparameters
    out_json = REPO_ROOT / "artifacts" / "metrics" / "phobert_best_params.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
        
    logger.info("Đã lưu tham số tốt nhất ra: %s", out_json)


if __name__ == "__main__":
    main()
