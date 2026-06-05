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
    parser = argparse.add_argument_group("Tuning Arguments")
    parser.add_argument("--n_trials", type=int, default=5, help="Số lần thử nghiệm (trials) của Optuna.")
    parser.add_argument("--epochs", type=int, default=2, help="Số epoch mỗi lần thử (nên để ít cho lẹ).")
    parser.add_argument("--sample_size", type=int, default=2000, help="Giới hạn số mẫu train để tuning nhanh.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    output_dir = REPO_ROOT / "artifacts" / "model" / "tuned" / "phobert_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_df, val_df = load_datasets(text_column="cleaned_text", label_column="sentiment_label")
    
    # Subsample to speed up tuning
    if len(train_df) > args.sample_size:
        train_df = train_df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
    if len(val_df) > args.sample_size // 4:
        val_df = val_df.sample(n=args.sample_size // 4, random_state=42).reset_index(drop=True)
        
    train_int_labels = [LABEL_MAP[lbl] for lbl in train_df["sentiment_label"]]
    class_counts = compute_class_counts(train_int_labels)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    train_dataset = PhoBertReviewDataset.from_dataframe(
        df=train_df, tokenizer=tokenizer, pad_to_max_length=False
    )
    val_dataset = PhoBertReviewDataset.from_dataframe(
        df=val_df, tokenizer=tokenizer, pad_to_max_length=False
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
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        disable_tqdm=True,
    )

    trainer = FocalLossTrainer(
        class_counts=class_counts,
        gamma=2.0,
        num_classes=NUM_LABELS,
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(ID_TO_LABEL),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16]),
        }

    logger.info("Bắt đầu quá trình Tuning với %d trials...", args.n_trials)
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
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
    import argparse
    import sys
    sys.argv = [sys.argv[0]] # Clear args to avoid conflict with parser
    main()
