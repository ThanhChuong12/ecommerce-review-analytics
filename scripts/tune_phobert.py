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

    # 1. Load data — load_datasets trả về (train, val, test); tuning chỉ cần train + val.
    train_df, val_df, _test_df = load_datasets(
        text_column="cleaned_text", label_column="sentiment_label"
    )

    # Subsample CÓ PHÂN TẦNG (stratified) để tuning nhanh. sample_size <= 0 nghĩa là dùng toàn bộ.
    # Stratify là bắt buộc ở đây: với phân bố ~94/5/1, subsample ngẫu nhiên có thể xoá sạch
    # lớp hiếm khiến eval_f1_macro chỉ còn là nhiễu, dẫn tới chọn sai siêu tham số.
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
        save_strategy="no",          # Không lưu checkpoint khi tune → tránh đầy đĩa
        logging_strategy="epoch",
        load_best_model_at_end=False,  # Phải False khi save_strategy="no"
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        disable_tqdm=False,
    )

    # gamma sẽ được Optuna ghi đè trực tiếp lên trainer.gamma trong mỗi trial (xem optuna_hp_space)
    trainer = FocalLossTrainer(
        class_counts=class_counts,
        gamma=2.0,  # giá trị khởi tạo, không quan trọng vì sẽ bị ghi đè mỗi trial
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
        # Ghi đè trực tiếp self.gamma của trainer. Đối tượng trainer được tái sử dụng
        # qua các trial (chỉ model bị tạo lại bởi model_init), nên FocalLossTrainer.compute_loss
        # sẽ đọc đúng gamma mới ở mỗi trial. gamma vẫn được Optuna lưu vào best_params.
        trainer.gamma = trial.suggest_float("gamma", 1.0, 3.0)
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
        }

    # Tối ưu trực tiếp theo macro-F1 (mặc định của HF là tổng các metric — kém rõ ràng).
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
