"""
PhoBERT Sentiment Classification – Training Script.

End-to-end MLOps-grade training pipeline for the Vietnamese e-commerce
review sentiment model based on ``vinai/phobert-base-v2``.

Key features:
    * Global reproducibility seed (Python, NumPy, PyTorch, CUDA).
    * Macro-F1 / Precision / Recall metrics (appropriate for ~94/5/1 skew).
    * EarlyStoppingCallback on ``eval_loss`` to prevent overfitting.
    * Dynamic class-weight (alpha) computation from training labels.
    * Dynamic padding via ``DataCollatorWithPadding`` to save VRAM.
    * Saves the best checkpoint to ``ai_engine/models/weights/phobert_best/``.

Usage::

    python scripts/train_phobert.py \\
        --data_path processed_labeled_reviews.csv \\
        --text_column cleaned_text \\
        --label_column sentiment_label \\
        --output_dir ai_engine/models/weights/phobert_best \\
        --epochs 4 \\
        --batch_size 16 \\
        --lr 2e-5 \\
        --max_length 256 \\
        --seed 42

All arguments have sensible defaults; the script is runnable without any flags
if the CSV is in the working directory as ``processed_labeled_reviews.csv``.
"""

import argparse
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Local imports – adjust path so the script works from the repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ai_engine.data.phobert_dataset import LABEL_MAP, ID_TO_LABEL, PhoBertReviewDataset  # noqa: E402
from ai_engine.models.phobert_trainer import FocalLossTrainer  # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_CHECKPOINT = "vinai/phobert-base-v2"
NUM_LABELS = 3


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for full reproducibility.

    Sets seeds for Python's built-in ``random``, NumPy, PyTorch (CPU and
    all CUDA devices), and enables deterministic CuDNN behaviour.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN determinism (may slow training slightly).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d.", seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(id2label: Dict[int, str]):
    """Return a ``compute_metrics`` closure for HuggingFace Trainer.

    For a severely imbalanced dataset (94/5/1), accuracy is misleading – a
    trivial all-Positive classifier achieves ~94 %.  We therefore report:

    * **Macro-F1**: Equally weights all three classes regardless of support.
      This is the primary metric for early stopping and model selection.
    * **Macro-Precision** and **Macro-Recall**: Break F1 into its components.
    * **Per-class F1** (logged separately): Surfaces whether the minority
      classes (Negative, Neutral) are being learned.

    Args:
        id2label: Mapping from integer class index to label name string.

    Returns:
        A function compatible with HuggingFace Trainer's ``compute_metrics``
        parameter signature ``(EvalPrediction) -> Dict[str, float]``.
    """

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # Per-class breakdown (printed to logs for diagnostics).
        report = classification_report(
            labels,
            preds,
            target_names=[id2label[i] for i in range(NUM_LABELS)],
            zero_division=0,
        )
        logger.info("\n%s", report)

        return {
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "precision_macro": precision_score(
                labels, preds, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                labels, preds, average="macro", zero_division=0
            ),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_datasets(
    text_column: str,
    label_column: str,
) -> tuple:
    """Load the pre-split CSVs, drop bad rows, and return train/val splits.
    """
    train_path = REPO_ROOT / "data" / "processed" / "processed_labeled_text_train.csv"
    val_path = REPO_ROOT / "data" / "processed" / "processed_labeled_text_val.csv"
    test_path = REPO_ROOT / "data" / "processed" / "processed_labeled_text_test.csv"
    
    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing pre-split datasets in data/processed/")

    logger.info("Loading pre-split datasets ...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    valid_labels = set(LABEL_MAP.keys())
    
    train_df = train_df.dropna(subset=[text_column, label_column])
    train_df = train_df[train_df[label_column].isin(valid_labels)].reset_index(drop=True)
    
    val_df = val_df.dropna(subset=[text_column, label_column])
    val_df = val_df[val_df[label_column].isin(valid_labels)].reset_index(drop=True)

    test_df = test_df.dropna(subset=[text_column, label_column])
    test_df = test_df[test_df[label_column].isin(valid_labels)].reset_index(drop=True)

    logger.info("Split → train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))
    
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------

def compute_class_counts(labels: list) -> Dict[int, int]:
    """Count samples per integer class index.

    Args:
        labels: List of integer class labels.

    Returns:
        Dictionary mapping class index → sample count.
    """
    counts = Counter(labels)
    return {int(k): int(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with sensible defaults."""
    parser = argparse.ArgumentParser(
        description="Train PhoBERT for Vietnamese e-commerce sentiment analysis."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/processed_labeled_all.csv",
        help="Path to the cleaned CSV dataset.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="cleaned_text",
        help="CSV column containing review text.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="sentiment_label",
        help="CSV column containing Vietnamese sentiment labels.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/model/tuned/phobert",
        help="Directory where the best model checkpoint is saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Per-device training and evaluation batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Peak learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="L2 weight decay for AdamW (applied to non-bias params).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum tokenisation length (sequences are truncated, not padded here).",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Fraction of data held out for validation.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Focal Loss focusing parameter gamma.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
        help="Number of evaluations with no improvement before stopping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log training loss every N steps.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the full training pipeline."""
    args = parse_args()

    # ---- Seed ---------------------------------------------------------------
    set_seed(args.seed)

    # ---- Output directory ---------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Model will be saved to: %s", output_dir.resolve())

    # ---- Data ---------------------------------------------------------------
    train_df, val_df, test_df = load_datasets(
        text_column=args.text_column,
        label_column=args.label_column,
    )

    # Integer labels for the training set (needed for class-weight computation).
    train_int_labels = [LABEL_MAP[lbl] for lbl in train_df[args.label_column]]
    class_counts = compute_class_counts(train_int_labels)
    logger.info("Training class counts (int → count): %s", class_counts)

    # ---- Tokeniser ----------------------------------------------------------
    logger.info("Loading tokeniser: %s …", MODEL_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # ---- Datasets -----------------------------------------------------------
    # Dynamic padding: no padding here; DataCollatorWithPadding pads each batch.
    train_dataset = PhoBertReviewDataset.from_dataframe(
        df=train_df,
        text_column=args.text_column,
        label_column=args.label_column,
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_max_length=False,  # dynamic padding
    )
    val_dataset = PhoBertReviewDataset.from_dataframe(
        df=val_df,
        text_column=args.text_column,
        label_column=args.label_column,
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_max_length=False,
    )
    test_dataset = PhoBertReviewDataset.from_dataframe(
        df=test_df,
        text_column=args.text_column,
        label_column=args.label_column,
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_max_length=False,
    )
    logger.info(
        "Datasets ready – train: %d, val: %d, test: %d", len(train_dataset), len(val_dataset), len(test_dataset)
    )

    # Dynamic-padding collator: pads each batch to its longest sequence.
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # ---- Model --------------------------------------------------------------
    logger.info("Loading model: %s …", MODEL_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
    )

    # ---- Training arguments -------------------------------------------------
    # Evaluation happens every epoch; ``load_best_model_at_end=True`` combined
    # with EarlyStoppingCallback ensures the best checkpoint is preserved.
    # Load tuned parameters if available
    tuned_params_file = REPO_ROOT / "artifacts" / "metrics" / "phobert_best_params.json"
    lr = args.lr
    weight_decay = args.weight_decay
    warmup_ratio = 0.1
    batch_size = args.batch_size
    
    if tuned_params_file.exists():
        with open(tuned_params_file, "r") as f:
            best_params = json.load(f)
            lr = best_params.get("learning_rate", lr)
            weight_decay = best_params.get("weight_decay", weight_decay)
            warmup_ratio = best_params.get("warmup_ratio", warmup_ratio)
            batch_size = best_params.get("per_device_train_batch_size", batch_size)
            logger.info("Loaded tuned parameters: LR=%f, WD=%f, Warmup=%f, BatchSize=%d", lr, weight_decay, warmup_ratio, batch_size)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,             # ~10 % of steps for LR warm-up
        lr_scheduler_type="cosine",   # cosine decay after warm-up
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),     # AMP on CUDA for speed/VRAM
        seed=args.seed,
        report_to="none",                   # disable W&B / MLflow by default
        save_total_limit=2,                 # keep only last 2 checkpoints
        dataloader_num_workers=0,           # safe default for Windows
    )

    # ---- Trainer ------------------------------------------------------------
    trainer = FocalLossTrainer(
        class_counts=class_counts,
        gamma=args.gamma,
        num_classes=NUM_LABELS,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(ID_TO_LABEL),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )

    # ---- Train --------------------------------------------------------------
    logger.info("Starting training …")
    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # ---- Save best model ----------------------------------------------------
    logger.info("Saving best model to %s …", output_dir.resolve())
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training metrics alongside the model.
    metrics_path = output_dir / "phobert_train_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        for k, v in train_result.metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info("Training metrics saved to %s", metrics_path)

    metrics_txt_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_txt_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- Final evaluation on Validation Set ---------------------------------
    logger.info("Đánh giá mô hình trên tập Val ...")
    val_metrics = trainer.evaluate()
    logger.info("Kết quả tập Val: %s", val_metrics)
    
    val_path = metrics_txt_dir / "phobert_val_metrics.txt"
    with open(val_path, "w", encoding="utf-8") as f:
        for k, v in val_metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info("Đã lưu kết quả Val ra %s", val_path)

    # ---- Final evaluation on Test Set ---------------------------------------
    logger.info("Đánh giá mô hình trên tập Test ...")
    test_result = trainer.predict(test_dataset)
    test_metrics = test_result.metrics
    logger.info("Kết quả tập Test: %s", test_metrics)

    eval_path = metrics_txt_dir / "phobert_test_metrics.txt"
    with open(eval_path, "w", encoding="utf-8") as f:
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info("Đã lưu kết quả Test ra %s", eval_path)
    
    plot_out_dir = REPO_ROOT / "artifacts" / "plot"
    plot_out_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- Vẽ biểu đồ (Learning Curves & Confusion Matrix) --------------------
    logger.info("Vẽ biểu đồ Learning Curves và Confusion Matrix...")
    
    # 1. Learning Curves
    history = trainer.state.log_history
    train_loss = [h["loss"] for h in history if "loss" in h and "epoch" in h]
    val_loss = [h["eval_loss"] for h in history if "eval_loss" in h and "epoch" in h]
    val_f1 = [h["eval_f1_macro"] for h in history if "eval_f1_macro" in h and "epoch" in h]
    epochs_train = [h["epoch"] for h in history if "loss" in h]
    epochs_val = [h["epoch"] for h in history if "eval_loss" in h]

    # Plot Loss Curve
    plt.figure(figsize=(8, 6))
    if train_loss and epochs_train:
        plt.plot(epochs_train, train_loss, label='Mất mát Huấn luyện', marker='o', color='#1f77b4')
    if val_loss and epochs_val:
        plt.plot(epochs_val, val_loss, label='Mất mát Kiểm định', marker='o', color='#ff7f0e')
    plt.title('Biểu đồ hàm mất mát của mô hình PhoBERT', fontsize=15, pad=15)
    plt.xlabel('Vòng lặp', fontsize=12)
    plt.ylabel('Giá trị mất mát', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_out_dir / "phobert_loss_curve.png", dpi=300)
    plt.close()
    
    # Plot F1-Score Curve
    plt.figure(figsize=(8, 6))
    if val_f1 and epochs_val:
        plt.plot(epochs_val, val_f1, label='Điểm F1 kiểm định', marker='s', color='#2ca02c')
    plt.title('Biểu đồ điểm F1 của mô hình PhoBERT', fontsize=15, pad=15)
    plt.xlabel('Vòng lặp', fontsize=12)
    plt.ylabel('Điểm F1', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_out_dir / "phobert_f1_curve.png", dpi=300)
    plt.close()

    # 2. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    y_true = test_result.label_ids
    y_pred = np.argmax(test_result.predictions, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    class_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})
    plt.title('Ma trận nhầm lẫn của mô hình PhoBERT trên tập kiểm tra', fontsize=15, pad=15)
    plt.xlabel('Nhãn dự đoán', fontsize=12)
    plt.ylabel('Nhãn thực tế', fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_out_dir / "phobert_confusion_matrix.png", dpi=300)
    plt.close()
    
    logger.info("Done! Đã lưu biểu đồ vào thư mục artifacts/plot/")


if __name__ == "__main__":
    main()
