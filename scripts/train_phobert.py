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

def load_and_split(
    data_path: str,
    text_column: str,
    label_column: str,
    val_size: float,
    seed: int,
) -> tuple:
    """Load the CSV, drop bad rows, and return stratified train/val splits.

    Args:
        data_path: Path to the cleaned CSV file.
        text_column: Column name containing review text.
        label_column: Column name containing Vietnamese sentiment labels.
        val_size: Fraction of data reserved for validation (0 < val_size < 1).
        seed: Random seed for the stratified split.

    Returns:
        Tuple of ``(train_df, val_df)`` pandas DataFrames.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    logger.info("Loading dataset from %s …", path)
    df = pd.read_csv(path)
    logger.info("Raw shape: %s", df.shape)

    # Validate columns.
    for col in (text_column, label_column):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available: {df.columns.tolist()}"
            )

    # Drop rows with missing text or unknown labels.
    valid_labels = set(LABEL_MAP.keys())
    df = df.dropna(subset=[text_column, label_column])
    df = df[df[label_column].isin(valid_labels)].reset_index(drop=True)
    logger.info("Usable rows after cleaning: %d", len(df))

    # Log class distribution.
    dist = df[label_column].value_counts()
    logger.info("Label distribution:\n%s", dist.to_string())

    # Stratified split to preserve class ratios in both sets.
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df[label_column],
        random_state=seed,
    )
    logger.info(
        "Split → train: %d rows, val: %d rows", len(train_df), len(val_df)
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


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
        default="processed_labeled_reviews.csv",
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
        default="ai_engine/models/weights/phobert_best",
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
    train_df, val_df = load_and_split(
        data_path=args.data_path,
        text_column=args.text_column,
        label_column=args.label_column,
        val_size=args.val_size,
        seed=args.seed,
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
    logger.info(
        "Datasets ready – train: %d, val: %d", len(train_dataset), len(val_dataset)
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
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=0.1,             # ~10 % of steps for LR warm-up
        lr_scheduler_type="cosine",   # cosine decay after warm-up
        evaluation_strategy="epoch",
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
    metrics_path = output_dir / "train_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        for k, v in train_result.metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info("Training metrics saved to %s", metrics_path)

    # ---- Final evaluation ---------------------------------------------------
    logger.info("Running final evaluation on validation set …")
    eval_metrics = trainer.evaluate()
    logger.info("Final eval metrics: %s", eval_metrics)

    eval_path = output_dir / "eval_metrics.txt"
    with open(eval_path, "w", encoding="utf-8") as f:
        for k, v in eval_metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info("Eval metrics saved to %s", eval_path)
    logger.info("Done! Best model checkpoint: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
