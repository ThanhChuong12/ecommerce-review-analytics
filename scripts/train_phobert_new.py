"""
PhoBERT Sentiment Classification – Training Script with Augmentation.

Phiên bản mở rộng của train_phobert.py, bổ sung:
    * Random Oversampling với Text Augmentation cho lớp thiểu số.
    * Kỹ thuật augment: Random Word Deletion + Random Word Swap.
    * Mục tiêu: cân bằng phân bố lớp trong tập train mà không làm méo dữ liệu thật.

Chiến lược cân bằng:
    - Xác định lớp đa số (majority) và lớp thiểu số (minority).
    - Với mỗi lớp thiểu số, tạo thêm mẫu augment cho đến khi đạt
      ``target_ratio`` so với lớp đa số (mặc định 40%).
    - Mỗi mẫu augment được tạo bằng cách kết hợp ngẫu nhiên:
        (1) Random Deletion: xóa mỗi từ với xác suất p.
        (2) Random Swap: hoán đổi n cặp từ liền kề ngẫu nhiên.

Usage::

    python scripts/train_phobert_new.py \\
        --target_ratio 0.4 \\
        --aug_delete_prob 0.15 \\
        --aug_swap_n 3

All arguments have sensible defaults; the script is runnable without any flags.
"""

import argparse
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
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
# Local imports
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d.", seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(id2label: Dict[int, str]):
    """Return a compute_metrics closure for HuggingFace Trainer."""

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        report = classification_report(
            labels,
            preds,
            target_names=[id2label[i] for i in range(NUM_LABELS)],
            zero_division=0,
        )
        logger.info("\n%s", report)

        per_class_f1 = f1_score(
            labels, preds, average=None, labels=list(range(NUM_LABELS)), zero_division=0
        )

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "precision_macro": precision_score(
                labels, preds, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                labels, preds, average="macro", zero_division=0
            ),
        }
        for i in range(NUM_LABELS):
            metrics[f"f1_{id2label[i]}"] = float(per_class_f1[i])
        return metrics

    return compute_metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_datasets(text_column: str, label_column: str) -> tuple:
    train_path = REPO_ROOT / "data" / "processed" / "processed_labeled_text_train.csv"
    val_path   = REPO_ROOT / "data" / "processed" / "processed_labeled_text_val.csv"
    test_path  = REPO_ROOT / "data" / "processed" / "processed_labeled_text_test.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing pre-split datasets in data/processed/")

    logger.info("Loading pre-split datasets ...")
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    valid_labels = set(LABEL_MAP.keys())

    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.dropna(subset=[text_column, label_column], inplace=True)
        df = df[df[label_column].isin(valid_labels)].reset_index(drop=True)

    train_df = train_df[train_df[label_column].isin(valid_labels)].reset_index(drop=True)
    val_df   = val_df[val_df[label_column].isin(valid_labels)].reset_index(drop=True)
    test_df  = test_df[test_df[label_column].isin(valid_labels)].reset_index(drop=True)

    logger.info("Split → train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Text Augmentation helpers
# ---------------------------------------------------------------------------

def random_deletion(words: List[str], p: float = 0.15) -> List[str]:
    """Xóa ngẫu nhiên mỗi từ với xác suất p. Giữ ít nhất 1 từ."""
    if len(words) == 1:
        return words
    result = [w for w in words if random.random() > p]
    return result if result else [random.choice(words)]


def random_swap(words: List[str], n: int = 3) -> List[str]:
    """Hoán đổi ngẫu nhiên n cặp từ liền kề."""
    if len(words) < 2:
        return words
    result = words.copy()
    for _ in range(n):
        idx = random.randint(0, len(result) - 2)
        result[idx], result[idx + 1] = result[idx + 1], result[idx]
    return result


def augment_text(text: str, delete_prob: float = 0.15, swap_n: int = 3) -> str:
    """Áp dụng random deletion và random swap lên văn bản."""
    words = str(text).split()
    words = random_deletion(words, p=delete_prob)
    words = random_swap(words, n=swap_n)
    return " ".join(words)


# ---------------------------------------------------------------------------
# Oversampling + Augmentation
# ---------------------------------------------------------------------------

def oversample_with_augmentation(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    target_ratio: float = 0.4,
    delete_prob: float = 0.15,
    swap_n: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Oversample các lớp thiểu số bằng text augmentation.

    Với mỗi lớp thiểu số, tạo thêm mẫu augment cho đến khi số lượng
    đạt ``target_ratio * majority_count``.

    Args:
        df: DataFrame gốc.
        text_column: Tên cột văn bản.
        label_column: Tên cột nhãn.
        target_ratio: Tỉ lệ mục tiêu so với lớp đa số (0.0 → 1.0).
        delete_prob: Xác suất xóa từ trong random deletion.
        swap_n: Số cặp từ hoán đổi trong random swap.
        seed: Seed cho random.

    Returns:
        DataFrame đã được bổ sung các mẫu augment.
    """
    random.seed(seed)
    np.random.seed(seed)

    counts = df[label_column].value_counts()
    majority_count = counts.max()
    majority_label = counts.idxmax()
    target_count = int(majority_count * target_ratio)

    logger.info("Phân bố lớp TRƯỚC augmentation:\n%s", counts.to_string())
    logger.info(
        "Lớp đa số: '%s' (%d mẫu) | Mục tiêu thiểu số: %d mẫu (%.0f%%)",
        majority_label, majority_count, target_count, target_ratio * 100,
    )

    augmented_rows = []

    for label, count in counts.items():
        if label == majority_label:
            continue  # Bỏ qua lớp đa số

        needed = max(0, target_count - count)
        if needed == 0:
            logger.info("Lớp '%s' đủ mẫu, bỏ qua augmentation.", label)
            continue

        logger.info(
            "Lớp '%s': %d mẫu → cần thêm %d mẫu augment", label, count, needed
        )

        minority_df = df[df[label_column] == label].reset_index(drop=True)
        indices = np.random.randint(0, len(minority_df), size=needed)

        for idx in indices:
            row = minority_df.iloc[idx].copy()
            row[text_column] = augment_text(
                row[text_column], delete_prob=delete_prob, swap_n=swap_n
            )
            augmented_rows.append(row)

    if not augmented_rows:
        logger.info("Không có lớp nào cần augmentation.")
        return df

    aug_df = pd.DataFrame(augmented_rows)
    result_df = pd.concat([df, aug_df], ignore_index=True)

    new_counts = result_df[label_column].value_counts()
    logger.info("Phân bố lớp SAU augmentation:\n%s", new_counts.to_string())

    return result_df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------

def compute_class_counts(labels: list) -> Dict[int, int]:
    counts = Counter(labels)
    return {int(k): int(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PhoBERT với Oversampling + Text Augmentation."
    )
    parser.add_argument("--text_column",  type=str,   default="cleaned_text")
    parser.add_argument("--label_column", type=str,   default="sentiment_label")
    parser.add_argument("--output_dir",   type=str,   default="artifacts/model/tuned/phobert_aug")
    parser.add_argument("--epochs",       type=int,   default=4)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length",   type=int,   default=256)
    parser.add_argument("--gamma",        type=float, default=2.0)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--logging_steps",type=int,   default=50)
    # ---- Augmentation args ----
    parser.add_argument(
        "--target_ratio",
        type=float,
        default=0.4,
        help="Tỉ lệ mục tiêu của lớp thiểu số so với lớp đa số (0→1). Mặc định 0.4 = 40%%.",
    )
    parser.add_argument(
        "--aug_delete_prob",
        type=float,
        default=0.15,
        help="Xác suất xóa một từ trong random deletion.",
    )
    parser.add_argument(
        "--aug_swap_n",
        type=int,
        default=3,
        help="Số cặp từ hoán đổi trong random swap.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Model will be saved to: %s", output_dir.resolve())

    metrics_txt_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_txt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ---------------------------------------------------------------
    train_df, val_df, test_df = load_datasets(
        text_column=args.text_column,
        label_column=args.label_column,
    )

    # ---- Oversampling + Augmentation ----------------------------------------
    logger.info("=" * 60)
    logger.info("Bắt đầu Oversampling + Text Augmentation cho lớp thiểu số ...")
    train_df = oversample_with_augmentation(
        df=train_df,
        text_column=args.text_column,
        label_column=args.label_column,
        target_ratio=args.target_ratio,
        delete_prob=args.aug_delete_prob,
        swap_n=args.aug_swap_n,
        seed=args.seed,
    )
    logger.info("Tổng mẫu sau augmentation: %d", len(train_df))
    logger.info("=" * 60)

    # Integer labels sau augmentation
    train_int_labels = [LABEL_MAP[lbl] for lbl in train_df[args.label_column]]
    class_counts = compute_class_counts(train_int_labels)
    logger.info("Class counts sau augmentation (int → count): %s", class_counts)

    # ---- Tokeniser ----------------------------------------------------------
    logger.info("Loading tokeniser: %s …", MODEL_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # ---- Datasets -----------------------------------------------------------
    train_dataset = PhoBertReviewDataset.from_dataframe(
        df=train_df,
        text_column=args.text_column,
        label_column=args.label_column,
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_max_length=False,
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
        "Datasets ready – train (aug): %d, val: %d, test: %d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # ---- Model --------------------------------------------------------------
    logger.info("Loading model: %s …", MODEL_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
    )

    # ---- Load tuned hyperparameters if available ----------------------------
    tuned_params_file = REPO_ROOT / "artifacts" / "metrics" / "phobert_best_params.json"
    lr           = args.lr
    weight_decay = args.weight_decay
    warmup_ratio = 0.1
    batch_size   = args.batch_size
    num_epochs   = args.epochs
    gamma        = args.gamma

    if tuned_params_file.exists():
        with open(tuned_params_file, "r") as f:
            best_params = json.load(f)
        lr           = best_params.get("learning_rate",              lr)
        weight_decay = best_params.get("weight_decay",               weight_decay)
        warmup_ratio = best_params.get("warmup_ratio",               warmup_ratio)
        batch_size   = best_params.get("per_device_train_batch_size", batch_size)
        num_epochs   = best_params.get("num_train_epochs",           num_epochs)
        gamma        = best_params.get("gamma",                      gamma)
        logger.info(
            "Loaded tuned params → LR=%.2e, WD=%.4f, Warmup=%.3f, Batch=%d, Epochs=%d, Gamma=%.2f",
            lr, weight_decay, warmup_ratio, batch_size, num_epochs, gamma,
        )

    # ---- Training arguments -------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=0,
    )

    # ---- Trainer ------------------------------------------------------------
    trainer = FocalLossTrainer(
        class_counts=class_counts,
        gamma=gamma,
        num_classes=NUM_LABELS,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={"val": val_dataset, "train": train_dataset},
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(ID_TO_LABEL),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )

    # ---- Train --------------------------------------------------------------
    logger.info("Starting training (với augmented data) …")
    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # ---- Save best model ----------------------------------------------------
    logger.info("Saving best model to %s …", output_dir.resolve())
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # ---- Save training metrics ----------------------------------------------
    train_path = metrics_txt_dir / "phobert_aug_train_metrics.txt"
    with open(train_path, "w", encoding="utf-8") as f:
        for k, v in train_result.metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info("Training metrics saved to %s", train_path)

    # ---- Final evaluation on Validation Set ---------------------------------
    logger.info("Đánh giá mô hình trên tập Val ...")
    val_metrics = trainer.evaluate()
    val_path = metrics_txt_dir / "phobert_aug_val_metrics.txt"
    with open(val_path, "w", encoding="utf-8") as f:
        for k, v in val_metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info("Đã lưu kết quả Val ra %s", val_path)

    # ---- Final evaluation on Test Set ---------------------------------------
    logger.info("Đánh giá mô hình trên tập Test ...")
    test_result  = trainer.predict(test_dataset)
    test_metrics = test_result.metrics
    logger.info("Kết quả tập Test: %s", test_metrics)

    eval_path = metrics_txt_dir / "phobert_aug_test_metrics.txt"
    with open(eval_path, "w", encoding="utf-8") as f:
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info("Đã lưu kết quả Test ra %s", eval_path)

    # ---- Vẽ biểu đồ ---------------------------------------------------------
    plot_out_dir = REPO_ROOT / "artifacts" / "plot"
    plot_out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Vẽ biểu đồ Learning Curves và Confusion Matrix...")

    history      = trainer.state.log_history
    train_loss   = [h["loss"]               for h in history if "loss"               in h and "epoch" in h]
    val_loss     = [h["eval_val_loss"]      for h in history if "eval_val_loss"      in h]
    val_f1       = [h["eval_val_f1_macro"]  for h in history if "eval_val_f1_macro"  in h]
    train_f1     = [h["eval_train_f1_macro"] for h in history if "eval_train_f1_macro" in h]
    epochs_train = [h["epoch"]              for h in history if "loss"               in h]
    epochs_val   = [h["epoch"]              for h in history if "eval_val_loss"      in h]

    # Plot Loss Curve
    plt.figure(figsize=(8, 6))
    if train_loss and epochs_train:
        plt.plot(epochs_train, train_loss, label='Mất mát huấn luyện', marker='o', color='#1f77b4')
    if val_loss and epochs_val:
        plt.plot(epochs_val, val_loss,     label='Mất mát kiểm định',  marker='o', color='#ff7f0e')
    plt.title('Biểu đồ hàm mất mát của mô hình PhoBERT (Augmented)', fontsize=15, pad=15)
    plt.xlabel('Vòng lặp', fontsize=12)
    plt.ylabel('Giá trị mất mát', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_out_dir / "phobert_aug_loss_curve.png", dpi=300)
    plt.close()

    # Plot F1-Score Curve
    plt.figure(figsize=(8, 6))
    if train_f1 and epochs_val:
        plt.plot(epochs_val, train_f1, label='Điểm F1 huấn luyện', marker='o', color='#1f77b4')
    if val_f1 and epochs_val:
        plt.plot(epochs_val, val_f1,   label='Điểm F1 kiểm định',  marker='s', color='#ff7f0e')
    plt.title('Biểu đồ điểm F1 của mô hình PhoBERT (Augmented)', fontsize=15, pad=15)
    plt.xlabel('Vòng lặp', fontsize=12)
    plt.ylabel('Điểm F1', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_out_dir / "phobert_aug_f1_curve.png", dpi=300)
    plt.close()

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    y_true = test_result.label_ids
    y_pred = np.argmax(test_result.predictions, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    col_order = [LABEL_MAP['tích cực'], LABEL_MAP['trung lập'], LABEL_MAP['tiêu cực']]
    row_order = [LABEL_MAP['tiêu cực'], LABEL_MAP['trung lập'], LABEL_MAP['tích cực']]

    cm_reordered = cm[np.ix_(row_order, col_order)]
    x_labels = [ID_TO_LABEL[i] for i in col_order]
    y_labels = [ID_TO_LABEL[i] for i in row_order]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_reordered, annot=True, fmt='d', cmap='Blues',
                xticklabels=x_labels, yticklabels=y_labels, annot_kws={"size": 12})
    plt.title('Ma trận nhầm lẫn PhoBERT (Augmented) trên tập kiểm tra', fontsize=15, pad=15)
    plt.xlabel('Nhãn dự đoán', fontsize=12)
    plt.ylabel('Nhãn thực tế', fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_out_dir / "phobert_aug_confusion_matrix.png", dpi=300)
    plt.close()

    logger.info("Done! Đã lưu biểu đồ vào thư mục artifacts/plot/")


if __name__ == "__main__":
    main()
