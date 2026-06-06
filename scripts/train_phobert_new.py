"""
PhoBERT Sentiment Classification – Training Script with Advanced Augmentation.

Phiên bản mở rộng của train_phobert.py, bổ sung:
    * Back-translation (Việt → Anh → Việt) via deep-translator.
    * ContextualWordEmbsAug via nlpaug + PhoBERT (thay từ theo ngữ cảnh).

Chiến lược:
    - Với mỗi lớp thiểu số, lấy toàn bộ mẫu gốc và áp dụng:
        1. Back-translation một lần → mẫu mới tự nhiên, ý nghĩa được bảo toàn.
        2. ContextualWordEmbsAug → thay ~15% từ bằng từ tương đương theo PhoBERT.
    - Nếu vẫn chưa đủ target_ratio thì dùng thêm ContextualWordEmbsAug.

Cài thêm trước khi chạy::

    pip install nlpaug deep-translator

Usage::

    python scripts/train_phobert_new.py \\
        --target_ratio 0.4 \\
        --aug_p 0.15

All arguments have sensible defaults.
"""

import argparse
import logging
import os
import random
import sys
import time
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
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

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
    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        report = classification_report(
            labels, preds,
            target_names=[id2label[i] for i in range(NUM_LABELS)],
            zero_division=0,
        )
        logger.info("\n%s", report)

        per_class_f1 = f1_score(
            labels, preds, average=None,
            labels=list(range(NUM_LABELS)), zero_division=0,
        )

        metrics = {
            "accuracy":         accuracy_score(labels, preds),
            "f1_macro":         f1_score(labels, preds, average="macro",     zero_division=0),
            "precision_macro":  precision_score(labels, preds, average="macro", zero_division=0),
            "recall_macro":     recall_score(labels, preds, average="macro", zero_division=0),
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

    if not all(p.exists() for p in [train_path, val_path, test_path]):
        raise FileNotFoundError("Missing pre-split datasets in data/processed/")

    logger.info("Loading pre-split datasets ...")
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    valid_labels = set(LABEL_MAP.keys())
    train_df = train_df.dropna(subset=[text_column, label_column])
    train_df = train_df[train_df[label_column].isin(valid_labels)].reset_index(drop=True)
    val_df   = val_df.dropna(subset=[text_column, label_column])
    val_df   = val_df[val_df[label_column].isin(valid_labels)].reset_index(drop=True)
    test_df  = test_df.dropna(subset=[text_column, label_column])
    test_df  = test_df[test_df[label_column].isin(valid_labels)].reset_index(drop=True)

    logger.info("Split → train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Augmentation: Back-translation
# ---------------------------------------------------------------------------

def back_translate(text: str, bt_delay: float = 0.3) -> str:
    """Dịch Việt → Anh → Việt bằng GoogleTranslator (deep-translator).

    Args:
        text: Câu tiếng Việt gốc.
        bt_delay: Thời gian nghỉ (giây) giữa 2 lần gọi API để tránh rate-limit.

    Returns:
        Câu tiếng Việt sau khi dịch ngược. Trả về câu gốc nếu gặp lỗi.
    """
    try:
        from deep_translator import GoogleTranslator
        en_text = GoogleTranslator(source="vi", target="en").translate(text)
        time.sleep(bt_delay)
        vi_text = GoogleTranslator(source="en", target="vi").translate(en_text)
        time.sleep(bt_delay)
        return vi_text if vi_text and vi_text.strip() else text
    except Exception as e:
        logger.debug("Back-translation thất bại (%s), giữ nguyên câu gốc.", e)
        return text


# ---------------------------------------------------------------------------
# Augmentation: ContextualWordEmbsAug (nlpaug + PhoBERT)
# ---------------------------------------------------------------------------

def build_contextual_augmenter(aug_p: float = 0.15, device: str = "cpu"):
    """Khởi tạo ContextualWordEmbsAug dùng PhoBERT.

    PhoBERT (RoBERTa-based) mask ngẫu nhiên ~aug_p% số từ, rồi predict
    từ thay thế theo ngữ cảnh → câu mới giữ nguyên nghĩa nhưng khác từ ngữ.

    Args:
        aug_p: Tỉ lệ từ bị thay thế (0 → 1).
        device: 'cuda' hoặc 'cpu'.

    Returns:
        nlpaug augmenter object.
    """
    try:
        import nlpaug.augmenter.word as naw
        augmenter = naw.ContextualWordEmbsAug(
            model_path=MODEL_CHECKPOINT,
            model_type="roberta",   # PhoBERT kế thừa kiến trúc RoBERTa
            action="substitute",    # thay từ, không thêm/xóa
            aug_p=aug_p,
            device=device,
            batch_size=16,
            silence=True,
        )
        logger.info("ContextualWordEmbsAug khởi tạo thành công (device=%s, aug_p=%.2f).", device, aug_p)
        return augmenter
    except ImportError:
        logger.error("Thiếu thư viện nlpaug. Chạy: pip install nlpaug")
        raise


def contextual_augment(text: str, augmenter) -> str:
    """Áp dụng ContextualWordEmbsAug lên một câu.

    Returns:
        Câu đã được augment. Trả về câu gốc nếu gặp lỗi.
    """
    try:
        result = augmenter.augment(text)
        return result[0] if result else text
    except Exception as e:
        logger.debug("Contextual aug thất bại (%s), giữ nguyên câu gốc.", e)
        return text


# ---------------------------------------------------------------------------
# Oversampling kết hợp cả 2 kỹ thuật
# ---------------------------------------------------------------------------

def oversample_with_augmentation(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    target_ratio: float = 0.4,
    aug_p: float = 0.15,
    bt_delay: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Oversample các lớp thiểu số bằng Back-translation + ContextualWordEmbsAug.

    Chiến lược:
        1. Dùng Back-translation cho toàn bộ mẫu gốc của lớp thiểu số (1 lần/mẫu).
        2. Nếu vẫn còn thiếu, dùng ContextualWordEmbsAug để tạo thêm.

    Args:
        df: DataFrame gốc.
        text_column: Tên cột văn bản.
        label_column: Tên cột nhãn.
        target_ratio: Tỉ lệ mục tiêu so với lớp đa số (0 → 1).
        aug_p: Tỉ lệ từ bị thay thế trong ContextualWordEmbsAug.
        bt_delay: Độ trễ (giây) giữa các API call của back-translation.
        seed: Random seed.

    Returns:
        DataFrame đã được bổ sung mẫu augment, đã shuffle.
    """
    random.seed(seed)
    np.random.seed(seed)

    counts = df[label_column].value_counts()
    majority_count  = counts.max()
    majority_label  = counts.idxmax()
    target_count    = int(majority_count * target_ratio)

    logger.info("Phân bố lớp TRƯỚC augmentation:\n%s", counts.to_string())
    logger.info(
        "Lớp đa số: '%s' (%d mẫu) | Mục tiêu lớp thiểu số: %d mẫu (%.0f%%)",
        majority_label, majority_count, target_count, target_ratio * 100,
    )

    # Khởi tạo augmenter một lần, tái sử dụng cho tất cả lớp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    augmenter = build_contextual_augmenter(aug_p=aug_p, device=device)

    augmented_rows: List[pd.Series] = []

    for label, count in counts.items():
        if label == majority_label:
            continue

        needed = max(0, target_count - count)
        if needed == 0:
            logger.info("Lớp '%s' đủ mẫu, bỏ qua.", label)
            continue

        logger.info("Lớp '%s': %d mẫu → cần thêm %d mẫu.", label, count, needed)
        minority_df = df[df[label_column] == label].reset_index(drop=True)

        # ---- Bước 1: Back-translation cho toàn bộ mẫu gốc ----
        bt_count = min(len(minority_df), needed)
        logger.info("  [Back-translation] Tạo %d mẫu ...", bt_count)
        bt_indices = np.random.choice(len(minority_df), size=bt_count, replace=False)

        for i, idx in enumerate(bt_indices):
            row = minority_df.iloc[idx].copy()
            row[text_column] = back_translate(row[text_column], bt_delay=bt_delay)
            augmented_rows.append(row)
            if (i + 1) % 100 == 0:
                logger.info("    Back-translation: %d/%d xong", i + 1, bt_count)

        needed -= bt_count

        # ---- Bước 2: ContextualWordEmbsAug cho phần còn lại ----
        if needed > 0:
            logger.info("  [ContextualWordEmbsAug] Tạo thêm %d mẫu ...", needed)
            ca_indices = np.random.randint(0, len(minority_df), size=needed)

            for i, idx in enumerate(ca_indices):
                row = minority_df.iloc[idx].copy()
                row[text_column] = contextual_augment(row[text_column], augmenter)
                augmented_rows.append(row)
                if (i + 1) % 200 == 0:
                    logger.info("    ContextualAug: %d/%d xong", i + 1, needed)

    if not augmented_rows:
        logger.info("Không có lớp nào cần augmentation.")
        return df

    aug_df    = pd.DataFrame(augmented_rows)
    result_df = pd.concat([df, aug_df], ignore_index=True)

    logger.info("Phân bố lớp SAU augmentation:\n%s", result_df[label_column].value_counts().to_string())
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
        description="Train PhoBERT với Back-translation + ContextualWordEmbsAug."
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
    parser.add_argument("--early_stopping_patience", type=int,   default=2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--logging_steps",type=int,   default=50)
    # ---- Augmentation ----
    parser.add_argument(
        "--target_ratio", type=float, default=0.4,
        help="Tỉ lệ mục tiêu của lớp thiểu số so với lớp đa số. Mặc định 0.4.",
    )
    parser.add_argument(
        "--aug_p", type=float, default=0.15,
        help="Tỉ lệ từ bị thay thế trong ContextualWordEmbsAug. Mặc định 0.15.",
    )
    parser.add_argument(
        "--bt_delay", type=float, default=0.3,
        help="Độ trễ (giây) giữa các lần gọi Back-translation API. Mặc định 0.3.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir      = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_txt_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_txt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ---------------------------------------------------------------
    train_df, val_df, test_df = load_datasets(
        text_column=args.text_column,
        label_column=args.label_column,
    )

    # ---- Augmentation -------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Bắt đầu Augmentation (Back-translation + ContextualWordEmbsAug) ...")
    train_df = oversample_with_augmentation(
        df=train_df,
        text_column=args.text_column,
        label_column=args.label_column,
        target_ratio=args.target_ratio,
        aug_p=args.aug_p,
        bt_delay=args.bt_delay,
        seed=args.seed,
    )
    logger.info("Tổng mẫu sau augmentation: %d", len(train_df))
    logger.info("=" * 60)

    train_int_labels = [LABEL_MAP[lbl] for lbl in train_df[args.label_column]]
    class_counts     = compute_class_counts(train_int_labels)
    logger.info("Class counts sau augmentation: %s", class_counts)

    # ---- Tokeniser ----------------------------------------------------------
    logger.info("Loading tokeniser: %s …", MODEL_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # ---- Datasets -----------------------------------------------------------
    train_dataset = PhoBertReviewDataset.from_dataframe(
        df=train_df, text_column=args.text_column, label_column=args.label_column,
        tokenizer=tokenizer, max_length=args.max_length, pad_to_max_length=False,
    )
    val_dataset = PhoBertReviewDataset.from_dataframe(
        df=val_df,   text_column=args.text_column, label_column=args.label_column,
        tokenizer=tokenizer, max_length=args.max_length, pad_to_max_length=False,
    )
    test_dataset = PhoBertReviewDataset.from_dataframe(
        df=test_df,  text_column=args.text_column, label_column=args.label_column,
        tokenizer=tokenizer, max_length=args.max_length, pad_to_max_length=False,
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

    # ---- Load tuned hyperparameters -----------------------------------------
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
        lr           = best_params.get("learning_rate",               lr)
        weight_decay = best_params.get("weight_decay",                weight_decay)
        warmup_ratio = best_params.get("warmup_ratio",                warmup_ratio)
        batch_size   = best_params.get("per_device_train_batch_size", batch_size)
        num_epochs   = best_params.get("num_train_epochs",            num_epochs)
        gamma        = best_params.get("gamma",                       gamma)
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
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ],
    )

    # ---- Train --------------------------------------------------------------
    logger.info("Starting training (augmented data) …")
    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # ---- Save metrics -------------------------------------------------------
    for filename, metrics in [
        ("phobert_aug_train_metrics.txt", train_result.metrics),
        ("phobert_aug_val_metrics.txt",   trainer.evaluate()),
    ]:
        path = metrics_txt_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        logger.info("Đã lưu %s", path)

    test_result  = trainer.predict(test_dataset)
    test_path    = metrics_txt_dir / "phobert_aug_test_metrics.txt"
    with open(test_path, "w", encoding="utf-8") as f:
        for k, v in test_result.metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info("Đã lưu %s", test_path)

    # ---- Plots --------------------------------------------------------------
    plot_out_dir = REPO_ROOT / "artifacts" / "plot"
    plot_out_dir.mkdir(parents=True, exist_ok=True)

    history      = trainer.state.log_history
    train_loss   = [h["loss"]                for h in history if "loss"                in h and "epoch" in h]
    val_loss     = [h["eval_val_loss"]       for h in history if "eval_val_loss"       in h]
    val_f1       = [h["eval_val_f1_macro"]   for h in history if "eval_val_f1_macro"   in h]
    train_f1     = [h["eval_train_f1_macro"] for h in history if "eval_train_f1_macro" in h]
    epochs_train = [h["epoch"]               for h in history if "loss"                in h]
    epochs_val   = [h["epoch"]               for h in history if "eval_val_loss"       in h]

    # Loss curve
    plt.figure(figsize=(8, 6))
    if train_loss: plt.plot(epochs_train, train_loss, label="Mất mát huấn luyện", marker="o", color="#1f77b4")
    if val_loss:   plt.plot(epochs_val,   val_loss,   label="Mất mát kiểm định",  marker="o", color="#ff7f0e")
    plt.title("Biểu đồ hàm mất mát của mô hình PhoBERT (Augmented)", fontsize=15, pad=15)
    plt.xlabel("Vòng lặp", fontsize=12); plt.ylabel("Giá trị mất mát", fontsize=12)
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.7); plt.tight_layout()
    plt.savefig(plot_out_dir / "phobert_aug_loss_curve.png", dpi=300); plt.close()

    # F1 curve
    plt.figure(figsize=(8, 6))
    if train_f1: plt.plot(epochs_val, train_f1, label="Điểm F1 huấn luyện", marker="o", color="#1f77b4")
    if val_f1:   plt.plot(epochs_val, val_f1,   label="Điểm F1 kiểm định",  marker="s", color="#ff7f0e")
    plt.title("Biểu đồ điểm F1 của mô hình PhoBERT (Augmented)", fontsize=15, pad=15)
    plt.xlabel("Vòng lặp", fontsize=12); plt.ylabel("Điểm F1", fontsize=12)
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.7); plt.tight_layout()
    plt.savefig(plot_out_dir / "phobert_aug_f1_curve.png", dpi=300); plt.close()

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    y_true = test_result.label_ids
    y_pred = np.argmax(test_result.predictions, axis=1)
    cm        = confusion_matrix(y_true, y_pred)
    col_order = [LABEL_MAP["tích cực"], LABEL_MAP["trung lập"], LABEL_MAP["tiêu cực"]]
    row_order = [LABEL_MAP["tiêu cực"], LABEL_MAP["trung lập"], LABEL_MAP["tích cực"]]
    cm_r      = cm[np.ix_(row_order, col_order)]
    x_labels  = [ID_TO_LABEL[i] for i in col_order]
    y_labels  = [ID_TO_LABEL[i] for i in row_order]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_r, annot=True, fmt="d", cmap="Blues",
                xticklabels=x_labels, yticklabels=y_labels, annot_kws={"size": 12})
    plt.title("Ma trận nhầm lẫn PhoBERT (Augmented) trên tập kiểm tra", fontsize=15, pad=15)
    plt.xlabel("Nhãn dự đoán", fontsize=12); plt.ylabel("Nhãn thực tế", fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_out_dir / "phobert_aug_confusion_matrix.png", dpi=300); plt.close()

    logger.info("Done! Đã lưu biểu đồ vào artifacts/plot/")


if __name__ == "__main__":
    main()
