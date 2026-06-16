"""
evaluate_models.py
==================
Khung đánh giá mô hình chuẩn (Evaluation Framework) cho dự án
"Multimodal Review Analytics".

Các chỉ số được tính toán:
  - Confusion Matrix (ma trận nhầm lẫn)
  - Macro F1-Score
  - ROC-AUC (One-vs-Rest cho bài toán đa lớp)

Hỗ trợ đánh giá hai loại mô hình:
  1. TextEnsembleModel  — Phân tích cảm xúc (tích cực / tiêu cực / trung lập)
  2. ResNet50 (PyTorch) — Phát hiện hàng lỗi (defect / no-defect)

Usage (chạy từ thư mục gốc của project):
    # Đánh giá mô hình Text Ensemble đã lưu
    python scripts/evaluate_models.py --mode text \\
        --model-path ai_engine/models/ensemble_no_smote.pkl \\
        --data-path data/processed/reviews_labeled.csv \\
        --text-col cleaned_text --label-col sentiment

    # Đánh giá mô hình ResNet50 đã lưu
    python scripts/evaluate_models.py --mode image \\
        --model-path ai_engine/models/resnet50_defect.pth \\
        --data-path data/processed \\
        --batch-size 16

    # In ra tất cả chỉ số, không lưu plot
    python scripts/evaluate_models.py --mode text ... --no-plot
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Force UTF-8 stdout/stderr so Vietnamese text prints correctly on Windows ──
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Add project root to sys.path ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CORE METRIC FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Tính Confusion Matrix và trả về dưới dạng numpy array và DataFrame.

    Args:
        y_true: Nhãn thực tế.
        y_pred: Nhãn dự đoán.
        labels: Danh sách tên lớp theo thứ tự. Nếu None, tự suy ra từ dữ liệu.

    Returns:
        (cm_array, cm_df): Ma trận numpy và DataFrame có tên hàng/cột.
    """
    from sklearn.metrics import confusion_matrix

    if labels is not None:
        # Đảm bảo kiểu dữ liệu của labels khớp với y_true
        dtype = np.array(y_true).dtype
        try:
            unique_labels = [dtype.type(l) for l in labels]
        except (ValueError, TypeError):
            unique_labels = labels
    else:
        unique_labels = sorted(list(set(np.unique(y_true)) | set(np.unique(y_pred))))

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {l}" for l in unique_labels],
        columns=[f"Predicted: {l}" for l in unique_labels],
    )
    return cm, cm_df


def compute_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    zero_division: int = 0,
) -> Tuple[float, pd.DataFrame]:
    """Tính Macro F1-Score và báo cáo per-class.

    Args:
        y_true: Nhãn thực tế.
        y_pred: Nhãn dự đoán.
        labels: Tên lớp. Nếu None, tự suy ra.
        zero_division: Giá trị trả về khi denominator = 0 (0 hoặc 1).

    Returns:
        (macro_f1, report_df): Điểm Macro F1 tổng thể và DataFrame báo cáo chi tiết.
    """
    from sklearn.metrics import classification_report, f1_score

    if labels is not None:
        dtype = np.array(y_true).dtype
        try:
            unique_labels = [dtype.type(l) for l in labels]
        except (ValueError, TypeError):
            unique_labels = labels
    else:
        unique_labels = sorted(list(set(np.unique(y_true)) | set(np.unique(y_pred))))

    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=unique_labels, zero_division=zero_division)

    report_dict = classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=[str(l) for l in unique_labels],
        zero_division=zero_division,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).T

    return macro_f1, report_df


def compute_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_labels: Optional[List[str]] = None,
) -> Tuple[float, dict]:
    """Tính ROC-AUC theo chiến lược One-vs-Rest (OvR).

    Hỗ trợ cả bài toán nhị phân (2 lớp) và đa lớp (> 2 lớp).

    Args:
        y_true: Nhãn thực tế (dạng số nguyên hoặc chuỗi).
        y_proba: Xác suất dự đoán, shape (n_samples, n_classes).
        class_labels: Tên lớp tương ứng với từng cột của y_proba.
                      Nếu None, tự suy ra từ dữ liệu.

    Returns:
        (macro_auc, per_class_auc_dict): Macro AUC tổng thể và dict AUC từng lớp.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelBinarizer

    unique_labels = class_labels or sorted(list(np.unique(y_true)))
    n_classes = len(unique_labels)

    # FIX: Scikit-learn's LabelBinarizer sorts string classes alphabetically by default.
    # Alphabetical sorting maps "defect" to 0 and "no-defect" to 1, while y_proba columns
    # are ordered ["no-defect", "defect"] (indices 0 and 1). This mismatch inverted the
    # binary ROC-AUC calculation (e.g. producing 1 - AUC = 0.0012).
    # We resolve this by converting labels to integer indices according to unique_labels order.
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    y_true_int = np.array([label_to_idx[y] for y in y_true])

    lb = LabelBinarizer()
    lb.fit(range(n_classes))
    y_bin = lb.transform(y_true_int)

    # Với bài toán nhị phân, LabelBinarizer cho ra (n_samples, 1)
    # đại diện cho lớp 1 (tức unique_labels[1])
    if n_classes == 2:
        # y_proba có thể là (n_samples, 2) hoặc (n_samples, 1)
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            proba_pos = y_proba[:, 1]
        else:
            proba_pos = y_proba.ravel()

        macro_auc = float(roc_auc_score(y_bin, proba_pos))
        per_class_auc = {
            str(unique_labels[0]): None,      # lớp âm không có AUC riêng trong nhị phân
            str(unique_labels[1]): macro_auc,
        }
    else:
        # Đa lớp: OvR
        if y_proba.shape[1] != n_classes:
            raise ValueError(
                f"y_proba có {y_proba.shape[1]} cột nhưng có {n_classes} lớp. "
                "Kiểm tra lại class_labels."
            )
        macro_auc = float(
            roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
        )
        per_class_auc = {}
        for i, label in enumerate(unique_labels):
            try:
                auc = float(roc_auc_score(y_bin[:, i], y_proba[:, i]))
            except Exception:
                auc = float("nan")
            per_class_auc[str(label)] = auc

    return macro_auc, per_class_auc


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — REPORT PRINTER
# ════════════════════════════════════════════════════════════════════════════


def print_full_report(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    class_labels: Optional[List[str]] = None,
    save_plot: bool = True,
    plot_dir: str = "reports/figures",
) -> dict:
    """In đầy đủ báo cáo đánh giá và (tuỳ chọn) lưu biểu đồ.

    Args:
        model_name: Tên mô hình (dùng trong tiêu đề và tên file).
        y_true: Nhãn thực tế.
        y_pred: Nhãn dự đoán.
        y_proba: Xác suất dự đoán (None = bỏ qua tính ROC-AUC).
        class_labels: Tên các lớp theo thứ tự.
        save_plot: Nếu True, lưu biểu đồ Confusion Matrix vào plot_dir.
        plot_dir: Thư mục lưu biểu đồ.

    Returns:
        dict chứa các chỉ số: macro_f1, macro_auc, per_class_auc.
    """
    SEP = "=" * 70

    print(f"\n{SEP}")
    print(f"  EVALUATION REPORT — {model_name.upper()}")
    print(SEP)

    # ── 1. Confusion Matrix ──────────────────────────────────────────────────
    cm_array, cm_df = compute_confusion_matrix(y_true, y_pred, labels=class_labels)
    print("\n[1] CONFUSION MATRIX")
    print(cm_df.to_string())

    # ── 2. Macro F1-Score ────────────────────────────────────────────────────
    macro_f1, report_df = compute_macro_f1(y_true, y_pred, labels=class_labels)
    print(f"\n[2] CLASSIFICATION REPORT (Macro F1-Score = {macro_f1:.4f})")
    print(report_df.to_string(float_format="{:.4f}".format))

    # ── 3. ROC-AUC ──────────────────────────────────────────────────────────
    results: dict = {"macro_f1": macro_f1, "macro_auc": None, "per_class_auc": {}}

    if y_proba is not None:
        try:
            macro_auc, per_class_auc = compute_roc_auc(y_true, y_proba, class_labels=class_labels)
            results["macro_auc"] = macro_auc
            results["per_class_auc"] = per_class_auc

            print(f"\n[3] ROC-AUC (One-vs-Rest, Macro = {macro_auc:.4f})")
            for label, auc in per_class_auc.items():
                if auc is not None:
                    print(f"    {label:<20} AUC = {auc:.4f}")
                else:
                    print(f"    {label:<20} AUC = N/A (negative class in binary task)")
        except Exception as e:
            logger.warning("Cannot compute ROC-AUC: %s", e)
            print("\n[3] ROC-AUC: Could not be computed (missing probabilities or data error)")
    else:
        print("\n[3] ROC-AUC: Skipped (no predicted probabilities provided)")

    # ── 4. Lưu biểu đồ Confusion Matrix ─────────────────────────────────────
    if save_plot:
        _save_confusion_matrix_plot(cm_array, cm_df, model_name, plot_dir)

    print(f"\n{SEP}\n")
    return results


def _save_confusion_matrix_plot(
    cm_array: np.ndarray,
    cm_df: pd.DataFrame,
    model_name: str,
    plot_dir: str,
) -> None:
    """Lưu heatmap Confusion Matrix ra file PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(max(6, len(cm_df) * 2), max(5, len(cm_df) * 1.7)))

        short_labels = [c.replace("Predicted: ", "") for c in cm_df.columns]
        row_labels   = [r.replace("Actual: ", "")    for r in cm_df.index]

        sns.heatmap(
            cm_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=short_labels,
            yticklabels=row_labels,
            ax=ax,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("Actual Label", fontsize=12)
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_dir = Path(plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.lower().replace(" ", "_").replace("/", "-")
        out_path = out_dir / f"confusion_matrix_{safe_name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion Matrix saved to: %s", out_path)
        print(f"    [Plot] Confusion Matrix -> {out_path}")
    except ImportError:
        logger.warning(
            "matplotlib / seaborn not installed. "
            "Skipping plot (pip install matplotlib seaborn)."
        )


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — TEXT MODEL EVALUATOR
# ════════════════════════════════════════════════════════════════════════════


def evaluate_text_model(
    model_path: str,
    data_path: str,
    text_col: str = "cleaned_text",
    label_col: str = "sentiment",
    save_plot: bool = True,
    plot_dir: str = "reports/figures",
) -> dict:
    """Đánh giá TextEnsembleModel đã lưu trên tập test CSV.

    Args:
        model_path: Đường dẫn tới file .pkl của TextEnsembleModel.
        data_path: Đường dẫn tới file CSV chứa cột text và nhãn.
        text_col: Tên cột chứa văn bản đầu vào.
        label_col: Tên cột chứa nhãn thực tế.
        save_plot: Có lưu biểu đồ hay không.
        plot_dir: Thư mục lưu biểu đồ.

    Returns:
        dict chứa các chỉ số đánh giá.
    """
    from ai_engine.models.text_baseline import TextEnsembleModel

    logger.info("Loading text model from: %s", model_path)
    model = TextEnsembleModel.load(model_path)

    logger.info("Loading data from: %s", data_path)
    df = pd.read_csv(data_path)

    # Validate columns
    for col in [text_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV. "
                             f"Available columns: {list(df.columns)}")

    # Drop rows with missing values
    df = df.dropna(subset=[text_col, label_col])
    logger.info("Evaluation samples: %d", len(df))

    X = df[text_col]
    y_true = df[label_col].values

    # Dự đoán
    y_pred = model.predict(X)
    try:
        y_proba = model.predict_proba(X)
    except Exception:
        logger.warning("predict_proba not available; skipping ROC-AUC.")
        y_proba = None

    class_labels = sorted(list(np.unique(y_true)))

    return print_full_report(
        model_name="Text Ensemble (Sentiment)",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_labels=class_labels,
        save_plot=save_plot,
        plot_dir=plot_dir,
    )


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — IMAGE MODEL EVALUATOR (ResNet50)
# ════════════════════════════════════════════════════════════════════════════


def evaluate_image_model(
    model_path: str,
    data_path: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    save_plot: bool = True,
    plot_dir: str = "reports/figures",
) -> dict:
    """Đánh giá mô hình ResNet50 defect-detection đã lưu trên Test set (hoặc Validation split).

    Args:
        model_path: Đường dẫn tới checkpoint .pth.
        data_path: Thư mục dữ liệu (nếu chứa 'test/' sẽ dùng 'test/').
        batch_size: Batch size cho DataLoader.
        val_split: Tỉ lệ validation (nếu chia offline split).
        save_plot: Có lưu biểu đồ hay không.
        plot_dir: Thư mục lưu biểu đồ.

    Returns:
        dict chứa các chỉ số đánh giá.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from ai_engine.image_processing.defect_detection import (
        get_dataloaders,
        get_resnet50_model,
        ProductDefectDataset,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Detect split folder
    path = Path(data_path)
    test_dir = path / "test"

    if test_dir.exists():
        test_dataset = ProductDefectDataset(data_dir=str(test_dir), is_train=False, oversample_defect=1)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        logger.info("Loaded test set from subdirectory: %s (%d samples)", test_dir, len(test_dataset))
    elif (path / "defect").exists():
        _, loader = get_dataloaders(
            data_dir=data_path,
            batch_size=batch_size,
            val_split=val_split,
        )
        logger.info("Loaded validation split from %s (%d samples)", data_path, len(loader.dataset))
    else:
        test_dataset = ProductDefectDataset(data_dir=data_path, is_train=False, oversample_defect=1)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        logger.info("Loaded directly from: %s (%d samples)", data_path, len(test_dataset))

    # Load checkpoint
    logger.info("Loading checkpoint from: %s", model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = get_resnet50_model(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    threshold = checkpoint.get("threshold", 0.5)
    logger.info("Using decision threshold from checkpoint: %.3f", threshold)

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_proba: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            proba = F.softmax(outputs, dim=1).cpu().numpy()
            
            # Apply tuned threshold for binary classification
            preds = (proba[:, 1] >= threshold).astype(int)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_proba.append(proba)

    class_labels = ["no-defect", "defect"]
    y_true = np.array([class_labels[y] for y in all_labels])
    y_pred = np.array([class_labels[y] for y in all_preds])
    y_proba = np.vstack(all_proba)

    return print_full_report(
        model_name="ResNet50 (Defect Detection)",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_labels=class_labels,
        save_plot=save_plot,
        plot_dir=plot_dir,
    )


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4b — SPAM MODEL EVALUATOR (SpamHybridModel)
# ════════════════════════════════════════════════════════════════════════════


def evaluate_spam_model(
    model_path: str,
    data_path: str,
    text_col: str = "text",
    rating_col: str = "rating",
    label_col: str = "final_spam",
    save_plot: bool = True,
    plot_dir: str = "reports/figures",
) -> dict:
    """Danh gia SpamHybridModel tren CSV co ground-truth spam label.

    CSV can co cac cot: text, rating, va 1 cot ground-truth spam (0/1).
    Cot ground-truth co the la:
      - 'final_spam'  : nhan ket hop duoc ghi ra boi train_spam_model.py
      - 'is_spam'     : nhan thu cong hoac tu nguon khac

    Args:
        model_path: Duong dan toi file .pkl cua SpamHybridModel.
        data_path:  Duong dan toi CSV test.
        text_col:   Ten cot van ban (default: text).
        rating_col: Ten cot rating (default: rating).
        label_col:  Ten cot nhan spam ground-truth (default: final_spam).
        save_plot:  Co luu bieu do khong.
        plot_dir:   Thu muc luu bieu do.

    Returns:
        dict chua cac chi so danh gia.
    """
    from ai_engine.text_processing.spam_model import (
        SpamHybridModel,
        build_feature_matrix,
    )
    from ai_engine.text_processing.spam_filter import detect_spam

    logger.info("Loading spam model from: %s", model_path)
    model = SpamHybridModel.load(model_path)

    logger.info("Loading data from: %s", data_path)
    df = pd.read_csv(data_path)

    for col in [text_col, rating_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    df[text_col] = df[text_col].fillna("").astype(str)
    if text_col != "text" or rating_col != "rating":
        df = df.rename(columns={text_col: "text", rating_col: "rating"})

    y_true = df[label_col].values.astype(int)
    logger.info("Evaluation samples: %d | Spam: %d | Clean: %d",
                len(df), int(y_true.sum()), int((y_true == 0).sum()))

    # Chay rule-based de lay flag_details (can cho build_feature_matrix)
    logger.info("Running rule-based detection to build feature matrix...")
    df_flagged = detect_spam(df[["text", "rating"]], dup_threshold=0.85)

    X = build_feature_matrix(df_flagged, df["text"].tolist(), df["rating"].tolist())

    # Du doan
    iforest_pred = model.predict_anomaly(X)
    rule_is_spam = df_flagged["is_spam"].values.astype(int)
    y_pred = model.predict_final_spam(X, rule_is_spam)

    # Anomaly score lam xac suat (normalize ve [0,1])
    scores = model.anomaly_score(X)          # am = bat thuong
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    # Xac suat spam = 1 - normalized_score
    y_proba = np.column_stack([scores_norm, 1.0 - scores_norm])

    return print_full_report(
        model_name="SpamHybridModel (Rule-based + Isolation Forest)",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_labels=[0, 1],
        save_plot=save_plot,
        plot_dir=plot_dir,
    )


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — QUICK SANITY-CHECK (chạy trên dữ liệu giả)
# ════════════════════════════════════════════════════════════════════════════


def run_sanity_check() -> None:
    """Quick demo of all metrics on synthetic data to verify the framework."""
    logger.info("Running sanity-check on synthetic data...")

    rng = np.random.default_rng(42)

    # ── Binary (defect detection) — integer labels 0/1 ──────────────────────
    y_true_bin = rng.integers(0, 2, size=200)
    y_pred_bin = rng.integers(0, 2, size=200)
    y_proba_bin = rng.dirichlet(alpha=[1, 1], size=200)

    print_full_report(
        model_name="[Sanity] Binary Model",
        y_true=y_true_bin,
        y_pred=y_pred_bin,
        y_proba=y_proba_bin,
        class_labels=[0, 1],
        save_plot=False,
    )

    # ── Multiclass (sentiment) — string labels ───────────────────────────────
    labels_mc = np.array(["positive", "negative", "neutral"])
    y_true_mc = rng.choice(labels_mc, size=300)
    y_pred_mc = rng.choice(labels_mc, size=300)
    y_proba_mc = rng.dirichlet(alpha=[1, 1, 1], size=300)

    print_full_report(
        model_name="[Sanity] Multiclass Model",
        y_true=y_true_mc,
        y_pred=y_pred_mc,
        y_proba=y_proba_mc,
        class_labels=["positive", "negative", "neutral"],
        save_plot=False,
    )

    logger.info("Sanity-check complete.")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Khung đánh giá mô hình chuẩn — Confusion Matrix, Macro F1, ROC-AUC",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=False)

    # ── Sub-command: text ────────────────────────────────────────────────────
    text_p = subparsers.add_parser("text", help="Đánh giá TextEnsembleModel (sentiment)")
    text_p.add_argument("--model-path", required=True,
                        help="Đường dẫn tới file .pkl của TextEnsembleModel")
    text_p.add_argument("--data-path", required=True,
                        help="Đường dẫn tới file CSV test")
    text_p.add_argument("--text-col", default="cleaned_text",
                        help="Tên cột văn bản (default: cleaned_text)")
    text_p.add_argument("--label-col", default="sentiment",
                        help="Tên cột nhãn (default: sentiment)")
    text_p.add_argument("--no-plot", action="store_true",
                        help="Không lưu biểu đồ")
    text_p.add_argument("--plot-dir", default="reports/figures",
                        help="Thư mục lưu biểu đồ (default: reports/figures)")

    # ── Sub-command: image ───────────────────────────────────────────────────
    img_p = subparsers.add_parser("image", help="Đánh giá ResNet50 (defect detection)")
    img_p.add_argument("--model-path", required=True,
                       help="Đường dẫn tới checkpoint .pth")
    img_p.add_argument("--data-path", required=True,
                       help="Thư mục chứa subfolder defect/ và no-defect/")
    img_p.add_argument("--batch-size", type=int, default=16,
                       help="Batch size (default: 16)")
    img_p.add_argument("--val-split", type=float, default=0.2,
                       help="Tỉ lệ validation (default: 0.2 — phải khớp với lúc train)")
    img_p.add_argument("--no-plot", action="store_true",
                       help="Không lưu biểu đồ")
    img_p.add_argument("--plot-dir", default="reports/figures",
                       help="Thư mục lưu biểu đồ (default: reports/figures)")

    # ── Sub-command: spam ────────────────────────────────────────────────────
    spam_p = subparsers.add_parser("spam", help="Danh gia SpamHybridModel (Rule-based + IForest)")
    spam_p.add_argument("--model-path", required=True,
                        help="Duong dan toi file .pkl cua SpamHybridModel")
    spam_p.add_argument("--data-path", required=True,
                        help="CSV test co cot text, rating va ground-truth spam label")
    spam_p.add_argument("--text-col", default="text",
                        help="Ten cot van ban (default: text)")
    spam_p.add_argument("--rating-col", default="rating",
                        help="Ten cot rating (default: rating)")
    spam_p.add_argument("--label-col", default="final_spam",
                        help="Ten cot nhan spam ground-truth (default: final_spam)")
    spam_p.add_argument("--no-plot", action="store_true",
                        help="Khong luu bieu do")
    spam_p.add_argument("--plot-dir", default="reports/figures",
                        help="Thu muc luu bieu do (default: reports/figures)")

    # ── Sub-command: sanity ──────────────────────────────────────────────────
    subparsers.add_parser("sanity", help="Chạy sanity-check trên dữ liệu giả")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode is None or args.mode == "sanity":
        run_sanity_check()

    elif args.mode == "text":
        evaluate_text_model(
            model_path=args.model_path,
            data_path=args.data_path,
            text_col=args.text_col,
            label_col=args.label_col,
            save_plot=not args.no_plot,
            plot_dir=args.plot_dir,
        )

    elif args.mode == "image":
        evaluate_image_model(
            model_path=args.model_path,
            data_path=args.data_path,
            batch_size=args.batch_size,
            val_split=args.val_split,
            save_plot=not args.no_plot,
            plot_dir=args.plot_dir,
        )

    elif args.mode == "spam":
        evaluate_spam_model(
            model_path=args.model_path,
            data_path=args.data_path,
            text_col=args.text_col,
            rating_col=args.rating_col,
            label_col=args.label_col,
            save_plot=not args.no_plot,
            plot_dir=args.plot_dir,
        )


if __name__ == "__main__":
    main()
