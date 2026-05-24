"""
compare_models.py
=================
Khung đánh giá so sánh hiệu năng thực tế giữa:
  - Text Baseline (Weighted Soft-Voting Ensemble: LR + SVM + RF + TF-IDF)
  - PhoBERT (vinai/phobert-base-v2 fine-tuned cho sentiment)

Metrics tính toán trên cùng một hold-out test set:
  - Macro F1-Score, Weighted F1-Score, Accuracy
  - Per-class F1, Precision, Recall
  - ROC-AUC (One-vs-Rest)
  - Inference latency (ms/sample trung bình)
  - Confusion Matrix (lưu thành file PNG)

Output:
  - In bảng so sánh ra console
  - Lưu JSON report: artifacts/metrics/model_comparison_report.json
  - Lưu confusion matrix PNGs: artifacts/plots/

Usage (từ thư mục gốc):
    # So sánh với model đã train
    python scripts/compare_models.py \\
        --baseline-path artifacts/models/baselines/ensemble_no_smote_auto_weights.pkl \\
        --phobert-path ai_engine/models/weights/phobert_best \\
        --data-path data/processed/processed_labeled_reviews.csv

    # Chỉ chạy với baseline (chưa có PhoBERT)
    python scripts/compare_models.py \\
        --baseline-path artifacts/models/baselines/ensemble_no_smote_auto_weights.pkl \\
        --data-path data/processed/processed_labeled_reviews.csv \\
        --no-phobert

    # Sanity check với dữ liệu giả (không cần file model)
    python scripts/compare_models.py --sanity
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Force UTF-8 stdout/stderr ──────────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Add project root to path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
# pyrefly: ignore [missing-import]
from sklearn.model_selection import train_test_split
# pyrefly: ignore [missing-import]
from sklearn.preprocessing import LabelBinarizer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_DATA_PATH = str(_PROJECT_ROOT / "data" / "processed" / "processed_labeled_reviews.csv")
_TEXT_COL = "cleaned_text"
_LABEL_COL = "sentiment_label"
_TEST_SIZE = 0.20
_RANDOM_STATE = 42
_METRICS_DIR = str(_PROJECT_ROOT / "artifacts" / "metrics")
_PLOTS_DIR = str(_PROJECT_ROOT / "artifacts" / "plots")

# Label mapping: Vietnamese string → integer (dùng cho PhoBERT)
_LABEL_MAP = {
    "tích cực": 0,
    "tiêu cực": 1,
    "trung lập": 2,
}
_ID_TO_LABEL = {v: k for k, v in _LABEL_MAP.items()}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ModelReport:
    """Kết quả đánh giá đầy đủ của một mô hình."""
    model_name: str
    macro_f1: float
    weighted_f1: float
    accuracy: float
    macro_auc: Optional[float]
    per_class_f1: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    latency_ms_mean: float          # Thời gian inference trung bình (ms/sample)
    latency_ms_p95: float           # P95 latency
    n_samples: int
    target_met: bool                 # F1 macro ≥ 0.82 (target của dự án)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_roc_auc_multiclass(y_true_str, y_proba, class_labels) -> Optional[float]:
    """Tính macro OvR ROC-AUC cho bài toán đa lớp.

    Args:
        y_true_str: Nhãn thực tế (string).
        y_proba: Xác suất dự đoán shape (n_samples, n_classes).
        class_labels: Danh sách tên lớp theo đúng thứ tự cột.

    Returns:
        Macro AUC hoặc None nếu không tính được.
    """
    try:
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y_true_str)
        if y_proba.shape[1] != len(class_labels):
            return None
        return float(roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"))
    except Exception as exc:
        logger.warning("Không thể tính ROC-AUC: %s", exc)
        return None


def _measure_latency(
    predict_fn,
    X_sample,
    n_warmup: int = 5,
    n_repeat: int = 50,
) -> Tuple[float, float]:
    """Đo latency inference theo ms/sample.

    Args:
        predict_fn: Hàm nhận X và trả về predictions.
        X_sample: Mẫu dữ liệu để đo (nên dùng một batch nhỏ).
        n_warmup: Số lần chạy warmup (không tính vào kết quả).
        n_repeat: Số lần đo để lấy trung bình.

    Returns:
        (mean_ms, p95_ms): Latency trung bình và P95 tính theo ms/sample.
    """
    n = len(X_sample) if hasattr(X_sample, '__len__') else 1

    # Warmup
    for _ in range(n_warmup):
        predict_fn(X_sample)

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        predict_fn(X_sample)
        times.append((time.perf_counter() - t0) * 1000 / max(n, 1))

    return float(np.mean(times)), float(np.percentile(times, 95))


def _save_confusion_matrix_plot(
    cm: np.ndarray,
    class_labels: List[str],
    model_name: str,
    plots_dir: str,
) -> Optional[str]:
    """Lưu Confusion Matrix dạng heatmap PNG.

    Args:
        cm: Numpy array confusion matrix.
        class_labels: Tên các lớp.
        model_name: Tên mô hình (dùng cho tiêu đề & tên file).
        plots_dir: Thư mục lưu.

    Returns:
        Đường dẫn file PNG đã lưu, hoặc None nếu lỗi.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(max(6, len(class_labels) * 2),
                                        max(5, len(class_labels) * 1.7)))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_xlabel("Nhãn dự đoán", fontsize=12)
        ax.set_ylabel("Nhãn thực tế", fontsize=12)
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        os.makedirs(plots_dir, exist_ok=True)
        safe_name = model_name.lower().replace(" ", "_").replace("/", "-")
        out_path = os.path.join(plots_dir, f"cm_{safe_name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion Matrix đã lưu → %s", out_path)
        return out_path
    except ImportError:
        logger.warning("matplotlib/seaborn chưa cài. Bỏ qua vẽ biểu đồ.")
        return None


def _print_model_section(report: ModelReport) -> None:
    """In chi tiết đánh giá của một mô hình."""
    sep = "─" * 60
    target_icon = "✅" if report.target_met else "❌"

    print(f"\n  {'MODEL':>8}: {report.model_name}")
    print(f"  {sep}")
    print(f"  {'Macro F1':<20}: {report.macro_f1:.4f} {target_icon} (target ≥ 0.82)")
    print(f"  {'Weighted F1':<20}: {report.weighted_f1:.4f}")
    print(f"  {'Accuracy':<20}: {report.accuracy:.4f}")
    if report.macro_auc is not None:
        print(f"  {'ROC-AUC (OvR)':<20}: {report.macro_auc:.4f}")
    print(f"  {'Latency (mean)':<20}: {report.latency_ms_mean:.2f} ms/sample")
    print(f"  {'Latency (P95)':<20}: {report.latency_ms_p95:.2f} ms/sample")
    print(f"  {'Test samples':<20}: {report.n_samples}")
    print(f"\n  Per-class metrics:")
    header = f"    {'Lớp':<15} {'F1':>8} {'Precision':>12} {'Recall':>10}"
    print(header)
    print("    " + "─" * 48)
    for cls in report.per_class_f1:
        print(
            f"    {cls:<15}"
            f" {report.per_class_f1.get(cls, 0.0):>8.4f}"
            f" {report.per_class_precision.get(cls, 0.0):>12.4f}"
            f" {report.per_class_recall.get(cls, 0.0):>10.4f}"
        )


# ── Baseline evaluator ────────────────────────────────────────────────────────

def evaluate_baseline(
    model_path: str,
    X_test: pd.Series,
    y_test_str: np.ndarray,
    class_labels: List[str],
    plots_dir: str,
) -> ModelReport:
    """Đánh giá TextEnsembleModel (baseline).

    Args:
        model_path: Đường dẫn tới file .pkl.
        X_test: Văn bản test (pd.Series).
        y_test_str: Nhãn thực tế dạng string.
        class_labels: Danh sách tên lớp.
        plots_dir: Thư mục lưu confusion matrix.

    Returns:
        ModelReport đầy đủ.
    """
    import joblib
    logger.info("Đang tải Baseline từ: %s", model_path)
    model = joblib.load(model_path)

    # Predictions
    logger.info("Đang dự đoán với Baseline trên %d mẫu...", len(X_test))
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = None
        logger.warning("predict_proba không khả dụng cho baseline — bỏ qua ROC-AUC.")

    # Metrics
    macro_f1 = float(f1_score(y_test_str, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test_str, y_pred, average="weighted", zero_division=0))
    acc = float(accuracy_score(y_test_str, y_pred))

    report_dict = classification_report(
        y_test_str, y_pred, zero_division=0, output_dict=True,
    )
    per_class_f1 = {k: round(v["f1-score"], 4)
                    for k, v in report_dict.items()
                    if k in class_labels}
    per_class_prec = {k: round(v["precision"], 4)
                      for k, v in report_dict.items()
                      if k in class_labels}
    per_class_rec = {k: round(v["recall"], 4)
                     for k, v in report_dict.items()
                     if k in class_labels}

    macro_auc = None
    if y_proba is not None:
        macro_auc = _compute_roc_auc_multiclass(y_test_str, y_proba, class_labels)

    # Latency (trên 100 mẫu đầu để nhanh)
    sample_size = min(100, len(X_test))
    X_sample = X_test.iloc[:sample_size]
    lat_mean, lat_p95 = _measure_latency(
        predict_fn=lambda x: model.predict(x),
        X_sample=X_sample,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test_str, y_pred, labels=class_labels)
    _save_confusion_matrix_plot(cm, class_labels, "Text Baseline", plots_dir)

    logger.info("Baseline — Macro F1: %.4f | AUC: %s | Latency: %.2fms",
                macro_f1, f"{macro_auc:.4f}" if macro_auc else "N/A", lat_mean)

    return ModelReport(
        model_name="Text Baseline (LR+SVM+RF Ensemble)",
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        accuracy=acc,
        macro_auc=macro_auc,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_prec,
        per_class_recall=per_class_rec,
        latency_ms_mean=lat_mean,
        latency_ms_p95=lat_p95,
        n_samples=len(X_test),
        target_met=macro_f1 >= 0.82,
    )


# ── PhoBERT evaluator ─────────────────────────────────────────────────────────

def evaluate_phobert(
    model_dir: str,
    X_test: pd.Series,
    y_test_str: np.ndarray,
    class_labels: List[str],
    plots_dir: str,
    batch_size: int = 32,
    max_length: int = 256,
) -> ModelReport:
    """Đánh giá PhoBERT fine-tuned model.

    Args:
        model_dir: Thư mục chứa checkpoint PhoBERT (saved_model hoặc checkpoint).
        X_test: Văn bản test (pd.Series).
        y_test_str: Nhãn thực tế dạng string.
        class_labels: Danh sách tên lớp (theo đúng thứ tự LABEL_MAP).
        plots_dir: Thư mục lưu confusion matrix.
        batch_size: Batch size cho inference.
        max_length: Độ dài token tối đa.

    Returns:
        ModelReport đầy đủ.
    """
    logger.info("Đang tải PhoBERT từ: %s", model_dir)

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise ImportError("Cần cài torch và transformers: pip install torch transformers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()

    # Chuyển nhãn string → integer
    y_test_int = np.array([_LABEL_MAP.get(lbl, -1) for lbl in y_test_str])

    # Batch inference
    texts_list = X_test.fillna("").tolist()
    all_preds = []
    all_proba = []

    logger.info("Đang inference PhoBERT trên %d mẫu (batch_size=%d)...",
                len(texts_list), batch_size)

    with torch.no_grad():
        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i:i + batch_size]
            encoding = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoding = {k: v.to(device) for k, v in encoding.items()}
            outputs = model(**encoding)
            logits = outputs.logits
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(proba, axis=-1)
            all_preds.extend(preds.tolist())
            all_proba.append(proba)

    y_pred_int = np.array(all_preds)
    y_proba = np.vstack(all_proba)

    # Map predictions back to strings
    y_pred_str = np.array([_ID_TO_LABEL.get(p, str(p)) for p in y_pred_int])

    # Metrics
    macro_f1 = float(f1_score(y_test_str, y_pred_str, average="macro",
                               zero_division=0, labels=class_labels))
    weighted_f1 = float(f1_score(y_test_str, y_pred_str, average="weighted",
                                  zero_division=0, labels=class_labels))
    acc = float(accuracy_score(y_test_str, y_pred_str))

    report_dict = classification_report(
        y_test_str, y_pred_str, labels=class_labels, zero_division=0, output_dict=True,
    )
    per_class_f1 = {k: round(v["f1-score"], 4)
                    for k, v in report_dict.items()
                    if k in class_labels}
    per_class_prec = {k: round(v["precision"], 4)
                      for k, v in report_dict.items()
                      if k in class_labels}
    per_class_rec = {k: round(v["recall"], 4)
                     for k, v in report_dict.items()
                     if k in class_labels}

    macro_auc = _compute_roc_auc_multiclass(y_test_str, y_proba, class_labels)

    # Latency (trên 20 mẫu để tránh OOM khi đo nhiều lần)
    sample_size = min(20, len(texts_list))
    X_lat_sample = texts_list[:sample_size]

    def _phobert_predict_batch(batch_texts):
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        return np.argmax(out.logits.cpu().numpy(), axis=-1)

    lat_mean, lat_p95 = _measure_latency(
        predict_fn=_phobert_predict_batch,
        X_sample=X_lat_sample,
        n_warmup=2,
        n_repeat=10,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test_str, y_pred_str, labels=class_labels)
    _save_confusion_matrix_plot(cm, class_labels, "PhoBERT", plots_dir)

    logger.info("PhoBERT — Macro F1: %.4f | AUC: %s | Latency: %.2fms",
                macro_f1, f"{macro_auc:.4f}" if macro_auc else "N/A", lat_mean)

    return ModelReport(
        model_name="PhoBERT (vinai/phobert-base-v2 fine-tuned)",
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        accuracy=acc,
        macro_auc=macro_auc,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_prec,
        per_class_recall=per_class_rec,
        latency_ms_mean=lat_mean,
        latency_ms_p95=lat_p95,
        n_samples=len(X_test),
        target_met=macro_f1 >= 0.82,
    )


# ── Comparison table printer ──────────────────────────────────────────────────

def print_comparison_table(reports: List[ModelReport]) -> None:
    """In bảng so sánh tất cả mô hình."""
    print("\n" + "=" * 90)
    print("  SO SÁNH HIỆU NĂNG: Text Baseline vs PhoBERT".center(90))
    print("=" * 90)

    # In chi tiết từng mô hình
    for report in reports:
        _print_model_section(report)

    # In bảng tổng kết
    print("\n" + "─" * 90)
    print("  BẢNG TÓM TẮT".center(90))
    print("─" * 90)

    col_widths = [35, 10, 12, 10, 10, 14, 14]
    headers = ["Mô hình", "MacroF1", "WeightedF1", "Accuracy", "ROC-AUC",
               "Latency(ms)", "Target(≥0.82)"]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    row_fmt = "|" + "|".join(f" {{:<{w}}} " for w in col_widths) + "|"

    print(sep)
    print(row_fmt.format(*headers))
    print(sep)

    for r in reports:
        auc_str = f"{r.macro_auc:.4f}" if r.macro_auc is not None else "N/A"
        target_str = "✅ YES" if r.target_met else "❌ NO"
        print(row_fmt.format(
            r.model_name[:35],
            f"{r.macro_f1:.4f}",
            f"{r.weighted_f1:.4f}",
            f"{r.accuracy:.4f}",
            auc_str,
            f"{r.latency_ms_mean:.2f}",
            target_str,
        ))

    print(sep)

    if len(reports) >= 2:
        best = max(reports, key=lambda r: r.macro_f1)
        # Hiệu số F1
        sorted_by_f1 = sorted(reports, key=lambda r: r.macro_f1, reverse=True)
        if len(sorted_by_f1) >= 2:
            diff = sorted_by_f1[0].macro_f1 - sorted_by_f1[1].macro_f1
            speed_ratio = sorted_by_f1[1].latency_ms_mean / max(sorted_by_f1[0].latency_ms_mean, 0.001)
            print(f"\n  ★ Mô hình tốt nhất (F1): {best.model_name} "
                  f"(Macro F1 = {best.macro_f1:.4f})")
            print(f"  ↑ Chênh lệch F1 so với mô hình thứ hai: +{diff:.4f}")
            print(f"  ⚡ Tỉ lệ tốc độ ({sorted_by_f1[1].model_name[:20]} / "
                  f"{sorted_by_f1[0].model_name[:20]}): {speed_ratio:.1f}x")

    print("=" * 90 + "\n")


# ── Save report ───────────────────────────────────────────────────────────────

def save_comparison_report(reports: List[ModelReport], metrics_dir: str) -> str:
    """Lưu báo cáo so sánh ra JSON.

    Args:
        reports: Danh sách ModelReport.
        metrics_dir: Thư mục lưu.

    Returns:
        Đường dẫn file JSON.
    """
    os.makedirs(metrics_dir, exist_ok=True)
    out_path = os.path.join(metrics_dir, "model_comparison_report.json")

    best = max(reports, key=lambda r: r.macro_f1)
    payload = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "best_model": best.model_name,
        "reports": [asdict(r) for r in reports],
        "summary": {
            r.model_name: {
                "macro_f1": r.macro_f1,
                "weighted_f1": r.weighted_f1,
                "accuracy": r.accuracy,
                "macro_auc": r.macro_auc,
                "latency_ms_mean": r.latency_ms_mean,
                "target_met": r.target_met,
            }
            for r in reports
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Báo cáo so sánh đã lưu → %s", out_path)
    return out_path


# ── Sanity check ──────────────────────────────────────────────────────────────

def run_sanity_check() -> None:
    """Chạy kiểm tra nhanh framework bằng dữ liệu tổng hợp."""
    logger.info("Chạy sanity-check với dữ liệu tổng hợp...")
    rng = np.random.default_rng(42)
    class_labels = ["tích cực", "tiêu cực", "trung lập"]
    n = 300

    y_true = rng.choice(class_labels, size=n, p=[0.75, 0.15, 0.10])
    y_proba = rng.dirichlet(alpha=[3, 1, 1], size=n)
    y_pred_int = np.argmax(y_proba, axis=1)
    y_pred = np.array([class_labels[i] for i in y_pred_int])

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0,
                               labels=class_labels))

    report_dict = classification_report(
        y_true, y_pred, labels=class_labels, zero_division=0, output_dict=True,
    )
    per_class_f1 = {k: round(v["f1-score"], 4)
                    for k, v in report_dict.items()
                    if k in class_labels}
    per_class_prec = {k: round(v["precision"], 4)
                      for k, v in report_dict.items()
                      if k in class_labels}
    per_class_rec = {k: round(v["recall"], 4)
                     for k, v in report_dict.items()
                     if k in class_labels}

    macro_auc = _compute_roc_auc_multiclass(y_true, y_proba, class_labels)

    fake_report = ModelReport(
        model_name="[Sanity] Fake Model",
        macro_f1=macro_f1,
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted",
                                    zero_division=0, labels=class_labels)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_auc=macro_auc,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_prec,
        per_class_recall=per_class_rec,
        latency_ms_mean=1.23,
        latency_ms_p95=2.45,
        n_samples=n,
        target_met=macro_f1 >= 0.82,
    )

    print_comparison_table([fake_report])
    logger.info("Sanity-check hoàn thành — Framework hoạt động bình thường.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="So sánh Text Baseline vs PhoBERT trên cùng test set",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default=None,
        help="Đường dẫn tới file .pkl của TextEnsembleModel",
    )
    parser.add_argument(
        "--phobert-path",
        type=str,
        default=None,
        help="Thư mục chứa checkpoint PhoBERT fine-tuned",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=_DATA_PATH,
        help=f"Đường dẫn CSV dữ liệu (default: {_DATA_PATH})",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default=_TEXT_COL,
        help=f"Tên cột văn bản (default: {_TEXT_COL})",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=_LABEL_COL,
        help=f"Tên cột nhãn (default: {_LABEL_COL})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=_TEST_SIZE,
        help="Tỉ lệ test set (default: 0.20)",
    )
    parser.add_argument(
        "--no-phobert",
        action="store_true",
        help="Bỏ qua đánh giá PhoBERT (chỉ đánh giá baseline)",
    )
    parser.add_argument(
        "--phobert-batch-size",
        type=int,
        default=32,
        help="Batch size cho PhoBERT inference (default: 32)",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=_METRICS_DIR,
        help=f"Thư mục lưu JSON report (default: {_METRICS_DIR})",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=_PLOTS_DIR,
        help=f"Thư mục lưu confusion matrix PNG (default: {_PLOTS_DIR})",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Chạy sanity-check với dữ liệu tổng hợp (không cần file model)",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Pipeline chính: tải dữ liệu → đánh giá → so sánh → lưu báo cáo."""
    args = parse_args()

    if args.sanity:
        run_sanity_check()
        return

    # Validate inputs
    if args.baseline_path is None and args.phobert_path is None:
        print("Lỗi: Cần cung cấp ít nhất một trong: --baseline-path hoặc --phobert-path")
        print("  Hoặc dùng --sanity để kiểm tra framework với dữ liệu giả.")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("  SO SÁNH MÔ HÌNH: Baseline vs PhoBERT")
    logger.info("=" * 70)

    # 1. Tải dữ liệu
    if not os.path.exists(args.data_path):
        logger.error("Không tìm thấy file dữ liệu: %s", args.data_path)
        sys.exit(1)

    logger.info("Đang tải dữ liệu từ: %s", args.data_path)
    df = pd.read_csv(args.data_path, encoding="utf-8-sig")
    logger.info("Raw shape: %s", df.shape)

    # Validate columns
    for col in [args.text_col, args.label_col]:
        if col not in df.columns:
            logger.error("Thiếu cột '%s'. Các cột hiện có: %s", col, list(df.columns))
            sys.exit(1)

    df = df.dropna(subset=[args.text_col, args.label_col])

    # Xác định class labels từ dữ liệu
    known_labels = list(_LABEL_MAP.keys())
    df = df[df[args.label_col].isin(known_labels)].reset_index(drop=True)

    logger.info("Usable samples: %d", len(df))
    logger.info("Phân phối nhãn:\n%s", df[args.label_col].value_counts().to_string())

    # 2. Stratified split (cùng seed để baseline và PhoBERT dùng cùng test set)
    _, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=_RANDOM_STATE,
        stratify=df[args.label_col],
    )
    logger.info("Test set: %d mẫu", len(test_df))

    X_test = test_df[args.text_col].fillna("")
    y_test_str = test_df[args.label_col].values

    # Xác định class_labels theo thứ tự _LABEL_MAP
    class_labels = [lbl for lbl in known_labels if lbl in np.unique(y_test_str)]

    reports: List[ModelReport] = []

    # 3. Đánh giá Baseline
    if args.baseline_path:
        if not os.path.exists(args.baseline_path):
            logger.error("Không tìm thấy baseline model: %s", args.baseline_path)
        else:
            try:
                bl_report = evaluate_baseline(
                    model_path=args.baseline_path,
                    X_test=X_test,
                    y_test_str=y_test_str,
                    class_labels=class_labels,
                    plots_dir=args.plots_dir,
                )
                reports.append(bl_report)
            except Exception as exc:
                logger.error("Lỗi khi đánh giá baseline: %s", exc, exc_info=True)

    # 4. Đánh giá PhoBERT
    if args.phobert_path and not args.no_phobert:
        if not os.path.exists(args.phobert_path):
            logger.error("Không tìm thấy PhoBERT model: %s", args.phobert_path)
        else:
            try:
                pb_report = evaluate_phobert(
                    model_dir=args.phobert_path,
                    X_test=X_test,
                    y_test_str=y_test_str,
                    class_labels=class_labels,
                    plots_dir=args.plots_dir,
                    batch_size=args.phobert_batch_size,
                )
                reports.append(pb_report)
            except Exception as exc:
                logger.error("Lỗi khi đánh giá PhoBERT: %s", exc, exc_info=True)

    if not reports:
        logger.error("Không có mô hình nào được đánh giá thành công.")
        sys.exit(1)

    # 5. In bảng so sánh
    print_comparison_table(reports)

    # 6. Lưu báo cáo JSON
    report_path = save_comparison_report(reports, args.metrics_dir)
    logger.info("Hoàn thành! Báo cáo tại: %s", report_path)


if __name__ == "__main__":
    main()
