"""
tune_spam_model.py
==================
Tinh chỉnh siêu tham số cho SpamHybridModel (Rule-based + Isolation Forest).

Vì Isolation Forest là unsupervised (không có nhãn ground-truth thật),
chiến lược tuning dùng nhãn rule-based `is_spam` làm **proxy ground-truth**
để đánh giá mức độ đồng thuận giữa IForest và bộ quy tắc.

Các tham số được tuning:
  1. IsolationForest:
       - contamination  : [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
       - n_estimators   : [100, 150, 200, 300]
       - max_samples    : ["auto", 0.7, 1.0]
  2. Rule-based:
       - dup_threshold  : [0.75, 0.80, 0.85, 0.90]  (cosine similarity cho duplicate seeding)

Output:
  - Bảng so sánh tất cả cấu hình (precision, recall, F1 vs rule-based)
  - Best config lưu vào artifacts/metrics/spam_tuning_results.json
  - Best model lưu vào artifacts/models/tuned/tuned_spam_iforest.pkl

Usage (từ thư mục gốc):
    py scripts/tune_spam_model.py
    py scripts/tune_spam_model.py --data-path data/processed/reviews_flagged.csv
    py scripts/tune_spam_model.py --quick  # nhanh hơn, grid nhỏ hơn
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Force UTF-8 stdout/stderr ──────────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from ai_engine.text_processing.spam_filter import detect_spam

# ── Import feature builder from train_spam_model ─────────────────────────────
from scripts.train_spam_model import (
    build_feature_matrix,
    extract_structural_features,
    SpamHybridModel,
    RULE_FLAG_COLS,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_DATA_PATH = str(ROOT / "data" / "processed" / "spam_labeled_text.csv")
_ARTIFACTS_DIR = str(ROOT / "artifacts" / "spam")
_METRICS_DIR = str(ROOT / "artifacts" / "spam")
_RANDOM_STATE = 42


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SpamTuningConfig:
    """Một cấu hình tham số cần thử."""
    contamination: float
    n_estimators: int
    max_samples: object   # "auto" | float
    dup_threshold: float


@dataclass
class SpamTuningResult:
    """Kết quả đánh giá một cấu hình."""
    contamination: float
    n_estimators: int
    max_samples: object
    dup_threshold: float
    # Metrics so với rule-based ground-truth (is_spam)
    iforest_precision: float   # precision của IForest vs rule labels
    iforest_recall: float
    iforest_f1: float
    # Metrics của final spam (union rule + iforest) vs rule labels
    final_precision: float
    final_recall: float
    final_f1: float
    # Tỉ lệ spam bắt được
    rule_spam_rate: float       # % rule-based flagged
    iforest_spam_rate: float    # % IForest flagged
    final_spam_rate: float      # % final flagged
    elapsed_seconds: float


# ── Grid definitions ──────────────────────────────────────────────────────────

def get_spam_grid(quick: bool = False) -> List[SpamTuningConfig]:
    """Trả về danh sách cấu hình cần thử.

    Args:
        quick: Nếu True, dùng grid nhỏ hơn.

    Returns:
        Danh sách SpamTuningConfig.
    """
    if quick:
        contaminations = [0.08, 0.10, 0.15]
        n_estimators_list = [100, 200]
        max_samples_list = ["auto"]
        dup_thresholds = [0.80, 0.85]
    else:
        contaminations = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
        n_estimators_list = [100, 150, 200, 300]
        max_samples_list = ["auto", 0.7, 1.0]
        dup_thresholds = [0.75, 0.80, 0.85, 0.90]

    configs = []
    for cont, n_est, max_s, dup in product(
        contaminations, n_estimators_list, max_samples_list, dup_thresholds
    ):
        configs.append(SpamTuningConfig(
            contamination=cont,
            n_estimators=n_est,
            max_samples=max_s,
            dup_threshold=dup,
        ))

    logger.info("Tổng số cấu hình cần thử: %d", len(configs))
    return configs


# ── Data loading ──────────────────────────────────────────────────────────────

def load_spam_data(data_path: str) -> pd.DataFrame:
    """Tải dữ liệu đã có cột rule-based `is_spam`.

    Nếu file chưa có cột `is_spam`, sẽ chạy detect_spam() để tạo.

    Args:
        data_path: Đường dẫn tới CSV.

    Returns:
        DataFrame có cột: text, rating, is_spam.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path.resolve()}")

    logger.info("Đang tải dữ liệu từ: %s", path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Loaded %d rows, columns: %s", len(df), df.columns.tolist())

    # Chuẩn hóa tên cột
    if "text" not in df.columns:
        # Tìm cột text có thể có tên khác
        text_candidates = [c for c in df.columns if "text" in c.lower() or "review" in c.lower()]
        if text_candidates:
            df = df.rename(columns={text_candidates[0]: "text"})
        else:
            raise ValueError(f"Không tìm thấy cột text. Các cột: {df.columns.tolist()}")

    if "rating" not in df.columns:
        rating_candidates = [c for c in df.columns if "rating" in c.lower() or "star" in c.lower()]
        if rating_candidates:
            df = df.rename(columns={rating_candidates[0]: "rating"})

    df["text"] = df["text"].fillna("").astype(str)
    df["rating"] = pd.to_numeric(df.get("rating", pd.Series([3] * len(df))), errors="coerce").fillna(3)

    # Loại bỏ rows quá ngắn hoặc rỗng
    before = len(df)
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    logger.info("Sau khi loại bỏ empty text: %d rows (bỏ %d)", len(df), before - len(df))

    # Giới hạn mẫu để tuning nhanh (lấy tối đa 5000 rows để tránh OOM)
    # (Đã comment lại theo yêu cầu chạy FULL toàn bộ dữ liệu của user)
    # if len(df) > 5_000:
    #     logger.info("Giới hạn %d → 5000 rows để tuning nhanh hơn.", len(df))
    #     df = df.sample(n=5_000, random_state=_RANDOM_STATE).reset_index(drop=True)

    return df


# ── Evaluate one config ───────────────────────────────────────────────────────

def evaluate_config(
    config: SpamTuningConfig,
    df: pd.DataFrame,
    X_precomputed: Optional[np.ndarray] = None,
    df_flagged_cache: Optional[Dict[float, pd.DataFrame]] = None,
) -> SpamTuningResult:
    """Đánh giá một cấu hình tham số.

    Nếu dup_threshold giống cấu hình trước, dùng lại df_flagged từ cache
    để tránh chạy lại detect_spam() tốn thời gian.

    Args:
        config: Cấu hình cần đánh giá.
        df: DataFrame gốc (text, rating).
        X_precomputed: Feature matrix đã tính (nếu dup_threshold không đổi).
        df_flagged_cache: Cache {dup_threshold → df_flagged} để tái sử dụng.

    Returns:
        SpamTuningResult với đầy đủ metrics.
    """
    t0 = time.perf_counter()

    # Step 1: Rule-based detection (có thể dùng cache)
    if df_flagged_cache is not None and config.dup_threshold in df_flagged_cache:
        df_flagged = df_flagged_cache[config.dup_threshold]
    else:
        df_flagged = detect_spam(df[["text", "rating"]], dup_threshold=config.dup_threshold)
        if df_flagged_cache is not None:
            df_flagged_cache[config.dup_threshold] = df_flagged

    rule_is_spam = df_flagged["is_spam"].values.astype(int)

    # Step 2: Feature matrix (tái sử dụng nếu dup_threshold giống)
    if X_precomputed is not None and config.dup_threshold == list(df_flagged_cache.keys())[0]:
        X = X_precomputed
    else:
        X = build_feature_matrix(df_flagged, df["text"].tolist(), df["rating"].tolist())

    # Step 3: Isolation Forest với config hiện tại
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iforest = IsolationForest(
        contamination=config.contamination,
        n_estimators=config.n_estimators,
        max_samples=config.max_samples,
        random_state=_RANDOM_STATE,
        n_jobs=-1,
    )
    iforest.fit(X_scaled)
    iforest_pred = iforest.predict(X_scaled)   # 1=normal, -1=anomaly
    iforest_spam = (iforest_pred == -1).astype(int)

    # Step 4: Final spam = rule OR iforest
    final_spam = np.clip(rule_is_spam + iforest_spam, 0, 1)

    n = len(rule_is_spam)
    rule_spam_rate = float(rule_is_spam.sum()) / n
    iforest_spam_rate = float(iforest_spam.sum()) / n
    final_spam_rate = float(final_spam.sum()) / n

    # Metrics của IForest vs rule-based (dùng rule_is_spam làm proxy GT)
    # Nếu không có spam nào trong rule, tránh division by zero
    if rule_is_spam.sum() == 0:
        iforest_prec = iforest_rec = iforest_f1 = 0.0
        final_prec = final_rec = final_f1 = 0.0
    else:
        iforest_prec = float(precision_score(rule_is_spam, iforest_spam, zero_division=0))
        iforest_rec  = float(recall_score(rule_is_spam, iforest_spam, zero_division=0))
        iforest_f1   = float(f1_score(rule_is_spam, iforest_spam, zero_division=0))
        final_prec   = float(precision_score(rule_is_spam, final_spam, zero_division=0))
        final_rec    = float(recall_score(rule_is_spam, final_spam, zero_division=0))
        final_f1     = float(f1_score(rule_is_spam, final_spam, zero_division=0))

    elapsed = time.perf_counter() - t0

    return SpamTuningResult(
        contamination=config.contamination,
        n_estimators=config.n_estimators,
        max_samples=config.max_samples,
        dup_threshold=config.dup_threshold,
        iforest_precision=round(iforest_prec, 4),
        iforest_recall=round(iforest_rec, 4),
        iforest_f1=round(iforest_f1, 4),
        final_precision=round(final_prec, 4),
        final_recall=round(final_rec, 4),
        final_f1=round(final_f1, 4),
        rule_spam_rate=round(rule_spam_rate, 4),
        iforest_spam_rate=round(iforest_spam_rate, 4),
        final_spam_rate=round(final_spam_rate, 4),
        elapsed_seconds=round(elapsed, 2),
    )


# ── Print results ─────────────────────────────────────────────────────────────

def print_top_results(results: List[SpamTuningResult], top_n: int = 10) -> None:
    """In top N cấu hình tốt nhất (theo IForest F1)."""
    sorted_r = sorted(results, key=lambda r: r.iforest_f1, reverse=True)[:top_n]

    print("\n" + "=" * 100)
    print("  TOP {} CẤU HÌNH TỐT NHẤT — SpamHybridModel Tuning".format(top_n).center(100))
    print("=" * 100)

    col_widths = [8, 8, 8, 8, 10, 10, 10, 12, 12, 12]
    headers = ["cont.", "n_est", "max_s", "dup_t",
               "IF_prec", "IF_rec", "IF_F1",
               "final_prec", "final_rec", "final_F1"]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    row_fmt = "|" + "|".join(f" {{:<{w}}} " for w in col_widths) + "|"

    print(sep)
    print(row_fmt.format(*headers))
    print(sep)

    for r in sorted_r:
        max_s_str = str(r.max_samples) if isinstance(r.max_samples, str) else f"{r.max_samples:.1f}"
        print(row_fmt.format(
            str(r.contamination),
            str(r.n_estimators),
            max_s_str,
            str(r.dup_threshold),
            f"{r.iforest_precision:.4f}",
            f"{r.iforest_recall:.4f}",
            f"{r.iforest_f1:.4f}",
            f"{r.final_precision:.4f}",
            f"{r.final_recall:.4f}",
            f"{r.final_f1:.4f}",
        ))

    print(sep)
    best = sorted_r[0]
    print(f"\n  ★ Best config:")
    print(f"     contamination = {best.contamination}")
    print(f"     n_estimators  = {best.n_estimators}")
    print(f"     max_samples   = {best.max_samples}")
    print(f"     dup_threshold = {best.dup_threshold}")
    print(f"     IForest F1    = {best.iforest_f1:.4f}")
    print(f"     Final F1      = {best.final_f1:.4f}")
    print(f"     Spam rates: rule={best.rule_spam_rate:.1%} | iforest={best.iforest_spam_rate:.1%} | final={best.final_spam_rate:.1%}")
    print("=" * 100 + "\n")


# ── Save best model ───────────────────────────────────────────────────────────

def train_and_save_best(
    best: SpamTuningResult,
    df: pd.DataFrame,
    artifacts_dir: str,
) -> str:
    """Train best config trên toàn bộ data và lưu model.

    Args:
        best: Cấu hình tốt nhất.
        df: DataFrame gốc.
        artifacts_dir: Thư mục lưu.

    Returns:
        Đường dẫn file .pkl đã lưu.
    """
    logger.info("Training best config trên toàn bộ %d samples...", len(df))

    df_flagged = detect_spam(df[["text", "rating"]], dup_threshold=best.dup_threshold)
    X = build_feature_matrix(df_flagged, df["text"].tolist(), df["rating"].tolist())

    model = SpamHybridModel(
        contamination=best.contamination,
        n_estimators=best.n_estimators,
        random_state=_RANDOM_STATE,
    )
    # Override max_samples nếu khác "auto"
    if best.max_samples != "auto":
        model.iforest = IsolationForest(
            contamination=best.contamination,
            n_estimators=best.n_estimators,
            max_samples=best.max_samples,
            random_state=_RANDOM_STATE,
            n_jobs=-1,
        )
        X_scaled = model.scaler.fit_transform(X)
        model.iforest.fit(X_scaled)
    else:
        model.fit(X)

    import os
    os.makedirs(artifacts_dir, exist_ok=True)
    save_path = str(Path(artifacts_dir) / "tuned_spam_iforest.pkl")
    model.save(save_path)
    logger.info("Best spam model đã lưu → %s", save_path)
    return save_path


# ── Save metrics JSON ─────────────────────────────────────────────────────────

def save_spam_metrics(
    results: List[SpamTuningResult],
    best: SpamTuningResult,
    metrics_dir: str,
) -> str:
    """Lưu kết quả tuning spam ra JSON.

    Args:
        results: Toàn bộ kết quả.
        best: Cấu hình tốt nhất.
        metrics_dir: Thư mục lưu.

    Returns:
        Đường dẫn JSON.
    """
    import os
    os.makedirs(metrics_dir, exist_ok=True)
    out_path = str(Path(metrics_dir) / "spam_tuning_results.json")

    payload = {
        "best_config": asdict(best),
        "n_configs_tested": len(results),
        "top10": [asdict(r) for r in sorted(results, key=lambda r: r.iforest_f1, reverse=True)[:10]],
        "all_results": [asdict(r) for r in results],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Spam tuning results → %s", out_path)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning cho SpamHybridModel (Isolation Forest)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data-path", type=str, default=_DATA_PATH,
        help=f"CSV có cột text, rating (default: {_DATA_PATH})",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Dùng grid nhỏ hơn để chạy nhanh (dev/debug)",
    )
    parser.add_argument(
        "--artifacts-dir", type=str, default=_ARTIFACTS_DIR,
        help=f"Thư mục lưu model tốt nhất (default: {_ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--metrics-dir", type=str, default=_METRICS_DIR,
        help=f"Thư mục lưu JSON (default: {_METRICS_DIR})",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Số cấu hình tốt nhất in ra bảng (default: 10)",
    )
    parser.add_argument(
        "--no-save-model", action="store_true",
        help="Không lưu model best (chỉ in kết quả)",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Pipeline: tải data → sweep configs → báo cáo → lưu best model."""
    args = parse_args()

    logger.info("=" * 70)
    logger.info("  SPAM MODEL HYPERPARAMETER TUNING")
    logger.info("  Mode: %s", "QUICK" if args.quick else "FULL")
    logger.info("=" * 70)

    # 1. Tải data
    try:
        df = load_spam_data(args.data_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Lỗi tải data: %s", exc)
        sys.exit(1)

    # 2. Lấy grid
    configs = get_spam_grid(quick=args.quick)
    logger.info("Bắt đầu sweep %d cấu hình...", len(configs))

    # 3. Cache để tránh chạy lại detect_spam() cho cùng dup_threshold
    df_flagged_cache: Dict[float, pd.DataFrame] = {}
    X_cache: Dict[float, np.ndarray] = {}

    results: List[SpamTuningResult] = []
    n_done = 0

    for config in configs:
        try:
            # Precompute df_flagged + X nếu chưa có trong cache
            if config.dup_threshold not in df_flagged_cache:
                logger.info("  [detect_spam] dup_threshold=%.2f ...", config.dup_threshold)
                df_flagged = detect_spam(df[["text", "rating"]], dup_threshold=config.dup_threshold)
                df_flagged_cache[config.dup_threshold] = df_flagged
                X = build_feature_matrix(df_flagged, df["text"].tolist(), df["rating"].tolist())
                X_cache[config.dup_threshold] = X

            df_flagged = df_flagged_cache[config.dup_threshold]
            X = X_cache[config.dup_threshold]

            rule_is_spam = df_flagged["is_spam"].values.astype(int)

            # Train IForest với config
            t0 = time.perf_counter()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            iforest = IsolationForest(
                contamination=config.contamination,
                n_estimators=config.n_estimators,
                max_samples=config.max_samples,
                random_state=_RANDOM_STATE,
                n_jobs=-1,
            )
            iforest.fit(X_scaled)
            iforest_pred = iforest.predict(X_scaled)
            iforest_spam = (iforest_pred == -1).astype(int)
            final_spam = np.clip(rule_is_spam + iforest_spam, 0, 1)
            elapsed = time.perf_counter() - t0

            n = len(rule_is_spam)
            if rule_is_spam.sum() == 0:
                iforest_prec = iforest_rec = iforest_f1 = 0.0
                final_prec = final_rec = final_f1 = 0.0
            else:
                iforest_prec = float(precision_score(rule_is_spam, iforest_spam, zero_division=0))
                iforest_rec  = float(recall_score(rule_is_spam, iforest_spam, zero_division=0))
                iforest_f1   = float(f1_score(rule_is_spam, iforest_spam, zero_division=0))
                final_prec   = float(precision_score(rule_is_spam, final_spam, zero_division=0))
                final_rec    = float(recall_score(rule_is_spam, final_spam, zero_division=0))
                final_f1     = float(f1_score(rule_is_spam, final_spam, zero_division=0))

            result = SpamTuningResult(
                contamination=config.contamination,
                n_estimators=config.n_estimators,
                max_samples=config.max_samples,
                dup_threshold=config.dup_threshold,
                iforest_precision=round(iforest_prec, 4),
                iforest_recall=round(iforest_rec, 4),
                iforest_f1=round(iforest_f1, 4),
                final_precision=round(final_prec, 4),
                final_recall=round(final_rec, 4),
                final_f1=round(final_f1, 4),
                rule_spam_rate=round(float(rule_is_spam.sum()) / n, 4),
                iforest_spam_rate=round(float(iforest_spam.sum()) / n, 4),
                final_spam_rate=round(float(final_spam.sum()) / n, 4),
                elapsed_seconds=round(elapsed, 2),
            )
            results.append(result)

            n_done += 1
            if n_done % 20 == 0 or n_done == len(configs):
                logger.info("  Đã chạy %d/%d cấu hình...", n_done, len(configs))

        except Exception as exc:
            logger.warning("  Lỗi với config %s: %s", config, exc)

    if not results:
        logger.error("Không có cấu hình nào chạy được.")
        sys.exit(1)

    # 4. Báo cáo
    print_top_results(results, top_n=min(args.top_n, len(results)))

    # 5. Lưu metrics
    best = max(results, key=lambda r: r.iforest_f1)
    metrics_path = save_spam_metrics(results, best, args.metrics_dir)
    logger.info("Metrics đã lưu → %s", metrics_path)

    # 6. Lưu best model
    if not args.no_save_model:
        try:
            save_path = train_and_save_best(best, df, args.artifacts_dir)
            logger.info("Best model đã lưu → %s", save_path)
        except Exception as exc:
            logger.error("Không thể lưu best model: %s", exc, exc_info=True)

    logger.info("Hoàn thành Spam Tuning!")


if __name__ == "__main__":
    main()
