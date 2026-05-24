"""
tune_text_hyperparams.py
========================
Tinh chỉnh siêu tham số (Hyperparameter Tuning) bằng GridSearchCV
cho các mô hình Text Baseline trong dự án ecommerce-review-analytics.

Các mô hình được tuning:
  1. Logistic Regression  + TF-IDF
  2. LinearSVC (Calibrated) + TF-IDF
  3. Random Forest          + TF-IDF
  4. Ensemble tốt nhất từ các mô hình trên

Chiến lược:
  - Tất cả Grid Search chạy trên StratifiedKFold(5) với scoring='f1_macro'
  - Sau khi tìm best params, retrain trên toàn bộ train set
  - Lưu best model và báo cáo kết quả vào artifacts/

Usage (từ thư mục gốc project):
    python scripts/tune_text_hyperparams.py
    python scripts/tune_text_hyperparams.py --cv 3 --n-jobs -1
    python scripts/tune_text_hyperparams.py --quick   # quick mode: nhỏ hơn grid
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import io
import joblib
import numpy as np
import pandas as pd

# ── Force UTF-8 stdout/stderr (avoid cp1252 UnicodeEncodeError on Windows) ────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Ensure project root is importable ─────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE
# pyrefly: ignore [missing-import]
from imblearn.pipeline import Pipeline as ImbPipeline
# pyrefly: ignore [missing-import]
from sklearn.calibration import CalibratedClassifierCV
# pyrefly: ignore [missing-import]
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# pyrefly: ignore [missing-import]
from sklearn.feature_extraction.text import TfidfVectorizer
# pyrefly: ignore [missing-import]
from sklearn.linear_model import LogisticRegression
# pyrefly: ignore [missing-import]
from sklearn.metrics import classification_report, f1_score
# pyrefly: ignore [missing-import]
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
# pyrefly: ignore [missing-import]
from sklearn.svm import LinearSVC

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_DATA_PATH = str(_PROJECT_ROOT / "data" / "processed" / "processed_labeled_reviews.csv")
_ARTIFACTS_DIR = str(_PROJECT_ROOT / "artifacts" / "models" / "tuned")
_METRICS_DIR = str(_PROJECT_ROOT / "artifacts" / "metrics")
_TEXT_COL = "cleaned_text"
_LABEL_COL = "sentiment_label"
_TEST_SIZE = 0.20
_RANDOM_STATE = 42


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TuningResult:
    """Lưu kết quả tuning của một mô hình."""
    model_name: str
    best_params: Dict[str, Any]
    cv_best_f1: float        # F1 trên cross-validation (best_score_)
    test_macro_f1: float     # F1 trên tập test hold-out 20%
    test_weighted_f1: float
    elapsed_seconds: float
    save_path: str


# ── Full Grid Definitions ─────────────────────────────────────────────────────

def get_grids(quick: bool = False) -> Dict[str, Tuple[ImbPipeline, List[Dict]]]:
    """Trả về dict {tên_mô_hình: (pipeline, param_grid)}.

    Args:
        quick: Nếu True, dùng grid nhỏ hơn (chạy nhanh hơn cho dev/debug).
    """
    # ── TF-IDF common options ────────────────────────────────────────────────
    tfidf_base = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.90)

    if quick:
        tfidf_params = [
            {
                "tfidf__max_features": [10_000],
                "tfidf__ngram_range": [(1, 2)],
            }
        ]
    else:
        tfidf_params = [
            {
                "tfidf__max_features": [10_000, 20_000],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
            }
        ]

    # ════════════════════════════════════════════
    # 1. Logistic Regression
    # ════════════════════════════════════════════
    lr_pipeline = ImbPipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.90)),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1_500,
            random_state=_RANDOM_STATE,
        )),
    ])

    if quick:
        lr_grid = [
            {
                **tfidf_params[0],
                "clf__C": [0.5, 1.0],
                "clf__solver": ["lbfgs"],
                "clf__penalty": ["l2"],
            }
        ]
    else:
        lr_grid = [
            {
                **tfidf_params[0],
                "clf__C": [0.1, 0.5, 1.0, 5.0, 10.0],
                "clf__solver": ["lbfgs", "saga"],
                "clf__penalty": ["l2"],
            },
            {
                **tfidf_params[0],
                "clf__C": [0.1, 0.5, 1.0, 5.0],
                "clf__solver": ["saga"],
                "clf__penalty": ["l1"],
            },
        ]

    # ════════════════════════════════════════════
    # 2. LinearSVC (Calibrated)
    # ════════════════════════════════════════════
    svm_raw = LinearSVC(class_weight="balanced", dual="auto", random_state=_RANDOM_STATE)
    svm_pipeline = ImbPipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.90)),
        ("clf", CalibratedClassifierCV(svm_raw, cv=3, method="isotonic")),
    ])

    if quick:
        svm_grid = [
            {
                **tfidf_params[0],
                "clf__estimator__C": [0.5, 1.0],
                "clf__estimator__max_iter": [2_000],
            }
        ]
    else:
        svm_grid = [
            {
                **tfidf_params[0],
                "clf__estimator__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                "clf__estimator__max_iter": [2_000, 5_000],
            }
        ]

    # ════════════════════════════════════════════
    # 3. Random Forest
    # ════════════════════════════════════════════
    rf_pipeline = ImbPipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.90)),
        ("clf", RandomForestClassifier(
            class_weight="balanced",
            n_jobs=-1,
            random_state=_RANDOM_STATE,
        )),
    ])

    if quick:
        rf_grid = [
            {
                **tfidf_params[0],
                "clf__n_estimators": [100],
                "clf__max_depth": [None, 20],
                "clf__min_samples_leaf": [1],
            }
        ]
    else:
        rf_grid = [
            {
                **tfidf_params[0],
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [None, 20, 40],
                "clf__min_samples_leaf": [1, 2, 5],
                "clf__max_features": ["sqrt", "log2"],
            }
        ]

    # ════════════════════════════════════════════
    # 4. SMOTE variants (ImbPipeline: tfidf → smote → clf)
    # Áp dụng SMOTE chỉ trên training folds (ImbPipeline handles this correctly)
    # ════════════════════════════════════════════
    if quick:
        smote_k_grid = [3]        # ít mẫu minority → k nhỏ
    else:
        smote_k_grid = [3, 5]

    lr_smote_pipeline = ImbPipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.90)),
        ("smote", SMOTE(random_state=_RANDOM_STATE)),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1_500,
            random_state=_RANDOM_STATE,
        )),
    ])
    lr_smote_grid = [{**g, "smote__k_neighbors": smote_k_grid} for g in lr_grid]

    svm_raw_smote = LinearSVC(class_weight="balanced", dual="auto", random_state=_RANDOM_STATE)
    svm_smote_pipeline = ImbPipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.90)),
        ("smote", SMOTE(random_state=_RANDOM_STATE)),
        ("clf", CalibratedClassifierCV(svm_raw_smote, cv=3, method="isotonic")),
    ])
    svm_smote_grid = [{**g, "smote__k_neighbors": smote_k_grid} for g in svm_grid]

    rf_smote_pipeline = ImbPipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.90)),
        ("smote", SMOTE(random_state=_RANDOM_STATE)),
        ("clf", RandomForestClassifier(
            class_weight="balanced",
            n_jobs=-1,
            random_state=_RANDOM_STATE,
        )),
    ])
    rf_smote_grid = [{**g, "smote__k_neighbors": smote_k_grid} for g in rf_grid]

    return {
        "LogisticRegression": (lr_pipeline, lr_grid),
        "LinearSVC_Calibrated": (svm_pipeline, svm_grid),
        "RandomForest": (rf_pipeline, rf_grid),
        "LogisticRegression_SMOTE": (lr_smote_pipeline, lr_smote_grid),
        "LinearSVC_Calibrated_SMOTE": (svm_smote_pipeline, svm_smote_grid),
        "RandomForest_SMOTE": (rf_smote_pipeline, rf_smote_grid),
    }


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data(data_path: str) -> Tuple[pd.Series, pd.Series]:
    """Tải dữ liệu từ CSV.

    Args:
        data_path: Đường dẫn tới file CSV.

    Returns:
        (X, y) — Series văn bản và nhãn.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        KeyError: Nếu thiếu cột.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")

    logger.info("Đang tải dữ liệu từ: %s", data_path)
    df = pd.read_csv(data_path, encoding="utf-8-sig")

    missing = {_TEXT_COL, _LABEL_COL} - set(df.columns)
    if missing:
        raise KeyError(f"Thiếu cột: {missing}. Các cột hiện có: {list(df.columns)}")

    before = len(df)
    df = df.dropna(subset=[_TEXT_COL, _LABEL_COL])
    logger.info("Đã tải %d mẫu (loại bỏ %d NaN).", len(df), before - len(df))
    logger.info("Phân phối nhãn:\n%s", df[_LABEL_COL].value_counts().to_string())
    return df[_TEXT_COL], df[_LABEL_COL]


# ── Run GridSearchCV for one model ────────────────────────────────────────────

def tune_one_model(
    name: str,
    pipeline: ImbPipeline,
    param_grid: List[Dict],
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    cv: int,
    n_jobs: int,
    artifacts_dir: str,
) -> TuningResult:
    """Chạy GridSearchCV cho một mô hình rồi đánh giá trên test set.

    Args:
        name: Tên mô hình (dùng để đặt tên file).
        pipeline: Pipeline scikit-learn / imbalanced-learn.
        param_grid: Danh sách dict tham số để grid search.
        X_train: Văn bản train.
        y_train: Nhãn train.
        X_test: Văn bản test.
        y_test: Nhãn test.
        cv: Số fold cross-validation.
        n_jobs: Số workers song song (-1 = tất cả CPU).
        artifacts_dir: Thư mục lưu model đã tuned.

    Returns:
        TuningResult với đầy đủ metrics.
    """
    logger.info("\n%s\n  TUNING: %s\n%s", "=" * 65, name, "=" * 65)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=_RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=skf,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,          # tự retrain best model trên toàn train set
        return_train_score=False,
        error_score="raise",
    )

    t_start = time.perf_counter()
    grid_search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t_start

    best_params = grid_search.best_params_
    cv_best_f1 = float(grid_search.best_score_)

    logger.info("  Best CV Macro-F1 = %.4f", cv_best_f1)
    logger.info("  Best params: %s", best_params)

    # Evaluate on hold-out test set
    y_pred = grid_search.predict(X_test)
    test_macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    logger.info(
        "  Test Macro-F1 = %.4f | Weighted-F1 = %.4f",
        test_macro_f1, test_weighted_f1,
    )
    logger.info(
        "\nClassification Report — %s:\n%s",
        name,
        classification_report(y_test, y_pred, zero_division=0),
    )

    # Save best estimator
    os.makedirs(artifacts_dir, exist_ok=True)
    save_path = os.path.join(artifacts_dir, f"tuned_{name.lower()}.pkl")
    joblib.dump(grid_search.best_estimator_, save_path)
    logger.info("  Model đã lưu → %s", save_path)

    return TuningResult(
        model_name=name,
        best_params=best_params,
        cv_best_f1=cv_best_f1,
        test_macro_f1=test_macro_f1,
        test_weighted_f1=test_weighted_f1,
        elapsed_seconds=elapsed,
        save_path=save_path,
    )


# ── Print summary table ───────────────────────────────────────────────────────

def print_summary(results: List[TuningResult]) -> None:
    """In bảng tổng kết so sánh các mô hình đã tuned."""
    col_widths = [22, 12, 14, 14, 12]
    headers = ["Model", "CV F1", "Test Macro-F1", "Weighted-F1", "Time (s)"]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    row_fmt = "|" + "|".join(f" {{:<{w}}} " for w in col_widths) + "|"

    print("\n" + "=" * 80)
    print("  KẾT QUẢ HYPERPARAMETER TUNING — Text Baseline Models".center(80))
    print("=" * 80)
    print(sep)
    print(row_fmt.format(*headers))
    print(sep)

    for r in results:
        print(row_fmt.format(
            r.model_name[:22],
            f"{r.cv_best_f1:.4f}",
            f"{r.test_macro_f1:.4f}",
            f"{r.test_weighted_f1:.4f}",
            f"{r.elapsed_seconds:.1f}s",
        ))

    print(sep)
    best = max(results, key=lambda r: r.test_macro_f1)
    print(f"\n  ★ Mô hình tốt nhất: {best.model_name} "
          f"(Test Macro-F1 = {best.test_macro_f1:.4f})")
    print(f"     Best params: {best.best_params}")
    print("=" * 80 + "\n")


# ── Save metrics JSON ─────────────────────────────────────────────────────────

def save_metrics(results: List[TuningResult], metrics_dir: str) -> str:
    """Lưu kết quả tuning ra file JSON.

    Args:
        results: Danh sách TuningResult.
        metrics_dir: Thư mục lưu.

    Returns:
        Đường dẫn tới file JSON đã lưu.
    """
    os.makedirs(metrics_dir, exist_ok=True)
    out_path = os.path.join(metrics_dir, "text_hyperparameter_tuning_results.json")

    payload = {
        "results": [asdict(r) for r in results],
        "best_model": max(results, key=lambda r: r.test_macro_f1).model_name,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Metrics đã lưu → %s", out_path)
    return out_path


# ── Build tuned ensemble from best params ─────────────────────────────────────

def build_and_save_tuned_ensemble(
    results: List[TuningResult],
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    artifacts_dir: str,
) -> Optional[TuningResult]:
    """Tạo VotingClassifier từ 3 mô hình đã tuned và đánh giá trên test set.

    Sử dụng soft voting với trọng số tỉ lệ với Test Macro-F1 của từng mô hình.
    Lưu ensemble đã fit vào artifacts_dir.

    Args:
        results: Danh sách TuningResult (đã có best_params).
        X_train: Văn bản train.
        y_train: Nhãn train.
        X_test: Văn bản test.
        y_test: Nhãn test.
        artifacts_dir: Thư mục lưu.

    Returns:
        TuningResult của ensemble, hoặc None nếu không thể tạo.
    """
    logger.info("\n%s\n  BUILDING TUNED ENSEMBLE\n%s", "=" * 65, "=" * 65)

    try:
        result_map = {r.model_name: r for r in results}
        grids = get_grids(quick=False)

        # Lấy best params cho mỗi mô hình
        def _make_lr(params: Dict) -> ImbPipeline:
            tfidf = TfidfVectorizer(
                sublinear_tf=True, min_df=2, max_df=0.90,
                max_features=params.get("tfidf__max_features", 10_000),
                ngram_range=params.get("tfidf__ngram_range", (1, 2)),
            )
            clf = LogisticRegression(
                class_weight="balanced",
                max_iter=1_500,
                random_state=_RANDOM_STATE,
                C=params.get("clf__C", 1.0),
                solver=params.get("clf__solver", "lbfgs"),
                penalty=params.get("clf__penalty", "l2"),
            )
            return ImbPipeline([("tfidf", tfidf), ("clf", clf)])

        def _make_svm(params: Dict) -> ImbPipeline:
            tfidf = TfidfVectorizer(
                sublinear_tf=True, min_df=2, max_df=0.90,
                max_features=params.get("tfidf__max_features", 10_000),
                ngram_range=params.get("tfidf__ngram_range", (1, 2)),
            )
            svm_raw = LinearSVC(
                class_weight="balanced", dual="auto",
                random_state=_RANDOM_STATE,
                C=params.get("clf__estimator__C", 1.0),
                max_iter=params.get("clf__estimator__max_iter", 2_000),
            )
            clf = CalibratedClassifierCV(svm_raw, cv=3, method="isotonic")
            return ImbPipeline([("tfidf", tfidf), ("clf", clf)])

        def _make_rf(params: Dict) -> ImbPipeline:
            tfidf = TfidfVectorizer(
                sublinear_tf=True, min_df=2, max_df=0.90,
                max_features=params.get("tfidf__max_features", 10_000),
                ngram_range=params.get("tfidf__ngram_range", (1, 2)),
            )
            clf = RandomForestClassifier(
                class_weight="balanced", n_jobs=-1,
                random_state=_RANDOM_STATE,
                n_estimators=params.get("clf__n_estimators", 200),
                max_depth=params.get("clf__max_depth", None),
                min_samples_leaf=params.get("clf__min_samples_leaf", 1),
                max_features=params.get("clf__max_features", "sqrt"),
            )
            return ImbPipeline([("tfidf", tfidf), ("clf", clf)])

        # Lấy kết quả bằng tiền tố để hỗ trợ cả bản SMOTE và không SMOTE
        lr_res = next((r for r in results if r.model_name.startswith("LogisticRegression")), None)
        svm_res = next((r for r in results if r.model_name.startswith("LinearSVC")), None)
        rf_res = next((r for r in results if r.model_name.startswith("RandomForest")), None)

        if not (lr_res and svm_res and rf_res):
            raise KeyError("Không đủ 3 base models (LR, SVM, RF) để tạo ensemble.")

        # Voting weights dựa trên Test Macro-F1
        weights = [lr_res.test_macro_f1, svm_res.test_macro_f1, rf_res.test_macro_f1]
        logger.info("  Ensemble weights [LR, SVM, RF]: %s", [round(w, 4) for w in weights])

        # Hỗ trợ thêm SMOTE vào pipeline nếu trong best_params có smote__k_neighbors
        def _add_smote(pipe, params):
            if "smote__k_neighbors" in params:
                from imblearn.over_sampling import SMOTE
                pipe.steps.insert(1, ("smote", SMOTE(k_neighbors=params["smote__k_neighbors"], random_state=_RANDOM_STATE)))
            return pipe

        lr_pipe = _add_smote(_make_lr(lr_res.best_params), lr_res.best_params)
        svm_pipe = _add_smote(_make_svm(svm_res.best_params), svm_res.best_params)
        rf_pipe = _add_smote(_make_rf(rf_res.best_params), rf_res.best_params)

        # Soft Voting Ensemble — NOTE: cần transform về dense feature trước
        # Vì VotingClassifier không hỗ trợ trực tiếp pipeline với sparse output
        # nên ta tách bước TF-IDF ra và dùng một shared vectorizer

        # Lấy best TF-IDF params từ LR (tốt nhất) để dùng chung
        best_tfidf_params = {
            k.replace("tfidf__", ""): v
            for k, v in lr_res.best_params.items()
            if k.startswith("tfidf__")
        }

        # Rebuild estimators để dùng trong VotingClassifier (không có tfidf step riêng)
        lr_only = LogisticRegression(
            class_weight="balanced", max_iter=1_500, random_state=_RANDOM_STATE,
            C=lr_res.best_params.get("clf__C", 1.0),
            solver=lr_res.best_params.get("clf__solver", "lbfgs"),
            penalty=lr_res.best_params.get("clf__penalty", "l2"),
        )
        svm_raw_only = LinearSVC(
            class_weight="balanced", dual="auto", random_state=_RANDOM_STATE,
            C=svm_res.best_params.get("clf__estimator__C", 1.0),
            max_iter=svm_res.best_params.get("clf__estimator__max_iter", 2_000),
        )
        svm_only = CalibratedClassifierCV(svm_raw_only, cv=3, method="isotonic")
        rf_only = RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=_RANDOM_STATE,
            n_estimators=rf_res.best_params.get("clf__n_estimators", 200),
            max_depth=rf_res.best_params.get("clf__max_depth", None),
            min_samples_leaf=rf_res.best_params.get("clf__min_samples_leaf", 1),
            max_features=rf_res.best_params.get("clf__max_features", "sqrt"),
        )

        voter = VotingClassifier(
            estimators=[("lr", lr_only), ("svm", svm_only), ("rf", rf_only)],
            voting="soft",
            weights=weights,
            n_jobs=-1,
        )

        shared_tfidf = TfidfVectorizer(
            sublinear_tf=True, min_df=2, max_df=0.90,
            **best_tfidf_params,
        )

        ensemble_pipeline = ImbPipeline([
            ("tfidf", shared_tfidf),
            ("ensemble", voter),
        ])

        t_start = time.perf_counter()
        ensemble_pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - t_start

        y_pred = ensemble_pipeline.predict(X_test)
        test_macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        test_weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        logger.info(
            "  Tuned Ensemble — Test Macro-F1 = %.4f | Weighted-F1 = %.4f",
            test_macro_f1, test_weighted_f1,
        )
        logger.info(
            "\nClassification Report — Tuned Ensemble:\n%s",
            classification_report(y_test, y_pred, zero_division=0),
        )

        save_path = os.path.join(artifacts_dir, "tuned_voting_ensemble.pkl")
        joblib.dump(ensemble_pipeline, save_path)
        logger.info("  Ensemble đã lưu → %s", save_path)

        return TuningResult(
            model_name="TunedVotingEnsemble",
            best_params={"weights": [round(w, 4) for w in weights], **best_tfidf_params},
            cv_best_f1=float(np.mean([r.cv_best_f1 for r in results])),
            test_macro_f1=test_macro_f1,
            test_weighted_f1=test_weighted_f1,
            elapsed_seconds=elapsed,
            save_path=save_path,
        )

    except Exception as exc:
        logger.error("Không thể tạo tuned ensemble: %s", exc, exc_info=True)
        return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GridSearchCV Hyperparameter Tuning — Text Baseline Models",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=_DATA_PATH,
        help=f"Đường dẫn CSV dữ liệu (default: {_DATA_PATH})",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Số fold StratifiedKFold (default: 5)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Số workers song song (-1 = tất cả CPU, default: -1)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Chạy grid nhỏ để debug nhanh (ít tham số hơn)",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Bỏ qua bước xây dựng ensemble cuối",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=_ARTIFACTS_DIR,
        help=f"Thư mục lưu model (default: {_ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=_METRICS_DIR,
        help=f"Thư mục lưu metrics JSON (default: {_METRICS_DIR})",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Pipeline chính: tải dữ liệu → GridSearch từng mô hình → Ensemble → Báo cáo."""
    args = parse_args()

    logger.info("=" * 70)
    logger.info("  HYPERPARAMETER TUNING — Text Baseline Models")
    logger.info("  Mode: %s | CV Folds: %d | Workers: %d",
                "QUICK" if args.quick else "FULL", args.cv, args.n_jobs)
    logger.info("=" * 70)

    # 1. Tải dữ liệu
    try:
        X, y = load_data(args.data_path)
    except (FileNotFoundError, KeyError) as exc:
        logger.error("Lỗi tải dữ liệu: %s", exc)
        sys.exit(1)

    # 2. Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=_TEST_SIZE,
        random_state=_RANDOM_STATE,
        stratify=y,
    )
    logger.info("Train: %d mẫu | Test: %d mẫu", len(X_train), len(X_test))

    # 3. Lấy grids
    grids = get_grids(quick=args.quick)

    # 4. Chạy GridSearchCV cho từng mô hình
    results: List[TuningResult] = []
    for model_name, (pipeline, param_grid) in grids.items():
        try:
            result = tune_one_model(
                name=model_name,
                pipeline=pipeline,
                param_grid=param_grid,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv=args.cv,
                n_jobs=args.n_jobs,
                artifacts_dir=args.artifacts_dir,
            )
            results.append(result)
        except Exception as exc:
            logger.error("Lỗi khi tuning %s: %s", model_name, exc, exc_info=True)

    if not results:
        logger.error("Không có mô hình nào được tune thành công. Dừng.")
        sys.exit(1)

    # 5. Xây dựng Tuned Ensemble từ best variant mỗi base model
    # Với mỗi base model (LR, SVM, RF), chọn variant tốt nhất (SMOTE vs không)
    if not args.no_ensemble:
        base_models = ["LogisticRegression", "LinearSVC_Calibrated", "RandomForest"]
        best_per_base: List[TuningResult] = []
        for base in base_models:
            candidates = [
                r for r in results
                if r.model_name == base or r.model_name == f"{base}_SMOTE"
            ]
            if candidates:
                best = max(candidates, key=lambda r: r.test_macro_f1)
                best_per_base.append(best)
                logger.info(
                    "  Best variant cho %s: %s (F1=%.4f)",
                    base, best.model_name, best.test_macro_f1
                )

        if len(best_per_base) == 3:
            ensemble_result = build_and_save_tuned_ensemble(
                results=best_per_base,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                artifacts_dir=args.artifacts_dir,
            )
            if ensemble_result:
                results.append(ensemble_result)

    # 6. In bảng tổng kết
    print_summary(results)

    # 7. Lưu metrics
    metrics_path = save_metrics(results, args.metrics_dir)
    logger.info("Hoàn thành! Metrics đã lưu tại: %s", metrics_path)


if __name__ == "__main__":
    main()
