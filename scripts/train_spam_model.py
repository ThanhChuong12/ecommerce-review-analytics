"""
train_spam_model.py
===================
Huan luyen mo hinh phat hien Spam / Seeding ket hop:
  - Rule-based Score  : Dung spam_filter.py de tinh so luong rule bi vi pham
  - Isolation Forest  : Phat hien bat thuong dua tren cac dac trung cau truc van ban

Chien luoc ket hop (Hybrid):
  1. Rule-based flags tu spam_filter.detect_spam() -> feature vector (21 flags)
  2. Structural features tu chinh van ban (do dai, ty le emoji, TTR, ...)
  3. Isolation Forest huan luyen tren ca 2 nhom feature de bat cac review di thuong
     ma rule-based co the bo sot (e.g. seeding tinh vi, khong vi pham ro rang)
  4. Final label: spam = (rule_based_is_spam == 1) OR (iforest_pred == -1)

Usage (chay tu thu muc goc):
    py scripts/train_spam_model.py --data-path data/processed/reviews.csv
    py scripts/train_spam_model.py --data-path data/processed/reviews.csv \\
        --contamination 0.15 --save-path ai_engine/models/spam_iforest.pkl
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Removed global stdout/stderr reconfiguration to prevent import issues
# ── Add project root to sys.path ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_engine.text_processing.spam_filter import (
    detect_spam,
    get_emoji_ratio,
    get_special_char_ratio,
    get_type_token_ratio,
    get_uppercase_ratio,
    get_digit_ratio,
    count_words,
    count_chars,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from ai_engine.text_processing.spam_model import (
    extract_structural_features,
    build_feature_matrix,
    SpamHybridModel,
    RULE_FLAG_COLS
)


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — EVALUATION REPORT
# ════════════════════════════════════════════════════════════════════════════

def print_spam_report(
    df_flagged: pd.DataFrame,
    final_spam: np.ndarray,
    iforest_pred: np.ndarray,
) -> None:
    """In bao cao ket qua phat hien spam."""
    SEP = "=" * 65
    total = len(df_flagged)
    rule_spam = int(df_flagged["is_spam"].sum())
    iforest_spam = int((iforest_pred == -1).sum())
    final_count = int(final_spam.sum())

    print(f"\n{SEP}")
    print("  SPAM DETECTION REPORT")
    print(SEP)
    print(f"  Total reviews        : {total:>8,}")
    print(f"  Rule-based spam      : {rule_spam:>8,}  ({rule_spam/total*100:.1f}%)")
    print(f"  IForest anomalies    : {iforest_spam:>8,}  ({iforest_spam/total*100:.1f}%)")
    print(f"  Final spam (union)   : {final_count:>8,}  ({final_count/total*100:.1f}%)")
    print(f"  Clean reviews        : {total-final_count:>8,}  ({(total-final_count)/total*100:.1f}%)")

    # Breakdown per rule
    flag_details = df_flagged.attrs.get("flag_details")
    if flag_details is not None:
        print(f"\n  {'Rule':<30} {'Count':>8}  {'%':>6}")
        print(f"  {'-'*30} {'-'*8}  {'-'*6}")
        for col in RULE_FLAG_COLS:
            if col in flag_details.columns:
                cnt = int(flag_details[col].sum())
                if cnt > 0:
                    print(f"  {col:<30} {cnt:>8,}  {cnt/total*100:>5.1f}%")

    print(f"\n{SEP}\n")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Force UTF-8 stdout/stderr ──────────────────────────────────────────
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Train spam/seeding detection model (Rule-based + Isolation Forest)"
    )
    parser.add_argument(
        "--data-path", default="data/processed/spam_train.csv",
        help="CSV file with columns: text, rating (and optionally is_spam for reference)",
    )
    parser.add_argument(
        "--text-col", default="text",
        help="Column name for review text (default: text)",
    )
    parser.add_argument(
        "--rating-col", default="rating",
        help="Column name for star rating (default: rating)",
    )
    parser.add_argument(
        "--contamination", type=float, default=0.1,
        help="Estimated fraction of spam in dataset for IForest (default: 0.1)",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200,
        help="Number of trees in Isolation Forest (default: 200)",
    )
    parser.add_argument(
        "--max-samples", default="auto",
        help="Number of samples to draw to train each base estimator (float or 'auto'). (default: auto)",
    )
    parser.add_argument(
        "--save-path", default="ai_engine/models/spam_iforest.pkl",
        help="Path to save the trained model (default: ai_engine/models/spam_iforest.pkl)",
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="If provided, save the annotated DataFrame with spam labels to this CSV path",
    )
    parser.add_argument(
        "--dup-threshold", type=float, default=0.85,
        help="Cosine similarity threshold for duplicate seeding detection (default: 0.85)",
    )
    args = parser.parse_args()

    # Chuyển đổi max_samples thành số float nếu là chuỗi số
    max_samples_val = args.max_samples
    if max_samples_val != "auto":
        try:
            max_samples_val = float(max_samples_val)
        except ValueError:
            pass

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading data from: %s", args.data_path)
    df = pd.read_csv(args.data_path)

    for col in [args.text_col, args.rating_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    # Rename for spam_filter compatibility if needed
    if args.text_col != "text" or args.rating_col != "rating":
        df = df.rename(columns={args.text_col: "text", args.rating_col: "rating"})

    df["text"] = df["text"].fillna("").astype(str)
    logger.info("Total samples: %d", len(df))

    # ── Step 1: Rule-based detection ───────────────────────────────────────
    logger.info("Running rule-based spam detection...")
    df_flagged = detect_spam(df, dup_threshold=args.dup_threshold)

    # ── Step 2: Feature matrix ─────────────────────────────────────────────
    logger.info("Building feature matrix...")
    texts = df_flagged["text"].tolist()
    ratings = df_flagged["rating"].tolist()
    X = build_feature_matrix(df_flagged, texts, ratings)

    # ── Step 3: Train Isolation Forest ─────────────────────────────────────
    model = SpamHybridModel(
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        max_samples=max_samples_val,
    )
    model.fit(X)

    # ── Step 4: Final prediction ───────────────────────────────────────────
    iforest_pred = model.predict_anomaly(X)
    rule_is_spam = df_flagged["is_spam"].values.astype(int)
    final_spam = model.predict_final_spam(X, rule_is_spam)

    df_flagged["iforest_anomaly"] = (iforest_pred == -1).astype(int)
    df_flagged["anomaly_score"] = model.anomaly_score(X)
    df_flagged["final_spam"] = final_spam

    # ── Step 5: Report ─────────────────────────────────────────────────────
    print_spam_report(df_flagged, final_spam, iforest_pred)

    # ── Step 6: Save model ─────────────────────────────────────────────────
    model.save(args.save_path)

    # ── Step 7: Optional CSV output ────────────────────────────────────────
    if args.output_csv:
        # Drop attrs (not serializable to CSV) before saving
        out_df = df_flagged[["text", "rating", "is_spam",
                              "iforest_anomaly", "anomaly_score", "final_spam"]].copy()
        flag_details = df_flagged.attrs.get("flag_details")
        if flag_details is not None:
            out_df = pd.concat([out_df, flag_details], axis=1)
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info("Annotated CSV saved -> %s", out_path)


if __name__ == "__main__":
    main()
