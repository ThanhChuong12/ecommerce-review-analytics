"""
explain_spam.py
===============
Load pre-trained Isolation Forest model, run inference on CSV, and output
results with additional explanation columns (spam_source, spam_rules).

Usage:
  python scripts/explain_spam.py --data-path <path> [--model-path <path>] [--output-csv <path>]
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_engine.text_processing.spam_filter import detect_spam
from ai_engine.text_processing.spam_model import (
    RULE_FLAG_COLS,
    SpamHybridModel,
    build_feature_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = ROOT / "artifacts" / "spam" / "tuned_spam_iforest.pkl"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "spam_explained.csv"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _spam_source(rule: bool, iforest: bool) -> str:
    if rule and iforest:
        return "both"
    if rule:
        return "rule"
    if iforest:
        return "iforest"
    return ""


def build_spam_source_cols(
    df_flagged: pd.DataFrame,
    iforest_pred,          # np.ndarray -1/1
) -> pd.DataFrame:
    """Return DataFrame with spam_source and spam_rules columns."""
    flag_details = df_flagged.attrs.get("flag_details")
    rule_arr = df_flagged["is_spam"].values.astype(bool)
    iforest_arr = (iforest_pred == -1)

    spam_source = [_spam_source(r, i) for r, i in zip(rule_arr, iforest_arr)]

    if flag_details is not None:
        triggered_rules = []
        for _, row in flag_details.iterrows():
            rules = [col for col in RULE_FLAG_COLS if col in row.index and row[col]]
            triggered_rules.append(", ".join(rules) if rules else "")
    else:
        triggered_rules = [""] * len(df_flagged)

    return pd.DataFrame(
        {"spam_source": spam_source, "spam_rules": triggered_rules},
        index=df_flagged.index,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Spam inference with explanation columns (spam_source, spam_rules)."
    )
    parser.add_argument(
        "--data-path", required=True,
        help="Input CSV with text and rating columns",
    )
    parser.add_argument(
        "--model-path", default=str(DEFAULT_MODEL),
        help=f"Path to model pkl (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-csv", default=str(DEFAULT_OUTPUT),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--text-col", default="text",
        help="Text column name (default: text)",
    )
    parser.add_argument(
        "--rating-col", default="rating",
        help="Rating column name (default: rating)",
    )
    parser.add_argument(
        "--dup-threshold", type=float, default=0.85,
        help="Cosine threshold for duplicate seeding (default: 0.85)",
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Reading data from: %s", args.data_path)
    df = pd.read_csv(args.data_path)

    for col in [args.text_col, args.rating_col]:
        if col not in df.columns:
            raise ValueError(f"Không tìm thấy cột '{col}'. Có: {list(df.columns)}")

    if args.text_col != "text" or args.rating_col != "rating":
        df = df.rename(columns={args.text_col: "text", args.rating_col: "rating"})

    df["text"] = df["text"].fillna("").astype(str)
    logger.info("Total samples: %d", len(df))

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = args.model_path
    logger.info("Loading model from: %s", model_path)
    model: SpamHybridModel = SpamHybridModel.load(model_path)

    # ── Step 1: Rule-based ────────────────────────────────────────────────────
    logger.info("Running rule-based spam detection...")
    df_flagged = detect_spam(df, dup_threshold=args.dup_threshold)

    # ── Step 2: Feature matrix ────────────────────────────────────────────────
    logger.info("Building feature matrix...")
    texts = df_flagged["text"].tolist()
    ratings = df_flagged["rating"].tolist()
    X = build_feature_matrix(df_flagged, texts, ratings)

    # ── Step 3: IForest predict (no retraining) ───────────────────────────────
    logger.info("Predicting anomalies...")
    iforest_pred = model.predict_anomaly(X)
    anomaly_scores = model.anomaly_score(X)

    rule_spam = df_flagged["is_spam"].values.astype(int)
    import numpy as np
    iforest_spam = (iforest_pred == -1).astype(int)
    final_spam = np.clip(rule_spam + iforest_spam, 0, 1)

    df_flagged["iforest_anomaly"] = iforest_spam
    df_flagged["anomaly_score"] = anomaly_scores
    df_flagged["final_spam"] = final_spam

    # ── Step 4: Build source + rule columns ───────────────────────────────────
    extra_cols = build_spam_source_cols(df_flagged, iforest_pred)

    # ── Step 5: Quick report ──────────────────────────────────────────────────
    total = len(df_flagged)
    n_rule = int(rule_spam.sum())
    n_iforest = int(iforest_spam.sum())
    n_final = int(final_spam.sum())
    n_both = int((extra_cols["spam_source"] == "both").sum())

    print("\n" + "=" * 60)
    print("  SPAM EXPLAIN REPORT")
    print("=" * 60)
    print(f"  Total reviews     : {total:>8,}")
    print(f"  Rule-based spam   : {n_rule:>8,}  ({n_rule/total*100:.1f}%)")
    print(f"  IForest anomaly   : {n_iforest:>8,}  ({n_iforest/total*100:.1f}%)")
    print(f"  Both              : {n_both:>8,}  ({n_both/total*100:.1f}%)")
    print(f"  Final spam (union): {n_final:>8,}  ({n_final/total*100:.1f}%)")
    print(f"  Clean             : {total-n_final:>8,}  ({(total-n_final)/total*100:.1f}%)")
    print("=" * 60 + "\n")

    # Top rules triggered
    flag_details = df_flagged.attrs.get("flag_details")
    if flag_details is not None:
        print(f"  {'Rule':<30} {'Count':>8}  {'%':>6}")
        print(f"  {'-'*30} {'-'*8}  {'-'*6}")
        for col in RULE_FLAG_COLS:
            if col in flag_details.columns:
                cnt = int(flag_details[col].sum())
                if cnt > 0:
                    print(f"  {col:<30} {cnt:>8,}  {cnt/total*100:>5.1f}%")
        print()

    # ── Step 6: Save CSV ──────────────────────────────────────────────────────
    flag_details = df_flagged.attrs.get("flag_details")
    out_df = df_flagged[["text", "rating", "is_spam",
                          "iforest_anomaly", "anomaly_score", "final_spam"]].copy()
    if flag_details is not None:
        out_df = pd.concat([out_df, flag_details], axis=1)
    out_df = pd.concat([out_df, extra_cols], axis=1)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Saved CSV to %s", out_path)
    logger.info(
        "spam_source breakdown: rule=%d, iforest=%d, both=%d",
        int((extra_cols["spam_source"] == "rule").sum()),
        int((extra_cols["spam_source"] == "iforest").sum()),
        n_both,
    )


if __name__ == "__main__":
    main()
