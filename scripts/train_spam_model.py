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

# ── Rule flag columns (21 flags from spam_filter) ───────────────────────────
RULE_FLAG_COLS: List[str] = [
    "ai_template", "template_repeat", "mostly_template",
    "xu_farming",
    "too_short", "too_long", "emoji_only",
    "keyboard_spam", "word_repetition",
    "too_many_special", "too_many_uppercase", "only_digits_or_punct",
    "random_keyboard", "non_informative_short",
    "short_generic", "off_topic", "competitor_promo",
    "external_link", "contact_info",
    "rating_mismatch", "duplicate_seeding",
]


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def extract_structural_features(texts: List[str], ratings: List) -> np.ndarray:
    """Trich xuat cac dac trung cau truc tu van ban.

    Cac dac trung nay doc lap voi rule-based, giup Isolation Forest bat
    cac review bat thuong ma rule don le chua cover.

    Features:
        0  word_count        - So luong tu
        1  char_count        - So luong ky tu (khong ke khoang trang)
        2  emoji_ratio       - Ty le emoji / tong ky tu
        3  special_char_ratio- Ty le ky tu dac biet
        4  uppercase_ratio   - Ty le chu hoa
        5  digit_ratio       - Ty le chu so
        6  type_token_ratio  - Do da dang tu vung (TTR)
        7  avg_word_len      - Do dai tu trung binh
        8  rating_norm       - Rating chuan hoa ve [0, 1]
    """
    features = []
    for text, rating in zip(texts, ratings):
        text_str = str(text) if text else ""
        words = text_str.split()
        avg_word_len = (
            np.mean([len(w) for w in words]) if words else 0.0
        )
        try:
            rating_norm = (float(rating) - 1.0) / 4.0  # scale 1-5 -> 0-1
        except (ValueError, TypeError):
            rating_norm = 0.5

        features.append([
            count_words(text_str),
            count_chars(text_str),
            get_emoji_ratio(text_str),
            get_special_char_ratio(text_str),
            get_uppercase_ratio(text_str),
            get_digit_ratio(text_str),
            get_type_token_ratio(text_str),
            avg_word_len,
            rating_norm,
        ])
    return np.array(features, dtype=np.float32)


def build_feature_matrix(
    df_flagged: pd.DataFrame,
    texts: List[str],
    ratings: List,
) -> np.ndarray:
    """Ket hop rule flags va structural features thanh 1 feature matrix.

    Args:
        df_flagged: DataFrame da chay qua detect_spam() (co flag_details trong .attrs).
        texts: Danh sach van ban goc.
        ratings: Danh sach diem so.

    Returns:
        np.ndarray shape (n_samples, n_rule_flags + n_structural_features)
    """
    flag_details = df_flagged.attrs.get("flag_details")
    if flag_details is None:
        raise RuntimeError(
            "flag_details not found in df.attrs. "
            "Make sure detect_spam() was called before build_feature_matrix()."
        )

    # Rule-based binary flags (21 cols) — gia tri 0/1
    rule_feats = flag_details[RULE_FLAG_COLS].values.astype(np.float32)

    # Rule score aggregate (tong so rule bi vi pham)
    rule_score = rule_feats.sum(axis=1, keepdims=True)

    # Structural features (9 cols)
    struct_feats = extract_structural_features(texts, ratings)

    X = np.hstack([rule_feats, rule_score, struct_feats])
    logger.info("Feature matrix shape: %s", X.shape)
    return X


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════

class SpamHybridModel:
    """Mo hinh phat hien spam ket hop Rule-based va Isolation Forest.

    Attributes:
        contamination: Ti le mau bat thuong uoc tinh trong tap du lieu.
        n_estimators:  So luong cay trong Isolation Forest.
        random_state:  Hat giong ngau nhien.
        scaler:        StandardScaler cho structural features.
        iforest:       Isolation Forest da huan luyen.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 200,
        max_samples: float | str = "auto",
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.iforest: IsolationForest | None = None

    def fit(self, X: np.ndarray) -> "SpamHybridModel":
        """Huan luyen Isolation Forest tren feature matrix.

        Chuc nang nay chi huan luyen phan anomaly detection. Rule-based
        flags duoc dua truc tiep vao feature matrix, nen model hoc duoc
        ca ket hop cua quy tac va dac trung cau truc.

        Args:
            X: Feature matrix tu build_feature_matrix().

        Returns:
            self, de chaining.
        """
        logger.info(
            "Training Isolation Forest (n_estimators=%d, contamination=%.2f)...",
            self.n_estimators, self.contamination,
        )
        X_scaled = self.scaler.fit_transform(X)
        self.iforest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.iforest.fit(X_scaled)
        logger.info("Isolation Forest training complete.")
        return self

    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Du doan bat thuong.

        Returns:
            np.ndarray: 1 = binh thuong (inlier), -1 = bat thuong (outlier/spam).
        """
        if self.iforest is None:
            raise RuntimeError("Model chua duoc huan luyen. Goi .fit() truoc.")
        X_scaled = self.scaler.transform(X)
        return self.iforest.predict(X_scaled)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Tra ve anomaly score (cang am cang bat thuong).

        Returns:
            np.ndarray: Mang score, shape (n_samples,).
        """
        if self.iforest is None:
            raise RuntimeError("Model chua duoc huan luyen. Goi .fit() truoc.")
        X_scaled = self.scaler.transform(X)
        return self.iforest.score_samples(X_scaled)

    def predict_final_spam(
        self,
        X: np.ndarray,
        rule_is_spam: np.ndarray,
    ) -> np.ndarray:
        """Final spam label = rule_is_spam OR iforest_anomaly.

        Args:
            X: Feature matrix.
            rule_is_spam: Mang 0/1 tu detect_spam() column 'is_spam'.

        Returns:
            np.ndarray: Mang 0/1, 1 = spam (theo bat ky tieu chi nao).
        """
        iforest_pred = self.predict_anomaly(X)
        iforest_spam = (iforest_pred == -1).astype(int)
        return np.clip(rule_is_spam + iforest_spam, 0, 1)

    def save(self, path: str) -> None:
        """Luu model xuong disk bang joblib."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        logger.info("Spam model saved -> %s", save_path)

    @classmethod
    def load(cls, path: str) -> "SpamHybridModel":
        """Tai model tu disk."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        logger.info("Loading spam model <- %s", path)
        return joblib.load(path)


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
        "--data-path", required=True,
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
