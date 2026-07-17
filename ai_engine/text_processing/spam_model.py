"""
spam_model.py
=============
Hybrid Spam Detection Model combining rule-based flags with Isolation Forest.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ai_engine.text_processing.spam_filter import (
    count_chars,
    count_words,
    get_digit_ratio,
    get_emoji_ratio,
    get_special_char_ratio,
    get_type_token_ratio,
    get_uppercase_ratio,
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

def extract_structural_features(texts: List[str], ratings: List) -> np.ndarray:
    """Trich xuat cac dac trung cau truc tu van ban."""
    features = []
    for text, rating in zip(texts, ratings):
        text_str = str(text) if text else ""

        features.append([
            get_emoji_ratio(text_str),
            get_special_char_ratio(text_str),
            get_uppercase_ratio(text_str),
            get_type_token_ratio(text_str),
            get_digit_ratio(text_str),
        ])
    return np.array(features, dtype=np.float32)

def build_feature_matrix(
    df_flagged: pd.DataFrame,
    texts: List[str],
    ratings: List,
) -> np.ndarray:
    """Ket hop rule flags va structural features thanh 1 feature matrix."""
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

    # Structural features (5 cols)
    struct_feats = extract_structural_features(texts, ratings)

    X = np.hstack([rule_feats, rule_score, struct_feats])
    logger.info("Feature matrix shape: %s", X.shape)
    return X


class SpamHybridModel:
    """Mo hinh phat hien spam ket hop Rule-based va Isolation Forest."""

    def __init__(
        self,
        contamination: float = 0.10,
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
        if self.iforest is None:
            raise RuntimeError("Model chua duoc huan luyen. Goi .fit() truoc.")
        X_scaled = self.scaler.transform(X)
        return self.iforest.predict(X_scaled)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if self.iforest is None:
            raise RuntimeError("Model chua duoc huan luyen. Goi .fit() truoc.")
        X_scaled = self.scaler.transform(X)
        return self.iforest.score_samples(X_scaled)

    def predict_final_spam(
        self,
        X: np.ndarray,
        rule_is_spam: np.ndarray,
    ) -> np.ndarray:
        iforest_pred = self.predict_anomaly(X)
        iforest_spam = (iforest_pred == -1).astype(int)
        # User requested to ONLY use the model (Isolation Forest)
        return iforest_spam

    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        logger.info("Spam model saved -> %s", save_path)

    @classmethod
    def load(cls, path: str) -> "SpamHybridModel":
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        logger.info("Loading spam model <- %s", path)
        return joblib.load(path)
