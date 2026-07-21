"""Hybrid spam detection model combining rule-based flags with Isolation Forest."""

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
    get_digit_ratio,
    get_emoji_ratio,
    get_special_char_ratio,
    get_type_token_ratio,
    get_uppercase_ratio,
)

logger = logging.getLogger(__name__)

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
    """Extract statistical and structural features from raw texts."""
    features = []
    for text, _ in zip(texts, ratings):
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
    """Combine rule-based flags and structural metrics into a single feature matrix."""
    flag_details = df_flagged.attrs.get("flag_details")
    if flag_details is None:
        raise RuntimeError(
            "flag_details missing in DataFrame attributes. "
            "Execute detect_spam() prior to build_feature_matrix()."
        )

    rule_feats = flag_details[RULE_FLAG_COLS].values.astype(np.float32)
    rule_score = rule_feats.sum(axis=1, keepdims=True)
    struct_feats = extract_structural_features(texts, ratings)

    X = np.hstack([rule_feats, rule_score, struct_feats])
    logger.info("Built feature matrix with shape: %s", X.shape)
    return X


class SpamHybridModel:
    """Isolation Forest anomaly detection model for identifying spam reviews."""

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
        """Fit scaler and Isolation Forest model on feature matrix."""
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
        logger.info("Isolation Forest training completed.")
        return self

    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly status (-1 for anomaly/spam, 1 for normal)."""
        if self.iforest is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.iforest.predict(X_scaled)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores for feature matrix."""
        if self.iforest is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.iforest.score_samples(X_scaled)

    def predict_final_spam(
        self,
        X: np.ndarray,
        rule_is_spam: np.ndarray,
    ) -> np.ndarray:
        """Predict binary spam status (1 for spam, 0 for legitimate)."""
        iforest_pred = self.predict_anomaly(X)
        return (iforest_pred == -1).astype(int)

    def save(self, path: str) -> None:
        """Persist model instance to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        logger.info("Saved spam model to '%s'", save_path)

    @classmethod
    def load(cls, path: str) -> "SpamHybridModel":
        """Load persisted model instance from disk."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        logger.info("Loading spam model from '%s'", path)
        return joblib.load(path)
