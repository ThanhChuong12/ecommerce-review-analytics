"""Module for the text baseline ensemble model.

Provides a unified interface for a soft-voting ensemble model combining
linear and ensemble estimators (Logistic Regression, Calibrated LinearSVC,
and Random Forest) with TF-IDF vectorization.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

# Model hyperparameter constants
_TFIDF_MAX_FEATURES = 15_000
_TFIDF_NGRAM_RANGE = (1, 2)
_RF_N_ESTIMATORS = 200
_CV_FOLDS = 5
_DEFAULT_RANDOM_STATE = 42


class EstimatorFactory:
    """Factory for building base classifiers used in the ensemble."""

    def __init__(self, random_state: int = _DEFAULT_RANDOM_STATE) -> None:
        self.random_state = random_state

    def create_logistic_regression(self) -> LogisticRegression:
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            solver="lbfgs",
            random_state=self.random_state,
        )

    def create_calibrated_svm(self) -> CalibratedClassifierCV:
        svm_raw = LinearSVC(
            class_weight="balanced",
            dual="auto",
            max_iter=2_000,
            random_state=self.random_state,
        )
        return CalibratedClassifierCV(svm_raw, cv=3, method="isotonic")

    def create_random_forest(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=_RF_N_ESTIMATORS,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state,
        )

    def create_all(self) -> Tuple[LogisticRegression, CalibratedClassifierCV, RandomForestClassifier]:
        return (
            self.create_logistic_regression(),
            self.create_calibrated_svm(),
            self.create_random_forest(),
        )


class CrossValWeightStrategy:
    """Strategy to calculate ensemble weights based on cross-validation F1-scores."""

    def __init__(self, estimator_factory: EstimatorFactory, cv: int = _CV_FOLDS) -> None:
        self.factory = estimator_factory
        self.cv = cv

    def calculate_weights(self, X: pd.Series, y: pd.Series) -> List[float]:
        """Runs cross-validation for each estimator and returns macro F1-scores."""
        if X.empty or y.empty:
            raise ValueError("Training data X and y must not be empty.")

        logger.info("Calculating auto-weights via %d-fold cross-validation...", self.cv)

        vectorizer = TfidfVectorizer(
            max_features=_TFIDF_MAX_FEATURES,
            ngram_range=_TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
            min_df=3,
            max_df=0.85,
        )
        X_tfidf = vectorizer.fit_transform(X)

        estimators = [
            ("LR", self.factory.create_logistic_regression()),
            ("SVM", self.factory.create_calibrated_svm()),
            ("RF", self.factory.create_random_forest()),
        ]

        scores = []
        for name, model in estimators:
            cv_scores = cross_val_score(
                model,
                X_tfidf,
                y,
                cv=self.cv,
                scoring="f1_macro",
                n_jobs=-1,
            )
            mean_score = float(cv_scores.mean())
            logger.info("  %s -> mean macro-F1: %.4f (std: %.4f)", name, mean_score, cv_scores.std())
            scores.append(mean_score)

        return [round(score, 4) for score in scores]


class TextEnsembleModel:
    """Text classification ensemble model wrapping a scikit-learn pipeline.

    Combines TF-IDF feature extraction, optional SMOTE over-sampling,
    and a soft-voting ensemble of Logistic Regression, Calibrated LinearSVC,
    and Random Forest.
    """

    def __init__(
        self,
        use_smote: bool = False,
        weights: Optional[List[float]] = None,
        random_state: int = _DEFAULT_RANDOM_STATE,
    ) -> None:
        self.use_smote = use_smote
        self.weights = weights
        self.random_state = random_state
        self.pipeline: Optional[ImbPipeline] = None

    def _build_base_estimators(
        self,
    ) -> Tuple[LogisticRegression, CalibratedClassifierCV, RandomForestClassifier]:
        """Constructs base estimators using EstimatorFactory."""
        factory = EstimatorFactory(random_state=self.random_state)
        return factory.create_all()

    def _build_pipeline(self, weights: Optional[List[float]]) -> ImbPipeline:
        """Assembles the full imbalanced-learn pipeline."""
        logger.info("Initializing pipeline with SMOTE=%s and weights=%s", self.use_smote, weights)

        vectorizer = TfidfVectorizer(
            max_features=_TFIDF_MAX_FEATURES,
            ngram_range=_TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
            min_df=3,
            max_df=0.85,
        )

        lr, svm, rf = self._build_base_estimators()
        ensemble = VotingClassifier(
            estimators=[("lr", lr), ("svm", svm), ("rf", rf)],
            voting="soft",
            weights=weights,
            n_jobs=-1,
        )

        steps = [("tfidf", vectorizer)]
        if self.use_smote:
            steps.append(("smote", SMOTE(random_state=self.random_state)))
        steps.append(("ensemble", ensemble))

        return ImbPipeline(steps=steps)

    def compute_auto_weights(self, X: pd.Series, y: pd.Series, cv: int = _CV_FOLDS) -> List[float]:
        """Calculates default F1-proportional weights for the ensemble."""
        factory = EstimatorFactory(random_state=self.random_state)
        strategy = CrossValWeightStrategy(factory, cv=cv)
        return strategy.calculate_weights(X, y)

    def fit(self, X: pd.Series, y: pd.Series) -> TextEnsembleModel:
        """Fits the ensemble pipeline on training text data."""
        if self.weights is None:
            self.weights = self.compute_auto_weights(X, y)

        self.pipeline = self._build_pipeline(weights=self.weights)
        logger.info("Fitting ensemble model on %d samples...", len(X))
        self.pipeline.fit(X, y)
        logger.info("Ensemble training completed successfully.")
        return self

    def predict(self, X: pd.Series) -> np.ndarray:
        """Predicts class labels for input text."""
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before calling predict.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """Predicts class probabilities for input text."""
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        return self.pipeline.predict_proba(X)

    def save(self, filepath: str) -> None:
        """Serializes the model to disk using joblib."""
        if self.pipeline is None:
            raise RuntimeError("Untrained model cannot be saved.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Saved model to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> TextEnsembleModel:
        """Loads a model instance from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        logger.info("Loaded model from %s", filepath)
        return joblib.load(filepath)