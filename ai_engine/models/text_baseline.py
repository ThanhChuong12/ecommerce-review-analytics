"""Weighted Soft-Voting Ensemble model for sentiment text classification.

This module provides the :class:`TextEnsembleModel`, a production-quality
pipeline that combines Logistic Regression, a Calibrated LinearSVC, and a
Random Forest under a **Soft-Voting Ensemble** strategy.  Optionally, SMOTE
over-sampling can be injected between the TF-IDF step and the ensemble to
handle class imbalance.

Automatic Weight Computation
-----------------------------
When ``weights`` is ``None`` (default), :meth:`TextEnsembleModel.fit` will
call :meth:`TextEnsembleModel.compute_auto_weights` before training the full
ensemble.  That helper runs :func:`~sklearn.model_selection.cross_val_score`
on each base estimator independently over the TF-IDF representations and uses
their **macro-F1 scores** as ensemble weights — ensuring that stronger
models have proportionally louder votes.

Usage Example
-------------
>>> from ai_engine.models.text_baseline import TextEnsembleModel
>>> model = TextEnsembleModel(use_smote=False)   # weights computed automatically
>>> model.fit(X_train, y_train)
>>> preds = model.predict(X_test)
>>> model.save("artifacts/models/ensemble_no_smote.pkl")
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
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
from sklearn.model_selection import cross_val_score
# pyrefly: ignore [missing-import]
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TFIDF_MAX_FEATURES: int = 15_000
_TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)
_RF_N_ESTIMATORS: int = 200
_CV_FOLDS: int = 5
_DEFAULT_RANDOM_STATE: int = 42


class TextEnsembleModel:
    """Weighted Soft-Voting Ensemble for text sentiment classification.

    The pipeline is composed of three stages:

    1. **Feature extraction** — TF-IDF with unigram + bigram, sublinear TF
       scaling, and vocabulary capped at :data:`_TFIDF_MAX_FEATURES`.
    2. **Optional over-sampling** — SMOTE is inserted *after* TF-IDF when
       ``use_smote=True``, operating on the sparse feature matrix.
    3. **Soft-Voting Ensemble** — Three base estimators vote via averaged
       predicted probabilities:

       - ``lr``: :class:`~sklearn.linear_model.LogisticRegression`
       - ``svm``: :class:`~sklearn.svm.LinearSVC` wrapped in
         :class:`~sklearn.calibration.CalibratedClassifierCV` (required for
         ``predict_proba`` support)
       - ``rf``: :class:`~sklearn.ensemble.RandomForestClassifier`

    Attributes:
        use_smote (bool): Whether SMOTE over-sampling is active.
        weights (Optional[List[float]]): Ensemble voting weights for
            ``[lr, svm, rf]``.  ``None`` triggers automatic computation.
        random_state (int): Global random seed for full reproducibility.
        pipeline (ImbPipeline): The assembled imbalanced-learn pipeline.
            Available after :meth:`fit` is called.
    """

    def __init__(
        self,
        use_smote: bool = False,
        weights: Optional[List[float]] = None,
        random_state: int = _DEFAULT_RANDOM_STATE,
    ) -> None:
        """Initialises the TextEnsembleModel.

        Args:
            use_smote (bool): If ``True``, SMOTE is inserted after TF-IDF.
                Defaults to ``False`` (cost-sensitive ``class_weight='balanced'``
                on all estimators is the primary imbalance strategy).
            weights (Optional[List[float]]): Explicit voting weights for
                ``[LR, SVM, RF]``.  When ``None``, weights are derived
                automatically from per-estimator macro-F1 cross-validation
                scores inside :meth:`fit`.
            random_state (int): Random seed propagated to all stochastic
                components. Defaults to 42.
        """
        self.use_smote = use_smote
        self.weights = weights
        self.random_state = random_state
        # Pipeline is built lazily in fit() so weights can be updated first.
        self.pipeline: Optional[ImbPipeline] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_base_estimators(
        self,
    ) -> tuple[LogisticRegression, CalibratedClassifierCV, RandomForestClassifier]:
        """Constructs the three base estimators used by the ensemble.

        All estimators use ``class_weight='balanced'`` to handle class
        imbalance via cost-sensitive learning.

        Returns:
            Tuple of (lr, svm_calibrated, rf) estimator instances.
        """
        lr = LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            solver="lbfgs",
            random_state=self.random_state,
        )

        # LinearSVC is much faster than SVC(kernel='linear') on sparse matrices
        # but lacks predict_proba.  CalibratedClassifierCV adds calibrated
        # probability outputs via isotonic / sigmoid regression.
        svm_raw = LinearSVC(
            class_weight="balanced",
            dual="auto",
            max_iter=2_000,
            random_state=self.random_state,
        )
        svm_calibrated = CalibratedClassifierCV(svm_raw, cv=3, method="isotonic")

        rf = RandomForestClassifier(
            n_estimators=_RF_N_ESTIMATORS,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state,
        )

        return lr, svm_calibrated, rf

    def _build_pipeline(self, weights: Optional[List[float]]) -> ImbPipeline:
        """Assembles the full imblearn pipeline.

        Args:
            weights (Optional[List[float]]): Voting weights to assign to
                ``[lr, svm, rf]`` inside :class:`~sklearn.ensemble.VotingClassifier`.

        Returns:
            ImbPipeline: The fully configured imbalanced-learn pipeline.
        """
        logger.info(
            "Building Ensemble Pipeline — SMOTE=%s | Weights=%s",
            self.use_smote,
            weights,
        )

        # Stage 1 — Feature extraction
        tfidf = TfidfVectorizer(
            max_features=_TFIDF_MAX_FEATURES,
            ngram_range=_TFIDF_NGRAM_RANGE,
            sublinear_tf=True,    # log(1 + tf) — dampens very frequent terms
            min_df=3,
            max_df=0.85,
        )

        # Stage 3 — Ensemble (weights applied here)
        lr, svm_calibrated, rf = self._build_base_estimators()

        ensemble = VotingClassifier(
            estimators=[("lr", lr), ("svm", svm_calibrated), ("rf", rf)],
            voting="soft",
            weights=weights,
            n_jobs=-1,
        )

        # Assemble steps
        steps: list[tuple[str, object]] = [("tfidf", tfidf)]

        # Stage 2 (optional) — SMOTE operates on TF-IDF sparse output
        if self.use_smote:
            steps.append(("smote", SMOTE(random_state=self.random_state)))

        steps.append(("ensemble", ensemble))
        return ImbPipeline(steps=steps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_auto_weights(
        self,
        X: pd.Series,
        y: pd.Series,
        cv: int = _CV_FOLDS,
    ) -> List[float]:
        """Derives ensemble weights from individual estimator CV performance.

        Runs :func:`~sklearn.model_selection.cross_val_score` (macro-F1) on
        each base estimator independently — **after** TF-IDF transformation —
        and returns a weight list proportional to those scores.

        The weights are **not** normalised to sum to 1; ``VotingClassifier``
        normalises internally.  A model with twice the F1 of another will
        receive twice the weight.

        Args:
            X (pd.Series): Raw text feature column.
            y (pd.Series): Target label column.
            cv (int): Number of cross-validation folds. Defaults to 5.

        Returns:
            List[float]: Voting weights ``[w_lr, w_svm, w_rf]`` rounded to 4
            decimal places.

        Raises:
            ValueError: If ``X`` or ``y`` is empty.
        """
        if X.empty or y.empty:
            raise ValueError("X and y must not be empty for weight computation.")

        logger.info(
            "Computing auto-weights via %d-fold CV (macro-F1) on each base estimator…",
            cv,
        )

        # TF-IDF transform once for efficiency
        tfidf = TfidfVectorizer(
            max_features=_TFIDF_MAX_FEATURES,
            ngram_range=_TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
            min_df=3,
            max_df=0.85,
        )
        X_tfidf = tfidf.fit_transform(X)

        lr, svm_calibrated, rf = self._build_base_estimators()

        estimator_map = [("LR", lr), ("SVM (Calibrated)", svm_calibrated), ("RF", rf)]
        f1_scores: List[float] = []

        for name, estimator in estimator_map:
            scores = cross_val_score(
                estimator,
                X_tfidf,
                y,
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
            )
            mean_f1 = float(scores.mean())
            logger.info("  %s → mean macro-F1 = %.4f (±%.4f)", name, mean_f1, scores.std())
            f1_scores.append(mean_f1)

        weights = [round(s, 4) for s in f1_scores]
        logger.info("Auto-computed weights [LR, SVM, RF]: %s", weights)
        return weights

    def fit(self, X: pd.Series, y: pd.Series) -> "TextEnsembleModel":
        """Trains the ensemble pipeline on raw text data.

        If ``self.weights`` is ``None``, :meth:`compute_auto_weights` is
        called first to derive F1-proportional weights from cross-validation.

        Args:
            X (pd.Series): Training text features (raw, unprocessed strings).
            y (pd.Series): Target sentiment labels.

        Returns:
            TextEnsembleModel: ``self``, for method chaining.
        """
        if self.weights is None:
            self.weights = self.compute_auto_weights(X, y)

        self.pipeline = self._build_pipeline(weights=self.weights)

        logger.info(
            "Training Ensemble (SMOTE=%s, Weights=%s) — this may take several minutes…",
            self.use_smote,
            self.weights,
        )
        self.pipeline.fit(X, y)
        logger.info("Ensemble training complete.")
        return self

    def predict(self, X: pd.Series) -> np.ndarray:
        """Predicts class labels for the given text samples.

        Args:
            X (pd.Series): Text features to classify.

        Returns:
            np.ndarray: Predicted class labels.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.pipeline is None:
            raise RuntimeError("Model must be trained (call .fit()) before predict().")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """Returns class probability estimates for the given text samples.

        Probabilities are the soft-voted averages from all base estimators.

        Args:
            X (pd.Series): Text features to score.

        Returns:
            np.ndarray: Shape ``(n_samples, n_classes)`` probability matrix.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.pipeline is None:
            raise RuntimeError("Model must be trained (call .fit()) before predict_proba().")
        return self.pipeline.predict_proba(X)

    def save(self, filepath: str) -> None:
        """Serialises the trained pipeline to disk using joblib.

        Creates all intermediate directories if they do not exist.

        Args:
            filepath (str): Destination path (e.g. ``"artifacts/models/ensemble.pkl"``).

        Raises:
            RuntimeError: If the model has not been trained yet.
            OSError: If the file cannot be written.
        """
        if self.pipeline is None:
            raise RuntimeError("Cannot save an untrained model.  Call .fit() first.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Model saved → %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "TextEnsembleModel":
        """Deserialises a :class:`TextEnsembleModel` instance from disk.

        Args:
            filepath (str): Path to a previously saved ``.pkl`` file.

        Returns:
            TextEnsembleModel: The restored model instance.

        Raises:
            FileNotFoundError: If the file does not exist at ``filepath``.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model file found at: {filepath}")
        logger.info("Loading model ← %s", filepath)
        return joblib.load(filepath)