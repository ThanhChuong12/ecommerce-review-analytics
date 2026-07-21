"""Sparse text vectorization utilities using scikit-learn TF-IDF."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Union

import joblib
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

TextLike = Union[Sequence[str], Iterable[str]]
PathLike = Union[str, Path]


class TextVectorizer:
    """Production-ready TF-IDF vectorizer for sparse text representations."""

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: Union[int, float] = 3,
        max_df: Union[int, float] = 0.90,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            token_pattern=r"(?u)\b\w+\b",
        )
        self._is_fitted = False

    def _to_text_list(self, texts: TextLike) -> List[str]:
        """Convert input text container into a list of strings."""
        if texts is None:
            return []
        if hasattr(texts, "tolist"):
            texts = texts.tolist()
        return ["" if t is None else str(t) for t in texts]

    def fit_transform(self, texts: TextLike) -> csr_matrix:
        """Fit the vectorizer on text data and return sparse TF-IDF feature matrix."""
        data = self._to_text_list(texts)
        if not data:
            self._is_fitted = True
            return csr_matrix((0, 0))

        matrix = self.vectorizer.fit_transform(data)
        self._is_fitted = True
        return matrix.tocsr()

    def transform(self, texts: TextLike) -> csr_matrix:
        """Transform text entries into sparse TF-IDF feature matrix using fitted vocabulary."""
        if not self._is_fitted:
            raise RuntimeError("TextVectorizer is not fitted. Call fit_transform() or load_model().")

        data = self._to_text_list(texts)
        if not data:
            return csr_matrix((0, 0))

        return self.vectorizer.transform(data).tocsr()

    def save_model(self, path: PathLike) -> None:
        """Persist fitted vectorizer instance to disk using joblib."""
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.vectorizer, target)
        except Exception as exc:
            raise RuntimeError(f"Failed to save vectorizer to {path}: {exc}") from exc

    def load_model(self, path: PathLike) -> None:
        """Load a persisted vectorizer instance from disk."""
        try:
            self.vectorizer = joblib.load(path)
            self._is_fitted = True
        except Exception as exc:
            raise RuntimeError(f"Failed to load vectorizer from {path}: {exc}") from exc
