"""
Sparse text vectorization utilities using scikit-learn TF-IDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Union

import joblib
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


TextLike = Union[Sequence[str], Iterable[str]]
PathLike = Union[str, Path]


class TextVectorizer:
	"""
	Production-ready TF-IDF vectorizer for sparse text representations.

	This wrapper adds safety checks, consistent defaults, and simple
	serialization for downstream ML pipelines.
	"""

	def __init__(
		self,
		max_features: int = 10000,
		ngram_range: tuple[int, int] = (1, 2),
		min_df: Union[int, float] = 3,
		max_df: Union[int, float] = 0.90,
	) -> None:
		"""
		Initialize the TF-IDF vectorizer.

		Args:
			max_features: Maximum vocabulary size to control memory usage.
			ngram_range: N-gram range to capture uni-grams and bi-grams.
			min_df: Ignore terms that appear in fewer than this many documents.
			max_df: Ignore terms that appear in more than this proportion of documents.
		"""
		self.vectorizer = TfidfVectorizer(
			max_features=max_features,
			ngram_range=ngram_range,
			min_df=min_df,
			max_df=max_df,
			# Preserve 1-character sentiment words and underscore tokens.
			token_pattern=r"(?u)\b\w+\b",
		)
		self._is_fitted = False

	def _to_text_list(self, texts: TextLike) -> List[str]:
		"""
		Convert input text container to a list of strings.

		Args:
			texts: Sequence or iterable of strings.

		Returns:
			List[str]: Normalized list of strings.
		"""
		if texts is None:
			return []

		if hasattr(texts, "tolist"):
			texts = texts.tolist()

		return ["" if t is None else str(t) for t in texts]

	def fit_transform(self, texts: TextLike) -> csr_matrix:
		"""
		Fit the vectorizer on training data and return TF-IDF features.

		Args:
			texts: Training text data.

		Returns:
			csr_matrix: Sparse TF-IDF features.
		"""
		data = self._to_text_list(texts)
		if not data:
			self._is_fitted = True
			return csr_matrix((0, 0))

		matrix = self.vectorizer.fit_transform(data)
		self._is_fitted = True
		return matrix.tocsr()

	def transform(self, texts: TextLike) -> csr_matrix:
		"""
		Transform new text data using the learned vocabulary.

		Args:
			texts: Unseen text data.

		Returns:
			csr_matrix: Sparse TF-IDF features.

		Raises:
			RuntimeError: If called before the vectorizer is fitted or loaded.
		"""
		if not self._is_fitted:
			raise RuntimeError("TextVectorizer is not fitted. Call fit_transform() or load_model().")

		data = self._to_text_list(texts)
		if not data:
			return csr_matrix((0, 0))

		return self.vectorizer.transform(data).tocsr()

	def save_model(self, path: PathLike) -> None:
		"""
		Persist the fitted vectorizer to disk using joblib.

		Args:
			path: Output file path.

		Raises:
			RuntimeError: If saving fails due to I/O issues.
		"""
		try:
			target = Path(path)
			target.parent.mkdir(parents=True, exist_ok=True)
			joblib.dump(self.vectorizer, target)
		except Exception as exc:
			raise RuntimeError(f"Failed to save vectorizer to {path}: {exc}") from exc

	def load_model(self, path: PathLike) -> None:
		"""
		Load a vectorizer from disk using joblib.

		Args:
			path: Input file path.

		Raises:
			RuntimeError: If loading fails or file is missing/corrupted.
		"""
		try:
			self.vectorizer = joblib.load(path)
			self._is_fitted = True
		except Exception as exc:
			raise RuntimeError(f"Failed to load vectorizer from {path}: {exc}") from exc
