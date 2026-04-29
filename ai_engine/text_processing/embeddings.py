"""
Dense text embedding utilities for Vietnamese semantic representations.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


TextInput = Union[str, List[str]]


class DeepEmbedder:
	"""
	High-throughput embedding wrapper for Sentence-Transformers models.

	The class loads a pre-trained Vietnamese SBERT model and exposes
	simple methods for encoding and cosine similarity.
	"""

	def __init__(self, model_name: str = "keepitreal/vietnamese-sbert") -> None:
		"""
		Initialize the embedder and load the model to the best device.

		Args:
			model_name: HuggingFace model identifier for Sentence-Transformers.
		"""
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model = SentenceTransformer(model_name, device=self.device)

	def encode(self, texts: TextInput) -> np.ndarray:
		"""
		Encode text(s) into dense vector representations.

		Args:
			texts: A single string or list of strings.

		Returns:
			np.ndarray: Dense embeddings with shape (n, dim).
		"""
		if texts is None:
			return np.empty((0, 0), dtype=np.float32)

		if isinstance(texts, str):
			texts = [texts]

		if not texts:
			return np.empty((0, 0), dtype=np.float32)

		embeddings = self.model.encode(
			texts,
			convert_to_numpy=True,
			show_progress_bar=False,
			normalize_embeddings=True,
		)
		return embeddings.astype(np.float32, copy=False)

	def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
		"""
		Compute cosine similarity using Sentence-Transformers utilities.

		Args:
			vec1: First embedding matrix or vector.
			vec2: Second embedding matrix or vector.

		Returns:
			np.ndarray: Cosine similarity matrix.
		"""
		if vec1.size == 0 or vec2.size == 0:
			return np.empty((0, 0), dtype=np.float32)

		tensor1 = torch.from_numpy(vec1)
		tensor2 = torch.from_numpy(vec2)
		scores = util.cos_sim(tensor1, tensor2)
		return scores.cpu().numpy().astype(np.float32, copy=False)
