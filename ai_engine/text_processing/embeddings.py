"""Dense text embedding utilities for Vietnamese semantic representations."""

from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

TextInput = Union[str, List[str]]


class DeepEmbedder:
    """Sentence-Transformers wrapper for dense Vietnamese text embeddings."""

    def __init__(self, model_name: str = "keepitreal/vietnamese-sbert") -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: TextInput, batch_size: int = 32) -> np.ndarray:
        """Encode string or list of strings into normalized float32 numpy embeddings."""
        if texts is None:
            return np.empty((0, 0), dtype=np.float32)

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32, copy=False)

    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between two numpy embedding arrays."""
        if vec1.size == 0 or vec2.size == 0:
            return np.empty((0, 0), dtype=np.float32)

        tensor1 = torch.from_numpy(vec1)
        tensor2 = torch.from_numpy(vec2)
        scores = util.cos_sim(tensor1, tensor2)
        return scores.cpu().numpy().astype(np.float32, copy=False)
