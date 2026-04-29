"""
Semantic review analysis using zero-shot classification and dense embeddings.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from transformers import pipeline

from ai_engine.llm_integration.llm_client import ask_llm
from ai_engine.text_processing.embeddings import DeepEmbedder


class NextGenReviewAnalyzer:
  """
  High-level review analyzer combining semantic aspect extraction and sentiment.

  This class is designed to be a production-grade facade for downstream
  analytics systems with clear, deterministic fallbacks.
  """

  aspect_anchors: Dict[str, str] = {
    "shipping": "giao hàng, đóng gói, vận chuyển",
    "product": "chất lượng sản phẩm, mẫu mã",
    "price": "giá cả, khuyến mãi, ưu đãi",
    "service": "dịch vụ, nhân viên, chăm sóc khách hàng",
  }

  def __init__(self) -> None:
    """Initialize embedder, zero-shot classifier, and anchor embeddings."""
    self.embedder = DeepEmbedder()
    device = 0 if torch.cuda.is_available() else -1
    self.zero_shot = pipeline(
      "zero-shot-classification",
      model="joeddav/xlm-roberta-large-xnli",
      device=device,
    )

    anchor_texts = list(self.aspect_anchors.values())
    self._anchor_vectors = self.embedder.encode(anchor_texts)

  def extract_aspects(self, text: str, threshold: float = 0.65) -> List[str]:
    """
    Extract aspects based on cosine similarity to anchor phrases.

    Args:
      text: Input review text.
      threshold: Similarity cutoff for aspect selection.

    Returns:
      List[str]: List of aspect keys exceeding the threshold.
    """
    if not text:
      return []

    vector = self.embedder.encode(text)
    if vector.size == 0 or self._anchor_vectors.size == 0:
      return []

    scores = self.embedder.compute_cosine_similarity(vector, self._anchor_vectors)
    if scores.size == 0:
      return []

    similarities = scores[0]
    aspects = []
    for idx, key in enumerate(self.aspect_anchors.keys()):
      if similarities[idx] > threshold:
        aspects.append(key)
    return aspects

  def predict_sentiment(self, text: str) -> str:
    """
    Predict sentiment using a zero-shot classifier with LLM fallback.

    Args:
      text: Input review text.

    Returns:
      str: Predicted sentiment label.
    """
    if not text:
      return "trung lập"

    labels = ["tích cực", "tiêu cực", "trung lập"]
    try:
      result = self.zero_shot(text, candidate_labels=labels)
    except Exception:
      return ask_llm(text)

    scores = np.array(result.get("scores", []), dtype=np.float32)
    if scores.size != len(labels):
      return ask_llm(text)

    max_score = float(scores.max())
    if max_score < 0.45:
      return ask_llm(text)

    if np.allclose(scores, scores.mean(), atol=0.05):
      return ask_llm(text)

    best_idx = int(scores.argmax())
    return labels[best_idx]

  def analyze_review(self, text: str) -> Dict[str, object]:
    """
    Analyze a review and return a structured result.

    Args:
      text: Input review text.

    Returns:
      Dict[str, object]: Structured analysis result.
    """
    aspects = self.extract_aspects(text)
    sentiment = self.predict_sentiment(text)

    return {
      "aspects": aspects,
      "sentiment": sentiment,
    }
