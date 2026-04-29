"""
Semantic review analysis using zero-shot classification and dense embeddings.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import pipeline

from ai_engine.llm_integration.llm_client import LLMFallbackClient
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
        """
        Initialize embedder, zero-shot classifier, and anchor embeddings.

        Uses dynamic device resolution to support CUDA, Apple MPS, and CPU.
        """
        self.embedder = DeepEmbedder()
        self.llm_client = LLMFallbackClient()

        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

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

    def _fallback_sentiment(self, text: str) -> str:
        return self.llm_client.analyze(text).get("sentiment", "trung lập")

    def predict_sentiment(self, text: str) -> str:
        """
        Predict sentiment using zero-shot classification with safe fallbacks.

        Corrects the pipeline sorting behavior by using the top label directly
        from the result instead of using argmax on scores.
        """
        if not text:
            return "trung lập"

        labels = ["tích cực", "tiêu cực", "trung lập"]
        try:
            result = self.zero_shot(text, candidate_labels=labels)
        except Exception:
            return self._fallback_sentiment(text)

        result_labels = result.get("labels", [])
        result_scores = result.get("scores", [])
        if not result_labels or not result_scores:
            return self._fallback_sentiment(text)

        top_label = result_labels[0]
        top_score = float(result_scores[0])

        if top_score < 0.45:
            return self._fallback_sentiment(text)

        if len(result_scores) > 1 and (top_score - float(result_scores[1])) < 0.05:
            return self._fallback_sentiment(text)

        return top_label

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
