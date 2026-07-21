"""Semantic review analysis using zero-shot classification, heuristics, and dense embeddings."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from transformers import pipeline

from ai_engine.llm_integration.llm_client import LLMBudget, LLMFallbackClient
from ai_engine.text_processing.embeddings import DeepEmbedder


class DataSource(str, Enum):
    """Identifies input CSV origin for tuning heuristic labeling rules."""

    ALL_REVIEWS = "all_reviews"
    ALL_BAD_REVIEWS = "all_bad_reviews"
    ALL_GOOD_REVIEWS = "all_good_reviews"


POSITIVE_LEXICON: List[str] = [
    "tốt", "tuyệt", "đẹp", "ưng", "nhanh", "chất lượng", "ok",
    "xuất sắc", "đỉnh", "hài lòng", "ổn", "xịn", "rẻ", "thơm", "ngon",
    "10 điểm", "hoàn hảo", "chắc chắn", "thích", "dễ thương", "tuyệt vời",
    "mịn", "rõ nét", "nhỏ gọn", "tiện lợi", "chuẩn", "chính hãng", "uy tín",
    "đáng tiền", "dễ xài", "cứng cáp", "êm", "mượt", "bền", "đúng hàng",
    "đúng mô tả", "giao nhanh", "đóng gói cẩn thận",
]

NEGATIVE_LEXICON: List[str] = [
    "tệ", "chậm", "xấu", "kém", "thất vọng", "lỗi", "bẩn",
    "hư", "trễ", "chán", "móp", "rách", "fake", "đắt", "mắc",
    "lừa đảo", "dỏm", "giả", "sai", "nhầm",
    "không ưng", "không đẹp", "phí tiền", "xước", "vỡ", "nứt",
    "kém chất lượng", "không giống", "khác hình", "tức", "hỏng",
    "bay màu", "nhão", "xù", "giãn", "đứt", "thô", "ngứa",
    "không đúng", "giao sai", "thiếu hàng",
]

_NEGATION_GROUP = r"(?:không|chưa|chẳng\s*có|đếch)"
_pos_regex_list = [w.replace(" ", r"\s+") for w in POSITIVE_LEXICON]
_neg_regex_list = [w.replace(" ", r"\s+") for w in NEGATIVE_LEXICON]

POS_PATTERN = re.compile(r"\b(" + "|".join(_pos_regex_list) + r")\b", re.IGNORECASE)
NEG_PATTERN = re.compile(r"\b(" + "|".join(_neg_regex_list) + r")\b", re.IGNORECASE)

NEGATED_POS_PATTERN = re.compile(
    rf"\b{_NEGATION_GROUP}\s+(" + "|".join(_pos_regex_list) + r")\b", re.IGNORECASE
)
NEGATED_NEG_PATTERN = re.compile(
    rf"\b{_NEGATION_GROUP}\s+(" + "|".join(_neg_regex_list) + r")\b", re.IGNORECASE
)


def has_true_positive(text: str) -> bool:
    """Return True if text contains genuine positive sentiment expressions."""
    text_without_neg_pos = NEGATED_POS_PATTERN.sub("", text)
    return bool(POS_PATTERN.search(text_without_neg_pos))


def has_true_negative(text: str) -> bool:
    """Return True if text contains genuine negative sentiment expressions."""
    if NEGATED_POS_PATTERN.search(text):
        return True
    text_without_neg_neg = NEGATED_NEG_PATTERN.sub("", text)
    return bool(NEG_PATTERN.search(text_without_neg_neg))


_SOURCE_RATING_BANDS: Dict[DataSource, tuple[int, int]] = {
    DataSource.ALL_REVIEWS: (4, 2),
    DataSource.ALL_BAD_REVIEWS: (4, 2),
    DataSource.ALL_GOOD_REVIEWS: (4, 2),
}

_SOURCE_MIXED_TOLERANCE: Dict[DataSource, bool] = {
    DataSource.ALL_REVIEWS: False,
    DataSource.ALL_BAD_REVIEWS: True,
    DataSource.ALL_GOOD_REVIEWS: True,
}


def assign_heuristic_label(
    row: Any,
    source: DataSource = DataSource.ALL_REVIEWS,
) -> str:
    """Assign weakly supervised sentiment label using rating priors and lexical matching."""
    try:
        if isinstance(row, str):
            text = row.lower().strip()
            rating: int = 0
        else:
            text = str(row.get("cleaned_text", "")).lower().strip()
            raw_rating = row.get("rating", 0)
            rating = int(raw_rating) if str(raw_rating).strip().isdigit() else 0
    except (ValueError, AttributeError, TypeError):
        return "ambiguous"

    has_pos = has_true_positive(text)
    has_neg = has_true_negative(text)
    lenient = _SOURCE_MIXED_TOLERANCE[source]

    if rating == 0:
        if has_pos and not has_neg:
            return "tích cực"
        if has_neg and not has_pos:
            return "tiêu cực"
        if source == DataSource.ALL_BAD_REVIEWS:
            return "tiêu cực"
        if source == DataSource.ALL_GOOD_REVIEWS:
            return "tích cực"
        return "ambiguous"

    if rating >= 4:
        if not has_neg:
            return "tích cực"
        if lenient:
            if has_pos:
                return "tích cực"
            return "ambiguous"
        return "ambiguous"

    if rating <= 2:
        if not has_pos:
            return "tiêu cực"
        if lenient:
            if has_neg:
                return "tiêu cực"
            return "ambiguous"
        return "ambiguous"

    if rating == 3:
        if has_pos and not has_neg:
            return "tích cực"
        if has_neg and not has_pos:
            return "tiêu cực"
        if not has_pos and not has_neg:
            return "trung lập"
        return "ambiguous"

    return "ambiguous"


class NextGenReviewAnalyzer:
    """Multi-stage sentiment analyzer integrating heuristics, zero-shot XLM-R, and LLM fallback."""

    aspect_anchors: Dict[str, str] = {
        "shipping": "giao hàng, đóng gói, vận chuyển",
        "product": "chất lượng sản phẩm, mẫu mã",
        "price": "giá cả, khuyến mãi, ưu đãi",
        "service": "dịch vụ, nhân viên, chăm sóc khách hàng",
    }

    def __init__(self) -> None:
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
        """Extract domain aspects matching anchor phrase embedding similarity."""
        if not text:
            return []

        vector = self.embedder.encode(text)
        if vector.size == 0 or self._anchor_vectors.size == 0:
            return []

        scores = self.embedder.compute_cosine_similarity(vector, self._anchor_vectors)
        if scores.size == 0:
            return []

        similarities = scores[0]
        return [
            key
            for idx, key in enumerate(self.aspect_anchors.keys())
            if similarities[idx] > threshold
        ]

    def _rating_to_prior(self, rating: Optional[int]) -> Optional[str]:
        """Map numeric star rating to prior sentiment category."""
        if rating is None:
            return None
        if rating >= 4:
            return "tích cực"
        if rating <= 2:
            return "tiêu cực"
        return None

    def _fallback_sentiment(
        self, text: str, rating: Optional[int] = None
    ) -> str:
        """Resolve ambiguous sentiment via budget-aware LLM or rating prior."""
        prior = self._rating_to_prior(rating)

        if LLMBudget.is_exhausted():
            return prior if prior is not None else LLMBudget._exhausted_label

        remaining = LLMBudget.remaining()
        total = LLMBudget._max_calls
        low_budget = total > 0 and remaining < max(1, int(total * 0.20))
        if low_budget and prior is not None:
            return prior

        payload: Dict[str, Any] = {"text": text}
        if rating is not None:
            payload["rating"] = rating
        return self.llm_client.analyze(payload).get("sentiment", "trung lập")

    def predict_sentiment(
        self, text: str, rating: Optional[int] = None
    ) -> tuple[str, Optional[Dict[str, float]]]:
        """Predict sentiment label and probabilities using zero-shot model and fallbacks."""
        if not text:
            return "trung lập", None

        prior = self._rating_to_prior(rating)

        if LLMBudget.is_exhausted():
            return (prior if prior is not None else "trung lập"), None

        candidate_labels = ["tích cực", "tiêu cực", "trung lập"]
        try:
            result = self.zero_shot(text, candidate_labels=candidate_labels)
        except Exception:
            if prior is not None:
                return prior, None
            return self._fallback_sentiment(text, rating), None

        result_labels: List[str] = result.get("labels", [])
        result_scores: List[float] = result.get("scores", [])

        if not result_labels or not result_scores:
            if prior is not None:
                return prior, None
            return self._fallback_sentiment(text, rating), None

        top_label: str = result_labels[0]
        top_score: float = float(result_scores[0])
        score_gap: float = (
            top_score - float(result_scores[1])
            if len(result_scores) > 1
            else 1.0
        )

        if top_score >= 0.35 and score_gap >= 0.08:
            return top_label, dict(zip(result_labels, result_scores))

        if prior is not None:
            return prior, None

        return self._fallback_sentiment(text, rating), None

    def analyze_review(
        self,
        text: str,
        rating: Optional[int] = None,
        source: DataSource = DataSource.ALL_REVIEWS,
    ) -> Dict[str, Any]:
        """Analyze review text and return aspect, sentiment, and method details."""
        row: Dict[str, Any] = {"cleaned_text": text, "rating": rating or 0}
        heuristic = assign_heuristic_label(row, source=source)

        if heuristic != "ambiguous":
            aspects = self.extract_aspects(text)
            probs = {"tích cực": 0.0, "tiêu cực": 0.0, "trung lập": 0.0}
            if heuristic in probs:
                probs[heuristic] = 1.0

            return {
                "aspects": aspects,
                "sentiment": heuristic,
                "method": "heuristic",
                "probabilities": probs,
            }

        aspects = self.extract_aspects(text)
        sentiment, probs = self.predict_sentiment(text, rating=rating)

        if probs is None:
            probs = {"tích cực": 0.0, "tiêu cực": 0.0, "trung lập": 0.0}
            if sentiment in probs:
                probs[sentiment] = 1.0

        return {
            "aspects": aspects,
            "sentiment": sentiment,
            "method": "model",
            "probabilities": probs,
        }
