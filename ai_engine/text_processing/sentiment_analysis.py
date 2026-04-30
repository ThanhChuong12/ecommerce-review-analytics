"""
Semantic review analysis using zero-shot classification and dense embeddings.
"""

from __future__ import annotations

import re
from typing import Dict, List, Any

import torch
from transformers import pipeline

from ai_engine.llm_integration.llm_client import LLMFallbackClient
from ai_engine.text_processing.embeddings import DeepEmbedder


# Define domain-specific sentiment lexicons
POSITIVE_LEXICON = [
    "tốt", "tuyệt", "đẹp", "ưng", "nhanh", "chất lượng", "ok", 
    "xuất sắc", "đỉnh", "hài lòng", "ổn", "xịn", "rẻ", "thơm", "ngon", "10 điểm", "hoàn hảo", "chắc chắn",
    "thích", "dễ thương", "tuyệt vời", "mịn", "rõ nét", "nhỏ gọn", "tiện lợi", "chuẩn", "chính hãng", "uy tín", "đáng tiền", "dễ xài", "cứng cáp"
]

NEGATIVE_LEXICON = [
    "tệ", "chậm", "xấu", "kém", "thất vọng", "lỗi", "bẩn", 
    "hư", "trễ", "chán", "móp", "rách", "fake", "đắt", "mắc", 
    "lừa đảo", "dỏm", "giả", "dìm", "sai", "nhầm",
    "không ưng", "không đẹp", "phí tiền", "xước", "vỡ", "nứt", "kém chất lượng", "không giống", "khác hình", "tức", "hỏng"
]

# Compile regex patterns for negations before sentiments
NEGATION_MODIFIERS = r"(không|chưa|chẳng.có|đếch)"

# Format lexicons for regex (handle spaces if any)
_pos_regex_list = [w.replace(" ", r"\s+") for w in POSITIVE_LEXICON]
_neg_regex_list = [w.replace(" ", r"\s+") for w in NEGATIVE_LEXICON]

# Exact word boundary matching for sentiments
POS_PATTERN = re.compile(r"\b(" + "|".join(_pos_regex_list) + r")\b", re.IGNORECASE)
NEG_PATTERN = re.compile(r"\b(" + "|".join(_neg_regex_list) + r")\b", re.IGNORECASE)

# E.g., Match "không tốt", "chưa hài lòng", "không đẹp"
NEGATED_POS_PATTERN = re.compile(rf"\b{NEGATION_MODIFIERS}\s+(" + "|".join(_pos_regex_list) + r")\b", re.IGNORECASE)
# Match "không xấu", "không trễ"
NEGATED_NEG_PATTERN = re.compile(rf"\b{NEGATION_MODIFIERS}\s+(" + "|".join(_neg_regex_list) + r")\b", re.IGNORECASE)

def has_true_positive(text: str) -> bool:
    """Check if there are true positive words not preceded by negation."""
    # Temporarily remove negated positive words (e.g., 'không tốt' -> remove) to avoid false positives
    text_without_neg_pos = NEGATED_POS_PATTERN.sub("", text)
    return bool(POS_PATTERN.search(text_without_neg_pos))

def has_true_negative(text: str) -> bool:
    """Check if there are true negative words, or negated positive words."""
    # 'không tốt' acts as negative
    if NEGATED_POS_PATTERN.search(text):
        return True
    
    text_without_neg_neg = NEGATED_NEG_PATTERN.sub("", text)
    return bool(NEG_PATTERN.search(text_without_neg_neg))

def assign_heuristic_label(row: Any) -> str:
    """
    Assign weakly supervised labels using N-gram negation lookaround and rating gravity.
    Returns 'ambiguous' when rating contradicts text or context is mixed/unclear.
    Relaxed to reduce LLM API calls for clear ratings.
    """
    try:
        if isinstance(row, str):
            text = row.lower()
            rating = 0
        else:
            text = str(row.get('cleaned_text', '')).lower()
            rating = int(row.get('rating', 0))
    except (ValueError, AttributeError):
        return 'ambiguous'
    
    has_pos = has_true_positive(text)
    has_neg = has_true_negative(text)
    
    # If no rating provided
    if rating == 0:
        if has_pos and not has_neg: return 'tích cực'
        if has_neg and not has_pos: return 'tiêu cực'
        return 'ambiguous'

    # Safe assumption for 4-5 stars: if it doesn't contain negative words, it's positive.
    if rating >= 4:
        if has_neg:
            return 'ambiguous' # Mixed or contradiction
        return 'tích cực'
        
    # Safe assumption for 1-2 stars: if it doesn't contain positive words, it's negative.
    if rating <= 2:
        if has_pos:
            return 'ambiguous' # Mixed or contradiction
        return 'tiêu cực'
        
    # For 3 stars, rely purely on lexical presence
    if rating == 3:
        if has_pos and not has_neg: return 'tích cực'
        if has_neg and not has_pos: return 'tiêu cực'
        if not has_pos and not has_neg: return 'trung lập'
        return 'ambiguous'
        
    return 'ambiguous'


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
