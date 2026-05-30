"""
Semantic review analysis using zero-shot classification and dense embeddings.

Supports three data source modes:
  - "all_reviews"      : Mixed ratings (1-5) from Tiki — full heuristic + LLM pipeline.
  - "all_bad_reviews"  : Rating 1-2 from Shopee — rating-dominant negative labeling.
  - "all_good_reviews" : Rating 4-5 from Shopee — rating-dominant positive labeling.

The ``DataSource`` enum is used to communicate which file a row comes from so
that ``assign_heuristic_label`` can apply appropriate confidence weights and
avoid wasting LLM calls on clear-cut cases.

LLM call budget
~~~~~~~~~~~~~~~
To cap the number of API calls during a batch labeling run, configure
:class:`~ai_engine.llm_integration.llm_client.LLMBudget` **before** creating
a :class:`NextGenReviewAnalyzer`::

    from ai_engine.llm_integration.llm_client import LLMBudget
    LLMBudget.configure(max_calls=300)   # hard cap for this run

When the budget is exhausted every subsequent ``predict_sentiment`` call
returns the rating prior (if available) or ``"trung lập"`` without touching
the network.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from transformers import pipeline

from ai_engine.llm_integration.llm_client import LLMBudget, LLMFallbackClient
from ai_engine.text_processing.embeddings import DeepEmbedder


# ---------------------------------------------------------------------------
# Data-source declaration
# ---------------------------------------------------------------------------

class DataSource(str, Enum):
    """Identifies the origin CSV so labeling confidence can be tuned per file.

    Attributes:
        ALL_REVIEWS:     Mixed-rating Tiki dataset (all_reviews.csv).
        ALL_BAD_REVIEWS: Low-rating Shopee dataset (all_bad_reviews.csv).
                         Contains only rating 1–2; strong prior → tiêu cực.
        ALL_GOOD_REVIEWS: High-rating Shopee dataset (all_good_reviews.csv).
                          Contains only rating 4–5; strong prior → tích cực.
    """

    ALL_REVIEWS = "all_reviews"
    ALL_BAD_REVIEWS = "all_bad_reviews"
    ALL_GOOD_REVIEWS = "all_good_reviews"


# ---------------------------------------------------------------------------
# Domain-specific sentiment lexicons
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------

# Fixed: replaced `.` wildcard with `\s*` so "chẳng có" is matched correctly.
_NEGATION_GROUP = r"(?:không|chưa|chẳng\s*có|đếch)"

_pos_regex_list = [w.replace(" ", r"\s+") for w in POSITIVE_LEXICON]
_neg_regex_list = [w.replace(" ", r"\s+") for w in NEGATIVE_LEXICON]

# Base patterns — exact word boundary matching
POS_PATTERN = re.compile(
    r"\b(" + "|".join(_pos_regex_list) + r")\b", re.IGNORECASE
)
NEG_PATTERN = re.compile(
    r"\b(" + "|".join(_neg_regex_list) + r")\b", re.IGNORECASE
)

# Negated-positive: "không tốt", "chưa hài lòng", "chẳng có gì đẹp"
NEGATED_POS_PATTERN = re.compile(
    rf"\b{_NEGATION_GROUP}\s+(" + "|".join(_pos_regex_list) + r")\b",
    re.IGNORECASE,
)
# Negated-negative: "không xấu", "chưa tệ"
NEGATED_NEG_PATTERN = re.compile(
    rf"\b{_NEGATION_GROUP}\s+(" + "|".join(_neg_regex_list) + r")\b",
    re.IGNORECASE,
)


def has_true_positive(text: str) -> bool:
    """Return ``True`` if *text* contains at least one genuine positive signal.

    Positive words preceded by a negation (e.g. "không tốt") are first masked
    out so they do not produce a false positive signal.
    """
    text_without_neg_pos = NEGATED_POS_PATTERN.sub("", text)
    return bool(POS_PATTERN.search(text_without_neg_pos))


def has_true_negative(text: str) -> bool:
    """Return ``True`` if *text* contains at least one genuine negative signal.

    "Negated positives" (e.g. "không tốt") count as negative.
    "Negated negatives" (e.g. "không xấu") are masked before checking.
    """
    if NEGATED_POS_PATTERN.search(text):
        return True
    text_without_neg_neg = NEGATED_NEG_PATTERN.sub("", text)
    return bool(NEG_PATTERN.search(text_without_neg_neg))


# ---------------------------------------------------------------------------
# Heuristic labeling — rating-gravity with per-source confidence
# ---------------------------------------------------------------------------

# Rating thresholds for "dominant" label assignment per data source.
# (min_rating_for_positive, max_rating_for_negative)
_SOURCE_RATING_BANDS: Dict[DataSource, tuple[int, int]] = {
    DataSource.ALL_REVIEWS:      (4, 2),   # standard bands
    DataSource.ALL_BAD_REVIEWS:  (4, 2),   # same bands, but source guarantees 1-2
    DataSource.ALL_GOOD_REVIEWS: (4, 2),   # same bands, but source guarantees 4-5
}

# Per-source tolerance: how much positive lexicon in a 1-2 star review is still
# acceptable before we call it 'ambiguous'.  Shopee bad/good reviews are
# scraped by rating, so we trust the rating signal more strongly.
_SOURCE_MIXED_TOLERANCE: Dict[DataSource, bool] = {
    DataSource.ALL_REVIEWS:      False,  # strict — any mix → ambiguous
    DataSource.ALL_BAD_REVIEWS:  True,   # lenient — rating dominant, text may praise style
    DataSource.ALL_GOOD_REVIEWS: True,   # lenient — rating dominant, text may note minor issues
}


def assign_heuristic_label(
    row: Any,
    source: DataSource = DataSource.ALL_REVIEWS,
) -> str:
    """Assign a weakly supervised sentiment label using rating gravity and lexical signals.

    The function is designed to work with all three data sources:

    * **all_reviews** (mixed Tiki data): strict contradiction detection — any
      mixed signal returns ``"ambiguous"`` for downstream LLM resolution.
    * **all_bad_reviews** (Shopee rating 1–2): lenient mode — the dataset is
      pre-filtered to low ratings, so incidental positive words (e.g. praising
      design while complaining about quality) do **not** trigger ``"ambiguous"``.
      Only genuine contradictions (strong positive text + rating 1) escalate.
    * **all_good_reviews** (Shopee rating 4–5): lenient mode — minor complaints
      in otherwise positive reviews do not trigger ``"ambiguous"``.

    Args:
        row: Either a ``str`` (plain text, no rating) or a ``dict``-like object
             with ``"cleaned_text"`` and ``"rating"`` keys.
        source: The :class:`DataSource` the row originates from.

    Returns:
        One of ``"tích cực"``, ``"tiêu cực"``, ``"trung lập"``, or
        ``"ambiguous"`` (requires LLM resolution).
    """
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

    # ------------------------------------------------------------------
    # No rating available — rely purely on lexical signals
    # ------------------------------------------------------------------
    if rating == 0:
        if has_pos and not has_neg:
            return "tích cực"
        if has_neg and not has_pos:
            return "tiêu cực"
        # For dedicated bad/good sources with missing rating, trust the source.
        if source == DataSource.ALL_BAD_REVIEWS:
            return "tiêu cực"
        if source == DataSource.ALL_GOOD_REVIEWS:
            return "tích cực"
        return "ambiguous"

    # ------------------------------------------------------------------
    # Rating 4–5 star → positive prior
    # ------------------------------------------------------------------
    if rating >= 4:
        if not has_neg:
            return "tích cực"
        if lenient:
            # Minor complaint in a high-rating review: still positive.
            # Only escalate if negative signals clearly dominate.
            if has_pos:
                return "tích cực"   # e.g. "giao nhanh nhưng hơi chậm" — overall ok
            return "ambiguous"      # no positive words + negative signal — unusual
        return "ambiguous"          # strict: any negative → needs LLM

    # ------------------------------------------------------------------
    # Rating 1–2 star → negative prior
    # ------------------------------------------------------------------
    if rating <= 2:
        if not has_pos:
            return "tiêu cực"
        if lenient:
            # Incidental positive words in a bad review (e.g. "chất vải oke nhưng size sai").
            # Only escalate when the positive signal clearly dominates.
            if has_neg:
                return "tiêu cực"  # both present but rating confirms negative
            # Positive-only text with rating 1-2 (e.g. pure sarcasm): ambiguous
            return "ambiguous"
        return "ambiguous"         # strict: any positive → needs LLM

    # ------------------------------------------------------------------
    # Rating 3 star → neutral prior; rely on lexical signals
    # ------------------------------------------------------------------
    if rating == 3:
        if has_pos and not has_neg:
            return "tích cực"
        if has_neg and not has_pos:
            return "tiêu cực"
        if not has_pos and not has_neg:
            return "trung lập"
        return "ambiguous"

    return "ambiguous"


# ---------------------------------------------------------------------------
# NextGenReviewAnalyzer — production-grade facade
# ---------------------------------------------------------------------------

class NextGenReviewAnalyzer:
    """High-level review analyzer combining semantic aspect extraction and sentiment.

    Combines three prediction layers:
    1. **Heuristic** — fast regex-based rule engine (``assign_heuristic_label``).
    2. **Zero-shot** — ``joeddav/xlm-roberta-large-xnli`` via HuggingFace
       Transformers for ambiguous cases.
    3. **LLM fallback** — ``LLMFallbackClient`` when zero-shot confidence is low.

    The ``rating`` parameter is propagated through all layers so that the model
    always has access to the strongest available signal.

    This class is designed to be a production-grade facade for downstream
    analytics systems with clear, deterministic fallbacks.
    """

    aspect_anchors: Dict[str, str] = {
        "shipping": "giao hàng, đóng gói, vận chuyển",
        "product":  "chất lượng sản phẩm, mẫu mã",
        "price":    "giá cả, khuyến mãi, ưu đãi",
        "service":  "dịch vụ, nhân viên, chăm sóc khách hàng",
    }

    def __init__(self) -> None:
        """Initialize embedder, zero-shot classifier, and anchor embeddings.

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

    # ------------------------------------------------------------------
    # Aspect extraction
    # ------------------------------------------------------------------

    def extract_aspects(self, text: str, threshold: float = 0.65) -> List[str]:
        """Extract review aspects based on cosine similarity to anchor phrases.

        Args:
            text: Input review text.
            threshold: Minimum cosine similarity for an aspect to be reported.

        Returns:
            List of aspect keys (e.g. ``["shipping", "product"]``) whose anchor
            embedding exceeds *threshold* similarity to *text*.
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
        return [
            key
            for idx, key in enumerate(self.aspect_anchors.keys())
            if similarities[idx] > threshold
        ]

    # ------------------------------------------------------------------
    # Sentiment prediction
    # ------------------------------------------------------------------

    def _rating_to_prior(self, rating: Optional[int]) -> Optional[str]:
        """Convert a numeric star rating to a sentiment prior label.

        Returns ``None`` when the rating is absent or neutral (3 stars).
        """
        if rating is None:
            return None
        if rating >= 4:
            return "tích cực"
        if rating <= 2:
            return "tiêu cực"
        return None  # rating 3 → no strong prior

    def _fallback_sentiment(
        self, text: str, rating: Optional[int] = None
    ) -> str:
        """Return a sentiment label using the rating prior or LLM as a last resort.

        Decision order (lowest API cost first):

        1. **Budget exhausted** → return rating prior or ``"trung lập"`` immediately.
        2. **Rating prior available** and budget is under 20 % remaining → use prior
           to preserve quota for genuinely ambiguous cases.
        3. **LLM call** with rating hint embedded in the prompt.

        Args:
            text: Review text to analyse.
            rating: Star rating (1–5) to pass as context, if available.

        Returns:
            Sentiment label: ``"tích cực"``, ``"tiêu cực"``, or ``"trung lập"``.
        """
        prior = self._rating_to_prior(rating)

        # Always skip LLM when budget is exhausted
        if LLMBudget.is_exhausted():
            return prior if prior is not None else LLMBudget._exhausted_label

        # Preserve remaining budget when it drops below 20 %
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
    ) -> str:
        """Predict sentiment using zero-shot classification with budget-aware fallbacks.

        Decision flow (ordered by cost — cheapest first):

        1. Empty text → ``"trung lập"`` (no model call).
        2. Zero-shot model call (local GPU/CPU, no API cost).
           - Exception during inference → ``_fallback_sentiment`` (may use rating prior).
           - No labels returned      → ``_fallback_sentiment``.
        3. **Top score ≥ 0.35 AND gap ≥ 0.08** → accept zero-shot result directly.
        4. **Rating prior available** (rating ≤ 2 or ≥ 4) → use prior without LLM.
        5. **Top score ≥ 0.35 AND gap < 0.08** but no prior → LLM tiebreaker.
        6. **Top score < 0.35** (low confidence) → LLM fallback (budget gated).

        The thresholds (0.35 confidence, 0.08 gap) are deliberately lower than
        the original (0.45 / 0.05) to maximise cases resolved without LLM
        while maintaining label quality.

        Args:
            text: Input review text.
            rating: Optional star rating (1–5) to use as a tiebreaker signal.

        Returns:
            One of ``"tích cực"``, ``"tiêu cực"``, or ``"trung lập"``.
        """
        if not text:
            return "trung lập"

        prior = self._rating_to_prior(rating)

        # Short-circuit: budget exhausted → rating prior or neutral
        if LLMBudget.is_exhausted():
            return prior if prior is not None else "trung lập"

        candidate_labels = ["tích cực", "tiêu cực", "trung lập"]
        try:
            result = self.zero_shot(text, candidate_labels=candidate_labels)
        except Exception:
            # Model error → use rating prior before touching LLM
            if prior is not None:
                return prior
            return self._fallback_sentiment(text, rating)

        result_labels: List[str] = result.get("labels", [])
        result_scores: List[float] = result.get("scores", [])

        if not result_labels or not result_scores:
            if prior is not None:
                return prior
            return self._fallback_sentiment(text, rating)

        top_label: str = result_labels[0]
        top_score: float = float(result_scores[0])
        score_gap: float = (
            top_score - float(result_scores[1])
            if len(result_scores) > 1
            else 1.0
        )

        # High confidence zero-shot result — accept without LLM
        if top_score >= 0.35 and score_gap >= 0.08:
            return top_label

        # Moderate confidence but near-tie OR low confidence:
        # prefer rating prior to avoid LLM call
        if prior is not None:
            return prior

        # No prior available — LLM is the only remaining option
        return self._fallback_sentiment(text, rating)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_review(
        self,
        text: str,
        rating: Optional[int] = None,
        source: DataSource = DataSource.ALL_REVIEWS,
    ) -> Dict[str, Any]:
        """Analyze a review and return a structured result.

        The heuristic label is attempted first.  Only when it returns
        ``"ambiguous"`` does the zero-shot / LLM pipeline execute, keeping
        inference costs proportional to actual label uncertainty.

        Args:
            text: Input review text (should be pre-cleaned).
            rating: Star rating (1–5) accompanying the review, if available.
            source: The :class:`DataSource` origin of the row, used to tune
                    heuristic confidence thresholds.

        Returns:
            Dict with keys:

            * ``"aspects"``  – ``List[str]`` of detected aspect keys.
            * ``"sentiment"`` – ``str`` sentiment label.
            * ``"method"``   – ``"heuristic"`` or ``"model"`` indicating which
              layer produced the final label.
        """
        # Build a row dict so assign_heuristic_label can read both fields.
        row: Dict[str, Any] = {"cleaned_text": text, "rating": rating or 0}
        heuristic = assign_heuristic_label(row, source=source)

        if heuristic != "ambiguous":
            aspects = self.extract_aspects(text)
            return {
                "aspects":   aspects,
                "sentiment": heuristic,
                "method":    "heuristic",
            }

        # Ambiguous → full model pipeline
        aspects = self.extract_aspects(text)
        sentiment = self.predict_sentiment(text, rating=rating)
        return {
            "aspects":   aspects,
            "sentiment": sentiment,
            "method":    "model",
        }
