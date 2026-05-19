"""
Cross-Modal Fusion Engine.

Calculates a Trust Score (0-100) by fusing signals from:
- Text Modality (PhoBERT sentiment probabilities)
- Image Modality (ResNet intact/broken probabilities)
- Authenticity (Spam detection boolean)

Key Upgrades:
- Dynamic Weighting: Handles missing images gracefully by redistributing weights.
- Spam Penalty: Instantly severely penalizes the trust score if spam is detected.
- Conflict Resolution: Flags reviews where text is positive but the image is broken.
- Typed Output: Uses dataclass for structured, type-safe API responses.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Typed output structure for the Cross-Modal Fusion Engine."""
    final_score: float
    is_conflict: bool
    reason_code: str


class TrustScoreCalculator:
    """
    Fuses multiple AI signals to produce a unified Trust Score.
    """

    def __init__(
        self, 
        base_text_weight: float = 0.4, 
        base_image_weight: float = 0.4, 
        base_auth_weight: float = 0.2
    ) -> None:
        """
        Initialize weights for the multimodal components.
        """
        self.base_text_weight = base_text_weight
        self.base_image_weight = base_image_weight
        self.base_auth_weight = base_auth_weight

    def calculate(
        self, 
        text_probs: Dict[str, float], 
        image_probs: Optional[Dict[str, float]], 
        is_spam: bool
    ) -> FusionResult:
        """
        Calculate the multimodal trust score.

        Args:
            text_probs: Dictionary with sentiment keys ('positive', 'negative', 'neutral').
            image_probs: Dictionary with visual keys ('intact', 'broken'). Can be None.
            is_spam: Boolean flag from the anti-spam detection module.

        Returns:
            A FusionResult containing the final score, conflict flag, and reasoning.
        """
        # 1. Spam Penalty (Instant severe penalty)
        if is_spam:
            logger.warning("Spam detected! Applying severe penalty.")
            return FusionResult(
                final_score=5.0, 
                is_conflict=False, 
                reason_code="SPAM_DETECTED"
            )

        # 2. Dynamic Weighting (Handle missing image modality)
        if image_probs is None:
            logger.info("Image modality missing. Redistributing image weight to text.")
            weight_text = self.base_text_weight + self.base_image_weight
            weight_image = 0.0
            image_intact_prob = 0.0
            image_broken_prob = 0.0
            base_reason = "MISSING_IMAGE"
        else:
            weight_text = self.base_text_weight
            weight_image = self.base_image_weight
            image_intact_prob = image_probs.get("intact", 0.0)
            image_broken_prob = image_probs.get("broken", 0.0)
            base_reason = "MULTIMODAL_OK"

        text_positive_prob = text_probs.get("positive", 0.0)
        
        # 3. Calculate Base Components (Scale 0 to 100)
        score_text = text_positive_prob * 100.0
        score_image = image_intact_prob * 100.0
        score_auth = 100.0  # Since is_spam is False

        # Weighted sum of components
        final_score = (
            (score_text * weight_text) + 
            (score_image * weight_image) + 
            (score_auth * self.base_auth_weight)
        )

        # 4. Multimodal Conflict Detection
        is_conflict = False
        reason_code = base_reason

        # Define conflict: Text says the product is great (>60% pos) 
        # but the image clearly shows a broken item (>60% broken)
        if image_probs is not None:
            if text_positive_prob > 0.6 and image_broken_prob > 0.6:
                logger.warning("Multimodal conflict! Positive text but Broken image.")
                is_conflict = True
                reason_code = "MULTIMODAL_CONFLICT"
                final_score *= 0.5  # Apply a 50% penalty for conflicting signals

        # 5. Final Score Adjustments and Reasoning
        final_score = max(0.0, min(round(final_score, 2), 100.0))

        if not is_conflict and image_probs is not None:
            if final_score >= 80.0:
                reason_code = "HIGH_TRUST"
            elif final_score >= 50.0:
                reason_code = "MODERATE_TRUST"
            else:
                reason_code = "LOW_TRUST"

        return FusionResult(
            final_score=final_score,
            is_conflict=is_conflict,
            reason_code=reason_code
        )