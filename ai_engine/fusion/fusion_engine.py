"""
Cross-Modal Fusion Engine v2.0.

Calculates a Trust Score (0-100) by fusing signals from:
- Text Modality (PhoBERT sentiment probabilities)
- Image Modality (ResNet defect/no_defect probabilities)
- Image Meta (CLIP relevance boolean)
- Authenticity Meta (Spam detection boolean)

Key Upgrades:
- Pydantic Input Normalization: Enforces structured type validation.
- Dynamic Weighting: Handles irrelevant or missing images gracefully by redistributing weights.
- Spam Penalty: Instantly severely penalizes the trust score if spam is detected.
- Conflict Resolution: Flags reviews with positive text but defective image, or negative text but perfect image.
- Flags Output: Returns list of strings for Frontend notification/rendering.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- Input Models ---

class TextProbs(BaseModel):
    positive: float
    negative: float
    neutral: float


class ImageProbs(BaseModel):
    defect: float
    no_defect: float


class ImageMeta(BaseModel):
    is_irrelevant: bool


class AuthMeta(BaseModel):
    is_spam: bool


class FusionInput(BaseModel):
    text_probs: TextProbs
    image_probs: Optional[ImageProbs] = None
    image_meta: Optional[ImageMeta] = None
    auth_meta: AuthMeta


# --- Output Structure ---

@dataclass
class FusionResult:
    """Typed output structure for the Cross-Modal Fusion Engine."""
    final_score: float
    is_conflict: bool
    flags: List[str]
    reason_code: str  # Kept for backward compatibility with LLM Recommendation client


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

    def calculate(self, inputs: FusionInput) -> FusionResult:
        """
        Calculate the multimodal trust score.

        Args:
            inputs: A FusionInput containing text probabilities, image probabilities,
                    image metadata, and authenticity metadata.

        Returns:
            A FusionResult containing the final score, conflict flag, flags list, and reasoning.
        """
        # Step 1: The Gatekeeper (Absolute risk filter)
        if inputs.auth_meta.is_spam:
            logger.warning("Spam detected! Applying severe penalty.")
            return FusionResult(
                final_score=5.0, 
                is_conflict=False, 
                flags=["RISK: Fraudulent Review (Spam/Seeding)"],
                reason_code="SPAM_DETECTED"
            )

        # Step 2: CLIP Intervention & Dynamic Weighting
        flags = []
        is_conflict = False

        if inputs.image_probs is None:
            logger.info("Image modality missing. Redistributing image weight to text.")
            weight_text = self.base_text_weight + self.base_image_weight
            weight_image = 0.0
            image_score = 0.0
            reason_code = "MISSING_IMAGE"
            use_image = False
        elif inputs.image_meta is not None and inputs.image_meta.is_irrelevant:
            logger.info("Irrelevant image detected. Ignoring image probabilities and redistributing image weight to text.")
            weight_text = self.base_text_weight + self.base_image_weight
            weight_image = 0.0
            image_score = 0.0
            flags.append("WARNING: Irrelevant Product Image")
            reason_code = "IRRELEVANT_IMAGE"
            use_image = False
        else:
            weight_text = self.base_text_weight
            weight_image = self.base_image_weight
            image_score = inputs.image_probs.no_defect * 100.0
            reason_code = "MULTIMODAL_OK"
            use_image = True

        # Step 4: Trust Score Calculation - Text and Authenticity
        score_text = (inputs.text_probs.positive * 100.0) + (inputs.text_probs.neutral * 50.0)
        score_auth = 100.0  # Since is_spam is False

        # Weighted sum of components
        final_score = (
            (score_text * weight_text) + 
            (image_score * weight_image) + 
            (score_auth * self.base_auth_weight)
        )

        # Step 3: Conflict Detection (Only when relevant image is present)
        if use_image and inputs.image_probs is not None:
            # Conflict 1: Positive Text - Defective Image
            if inputs.text_probs.positive > 0.6 and inputs.image_probs.defect > 0.6:
                logger.warning("Multimodal conflict! Positive text but defect image.")
                is_conflict = True
                final_score *= 0.5  # Apply 50% penalty
                flags.append("CONFLICT: Suspicious praise due to defective product image")
                reason_code = "MULTIMODAL_CONFLICT"
            
            # Conflict 2: Negative Text - No-Defect Image
            elif inputs.text_probs.negative > 0.6 and inputs.image_probs.no_defect > 0.8:
                logger.warning("Multimodal notice! Negative text but no-defect image.")
                is_conflict = True
                # No penalty, but attach warning flag
                flags.append("NOTICE: Customer complaint but no visible product defects")
                reason_code = "MULTIMODAL_CONFLICT"

        # Final Score Adjustments and clamp
        final_score = max(0.0, min(round(final_score, 2), 100.0))

        # Adjust reason code for backward compatibility if no conflict
        if not is_conflict and use_image:
            if final_score >= 80.0:
                reason_code = "HIGH_TRUST"
            elif final_score >= 50.0:
                reason_code = "MODERATE_TRUST"
            else:
                reason_code = "LOW_TRUST"

        return FusionResult(
            final_score=final_score,
            is_conflict=is_conflict,
            flags=flags,
            reason_code=reason_code
        )