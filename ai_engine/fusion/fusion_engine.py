"""
Cross-Modal Fusion Engine v2.0.

Calculates a Trust Score (0-100) by fusing signals from:
- Text Modality (PhoBERT sentiment probabilities)
- Image Modality (ResNet defect/no_defect probabilities)
- Image Meta (CLIP relevance boolean)
- Authenticity Meta (Spam detection boolean)
"""

import logging
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# --- Type Constraints ---
ReasonCode = Literal[
    "SPAM_DETECTED",
    "MISSING_IMAGE",
    "IRRELEVANT_IMAGE",
    "MULTIMODAL_CONFLICT",
    "HIGH_TRUST",
    "MODERATE_TRUST",
    "LOW_TRUST",
    "MULTIMODAL_OK"
]


# --- Input Models ---
class TextProbs(BaseModel):
    positive: float = Field(ge=0.0, le=1.0)
    negative: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)

    @model_validator(mode='after')
    def check_sum(self) -> 'TextProbs':
        total = self.positive + self.negative + self.neutral
        # Allow small epsilon for floating point inaccuracies
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Text probabilities must sum to 1.0, got {total}")
        return self


class ImageProbs(BaseModel):
    defect: float = Field(ge=0.0, le=1.0)
    no_defect: float = Field(ge=0.0, le=1.0)

    @model_validator(mode='after')
    def check_sum(self) -> 'ImageProbs':
        total = self.defect + self.no_defect
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Image probabilities must sum to 1.0, got {total}")
        return self


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
class FusionResult(BaseModel):
    """Consistent Pydantic output structure for the Fusion Engine."""
    final_score: float = Field(ge=0.0, le=100.0)
    is_conflict: bool
    flags: List[str]
    reason_code: ReasonCode


class TrustScoreCalculator:
    """
    Fuses multiple AI signals to produce a unified Trust Score.
    """
    
    # Configuration constants to replace magic numbers
    SPAM_PENALTY_SCORE = 5.0
    CONFLICT_TEXT_POS_THRESHOLD = 0.6
    CONFLICT_IMG_DEFECT_THRESHOLD = 0.6
    CONFLICT_PENALTY_MULTIPLIER = 0.5
    NOTICE_TEXT_NEG_THRESHOLD = 0.6
    NOTICE_IMG_PERFECT_THRESHOLD = 0.8
    NEUTRAL_SCORE_MULTIPLIER = 50.0
    HIGH_TRUST_THRESHOLD = 80.0
    MODERATE_TRUST_THRESHOLD = 50.0

    def __init__(
        self, 
        base_text_weight: float = 0.4, 
        base_image_weight: float = 0.4, 
        base_auth_weight: float = 0.2
    ) -> None:
        total_weight = base_text_weight + base_image_weight + base_auth_weight
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Base weights must sum to 1.0, got {total_weight}")

        self.base_text_weight = base_text_weight
        self.base_image_weight = base_image_weight
        self.base_auth_weight = base_auth_weight

    def calculate(self, inputs: FusionInput) -> FusionResult:
        # Step 1: The Gatekeeper (Absolute risk filter)
        if inputs.auth_meta.is_spam:
            logger.warning("Spam detected! Applying severe penalty.")
            return FusionResult(
                final_score=self.SPAM_PENALTY_SCORE,
                is_conflict=False,
                flags=["RISK: Fraudulent Review (Spam/Seeding)"],
                reason_code="SPAM_DETECTED"
            )

        flags = []
        is_conflict = False

        # Step 2: Image Modality Routing & Dynamic Weighting
        if inputs.image_probs is None:
            logger.info("Image modality missing. Redistributing weight to text.")
            weight_text = self.base_text_weight + self.base_image_weight
            weight_image = 0.0
            image_score = 0.0
            reason_code = "MISSING_IMAGE"
            use_image = False
            
        elif inputs.image_meta is not None and inputs.image_meta.is_irrelevant:
            logger.info("Irrelevant image detected. Redistributing weight to text.")
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

        # Step 3: Base Trust Score Calculation
        score_text = (inputs.text_probs.positive * 100.0) + (inputs.text_probs.neutral * self.NEUTRAL_SCORE_MULTIPLIER)
        score_auth = 100.0  # Safe assumption: is_spam is False at this point

        final_score = (
            (score_text * weight_text) + 
            (image_score * weight_image) + 
            (score_auth * self.base_auth_weight)
        )

        # Step 4: Conflict Detection (Evaluated only if relevant image is present)
        if use_image:
            # Conflict 1: Positive Text - Defective Image
            if (inputs.text_probs.positive > self.CONFLICT_TEXT_POS_THRESHOLD and 
                inputs.image_probs.defect > self.CONFLICT_IMG_DEFECT_THRESHOLD):
                
                logger.warning("Multimodal conflict: Positive text but defective image.")
                is_conflict = True
                final_score *= self.CONFLICT_PENALTY_MULTIPLIER
                flags.append("CONFLICT: Suspicious praise due to defective product image")
                reason_code = "MULTIMODAL_CONFLICT"
            
            # Conflict 2: Negative Text - No-Defect Image
            elif (inputs.text_probs.negative > self.NOTICE_TEXT_NEG_THRESHOLD and 
                  inputs.image_probs.no_defect > self.NOTICE_IMG_PERFECT_THRESHOLD):
                
                logger.warning("Multimodal notice: Negative text but perfect image.")
                is_conflict = True
                flags.append("NOTICE: Customer complaint but no visible product defects")
                reason_code = "MULTIMODAL_CONFLICT"

        # Step 5: Final Score Clamping & Trust Level Assignment
        final_score = max(0.0, min(round(final_score, 2), 100.0))

        if not is_conflict and use_image:
            if final_score >= self.HIGH_TRUST_THRESHOLD:
                reason_code = "HIGH_TRUST"
            elif final_score >= self.MODERATE_TRUST_THRESHOLD:
                reason_code = "MODERATE_TRUST"
            else:
                reason_code = "LOW_TRUST"

        return FusionResult(
            final_score=final_score,
            is_conflict=is_conflict,
            flags=flags,
            reason_code=reason_code
        )