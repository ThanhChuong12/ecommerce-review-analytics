"""
ai_engine.denoising
====================
Multimodal Denoising Module — Adapted from MDSBR (RecSys'25).

Provides diffusion-based feature denoising for text and image embeddings
extracted from pre-trained models (PhoBERT, MobileNetV3/ResNet50).
"""

from ai_engine.denoising.feature_denoiser import (
    FeatureDenoiser,
    GaussianDiffusionDenoiser,
    TaskGuidedGating,
    MultimodalAlignmentLayer,
    DenoisingMLP,
)

__all__ = [
    "FeatureDenoiser",
    "GaussianDiffusionDenoiser",
    "TaskGuidedGating",
    "MultimodalAlignmentLayer",
    "DenoisingMLP",
]
