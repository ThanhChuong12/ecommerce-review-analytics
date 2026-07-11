"""
feature_denoiser.py
====================
Multimodal Feature Denoiser — Adapted from MDSBR (RecSys'25).

Paper: "MDSBR: Multimodal Denoising for Session-based Recommendation"
Source: https://github.com/YutongLi2024/MDSBR

Core idea: Features extracted from pre-trained models (PhoBERT, CLIP,
MobileNetV3) are inherently noisy due to label errors, task mismatch,
and over-inclusion of irrelevant content. This module uses diffusion-based
denoising to progressively refine those features.

Adaptation for ecommerce-review-analytics:
  - MDSBR's session-based recommendation → Review-level feature denoising
  - Interest-Guided Denoising → Task-Guided Gating (sentiment/defect task signal)
  - Multimodal Alignment → Text ↔ Image alignment via InfoNCE + distribution matching

Components (adapted from MDSBR):
  1. DenoisingMLP           ← model_in_diffusion.py::DNN
  2. GaussianDiffusionDenoiser ← diffusion_new.py::GaussianDiffusion
  3. TaskGuidedGating       ← sessionG_diff.py::gate_v, gate_t
  4. MultimodalAlignmentLayer ← sessionG_diff.py::align_vt + loss.py::InfoNCE
  5. FeatureDenoiser        ← Tich hop tat ca cac component tren
"""

from __future__ import annotations

import enum
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — TIMESTEP EMBEDDING
#  Source: model_in_diffusion.py::timestep_embedding (MDSBR)
# ════════════════════════════════════════════════════════════════════════════

def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Encodes diffusion timestep ``t`` into a continuous vector so the
    denoising network knows *how much* noise to remove.

    Args:
        timesteps: 1-D tensor of N timestep indices (one per batch element).
        dim: Dimension of the output embedding.
        max_period: Controls the minimum frequency of the embeddings.

    Returns:
        Tensor of shape ``(N, dim)`` with positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DENOISING MLP (Reverse Diffusion Backbone)
#  Source: model_in_diffusion.py::DNN (MDSBR)
#  Adaptation: Simplified for feature-vector denoising (removed UNet/ResNet)
# ════════════════════════════════════════════════════════════════════════════

class DenoisingMLP(nn.Module):
    """MLP backbone for the reverse diffusion process.

    Takes a noisy feature vector concatenated with a timestep embedding
    and predicts either the clean signal (x₀) or the noise (ε).

    Architecture::

        [noisy_feature ‖ time_emb] → Linear → Tanh → Linear → Tanh → Project → output

    Adapted from ``DNN`` in MDSBR's ``model_in_diffusion.py``.
    """

    def __init__(
        self,
        feature_dim: int,
        time_emb_dim: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.time_emb_dim = time_emb_dim

        # Timestep embedding projection
        self.emb_layer = nn.Linear(time_emb_dim, time_emb_dim)

        # Encoder: [feature + time_emb] → hidden
        in_dim = feature_dim + time_emb_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Decoder: hidden → feature_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization (following MDSBR convention)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.normal_(module.bias, std=1e-3)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Noisy feature tensor ``(batch, feature_dim)``.
            timesteps: Timestep indices ``(batch,)``.

        Returns:
            Predicted clean feature or noise ``(batch, feature_dim)``.
        """
        # Encode timestep
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)

        # Dropout on input (regularization during training)
        x = self.drop(x)

        # Concat feature + time embedding
        h = torch.cat([x, emb], dim=-1)

        # Encode → Decode
        h = self.encoder(h)
        h = self.decoder(h)
        return h


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — GAUSSIAN DIFFUSION DENOISER
#  Source: diffusion_new.py::GaussianDiffusion (MDSBR)
#  Adaptation: Removed session dependency, standalone feature denoiser
# ════════════════════════════════════════════════════════════════════════════

class ModelMeanType(enum.Enum):
    """What the denoising model predicts."""
    START_X = enum.auto()   # Model predicts x₀ (clean signal)
    EPSILON = enum.auto()   # Model predicts ε (noise)


def _betas_for_alpha_bar(
    num_steps: int,
    alpha_bar_fn,
    max_beta: float = 0.999,
) -> np.ndarray:
    """Create a cosine beta schedule (Nichol & Dhariwal, 2021)."""
    betas = []
    for i in range(num_steps):
        t1 = i / num_steps
        t2 = (i + 1) / num_steps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return np.array(betas)


class GaussianDiffusionDenoiser(nn.Module):
    """Gaussian Diffusion process for feature denoising.

    Implements the forward (noise addition) and reverse (denoising) diffusion
    processes. During training, noise is added to clean features and the model
    learns to remove it. At inference, the reverse process progressively
    denoises the features.

    Adapted from ``GaussianDiffusion`` in MDSBR's ``diffusion_new.py``.

    Args:
        feature_dim: Dimension of input feature vectors.
        noise_steps: Number of diffusion timesteps (T).
        noise_schedule: Beta schedule type (``"linear"`` or ``"cosine"``).
        noise_scale: Scaling factor for noise levels.
        noise_min: Minimum noise level (for linear schedule).
        noise_max: Maximum noise level (for linear schedule).
        mean_type: What the model predicts (``"x0"`` or ``"eps"``).
        hidden_dim: Hidden dimension of the denoising MLP.
    """

    def __init__(
        self,
        feature_dim: int,
        noise_steps: int = 5,
        noise_schedule: str = "linear",
        noise_scale: float = 0.001,
        noise_min: float = 0.0005,
        noise_max: float = 0.005,
        mean_type: str = "x0",
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.noise_steps = noise_steps
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.mean_type = (
            ModelMeanType.START_X if mean_type == "x0" else ModelMeanType.EPSILON
        )

        # Build the denoising backbone
        self.model = DenoisingMLP(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        )

        # Compute noise schedule
        betas = self._get_betas()
        betas = torch.tensor(betas, dtype=torch.float64)
        # Fix first beta to small value (MDSBR convention)
        betas[0] = 1e-5

        # Pre-compute diffusion constants
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float64), alphas_cumprod[:-1]]
        )

        # Register as buffers (moved to device with model, not trained)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1),
        )

        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(
                torch.cat([posterior_variance[1:2], posterior_variance[1:]])
            ),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod),
        )

    def _get_betas(self) -> np.ndarray:
        """Compute noise schedule betas."""
        if self.noise_schedule == "linear":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            return np.linspace(start, end, self.noise_steps, dtype=np.float64)
        elif self.noise_schedule == "cosine":
            return _betas_for_alpha_bar(
                self.noise_steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

    def _extract(
        self,
        arr: torch.Tensor,
        timesteps: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> torch.Tensor:
        """Extract values from a 1-D array for a batch of timestep indices."""
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    # ── Forward diffusion: q(x_t | x_0) ────────────────────────────────────

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add noise to clean features (forward diffusion).

        Implements: x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε

        Args:
            x_start: Clean features ``(batch, feature_dim)``.
            t: Timestep indices ``(batch,)``.
            noise: Optional pre-generated noise.

        Returns:
            Noisy features ``(batch, feature_dim)``.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # ── Reverse diffusion: p(x_{t-1} | x_t) ───────────────────────────────

    def _predict_xstart_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """Recover x_0 from x_t and predicted noise."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _p_mean_variance(
        self, x_t: torch.Tensor, t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute mean and variance of p(x_{t-1} | x_t)."""
        model_output = self.model(x_t, t)

        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        else:  # EPSILON
            pred_xstart = self._predict_xstart_from_eps(x_t, t, model_output)

        # Posterior mean
        model_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * pred_xstart
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        model_variance = self._extract(self.posterior_variance, t, x_t.shape)
        model_log_variance = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(
        self,
        x_start: torch.Tensor,
        sampling_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Denoise features via reverse diffusion.

        Args:
            x_start: Input features to denoise ``(batch, feature_dim)``.
            sampling_steps: Number of reverse steps (default: all steps).

        Returns:
            Denoised features ``(batch, feature_dim)``.
        """
        steps = sampling_steps if sampling_steps is not None else self.noise_steps
        steps = min(steps, self.noise_steps)

        if steps == 0:
            return x_start

        # Add noise to the starting point
        t = torch.tensor(
            [steps - 1] * x_start.shape[0], device=x_start.device
        )
        x_t = self.q_sample(x_start, t)

        # Reverse iterate
        for i in reversed(range(steps)):
            t_batch = torch.tensor(
                [i] * x_t.shape[0], device=x_t.device
            )
            out = self._p_mean_variance(x_t, t_batch)
            x_t = out["mean"]  # Deterministic sampling (no extra noise)

        return x_t

    # ── Training loss ──────────────────────────────────────────────────────

    def training_losses(
        self, x_start: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute diffusion training loss.

        Adds noise at a random timestep, then asks the model to predict
        either the clean signal (x₀) or the noise (ε).

        Args:
            x_start: Clean features ``(batch, feature_dim)``.

        Returns:
            Dict with ``"loss"`` key containing per-sample MSE loss.
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        # Sample random timesteps
        t = torch.randint(0, self.noise_steps, (batch_size,), device=device)
        noise = torch.randn_like(x_start)

        # Forward diffusion
        x_t = self.q_sample(x_start, t, noise)

        # Model prediction
        model_output = self.model(x_t, t)

        # Target depends on mean_type
        if self.mean_type == ModelMeanType.START_X:
            target = x_start
        else:  # EPSILON
            target = noise

        # Per-sample MSE
        loss = F.mse_loss(model_output, target, reduction="none")
        loss = loss.mean(dim=-1)  # Mean over feature dimension

        return {"loss": loss}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience forward: denoise the input features."""
        return self.p_sample(x)


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — TASK-GUIDED GATING
#  Source: sessionG_diff.py::gate_v, gate_t (MDSBR)
#  Adaptation: User interest → Task signal (sentiment / defect detection)
# ════════════════════════════════════════════════════════════════════════════

class TaskGuidedGating(nn.Module):
    """Gating mechanism that filters features based on task context.

    In MDSBR, this was "Interest-Guided Denoising" where user interest
    acted as a gate. Here we replace user interest with a **task signal**
    — the overall review context that guides which features are relevant
    for the downstream task (sentiment analysis or defect detection).

    Gate operation::

        gate = σ(W · task_signal + b)
        output = gate ⊙ feature

    Adapted from ``gate_v`` and ``gate_t`` in MDSBR's ``sessionG_diff.py``.

    Args:
        feature_dim: Dimension of the feature vector.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        feature: torch.Tensor,
        task_signal: torch.Tensor,
    ) -> torch.Tensor:
        """Apply task-guided gating.

        Args:
            feature: Feature tensor ``(batch, feature_dim)``.
            task_signal: Task context tensor ``(batch, feature_dim)``.
                Typically the mean-pooled multimodal representation.

        Returns:
            Gated feature ``(batch, feature_dim)``.
        """
        gate_weights = self.gate(task_signal)
        return gate_weights * feature


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — MULTIMODAL ALIGNMENT LAYER
#  Source: sessionG_diff.py::align_vt + loss.py::InfoNCE, MatchLoss (MDSBR)
#  Adaptation: Kept distribution matching + InfoNCE contrastive alignment
# ════════════════════════════════════════════════════════════════════════════

class MultimodalAlignmentLayer(nn.Module):
    """Cross-modal alignment layer for text and image features.

    Enforces coherence between text and image modalities through two
    complementary losses:

    1. **Distribution Matching** — Aligns mean and variance of text and
       image feature distributions. (From ``align_vt`` in ``sessionG_diff.py``)

    2. **Contrastive Alignment (InfoNCE)** — Pulls matched text-image pairs
       closer and pushes unmatched pairs apart. (From ``InfoNCE`` in ``loss.py``)

    Args:
        feature_dim: Common feature dimension after projection.
        temperature: Temperature for InfoNCE softmax.
    """

    def __init__(
        self,
        feature_dim: int,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature

    def distribution_matching_loss(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """Align distributions by matching mean and variance.

        Adapted from ``align_vt()`` in MDSBR's ``sessionG_diff.py``:
        ``vt_loss = (|var₁ - var₂| + |mean₁ - mean₂|).mean()``

        Args:
            text_features: ``(batch, feature_dim)``
            image_features: ``(batch, feature_dim)``

        Returns:
            Scalar distribution matching loss.
        """
        text_var, text_mean = torch.var(text_features), torch.mean(text_features)
        img_var, img_mean = torch.var(image_features), torch.mean(image_features)

        loss = (torch.abs(text_var - img_var) + torch.abs(text_mean - img_mean))
        return loss

    def infonce_loss(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE contrastive loss for cross-modal alignment.

        Adapted from ``InfoNCE`` class in MDSBR's ``loss.py``.
        Uses standard cross-entropy formulation for numerical stability.
        Assumes i-th sample in features1 is paired with i-th in features2.

        Args:
            features1: Features ``(batch, feature_dim)``
            features2: Features ``(batch, feature_dim)``

        Returns:
            Scalar InfoNCE loss.
        """
        # L2 normalize
        f1 = F.normalize(features1, dim=1)
        f2 = F.normalize(features2, dim=1)

        # Scaled similarity matrix (logits)
        logits = torch.matmul(f1, f2.T) / self.temperature

        # Labels: positive pairs are on the diagonal (i-th matches i-th)
        labels = torch.arange(len(f1), device=f1.device)

        # Standard cross-entropy handles log-sum-exp internally (numerically stable)
        loss = F.cross_entropy(logits, labels)

        return loss

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        alignment_weight: float = 0.5,
    ) -> torch.Tensor:
        """Compute combined alignment loss.

        Args:
            text_features: ``(batch, feature_dim)``
            image_features: ``(batch, feature_dim)``
            alignment_weight: Balance between distribution matching and InfoNCE.
                0 = only distribution matching, 1 = only InfoNCE.

        Returns:
            Combined alignment loss (scalar).
        """
        dist_loss = self.distribution_matching_loss(text_features, image_features)
        contrastive_loss = self.infonce_loss(text_features, image_features)

        return (1 - alignment_weight) * dist_loss + alignment_weight * contrastive_loss


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — FEATURE DENOISER (Main Integrated Module)
#  Source: Combines all MDSBR components into a single module
# ════════════════════════════════════════════════════════════════════════════

class FeatureDenoiser(nn.Module):
    """Multimodal Feature Denoiser for ecommerce review analytics.

    Integrates all MDSBR-inspired components into a single module that:
    1. Projects text/image features to a common hidden space
    2. Applies Gaussian Diffusion denoising to each modality
    3. Filters features through Task-Guided Gating
    4. Optionally aligns cross-modal features

    Example::

        denoiser = FeatureDenoiser(text_dim=768, image_dim=2048, hidden_dim=512)

        # Inference: denoise features
        clean_text, clean_image = denoiser.denoise_multimodal(text_emb, image_emb)

        # Training: compute total loss
        loss_dict = denoiser.training_step(text_emb, image_emb)
        loss_dict["total_loss"].backward()

    Args:
        text_dim: Dimension of text embeddings (768 for PhoBERT).
        image_dim: Dimension of image embeddings (2048 for ResNet50).
        hidden_dim: Common hidden dimension for denoising.
        noise_steps: Number of diffusion timesteps.
        noise_schedule: Beta schedule (``"linear"`` or ``"cosine"``).
        noise_scale: Noise scaling factor.
        diffusion_weight: Weight of diffusion loss in total loss.
        alignment_weight: Weight of alignment loss in total loss.
        reconstruction_weight: Weight of reconstruction loss.
    """

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 2048,
        hidden_dim: int = 512,
        noise_steps: int = 5,
        noise_schedule: str = "cosine",
        noise_scale: float = 0.001,
        diffusion_weight: float = 1.0,
        alignment_weight: float = 0.01,
        reconstruction_weight: float = 0.5,
    ) -> None:
        super().__init__()

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.diffusion_weight = diffusion_weight
        self.alignment_weight = alignment_weight
        self.reconstruction_weight = reconstruction_weight

        # ── Projection layers (map to common space) ────────────────────────
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ── Diffusion denoisers (one per modality) ─────────────────────────
        self.text_diffusion = GaussianDiffusionDenoiser(
            feature_dim=hidden_dim,
            noise_steps=noise_steps,
            noise_schedule=noise_schedule,
            noise_scale=noise_scale,
            hidden_dim=hidden_dim,
        )
        self.image_diffusion = GaussianDiffusionDenoiser(
            feature_dim=hidden_dim,
            noise_steps=noise_steps,
            noise_schedule=noise_schedule,
            noise_scale=noise_scale,
            hidden_dim=hidden_dim,
        )

        # ── Task-Guided Gating ─────────────────────────────────────────────
        self.text_gate = TaskGuidedGating(hidden_dim)
        self.image_gate = TaskGuidedGating(hidden_dim)

        # ── Common-Specific Decomposition (from MDSBR forward()) ───────────
        # Attention to compute common component from gated features
        self.query_common = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        # ── Multimodal Alignment ───────────────────────────────────────────
        self.alignment = MultimodalAlignmentLayer(
            feature_dim=hidden_dim,
        )

        # ── Output projections (back to original dims) ─────────────────────
        self.text_output = nn.Linear(hidden_dim, text_dim)
        self.image_output = nn.Linear(hidden_dim, image_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection layers."""
        for module in [self.text_projection, self.image_projection,
                       self.text_output, self.image_output, self.query_common]:
            for sub in module.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.xavier_normal_(sub.weight)
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)

    # ── Single-modality denoising ──────────────────────────────────────────

    def denoise_text(self, text_features: torch.Tensor) -> torch.Tensor:
        """Denoise text features only (no cross-modal alignment).

        Args:
            text_features: ``(batch, text_dim)`` — e.g. PhoBERT embeddings.

        Returns:
            Denoised text features ``(batch, text_dim)``.
        """
        projected = self.text_projection(text_features)
        denoised = self.text_diffusion.p_sample(projected)
        # Skip connection: add residual
        denoised = denoised + projected
        return self.text_output(denoised)

    def denoise_image(self, image_features: torch.Tensor) -> torch.Tensor:
        """Denoise image features only (no cross-modal alignment).

        Args:
            image_features: ``(batch, image_dim)`` — e.g. MobileNetV3 embeddings.

        Returns:
            Denoised image features ``(batch, image_dim)``.
        """
        projected = self.image_projection(image_features)
        denoised = self.image_diffusion.p_sample(projected)
        denoised = denoised + projected
        return self.image_output(denoised)

    # ── Multimodal denoising ───────────────────────────────────────────────

    def denoise_multimodal(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Denoise both modalities with cross-modal gating and alignment.

        Pipeline (adapted from MDSBR ``sessionG_diff.py`` forward()):
        1. Project to common space
        2. Diffusion denoise each modality
        3. Task-guided gating (using mean of both modalities as task signal)
        4. Common-specific decomposition
        5. Fuse and project back

        Args:
            text_features: ``(batch, text_dim)``
            image_features: ``(batch, image_dim)``

        Returns:
            Tuple of (denoised_text, denoised_image) in original dimensions.
        """
        # Step 1: Project to common hidden space
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)

        # Step 2: Diffusion denoise
        text_denoised = self.text_diffusion.p_sample(text_proj)
        image_denoised = self.image_diffusion.p_sample(image_proj)

        # Step 3: Task-guided gating
        # Task signal = mean of original projections (represents overall review)
        task_signal = (text_proj + image_proj) / 2.0
        text_gated = self.text_gate(text_denoised, task_signal)
        image_gated = self.image_gate(image_denoised, task_signal)

        # Step 4: Common-specific decomposition (from MDSBR forward())
        # Compute attention weights for common component
        common_weights = torch.cat([
            self.query_common(text_gated),
            self.query_common(image_gated),
        ], dim=-1)
        common_weights = F.softmax(common_weights, dim=-1)

        # Common component = weighted sum of gated features
        common = (
            common_weights[:, 0:1] * text_gated
            + common_weights[:, 1:2] * image_gated
        )

        # Specific components
        text_specific = text_gated - common
        image_specific = image_gated - common

        # Fuse: specific + common + residual
        text_fused = text_specific + common + text_proj
        image_fused = image_specific + common + image_proj

        # Step 5: Project back to original dimensions
        text_out = self.text_output(text_fused)
        image_out = self.image_output(image_fused)

        return text_out, image_out

    # ── Training ───────────────────────────────────────────────────────────

    def training_step(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses for the denoiser.

        Loss components:
        - ``diffusion_loss``: MSE denoising loss (text + image)
        - ``alignment_loss``: Cross-modal alignment loss
        - ``total_loss``: Weighted sum

        Args:
            text_features: ``(batch, text_dim)``
            image_features: ``(batch, image_dim)``

        Returns:
            Dict with ``total_loss``, ``diffusion_loss``, ``alignment_loss``.
        """
        # Project to common space
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)

        # Diffusion losses
        text_diff_loss = self.text_diffusion.training_losses(text_proj)["loss"].mean()
        image_diff_loss = self.image_diffusion.training_losses(image_proj)["loss"].mean()
        diffusion_loss = (text_diff_loss + image_diff_loss) / 2.0

        # Alignment loss
        alignment_loss = self.alignment(text_proj, image_proj)

        # Reconstruction loss — ensure denoised output preserves original signal
        text_denoised, image_denoised = self.denoise_multimodal(text_features, image_features)
        recon_loss = (
            F.mse_loss(text_denoised, text_features)
            + F.mse_loss(image_denoised, image_features)
        ) / 2.0

        # Total loss
        total_loss = (
            self.diffusion_weight * diffusion_loss
            + self.alignment_weight * alignment_loss
            + self.reconstruction_weight * recon_loss
        )

        return {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "alignment_loss": alignment_loss,
            "reconstruction_loss": recon_loss,
            "text_diffusion_loss": text_diff_loss,
            "image_diffusion_loss": image_diff_loss,
        }

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass — denoise features.

        Args:
            text_features: ``(batch, text_dim)``
            image_features: Optional ``(batch, image_dim)``

        Returns:
            Tuple of (denoised_text, denoised_image).
            ``denoised_image`` is None when ``image_features`` is None.
        """
        if image_features is not None:
            return self.denoise_multimodal(text_features, image_features)
        else:
            return self.denoise_text(text_features), None
