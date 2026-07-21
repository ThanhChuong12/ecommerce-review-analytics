"""
Custom HuggingFace Trainer with Focal Loss for Imbalanced Sentiment Data.

Provides FocalLossTrainer with class-weighted Focal Loss to handle severely
skewed datasets. Features dynamic alpha computation, device synchronization, 
and numerical stability.
"""

import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focal Loss implementation (pure PyTorch, no external deps)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multi-class Focal Loss with per-class alpha weighting.

    FL = -alpha_c * (1 - p_c)^gamma * log(p_c)

    Args:
        alpha: Per-class weights tensor (normalized to sum to num_classes).
        gamma: Focusing parameter (default: 2.0).
        num_classes: Number of classes.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: float = 2.0,
        num_classes: int = 3,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if alpha.shape[0] != num_classes:
            raise ValueError(
                f"alpha must have length {num_classes}, got {alpha.shape[0]}."
            )
        # Register as buffer so it moves with .to(device) automatically.
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.num_classes = num_classes
        self.eps = eps

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute the mean focal loss over the batch.

        Args:
            logits: Raw model output of shape ``(batch_size, num_classes)``.
            labels: Ground-truth class indices of shape ``(batch_size,)``.

        Returns:
            Scalar loss tensor (mean over the batch).
        """
        # Device sync – both tensors must be on the same device as logits.
        labels = labels.to(logits.device)
        alpha = self.alpha.to(logits.device)

        # Compute class probabilities via softmax.
        probs = F.softmax(logits, dim=-1)  # (B, C)

        # Clamp to [eps, 1-eps] for numerical stability before log.
        probs = probs.clamp(min=self.eps, max=1.0 - self.eps)

        # Gather p_c: probability assigned to the *true* class.
        # Shape: (B, 1) → (B,)
        p_t = probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)

        # Per-sample alpha weight for the true class.
        alpha_t = alpha[labels]  # (B,)

        # Focal modulation factor: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Final focal loss per sample.
        loss = -alpha_t * focal_weight * torch.log(p_t)

        return loss.mean()


# ---------------------------------------------------------------------------
# Trainer subclass
# ---------------------------------------------------------------------------

class FocalLossTrainer(Trainer):
    """Trainer using class-weighted Focal Loss.

    Alpha weights can be explicit, computed from class_counts (inverse-frequency),
    or default to uniform.

    Args:
        alpha: Explicit class weights tensor.
        class_counts: Dict of counts per class (used if alpha is None).
        gamma: Focusing parameter.
        num_classes: Number of classes.
        **kwargs: Passed to Trainer.
    """

    def __init__(
        self,
        *args,
        alpha: Optional[torch.Tensor] = None,
        class_counts: Optional[Dict[int, int]] = None,
        gamma: float = 2.0,
        num_classes: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_classes = num_classes
        self.gamma = gamma

        # Resolve alpha weights.
        resolved_alpha = self._resolve_alpha(alpha, class_counts, num_classes)

        logger.info(
            "FocalLossTrainer initialised with gamma=%.1f and alpha=%s",
            gamma,
            resolved_alpha.tolist(),
        )

        # Instantiate the focal loss module (alpha normalisation happens inside).
        self.focal_loss = FocalLoss(
            alpha=resolved_alpha,
            gamma=gamma,
            num_classes=num_classes,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_alpha(
        alpha: Optional[torch.Tensor],
        class_counts: Optional[Dict[int, int]],
        num_classes: int,
    ) -> torch.Tensor:
        """Resolve alpha tensor: 1. explicit alpha -> 2. inverse counts -> 3. uniform.
        
        Returns: FloatTensor of shape (num_classes,) on CPU.
        """
        if alpha is not None:
            alpha = alpha.float()
            if alpha.shape[0] != num_classes:
                raise ValueError(
                    f"Provided alpha has length {alpha.shape[0]}, "
                    f"but num_classes={num_classes}."
                )
            logger.info("Using explicit alpha: %s", alpha.tolist())
            return alpha

        if class_counts is not None:
            # Use Square Root Smoothing to prevent tiny alpha for majority class
            counts = torch.tensor([class_counts.get(c, 1) for c in range(num_classes)], dtype=torch.float)
            smoothed_counts = torch.sqrt(counts)
            
            # Inverse frequency weighting with square root counts
            weights = 1.0 / smoothed_counts
            
            # Normalize so that weights sum to num_classes
            weights = weights / weights.sum() * num_classes
            
            logger.info(
                "Computed square-root smoothed alpha from class_counts %s → %s",
                class_counts,
                weights.tolist(),
            )
            return weights

        logger.warning(
            "Neither alpha nor class_counts provided. "
            "Using uniform weights – this may not help with class imbalance."
        )
        return torch.ones(num_classes, dtype=torch.float)

    # ------------------------------------------------------------------
    # Override: compute_loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, SequenceClassifierOutput]]:
        """Compute Focal Loss for a batch.

        Pops labels from inputs to extract raw logits, then computes custom loss.
        """
        # Pop labels before forwarding – we compute loss ourselves.
        labels = inputs.pop("labels")

        outputs: SequenceClassifierOutput = model(**inputs)
        logits: torch.Tensor = outputs.logits  # (B, num_classes)

        # Compute focal loss (device sync handled inside FocalLoss.forward).
        loss = self.focal_loss(logits, labels)

        # Restore labels in inputs so HuggingFace's internal bookkeeping
        # (e.g. prediction_loop) still has access to them.
        inputs["labels"] = labels

        return (loss, outputs) if return_outputs else loss
