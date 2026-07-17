"""
Custom HuggingFace Trainer with Focal Loss for Imbalanced Sentiment Data.

This module provides ``FocalLossTrainer``, a subclass of HuggingFace's
``transformers.Trainer`` that replaces the default cross-entropy loss with
a class-weighted Focal Loss.  Focal Loss down-weights easy (well-classified)
examples and focuses learning on hard, misclassified ones, which is
especially beneficial for the severely skewed dataset (≈94 % Positive,
≈5 % Negative, ≈1 % Neutral).

Key design choices:
    * **Dynamic alpha** – class weights are computed from inverse class
      frequencies at runtime if not provided explicitly, eliminating magic
      hard-coded constants.
    * **Device synchronisation** – ``alpha`` and ``labels`` are always moved
      to the same device as the model's logits, preventing CPU/GPU crashes.
    * **Numerical stability** – predicted probabilities are clamped before
      the log to avoid ``log(0)`` producing ``-inf`` in the loss.
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

    Focal Loss was introduced by Lin et al. (2017) for dense object
    detection, but generalises well to any classification problem with class
    imbalance.  For a sample with true class ``c``, the loss is:

        FL = -alpha_c * (1 - p_c)^gamma * log(p_c)

    where ``p_c`` is the model's predicted probability for the true class.

    Args:
        alpha: 1-D tensor of per-class weights of shape ``(num_classes,)``.
            Higher values penalise misclassification of that class more
            heavily.  Will be normalised to sum to ``num_classes`` so the
            effective learning rate scale is preserved.
        gamma: Focusing parameter ≥ 0.  ``gamma=0`` reduces to weighted
            cross-entropy.  ``gamma=2`` is the value recommended in the
            original paper and works well in practice.
        num_classes: Number of output classes.
        eps: Small constant added for numerical stability when clamping
            probabilities before taking ``log``.

    References:
        Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection.
        arXiv:1708.02002.
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
    """HuggingFace Trainer that uses class-weighted Focal Loss.

    Overrides ``compute_loss`` to replace cross-entropy with Focal Loss,
    which is better suited to the ~94/5/1 class imbalance of the Vietnamese
    e-commerce sentiment dataset.

    Alpha weights can be:

    1. **Provided explicitly** – Pass a pre-computed ``alpha`` tensor whose
       values reflect domain knowledge or a separate calibration run.
    2. **Computed from data** – Pass ``class_counts`` (a list/dict of sample
       counts per class) and the trainer will use inverse-frequency weighting
       (``alpha_c = total / (num_classes * count_c)``).
    3. **Both omitted** – Equal weights are used (``alpha = [1, 1, 1]``),
       which degrades to unweighted Focal Loss.

    Args:
        alpha: Optional 1-D ``torch.Tensor`` of shape ``(num_classes,)`` with
            pre-computed class weights.  Takes precedence over
            ``class_counts``.
        class_counts: Optional mapping from class index (int) to sample
            count (int).  Used to compute inverse-frequency alpha when
            ``alpha`` is not given.
        gamma: Focal Loss focusing parameter.  Defaults to 2.0.
        num_classes: Number of sentiment classes.  Defaults to 3.
        **kwargs: All remaining keyword arguments are forwarded to the base
            ``Trainer.__init__``.

    Example::

        counts = {0: 44_490, 1: 2_360, 2: 470}   # rough distribution
        trainer = FocalLossTrainer(
            class_counts=counts,
            gamma=2.0,
            num_classes=3,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
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
        """Determine the alpha tensor from available information.

        Priority:
            1. Explicit ``alpha`` tensor – used as-is.
            2. ``class_counts`` dict – inverse-frequency weighting.
            3. Fallback – uniform weights (ones).

        Args:
            alpha: Optional explicit weights.
            class_counts: Optional {class_idx: count} mapping.
            num_classes: Number of classes (used for shape validation and
                fallback creation).

        Returns:
            A ``torch.FloatTensor`` of shape ``(num_classes,)`` on CPU.

        Raises:
            ValueError: If ``alpha`` has the wrong length.
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
        """Compute Focal Loss for a batch of inputs.

        Replaces HuggingFace's default cross-entropy loss.  The model is
        called with ``labels`` removed from ``inputs`` so we can extract the
        raw logits and compute our custom loss.

        Args:
            model: The sequence classification model.
            inputs: Batch dictionary containing ``input_ids``,
                ``attention_mask``, and ``labels``.
            return_outputs: If ``True``, also return the model's
                ``SequenceClassifierOutput`` alongside the loss scalar.
            **kwargs: Absorbed for forward-compatibility with future
                HuggingFace Trainer signatures.

        Returns:
            ``loss`` if ``return_outputs=False``, else ``(loss, outputs)``.
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
