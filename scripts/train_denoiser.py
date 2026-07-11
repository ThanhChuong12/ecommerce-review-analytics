"""
train_denoiser.py
==================
Huan luyen FeatureDenoiser — module lam sach feature embeddings
tu pre-trained models (PhoBERT, MobileNetV3) bang Gaussian Diffusion.

Tu tuong lay tu MDSBR (RecSys'25):
  - Diffusion-based denoising: them noise → hoc khu noise → feature sach hon
  - Task-Guided Gating: loc feature theo task context
  - Multimodal Alignment: can chinh text ↔ image features

Usage:
    # Train voi pre-extracted embeddings
    python scripts/train_denoiser.py \\
        --text-embeddings data/processed/text_embeddings.pt \\
        --image-embeddings data/processed/image_embeddings.pt \\
        --epochs 50 --lr 1e-4 \\
        --save-path ai_engine/models/feature_denoiser.pt

    # Train voi synthetic data (demo/test)
    python scripts/train_denoiser.py --demo --epochs 20

    # Chi denoise text (khong can image)
    python scripts/train_denoiser.py \\
        --text-embeddings data/processed/text_embeddings.pt \\
        --text-only --epochs 30
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Add project root to sys.path ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_engine.denoising.feature_denoiser import FeatureDenoiser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_embeddings(path: str) -> torch.Tensor:
    """Load pre-extracted embeddings from a .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        # Support dict format with "embeddings" key
        data = data.get("embeddings", data.get("features", next(iter(data.values()))))
    return data.float()


def generate_synthetic_data(
    n_samples: int = 1000,
    text_dim: int = 768,
    image_dim: int = 1280,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic embeddings for demo/testing."""
    logger.info(
        "Generating synthetic data: %d samples, text_dim=%d, image_dim=%d",
        n_samples, text_dim, image_dim,
    )
    # Simulate noisy pre-trained features with some structure
    # Clean signal
    text_clean = torch.randn(n_samples, text_dim) * 0.5
    image_clean = torch.randn(n_samples, image_dim) * 0.5

    # Add noise (simulating pre-trained model noise)
    text_noisy = text_clean + torch.randn_like(text_clean) * 0.3
    image_noisy = image_clean + torch.randn_like(image_clean) * 0.3

    return text_noisy, image_noisy


# ════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def train(
    denoiser: FeatureDenoiser,
    text_data: torch.Tensor,
    image_data: torch.Tensor | None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cpu",
    patience: int = 15,
    val_ratio: float = 0.1,
) -> list[dict]:
    """Train the FeatureDenoiser with early stopping.

    Args:
        denoiser: The FeatureDenoiser module.
        text_data: Text embeddings (n_samples, text_dim).
        image_data: Image embeddings (n_samples, image_dim) or None.
        epochs: Maximum number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        device: Device to train on.
        patience: Early stopping patience (epochs without improvement).
        val_ratio: Fraction of data for validation.

    Returns:
        List of per-epoch metric dicts.
    """
    denoiser = denoiser.to(device)

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # ── Train/Val split ────────────────────────────────────────────────────
    n_total = len(text_data)
    n_val = max(int(n_total * val_ratio), batch_size)
    n_train = n_total - n_val

    # Shuffle indices
    perm = torch.randperm(n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    text_train = text_data[train_idx].to(device)
    text_val = text_data[val_idx].to(device)

    if image_data is not None:
        image_train = image_data[train_idx].to(device)
        image_val = image_data[val_idx].to(device)
        train_dataset = TensorDataset(text_train, image_train)
        val_dataset = TensorDataset(text_val, image_val)
    else:
        image_train = None
        image_val = None
        train_dataset = TensorDataset(text_train)
        val_dataset = TensorDataset(text_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    history = []
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    SEP = "=" * 60

    logger.info(SEP)
    logger.info("  TRAINING FeatureDenoiser")
    logger.info(SEP)
    logger.info("  Train      : %d", n_train)
    logger.info("  Val        : %d", n_val)
    logger.info("  Text dim   : %d", text_data.shape[1])
    if image_data is not None:
        logger.info("  Image dim  : %d", image_data.shape[1])
    logger.info("  Hidden dim : %d", denoiser.hidden_dim)
    logger.info("  Epochs     : %d (patience=%d)", epochs, patience)
    logger.info("  Batch size : %d", batch_size)
    logger.info("  LR         : %.2e", lr)
    logger.info("  Device     : %s", device)
    logger.info(SEP)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ── Training ───────────────────────────────────────────────────────
        denoiser.train()
        epoch_losses = {"total_loss": 0.0, "diffusion_loss": 0.0, "alignment_loss": 0.0}
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if image_data is not None:
                text_batch, image_batch = batch
                losses = denoiser.training_step(text_batch, image_batch)
            else:
                (text_batch,) = batch
                text_proj = denoiser.text_projection(text_batch)
                diff_loss = denoiser.text_diffusion.training_losses(text_proj)["loss"].mean()
                losses = {
                    "total_loss": diff_loss,
                    "diffusion_loss": diff_loss,
                    "alignment_loss": torch.tensor(0.0, device=device),
                }

            losses["total_loss"].backward()
            nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            optimizer.step()

            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            n_batches += 1

        scheduler.step()

        for key in epoch_losses:
            epoch_losses[key] /= max(n_batches, 1)

        # ── Validation ─────────────────────────────────────────────────────
        denoiser.eval()
        val_losses = {"total_loss": 0.0, "diffusion_loss": 0.0, "alignment_loss": 0.0}
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if image_data is not None:
                    text_batch, image_batch = batch
                    losses = denoiser.training_step(text_batch, image_batch)
                else:
                    (text_batch,) = batch
                    text_proj = denoiser.text_projection(text_batch)
                    diff_loss = denoiser.text_diffusion.training_losses(text_proj)["loss"].mean()
                    losses = {
                        "total_loss": diff_loss,
                        "diffusion_loss": diff_loss,
                        "alignment_loss": torch.tensor(0.0, device=device),
                    }

                for key in val_losses:
                    val_losses[key] += losses[key].item()
                n_val_batches += 1

        for key in val_losses:
            val_losses[key] /= max(n_val_batches, 1)

        # ── Early stopping check ───────────────────────────────────────────
        val_total = val_losses["total_loss"]
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_state_dict = {k: v.clone() for k, v in denoiser.state_dict().items()}
            epochs_without_improvement = 0
            marker = " ★"
        else:
            epochs_without_improvement += 1
            marker = ""

        epoch_losses["val_total_loss"] = val_total
        epoch_losses["val_diff_loss"] = val_losses["diffusion_loss"]
        epoch_losses["val_align_loss"] = val_losses["alignment_loss"]
        epoch_losses["epoch"] = epoch
        epoch_losses["lr"] = scheduler.get_last_lr()[0]
        history.append(epoch_losses)

        # Log progress
        if epoch % max(1, epochs // 20) == 0 or epoch == 1 or marker:
            logger.info(
                "  Epoch %3d/%d | train=%.6f | val=%.6f | diff=%.6f | align=%.6f | lr=%.2e%s",
                epoch, epochs,
                epoch_losses["total_loss"], val_total,
                epoch_losses["diffusion_loss"], epoch_losses["alignment_loss"],
                epoch_losses["lr"], marker,
            )

        if epochs_without_improvement >= patience:
            logger.info("  Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    # Restore best model
    if best_state_dict is not None:
        denoiser.load_state_dict(best_state_dict)
        logger.info("  Restored best model (val_loss=%.6f)", best_val_loss)

    elapsed = time.time() - start_time
    logger.info(SEP)
    logger.info("  Training complete in %.1f seconds", elapsed)
    logger.info("  Best val loss: %.6f", best_val_loss)
    logger.info(SEP)

    return history


# ════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_denoising(
    denoiser: FeatureDenoiser,
    text_data: torch.Tensor,
    image_data: torch.Tensor | None,
    device: str = "cpu",
) -> dict:
    """Evaluate denoising quality by measuring feature consistency."""
    denoiser.eval()
    denoiser = denoiser.to(device)
    text_data = text_data.to(device)

    if image_data is not None:
        image_data = image_data.to(device)
        text_out, image_out = denoiser.denoise_multimodal(text_data, image_data)

        # Cosine similarity between original and denoised
        text_cos = torch.nn.functional.cosine_similarity(text_data, text_out, dim=-1).mean().item()
        image_cos = torch.nn.functional.cosine_similarity(image_data, image_out, dim=-1).mean().item()

        # MSE between original and denoised
        text_mse = torch.nn.functional.mse_loss(text_data, text_out).item()
        image_mse = torch.nn.functional.mse_loss(image_data, image_out).item()

        metrics = {
            "text_cosine_similarity": round(text_cos, 4),
            "image_cosine_similarity": round(image_cos, 4),
            "text_mse": round(text_mse, 6),
            "image_mse": round(image_mse, 6),
        }
    else:
        text_out = denoiser.denoise_text(text_data)
        text_cos = torch.nn.functional.cosine_similarity(text_data, text_out, dim=-1).mean().item()
        text_mse = torch.nn.functional.mse_loss(text_data, text_out).item()
        metrics = {
            "text_cosine_similarity": round(text_cos, 4),
            "text_mse": round(text_mse, 6),
        }

    return metrics


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Force UTF-8 stdout/stderr on Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Train FeatureDenoiser (MDSBR-inspired multimodal denoising)"
    )
    parser.add_argument(
        "--text-embeddings", type=str, default=None,
        help="Path to text embeddings .pt file",
    )
    parser.add_argument(
        "--image-embeddings", type=str, default=None,
        help="Path to image embeddings .pt file",
    )
    parser.add_argument("--text-only", action="store_true", help="Only denoise text")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data for demo")
    parser.add_argument("--n-samples", type=int, default=1000, help="Synthetic sample count")
    parser.add_argument("--text-dim", type=int, default=768, help="Text embedding dim")
    parser.add_argument("--image-dim", type=int, default=2048, help="Image embedding dim")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Denoiser hidden dim")
    parser.add_argument("--noise-steps", type=int, default=5, help="Diffusion timesteps")
    parser.add_argument(
        "--noise-schedule", type=str, default="cosine",
        choices=["linear", "cosine"], help="Noise schedule type",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--save-path", type=str, default="ai_engine/models/feature_denoiser.pt",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--metrics-path", type=str, default=None,
        help="Path to save training metrics JSON",
    )
    args = parser.parse_args()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load or generate data ──────────────────────────────────────────────
    if args.demo:
        text_data, image_data = generate_synthetic_data(
            n_samples=args.n_samples,
            text_dim=args.text_dim,
            image_dim=args.image_dim,
        )
        if args.text_only:
            image_data = None
    else:
        if args.text_embeddings is None:
            parser.error("--text-embeddings is required (or use --demo)")

        text_data = load_embeddings(args.text_embeddings)
        args.text_dim = text_data.shape[1]

        if args.text_only or args.image_embeddings is None:
            image_data = None
        else:
            image_data = load_embeddings(args.image_embeddings)
            args.image_dim = image_data.shape[1]

            # Ensure same number of samples
            min_samples = min(len(text_data), len(image_data))
            text_data = text_data[:min_samples]
            image_data = image_data[:min_samples]

    # ── Build model ────────────────────────────────────────────────────────
    denoiser = FeatureDenoiser(
        text_dim=args.text_dim,
        image_dim=args.image_dim,
        hidden_dim=args.hidden_dim,
        noise_steps=args.noise_steps,
        noise_schedule=args.noise_schedule,
    )

    param_count = sum(p.numel() for p in denoiser.parameters())
    logger.info("Model parameters: %s", f"{param_count:,}")

    # ── Train ──────────────────────────────────────────────────────────────
    history = train(
        denoiser=denoiser,
        text_data=text_data,
        image_data=image_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        patience=args.patience,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    eval_metrics = evaluate_denoising(denoiser, text_data, image_data, device=device)
    logger.info("Denoising evaluation: %s", json.dumps(eval_metrics, indent=2))

    # ── Save model ─────────────────────────────────────────────────────────
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": denoiser.cpu().state_dict(),
        "config": {
            "text_dim": args.text_dim,
            "image_dim": args.image_dim,
            "hidden_dim": args.hidden_dim,
            "noise_steps": args.noise_steps,
            "noise_schedule": args.noise_schedule,
        },
        "eval_metrics": eval_metrics,
        "training_history": history,
    }
    torch.save(checkpoint, save_path)
    logger.info("Model saved → %s", save_path)

    # ── Save metrics ───────────────────────────────────────────────────────
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {"eval_metrics": eval_metrics, "training_history": history},
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info("Metrics saved → %s", metrics_path)


if __name__ == "__main__":
    main()
