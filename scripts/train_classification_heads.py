"""
train_classification_heads.py
==============================
Train classification heads on top of denoised embeddings, then evaluate
on test set. Compares performance BEFORE vs AFTER denoiser.

Usage:
    py scripts/train_classification_heads.py

Output:
    - artifacts/models/text_sentiment_head.pt
    - artifacts/models/image_defect_head.pt
    - Evaluation metrics printed to console
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_engine.denoising.feature_denoiser import FeatureDenoiser


# ════════════════════════════════════════════════════════════════════════════
#  MLP Classification Head
# ════════════════════════════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    """MLP head for classification on top of embeddings."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════════
#  Training & Evaluation
# ════════════════════════════════════════════════════════════════════════════

def train_head(
    head: ClassificationHead,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    class_weights: torch.Tensor | None = None,
    warmup_epochs: int = 10,
    label_smoothing: float = 0.1,
) -> dict:
    """Train a classification head with early stopping."""
    device = train_x.device
    head = head.to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    # Pure Cosine Annealing (with optional warmup)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=lr * 0.01
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = head(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weights, label_smoothing=label_smoothing)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        head.eval()
        with torch.no_grad():
            val_logits = head(val_x)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_y).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            no_improve = 0
            marker = " ★"
        else:
            no_improve += 1
            marker = ""

        if epoch % 10 == 0 or epoch == 1 or marker:
            logger.info(
                "  Epoch %3d | loss=%.4f | val_acc=%.4f%s",
                epoch, total_loss / len(train_loader), val_acc, marker,
            )

        if no_improve >= patience:
            logger.info("  Early stopping at epoch %d", epoch)
            break

    if best_state:
        head.load_state_dict(best_state)

    return {"best_val_acc": best_val_acc}


def evaluate_head(
    head: ClassificationHead,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    class_names: list[str],
) -> dict:
    """Evaluate on test set with detailed metrics."""
    head.eval()
    with torch.no_grad():
        logits = head(test_x)
        preds = logits.argmax(dim=1)

    correct = (preds == test_y).float()
    accuracy = correct.mean().item()

    # Per-class metrics
    num_classes = len(class_names)
    per_class = {}
    for i, name in enumerate(class_names):
        mask = test_y == i
        if mask.sum() == 0:
            continue
        class_acc = correct[mask].mean().item()
        # Precision & Recall
        pred_mask = preds == i
        tp = (pred_mask & mask).sum().item()
        fp = (pred_mask & ~mask).sum().item()
        fn = (~pred_mask & mask).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[name] = {
            "accuracy": round(class_acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(mask.sum()),
        }

    # Macro F1
    f1s = [v["f1"] for v in per_class.values()]
    macro_f1 = np.mean(f1s) if f1s else 0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Train classification heads on denoised embeddings")
    parser.add_argument("--denoiser-path", default="artifacts/models/denoiser/feature_denoiser.pt")
    parser.add_argument("--paired-csv", default="data/processed/paired_text_image.csv")
    parser.add_argument("--text-embeddings", default="data/processed/paired_text_embeddings.pt")
    parser.add_argument("--image-embeddings", default="data/processed/paired_image_embeddings.pt")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--save-dir", default="artifacts/models/denoiser")
    args = parser.parse_args()

    device = "cpu"
    SEP = "=" * 60

    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_csv(ROOT / args.paired_csv, encoding="utf-8-sig")
    text_emb = torch.load(ROOT / args.text_embeddings, weights_only=False)["embeddings"]
    image_emb = torch.load(ROOT / args.image_embeddings, weights_only=False)["embeddings"]

    logger.info("Loaded %d pairs, text=%s, image=%s", len(df), text_emb.shape, image_emb.shape)

    # ── Encode labels ──────────────────────────────────────────────────────
    sentiment_classes = sorted(df["sentiment_label"].dropna().unique().tolist())
    defect_classes = sorted(df["image_label"].dropna().unique().tolist())

    sentiment_map = {c: i for i, c in enumerate(sentiment_classes)}
    defect_map = {c: i for i, c in enumerate(defect_classes)}

    df["sentiment_idx"] = df["sentiment_label"].map(sentiment_map)
    df["defect_idx"] = df["image_label"].map(defect_map)

    logger.info("Sentiment classes: %s", sentiment_classes)
    logger.info("Defect classes: %s", defect_classes)

    # ── Split by 'split' column ────────────────────────────────────────────
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"

    # ── Load denoiser ──────────────────────────────────────────────────────
    denoiser_path = ROOT / args.denoiser_path
    checkpoint = torch.load(denoiser_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    denoiser = FeatureDenoiser(
        text_dim=config["text_dim"],
        image_dim=config["image_dim"],
        hidden_dim=config["hidden_dim"],
        noise_steps=config["noise_steps"],
        noise_schedule=config.get("noise_schedule", "cosine"),
    )
    denoiser.load_state_dict(checkpoint["model_state_dict"])
    denoiser.eval()
    logger.info("Loaded denoiser from %s (hidden_dim=%d)", denoiser_path, config["hidden_dim"])

    # ── Denoise all embeddings ─────────────────────────────────────────────
    with torch.no_grad():
        text_clean, image_clean = denoiser(text_emb, image_emb)

    logger.info("Denoised: text=%s, image=%s", text_clean.shape, image_clean.shape)

    # ════════════════════════════════════════════════════════════════════════
    #  TEXT SENTIMENT HEAD
    # ════════════════════════════════════════════════════════════════════════
    logger.info(SEP)
    logger.info("  TRAINING TEXT SENTIMENT HEAD")
    logger.info(SEP)

    sentiment_labels = torch.tensor(df["sentiment_idx"].values, dtype=torch.long)

    # Compute class weights (inverse frequency) to handle imbalance
    train_sentiment = sentiment_labels[train_mask]
    counts = torch.bincount(train_sentiment, minlength=len(sentiment_classes)).float()
    sentiment_weights = (counts.sum() / (len(sentiment_classes) * counts)).clamp(max=10.0)
    logger.info("Sentiment class weights: %s", dict(zip(sentiment_classes, sentiment_weights.tolist())))

    for label, embeddings, name in [
        ("BEFORE denoiser", text_emb, "raw"),
        ("AFTER denoiser", text_clean, "denoised"),
    ]:
        logger.info("\n--- %s ---", label)

        head = ClassificationHead(
            input_dim=embeddings.shape[1],
            num_classes=len(sentiment_classes),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )

        train_head(
            head,
            train_x=embeddings[train_mask],
            train_y=sentiment_labels[train_mask],
            val_x=embeddings[val_mask],
            val_y=sentiment_labels[val_mask],
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            class_weights=sentiment_weights,
            warmup_epochs=args.warmup_epochs,
            label_smoothing=args.label_smoothing,
        )

        test_metrics = evaluate_head(
            head,
            test_x=embeddings[test_mask],
            test_y=sentiment_labels[test_mask],
            class_names=sentiment_classes,
        )

        logger.info("  [%s] Test Accuracy: %.4f | Macro F1: %.4f",
                     label, test_metrics["accuracy"], test_metrics["macro_f1"])
        for cls, m in test_metrics["per_class"].items():
            logger.info("    %-12s P=%.3f R=%.3f F1=%.3f (n=%d)", cls, m["precision"], m["recall"], m["f1"], m["support"])

        if name == "raw":
            # Save raw embedding head (no denoiser needed at inference)
            save_path = ROOT / args.save_dir / "text_sentiment_head.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": head.state_dict(),
                "input_dim": embeddings.shape[1],
                "num_classes": len(sentiment_classes),
                "class_names": sentiment_classes,
                "test_metrics": test_metrics,
            }, save_path)
            logger.info("  Saved → %s", save_path)

    # ════════════════════════════════════════════════════════════════════════
    #  IMAGE DEFECT HEAD (BINARY: no-defect vs defect)
    #  Giống pipeline cũ ResNet50: intact=0 (no-defect), còn lại=1 (defect)
    # ════════════════════════════════════════════════════════════════════════
    logger.info(SEP)
    logger.info("  TRAINING IMAGE DEFECT HEAD (BINARY)")
    logger.info(SEP)

    # Convert 4-class → binary: intact=0, others=1
    binary_defect_classes = ["no-defect", "defect"]
    intact_idx = defect_classes.index("intact")
    binary_defect_labels = (torch.tensor(df["defect_idx"].values, dtype=torch.long) != intact_idx).long()
    logger.info("Binary labels: no-defect=%d, defect=%d",
                (binary_defect_labels == 0).sum(), (binary_defect_labels == 1).sum())

    # Class weights for binary
    train_bin = binary_defect_labels[train_mask]
    counts_bin = torch.bincount(train_bin, minlength=2).float()
    binary_weights = (counts_bin.sum() / (2 * counts_bin)).clamp(max=10.0)
    logger.info("Binary class weights: no-defect=%.3f, defect=%.3f",
                binary_weights[0].item(), binary_weights[1].item())

    for label, embeddings, name in [
        ("BEFORE denoiser", image_emb, "raw"),
        ("AFTER denoiser", image_clean, "denoised"),
    ]:
        logger.info("\n--- %s ---", label)

        head = ClassificationHead(
            input_dim=embeddings.shape[1],
            num_classes=2,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )

        train_head(
            head,
            train_x=embeddings[train_mask],
            train_y=binary_defect_labels[train_mask],
            val_x=embeddings[val_mask],
            val_y=binary_defect_labels[val_mask],
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            class_weights=binary_weights,
            warmup_epochs=args.warmup_epochs,
            label_smoothing=args.label_smoothing,
        )

        test_metrics = evaluate_head(
            head,
            test_x=embeddings[test_mask],
            test_y=binary_defect_labels[test_mask],
            class_names=binary_defect_classes,
        )

        logger.info("  [%s] Test Accuracy: %.4f | Macro F1: %.4f",
                     label, test_metrics["accuracy"], test_metrics["macro_f1"])
        for cls, m in test_metrics["per_class"].items():
            logger.info("    %-12s P=%.3f R=%.3f F1=%.3f (n=%d)", cls, m["precision"], m["recall"], m["f1"], m["support"])

        if name == "denoised":
            save_path = ROOT / args.save_dir / "image_defect_head.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": head.state_dict(),
                "input_dim": embeddings.shape[1],
                "num_classes": 2,
                "class_names": binary_defect_classes,
                "test_metrics": test_metrics,
            }, save_path)
            logger.info("  Saved → %s", save_path)

    logger.info(SEP)
    logger.info("  DONE! All heads trained and evaluated.")
    logger.info(SEP)


if __name__ == "__main__":
    main()
