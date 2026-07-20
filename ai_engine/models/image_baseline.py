"""Transfer Learning baseline for defect detection (damaged box classification).

Supports two backbone options:
  - ResNet50   — higher accuracy, slower inference
  - MobileNetV3-Large — lightweight, faster inference (< 50ms)

Both backbones are pretrained on ImageNet and fine-tuned by replacing
the final classification head to output 4 classes:
  intact | damaged | wrong_item | irrelevant

Label schema mirrors Review.label in the Node.js DB model.

Usage
-----
>>> from ai_engine.models.image_baseline import ImageBaselineModel
>>> model = ImageBaselineModel(backbone="resnet50")
>>> model.fit("image_labeling/data/labeled", epochs=10)
>>> result = model.predict("path/to/image.jpg")
>>> model.save("ai_engine/models/weights/resnet50_defect.pt")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
# pyrefly: ignore [missing-import]
from torchvision import datasets, models, transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# 4 labels mapping to schema DB (Review.label)
CLASS_NAMES = ["intact", "damaged", "wrong_item", "irrelevant"]
NUM_CLASSES = len(CLASS_NAMES)

# ImageNet mean/std used for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard input image size for backbones
IMAGE_SIZE = 224


def _build_transforms(is_train: bool) -> transforms.Compose:
    """Create image transform pipeline for training or inference.

    Train: strong augmentation to reduce overfitting.
    Eval/Inference: resize + center crop + normalize.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _build_backbone(backbone: str, num_classes: int = 4) -> Tuple[nn.Module, int]:
    """Load pretrained backbone and return (model, number of features in final layer).

    2-stage fine-tuning strategy:
      - Freeze the entire backbone, train only the head (first epochs).
      - Unfreeze last block so the backbone adapts to the domain (subsequent epochs).
    The caller unfreezes it via _unfreeze_last_block() after a few epochs.
    """
    if backbone == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in net.parameters():
            param.requires_grad = False
        in_features = net.fc.in_features  # 2048
        net.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )
        return net, in_features

    if backbone == "mobilenet_v3":
        net = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        # Freeze all features first
        for param in net.features.parameters():
            param.requires_grad = False
        # Get input features from classifier[0].in_features (960)
        # classifier[0] is Linear(960, 1280), [1] Hardswish, [2] Dropout, [3] Linear(1280,1000)
        in_features = net.classifier[0].in_features  # 960
        # Replace the entire classifier block
        # Add bottleneck 512 for capacity
        net.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )
        # Ensure classifier is trainable
        for param in net.classifier.parameters():
            param.requires_grad = True
        return net, in_features

    if backbone == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Freeze all features
        for param in net.features.parameters():
            param.requires_grad = False
        in_features = net.classifier[1].in_features  # 1280
        # Replace classifier
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )
        for param in net.classifier.parameters():
            param.requires_grad = True
        return net, in_features

    raise ValueError(
        f"Invalid backbone: '{backbone}'. "
        "Choose 'resnet50', 'mobilenet_v3', or 'efficientnet_b0'."
    )


def _unfreeze_last_block(net: nn.Module, backbone: str) -> list:
    """Unfreeze last conv block.

    Returns the list of parameter groups for differential learning rate.
    """
    backbone_params = []
    if backbone == "resnet50":
        # Unfreeze layer4 (last residual block)
        for param in net.layer4.parameters():  # type: ignore[attr-defined]
            param.requires_grad = True
            backbone_params.append(param)
        logger.info("Unfroze ResNet50 layer4 (%d param groups)", len(backbone_params))
    elif backbone == "mobilenet_v3":
        # MobileNetV3-Large features has 17 blocks (index 0-16).
        # Block 16 is expansion Conv (1x1 -> 960), 13-15 is final InvertedResidual.
        # Use net.features[13:] to unfreeze last 4 blocks.
        features = net.features  # type: ignore[attr-defined]
        for block in features[13:]:  # type: ignore[index]
            for param in block.parameters():
                param.requires_grad = True
                backbone_params.append(param)
        logger.info("Unfroze MobileNetV3 features[13:] (%d params)", len(backbone_params))
    elif backbone == "efficientnet_b0":
        # EfficientNet-B0 has 9 blocks (0-8), unfreeze last 2 blocks
        features = net.features  # type: ignore[attr-defined]
        for block in features[7:]:  # type: ignore[index]
            for param in block.parameters():
                param.requires_grad = True
                backbone_params.append(param)
        logger.info("Unfroze EfficientNet-B0 features[7:] (%d params)", len(backbone_params))
    return backbone_params


# -- Helper functions

def _count_params(net: nn.Module, backbone: str) -> dict:
    """Print and return model parameter statistics."""
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen = total - trainable
    activation_map = {
        "mobilenet_v3": "Hardswish",
        "resnet50": "ReLU",
        "efficientnet_b0": "SiLU (Swish)",
    }
    activation = activation_map.get(backbone, "ReLU")
    logger.info("=" * 55)
    logger.info("Model: %s", backbone.upper())
    logger.info("  Total parameters:     %s", f"{total:,}")
    logger.info("  Trainable parameters: %s (%.1f%%)", f"{trainable:,}", trainable / total * 100)
    logger.info("  Frozen parameters:    %s (%.1f%%)", f"{frozen:,}", frozen / total * 100)
    logger.info("  Activation function:  %s", activation)
    logger.info("=" * 55)
    return {"total": total, "trainable": trainable, "frozen": frozen, "activation": activation}


def _plot_learning_curves(history: dict, save_dir: str, backbone: str, filename: str | None = None) -> None:
    """Plot Loss, Accuracy, Macro-F1 curves by epoch and save as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed — skipping Learning Curves.")
        return
    os.makedirs(save_dir, exist_ok=True)
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Learning Curves — {backbone.upper()}", fontsize=13, fontweight="bold")
 
    # Loss
    axes[0].plot(ep, history["train_loss"], "b-o", markersize=4, label="Train")
    axes[0].plot(ep, history["val_loss"],   "r-o", markersize=4, label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(ep, history["train_acc"], "b-o", markersize=4, label="Train")
    axes[1].plot(ep, history["val_acc"],   "r-o", markersize=4, label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # Macro-F1 (val only)
    axes[2].plot(ep, history["val_f1"], "g-o", markersize=4, label="Val Macro-F1")
    axes[2].set_title("Val Macro-F1"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("F1")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_name = filename if filename else f"{backbone}_learning_curves.png"
    out = os.path.join(save_dir, out_name)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Learning Curves → %s", out)


def _plot_confusion_matrix(
    all_labels: list, all_preds: list, class_names: list,
    save_dir: str, backbone: str, filename: str | None = None,
) -> None:
    """Plot Confusion Matrix (raw + normalized) and save as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import confusion_matrix
    except ImportError:
        logger.warning("matplotlib is not installed — skipping Confusion Matrix.")
        return
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Confusion Matrix — {backbone.upper()}", fontsize=13, fontweight="bold")
    ticks = range(len(class_names))

    for ax, data, title, fmt in [
        (axes[0], cm,      "Raw Counts",          "d"),
        (axes[1], cm_norm, "Normalized (Recall)", ".2f"),
    ]:
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=(1 if fmt == ".2f" else None))
        plt.colorbar(im, ax=ax)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title)
        thresh = data.max() / 2
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black", fontsize=9)

    plt.tight_layout()
    out_name = filename if filename else f"{backbone}_confusion_matrix.png"
    out = os.path.join(save_dir, out_name)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion Matrix → %s", out)


def _save_error_analysis(
    val_paths: list, all_labels: list, all_preds: list,
    class_names: list, save_dir: str, backbone: str, max_samples: int = 20,
) -> None:
    """Save JSON + images of misclassified samples for error analysis."""
    error_dir = os.path.join(save_dir, f"{backbone}_error_analysis")
    os.makedirs(error_dir, exist_ok=True)

    errors = [
        {"image": str(val_paths[i]),
         "true": class_names[all_labels[i]],
         "pred": class_names[all_preds[i]]}
        for i in range(len(all_preds))
        if all_preds[i] != all_labels[i]
    ]
    error_pairs = Counter(
        f"{class_names[t]}->{class_names[p]}"
        for t, p in zip(all_labels, all_preds) if t != p
    )
    summary = {
        "total_errors": len(errors),
        "error_rate": round(len(errors) / max(len(all_labels), 1), 4),
        "top_error_patterns": dict(error_pairs.most_common(8)),
        "samples": errors[:max_samples],
    }
    with open(os.path.join(error_dir, "error_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Copy error images
    copied = 0
    for err in errors[:max_samples]:
        src = Path(err["image"])
        if src.exists():
            dest = os.path.join(error_dir, f"{err['true']}_as_{err['pred']}_{src.name}")
            try:
                shutil.copy2(src, dest)
                copied += 1
            except OSError:
                pass

    logger.info(
        "Error analysis: %d/%d incorrect (%.1f%%) | %d images saved → %s",
        len(errors), len(all_labels), summary["error_rate"] * 100, copied, error_dir,
    )
    for pattern, cnt in list(error_pairs.most_common(5)):
        logger.info("  %s: %d times", pattern, cnt)


class ImageBaselineModel:
    """Transfer Learning model for product packaging condition detection."""

    def __init__(
        self,
        backbone: Literal["resnet50", "mobilenet_v3"] = "resnet50",
        device: Optional[str] = None,
    ) -> None:
        self.backbone = backbone
        # Select GPU if available, fallback to CPU
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = CLASS_NAMES
        self.model: Optional[nn.Module] = None
        self.threshold: float = 0.5
        logger.info("ImageBaselineModel initialized — backbone=%s | device=%s | threshold=%.3f", backbone, self.device, self.threshold)

    def _get_model(self, num_classes: int = 4) -> nn.Module:
        """Initialize model if not present and send to device."""
        if self.model is None:
            net, _ = _build_backbone(self.backbone, num_classes=num_classes)
            self.model = net.to(self.device)
        return self.model

    def fit(
        self,
        data_dir: str,
        val_dir: str | None = None,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        val_split: float = 0.2,
        patience: int = 3,
        subset_ratio: float = 1.0,
        results_dir: str = "ai_engine/models/results",  # output directory for plots and history
        class_weight_mode: str = "sqrt",
        use_sampler: bool = False,
        threshold_mode: str = "maximize_macro_f1_subject_to_recall",
        learning_curves_name: str | None = None,
        confusion_matrix_name: str | None = None,
        training_history_name: str | None = None,
        threshold_tuning_name: str | None = None,
    ) -> "ImageBaselineModel":
        """Fine-tune backbone on labeled image data.

        The data_dir folder structure must follow ImageFolder format.

        Args:
            data_dir: Path to directory containing subfolders by label (or train folder).
            val_dir: Path to validation directory (if using a physical split).
            epochs: Maximum number of epochs.
            batch_size: Mini-batch size.
            lr: Learning rate.
            val_split: Validation split ratio (when using random split).
            patience: Patience epochs for early stopping.
            subset_ratio: Ratio of training data to use.
            results_dir: Results directory.
            class_weight_mode: Class weighting method for loss function ('none', 'balanced', 'sqrt').
            use_sampler: Whether to use WeightedRandomSampler.
            threshold_mode: Selection mode for the best threshold on the validation set.
            learning_curves_name: Filename for learning curves plot.
            confusion_matrix_name: Filename for confusion matrix plot.
            training_history_name: Filename for training history.
            threshold_tuning_name: Filename for threshold tuning results.
        """
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.metrics import classification_report as sk_report

        if val_dir is not None:
            # Physical split mode
            train_dataset = datasets.ImageFolder(
                root=data_dir,
                transform=_build_transforms(is_train=True),
            )
            val_dataset = datasets.ImageFolder(
                root=val_dir,
                transform=_build_transforms(is_train=False),
            )
            self.class_names = train_dataset.classes
            num_classes = len(self.class_names)
            logger.info("Physical Split Mode: Train: %d, Val: %d | Classes: %s", 
                        len(train_dataset), len(val_dataset), self.class_names)
            
            train_idx = list(range(len(train_dataset)))
            val_idx = list(range(len(val_dataset)))
            all_targets = train_dataset.targets
            
            if 0.0 < subset_ratio < 1.0:
                sss_sub = StratifiedShuffleSplit(
                    n_splits=1, test_size=(1.0 - subset_ratio), random_state=42
                )
                keep_local, _ = next(sss_sub.split(range(len(train_idx)), all_targets))
                train_idx = [train_idx[i] for i in keep_local]
                logger.info(
                    "subset_ratio=%.2f → training set reduced to %d images",
                    subset_ratio, len(train_idx)
                )
        else:
            # Random split mode
            train_dataset = datasets.ImageFolder(
                root=data_dir,
                transform=_build_transforms(is_train=True),
            )
            val_dataset = datasets.ImageFolder(
                root=data_dir,
                transform=_build_transforms(is_train=False),
            )
            self.class_names = train_dataset.classes
            num_classes = len(self.class_names)
            logger.info("Random Split Mode: Dataset: %d images | Classes: %s", len(train_dataset), self.class_names)
            
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
            all_targets = train_dataset.targets
            train_idx, val_idx = next(sss.split(range(len(train_dataset)), all_targets))
            
            if 0.0 < subset_ratio < 1.0:
                sss_sub = StratifiedShuffleSplit(
                    n_splits=1, test_size=(1.0 - subset_ratio), random_state=42
                )
                sub_targets = [all_targets[i] for i in train_idx]
                keep_local, _ = next(sss_sub.split(range(len(train_idx)), sub_targets))
                train_idx = [train_idx[i] for i in keep_local]
                logger.info(
                    "subset_ratio=%.2f → training set reduced to %d images (validation remains %d images)",
                    subset_ratio, len(train_idx), len(val_idx),
                )

        train_set = Subset(train_dataset, train_idx)  # augmented transforms
        val_set = Subset(val_dataset, val_idx)         # eval transforms (separate!)

        n_train = len(train_idx)
        n_val = len(val_idx)

        # -- Adjust pin_memory and num_workers by platform
        use_pin = torch.cuda.is_available()
        n_workers = 0 if os.name == 'nt' else 2

        # WeightedRandomSampler vs Standard shuffling
        if use_sampler:
            train_targets = [all_targets[i] for i in train_idx]
            class_counts = torch.bincount(torch.tensor(train_targets), minlength=num_classes).float()
            sample_class_weights = 1.0 / class_counts.clamp(min=1)
            sample_weights = sample_class_weights[torch.tensor(train_targets)]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(
                train_set, batch_size=batch_size, sampler=sampler,
                num_workers=n_workers, pin_memory=use_pin,
            )
            logger.info("Using WeightedRandomSampler for class balance during training.")
        else:
            train_loader = DataLoader(
                train_set, batch_size=batch_size, shuffle=True,
                num_workers=n_workers, pin_memory=use_pin,
            )
            logger.info("Using standard shuffling (no sampler) for training.")

        # Class counts & weights
        train_targets = [all_targets[i] for i in train_idx]
        class_counts = torch.bincount(torch.tensor(train_targets), minlength=num_classes).float()
        logger.info("Class counts (train): %s", dict(zip(train_dataset.classes, class_counts.int().tolist())))

        if class_weight_mode == "none":
            loss_weights = None
            logger.info("Loss weights: None (no class weighting)")
        elif class_weight_mode == "balanced":
            loss_weights = 1.0 / class_counts.clamp(min=1)
            loss_weights = loss_weights / loss_weights.sum() * num_classes
            logger.info("Loss weights (balanced): %s",
                        {c: round(w, 3) for c, w in zip(train_dataset.classes, loss_weights.tolist())})
        elif class_weight_mode == "sqrt":
            loss_weights = 1.0 / torch.sqrt(class_counts.clamp(min=1))
            loss_weights = loss_weights / loss_weights.sum() * num_classes
            logger.info("Loss weights (sqrt): %s",
                        {c: round(w, 3) for c, w in zip(train_dataset.classes, loss_weights.tolist())})
        else:
            raise ValueError(f"Unknown class_weight_mode: {class_weight_mode}")

        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=n_workers, pin_memory=use_pin,
        )

        # Store val paths for error analysis
        val_paths = [val_dataset.imgs[i][0] for i in val_idx]

        net = self._get_model(num_classes=num_classes)

        # Count parameters
        param_info = _count_params(net, self.backbone)

        # Initialize training history
        history: dict = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "val_f1": [], "lr": [],
        }

        # Stage 1: train head only (backbone frozen)
        head_params = list(filter(lambda p: p.requires_grad, net.parameters()))
        optimizer = torch.optim.Adam(head_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        
        # Train criterion: CrossEntropyLoss with class weights
        weight_tensor = loss_weights.to(self.device) if loss_weights is not None else None
        criterion = nn.CrossEntropyLoss(
            weight=weight_tensor, label_smoothing=0.05
        )
        # Val criterion does not use class weights for fair evaluation
        val_criterion = nn.CrossEntropyLoss()

        backbone_unfrozen = False
        best_val_loss = float("inf")
        best_val_f1 = 0.0
        epochs_no_improve = 0
        # Set patience (loss can fluctuate with imbalanced data)
        effective_patience = max(patience, 5)
        import tempfile
        best_weights_path = os.path.join(tempfile.gettempdir(), f"{self.backbone}_best.pt")

        for epoch in range(1, epochs + 1):
            # Stage 2: unfreeze last block after epoch 3
            if not backbone_unfrozen and epoch > 3:
                backbone_params = _unfreeze_last_block(net, self.backbone)
                if backbone_params:
                    optimizer.add_param_group({"params": backbone_params, "lr": lr / 10})
                    backbone_unfrozen = True
                    logger.info("Epoch %d: backbone unfrozen, differential LR=%.2e", epoch, lr / 10)

            net.train()
            train_loss, train_correct = 0.0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = net(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                # FIX: Gradient clipping — prevent explosion when unfreezing backbone
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()

            net.eval()
            val_loss, val_correct = 0.0, 0
            val_all_preds: list = []
            val_all_labels: list = []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    outputs = net(imgs)
                    # FIX #4: Use val_criterion (unweighted) for early stopping
                    loss = val_criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)
                    preds = outputs.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    val_all_preds.extend(preds.cpu().tolist())
                    val_all_labels.extend(labels.cpu().tolist())

            from sklearn.metrics import f1_score
            train_acc = train_correct / n_train
            val_acc = val_correct / n_val
            val_loss_avg = val_loss / n_val
            # Calculate Macro-F1 to monitor class-imbalanced performance better
            val_macro_f1 = f1_score(val_all_labels, val_all_preds, average="macro", zero_division=0)

            logger.info(
                "Epoch %02d/%02d — train_acc=%.4f | val_acc=%.4f | val_loss=%.4f | val_macro_f1=%.4f",
                epoch, epochs, train_acc, val_acc, val_loss_avg, val_macro_f1,
            )

            # Log history
            history["train_loss"].append(round(train_loss / n_train, 5))
            history["val_loss"].append(round(val_loss_avg, 5))
            history["train_acc"].append(round(train_acc, 5))
            history["val_acc"].append(round(val_acc, 5))
            history["val_f1"].append(round(val_macro_f1, 5))
            history["lr"].append(optimizer.param_groups[0]["lr"])

            scheduler.step(val_loss_avg)

            # Checkpoint when F1 or val_loss is best
            improved = False
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                improved = True
            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                improved = True

            if improved:
                epochs_no_improve = 0
                torch.save(net.state_dict(), best_weights_path)
                logger.info("  ✓ Checkpoint saved (val_loss=%.4f | val_f1=%.4f)",
                            best_val_loss, best_val_f1)
            else:
                epochs_no_improve += 1
                logger.info("  No improvement (%d/%d)", epochs_no_improve, effective_patience)

            if epochs_no_improve >= effective_patience:
                logger.info("Early stopping at epoch %d (patience=%d).", epoch, effective_patience)
                break

        # Restore best weights from checkpoint
        net.load_state_dict(torch.load(best_weights_path, map_location=self.device))

        # --- Save training history JSON + plot Learning Curves ---
        os.makedirs(results_dir, exist_ok=True)
        history_data = {**history, "param_info": param_info, "backbone": self.backbone}
        hist_name = training_history_name if training_history_name else f"{self.backbone}_training_history.json"
        history_path = os.path.join(results_dir, hist_name)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2)
        logger.info("Training history → %s", history_path)
        _plot_learning_curves(history, results_dir, self.backbone, filename=learning_curves_name)

        # --- Threshold tuning on validation set ---
        tune_name = threshold_tuning_name if threshold_tuning_name else f"{self.backbone}_threshold_tuning.json"
        tune_path = os.path.join(results_dir, tune_name)
        self.tune_threshold(
            val_loader=val_loader,
            mode=threshold_mode,
            save_path=tune_path
        )

        # --- Final evaluation on VAL SET (no data leakage) ---
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION — val set only (%d images) using threshold %.3f:", n_val, self.threshold)
        net.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(self.device)
                outputs = net(imgs)
                probs = torch.softmax(outputs, dim=1)
                
                if len(self.class_names) == 2:
                    defect_idx = 0
                    if "defect" in self.class_names:
                        defect_idx = self.class_names.index("defect")
                    elif "damaged" in self.class_names:
                        defect_idx = self.class_names.index("damaged")
                        
                    prob_defect = probs[:, defect_idx]
                    if defect_idx == 1:
                        preds = (prob_defect >= self.threshold).long().cpu().tolist()
                    else:
                        preds = torch.where(
                            prob_defect >= self.threshold,
                            torch.zeros_like(prob_defect, dtype=torch.long),
                            torch.ones_like(prob_defect, dtype=torch.long)
                        ).cpu().tolist()
                else:
                    preds = outputs.argmax(dim=1).cpu().tolist()
                    
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        report_str = sk_report(all_labels, all_preds, target_names=self.class_names)
        report_dict = sk_report(all_labels, all_preds, target_names=self.class_names, output_dict=True)
        logger.info("\n%s", report_str)
        logger.info(
            "Val Accuracy: %.4f | Val Macro-F1: %.4f",
            report_dict["accuracy"], report_dict["macro avg"]["f1-score"],
        )
        logger.info("=" * 60)

        # --- Plot Confusion Matrix + Error Analysis ---
        _plot_confusion_matrix(all_labels, all_preds, self.class_names, results_dir, self.backbone, filename=confusion_matrix_name)
        _save_error_analysis(val_paths, all_labels, all_preds, self.class_names, results_dir, self.backbone)

        # Save val report for train script usage
        self._val_report = report_dict
        logger.info("Training complete. Best val_loss=%.4f | Selected Threshold=%.3f", best_val_loss, self.threshold)
        return self

    def tune_threshold(
        self,
        val_loader: DataLoader,
        mode: str = "maximize_macro_f1_subject_to_recall",
        save_path: str | None = None,
    ) -> float:
        """Runs threshold sweep on validation set and selects the best threshold.
        
        Args:
            val_loader: DataLoader for the validation set.
            mode: threshold selection mode:
                  - 'maximize_defect_f1'
                  - 'maximize_macro_f1'
                  - 'maximize_macro_f1_subject_to_recall'
            save_path: JSON path to save tuning results.
        """
        import numpy as np
        from sklearn.metrics import precision_recall_fscore_support, f1_score
        
        net = self._get_model(num_classes=len(self.class_names))
        net.eval()
        
        # 1. Collect all predictions (probabilities) and true labels on validation set
        all_probs = []
        all_labels = []
        
        # Find defect index
        defect_idx = 0
        if "defect" in self.class_names:
            defect_idx = self.class_names.index("defect")
        elif "damaged" in self.class_names:
            defect_idx = self.class_names.index("damaged")
            
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(self.device)
                outputs = net(imgs)
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs[:, defect_idx].cpu().tolist())
                all_labels.extend(labels.tolist())
                
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 2. Sweep thresholds from 0.05 to 0.95 with step 0.01
        thresholds = np.arange(0.05, 0.96, 0.01)
        sweep_results = []
        
        for thresh in thresholds:
            thresh = round(float(thresh), 3)
            if defect_idx == 1:
                preds = (all_probs >= thresh).astype(int)
            else:
                preds = np.where(all_probs >= thresh, 0, 1)
                
            p, r, f, _ = precision_recall_fscore_support(all_labels, preds, labels=[0, 1], zero_division=0)
            defect_p = float(p[defect_idx])
            defect_r = float(r[defect_idx])
            defect_f1 = float(f[defect_idx])
            macro_f1 = float(f1_score(all_labels, preds, average="macro", zero_division=0))
            
            sweep_results.append({
                "threshold": thresh,
                "defect_precision": round(defect_p, 4),
                "defect_recall": round(defect_r, 4),
                "defect_f1": round(defect_f1, 4),
                "macro_f1": round(macro_f1, 4)
            })
            
        # 3. Select best threshold based on selection mode
        best_threshold = 0.5
        if mode == "maximize_defect_f1":
            best_candidate = max(sweep_results, key=lambda x: x["defect_f1"])
            best_threshold = best_candidate["threshold"]
        elif mode == "maximize_macro_f1":
            best_candidate = max(sweep_results, key=lambda x: x["macro_f1"])
            best_threshold = best_candidate["threshold"]
        elif mode == "maximize_macro_f1_subject_to_recall":
            # Filter candidates with defect_recall >= 0.80
            candidates = [r for r in sweep_results if r["defect_recall"] >= 0.80]
            if candidates:
                best_candidate = max(candidates, key=lambda x: x["macro_f1"])
                best_threshold = best_candidate["threshold"]
            else:
                # Fallback: maximize defect recall
                max_recall = max(r["defect_recall"] for r in sweep_results)
                candidates_max_recall = [r for r in sweep_results if r["defect_recall"] == max_recall]
                best_candidate = max(candidates_max_recall, key=lambda x: x["macro_f1"])
                best_threshold = best_candidate["threshold"]
                
        logger.info(
            "Threshold sweep completed using mode '%s'. Best threshold selected: %.3f",
            mode, best_threshold
        )
        
        # 4. Save results to JSON
        if save_path:
            tuning_data = {
                "best_threshold": best_threshold,
                "selection_mode": mode,
                "sweep_results": sweep_results
            }
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(tuning_data, f, indent=2)
            logger.info("Saved threshold tuning results to: %s", save_path)
            
        self.threshold = best_threshold
        return best_threshold

    def predict(self, image_path: str) -> Dict[str, object]:
        """Predict label for a single image.

        Args:
            image_path: Path to the image file (.jpg/.png).

        Returns:
            dict with keys:
              - label (str): Predicted label, e.g., "defect".
              - confidence (float): Probability of the predicted label (0–1).
              - probabilities (dict): Complete class probability dictionary.
              - inference_ms (float): Inference time in milliseconds.
        """
        from PIL import Image

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        net = self._get_model(num_classes=len(self.class_names))
        net.eval()

        transform = _build_transforms(is_train=False)
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = net(tensor)                           # (1, num_classes)
            probs = torch.softmax(logits, dim=1)[0]       # (num_classes,)
        inference_ms = (time.perf_counter() - t0) * 1000

        threshold = getattr(self, "threshold", 0.5)
        if len(self.class_names) == 2:
            defect_idx = 0
            if "defect" in self.class_names:
                defect_idx = self.class_names.index("defect")
            elif "damaged" in self.class_names:
                defect_idx = self.class_names.index("damaged")
            
            prob_defect = probs[defect_idx].item()
            if prob_defect >= threshold:
                pred_idx = defect_idx
                confidence = prob_defect
            else:
                pred_idx = 1 - defect_idx
                confidence = 1.0 - prob_defect
        else:
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

        return {
            "label": self.class_names[pred_idx],
            "confidence": round(confidence, 4),
            "probabilities": {
                name: round(probs[i].item(), 4)
                for i, name in enumerate(self.class_names)
            },
            "inference_ms": round(inference_ms, 2),
        }

    def predict_batch(self, image_paths: list[str], batch_size: int = 32) -> list[Dict]:
        """Predict labels for multiple images at once — more efficient than predicting individually.

        Args:
            image_paths: List of image paths.
            batch_size: Number of images to process per forward pass.

        Returns:
            List of result dicts, in the same order as image_paths.
        """
        from PIL import Image

        net = self._get_model(num_classes=len(self.class_names))
        net.eval()
        transform = _build_transforms(is_train=False)

        results = []
        threshold = getattr(self, "threshold", 0.5)
        
        defect_idx = 0
        if len(self.class_names) == 2:
            if "defect" in self.class_names:
                defect_idx = self.class_names.index("defect")
            elif "damaged" in self.class_names:
                defect_idx = self.class_names.index("damaged")

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i: i + batch_size]
            tensors = []
            valid_paths = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(transform(img))
                    valid_paths.append(p)
                except Exception as e:
                    logger.warning("Skipping corrupted image %s: %s", p, e)

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors).to(self.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = net(batch_tensor)
                probs = torch.softmax(logits, dim=1)
            inference_ms = (time.perf_counter() - t0) * 1000 / len(tensors)

            for j, path in enumerate(valid_paths):
                if len(self.class_names) == 2:
                    prob_defect = probs[j][defect_idx].item()
                    if prob_defect >= threshold:
                        pred_idx = defect_idx
                        confidence = prob_defect
                    else:
                        pred_idx = 1 - defect_idx
                        confidence = 1.0 - prob_defect
                else:
                    pred_idx = probs[j].argmax().item()
                    confidence = probs[j][pred_idx].item()

                results.append({
                    "image_path": path,
                    "label": self.class_names[pred_idx],
                    "confidence": round(confidence, 4),
                    "probabilities": {
                        name: round(probs[j][k].item(), 4)
                        for k, name in enumerate(self.class_names)
                    },
                    "inference_ms": round(inference_ms, 2),
                })

        return results

    def evaluate(self, data_dir: str, batch_size: int = 32, threshold: float | None = None) -> Dict[str, float]:
        """Evaluate model on a test set (accuracy, per-class accuracy).

        Args:
            data_dir: Test images directory in ImageFolder format (subfolders by label).
            batch_size: Batch size for DataLoader.
            threshold: Classification threshold for defect class (if None, uses self.threshold or 0.5).

        Returns:
            dict containing overall_accuracy and per-class accuracy.
        """
        from sklearn.metrics import classification_report

        dataset = datasets.ImageFolder(
            root=data_dir,
            transform=_build_transforms(is_train=False),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        net = self._get_model(num_classes=len(self.class_names))
        net.eval()

        eval_thresh = threshold if threshold is not None else getattr(self, "threshold", 0.5)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                outputs = net(imgs)
                probs = torch.softmax(outputs, dim=1)
                
                if len(self.class_names) == 2:
                    defect_idx = 0
                    if "defect" in self.class_names:
                        defect_idx = self.class_names.index("defect")
                    elif "damaged" in self.class_names:
                        defect_idx = self.class_names.index("damaged")
                    
                    prob_defect = probs[:, defect_idx]
                    if defect_idx == 1:
                        preds = (prob_defect >= eval_thresh).long().cpu().tolist()
                    else:
                        preds = torch.where(
                            prob_defect >= eval_thresh,
                            torch.zeros_like(prob_defect, dtype=torch.long),
                            torch.ones_like(prob_defect, dtype=torch.long)
                        ).cpu().tolist()
                else:
                    preds = outputs.argmax(dim=1).cpu().tolist()
                
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        report = classification_report(
            all_labels, all_preds,
            target_names=dataset.classes,
            output_dict=True,
        )
        accuracy = report["accuracy"]
        logger.info("Evaluation (threshold=%.3f) — Overall Accuracy: %.4f", eval_thresh, accuracy)
        logger.info("\n%s", classification_report(all_labels, all_preds, target_names=dataset.classes))
        return report

    def save(self, filepath: str) -> None:
        """Save state dict + metadata to a .pt file.

        Saves backbone name, class_names, and threshold to reload the exact configuration.
        Supports: resnet50 | mobilenet_v3 | efficientnet_b0
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call .fit() first.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        threshold = getattr(self, "threshold", 0.5)
        torch.save({
            "backbone": self.backbone,
            "class_names": self.class_names,
            "state_dict": self.model.state_dict(),
            "threshold": threshold,
        }, filepath)
        logger.info("Model saved → %s (threshold=%.3f)", filepath, threshold)

    @classmethod
    def load(cls, filepath: str) -> "ImageBaselineModel":
        """Load model from a saved .pt file.

        Args:
            filepath: Path to the .pt file.

        Returns:
            ImageBaselineModel instance ready for inference.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        backbone = checkpoint["backbone"]
        class_names = checkpoint["class_names"]
        num_classes = len(class_names)

        instance = cls(backbone=backbone)
        instance.class_names = class_names
        net, _ = _build_backbone(backbone, num_classes=num_classes)
        net.load_state_dict(checkpoint["state_dict"])
        instance.model = net.to(instance.device)
        instance.threshold = checkpoint.get("threshold", 0.5)

        logger.info("Model loaded ← %s (backbone=%s, classes=%d, threshold=%.3f)", 
                    filepath, backbone, num_classes, instance.threshold)
        return instance
