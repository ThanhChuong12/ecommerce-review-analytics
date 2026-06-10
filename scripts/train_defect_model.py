"""
train_defect_model.py
---------------------
Train ResNet50 for binary defect detection: no-defect (0) vs defect (1).

This version is split-aware and runs training, threshold tuning on Validation,
and final metrics report generation on the Test set.
"""

import sys
import time
import argparse
import json
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)

from ai_engine.image_processing.defect_detection import (
    get_dataloaders,
    get_resnet50_model,
    FocalLoss,
    ProductDefectDataset,
)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for 1 epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set. Returns (loss, accuracy, all_preds, all_labels)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


def compute_defect_metrics(val_preds, val_labels):
    """Compute defect precision, recall, F1 and macro F1."""
    val_preds_np = np.array(val_preds)
    val_labels_np = np.array(val_labels)

    # Defect (class 1) metrics
    defect_correct = int(((val_preds_np == 1) & (val_labels_np == 1)).sum())
    defect_total_true = int((val_labels_np == 1).sum())
    defect_pred_total = int((val_preds_np == 1).sum())

    defect_recall = defect_correct / max(defect_total_true, 1)
    defect_precision = defect_correct / max(defect_pred_total, 1)
    defect_f1 = (
        2 * defect_precision * defect_recall / max(defect_precision + defect_recall, 1e-8)
    )
    macro_f1 = float(
        f1_score(val_labels_np, val_preds_np, average="macro", zero_division=0)
    )

    return defect_precision, defect_recall, defect_f1, macro_f1


def count_train_labels(train_loader):
    """
    Count class distribution in the training subset.
    Supports both torch.utils.data.Subset and ProductDefectDataset.
    """
    dataset = train_loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        indices = dataset.indices
        base_dataset = dataset.dataset
        labels = [base_dataset.labels[i] for i in indices]
    else:
        labels = dataset.labels
    n_defect = sum(1 for l in labels if l == 1)
    n_normal = sum(1 for l in labels if l == 0)
    return n_normal, n_defect


# ── Plotting Helpers ─────────────────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, val_f1s, figures_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        epochs = range(1, len(train_losses) + 1)

        # Loss curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
        plt.plot(epochs, val_losses, 'r-o', label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(figures_dir / "resnet50_split_loss_curve.png", dpi=150)
        plt.close()

        # Accuracy curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_accs, 'b-o', label='Train Acc')
        plt.plot(epochs, val_accs, 'r-o', label='Val Acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(figures_dir / "resnet50_split_accuracy_curve.png", dpi=150)
        plt.close()

        # Defect F1 curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, val_f1s, 'g-o', label='Val Defect F1')
        plt.title('Validation Defect F1 by Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(figures_dir / "resnet50_split_defect_f1_curve.png", dpi=150)
        plt.close()

        print(f"Training curves saved to {figures_dir}")
    except Exception as e:
        print(f"[WARNING] Could not plot training curves: {e}")


def plot_threshold_tuning(thresholds, precisions, recalls, f1s, best_threshold, figures_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, precisions, 'b--', label='Precision')
        plt.plot(thresholds, recalls, 'g-', label='Recall')
        plt.plot(thresholds, f1s, 'r-', linewidth=2, label='F1 Score')
        plt.axvline(x=best_threshold, color='black', linestyle=':', label=f'Best Threshold ({best_threshold:.3f})')
        plt.title('Threshold Tuning Curve on Validation Set')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(figures_dir / "resnet50_split_threshold_tuning.png", dpi=150)
        plt.close()
        print(f"Threshold tuning curve saved to {figures_dir}")
    except Exception as e:
        print(f"[WARNING] Could not plot threshold tuning curve: {e}")


def plot_confusion_matrix_test(cm, figures_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["no-defect", "defect"],
            yticklabels=["no-defect", "defect"]
        )
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix on Test Set')
        plt.tight_layout()
        plt.savefig(figures_dir / "confusion_matrix_resnet50_split_test.png", dpi=150)
        plt.close()
        print(f"Test confusion matrix saved to {figures_dir}")
    except Exception as e:
        print(f"[WARNING] Could not plot confusion matrix: {e}")


def plot_misclassified_examples(misclassified, figures_dir, max_images=16):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import cv2

        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        n_images = min(len(misclassified), max_images)
        if n_images == 0:
            print("No misclassified examples to plot.")
            return

        cols = 4
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows))
        if n_images == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i in range(len(axes)):
            if i < n_images:
                item = misclassified[i]
                img = cv2.imread(str(item["path"]))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img)
                axes[i].axis('off')
                true_name = "defect" if item["true_label"] == 1 else "no-defect"
                pred_name = "defect" if item["pred_label"] == 1 else "no-defect"
                axes[i].set_title(
                    f"True: {true_name}\nPred: {pred_name} (p={item['prob']:.3f})",
                    color="red", fontsize=10
                )
            else:
                axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(figures_dir / "resnet50_split_test_misclassified_examples.png", dpi=150)
        plt.close()
        print(f"Misclassified examples grid saved to {figures_dir}")
    except Exception as e:
        print(f"[WARNING] Could not plot misclassified examples: {e}")


# ── Threshold Tuning & Test Evaluation ───────────────────────────────────────

def tune_threshold_on_val(model, val_loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    thresholds = np.arange(0.05, 0.95, 0.025)
    precisions = []
    recalls = []
    f1s = []

    best_f1 = -1.0
    best_threshold = 0.5
    best_stats = {}

    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        tp = np.sum((preds == 1) & (all_labels == 1))
        fp = np.sum((preds == 1) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))

        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_stats = {
                "threshold": float(thresh),
                "val_precision": float(precision),
                "val_recall": float(recall),
                "val_f1": float(f1),
                "tp": int(tp), "fp": int(fp), "fn": int(fn)
            }

    print(f"\nValidation Threshold Sweep -> Best Threshold: {best_threshold:.3f} | Best F1: {best_f1:.4f}")
    return best_threshold, thresholds, precisions, recalls, f1s, best_stats


def evaluate_on_test_set(model, test_loader, threshold, device, figures_dir, json_path, md_path, best_epoch):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= threshold).astype(int)

    test_acc = accuracy_score(all_labels, preds)
    roc_auc = roc_auc_score(all_labels, all_probs)

    prec, rec, f1, support = precision_recall_fscore_support(all_labels, preds, average=None, labels=[0, 1])
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(all_labels, preds, average='macro')
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(all_labels, preds, average='weighted')

    cm = confusion_matrix(all_labels, preds)
    tn, fp, fn, tp = cm.ravel()

    # Collect misclassified examples
    misclassified = []
    paths = test_loader.dataset.image_paths
    for idx, (p, label, pred, prob) in enumerate(zip(paths, all_labels, preds, all_probs)):
        if label != pred:
            misclassified.append({
                "path": p,
                "true_label": int(label),
                "pred_label": int(pred),
                "prob": float(prob)
            })

    # Plot results
    plot_confusion_matrix_test(cm, figures_dir)
    plot_misclassified_examples(misclassified, figures_dir)

    metrics = {
        "best_epoch": best_epoch,
        "selected_threshold": float(threshold),
        "test_accuracy": float(test_acc),
        "test_precision_no_defect": float(prec[0]),
        "test_recall_no_defect": float(rec[0]),
        "test_f1_no_defect": float(f1[0]),
        "test_precision_defect": float(prec[1]),
        "test_recall_defect": float(rec[1]),
        "test_f1_defect": float(f1[1]),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        }
    }

    # Save JSON
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # Save Markdown report
    md_path = Path(md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    md_content = f"""# ResNet50 Split Test Evaluation Results

## Model Information
* **Best Epoch:** {best_epoch}
* **Selected Threshold from Validation:** {threshold:.3f}

## Final Test Metrics
| Metric | Value |
|---|---|
| Test Accuracy | {test_acc:.4f} |
| Test Precision (no-defect) | {prec[0]:.4f} |
| Test Recall (no-defect) | {rec[0]:.4f} |
| Test F1-Score (no-defect) | {f1[0]:.4f} |
| Test Precision (defect) | {prec[1]:.4f} |
| Test Recall (defect) | {rec[1]:.4f} |
| Test F1-Score (defect) | {f1[1]:.4f} |
| Macro Precision | {macro_prec:.4f} |
| Macro Recall | {macro_rec:.4f} |
| Macro F1-Score | {macro_f1:.4f} |
| Weighted F1-Score | {weighted_f1:.4f} |
| ROC-AUC | {roc_auc:.4f} |

## Confusion Matrix (Test Set)
* **True Negative (TN):** {tn} (no-defect correctly classified)
* **False Positive (FP):** {fp} (no-defect misclassified as defect)
* **False Negative (FN):** {fn} (defect misclassified as no-defect)
* **True Positive (TP):** {tp} (defect correctly classified)

| Actual \\ Predicted | Predicted: no-defect | Predicted: defect |
|---|---|---|
| **Actual: no-defect** | {tn} | {fp} |
| **Actual: defect** | {fn} | {tp} |

## Quality Gate Status
* **Defect Recall (Target >= 0.80):** {"PASS ✅" if rec[1] >= 0.80 else "FAIL ❌"} ({rec[1]:.4f})
* **Defect F1 (Target >= 0.85):** {"PASS ✅" if f1[1] >= 0.85 else "FAIL ❌"} ({f1[1]:.4f})
* **Macro F1 (Target >= 0.85):** {"PASS ✅" if macro_f1 >= 0.85 else "FAIL ❌"} ({macro_f1:.4f})
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nFinal Test Metrics saved to:")
    print(f"  JSON: {json_path}")
    print(f"  MD  : {md_path}")

    return metrics


# ── Main Training Script Entrypoint ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet50 for Defect Detection with pre-split Train/Val/Test directories."
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/image_dataset_split",
        help="Root split directory containing train/, val/, and test/ subdirs"
    )
    parser.add_argument("--train-dir", type=str, default=None, help="Explicit training subset directory")
    parser.add_argument("--val-dir", type=str, default=None, help="Explicit validation subset directory")
    parser.add_argument("--test-dir", type=str, default=None, help="Explicit test subset directory")
    parser.add_argument("--epochs", type=int, default=25, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="FC head learning rate")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation fraction (fallback split)")
    parser.add_argument(
        "--oversample", type=int, default=10,
        help="Times to repeat each defect image in train set (default=10)."
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true", default=False,
        help="Freeze ResNet50 backbone, train FC head only"
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=2.0,
        help="Focal Loss gamma"
    )
    parser.add_argument(
        "--patience", type=int, default=8,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--save-path", type=str, default="ai_engine/models/resnet50_defect_split_best.pth",
        help="Checkpoint save path"
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.5,
        help="Dropout in classification head"
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--unfreeze-layer4", action="store_true", default=True,
        help="Unfreeze ResNet50 layer4 + FC head only"
    )
    parser.add_argument(
        "--backbone-lr", type=float, default=1e-5,
        help="Learning rate for layer4 backbone"
    )
    parser.add_argument(
        "--metrics-output", type=str, default="reports/resnet50_split_test_metrics.json",
        help="Output JSON file path for final test evaluation metrics"
    )
    parser.add_argument(
        "--figures-dir", type=str, default="reports/figures",
        help="Directory to save generated metric figures"
    )
    args = parser.parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data Loaders ---
    print(f"\nLoading split data from: {args.data_dir}")
    print(f"Oversampling defect class: {args.oversample}x")

    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        oversample_defect=args.oversample,
        train_dir=args.train_dir,
        val_dir=args.val_dir
    )

    train_total = len(train_loader.dataset)
    val_total = len(val_loader.dataset)
    print(f"Train samples (after oversampling): {train_total} | Validation samples: {val_total}")

    # --- Class distribution in train set ---
    n_normal, n_defect = count_train_labels(train_loader)
    print(f"Train class distribution -> no-defect: {n_normal}, defect: {n_defect}")
    ratio = n_normal / max(n_defect, 1)
    print(f"Ratio no-defect:defect = {ratio:.1f}:1")

    # --- Class weights for FocalLoss ---
    if n_defect > 0 and n_normal > 0:
        weight_normal = train_total / (2.0 * n_normal)
        weight_defect = train_total / (2.0 * n_defect)
        class_weights = torch.tensor([weight_normal, weight_defect], dtype=torch.float32).to(device)
        print(f"Class weights -> no-defect: {weight_normal:.4f}, defect: {weight_defect:.4f}")
    else:
        class_weights = None
        print("[WARNING] One class has 0 samples — not using class weights.")

    # --- Val class distribution ---
    val_labels_all = []
    for _, lbl in val_loader:
        val_labels_all.extend(lbl.numpy())
    n_val_defect = sum(1 for l in val_labels_all if l == 1)
    n_val_normal = sum(1 for l in val_labels_all if l == 0)
    print(f"Val class distribution  -> no-defect: {n_val_normal}, defect: {n_val_defect}")

    # --- Load Test Loader ---
    test_dir = args.test_dir or (str(Path(args.data_dir) / "test") if (Path(args.data_dir) / "test").exists() else None)
    if test_dir and Path(test_dir).exists():
        test_dataset = ProductDefectDataset(data_dir=test_dir, is_train=False, oversample_defect=1)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(f"Test samples: {len(test_dataset)}")
    else:
        test_loader = None
        print("[INFO] Test directory not found. Skipping final test evaluation.")

    # --- Model Creation ---
    # Disable pretrained ImageNet weights download if we are resuming from checkpoint to avoid Kaggle no-internet crashes
    use_pretrained = (args.resume_from is None)

    if args.unfreeze_layer4:
        base_model = get_resnet50_model(
            num_classes=2,
            freeze_backbone=True,
            dropout_rate=args.dropout_rate,
            pretrained=use_pretrained
        )
        for name, param in base_model.named_parameters():
            if name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        model = base_model
        training_mode = "layer4"
    else:
        model = get_resnet50_model(
            num_classes=2,
            freeze_backbone=args.freeze_backbone,
            dropout_rate=args.dropout_rate,
            pretrained=use_pretrained
        )
        training_mode = "frozen" if args.freeze_backbone else "full"

    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"\nTraining mode  : {training_mode}")
    print(f"Trainable params: {trainable:,} / {total_p:,} ({100*trainable/total_p:.1f}%)")

    # --- Loss & Optimizer ---
    criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    print(f"Loss: FocalLoss (gamma={args.focal_gamma})")

    if args.unfreeze_layer4:
        backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("layer4")]
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("layer4")]
        optimizer = optim.Adam([
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.lr},
        ])
        print(f"Optimizer: Adam | FC lr={args.lr:.1e} | layer4 lr={args.backbone_lr:.1e}")
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        print(f"Optimizer: Adam | lr={args.lr:.1e}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # --- Resume/Load Weights from Checkpoint ---
    best_defect_f1_resume = 0.0
    resume_epoch = 0
    if args.resume_from and Path(args.resume_from).exists():
        print(f"\n[WEIGHTS] Loading from: {args.resume_from}")
        prev_ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        
        # Determine if it's a training checkpoint or a raw state dict
        if isinstance(prev_ckpt, dict) and "model_state_dict" in prev_ckpt:
            state_dict = prev_ckpt["model_state_dict"]
            resume_epoch = prev_ckpt.get("epoch", 0)
            best_defect_f1_resume = prev_ckpt.get("defect_f1", 0.0)
            print(f"  Detected training checkpoint. Resuming from epoch {resume_epoch} | best_defect_f1={best_defect_f1_resume:.4f}")
            
            # Load optimizer state if resuming
            try:
                optimizer.load_state_dict(prev_ckpt["optimizer_state_dict"])
                print("  Successfully loaded optimizer state.")
            except Exception as e:
                print(f"  [WARN] Could not restore optimizer state: {e}")
        else:
            state_dict = prev_ckpt
            print("  Detected raw state dict or pretrained weights file.")
            resume_epoch = 0
            best_defect_f1_resume = 0.0
            
        # Load state dict (strict try, non-strict fallback)
        try:
            model.load_state_dict(state_dict, strict=True)
            print("  Successfully loaded weights (strict mode).")
        except RuntimeError as e:
            print(f"  [WARN] Strict loading failed: {e}")
            print("  Attempting non-strict loading (strict=False) for backbone layers...")
            model.load_state_dict(state_dict, strict=False)
            print("  Successfully loaded matching backbone weights. FC head initialized randomly.")
    else:
        if args.resume_from:
            print(f"\n[ERROR] Checkpoint path not found: {args.resume_from}")
            sys.exit(1)
        best_defect_f1_resume = 0.0
        resume_epoch = 0

    # --- Training Loop ---
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    best_defect_f1 = best_defect_f1_resume
    best_ckpt_meta = {}
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_defect_f1s = []

    remaining_epochs = max(args.epochs - resume_epoch, 0)
    print(f"\n{'='*70}")
    print(f"  Training: {args.epochs} total epochs | resumed from epoch {resume_epoch}")
    print(f"  Remaining: {remaining_epochs} epochs | lr={args.lr} | patience={args.patience}")
    print(f"{'='*70}\n")

    if remaining_epochs > 0:
        for epoch in range(resume_epoch + 1, args.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            elapsed = time.time() - t0

            defect_precision, defect_recall, defect_f1, macro_f1 = compute_defect_metrics(val_preds, val_labels)

            # Keep history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            val_defect_f1s.append(defect_f1)

            print(
                f"Epoch [{epoch:02d}/{args.epochs}] "
                f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | "
                f"Defect P={defect_precision:.3f} R={defect_recall:.3f} F1={defect_f1:.3f} | "
                f"Macro F1={macro_f1:.3f} | LR={current_lr:.2e} | {elapsed:.1f}s"
            )

            # Checkpoint selection based on Validation defect F1
            if defect_f1 > best_defect_f1:
                best_defect_f1 = defect_f1
                patience_counter = 0

                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

                best_ckpt_meta = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_names": ["no-defect", "defect"],
                    "label_mapping": {"no-defect": 0, "defect": 1},
                    "train_count": train_total,
                    "val_count": val_total,
                    "train_defect_count": n_defect,
                    "train_normal_count": n_normal,
                    "val_defect_count": n_val_defect,
                    "val_normal_count": n_val_normal,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "defect_precision": defect_precision,
                    "defect_recall": defect_recall,
                    "defect_f1": defect_f1,
                    "macro_f1": macro_f1,
                    "threshold": 0.5,
                    "args": vars(args),
                    "training_mode": training_mode,
                    "device": str(device),
                    "gpu_name": gpu_name,
                }
                torch.save(best_ckpt_meta, args.save_path)
                print(f"  -> Saved best checkpoint (defect_f1={defect_f1:.3f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n[EARLY STOPPING] defect_f1 did not improve for {args.patience} epochs.")
                    break

        # Plot training stats
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, val_defect_f1s, args.figures_dir)

    print(f"\n{'='*70}")
    print(f"  Training stage finished!")
    print(f"{'='*70}\n")

    if not Path(args.save_path).exists():
        print("[WARNING] No checkpoint exists at the save path.")
        return

    # --- Load Best Model ---
    print(f"Loading best model checkpoint from {args.save_path} for tuning and test evaluation...")
    checkpoint = torch.load(args.save_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # --- Threshold Tuning on Validation ---
    best_threshold, thresholds, precisions, recalls, f1s, val_stats = tune_threshold_on_val(model, val_loader, device)
    plot_threshold_tuning(thresholds, precisions, recalls, f1s, best_threshold, args.figures_dir)

    # Update checkpoint with best threshold
    checkpoint["threshold"] = best_threshold
    torch.save(checkpoint, args.save_path)
    print(f"Checkpoint threshold updated to {best_threshold:.3f}")

    # --- Final Evaluation on Test Set ---
    if test_loader is not None:
        evaluate_on_test_set(
            model=model,
            test_loader=test_loader,
            threshold=best_threshold,
            device=device,
            figures_dir=args.figures_dir,
            json_path=args.metrics_output,
            md_path=Path(args.metrics_output).with_suffix(".md"),
            best_epoch=checkpoint.get("epoch")
        )
    else:
        print("[INFO] Test set is not available. Skipping official evaluation step.")


if __name__ == "__main__":
    main()