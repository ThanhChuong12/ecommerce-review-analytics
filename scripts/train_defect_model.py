"""
train_defect_model.py
---------------------
Train ResNet50 for binary defect detection: no-defect (0) vs defect (1).

v4 improvements over v3:
  - --unfreeze-layer4: freeze conv1/layer1-3, unfreeze layer4 + FC head
  - --backbone-lr: differential learning rate for backbone layers (default 1e-5)
  - checkpoint now stores: training_mode, device, gpu_name
  - all existing flags (--freeze-backbone, --no-freeze, --resume-from) unchanged

training_mode values saved in checkpoint:
  'frozen'  — only FC head trained (--freeze-backbone, default)
  'layer4'  — layer4 + FC trained (--unfreeze-layer4)
  'full'    — all layers trained (--no-freeze)

Usage:
    # Frozen FC head only (original CPU behaviour)
    python scripts/train_defect_model.py \\
        --data-dir data/image_dataset \\
        --epochs 30 --batch-size 32 --lr 0.001 \\
        --oversample 10 --patience 12 \\
        --save-path ai_engine/models/resnet50_defect.pth

    # Kaggle GPU — layer4 + FC fine-tuning (recommended)
    python scripts/train_defect_model.py \\
        --data-dir data/image_dataset \\
        --epochs 25 --batch-size 64 \\
        --lr 5e-4 --backbone-lr 1e-5 \\
        --oversample 10 --patience 8 \\
        --unfreeze-layer4 \\
        --save-path ai_engine/models/resnet50_defect_gpu_layer4.pth

    # Resume from existing checkpoint
    python scripts/train_defect_model.py \\
        --data-dir data/image_dataset \\
        --epochs 30 --batch-size 32 --lr 0.0005 \\
        --oversample 10 --patience 12 \\
        --resume-from ai_engine/models/resnet50_defect.pth \\
        --save-path ai_engine/models/resnet50_defect.pth
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from ai_engine.image_processing.defect_detection import (
    get_dataloaders,
    get_resnet50_model,
    FocalLoss,
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
    Works with Subset wrapping used by get_dataloaders().
    """
    try:
        subset = train_loader.dataset          # torch.utils.data.Subset
        indices = subset.indices               # list of indices into the underlying dataset
        base_dataset = subset.dataset          # ProductDefectDataset
        labels = [base_dataset.labels[i] for i in indices]
        n_defect = sum(1 for l in labels if l == 1)
        n_normal = sum(1 for l in labels if l == 0)
    except AttributeError:
        # Fallback: iterate through all batches (slow)
        n_defect = 0
        n_normal = 0
        for _, lbl in train_loader:
            n_defect += (lbl == 1).sum().item()
            n_normal += (lbl == 0).sum().item()
    return n_normal, n_defect


def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet50 for Defect Detection (v3 — full metadata checkpoint)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/image_dataset",
        help="ImageFolder directory with defect/ and no-defect/ subdirs (default: data/image_dataset)"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction")
    parser.add_argument(
        "--oversample", type=int, default=20,
        help="Times to repeat each defect image in train set (default=20). "
             "With ~10:1 imbalance, 20x brings ratio to ~1:1."
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true", default=True,
        help="Freeze ResNet50 backbone, train FC head only (default: True)"
    )
    parser.add_argument(
        "--no-freeze", dest="freeze_backbone", action="store_false",
        help="Unfreeze backbone to train all layers"
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=2.0,
        help="Focal Loss gamma (0 = standard CE). Higher = more focus on hard samples"
    )
    parser.add_argument(
        "--patience", type=int, default=12,
        help="Early stopping patience (epochs without defect_f1 improvement). "
             "Default 12 for safe long-running CPU training."
    )
    parser.add_argument(
        "--save-path", type=str, default="ai_engine/models/resnet50_defect.pth",
        help="Checkpoint save path"
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.5,
        help="Dropout in classification head"
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to checkpoint to resume training from (loads model+optimizer state)"
    )
    # ── Kaggle GPU fine-tuning options ──────────────────────────────────────
    parser.add_argument(
        "--unfreeze-layer4", action="store_true", default=False,
        help="Unfreeze ResNet50 layer4 + FC head only (conv1/layer1-3 stay frozen). "
             "Recommended for Kaggle GPU fine-tuning. Overrides --freeze-backbone."
    )
    parser.add_argument(
        "--backbone-lr", type=float, default=1e-5,
        help="Learning rate for unfrozen backbone layers when --unfreeze-layer4 is used. "
             "Default 1e-5 (smaller than FC lr to avoid catastrophic forgetting)."
    )
    args = parser.parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    print(f"\nLoading data from: {args.data_dir}")
    print(f"Oversampling defect class: {args.oversample}x")

    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        oversample_defect=args.oversample,
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

    # --- Model ---
    # Determine training mode: layer4 > full-unfreeze > frozen
    if args.unfreeze_layer4:
        # Start with everything frozen
        base_model = get_resnet50_model(
            num_classes=2,
            freeze_backbone=True,       # freeze all first
            dropout_rate=args.dropout_rate,
        )
        # Then selectively unfreeze layer4
        for name, param in base_model.named_parameters():
            # Unfreeze layer4 and the FC head (fc / classifier)
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
        )
        training_mode = "frozen" if args.freeze_backbone else "full"

    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"\nTraining mode  : {training_mode}")
    print(f"Freeze backbone: {args.freeze_backbone and not args.unfreeze_layer4}")
    print(f"Trainable params: {trainable:,} / {total_p:,} ({100*trainable/total_p:.1f}%)")

    # --- Loss & Optimizer ---
    criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    print(f"Loss: FocalLoss (gamma={args.focal_gamma})")

    if args.unfreeze_layer4:
        # Differential learning rates: layer4 gets backbone_lr, FC head gets lr
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and n.startswith("layer4")]
        head_params     = [p for n, p in model.named_parameters()
                           if p.requires_grad and not n.startswith("layer4")]
        optimizer = optim.Adam([
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params,     "lr": args.lr},
        ])
        print(f"Optimizer: Adam | FC lr={args.lr:.1e} | layer4 lr={args.backbone_lr:.1e}")
    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
        )
        print(f"Optimizer: Adam | lr={args.lr:.1e}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # --- Resume from checkpoint (optional) ---
    best_defect_f1_resume = 0.0
    resume_epoch = 0
    if args.resume_from and Path(args.resume_from).exists():
        print(f"\n[RESUME] Loading checkpoint from: {args.resume_from}")
        prev_ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(prev_ckpt["model_state_dict"])
        try:
            optimizer.load_state_dict(prev_ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"  [WARN] Could not restore optimizer state: {e}")
        resume_epoch = prev_ckpt.get("epoch", 0)
        best_defect_f1_resume = prev_ckpt.get("defect_f1", 0.0)
        print(f"  Resuming from epoch {resume_epoch} | best_defect_f1={best_defect_f1_resume:.4f}")
    else:
        best_defect_f1_resume = 0.0
        resume_epoch = 0

    # --- Training Loop ---
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    best_defect_f1 = best_defect_f1_resume
    best_ckpt_meta = {}
    patience_counter = 0

    remaining_epochs = max(args.epochs - resume_epoch, 0)
    print(f"\n{'='*70}")
    print(f"  Training: {args.epochs} total epochs | resumed from epoch {resume_epoch}")
    print(f"  Remaining: {remaining_epochs} epochs | lr={args.lr} | patience={args.patience}")
    print(f"{'='*70}\n")

    if remaining_epochs == 0:
        print("[INFO] Already reached target epochs. Nothing to train.")
        return

    for epoch in range(resume_epoch + 1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        defect_precision, defect_recall, defect_f1, macro_f1 = compute_defect_metrics(
            val_preds, val_labels
        )

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | "
            f"Defect P={defect_precision:.3f} R={defect_recall:.3f} F1={defect_f1:.3f} | "
            f"Macro F1={macro_f1:.3f} | LR={current_lr:.2e} | {elapsed:.1f}s"
        )

        # Save best model based on defect F1
        if defect_f1 > best_defect_f1:
            best_defect_f1 = defect_f1
            patience_counter = 0

            # GPU info
            gpu_name = "N/A"
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)

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
                "threshold": 0.5,     # default; update after tune_threshold.py
                "args": vars(args),
                "training_mode": training_mode,
                "device": str(device),
                "gpu_name": gpu_name,
            }
            torch.save(best_ckpt_meta, args.save_path)
            print(
                f"  -> Saved best (defect_f1={defect_f1:.3f} "
                f"P={defect_precision:.3f} R={defect_recall:.3f} macro_f1={macro_f1:.3f})"
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] defect_f1 did not improve for {args.patience} epochs.")
                break

    # --- Final Report ---
    print(f"\n{'='*70}")
    print(f"  Training complete! Best defect_f1: {best_defect_f1:.4f}")
    print(f"{'='*70}\n")

    if not Path(args.save_path).exists():
        print("[WARNING] No checkpoint was saved (model never improved defect_f1 above 0). "
              "Check dataset and class balance.")
        return

    # Load best and run final evaluation
    checkpoint = torch.load(args.save_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, final_preds, final_labels = evaluate(model, val_loader, criterion, device)

    print("Classification Report (Best Checkpoint on Validation Set):")
    print(classification_report(
        final_labels, final_preds,
        target_names=["no-defect", "defect"],
        zero_division=0,
    ))

    cm = confusion_matrix(final_labels, final_preds)
    print("Confusion Matrix:")
    print(f"  {'':>12} Pred:no-defect  Pred:defect")
    print(f"  True:no-defect  {cm[0][0]:>10}  {cm[0][1]:>10}")
    if cm.shape[0] > 1:
        print(f"  True:defect     {cm[1][0]:>10}  {cm[1][1]:>10}")

    print(f"\nCheckpoint saved to: {args.save_path}")
    print(f"  epoch={checkpoint.get('epoch')}")
    print(f"  defect_f1={checkpoint.get('defect_f1', 0):.4f}")
    print(f"  defect_recall={checkpoint.get('defect_recall', 0):.4f}")
    print(f"  macro_f1={checkpoint.get('macro_f1', 0):.4f}")
    print(f"  val_acc={checkpoint.get('val_acc', 0):.4f}")


if __name__ == "__main__":
    main()