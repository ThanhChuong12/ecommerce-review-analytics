"""
train_defect_model.py
---------------------
Script huan luyen mo hinh ResNet50 de phan loai hang loi (defect) va hang binh thuong (no-defect).

Phien ban v2 - Cai thien Recall cho lop defect:
  1. Oversampling: Lap lai anh defect nhieu lan de can bang du lieu
  2. Focal Loss: Tap trung vao cac mau kho (defect bi nham thanh no-defect)
  3. Freeze Backbone: Dong bang cac layer CNN, chi train FC layer
  4. Early Stopping: Dung train khi Val Loss khong giam de chong overfitting

Usage:
    # Mac dinh (da toi uu cho dataset mat can bang)
    python scripts/train_defect_model.py

    # Tuy chinh
    python scripts/train_defect_model.py --epochs 30 --batch-size 32 --lr 0.001 --oversample 15
"""

import sys
import time
import argparse
from pathlib import Path

# Them root project vao sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

from ai_engine.image_processing.defect_detection import (
    get_dataloaders,
    get_resnet50_model,
    FocalLoss,
)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Huan luyen 1 epoch. Tra ve (loss trung binh, accuracy)."""
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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, val_loader, criterion, device):
    """Danh gia tren tap Validation. Tra ve (loss, accuracy, all_preds, all_labels)."""
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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 for Defect Detection (v2)")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Path to data directory containing defect/ and no-defect/")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Max number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (higher because backbone is frozen)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data used for validation")
    parser.add_argument("--oversample", type=int, default=15,
                        help="Number of times to repeat each defect image (default=15 -> ~1:2 ratio)")
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
                        help="Freeze ResNet50 backbone, only train FC layer (default: True)")
    parser.add_argument("--no-freeze", dest="freeze_backbone", action="store_false",
                        help="Train all layers (no freeze)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Gamma for Focal Loss (0 = standard CE, higher = more focus on hard samples)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (stop after N epochs without improvement)")
    parser.add_argument("--save-path", type=str, default="ai_engine/models/resnet50_defect.pth",
                        help="Path to save the best model checkpoint")
    parser.add_argument("--dropout-rate", type=float, default=0.5,
                        help="Dropout rate in the classification head (default=0.5)")
    args = parser.parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    print(f"Loading data from: {args.data_dir}")
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

    # --- Count class distribution in training set ---
    train_labels = [train_loader.dataset.dataset.labels[i] for i in train_loader.dataset.indices]
    n_defect = sum(1 for l in train_labels if l == 1)
    n_normal = sum(1 for l in train_labels if l == 0)
    print(f"Train class distribution -> no-defect: {n_normal}, defect: {n_defect}")
    print(f"Ratio no-defect:defect = {n_normal/max(n_defect,1):.1f}:1")

    # --- Class weights for Focal Loss ---
    if n_defect > 0 and n_normal > 0:
        weight_normal = train_total / (2.0 * n_normal)
        weight_defect = train_total / (2.0 * n_defect)
        class_weights = torch.tensor([weight_normal, weight_defect], dtype=torch.float32).to(device)
        print(f"Class weights -> no-defect: {weight_normal:.4f}, defect: {weight_defect:.4f}")
    else:
        class_weights = None
        print("[WARNING] One class has 0 samples, not using class weights.")

    # --- Model ---
    model = get_resnet50_model(num_classes=2, freeze_backbone=args.freeze_backbone,
                               dropout_rate=args.dropout_rate)
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    # --- Loss & Optimizer ---
    criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    print(f"Using FocalLoss (gamma={args.focal_gamma})")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # --- Training Loop with Early Stopping ---
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    # ✅ Theo dõi Defect F1 — buoc model can bang ca Recall lan Precision
    # Tranh truong hop model "gian lan" bang cach bao tat ca la defect de recall = 100%
    best_defect_f1 = 0.0
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"  Training config: {args.epochs} epochs, lr={args.lr}, patience={args.patience}")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

        # Step scheduler (van dung val_loss de dieu chinh lr)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Tinh Defect Recall
        defect_correct = sum(1 for p, l in zip(val_preds, val_labels) if p == 1 and l == 1)
        defect_total = sum(1 for l in val_labels if l == 1)
        defect_recall = defect_correct / max(defect_total, 1)

        # Tinh Defect F1 de hien thi them
        defect_pred_total = sum(1 for p in val_preds if p == 1)
        defect_precision = defect_correct / max(defect_pred_total, 1)
        defect_f1 = (2 * defect_precision * defect_recall) / max(defect_precision + defect_recall, 1e-8)

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
            f"| Defect Recall: {defect_recall:.2f} F1: {defect_f1:.2f} "
            f"| LR: {current_lr:.6f} "
            f"| Time: {elapsed:.1f}s"
        )

        # ✅ Save model dua tren Defect F1 — can bang ca Recall lan Precision
        # F1 cao buoc model phai du doan dung defect (recall) ma khong bao nham qua nhieu (precision)
        if defect_f1 > best_defect_f1:
            best_defect_f1 = defect_f1
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "defect_recall": defect_recall,
                "defect_f1": defect_f1,
            }, args.save_path)
            print(f"  -> Saved best model (defect_f1={defect_f1:.2f}, recall={defect_recall:.2f}, val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] Defect F1 did not improve for {args.patience} epochs. Stopping.")
                break

    # --- Final Evaluation ---
    print(f"\n{'='*70}")
    print(f"  Training complete! Best defect_f1: {best_defect_f1:.4f}, val_acc: {best_val_acc:.4f}")
    print(f"{'='*70}\n")

    # Load best model and run final evaluation
    checkpoint = torch.load(args.save_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, final_preds, final_labels = evaluate(model, val_loader, criterion, device)

    print("Classification Report (Validation Set):")
    print(classification_report(
        final_labels, final_preds,
        target_names=["no-defect", "defect"],
        zero_division=0,
    ))

    print("Confusion Matrix:")
    cm = confusion_matrix(final_labels, final_preds)
    print(f"  {'':>12} Pred:no-defect  Pred:defect")
    print(f"  True:no-defect  {cm[0][0]:>10}  {cm[0][1]:>10}")
    if cm.shape[0] > 1:
        print(f"  True:defect     {cm[1][0]:>10}  {cm[1][1]:>10}")


if __name__ == "__main__":
    main()