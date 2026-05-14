"""
train_defect_model.py
---------------------
Script huan luyen mo hinh ResNet50 de phan loai hang loi (defect) va hang binh thuong (no-defect).

Usage:
    # Mac dinh: 10 epochs
    python scripts/train_defect_model.py
    
    # Tuy chinh
    python scripts/train_defect_model.py --epochs 20 --batch-size 32 --lr 0.0001
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
    parser = argparse.ArgumentParser(description="Train ResNet50 for Defect Detection")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Path to data directory containing defect/ and no-defect/")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data used for validation")
    parser.add_argument("--save-path", type=str, default="ai_engine/models/resnet50_defect.pth",
                        help="Path to save the best model checkpoint")
    args = parser.parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    print(f"Loading data from: {args.data_dir}")
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )

    train_total = len(train_loader.dataset)
    val_total = len(val_loader.dataset)
    print(f"Train samples: {train_total} | Validation samples: {val_total}")

    # --- Count class distribution in training set ---
    train_labels = [train_loader.dataset.dataset.labels[i] for i in train_loader.dataset.indices]
    n_defect = sum(1 for l in train_labels if l == 1)
    n_normal = sum(1 for l in train_labels if l == 0)
    print(f"Train class distribution -> no-defect: {n_normal}, defect: {n_defect}")

    # --- Class weights (inverse frequency) to handle imbalance ---
    if n_defect > 0 and n_normal > 0:
        weight_normal = train_total / (2.0 * n_normal)
        weight_defect = train_total / (2.0 * n_defect)
        class_weights = torch.tensor([weight_normal, weight_defect], dtype=torch.float32).to(device)
        print(f"Class weights -> no-defect: {weight_normal:.4f}, defect: {weight_defect:.4f}")
    else:
        class_weights = None
        print("[WARNING] One class has 0 samples, not using class weights.")

    # --- Model ---
    model = get_resnet50_model(num_classes=2)
    model = model.to(device)

    # --- Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler: giam lr khi val_loss khong giam sau 3 epoch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # --- Training Loop ---
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    print(f"\n{'='*60}")
    print(f"  Starting training: {args.epochs} epochs, lr={args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
            f"| LR: {current_lr:.6f} "
            f"| Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, args.save_path)
            print(f"  -> Saved best model (val_acc={val_acc:.4f}) to {args.save_path}")

    # --- Final Evaluation ---
    print(f"\n{'='*60}")
    print(f"  Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}\n")

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
    print(f"  True:defect     {cm[1][0]:>10}  {cm[1][1]:>10}")


if __name__ == "__main__":
    main()
