# -*- coding: utf-8 -*-
"""
tune_threshold.py
-----------------
Script to find the optimal threshold for ResNet50 Defect Detection.

Runs inference on the entire validation set across multiple thresholds,
plots the Precision-Recall curve, and finds the best balanced threshold.

Usage:
    python scripts/tune_threshold.py
    python scripts/tune_threshold.py --model-path ai_engine/models/resnet50_defect.pth
    python scripts/tune_threshold.py --target-recall 0.80
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Fix Windows console encoding (avoid UnicodeEncodeError with cp1252)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
)

from ai_engine.image_processing.defect_detection import (
    get_dataloaders,
    get_resnet50_model,
    FocalLoss,
)


def collect_val_probs(model_path: str, data_dir: str, val_split: float, seed: int,
                      batch_size: int, oversample: int):
    """
    Load model and run inference on the entire validation set.
    Returns (defect_probs, true_labels).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = get_resnet50_model(num_classes=2, freeze_backbone=True, dropout_rate=0.5)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    f1 = checkpoint.get("defect_f1", 0.0)
    recall = checkpoint.get("defect_recall", 0.0)
    print(f"Loaded checkpoint: epoch={epoch}, best_defect_f1={f1:.3f}, best_recall={recall:.3f}")

    # Build val loader (NO oversampling)
    _, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=val_split,
        oversample_defect=1,  # Do not oversample validation set
        seed=seed,
    )

    print(f"Val samples: {len(val_loader.dataset)}")

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(defect)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)


def find_best_threshold(probs, labels, target_recall: float = None):
    """
    Find the optimal threshold based on F1 or target recall.

    Args:
        probs: Array of P(defect) for each sample.
        labels: Ground truth labels (0/1).
        target_recall: If set, finds the minimum threshold that achieves recall >= target_recall.

    Returns:
        dict containing information on the optimal threshold.
    """
    thresholds = np.arange(0.05, 0.95, 0.025)

    results = []
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        results.append({
            "threshold": round(float(thresh), 3),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
        })

    # Print results table
    print(f"\n{'='*75}")
    print(f"  Threshold Sweep — Defect Class Performance")
    print(f"{'='*75}")
    print(f"  {'Threshold':>10} | {'Precision':>10} | {'Recall':>8} | {'F1':>8} | {'TP':>4} | {'FP':>4} | {'FN':>4}")
    print(f"  {'-'*73}")
    for r in results:
        marker = ""
        if r["threshold"] == 0.5:
            marker = "  <-- default"
        print(
            f"  {r['threshold']:>10.3f} | {r['precision']:>10.4f} | {r['recall']:>8.4f} | "
            f"{r['f1']:>8.4f} | {r['tp']:>4} | {r['fp']:>4} | {r['fn']:>4}{marker}"
        )

    # Find best threshold
    if target_recall is not None:
        # Find the smallest threshold achieving recall >= target
        candidates = [r for r in results if r["recall"] >= target_recall]
        if candidates:
            best = max(candidates, key=lambda r: r["f1"])
            print(f"\n  >> Target recall >= {target_recall}: best threshold = {best['threshold']}")
        else:
            best = max(results, key=lambda r: r["recall"])
            print(f"\n  [WARNING] Target recall {target_recall} not met. Best recall = {best['recall']} at threshold={best['threshold']}")
    else:
        # Maximize F1
        best = max(results, key=lambda r: r["f1"])
        print(f"\n  → Best F1 threshold = {best['threshold']} (F1={best['f1']:.4f})")

    return best, results


def print_final_report(probs, labels, threshold: float, model_path: str):
    """Print complete classification report with the selected threshold."""
    preds = (probs >= threshold).astype(int)

    print(f"\n{'='*70}")
    print(f"  Final Evaluation — Threshold = {threshold}")
    print(f"{'='*70}")

    print("\nClassification Report:")
    print(classification_report(
        labels, preds,
        target_names=["no-defect", "defect"],
        zero_division=0,
    ))

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(f"  {'':>12} Pred:no-defect  Pred:defect")
    print(f"  True:no-defect  {cm[0][0]:>10}  {cm[0][1]:>10}   (FP={cm[0][1]})")
    if cm.shape[0] > 1:
        print(f"  True:defect     {cm[1][0]:>10}  {cm[1][1]:>10}   (FN={cm[1][0]})")

    try:
        auc = roc_auc_score(labels, probs)
        print(f"\nROC-AUC score: {auc:.4f}")
    except Exception:
        pass

    defect_total = int(np.sum(labels == 1))
    tp = int(cm[1][1]) if cm.shape[0] > 1 else 0
    print(f"\nDefect caught: {tp}/{defect_total} = {tp/max(defect_total,1)*100:.1f}%")
    print(f"Model path: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Threshold Tuning for ResNet50 Defect Detection")
    parser.add_argument("--model-path", type=str,
                        default="ai_engine/models/resnet50_defect.pth",
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--data-dir", type=str, default="data/image_dataset",
                        help="Path to data directory (default: data/image_dataset)")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--oversample", type=int, default=15,
                        help="Oversample factor used during training (for consistent split)")
    parser.add_argument("--target-recall", type=float, default=None,
                        help="Target minimum recall for defect class (e.g. 0.80). "
                             "If set, picks threshold maximizing F1 while recall >= target.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Evaluate a specific threshold (skips sweep, just shows report)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  ResNet50 Defect Detection — Threshold Tuning")
    print(f"{'='*70}")
    print(f"  Model: {args.model_path}")
    print(f"  Data:  {args.data_dir}")

    # Collect val set probabilities
    probs, labels = collect_val_probs(
        model_path=args.model_path,
        data_dir=args.data_dir,
        val_split=args.val_split,
        seed=args.seed,
        batch_size=args.batch_size,
        oversample=args.oversample,
    )

    n_defect = int(np.sum(labels == 1))
    n_normal = int(np.sum(labels == 0))
    print(f"Val distribution: no-defect={n_normal}, defect={n_defect}")

    if args.threshold is not None:
        # Evaluate a specific threshold only
        print_final_report(probs, labels, args.threshold, args.model_path)
        return

    # Threshold sweep
    best, all_results = find_best_threshold(probs, labels, target_recall=args.target_recall)

    # Final report with the best threshold
    print_final_report(probs, labels, best["threshold"], args.model_path)

    print(f"\n{'='*70}")
    print(f"  [TIP] Recommended: Use threshold={best['threshold']} in detect_defect_resnet()")
    print(f"        Or set env var: DEFECT_THRESHOLD={best['threshold']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
