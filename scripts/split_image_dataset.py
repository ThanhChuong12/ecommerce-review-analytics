"""
split_image_dataset.py
----------------------
Split all images in labeled/labeled/ into 3 physical subsets:
  labeled/train/   — 70%  used to train models
  labeled/val/     — 15%  used for early stopping / tuning hyperparameters
  labeled/test/    — 15%  used for final evaluation (NOT touched during training)

Uses StratifiedShuffleSplit to ensure class ratios are identical across all 3 subsets.

⚠️  IMPORTANT: Only run this script ONCE.
    The test set must be fixed before any training starts.
    Subsequent runs will be blocked unless --force is specified.

Usage:
    python scripts/split_image_dataset.py
    python scripts/split_image_dataset.py --train 0.70 --val 0.15 --test 0.15
    python scripts/split_image_dataset.py --dry-run      # view distribution without copying
    python scripts/split_image_dataset.py --force        # overwrite if split already exists

Output structure:
    labeled/
    ├── labeled/        ← unchanged (original source)
    ├── train/          intact/ damaged/ wrong_item/ irrelevant/
    ├── val/            intact/ damaged/ wrong_item/ irrelevant/
    └── test/           intact/ damaged/ wrong_item/ irrelevant/
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Source: original labeled directory
SOURCE_DIR = PROJECT_ROOT / "labeled" / "labeled"

# Destination: 3 new subsets
SPLIT_DIRS = {
    "train": PROJECT_ROOT / "labeled" / "train",
    "val":   PROJECT_ROOT / "labeled" / "val",
    "test":  PROJECT_ROOT / "labeled" / "test",
}

# Manifest lock file to prevent subsequent runs
SPLIT_MANIFEST = PROJECT_ROOT / "labeled" / "split_manifest.json"

# Class names in ImageFolder format
CLASS_NAMES = ["intact", "damaged", "wrong_item", "irrelevant"]

# Allowed image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_images(source_dir: Path) -> Tuple[List[Path], List[str]]:
    """Collect all images and labels from source_dir (ImageFolder format)."""
    all_paths: List[Path] = []
    all_labels: List[str] = []

    for cls in CLASS_NAMES:
        cls_dir = source_dir / cls
        if not cls_dir.exists():
            logger.warning("Class directory not found: %s — skipping", cls_dir)
            continue
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        all_paths.extend(imgs)
        all_labels.extend([cls] * len(imgs))
        logger.info("  %-15s: %6d images", cls, len(imgs))

    return all_paths, all_labels


def _print_split_stats(name: str, labels: List[str]) -> None:
    """Print class distribution for a subset."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    logger.info("  %s (%d images):", name, total)
    for cls in CLASS_NAMES:
        n = counts.get(cls, 0)
        pct = n / total * 100 if total > 0 else 0
        logger.info("    %-15s: %5d (%.1f%%)", cls, n, pct)


def _copy_images(
    paths: List[Path],
    labels: List[str],
    split_name: str,
    dry_run: bool,
) -> int:
    """Copy images into labeled/<split_name>/<class>/. Returns the number of copied images."""
    dest_root = SPLIT_DIRS[split_name]
    copied = 0
    skipped = 0

    for img_path, label in zip(paths, labels):
        dest_dir = dest_root / label
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / img_path.name

        if dest_file.exists():
            skipped += 1
            continue

        if not dry_run:
            shutil.copy2(img_path, dest_file)
        copied += 1

    if skipped > 0:
        logger.info("  [%s] %d images already exist — skipping", split_name, skipped)

    return copied


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def split(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    dry_run: bool,
    force: bool,
) -> None:
    # Verify split ratios sum to 1.0
    total = round(train_ratio + val_ratio + test_ratio, 6)
    if abs(total - 1.0) > 1e-6:
        logger.error("Sum of train+val+test ratios must equal 1.0, currently = %.4f", total)
        sys.exit(1)

    # Check if already split
    if SPLIT_MANIFEST.exists() and not force:
        logger.error(
            "Dataset has already been split!\n"
            "  Manifest: %s\n"
            "  Use --force to overwrite (NOT recommended after training has started).",
            SPLIT_MANIFEST,
        )
        sys.exit(1)

    if not SOURCE_DIR.exists():
        logger.error(
            "Source directory not found: %s\n"
            "Ensure images are labeled and copied to labeled/labeled/",
            SOURCE_DIR,
        )
        sys.exit(1)

    # --- Collect images ---
    logger.info("=" * 60)
    logger.info("Reading images from: %s", SOURCE_DIR)
    all_paths, all_labels = _collect_images(SOURCE_DIR)
    total_images = len(all_paths)

    if total_images == 0:
        logger.error("No images found in %s", SOURCE_DIR)
        sys.exit(1)

    logger.info("Total: %d images", total_images)
    logger.info("=" * 60)

    # --- Split indices ---
    from sklearn.model_selection import StratifiedShuffleSplit

    # Step 1: Separate test set first (keeping it completely clean)
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    trainval_idx, test_idx = next(
        sss_test.split(range(total_images), all_labels)
    )

    # Step 2: Split trainval → train + val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio_adjusted, random_state=seed
    )
    trainval_labels = [all_labels[i] for i in trainval_idx]
    train_local_idx, val_local_idx = next(
        sss_val.split(range(len(trainval_idx)), trainval_labels)
    )

    train_idx = [trainval_idx[i] for i in train_local_idx]
    val_idx   = [trainval_idx[i] for i in val_local_idx]

    # Aggregate paths and labels for each split
    splits = {
        "train": ([all_paths[i] for i in train_idx], [all_labels[i] for i in train_idx]),
        "val":   ([all_paths[i] for i in val_idx],   [all_labels[i] for i in val_idx]),
        "test":  ([all_paths[i] for i in test_idx],  [all_labels[i] for i in test_idx]),
    }

    # --- Print statistics ---
    logger.info("Distribution after split (seed=%d):", seed)
    for split_name, (paths, labels) in splits.items():
        _print_split_stats(split_name, labels)

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN — No files were copied.")
        logger.info("Re-run WITHOUT --dry-run to perform the actual split.")
        return

    # --- Copy images ---
    logger.info("=" * 60)
    logger.info("Starting image copying...")

    total_copied = 0
    for split_name, (paths, labels) in splits.items():
        logger.info("  Copying '%s' subset (%d images)...", split_name, len(paths))
        copied = _copy_images(paths, labels, split_name, dry_run=False)
        total_copied += copied
        logger.info("  ✓ %s: %d images copied", split_name, copied)

    # --- Save manifest ---
    manifest_data = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "total_images": total_images,
        "split_counts": {
            name: len(paths) for name, (paths, _) in splits.items()
        },
        "class_counts": {
            name: {
                cls: labels.count(cls) for cls in CLASS_NAMES
            }
            for name, (_, labels) in splits.items()
        },
        "source_dir": str(SOURCE_DIR),
        "split_dirs": {k: str(v) for k, v in SPLIT_DIRS.items()},
    }
    SPLIT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("✅ Done! Copied %d / %d images.", total_copied, total_images)
    logger.info("Manifest saved at: %s", SPLIT_MANIFEST)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Train:    python scripts/train_image_baseline.py --data-dir labeled/train")
    logger.info("  2. Evaluate: python scripts/train_image_baseline.py --eval-only --data-dir labeled/test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split image dataset into train/val/test (run once)"
    )
    parser.add_argument("--train", type=float, default=0.70, help="Train ratio (default: 0.70)")
    parser.add_argument("--val",   type=float, default=0.15, help="Val ratio (default: 0.15)")
    parser.add_argument("--test",  type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument("--seed",  type=int,   default=42,   help="Random seed (default: 42)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print statistics, do not copy any files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite even if dataset has already been split (NOT recommended after training)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split(
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        dry_run=args.dry_run,
        force=args.force,
    )
