"""
split_image_dataset.py
----------------------
Tách toàn bộ ảnh trong labeled/labeled/ thành 3 tập vật lý:
  labeled/train/   — 70%  dùng để train model
  labeled/val/     — 15%  dùng để early stopping / tune hyperparams
  labeled/test/    — 15%  dùng để đánh giá cuối cùng (KHÔNG chạm trong training)

Dùng StratifiedShuffleSplit để đảm bảo tỉ lệ class giống nhau ở cả 3 tập.

⚠️  QUAN TRỌNG: Chỉ chạy script này MỘT LẦN DUY NHẤT.
    Test set phải được cố định trước khi bất kỳ training nào bắt đầu.
    Nếu chạy lại sẽ bị chặn trừ khi truyền --force.

Cách dùng:
    python scripts/split_image_dataset.py
    python scripts/split_image_dataset.py --train 0.70 --val 0.15 --test 0.15
    python scripts/split_image_dataset.py --dry-run      # xem phân bố mà không copy
    python scripts/split_image_dataset.py --force        # ghi đè nếu đã split rồi

Cấu trúc đầu ra:
    labeled/
    ├── labeled/        ← giữ nguyên (nguồn gốc)
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

# Thư mục gốc project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Nguồn: thư mục labeled gốc
SOURCE_DIR = PROJECT_ROOT / "labeled" / "labeled"

# Đích: 3 tập mới
SPLIT_DIRS = {
    "train": PROJECT_ROOT / "labeled" / "train",
    "val":   PROJECT_ROOT / "labeled" / "val",
    "test":  PROJECT_ROOT / "labeled" / "test",
}

# File lock để tránh chạy lại
SPLIT_MANIFEST = PROJECT_ROOT / "labeled" / "split_manifest.json"

# Class names theo ImageFolder format
CLASS_NAMES = ["intact", "damaged", "wrong_item", "irrelevant"]

# Định dạng ảnh được chấp nhận
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_images(source_dir: Path) -> Tuple[List[Path], List[str]]:
    """Thu thập tất cả ảnh và nhãn từ source_dir (ImageFolder format)."""
    all_paths: List[Path] = []
    all_labels: List[str] = []

    for cls in CLASS_NAMES:
        cls_dir = source_dir / cls
        if not cls_dir.exists():
            logger.warning("Không tìm thấy thư mục class: %s — bỏ qua", cls_dir)
            continue
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        all_paths.extend(imgs)
        all_labels.extend([cls] * len(imgs))
        logger.info("  %-15s: %6d ảnh", cls, len(imgs))

    return all_paths, all_labels


def _print_split_stats(name: str, labels: List[str]) -> None:
    """In phân bố class của 1 tập."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    logger.info("  %s (%d ảnh):", name, total)
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
    """Copy ảnh vào labeled/<split_name>/<class>/. Trả về số ảnh đã copy."""
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
        logger.info("  [%s] %d ảnh đã tồn tại — bỏ qua", split_name, skipped)

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
    # Kiểm tra tổng tỉ lệ
    total = round(train_ratio + val_ratio + test_ratio, 6)
    if abs(total - 1.0) > 1e-6:
        logger.error("Tổng tỉ lệ train+val+test phải bằng 1.0, hiện tại = %.4f", total)
        sys.exit(1)

    # Kiểm tra đã split chưa
    if SPLIT_MANIFEST.exists() and not force:
        logger.error(
            "Dataset đã được split trước đó!\n"
            "  Manifest: %s\n"
            "  Dùng --force để ghi đè (KHÔNG khuyến nghị sau khi đã train).",
            SPLIT_MANIFEST,
        )
        sys.exit(1)

    if not SOURCE_DIR.exists():
        logger.error(
            "Không tìm thấy thư mục nguồn: %s\n"
            "Hãy chắc chắn ảnh đã được gán nhãn và copy vào labeled/labeled/",
            SOURCE_DIR,
        )
        sys.exit(1)

    # --- Thu thập ảnh ---
    logger.info("=" * 60)
    logger.info("Đọc ảnh từ: %s", SOURCE_DIR)
    all_paths, all_labels = _collect_images(SOURCE_DIR)
    total_images = len(all_paths)

    if total_images == 0:
        logger.error("Không tìm thấy ảnh nào trong %s", SOURCE_DIR)
        sys.exit(1)

    logger.info("Tổng cộng: %d ảnh", total_images)
    logger.info("=" * 60)

    # --- Chia indices ---
    from sklearn.model_selection import StratifiedShuffleSplit

    # Bước 1: Tách test trước (giữ hoàn toàn sạch)
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    trainval_idx, test_idx = next(
        sss_test.split(range(total_images), all_labels)
    )

    # Bước 2: Chia trainval → train + val
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

    # Tập hợp paths và labels cho mỗi split
    splits = {
        "train": ([all_paths[i] for i in train_idx], [all_labels[i] for i in train_idx]),
        "val":   ([all_paths[i] for i in val_idx],   [all_labels[i] for i in val_idx]),
        "test":  ([all_paths[i] for i in test_idx],  [all_labels[i] for i in test_idx]),
    }

    # --- In thống kê ---
    logger.info("Phân bố sau khi split (seed=%d):", seed)
    for split_name, (paths, labels) in splits.items():
        _print_split_stats(split_name, labels)

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN — Không có file nào được copy.")
        logger.info("Chạy lại KHÔNG có --dry-run để thực hiện thật.")
        return

    # --- Copy ảnh ---
    logger.info("=" * 60)
    logger.info("Bắt đầu copy ảnh...")

    total_copied = 0
    for split_name, (paths, labels) in splits.items():
        logger.info("  Đang copy tập '%s' (%d ảnh)...", split_name, len(paths))
        copied = _copy_images(paths, labels, split_name, dry_run=False)
        total_copied += copied
        logger.info("  ✓ %s: %d ảnh đã copy", split_name, copied)

    # --- Lưu manifest ---
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
    logger.info("✅ Hoàn tất! Đã copy %d / %d ảnh.", total_copied, total_images)
    logger.info("Manifest lưu tại: %s", SPLIT_MANIFEST)
    logger.info("")
    logger.info("Các bước tiếp theo:")
    logger.info("  1. Train:    python scripts/train_image_baseline.py --data-dir labeled/train")
    logger.info("  2. Evaluate: python scripts/train_image_baseline.py --eval-only --data-dir labeled/test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tách dataset ảnh thành train/val/test (chạy 1 lần duy nhất)"
    )
    parser.add_argument("--train", type=float, default=0.70, help="Tỉ lệ train (mặc định: 0.70)")
    parser.add_argument("--val",   type=float, default=0.15, help="Tỉ lệ val   (mặc định: 0.15)")
    parser.add_argument("--test",  type=float, default=0.15, help="Tỉ lệ test  (mặc định: 0.15)")
    parser.add_argument("--seed",  type=int,   default=42,   help="Random seed (mặc định: 42)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chỉ in thống kê, không copy file nào",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ghi đè dù dataset đã split rồi (KHÔNG khuyến nghị sau khi đã train)",
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
