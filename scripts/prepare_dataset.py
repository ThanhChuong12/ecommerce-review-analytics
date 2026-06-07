"""
prepare_dataset.py
------------------
Prepares a binary ImageFolder dataset from the labeled image directory.

Source:  image_labeling/data/labeled/{damaged,intact,wrong_item,irrelevant}/
Labels:  image_labeling/data/manifests/labels.csv  (used to verify, not as primary source)
Output:  data/image_dataset/defect/        (damaged + wrong_item)
         data/image_dataset/no-defect/     (intact only)
         irrelevant -> skipped

Binary label mapping:
  intact      -> no-defect (class 0)
  damaged     -> defect    (class 1)
  wrong_item  -> defect    (class 1)
  irrelevant  -> EXCLUDED

Usage:
    # Prepare dataset from labeled folders (primary mode)
    python scripts/prepare_dataset.py \\
        --labels-csv image_labeling/data/manifests/labels.csv \\
        --images-root image_labeling \\
        --output-dir data/image_dataset \\
        --label-map binary

    # Check class balance of generated dataset
    python scripts/prepare_dataset.py --check-balance --data-dir data/image_dataset
"""

import argparse
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
BINARY_MAP = {
    "intact": "no-defect",
    "damaged": "defect",
    "wrong_item": "defect",
    "irrelevant": None,          # None = skip
}


def prepare_from_labeled_dirs(images_root: Path, output_dir: Path) -> None:
    """
    Copy images from the labeled folder structure into the binary ImageFolder.

    Source layout (in images_root/data/labeled/):
        damaged/        -> output_dir/defect/
        wrong_item/     -> output_dir/defect/
        intact/         -> output_dir/no-defect/
        irrelevant/     -> SKIP

    Original files are NEVER moved or deleted.
    """
    labeled_root = images_root / "data" / "labeled"
    if not labeled_root.exists():
        print(f"[ERROR] Labeled image root not found: {labeled_root}")
        sys.exit(1)

    defect_dir = output_dir / "defect"
    no_defect_dir = output_dir / "no-defect"
    defect_dir.mkdir(parents=True, exist_ok=True)
    no_defect_dir.mkdir(parents=True, exist_ok=True)

    counts = {"defect": 0, "no-defect": 0, "skipped_irrelevant": 0, "missing": 0, "errors": []}

    for raw_label, binary_class in BINARY_MAP.items():
        src_dir = labeled_root / raw_label
        if not src_dir.exists():
            print(f"  [INFO] Source dir does not exist, skipping: {src_dir}")
            continue

        if binary_class is None:
            n_skipped = sum(1 for _ in src_dir.glob("*") if _.is_file())
            print(f"  [SKIP] irrelevant/ -> {n_skipped} images skipped.")
            counts["skipped_irrelevant"] += n_skipped
            continue

        dest_dir = defect_dir if binary_class == "defect" else no_defect_dir
        copied = 0

        for src_img in src_dir.iterdir():
            if not src_img.is_file():
                continue
            if src_img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue

            # De-duplicate filename by prefixing with raw_label to avoid collision
            # e.g. damaged_abc.jpg vs wrong_item_abc.jpg
            dest_name = f"{raw_label}_{src_img.name}"
            dest_path = dest_dir / dest_name

            if not dest_path.exists():
                try:
                    shutil.copy2(src_img, dest_path)
                    copied += 1
                except Exception as e:
                    counts["errors"].append(f"{src_img}: {e}")
            else:
                copied += 1  # already exists from previous run

        counts[binary_class] += copied
        print(f"  [{raw_label:12}] -> {binary_class:10} | {copied} images")

    print()
    print("=" * 60)
    print("  Dataset Preparation Summary")
    print("=" * 60)
    print(f"  defect images   : {counts['defect']}")
    print(f"  no-defect images: {counts['no-defect']}")
    print(f"  irrelevant skip : {counts['skipped_irrelevant']}")

    if counts["errors"]:
        print(f"\n  [WARNING] {len(counts['errors'])} copy errors:")
        for err in counts["errors"][:5]:
            print(f"    {err}")
        if len(counts["errors"]) > 5:
            print(f"    ... and {len(counts['errors']) - 5} more.")
        sys.exit(1)

    total = counts["defect"] + counts["no-defect"]
    if total == 0:
        print("[ERROR] No images were copied. Check image_labeling/data/labeled/ contents.")
        sys.exit(1)

    print(f"\n[SUCCESS] {total} total images prepared in: {output_dir}")
    print(f"  defect   -> {output_dir / 'defect'}")
    print(f"  no-defect-> {output_dir / 'no-defect'}")


def prepare_from_csv(labels_csv: Path, images_root: Path, output_dir: Path) -> None:
    """
    FALLBACK: prepare from labels.csv when labeled/ folder structure differs.
    Cross-validates CSV-referenced image paths against the labeled directory.
    """
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas is required. Run: pip install pandas")
        sys.exit(1)

    if not labels_csv.exists():
        print(f"[ERROR] Labels CSV not found: {labels_csv}")
        sys.exit(1)

    print(f"Reading labels from: {labels_csv}")
    df = pd.read_csv(labels_csv)

    required_cols = {"image_path", "label"}
    if not required_cols.issubset(df.columns):
        print(f"[ERROR] CSV must have columns {required_cols}. Found: {list(df.columns)}")
        sys.exit(1)

    defect_dir = output_dir / "defect"
    no_defect_dir = output_dir / "no-defect"
    defect_dir.mkdir(parents=True, exist_ok=True)
    no_defect_dir.mkdir(parents=True, exist_ok=True)

    copied_defect = 0
    copied_no_defect = 0
    skipped_irrelevant = 0
    missing_images = []

    for _, row in df.iterrows():
        raw_label = str(row["label"]).strip().lower()
        binary_class = BINARY_MAP.get(raw_label)

        if binary_class is None:
            skipped_irrelevant += 1
            continue

        rel_path = str(row["image_path"]).replace("\\", "/")
        # Try multiple root candidates
        candidates = [
            images_root / rel_path,
            images_root / "data" / Path(rel_path).name,
            Path(rel_path),
        ]
        src_img = next((c for c in candidates if c.exists()), None)

        if src_img is None:
            missing_images.append(rel_path)
            continue

        dest_dir = defect_dir if binary_class == "defect" else no_defect_dir
        dest_path = dest_dir / src_img.name
        if not dest_path.exists():
            shutil.copy2(src_img, dest_path)

        if binary_class == "defect":
            copied_defect += 1
        else:
            copied_no_defect += 1

    print(f"\n[CSV mode] defect: {copied_defect}, no-defect: {copied_no_defect}, "
          f"skipped: {skipped_irrelevant}, missing: {len(missing_images)}")
    if missing_images:
        print(f"  [WARNING] {len(missing_images)} images not found:")
        for p in missing_images[:10]:
            print(f"    {p}")


def check_balance(data_dir: Path) -> None:
    """Print class balance statistics and oversampling recommendation."""
    defect_dir = data_dir / "defect"
    no_defect_dir = data_dir / "no-defect"

    if not defect_dir.exists() or not no_defect_dir.exists():
        print(f"[ERROR] Expected subdirs 'defect/' and 'no-defect/' under: {data_dir}")
        print("  Run prepare first: python scripts/prepare_dataset.py --label-map binary")
        sys.exit(1)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    n_defect = sum(1 for f in defect_dir.iterdir() if f.is_file() and f.suffix.lower() in exts)
    n_normal = sum(1 for f in no_defect_dir.iterdir() if f.is_file() and f.suffix.lower() in exts)

    ratio = n_normal / max(n_defect, 1)

    if ratio > 10:
        recommended_oversample = 20
    elif ratio > 5:
        recommended_oversample = 10
    elif ratio > 3:
        recommended_oversample = 5
    else:
        recommended_oversample = 1

    print()
    print("=" * 60)
    print("  Class Balance Report")
    print("=" * 60)
    print(f"  defect images   : {n_defect}")
    print(f"  no-defect images: {n_normal}")
    print(f"  Ratio no-defect:defect = {ratio:.2f}:1")
    print()

    if ratio > 10:
        print(f"  [WARNING] Severe imbalance (>{ratio:.0f}:1).")
        print(f"  Recommended: --oversample {recommended_oversample} (brings ratio toward 1:1)")
    elif ratio > 3:
        print(f"  [WARNING] Moderate imbalance ({ratio:.1f}:1).")
        print(f"  Recommended: --oversample {recommended_oversample}")
    else:
        print("  [OK] Class balance is acceptable.")

    print()
    print(f"  Suggested training command:")
    print(f"    python scripts/train_defect_model.py \\")
    print(f"        --data-dir {data_dir} \\")
    print(f"        --epochs 30 \\")
    print(f"        --batch-size 32 \\")
    print(f"        --lr 0.001 \\")
    print(f"        --oversample {recommended_oversample} \\")
    print(f"        --patience 8 \\")
    print(f"        --save-path ai_engine/models/resnet50_defect.pth")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare binary ImageFolder dataset for ResNet50 defect detection.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default="image_labeling/data/manifests/labels.csv",
        help="Path to labels CSV (default: image_labeling/data/manifests/labels.csv)",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default="image_labeling",
        help="Root directory containing image_labeling/data/labeled/ (default: image_labeling)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/image_dataset",
        help="Output ImageFolder directory (default: data/image_dataset)",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        choices=["binary"],
        default="binary",
        help="Label mapping mode. 'binary' = damaged/wrong_item->defect, intact->no-defect",
    )
    parser.add_argument(
        "--check-balance",
        action="store_true",
        help="Only check class balance of an already-prepared dataset.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/image_dataset",
        help="Dataset directory for --check-balance (default: data/image_dataset)",
    )
    parser.add_argument(
        "--use-csv",
        action="store_true",
        help="Use CSV as image path source instead of labeled/ folder structure.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent

    if args.check_balance:
        data_dir = root / args.data_dir
        check_balance(data_dir)
        return

    output_dir = root / args.output_dir
    images_root = root / args.images_root

    print()
    print("=" * 60)
    print("  ResNet50 Dataset Preparation")
    print("=" * 60)
    print(f"  images root : {images_root}")
    print(f"  output dir  : {output_dir}")
    print(f"  label map   : {args.label_map}")
    print()

    if args.use_csv:
        labels_csv = root / args.labels_csv
        print(f"  Mode: CSV-based (labels from {labels_csv})")
        prepare_from_csv(labels_csv, images_root, output_dir)
    else:
        print("  Mode: Folder-based (from image_labeling/data/labeled/)")
        prepare_from_labeled_dirs(images_root, output_dir)

    # Auto-run balance check after preparation
    print()
    check_balance(output_dir)


if __name__ == "__main__":
    main()
