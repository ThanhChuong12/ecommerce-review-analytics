"""
build_paired_dataset.py
========================
Ghép cặp text + image từ:
  - Text CSVs: data/processed/processed_labeled_text_{train,val,test}.csv
  - Image manifest: ecommerce-backup/labeled_all/manifests/images.csv
  - Image files: ecommerce-backup/labeled_all/labeled/{class}/

Output:
  - data/processed/paired_text_image.csv  (review_text, sentiment_label, image_path, image_label)
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
BACKUP = Path("d:/ecommerce-backup/labeled_all")


def build_paired_dataset(output_path: str = "data/processed/paired_text_image.csv") -> pd.DataFrame:
    """Build paired text-image dataset."""

    # ── Load all text CSVs ─────────────────────────────────────────────────
    text_dfs = []
    for split in ["train", "val", "test"]:
        csv_path = ROOT / f"data/processed/processed_labeled_text_{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["split"] = split
            text_dfs.append(df)
            logger.info("Loaded %s: %d rows", csv_path.name, len(df))

    all_text = pd.concat(text_dfs, ignore_index=True)
    all_text["text_stripped"] = all_text["text"].str.strip()
    logger.info("Total text reviews: %d", len(all_text))

    # ── Load image manifest ────────────────────────────────────────────────
    img_csv = BACKUP / "manifests" / "images.csv"
    img_df = pd.read_csv(img_csv, encoding="utf-8-sig")
    img_df["text_stripped"] = img_df["review_text"].str.strip()
    logger.info("Total image rows in manifest: %d", len(img_df))

    # ── Build file lookup from labeled folders ─────────────────────────────
    labeled_dir = BACKUP / "labeled"
    file_map = {}  # filename -> (full_path, class_label)
    for cls in ["intact", "damaged", "irrelevant", "wrong_item"]:
        cls_dir = labeled_dir / cls
        if not cls_dir.exists():
            continue
        for f in os.listdir(cls_dir):
            file_map[f] = (str(cls_dir / f), cls)

    logger.info("Total image files in labeled folders: %d", len(file_map))

    # ── Match image rows to actual files ───────────────────────────────────
    img_df["plain_name"] = img_df["image_path"].apply(lambda x: os.path.basename(str(x)))
    img_df["has_file"] = img_df["plain_name"].isin(file_map)
    img_df["full_path"] = img_df["plain_name"].map(lambda x: file_map.get(x, (None, None))[0])
    img_df["image_label"] = img_df["plain_name"].map(lambda x: file_map.get(x, (None, None))[1])

    img_with_files = img_df[img_df["has_file"]].copy()
    logger.info("Image rows with actual files: %d", len(img_with_files))

    # ── Match text with images ─────────────────────────────────────────────
    # Merge on stripped text
    matched = img_with_files.merge(
        all_text[["text_stripped", "text", "sentiment_label", "rating", "split"]].drop_duplicates("text_stripped"),
        on="text_stripped",
        how="inner",
    )
    logger.info("Total matched rows (text+image): %d", len(matched))

    # ── Take first image per review (for denoiser training) ────────────────
    # Sort by review_id + frame_index to get the most representative image
    matched = matched.sort_values(["review_id", "frame_index"])
    paired = matched.groupby("text_stripped").first().reset_index()

    logger.info("Unique text-image pairs (1 image/review): %d", len(paired))

    # ── Save output ────────────────────────────────────────────────────────
    # Find correct column names (might have _x/_y suffix from merge)
    rating_col = "rating_y" if "rating_y" in paired.columns else "rating"
    
    output = paired[["text", "sentiment_label", rating_col, "full_path", "image_label", "review_id", "split"]].copy()
    output.columns = ["text", "sentiment_label", "rating", "image_path", "image_label", "review_id", "split"]

    out_path = ROOT / output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False, encoding="utf-8-sig")

    logger.info("Saved paired dataset → %s (%d rows)", out_path, len(output))

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PAIRED DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total pairs: {len(output)}")
    print(f"  Splits: {output['split'].value_counts().to_dict()}")
    print(f"  Sentiment labels: {output['sentiment_label'].value_counts().to_dict()}")
    print(f"  Image labels: {output['image_label'].value_counts().to_dict()}")
    print("=" * 60)

    return output


if __name__ == "__main__":
    build_paired_dataset()
