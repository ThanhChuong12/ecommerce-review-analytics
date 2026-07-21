"""
scripts/split_spam_data.py
==========================
Preprocess and split spam dataset (Train/Val/Test).
- Input: spam_labeled_text.csv
- Ratio: 70/15/15
- Note: Labels are removed from Train set for unsupervised training.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "processed" / "spam_labeled_text.csv"
OUT_TRAIN = ROOT / "data" / "processed" / "spam_train.csv"
OUT_VAL = ROOT / "data" / "processed" / "spam_val.csv"
OUT_TEST = ROOT / "data" / "processed" / "spam_test.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info(f"Loading data from: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        logger.error(f"File not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    total_samples = len(df)
    logger.info(f"Total samples: {total_samples}")

    # Shuffle & Split 70-15-15
    # train: 70%, temp: 30%
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    # val: 50% of temp (15%), test: 50% of temp (15%)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    logger.info("Processing labels for datasets...")
    
    # ── Train set MUST NOT contain labels (Unsupervised) ──
    # Drop 'is_spam' column if it exists
    if "is_spam" in train_df.columns:
        train_df = train_df.drop(columns=["is_spam"])
    
    # Val and Test sets keep labels for evaluation
    
    logger.info(f"Train set (Unlabeled): {len(train_df)} samples")
    logger.info(f"Val set   (Labeled)  : {len(val_df)} samples")
    logger.info(f"Test set  (Labeled)  : {len(test_df)} samples")

    # Save files
    train_df.to_csv(OUT_TRAIN, index=False, encoding="utf-8-sig")
    val_df.to_csv(OUT_VAL, index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_TEST, index=False, encoding="utf-8-sig")

    logger.info("Dataset split and saving completed!")

if __name__ == "__main__":
    main()
