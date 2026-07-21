import os
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "processed" / "processed_labeled_all.csv"

OUTPUT_TRAIN = ROOT / "data" / "processed" / "processed_labeled_text_train.csv"
OUTPUT_VAL = ROOT / "data" / "processed" / "processed_labeled_text_val.csv"
OUTPUT_TEST = ROOT / "data" / "processed" / "processed_labeled_text_test.csv"

def main():
    logger.info(f"Reading data from: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        logger.error("Original data file not found!")
        sys.exit(1)
        
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df = df.dropna(subset=["cleaned_text", "sentiment_label"])
    
    logger.info(f"Total valid samples: {len(df)}")
    
    # Step 1: Split Train (70%) and Rest (30%)
    # Use stratify to keep label ratio
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30, 
        random_state=42, 
        stratify=df["sentiment_label"]
    )
    
    # Step 2: Split Rest (30%) into Val (15%) and Test (15%)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, # 50% of 30% is 15%
        random_state=42, 
        stratify=temp_df["sentiment_label"]
    )
    
    logger.info(f"Data split successfully (Stratified):")
    logger.info(f" - Train : {len(train_df)} samples ({len(train_df)/len(df):.1%})")
    logger.info(f" - Val   : {len(val_df)} samples ({len(val_df)/len(df):.1%})")
    logger.info(f" - Test  : {len(test_df)} samples ({len(test_df)/len(df):.1%})")
    
    logger.info("-" * 50)
    logger.info("LABEL RATIO BEFORE AND AFTER SPLITTING:")
    
    def get_dist(data):
        dist = data["sentiment_label"].value_counts(normalize=True) * 100
        return " | ".join([f"{k}: {v:.2f}%" for k, v in dist.items()])
        
    logger.info(f"[Original] {get_dist(df)}")
    logger.info(f"[Train]    {get_dist(train_df)}")
    logger.info(f"[Val]      {get_dist(val_df)}")
    logger.info(f"[Test]     {get_dist(test_df)}")
    logger.info("-" * 50)
    
    # Save to file
    train_df.to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8-sig")
    val_df.to_csv(OUTPUT_VAL, index=False, encoding="utf-8-sig")
    test_df.to_csv(OUTPUT_TEST, index=False, encoding="utf-8-sig")
    
    logger.info("Saved 3 CSV files:")
    logger.info(f"1. {OUTPUT_TRAIN}")
    logger.info(f"2. {OUTPUT_VAL}")
    logger.info(f"3. {OUTPUT_TEST}")

if __name__ == "__main__":
    main()
