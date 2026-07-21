import os
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import logging
import pandas as pd
from pathlib import Path

# Add root directory to sys.path to import ai_engine
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_engine.text_processing.spam_filter import detect_spam, summarize_spam

INPUT_CSV = ROOT / "data" / "processed" / "processed_labeled_all.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "spam_labeled_text.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info(f"Reading data from: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        logger.error(f"File not found: {INPUT_CSV}")
        return

    logger.info(f"Total samples: {len(df)}")
    
    # Run rule-based spam filter
    logger.info("Running spam filter...")
    df_flagged = detect_spam(df, dup_threshold=0.85)
    
    # Print statistics
    stats = summarize_spam(df_flagged)
    logger.info("=" * 50)
    logger.info("SPAM STATISTICS:")
    logger.info(f" - Total reviews: {stats['total_reviews']}")
    logger.info(f" - Spam: {stats['spam_count']} ({stats['spam_pct']}%)")
    logger.info(f" - Clean: {stats['clean_count']} ({stats['clean_pct']}%)")
    logger.info("=" * 50)

    # Save results
    df_flagged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"Saved complete results to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
