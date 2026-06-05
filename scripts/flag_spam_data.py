import os
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import logging
import pandas as pd
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import ai_engine
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_engine.text_processing.spam_filter import detect_spam, summarize_spam

INPUT_CSV = ROOT / "data" / "processed" / "processed_labeled_all.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "spam_labeled_text.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info(f"Đọc dữ liệu từ: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        logger.error(f"Không tìm thấy file: {INPUT_CSV}")
        return

    logger.info(f"Tổng số mẫu: {len(df)}")
    
    # Chạy bộ lọc rule-based spam
    logger.info("Bắt đầu chạy bộ lọc spam...")
    df_flagged = detect_spam(df, dup_threshold=0.85)
    
    # In thống kê
    stats = summarize_spam(df_flagged)
    logger.info("=" * 50)
    logger.info("THỐNG KÊ SPAM:")
    logger.info(f" - Tổng review: {stats['total_reviews']}")
    logger.info(f" - Spam: {stats['spam_count']} ({stats['spam_pct']}%)")
    logger.info(f" - Sạch: {stats['clean_count']} ({stats['clean_pct']}%)")
    logger.info("=" * 50)

    # Lưu kết quả
    df_flagged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"Đã lưu kết quả hoàn chỉnh ra: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
