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
    logger.info(f"Đọc dữ liệu từ: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        logger.error("Không tìm thấy file dữ liệu gốc!")
        sys.exit(1)
        
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df = df.dropna(subset=["cleaned_text", "sentiment_label"])
    
    logger.info(f"Tổng số mẫu hợp lệ: {len(df)}")
    
    # Bước 1: Tách Train (70%) và Phần còn lại (30%)
    # Dùng stratify để giữ nguyên tỉ lệ lớp (tích cực/tiêu cực/trung lập)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30, 
        random_state=42, 
        stratify=df["sentiment_label"]
    )
    
    # Bước 2: Tách phần còn lại (30%) thành Val (15%) và Test (15%)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, # 50% của 30% là 15%
        random_state=42, 
        stratify=temp_df["sentiment_label"]
    )
    
    logger.info(f"Chia dữ liệu thành công (Stratified):")
    logger.info(f" - Train : {len(train_df)} mẫu ({len(train_df)/len(df):.1%})")
    logger.info(f" - Val   : {len(val_df)} mẫu ({len(val_df)/len(df):.1%})")
    logger.info(f" - Test  : {len(test_df)} mẫu ({len(test_df)/len(df):.1%})")
    
    logger.info("-" * 50)
    logger.info("THỐNG KÊ TỈ LỆ NHÃN TRƯỚC VÀ SAU KHI CHIA:")
    
    def get_dist(data):
        dist = data["sentiment_label"].value_counts(normalize=True) * 100
        return " | ".join([f"{k}: {v:.2f}%" for k, v in dist.items()])
        
    logger.info(f"[Gốc]   {get_dist(df)}")
    logger.info(f"[Train] {get_dist(train_df)}")
    logger.info(f"[Val]   {get_dist(val_df)}")
    logger.info(f"[Test]  {get_dist(test_df)}")
    logger.info("-" * 50)
    
    # Lưu ra file
    train_df.to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8-sig")
    val_df.to_csv(OUTPUT_VAL, index=False, encoding="utf-8-sig")
    test_df.to_csv(OUTPUT_TEST, index=False, encoding="utf-8-sig")
    
    logger.info("Đã lưu 3 file CSV:")
    logger.info(f"1. {OUTPUT_TRAIN}")
    logger.info(f"2. {OUTPUT_VAL}")
    logger.info(f"3. {OUTPUT_TEST}")

if __name__ == "__main__":
    main()
