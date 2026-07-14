"""
scripts/split_spam_data.py
==========================
Tiền xử lý và chia tách tập dữ liệu đánh giá Spam (Train/Val/Test).
- Đầu vào: spam_labeled_text.csv
- Tỷ lệ: 70/15/15
- Đặc biệt: Loại bỏ cột nhãn khỏi tập Train để huấn luyện Unsupervised.
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
    logger.info(f"Đang tải dữ liệu từ: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        logger.error(f"Không tìm thấy file: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    total_samples = len(df)
    logger.info(f"Tổng số mẫu hiện có: {total_samples}")

    # Shuffle & Split 70-15-15
    # train: 70%, temp: 30%
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    # val: 50% của temp (15%), test: 50% của temp (15%)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    logger.info("Đang xử lý cấu trúc nhãn cho các tập dữ liệu...")
    
    # ── MỤC TIÊU BẢO MẬT: Tập Train KHÔNG được chứa nhãn (Unsupervised) ──
    # Xóa cột 'is_spam' (hoặc bất kỳ cột nhãn nào liên quan) nếu có
    if "is_spam" in train_df.columns:
        train_df = train_df.drop(columns=["is_spam"])
    
    # Tập Val và Test giữ nguyên nhãn để phục vụ đánh giá (Evaluation)
    
    logger.info(f"Tập Train (Không nhãn): {len(train_df)} mẫu")
    logger.info(f"Tập Val   (Có nhãn)   : {len(val_df)} mẫu")
    logger.info(f"Tập Test  (Có nhãn)   : {len(test_df)} mẫu")

    # Lưu file
    train_df.to_csv(OUT_TRAIN, index=False, encoding="utf-8-sig")
    val_df.to_csv(OUT_VAL, index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_TEST, index=False, encoding="utf-8-sig")

    logger.info("Hoàn tất chia tách và lưu file dữ liệu!")

if __name__ == "__main__":
    main()
