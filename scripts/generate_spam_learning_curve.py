"""
scripts/generate_spam_learning_curve.py
=======================================
Tự động tạo "Learning Curve" cho Đồ án (dành cho Isolation Forest).
Vẽ đường cong hiệu suất (F1-Score) dựa trên siêu tham số `contamination`.
Lưu ảnh và log vào `artifacts/spam/`.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Chèn thư mục gốc để import module
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_engine.text_processing.spam_filter import detect_spam
from ai_engine.text_processing.spam_model import build_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
TRAIN_CSV = ROOT / "data" / "processed" / "spam_train.csv"
VAL_CSV = ROOT / "data" / "processed" / "spam_val.csv"
OUT_DIR = ROOT / "artifacts" / "spam"
OUT_LOG = OUT_DIR / "spam_learning_logs.csv"
OUT_PLOT = OUT_DIR / "spam_learning_curve.png"

def load_and_extract(csv_path: Path, is_train: bool = False):
    logger.info(f"Loading {csv_path.name}...")
    df = pd.read_csv(csv_path)
    
    # 1. Chạy luật
    df_flagged = detect_spam(df, dup_threshold=0.85)
    
    # 2. Xây dựng ma trận đặc trưng
    X = build_feature_matrix(df_flagged, df["text"].tolist(), df["rating"].tolist())
    
    # Lấy ground truth (nếu không phải tập train)
    y_true = None
    if not is_train:
        y_true = df["is_spam"].values.astype(int)
    else:
        # Đối với tập train (không nhãn), ta lấy nhãn giả định từ luật để vẽ đường "Train F1"
        y_true = df_flagged["is_spam"].values.astype(int)
        
    return X, y_true

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, y_train_rule = load_and_extract(TRAIN_CSV, is_train=True)
    X_val, y_val_true = load_and_extract(VAL_CSV, is_train=False)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Thông số thay thế Epochs: Contamination
    contaminations = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    results = []
    
    logger.info("Bắt đầu mô phỏng quá trình học (Tuning Curve)...")
    for cont in contaminations:
        logger.info(f"  Training với contamination = {cont:.2f}")
        model = IsolationForest(
            n_estimators=200, 
            contamination=cont, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train_scaled)
        
        # Dự đoán
        pred_train = (model.predict(X_train_scaled) == -1).astype(int)
        pred_val = (model.predict(X_val_scaled) == -1).astype(int)
        
        # Tính F1-Score (Macro)
        f1_train = f1_score(y_train_rule, pred_train, average="macro")
        f1_val = f1_score(y_val_true, pred_val, average="macro")
        
        results.append({
            "Contamination": cont,
            "Train_F1": f1_train,
            "Validation_F1": f1_val
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUT_LOG, index=False)
    logger.info(f"Đã lưu logs vào {OUT_LOG}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["Contamination"], df_results["Train_F1"], marker='o', linestyle='-', color='blue', label='Train')
    plt.plot(df_results["Contamination"], df_results["Validation_F1"], marker='s', linestyle='-', color='orange', label='Validation')
    
    plt.title("Learning Curve - Spam Model (Isolation Forest)")
    plt.xlabel("Contamination (Tỷ lệ ngoại lệ kỳ vọng)")
    plt.ylabel("Macro F1-Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Highlight max validation F1
    max_val_idx = df_results["Validation_F1"].idxmax()
    best_cont = df_results.loc[max_val_idx, "Contamination"]
    best_f1 = df_results.loc[max_val_idx, "Validation_F1"]
    plt.annotate(f"Best: {best_f1:.3f}",
                 xy=(best_cont, best_f1),
                 xytext=(best_cont, best_f1 + 0.02),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300)
    logger.info(f"Đã lưu biểu đồ vào {OUT_PLOT}")
    logger.info("Hoàn tất!")

if __name__ == "__main__":
    main()
