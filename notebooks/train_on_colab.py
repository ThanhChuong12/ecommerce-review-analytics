# ============================================================
# COLAB TRAINING SCRIPT - Defect Detection with ResNet50
# ============================================================
# Huong dan su dung:
#   1. Upload file nay len Google Colab (hoac copy paste tung cell)
#   2. Bat GPU: Runtime -> Change runtime type -> GPU (T4)
#   3. Upload file training_images.zip len Google Drive
#   4. Chay tung cell tu tren xuong
# ============================================================

# %% [markdown]
# # Defect Detection - ResNet50 Training on Colab
# **Du an**: Multimodal Review Analytics  
# **Muc tieu**: Phan loai anh san pham loi (defect) vs binh thuong (no-defect)

# %% --- CELL 1: Kiem tra GPU ---
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[WARNING] No GPU detected! Go to Runtime -> Change runtime type -> GPU")

# %% --- CELL 2: Clone repo tu GitHub ---
!git clone -b feature/Image_Augmentation https://github.com/ThanhChuong12/ecommerce-review-analytics.git
%cd ecommerce-review-analytics

# %% --- CELL 3: Cai dat thu vien ---
!pip install -q albumentations opencv-python-headless scikit-learn

# %% --- CELL 4: Mount Google Drive va copy du lieu ---
from google.colab import drive
drive.mount('/content/drive')

# === QUAN TRONG ===
# Truoc khi chay cell nay, ban can:
# 1. Upload file training_images.zip len Google Drive (thu muc goc "My Drive")
# 2. Neu ban de o thu muc khac, sua duong dan ben duoi cho dung

import zipfile
import os

# Duong dan toi file zip tren Google Drive
# Sua lai cho dung neu ban de o thu muc khac
DRIVE_ZIP_PATH = "/content/drive/MyDrive/training_images.zip"

if os.path.exists(DRIVE_ZIP_PATH):
    print(f"Found zip file: {DRIVE_ZIP_PATH}")
    print("Extracting...")
    with zipfile.ZipFile(DRIVE_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall("data/processed/")
    print("Done!")
else:
    print(f"[ERROR] File not found: {DRIVE_ZIP_PATH}")
    print("Please upload training_images.zip to your Google Drive root folder.")

# Kiem tra so luong anh
defect_count = len(os.listdir("data/processed/defect")) if os.path.exists("data/processed/defect") else 0
no_defect_count = len(os.listdir("data/processed/no-defect")) if os.path.exists("data/processed/no-defect") else 0
print(f"\nDataset loaded:")
print(f"  defect:    {defect_count} images")
print(f"  no-defect: {no_defect_count} images")

# %% --- CELL 5: Bat dau Training ---
# Chay script training voi GPU
# Tuy chinh tham so neu can: --epochs, --batch-size, --lr
!python scripts/train_defect_model.py \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.0001

# %% --- CELL 6: Download model da train ---
# Sau khi train xong, download model ve may local
from google.colab import files
import shutil

# Copy model sang Drive de luu tru lau dai
MODEL_PATH = "ai_engine/models/resnet50_defect.pth"
DRIVE_SAVE_PATH = "/content/drive/MyDrive/resnet50_defect.pth"

if os.path.exists(MODEL_PATH):
    shutil.copy(MODEL_PATH, DRIVE_SAVE_PATH)
    print(f"Model saved to Google Drive: {DRIVE_SAVE_PATH}")
    
    # Hoac download truc tiep ve may
    # files.download(MODEL_PATH)
else:
    print("[ERROR] Model file not found. Training may not have completed.")
