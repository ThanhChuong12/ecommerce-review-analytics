# ============================================================
# COLAB TRAINING — MobileNetV3 Defect Detection (4-class)
# Ecommerce Review Analytics
# ============================================================
# TRƯỚC KHI CHẠY:
#   1. Runtime → Change runtime type → T4 GPU
#   2. Upload lên Google Drive (thư mục gốc My Drive):
#        - labeled_data.zip       (ảnh đã gán nhãn)
#        - image_baseline.py      (bản đã fix bugs — lấy từ ai_engine/models/)
# ============================================================


# %% [markdown]
# # MobileNetV3 Defect Detection — Colab Training
# **Classes:** intact | damaged | wrong_item | irrelevant
# **Dataset:** ~27,743 ảnh (ImageFolder format)
# **Estimated time:** ~20-40 phút (T4 GPU)


# %% --- CELL 1: Kiểm tra GPU ---
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    raise RuntimeError("Chưa bật GPU! Vào Runtime → Change runtime type → T4 GPU")


# %% --- CELL 2: Clone repo ---
import os

REPO_URL    = "https://github.com/ThanhChuong12/ecommerce-review-analytics.git"
REPO_BRANCH = "main"
REPO_DIR    = "/content/ecommerce-review-analytics"

if not os.path.exists(REPO_DIR):
    os.system(f"git clone -b {REPO_BRANCH} {REPO_URL} {REPO_DIR}")
else:
    os.system(f"git -C {REPO_DIR} pull")

os.chdir(REPO_DIR)
print("Working dir:", os.getcwd())


# %% --- CELL 3: Cài dependencies ---
os.system("pip install -q albumentations opencv-python-headless scikit-learn Pillow")
print("Dependencies installed.")


# %% --- CELL 4: Apply bug fixes cho image_baseline.py ---
# FIX: Code trên main chưa có 4 bug fixes của MobileNetV3.
# Cần upload file image_baseline.py (đã fix) lên Google Drive trước.
#
# Cách lấy file đã fix trên máy local:
#   ai_engine\models\image_baseline.py  →  upload lên Google Drive
# -------------------------------------------------------
import shutil
from google.colab import drive

drive.mount("/content/drive")

FIXED_BASELINE = "/content/drive/MyDrive/image_baseline.py"
TARGET         = "ai_engine/models/image_baseline.py"

if os.path.exists(FIXED_BASELINE):
    shutil.copy(FIXED_BASELINE, TARGET)
    print("✅ Đã apply image_baseline.py đã fix bugs")
else:
    print("⚠️  CẢNH BÁO: Không tìm thấy image_baseline.py trên Drive!")
    print("   Upload file ai_engine/models/image_baseline.py từ máy local lên Drive trước.")
    print("   Tiếp tục với code gốc (có thể có bugs).")


# %% --- CELL 5: Giải nén data ---
# Upload labeled_data.zip lên Google Drive trước (thư mục gốc My Drive)
# Cách tạo zip đúng cấu trúc (chạy trên máy local):
#   Compress-Archive -Path "labeled\labeled" -DestinationPath "labeled_data.zip"
# -------------------------------------------------------
import zipfile

DRIVE_ZIP   = "/content/drive/MyDrive/labeled.zip"
EXTRACT_DIR = "/content/ecommerce-review-analytics/labeled/labeled"

if not os.path.exists(EXTRACT_DIR):
    if not os.path.exists(DRIVE_ZIP):
        raise FileNotFoundError(f"Không tìm thấy {DRIVE_ZIP}. Upload labeled_data.zip lên Google Drive trước.")
    print(f"Extracting {DRIVE_ZIP} ...")
    with zipfile.ZipFile(DRIVE_ZIP, "r") as z:
        z.extractall("/content/ecommerce-review-analytics/labeled/")
    print("Done!")
else:
    print("Data đã tồn tại, bỏ qua giải nén.")

# Kiểm tra số ảnh mỗi class
print("\nDataset summary:")
total = 0
for cls in ["intact", "damaged", "wrong_item", "irrelevant"]:
    folder = os.path.join(EXTRACT_DIR, cls)
    count  = len(os.listdir(folder)) if os.path.exists(folder) else 0
    total += count
    status = "✅" if count > 0 else "❌"
    print(f"  {status} {cls:15s}: {count:,} ảnh")
print(f"  {'TOTAL':15s}: {total:,} ảnh")


# %% --- CELL 6: Train ---
# FIX: Dùng subprocess thay os.system() để hiển thị log real-time
import subprocess, sys

cmd = [
    "python", "scripts/train_image_baseline.py",
    "--backbone",   "mobilenet_v3",
    "--data-dir",   "labeled/labeled",
    "--epochs",     "20",
    "--lr",         "1e-3",
    "--batch-size", "64",
    "--val-split",  "0.2",
    "--patience",   "5",
]

print("Bắt đầu training...\n" + "=" * 60)
result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stdout, text=True)
print("=" * 60)
print("Training kết thúc với exit code:", result.returncode)


# %% --- CELL 7: Lưu model về Google Drive ---
MODEL_SRC   = "ai_engine/models/weights/mobilenet_v3_defect.pt"
RESULTS_SRC = "ai_engine/models/results/image_baseline_results.json"

DRIVE_SAVE_DIR = "/content/drive/MyDrive/ecommerce_models"
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

if os.path.exists(MODEL_SRC):
    shutil.copy(MODEL_SRC, f"{DRIVE_SAVE_DIR}/mobilenet_v3_defect.pt")
    print(f"✅ Model saved → {DRIVE_SAVE_DIR}/mobilenet_v3_defect.pt")
else:
    print("❌ Model không tìm thấy — kiểm tra log ở CELL 6.")

if os.path.exists(RESULTS_SRC):
    shutil.copy(RESULTS_SRC, f"{DRIVE_SAVE_DIR}/image_baseline_results.json")
    print(f"✅ Results saved → {DRIVE_SAVE_DIR}/image_baseline_results.json")


# %% --- CELL 8 (optional): Download model trực tiếp về máy ---
from google.colab import files
if os.path.exists(MODEL_SRC):
    files.download(MODEL_SRC)
