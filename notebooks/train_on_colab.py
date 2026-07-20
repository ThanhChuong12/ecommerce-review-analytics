# ============================================================
# COLAB TRAINING — MobileNetV3 Defect Detection (4-class)
# Ecommerce Review Analytics
# ============================================================
# BEFORE RUNNING:
#   1. Runtime → Change runtime type → T4 GPU
#   2. Upload to Google Drive (root directory My Drive):
#        - labeled_data.zip       (labeled images)
#        - image_baseline.py      (bug-fixed version — retrieved from ai_engine/models/)
# ============================================================


# %% [markdown]
# # MobileNetV3 Defect Detection — Colab Training
# **Classes:** intact | damaged | wrong_item | irrelevant
# **Dataset:** ~27,743 images (ImageFolder format)
# **Estimated time:** ~20-40 minutes (T4 GPU)


# %% --- CELL 1: Check GPU ---
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


# %% --- CELL 3: Install dependencies ---
os.system("pip install -q albumentations opencv-python-headless scikit-learn Pillow")
print("Dependencies installed.")


# %% --- CELL 4: Apply bug fixes for image_baseline.py ---
# FIX: Code on main does not have the 4 bug fixes for MobileNetV3.
# Need to upload the image_baseline.py file (fixed) to Google Drive first.
#
# How to get the fixed file on local machine:
#   ai_engine\models\image_baseline.py  →  upload to Google Drive
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


# %% --- CELL 5: Extract data ---
# Upload labeled_data.zip to Google Drive first (root directory My Drive)
# How to create zip with correct structure (run on local machine):
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

# Check number of images in each class
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
# FIX: Use subprocess instead of os.system() to display real-time logs
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


# %% --- CELL 7: Save model to Google Drive ---
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


# %% --- CELL 8 (optional): Download model directly to local machine ---
from google.colab import files
if os.path.exists(MODEL_SRC):
    files.download(MODEL_SRC)
