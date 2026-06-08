"""
train_mobilenet_kaggle.py
--------------------------
Kaggle Notebook script — MobileNetV3 Defect Detection (4-class)

SETUP TRƯỚC KHI CHẠY:
  Bước 1 — Tạo 2 Kaggle Datasets:
    [A] labeled-images   : upload thư mục labeled\labeled\ (4 folder: intact/ damaged/ wrong_item/ irrelevant/)
    [B] mobilenet-fixed  : upload file ai_engine\models\image_baseline.py (đã fix bugs)

  Bước 2 — Tạo Kaggle Notebook:
    - New Notebook → Script
    - Add Input → [A] labeled-images
    - Add Input → [B] mobilenet-fixed
    - Settings → Accelerator: GPU P100 (hoặc T4)
    - Paste toàn bộ file này vào script

  Bước 3 — Chạy notebook
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ── ANSI colors ────────────────────────────────────────────────────────────
OK   = "\033[92m✅ \033[0m"
WARN = "\033[93m⚠️  \033[0m"
ERR  = "\033[91m❌ \033[0m"

print("=" * 65)
print("  MobileNetV3 Defect Detection — Kaggle GPU Training")
print("=" * 65)


# ── STEP 1: Kiểm tra GPU ───────────────────────────────────────────────────
print("\n[Step 1] Checking GPU...")
import torch

if not torch.cuda.is_available():
    raise RuntimeError(f"{ERR}GPU không khả dụng! Vào Settings → Accelerator → GPU P100")

print(f"{OK}GPU: {torch.cuda.get_device_name(0)}")
print(f"{OK}VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── STEP 2: Clone repo ─────────────────────────────────────────────────────
print("\n[Step 2] Cloning repo...")

REPO_URL  = "https://github.com/ThanhChuong12/ecommerce-review-analytics.git"
WORK_DIR  = Path("/kaggle/working/project")

if not WORK_DIR.exists():
    os.system(f"git clone {REPO_URL} {WORK_DIR}")
else:
    print(f"{OK}Repo đã tồn tại, skip clone.")

os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))
print(f"{OK}Working dir: {WORK_DIR}")


# ── STEP 3: Apply bug fixes cho image_baseline.py ─────────────────────────
# Dataset [B] mobilenet-fixed phải chứa file image_baseline.py đã fix
print("\n[Step 3] Applying image_baseline.py bug fixes...")

# Tìm file đã fix trong /kaggle/input
FIXED_FILE = None
for p in Path("/kaggle/input").rglob("image_baseline.py"):
    FIXED_FILE = p
    break

TARGET = WORK_DIR / "ai_engine/models/image_baseline.py"

if FIXED_FILE:
    shutil.copy(FIXED_FILE, TARGET)
    print(f"{OK}Applied fix: {FIXED_FILE} → {TARGET}")
else:
    print(f"{WARN}Không tìm thấy image_baseline.py đã fix trong /kaggle/input!")
    print(f"   Tạo Kaggle Dataset chứa file image_baseline.py rồi Add Input vào notebook.")
    print(f"   Tiếp tục với code repo gốc (có thể có 4 bugs).")


# ── STEP 4: Cài thêm dependencies ─────────────────────────────────────────
print("\n[Step 4] Installing dependencies...")
os.system("pip install -q albumentations opencv-python-headless scikit-learn Pillow")
print(f"{OK}Dependencies ready.")


# ── STEP 5: Kiểm tra data ──────────────────────────────────────────────────
print("\n[Step 5] Locating labeled images...")

# Tìm thư mục chứa 4 class folders
DATA_DIR = None
CLASSES  = {"intact", "damaged", "wrong_item", "irrelevant"}

for candidate in Path("/kaggle/input").rglob("intact"):
    if candidate.is_dir():
        parent = candidate.parent
        subdirs = {d.name for d in parent.iterdir() if d.is_dir()}
        if CLASSES.issubset(subdirs):
            DATA_DIR = str(parent)
            break

if DATA_DIR is None:
    raise FileNotFoundError(
        f"{ERR}Không tìm thấy thư mục chứa 4 class folders (intact/damaged/wrong_item/irrelevant) "
        f"trong /kaggle/input. Kiểm tra lại dataset upload."
    )

print(f"{OK}Data found: {DATA_DIR}")
total = 0
for cls in ["intact", "damaged", "wrong_item", "irrelevant"]:
    n = len(list(Path(DATA_DIR, cls).glob("*")))
    total += n
    print(f"   {cls:15s}: {n:,} ảnh")
print(f"   {'TOTAL':15s}: {total:,} ảnh")


# ── STEP 6: Train ──────────────────────────────────────────────────────────
print("\n[Step 6] Starting training...\n" + "=" * 65)

cmd = [
    "python", "scripts/train_image_baseline.py",
    "--backbone",   "mobilenet_v3",
    "--data-dir",   DATA_DIR,
    "--epochs",     "20",
    "--lr",         "1e-3",
    "--batch-size", "64",      # P100 16GB: có thể tăng lên 128 nếu muốn nhanh hơn
    "--val-split",  "0.2",
    "--patience",   "5",
    "--weights-dir", str(WORK_DIR / "ai_engine/models/weights"),
]

result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stdout, text=True)
print("=" * 65)
print(f"Training exit code: {result.returncode}")

if result.returncode != 0:
    raise RuntimeError(f"{ERR}Training thất bại! Xem log bên trên.")


# ── STEP 7: Copy output ra /kaggle/working để download ────────────────────
print("\n[Step 7] Saving outputs...")

OUTPUT_DIR  = Path("/kaggle/working/outputs")
MODEL_SRC   = WORK_DIR / "ai_engine/models/weights/mobilenet_v3_defect.pt"
RESULTS_SRC = WORK_DIR / "ai_engine/models/results/image_baseline_results.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if MODEL_SRC.exists():
    shutil.copy(MODEL_SRC, OUTPUT_DIR / "mobilenet_v3_defect.pt")
    print(f"{OK}Model saved → /kaggle/working/outputs/mobilenet_v3_defect.pt")
else:
    print(f"{ERR}Model file không tìm thấy!")

if RESULTS_SRC.exists():
    shutil.copy(RESULTS_SRC, OUTPUT_DIR / "image_baseline_results.json")
    print(f"{OK}Results saved → /kaggle/working/outputs/image_baseline_results.json")

print("\n" + "=" * 65)
print("  HOÀN TẤT — Download outputs từ tab Output của Notebook")
print("=" * 65)
