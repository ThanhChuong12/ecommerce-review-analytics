r"""
train_mobilenet_kaggle.py
--------------------------
Kaggle Notebook script — MobileNetV3 Defect Detection (4-class)

SETUP TRƯỚC KHI CHẠY:
  Bước 1 — Tạo 2 Kaggle Datasets:
    [A] image-dataset-split : upload thư mục data\image_dataset_split\ chứa train/, val/, test/
    [B] mobilenetv3-model2-code-offline : upload file mobilenetv3_model2_code_offline.zip

  Bước 2 — Tạo Kaggle Notebook:
    - New Notebook → Script
    - Add Input → [A] labeled-images
    - Add Input → [B] mobilenetv3-model2-code-offline
    - Settings → Accelerator: GPU T4
    - Settings → Internet: OFF
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


# ── STEP 2: Extract offline code package ───────────────────────────────────
print("\n[Step 2] Locating and extracting offline code package...")

import zipfile

CODE_ZIP = None
# Tìm bất kỳ file zip nào trong /kaggle/input chứa file train_image_baseline.py hoặc có tên liên quan đến code/model2/mobilenet
for candidate in Path("/kaggle/input").rglob("*.zip"):
    if "code" in candidate.name.lower() or "mobilenet" in candidate.name.lower():
        CODE_ZIP = candidate
        break

if CODE_ZIP is None:
    # Fallback: lấy file zip đầu tiên tìm thấy
    for candidate in Path("/kaggle/input").rglob("*.zip"):
        CODE_ZIP = candidate
        break

if CODE_ZIP is None:
    raise FileNotFoundError(f"{ERR}Không tìm thấy file code offline dạng zip trong /kaggle/input! Đảm bảo đã add input dataset code offline.")

WORK_DIR = Path("/kaggle/working/project")
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir(parents=True, exist_ok=True)

print(f"{OK}Found offline code package: {CODE_ZIP}")
print(f"Extracting to {WORK_DIR}...")
with zipfile.ZipFile(CODE_ZIP, "r") as zip_ref:
    zip_ref.extractall(WORK_DIR)

os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))
print(f"{OK}Working dir: {WORK_DIR}")


# ── STEP 3: Verification ───────────────────────────────────────────────────
print("\n[Step 3] Verifying extracted codebase...")
if (WORK_DIR / "ai_engine/models/image_baseline.py").exists():
    print(f"{OK}Verified image_baseline.py exists.")
else:
    print(f"{WARN}image_baseline.py not found in extracted workspace!")


# ── STEP 4: Cài thêm dependencies ─────────────────────────────────────────
print("\n[Step 4] Installing dependencies...")
os.system("pip install -q albumentations opencv-python-headless scikit-learn Pillow")
print(f"{OK}Dependencies ready.")


# ── STEP 5: Kiểm tra data ──────────────────────────────────────────────────
print("\n[Step 5] Locating labeled images split...")

# Tìm thư mục chứa train/val/test
TRAIN_DIR = None
VAL_DIR = None
TEST_DIR = None
CLASSES = {"defect", "no-defect"}

for candidate in Path("/kaggle/input").rglob("defect"):
    if candidate.is_dir() and candidate.parent.name == "train":
        parent = candidate.parent.parent
        subdirs = {d.name for d in parent.iterdir() if d.is_dir()}
        if {"train", "val", "test"}.issubset(subdirs):
            if CLASSES.issubset({d.name for d in (parent / "train").iterdir() if d.is_dir()}):
                TRAIN_DIR = str(parent / "train")
                VAL_DIR = str(parent / "val")
                TEST_DIR = str(parent / "test")
                break

if TRAIN_DIR is None or VAL_DIR is None:
    raise FileNotFoundError(
        f"{ERR}Không tìm thấy cấu trúc split train/val/test với các nhãn defect/no-defect trong /kaggle/input!"
    )

print(f"{OK}Data splits found:")
print(f"   Train: {TRAIN_DIR}")
print(f"   Val:   {VAL_DIR}")
print(f"   Test:  {TEST_DIR}")

for split_name, split_path in [("Train", TRAIN_DIR), ("Val", VAL_DIR), ("Test", TEST_DIR)]:
    print(f"\n   --- {split_name} split counts ---")
    total = 0
    for cls in sorted(CLASSES):
        n = len(list(Path(split_path, cls).glob("*")))
        total += n
        print(f"      {cls:15s}: {n:,} ảnh")
    print(f"      {'TOTAL':15s}: {total:,} ảnh")


# ── STEP 6: Train ──────────────────────────────────────────────────────────
print("\n[Step 6] Starting training...\n" + "=" * 65)

cmd = [
    "python", "scripts/train_image_baseline.py",
    "--backbone",   "mobilenet_v3",
    "--data-dir",   TRAIN_DIR,
    "--val-dir",    VAL_DIR,
    "--epochs",     "15",
    "--lr",         "1e-3",
    "--batch-size", "32",
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
    shutil.copy(MODEL_SRC, OUTPUT_DIR / "mobilenet_v3_model2_defect.pt")
    print(f"{OK}Model saved → /kaggle/working/outputs/mobilenet_v3_model2_defect.pt")
else:
    print(f"{ERR}Model file không tìm thấy!")

if RESULTS_SRC.exists():
    shutil.copy(RESULTS_SRC, OUTPUT_DIR / "mobilenet_v3_model2_results.json")
    print(f"{OK}Results saved → /kaggle/working/outputs/mobilenet_v3_model2_results.json")

# Copy them artifacts voi ten Model 2 rieng biet
RESULTS_SRC_DIR = WORK_DIR / "ai_engine/models/results"
artifact_mapping = {
    "mobilenet_v3_learning_curves.png": "mobilenet_v3_model2_learning_curves.png",
    "mobilenet_v3_confusion_matrix.png": "mobilenet_v3_model2_confusion_matrix.png",
    "mobilenet_v3_training_history.json": "mobilenet_v3_model2_training_history.json",
}
for src_name, dest_name in artifact_mapping.items():
    src = RESULTS_SRC_DIR / src_name
    if src.exists():
        shutil.copy(src, OUTPUT_DIR / dest_name)
        print(f"{OK}Saved: {dest_name}")

print("\n" + "=" * 65)
print("  HOÀN TẤT — Download outputs từ tab Output của Notebook")
print("=" * 65)