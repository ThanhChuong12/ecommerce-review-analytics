r"""
train_mobilenet_kaggle.py
--------------------------
Kaggle Notebook script — MobileNetV3 Improved Defect Detection (Model 2)

SETUP BEFORE RUNNING:
  Step 1 — Create 2 Kaggle Datasets:
    [A] image-dataset-split: upload the data\image_dataset_split\ directory containing train/, val/, test/
    [B] mobilenetv3-model2-code-offline: upload the mobilenetv3_model2_code_offline.zip file

  Step 2 — Create Kaggle Notebook:
    - New Notebook → Script
    - Add Input → [A] labeled-images
    - Add Input → [B] mobilenetv3-model2-code-offline
    - Settings → Accelerator: GPU T4
    - Settings → Internet: OFF
    - Paste the entire content of this file into the script

  Step 3 — Run the notebook
"""

import os
import sys
import shutil
import json
import subprocess
from pathlib import Path

# ── ANSI colors ────────────────────────────────────────────────────────────
OK   = "\033[92m✅ \033[0m"
WARN = "\033[93m⚠️  \033[0m"
ERR  = "\033[91m❌ \033[0m"

print("=" * 75)
print("  MobileNetV3 Improved Defect Detection (Model 2) — Kaggle GPU Training")
print("=" * 75)


# ── STEP 1: Check GPU ──────────────────────────────────────────────────────
print("\n[Step 1] Checking GPU...")
import torch

if not torch.cuda.is_available():
    raise RuntimeError(f"{ERR}GPU không khả dụng! Vào Settings → Accelerator → GPU T4")

print(f"{OK}GPU: {torch.cuda.get_device_name(0)}")
print(f"{OK}VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── STEP 2: Locate offline code package or extracted code ──────────────────
print("\n[Step 2] Locating offline code package or extracted code...")

import zipfile

WORK_DIR = Path("/kaggle/working/project")
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir(parents=True, exist_ok=True)

required_rel_files = [
    Path("ai_engine/models/image_baseline.py"),
    Path("scripts/train_image_baseline.py"),
    Path("notebooks/train_mobilenet_kaggle.py"),
]

def is_project_root(path: Path) -> bool:
    return all((path / rel).exists() for rel in required_rel_files)

# Case A: Kaggle kept the uploaded code zip
CODE_ZIP = None
for candidate in Path("/kaggle/input").rglob("*.zip"):
    name = candidate.name.lower()
    if "code" in name or "mobilenet" in name or "model2" in name:
        CODE_ZIP = candidate
        break

if CODE_ZIP is not None:
    print(f"{OK}Found offline code ZIP: {CODE_ZIP}")
    print(f"Extracting to {WORK_DIR}...")
    with zipfile.ZipFile(CODE_ZIP, "r") as zip_ref:
        zip_ref.extractall(WORK_DIR)

# Case B: Kaggle already extracted the uploaded code dataset
else:
    print(f"{WARN}No code ZIP found. Searching for already-extracted code dataset...")

    PROJECT_ROOT = None
    for candidate in Path("/kaggle/input").rglob("ai_engine/models/image_baseline.py"):
        possible_root = candidate.parents[2]  # root/ai_engine/models/image_baseline.py
        if is_project_root(possible_root):
            PROJECT_ROOT = possible_root
            break

    if PROJECT_ROOT is None:
        print("\nAvailable /kaggle/input structure:")
        for p in Path("/kaggle/input").glob("*"):
            print(" -", p)
            for sub in list(p.glob("*"))[:20]:
                print("   -", sub)

        raise FileNotFoundError(
            f"{ERR}Không tìm thấy code offline trong /kaggle/input. "
            "Hãy kiểm tra dataset code đã được Add Input đúng chưa."
        )

    print(f"{OK}Found extracted code root: {PROJECT_ROOT}")
    print(f"Copying code to writable work dir: {WORK_DIR}")

    for item in PROJECT_ROOT.iterdir():
        dest = WORK_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, dest, ignore=shutil.ignore_patterns(
                "__pycache__", ".ipynb_checkpoints", ".git", "*.pt", "*.pth", "*.ckpt", "*.zip"
            ))
        else:
            shutil.copy2(item, dest)

os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))
print(f"{OK}Working dir: {WORK_DIR}")


# ── STEP 3: Verification ───────────────────────────────────────────────────
print("\n[Step 3] Verifying extracted codebase...")
if (WORK_DIR / "ai_engine/models/image_baseline.py").exists():
    print(f"{OK}Verified image_baseline.py exists.")
else:
    print(f"{WARN}image_baseline.py not found in extracted workspace!")


# ── STEP 4: Install dependencies ─────────────────────────────────────────
print("\n[Step 4] Installing dependencies...")
os.system("pip install -q albumentations opencv-python-headless scikit-learn Pillow")
print(f"{OK}Dependencies ready.")


# ── STEP 5: Check data ─────────────────────────────────────────────────────
print("\n[Step 5] Locating labeled images split...")

# Find directory containing train/val/test
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
print("\n[Step 6] Starting training (Validation threshold selection mode)...\n" + "=" * 75)

cmd_train = [
    "python", "scripts/train_image_baseline.py",
    "--backbone", "mobilenet_v3",
    "--data-dir", TRAIN_DIR,
    "--val-dir", VAL_DIR,
    "--epochs", "15",
    "--lr", "1e-3",
    "--batch-size", "32",
    "--patience", "5",
    "--subset-ratio", "1.0",
    "--class-weight-mode", "sqrt",
    "--threshold-mode", "maximize_macro_f1_subject_to_recall",
    "--weights-dir", str(WORK_DIR / "ai_engine/models/weights"),
    "--results-dir", str(WORK_DIR / "ai_engine/models/results"),
    "--weights-name", "mobilenet_v3_model2_improved_defect.pt",
    "--results-name", "mobilenet_v3_model2_improved_results.json",
    "--learning-curves-name", "mobilenet_v3_model2_improved_learning_curves.png",
    "--confusion-matrix-name", "mobilenet_v3_model2_improved_confusion_matrix.png",
    "--training-history-name", "mobilenet_v3_model2_improved_training_history.json",
    "--threshold-tuning-name", "mobilenet_v3_model2_improved_threshold_tuning.json",
]

result_train = subprocess.run(cmd_train, stdout=sys.stdout, stderr=sys.stdout, text=True)
print("=" * 75)
print(f"Training exit code: {result_train.returncode}")

if result_train.returncode != 0:
    raise RuntimeError(f"{ERR}Training thất bại! Xem log bên trên.")


# ── STEP 7: Official Test Set Evaluation using Val Threshold ───────────────
print("\n[Step 7] Evaluating on Test set using the validation-selected threshold...")
cmd_eval = [
    "python", "scripts/train_image_baseline.py",
    "--backbone", "mobilenet_v3",
    "--eval-test",
    "--test-dir", TEST_DIR,
    "--weights-dir", str(WORK_DIR / "ai_engine/models/weights"),
    "--weights-name", "mobilenet_v3_model2_improved_defect.pt",
    "--results-dir", str(WORK_DIR / "ai_engine/models/results"),
    "--results-name", "mobilenet_v3_model2_improved_test_results.json",
    "--threshold-file", str(WORK_DIR / "ai_engine/models/results/mobilenet_v3_model2_improved_threshold_tuning.json"),
]

result_eval = subprocess.run(cmd_eval, stdout=sys.stdout, stderr=sys.stdout, text=True)
print("=" * 75)
print(f"Evaluation exit code: {result_eval.returncode}")

if result_eval.returncode != 0:
    logger.warning(f"{WARN}Evaluation test set failed, checking fallback...")


# ── STEP 8: Save Output Files to /kaggle/working/outputs ───────────────────
print("\n[Step 8] Saving outputs...")

OUTPUT_DIR = Path("/kaggle/working/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Map result files from Project dir to outputs dir
artifact_mapping = {
    "ai_engine/models/weights/mobilenet_v3_model2_improved_defect.pt": "mobilenet_v3_model2_improved_defect.pt",
    "ai_engine/models/results/mobilenet_v3_model2_improved_results.json": "mobilenet_v3_model2_improved_results.json",
    "ai_engine/models/results/mobilenet_v3_model2_improved_learning_curves.png": "mobilenet_v3_model2_improved_learning_curves.png",
    "ai_engine/models/results/mobilenet_v3_model2_improved_confusion_matrix.png": "mobilenet_v3_model2_improved_confusion_matrix.png",
    "ai_engine/models/results/mobilenet_v3_model2_improved_training_history.json": "mobilenet_v3_model2_improved_training_history.json",
    "ai_engine/models/results/mobilenet_v3_model2_improved_threshold_tuning.json": "mobilenet_v3_model2_improved_threshold_tuning.json",
    "ai_engine/models/results/mobilenet_v3_model2_improved_test_results.json": "mobilenet_v3_model2_improved_test_results.json",
}

for src_rel, dest_name in artifact_mapping.items():
    src = WORK_DIR / src_rel
    if src.exists():
        shutil.copy(src, OUTPUT_DIR / dest_name)
        print(f"{OK}Saved: {dest_name}")
    else:
        print(f"{WARN}Artifact not found: {src_rel}")

print("\n" + "=" * 75)
print("  HOÀN TẤT — Download outputs từ tab Output của Notebook")
print("=" * 75)