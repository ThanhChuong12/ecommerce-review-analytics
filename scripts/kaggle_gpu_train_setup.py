"""
kaggle_gpu_train_setup.py
--------------------------
Kaggle environment setup script for ResNet50 GPU fine-tuning.

Run this as the FIRST cell/script inside a Kaggle Notebook after uploading
resnet50_kaggle_train.zip as a dataset.

Usage (inside Kaggle Notebook):
    python scripts/kaggle_gpu_train_setup.py
    # or equivalently:
    !python scripts/kaggle_gpu_train_setup.py
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path


# ── ANSI colours (Kaggle notebooks support them) ───────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

ok   = lambda s: f"{GREEN}✅  {s}{RESET}"
warn = lambda s: f"{YELLOW}⚠️   {s}{RESET}"
err  = lambda s: f"{RED}❌  {s}{RESET}"


def sep(title: str = "") -> None:
    line = "=" * 68
    if title:
        print(f"\n{BOLD}{line}{RESET}")
        print(f"{BOLD}  {title}{RESET}")
        print(f"{BOLD}{line}{RESET}")
    else:
        print(line)


# ── 1. Detect Kaggle input paths ───────────────────────────────────────────
sep("Step 1 — Detecting Kaggle environment")

KAGGLE_INPUT  = Path("/kaggle/input")
KAGGLE_WORK   = Path("/kaggle/working")
PROJECT_DIR   = KAGGLE_WORK / "resnet50_project"

IS_KAGGLE = KAGGLE_INPUT.exists() and KAGGLE_WORK.exists()
print(f"  Running inside Kaggle: {IS_KAGGLE}")
print(f"  /kaggle/input exists : {KAGGLE_INPUT.exists()}")
print(f"  /kaggle/working      : {KAGGLE_WORK}")

if not IS_KAGGLE:
    print(warn("Not running inside Kaggle. Paths will be simulated for testing."))
    KAGGLE_INPUT = Path(".")
    KAGGLE_WORK  = Path(".")
    PROJECT_DIR  = Path(".")


# ── 2. Find the uploaded dataset / ZIP ────────────────────────────────────
sep("Step 2 — Locating uploaded project data")

CANDIDATE_NAMES = [
    "resnet50-kaggle-train",
    "resnet50_kaggle_train",
    "resnet50kagglertrain",
]
INNER_FOLDER = "resnet50_kaggle_train"

found_source: Path | None = None
found_via_zip = False

# Walk top-level Kaggle dataset directories
for ds_dir in sorted(KAGGLE_INPUT.iterdir()):
    if not ds_dir.is_dir():
        continue
    print(f"  Scanning: {ds_dir}")

    # Case A: extracted folder exists directly
    candidate = ds_dir / INNER_FOLDER
    if candidate.is_dir():
        found_source = candidate
        print(ok(f"Found extracted project folder: {candidate}"))
        break

    # Case B: ZIP inside dataset directory
    zips = list(ds_dir.glob("*.zip"))
    if zips:
        zip_path = zips[0]
        print(f"  Found ZIP: {zip_path}")
        print(f"  Extracting to {KAGGLE_WORK} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(KAGGLE_WORK)
        candidate = KAGGLE_WORK / INNER_FOLDER
        if candidate.is_dir():
            found_source = candidate
            found_via_zip = True
            print(ok(f"Extracted project to: {candidate}"))
            break

if found_source is None:
    # Case C: fallback — look anywhere under /kaggle/input
    for p in KAGGLE_INPUT.rglob("scripts/train_defect_model.py"):
        found_source = p.parent.parent
        print(warn(f"Fallback: found project at {found_source}"))
        break

if found_source is None:
    print(err("Could not locate the project folder under /kaggle/input."))
    print("  Please verify the dataset was uploaded correctly.")
    print("  Expected structure: /kaggle/input/<dataset-slug>/resnet50_kaggle_train/")
    sys.exit(1)


# ── 3. Copy project to /kaggle/working (read-write) ───────────────────────
sep("Step 3 — Copying project to working directory")

if found_source != PROJECT_DIR:
    if PROJECT_DIR.exists():
        shutil.rmtree(PROJECT_DIR)
    shutil.copytree(found_source, PROJECT_DIR)
    print(ok(f"Project copied to: {PROJECT_DIR}"))
else:
    print(ok(f"Project already at working dir: {PROJECT_DIR}"))

# Set working directory
os.chdir(PROJECT_DIR)
print(f"  Working directory: {os.getcwd()}")


# ── 4. Add project to sys.path ─────────────────────────────────────────────
sep("Step 4 — Configuring sys.path")

proj_str = str(PROJECT_DIR)
if proj_str not in sys.path:
    sys.path.insert(0, proj_str)
    print(ok(f"Added to sys.path: {proj_str}"))
else:
    print(ok("Project already in sys.path"))


# ── 5. GPU check ───────────────────────────────────────────────────────────
sep("Step 5 — GPU availability")

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
    gpu_count = torch.cuda.device_count()
    print(f"  CUDA available : {cuda_ok}")
    print(f"  GPU name       : {gpu_name}")
    print(f"  GPU count      : {gpu_count}")
    if cuda_ok:
        print(ok("GPU ready for training"))
    else:
        print(err("No GPU detected! Enable GPU in Notebook settings: Notebook → Settings → Accelerator → GPU P100"))
except ImportError:
    print(warn("PyTorch not installed. Run: pip install torch torchvision"))


# ── 6. Verify required files ───────────────────────────────────────────────
sep("Step 6 — Verifying required files")

REQUIRED = [
    "scripts/train_defect_model.py",
    "scripts/tune_threshold.py",
    "scripts/evaluate_models.py",
    "scripts/prepare_dataset.py",
    "ai_engine/image_processing/defect_detection.py",
    "ai_engine/image_processing/augmentation/transforms.py",
    "ai_engine/models/resnet50_defect.pth",
    "data/image_dataset/defect",
    "data/image_dataset/no-defect",
]

all_ok = True
for rel in REQUIRED:
    p = PROJECT_DIR / rel
    if p.exists():
        print(ok(rel))
    else:
        print(err(f"MISSING: {rel}"))
        all_ok = False

if not all_ok:
    print(err("Some required files are missing. Re-upload the ZIP."))
    sys.exit(1)


# ── 7. Dataset image counts ────────────────────────────────────────────────
sep("Step 7 — Dataset image counts")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir()
               if f.is_file() and f.suffix.lower() in IMG_EXTS)

n_defect   = count_images(PROJECT_DIR / "data/image_dataset/defect")
n_no_defect = count_images(PROJECT_DIR / "data/image_dataset/no-defect")
ratio = n_no_defect / max(n_defect, 1)

print(f"  defect   : {n_defect} images")
print(f"  no-defect: {n_no_defect} images")
print(f"  ratio    : {ratio:.2f}:1  (imbalance)")

if n_defect < 100:
    print(err(f"defect class has only {n_defect} images — too few to train reliably"))
elif n_defect < 1000:
    print(warn(f"defect class has {n_defect} images — training may still be imbalanced"))
else:
    print(ok(f"defect class count ({n_defect}) acceptable for training"))


# ── 8. Backup CPU checkpoint ──────────────────────────────────────────────
sep("Step 8 — Backing up CPU checkpoint")

cpu_ckpt = PROJECT_DIR / "ai_engine/models/resnet50_defect.pth"
cpu_backup = PROJECT_DIR / "ai_engine/models/resnet50_defect_cpu_backup.pth"

if cpu_ckpt.exists():
    if not cpu_backup.exists():
        shutil.copy2(cpu_ckpt, cpu_backup)
        print(ok(f"CPU checkpoint backed up to: {cpu_backup.name}"))
    else:
        print(ok(f"CPU backup already exists: {cpu_backup.name}"))

    # Print CPU checkpoint metrics
    try:
        import torch
        ckpt = torch.load(cpu_ckpt, map_location="cpu", weights_only=False)
        print(f"\n  CPU checkpoint metrics:")
        print(f"    epoch        : {ckpt.get('epoch')}")
        print(f"    defect_f1    : {ckpt.get('defect_f1', 0):.4f}")
        print(f"    defect_recall: {ckpt.get('defect_recall', 0):.4f}")
        print(f"    macro_f1     : {ckpt.get('macro_f1', 0):.4f}")
        print(f"    val_acc      : {ckpt.get('val_acc', 0):.4f}")
    except Exception as e:
        print(warn(f"Could not read checkpoint metadata: {e}"))
else:
    print(warn("CPU checkpoint not found — skipping backup"))


# ── 9. Print recommended training commands ────────────────────────────────
sep("Step 9 — Recommended GPU training commands")

PROJ = str(PROJECT_DIR)

print(f"""
{'─'*68}
  Ensure you are in the project directory first:
    cd {PROJ}

  ── MAIN GPU TRAINING COMMAND ──────────────────────────────────
  python scripts/train_defect_model.py \\
      --data-dir data/image_dataset \\
      --epochs 25 \\
      --batch-size 64 \\
      --lr 5e-4 \\
      --backbone-lr 1e-5 \\
      --oversample 10 \\
      --patience 8 \\
      --unfreeze-layer4 \\
      --save-path ai_engine/models/resnet50_defect_gpu_layer4.pth

  If CUDA out-of-memory (OOM), retry with:
      --batch-size 32

  ── THRESHOLD TUNING ────────────────────────────────────────────
  python scripts/tune_threshold.py \\
      --model-path ai_engine/models/resnet50_defect_gpu_layer4.pth \\
      --data-dir data/image_dataset \\
      --val-split 0.2 --seed 42 --batch-size 64

  ── FINAL EVALUATION ────────────────────────────────────────────
  python scripts/evaluate_models.py image \\
      --model-path ai_engine/models/resnet50_defect_gpu_layer4.pth \\
      --data-path data/image_dataset \\
      --batch-size 64

  ── COPY OUTPUTS ────────────────────────────────────────────────
  mkdir -p outputs
  cp ai_engine/models/resnet50_defect_gpu_layer4.pth outputs/resnet50_defect_gpu_best.pth
  cp reports/figures/confusion_matrix_resnet50_*.png outputs/ 2>/dev/null || true

  Download /kaggle/working/resnet50_project/outputs/ after training.
{'─'*68}
""")

print(ok("Setup complete. GPU training environment is ready."))
