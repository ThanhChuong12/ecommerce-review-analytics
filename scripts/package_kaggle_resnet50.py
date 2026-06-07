"""
package_kaggle_resnet50.py
--------------------------
Creates a ZIP archive of project files required to run ResNet50
defect-detection GPU fine-tuning on Kaggle.

Usage:
    python scripts/package_kaggle_resnet50.py

Output:
    resnet50_kaggle_train.zip   (in project root)

ZIP internal layout:
    resnet50_kaggle_train/
        scripts/
        ai_engine/
        data/image_dataset/defect/
        data/image_dataset/no-defect/
        docs/
        reports/
        requirements.txt  (if present)
        ai_engine/__init__.py etc.
"""

import os
import sys
import zipfile
from pathlib import Path
import time

# ── Project root is one level above this script ────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ZIP_NAME = "resnet50_kaggle_train.zip"
ZIP_ROOT = "resnet50_kaggle_train"          # folder name inside the ZIP
OUTPUT_ZIP = ROOT / ZIP_NAME

# ── Files/dirs that must be present ─────────────────────────────────────────
REQUIRED_FILES = [
    "scripts/train_defect_model.py",
    "scripts/tune_threshold.py",
    "scripts/evaluate_models.py",
    "scripts/prepare_dataset.py",
    "ai_engine/image_processing/defect_detection.py",
    "ai_engine/image_processing/augmentation/transforms.py",
    "ai_engine/models/resnet50_defect.pth",
    "docs/resnet50_training_flow.md",
    "reports/resnet50_training_report.md",
]
REQUIRED_DIRS = [
    "data/image_dataset/defect",
    "data/image_dataset/no-defect",
]

# ── Optional files (included if they exist) ──────────────────────────────────
OPTIONAL_FILES = [
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "ai_engine/__init__.py",
    "ai_engine/image_processing/__init__.py",
    "ai_engine/image_processing/augmentation/__init__.py",
    "scripts/kaggle_gpu_train_setup.py",
    "scripts/package_kaggle_resnet50.py",
    "docs/kaggle_gpu_training_instructions.md",
    "reports/resnet50_gpu_training_report_template.md",
]

# ── Patterns to exclude ───────────────────────────────────────────────────────
EXCLUDE_DIRS  = {"__pycache__", ".git", ".venv", "venv", "env",
                 ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache", "node_modules"}
EXCLUDE_EXTS  = {".pyc", ".pyo", ".ipynb", ".log", ".tmp"}
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    if path.name in EXCLUDE_NAMES:
        return True
    if path.suffix.lower() in EXCLUDE_EXTS:
        return True
    return False


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir()
               if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


def verify_requirements() -> bool:
    ok = True
    print("\n" + "=" * 64)
    print("  Step 1 — Verifying required files")
    print("=" * 64)

    for rel in REQUIRED_FILES:
        p = ROOT / rel
        status = "[OK]     " if p.exists() else "[MISSING]"
        print(f"  {status}  {rel}")
        if not p.exists():
            ok = False

    print()
    for rel in REQUIRED_DIRS:
        p = ROOT / rel
        n = count_images(p)
        if p.exists():
            print(f"  [OK]       {rel}/  ({n} images)")
        else:
            print(f"  [MISSING]  {rel}/")
            ok = False

    print()
    print("  Optional files:")
    for rel in OPTIONAL_FILES:
        p = ROOT / rel
        status = "[OK]  " if p.exists() else "[SKIP]"
        print(f"  {status}  {rel}")

    return ok


def add_file(zf: zipfile.ZipFile, src: Path, arcname: str) -> None:
    if _should_exclude(src):
        return
    zf.write(src, arcname)


def add_dir(zf: zipfile.ZipFile, src_dir: Path, arc_prefix: str) -> int:
    count = 0
    for item in sorted(src_dir.rglob("*")):
        if _should_exclude(item):
            continue
        if item.is_file():
            rel = item.relative_to(ROOT)
            arc = f"{ZIP_ROOT}/{rel.as_posix()}"
            zf.write(item, arc)
            count += 1
    return count


def build_zip() -> None:
    print("\n" + "=" * 64)
    print("  Step 2 — Building ZIP")
    print("=" * 64)
    print(f"  Output: {OUTPUT_ZIP}\n")

    total_files = 0
    t0 = time.time()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:

        # ── Required single files ──────────────────────────────────────────
        for rel in REQUIRED_FILES:
            p = ROOT / rel
            if p.exists():
                arc = f"{ZIP_ROOT}/{rel}"
                zf.write(p, arc)
                total_files += 1
                print(f"  + {arc}")

        # ── Required dirs (image_dataset) ──────────────────────────────────
        for rel in REQUIRED_DIRS:
            p = ROOT / rel
            if p.exists():
                n = add_dir(zf, p, ZIP_ROOT)
                total_files += n
                print(f"  + {ZIP_ROOT}/{rel}/  ({n} files)")

        # ── Optional files ──────────────────────────────────────────────────
        for rel in OPTIONAL_FILES:
            p = ROOT / rel
            if p.exists():
                arc = f"{ZIP_ROOT}/{rel}"
                if arc not in [zi.filename for zi in zf.infolist()]:
                    zf.write(p, arc)
                    total_files += 1
                    print(f"  + {arc}  (optional)")

    elapsed = time.time() - t0
    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)

    print()
    print("=" * 64)
    print("  Step 3 — Summary")
    print("=" * 64)
    print(f"  ZIP path    : {OUTPUT_ZIP}")
    print(f"  ZIP size    : {size_mb:.1f} MB")
    print(f"  Total files : {total_files}")
    print(f"  Time        : {elapsed:.1f}s")

    # ── Dataset counts ──────────────────────────────────────────────────────
    n_defect   = count_images(ROOT / "data/image_dataset/defect")
    n_no_defect = count_images(ROOT / "data/image_dataset/no-defect")
    ratio = n_no_defect / max(n_defect, 1)
    print()
    print("  Dataset image counts:")
    print(f"    defect   : {n_defect}")
    print(f"    no-defect: {n_no_defect}")
    print(f"    ratio    : {ratio:.2f}:1  (no-defect:defect)")
    print()
    print("  Upload this ZIP to Kaggle:")
    print("    Notebook -> Add Input -> Upload Dataset -> select ZIP")
    print()
    print("  Then run inside Kaggle:")
    print("    python scripts/kaggle_gpu_train_setup.py")
    print()


def verify_zip() -> None:
    print("=" * 64)
    print("  Step 4 — Verifying ZIP contents")
    print("=" * 64)
    checks = {
        f"{ZIP_ROOT}/scripts/": False,
        f"{ZIP_ROOT}/ai_engine/": False,
        f"{ZIP_ROOT}/data/image_dataset/defect/": False,
        f"{ZIP_ROOT}/data/image_dataset/no-defect/": False,
        f"{ZIP_ROOT}/ai_engine/models/resnet50_defect.pth": False,
    }
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zf:
        names = zf.namelist()
        for key in list(checks.keys()):
            checks[key] = any(n.startswith(key) or n == key for n in names)

    for key, found in checks.items():
        status = "[OK]     " if found else "[MISSING]"
        print(f"  {status}  {key}")

    print()
    print("  Packaging complete." if all(checks.values())
          else "  WARNING: Some expected entries not found in ZIP.")


def main() -> None:
    print()
    print("=" * 64)
    print("  ResNet50 Kaggle Packaging — package_kaggle_resnet50.py")
    print("=" * 64)

    ok = verify_requirements()
    if not ok:
        print("\n[ERROR] Required files are missing. Fix them before packaging.")
        sys.exit(1)

    build_zip()
    verify_zip()


if __name__ == "__main__":
    main()
