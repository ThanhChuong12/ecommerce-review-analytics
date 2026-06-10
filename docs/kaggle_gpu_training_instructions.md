# Kaggle GPU Training Instructions — ResNet50 Defect Detection

> Purpose: Fine-tune ResNet50 on Kaggle GPU to improve defect detection beyond the CPU baseline.  
> CPU baseline: defect_f1 = 0.4175, defect_recall = 0.488, training_mode = frozen  
> Target: defect_f1 ≥ 0.85, defect_recall ≥ 0.80, macro_f1 ≥ 0.85

---

## Overview

The CPU-trained ResNet50 model plateaued because only the FC head was trained (4.3% of parameters). On Kaggle GPU we will unfreeze `layer4` of ResNet50 alongside the FC head (~6.4M trainable parameters) using differential learning rates on a clean **70/15/15 stratified dataset split**. This is called **layer4 fine-tuning**.

The entire training, validation early stopping, threshold tuning, and independent test-set evaluation are fully integrated into `scripts/train_defect_model.py`.

---

## Step 1 — Generate the ZIP locally

Run this on your local machine from the project root:

```bash
python scripts/package_kaggle_resnet50.py
```

Expected output:
```
resnet50_kaggle_train_split.zip   (in project root)
```

The script will print the ZIP path, file size, top-level contents, and excluded items. Verify before uploading.

---

## Step 2 — Upload to Kaggle

1. Go to [kaggle.com](https://www.kaggle.com) → **Datasets** → **New Dataset**
2. Name: `resnet50-kaggle-train-split` (Kaggle will slugify this to `resnet50-kaggle-train-split`)
3. Upload: `resnet50_kaggle_train_split.zip`
4. Set visibility to **Private**
5. Click **Create**

---

## Step 3 — Create a Kaggle Notebook

1. Go to **Notebooks** → **New Notebook** → **Script** (not notebook)
2. Click **Add Input** → your uploaded dataset → `resnet50-kaggle-train-split`
3. Go to **Settings** (right panel):
   - **Accelerator**: GPU P100 *(preferred — 16 GB VRAM, stable)* or GPU T4 x2
   - **Persistence**: Files

---

## Step 4 — Kaggle first cell (verify environment)

Paste and run this at the start of your Kaggle session:

```python
import os, sys
from pathlib import Path

print("Input folders:")
import subprocess
result = subprocess.run(["find", "/kaggle/input", "-maxdepth", "3", "-type", "d"],
                        capture_output=True, text=True)
print(result.stdout[:3000])

print("\nGPU:")
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout[:1000])

import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("GPU count:", torch.cuda.device_count())
```

---

## Step 5 — Run the setup script

```bash
python scripts/kaggle_gpu_train_setup.py
```

This setup script:
- Finds the uploaded data under `/kaggle/input`
- Extracts the ZIP to `/kaggle/working/resnet50_project` (read-write)
- Verifies GPU availability
- Verifies all required files and split dataset structure
- Backs up the CPU checkpoint to `resnet50_defect_cpu_backup.pth`
- Prints the recommended training commands

---

## Step 6 — Main GPU training command

```bash
cd /kaggle/working/resnet50_project

python scripts/train_defect_model.py \
    --data-dir data/image_dataset_split \
    --epochs 25 \
    --batch-size 64 \
    --lr 5e-4 \
    --backbone-lr 1e-5 \
    --oversample 10 \
    --patience 8 \
    --unfreeze-layer4 \
    --save-path ai_engine/models/resnet50_defect_gpu_layer4.pth \
    --metrics-output reports/resnet50_split_test_metrics.json \
    --figures-dir reports/figures
```

**What this unified script does:**
- **Layer4 Fine-Tuning**: freezes conv1/layer1-3, unfreezes layer4 + FC head (~6.4M params).
- **Split-Aware**: loads Train from `train/` subdir and Validation from `val/` subdir.
- **Oversampling**: only oversamples the Train set (10× ratio) and keeps Validation/Test clean.
- **Early Stopping**: stops training if validation defect F1 does not improve for 8 consecutive epochs.
- **Plotting**: generates epoch-by-epoch loss, accuracy, and F1 curves in `reports/figures/`.
- **Threshold Tuning**: sweeps probabilities on the Validation split (only) to select the optimal threshold.
- **Test Evaluation**: runs inference on the unseen Test set split using the tuned threshold.
- **Metrics Report**: generates `reports/resnet50_split_test_metrics.json` and `.md`.
- **Misclassified Grid**: saves a grid of misclassified test examples to analyze model errors.

### If CUDA out-of-memory (OOM), retry with:
```bash
python scripts/train_defect_model.py \
    --data-dir data/image_dataset_split \
    --epochs 25 \
    --batch-size 32 \
    --lr 5e-4 \
    --backbone-lr 1e-5 \
    --oversample 10 \
    --patience 8 \
    --unfreeze-layer4 \
    --save-path ai_engine/models/resnet50_defect_gpu_layer4.pth \
    --metrics-output reports/resnet50_split_test_metrics.json \
    --figures-dir reports/figures
```

---

## Step 7 — Copy and Download Output Files

Prepare the outputs folder for downloading:
```bash
mkdir -p /kaggle/working/resnet50_project/outputs

cp /kaggle/working/resnet50_project/ai_engine/models/resnet50_defect_gpu_layer4.pth \
   /kaggle/working/resnet50_project/outputs/resnet50_defect_gpu_best.pth

cp /kaggle/working/resnet50_project/reports/resnet50_split_test_metrics.* \
   /kaggle/working/resnet50_project/outputs/ 2>/dev/null || true

cp /kaggle/working/resnet50_project/reports/figures/*.png \
   /kaggle/working/resnet50_project/outputs/ 2>/dev/null || true
```

After your session ends:
1. Go to your Notebook → **Output** tab
2. Download the `outputs/` folder containing:
   - `resnet50_defect_gpu_best.pth`
   - `resnet50_split_test_metrics.json`
   - `resnet50_split_test_metrics.md`
   - All curves, confusion matrix, and misclassified example plots
3. Place `resnet50_defect_gpu_best.pth` into your local `ai_engine/models/` directory.

---

## Quality Gate

| Metric | Target | CPU Baseline |
|---|---|---|
| Defect Recall | ≥ 0.80 | 0.4880 |
| Defect F1 | ≥ 0.85 | 0.4175 |
| Macro F1 | ≥ 0.85 | 0.6908 |

---

## Files summary

| File | Purpose |
|------|---------|
| `scripts/package_kaggle_resnet50.py` | Creates upload ZIP locally |
| `scripts/kaggle_gpu_train_setup.py` | Kaggle environment setup |
| `scripts/train_defect_model.py` | Unified training, validation early-stopping, threshold tuning, and testing |
| `ai_engine/models/resnet50_defect.pth` | CPU baseline checkpoint (DO NOT OVERWRITE) |
| `ai_engine/models/resnet50_defect_gpu_layer4.pth` | GPU checkpoint (new file) |
| `reports/resnet50_train_val_test_report.md` | Academic split training report |
