# Kaggle GPU Training Instructions — ResNet50 Defect Detection

> Purpose: Fine-tune ResNet50 on Kaggle GPU to improve defect detection beyond the CPU baseline.  
> CPU baseline: defect_f1 = 0.4175, defect_recall = 0.488, training_mode = frozen  
> Target: defect_f1 ≥ 0.85, defect_recall ≥ 0.80

---

## Overview

The CPU-trained ResNet50 model plateaued because only the FC head was trained (4.3% of parameters). On Kaggle GPU we will unfreeze `layer4` of ResNet50 alongside the FC head (~6.4M trainable parameters) using differential learning rates. This is called **layer4 fine-tuning**.

---

## Step 1 — Generate the ZIP locally

Run this on your local machine from the project root:

```bash
python scripts/package_kaggle_resnet50.py
```

Expected output:
```
resnet50_kaggle_train.zip   (in project root)
```

The script will print the ZIP path, file size, and image counts. Verify before uploading.

---

## Step 2 — Upload to Kaggle

1. Go to [kaggle.com](https://www.kaggle.com) → **Datasets** → **New Dataset**
2. Name: `resnet50-kaggle-train` (Kaggle will slugify this)
3. Upload: `resnet50_kaggle_train.zip`
4. Set visibility to **Private**
5. Click **Create**

---

## Step 3 — Create a Kaggle Notebook

1. Go to **Notebooks** → **New Notebook** → **Script** (not notebook)
2. Click **Add Input** → your uploaded dataset → `resnet50-kaggle-train`
3. Go to **Settings** (right panel):
   - **Accelerator**: GPU P100 *(preferred — 16 GB VRAM, stable)*
   - If P100 is not available: GPU T4 x1
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

This script:
- Finds the uploaded data under `/kaggle/input`
- Extracts the ZIP if needed
- Copies everything to `/kaggle/working/resnet50_project` (read-write)
- Verifies GPU availability
- Verifies all required files
- Counts dataset images
- Backs up the CPU checkpoint to `resnet50_defect_cpu_backup.pth`
- Prints the recommended training commands

---

## Step 6 — Main GPU training command

```bash
cd /kaggle/working/resnet50_project

python scripts/train_defect_model.py \
    --data-dir data/image_dataset \
    --epochs 25 \
    --batch-size 64 \
    --lr 5e-4 \
    --backbone-lr 1e-5 \
    --oversample 10 \
    --patience 8 \
    --unfreeze-layer4 \
    --save-path ai_engine/models/resnet50_defect_gpu_layer4.pth
```

**What this does:**
- `--unfreeze-layer4`: freezes conv1/layer1-3, unfreezes layer4 + FC head (~6.4M params)
- `--lr 5e-4`: learning rate for FC head
- `--backbone-lr 1e-5`: smaller LR for layer4 (avoids catastrophic forgetting)
- `--oversample 10`: 10× defect oversampling → near-balanced training set
- `--patience 8`: stop if no defect F1 improvement for 8 consecutive epochs
- `--save-path`: new GPU checkpoint (does NOT overwrite CPU checkpoint)
- Checkpoint is saved on **defect F1 improvement only**, never on accuracy

### If CUDA out-of-memory (OOM), retry with:

```bash
python scripts/train_defect_model.py \
    --data-dir data/image_dataset \
    --epochs 25 \
    --batch-size 32 \
    --lr 5e-4 \
    --backbone-lr 1e-5 \
    --oversample 10 \
    --patience 8 \
    --unfreeze-layer4 \
    --save-path ai_engine/models/resnet50_defect_gpu_layer4.pth
```

---

## Step 7 — Threshold tuning

After training completes, find the best decision threshold:

```bash
python scripts/tune_threshold.py \
    --model-path ai_engine/models/resnet50_defect_gpu_layer4.pth \
    --data-dir data/image_dataset \
    --val-split 0.2 \
    --seed 42 \
    --batch-size 64
```

Note the best threshold printed (e.g., `Best F1 threshold = 0.XX`).

---

## Step 8 — Final evaluation

```bash
python scripts/evaluate_models.py image \
    --model-path ai_engine/models/resnet50_defect_gpu_layer4.pth \
    --data-path data/image_dataset \
    --batch-size 64
```

This generates:
- Classification report (precision/recall/F1 per class)
- Confusion matrix
- `reports/figures/confusion_matrix_resnet50_(defect_detection).png`

---

## Step 9 — Collect output files

```bash
mkdir -p /kaggle/working/resnet50_project/outputs

cp /kaggle/working/resnet50_project/ai_engine/models/resnet50_defect_gpu_layer4.pth \
   /kaggle/working/resnet50_project/outputs/resnet50_defect_gpu_best.pth

cp /kaggle/working/resnet50_project/reports/figures/confusion_matrix_resnet50_*.png \
   /kaggle/working/resnet50_project/outputs/ 2>/dev/null || true
```

---

## Step 10 — Download results from Kaggle

Kaggle saves outputs automatically. After your session ends:

1. Go to your Notebook → **Output** tab
2. Download:
   - `resnet50_project/outputs/resnet50_defect_gpu_best.pth`
   - `resnet50_project/outputs/confusion_matrix_resnet50_*.png`
3. Place the `.pth` file in your local `ai_engine/models/` directory
4. Fill in `reports/resnet50_gpu_training_report_template.md` with the final metrics

---

## Quality Gate

| Criterion | Target | CPU Baseline |
|-----------|--------|-------------|
| defect_f1 > 0.4175 | Minimum | 0.4175 |
| defect_f1 ≥ 0.85 | Goal | 0.4175 |
| defect_recall ≥ 0.80 | Goal (should not collapse) | 0.488 |

> ⚠️ **Do not report success based on accuracy alone.** The validation set is 10.75:1 imbalanced. Defect F1 and defect recall are the primary success metrics.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `--batch-size` from 64 to 32 |
| Dataset not found | Check `find /kaggle/input -maxdepth 4 -name "*.pth"` |
| Module not found | Verify `kaggle_gpu_train_setup.py` added project to `sys.path` |
| No GPU available | Go to Notebook Settings → Accelerator → GPU P100 |
| Recall collapses | Try lower threshold (e.g., 0.35–0.45) in `tune_threshold.py` |
| F1 doesn't improve | Try `--lr 1e-4 --backbone-lr 5e-6` with more patience |

---

## Files summary

| File | Purpose |
|------|---------|
| `scripts/package_kaggle_resnet50.py` | Creates upload ZIP locally |
| `scripts/kaggle_gpu_train_setup.py` | Kaggle environment setup |
| `scripts/train_defect_model.py` | Training (updated with `--unfreeze-layer4`) |
| `scripts/tune_threshold.py` | Post-training threshold search |
| `scripts/evaluate_models.py` | Final evaluation + confusion matrix |
| `ai_engine/models/resnet50_defect.pth` | CPU baseline (DO NOT OVERWRITE) |
| `ai_engine/models/resnet50_defect_gpu_layer4.pth` | GPU checkpoint (new file) |
| `reports/resnet50_gpu_training_report_template.md` | Report template to fill in |
