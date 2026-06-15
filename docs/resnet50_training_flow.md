# ResNet50 Defect Detection — Training Workflow

> Created: 2026-06-06  
> Branch: `feature/label-image-v2`  
> Purpose: Document the full reproducible pipeline from raw labels to a trained ResNet50 defect detection model.

---

## 1. Overview

This pipeline trains a ResNet50 binary classifier to detect product defects in review images.

| Item | Value |
|------|-------|
| Model | ResNet50 (ImageNet pretrained backbone + custom MLP head) |
| Classes | `0: no-defect`, `1: defect` |
| Source labels | `image_labeling/data/manifests/labels.csv` |
| Source images | `image_labeling/data/labeled/{damaged,intact,wrong_item,irrelevant}/` |
| Generated dataset | `data/image_dataset/{defect,no-defect}/` |
| Best checkpoint | `ai_engine/models/resnet50_defect.pth` |

---

## 2. Label Mapping

| Raw label (from labeling tool) | Binary class | Class index |
|-------------------------------|--------------|-------------|
| `intact`     | `no-defect` | 0 |
| `damaged`    | `defect`    | 1 |
| `wrong_item` | `defect`    | 1 |
| `irrelevant` | **EXCLUDED** | — |

**Source of truth**: `image_labeling/data/labeled/` subdirectories.  
The `labels.csv` is used to identify labels; actual images are read from the labeled folder structure.

---

## 3. Dataset Statistics (from labels.csv)

| Label | Count |
|-------|-------|
| intact | 17,851 |
| irrelevant | 8,232 (excluded) |
| damaged | 1,297 |
| wrong_item | 363 |
| **Total defect** | **1,660** |
| **Total no-defect** | **17,851** |
| **Imbalance ratio** | **~10.75:1** |

> **Class imbalance handling**: Use `--oversample 20` (oversampling defect 20x in train set).
> Also uses FocalLoss with class weights.

---

## 4. Preprocessing Pipeline

### Train Transforms (`is_train=True`, label=1 only)
```
HorizontalFlip (p=0.5)
Rotate ±15° (p=0.5, border constant)
GaussNoise (p=0.3)
RandomBrightnessContrast (p=0.3)
Resize(224, 224)
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ToTensorV2()
```

### Validation Transforms (`is_train=False` OR label=0)
```
Resize(224, 224)
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ToTensorV2()
```

**Important**: Validation set uses NO augmentation. Train/val split is deterministic (`random_state=42`, stratified).

---

## 5. Execution Order

```
Phase 0  Pull latest code from GitHub
Phase 1  Create this documentation file
Phase 2  Prepare dataset from labels.csv
Phase 3  Check class balance
Phase 4  Fix ResNet50 training code
Phase 5  Train ResNet50
Phase 6  Tune threshold
Phase 7  Evaluate final model
Phase 8  Write quality gate report
```

---

## 6. Commands

### Phase 2 — Prepare Dataset

```bash
python scripts/prepare_dataset.py \
    --labels-csv image_labeling/data/manifests/labels.csv \
    --images-root image_labeling \
    --output-dir data/image_dataset \
    --label-map binary
```

**Expected output:**
```
data/image_dataset/
├── defect/       # damaged + wrong_item
└── no-defect/    # intact only
```

### Phase 3 — Check Balance

```bash
python scripts/prepare_dataset.py \
    --check-balance \
    --data-dir data/image_dataset
```

### Phase 5 — Train

```bash
python scripts/train_defect_model.py \
    --data-dir data/image_dataset \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.001 \
    --oversample 20 \
    --patience 8 \
    --save-path ai_engine/models/resnet50_defect.pth
```

**Hyperparameter rationale:**
- `--oversample 20`: With 1660 defect vs 17851 no-defect (10.75:1 ratio), 20x oversample brings ratio to ~1:1
- `--patience 8`: Gives model more time to escape local minima vs old `patience=5`
- `--epochs 30`: Old model stopped at epoch 3; needs more epochs to converge
- `--lr 0.001`: Standard Adam LR for frozen backbone (only FC head trained by default)

### Phase 6 — Threshold Tuning

```bash
python scripts/tune_threshold.py \
    --model-path ai_engine/models/resnet50_defect.pth \
    --data-dir data/image_dataset \
    --val-split 0.2 \
    --seed 42
```

### Phase 7 — Evaluate

```bash
python scripts/evaluate_models.py image \
    --model-path ai_engine/models/resnet50_defect.pth \
    --data-path data/image_dataset \
    --batch-size 16
```

---

## 7. Model Architecture

```
ResNet50 (ImageNet pretrained backbone)
  └── All conv/bn layers (optionally frozen)
  └── fc = Sequential(
        Linear(2048 → 512),
        BatchNorm1d(512),
        ReLU(),
        Dropout(0.5),
        Linear(512 → 2)       # no-defect=0, defect=1
      )
```

**Key design notes:**
- Backbone frozen by default (`--freeze-backbone`, only 4.28% params trainable)
- FocalLoss (gamma=2.0) + dynamic class weights to handle imbalance
- Early stopping on **Defect F1** (not val_loss or val_acc)
- Checkpoint saved when defect_f1 improves

---

## 8. Output Artifacts

| Artifact | Path |
|----------|------|
| Prepared dataset (defect) | `data/image_dataset/defect/` |
| Prepared dataset (no-defect) | `data/image_dataset/no-defect/` |
| Best checkpoint | `ai_engine/models/resnet50_defect.pth` |
| Confusion matrix plot | `reports/figures/confusion_matrix_resnet50_(defect_detection).png` |
| Workflow documentation | `docs/resnet50_training_flow.md` |
| Final training report | `reports/resnet50_training_report.md` |

---

## 9. Quality Gate

| Metric | Minimum (pass over old) | Target |
|--------|------------------------|--------|
| defect_recall | > 0.637 | ≥ 0.80 |
| defect_f1 | > 0.378 | ≥ 0.85 |
| macro_f1 | — | ≥ 0.85 |

**Old checkpoint baseline (epoch=3):**
- val_acc: 85.8% (misleading due to class imbalance)  
- defect_recall: 0.637  
- defect_f1: 0.378  

If target not met, reasons may include:
- Dataset size (1660 defect images may still be insufficient)
- Label noise in `wrong_item` category
- Image quality variation (screenshots vs product photos)

---

## 10. File Locations

| File | Purpose |
|------|---------|
| `scripts/prepare_dataset.py` | Dataset preparation from labels.csv |
| `scripts/train_defect_model.py` | ResNet50 training |
| `scripts/tune_threshold.py` | Threshold optimization |
| `scripts/evaluate_models.py` | Model evaluation |
| `ai_engine/image_processing/defect_detection.py` | Dataset class, model arch, FocalLoss, inference |
| `ai_engine/image_processing/augmentation/transforms.py` | Albumentations pipelines |
| `image_labeling/data/manifests/labels.csv` | Ground truth labels (27,743 rows) |
| `image_labeling/data/labeled/` | Source labeled images |
| `data/image_dataset/` | Generated binary ImageFolder |
| `ai_engine/models/resnet50_defect.pth` | Trained model checkpoint |
