# ResNet50 GPU Training Report — Kaggle

> **Status:** ⬜ PENDING — Fill in after Kaggle training, threshold tuning, and evaluation complete.  
> Template version: 2026-06-07  
> CPU baseline for comparison: defect_f1=0.4175, defect_recall=0.488, macro_f1=0.6766

---

## 1. Environment

| Field | Value |
|-------|-------|
| Kaggle GPU name | *(e.g., Tesla P100-PCIE-16GB)* — UNVERIFIED until run |
| CUDA version | *(e.g., 12.1)* — UNVERIFIED |
| PyTorch version | *(e.g., 2.3.0+cu121)* — UNVERIFIED |
| CUDA available | *(True/False)* — UNVERIFIED |

---

## 2. Dataset

| Field | Value |
|-------|-------|
| defect images | 1,660 |
| no-defect images | 17,851 |
| Imbalance ratio (raw) | 10.75:1 |
| Oversample factor used | *(e.g., 10x)* |
| Effective train ratio after oversample | *(e.g., 1.1:1)* |
| Val split | 20% (3,903 samples) |
| Val defect count | 332 |
| Val no-defect count | 3,571 |

---

## 3. Training Configuration

| Parameter | Value |
|-----------|-------|
| Training command | *(paste full command used)* |
| training_mode | `layer4` (layer4 + FC unfrozen) |
| Trainable params | *(e.g., 6,407,682 / 24,559,170 — 26.1%)* |
| --epochs | 25 |
| --batch-size | *(64 or 32 if OOM)* |
| --lr (FC head) | 5e-4 |
| --backbone-lr (layer4) | 1e-5 |
| --oversample | 10 |
| --patience | 8 |
| Loss | FocalLoss (gamma=2.0) + class weights |
| Optimizer | Adam (differential LR) |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |

---

## 4. Training History

> Fill in per-epoch log from Kaggle output.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Defect P | Defect R | Defect F1 | Macro F1 | LR |
|-------|-----------|-----------|----------|---------|----------|----------|-----------|----------|----|
| 1 | | | | | | | | | |
| 2 | | | | | | | | | |
| … | | | | | | | | | |
| best | | | | | | | | | |

Early stopping triggered at epoch: *(fill in)*

---

## 5. Best Checkpoint

| Field | Value |
|-------|-------|
| **Checkpoint path** | `ai_engine/models/resnet50_defect_gpu_layer4.pth` |
| **Best epoch** | *(fill in)* |
| **training_mode** | `layer4` |
| **device** | `cuda` |
| **gpu_name** | *(fill in from checkpoint)* |

---

## 6. Threshold Tuning

> Run: `python scripts/tune_threshold.py --model-path ai_engine/models/resnet50_defect_gpu_layer4.pth ...`

| Field | Value |
|-------|-------|
| **Best threshold** | *(e.g., 0.45)* — UNVERIFIED until run |
| **At best threshold — Precision** | *(fill in)* |
| **At best threshold — Recall** | *(fill in)* |
| **At best threshold — F1** | *(fill in)* |
| ROC-AUC | *(fill in)* |

Full threshold sweep table:
*(paste from tune_threshold.py output)*

---

## 7. Final Evaluation Metrics

> Source: `python scripts/evaluate_models.py image ...`

### Confusion Matrix

|  | Predicted: no-defect | Predicted: defect |
|--|---------------------|------------------|
| **Actual: no-defect** | *(TN)* | *(FP)* |
| **Actual: defect** | *(FN)* | *(TP)* |

**Confusion matrix plot:** `reports/figures/confusion_matrix_resnet50_(defect_detection).png`

### Metrics Summary

| Metric | GPU Model (layer4) | CPU Baseline (frozen) | Change |
|--------|-------------------|-----------------------|--------|
| **Best epoch** | *(fill in)* | 8 | — |
| **training_mode** | layer4 | frozen | — |
| **defect_precision** | *(fill in)* | 0.3649 | *(+/- %)* |
| **defect_recall** | *(fill in)* | 0.4880 | *(+/- %)* |
| **defect_f1** | *(fill in)* | 0.4175 | *(+/- %)* |
| **macro_f1** | *(fill in)* | 0.6766 | *(+/- %)* |
| **val_acc** | *(fill in)* | 0.8842 | *(+/- %)* |
| **ROC-AUC** | *(fill in)* | 0.8230 | *(+/- %)* |
| **best_threshold** | *(fill in)* | 0.500 | — |

---

## 8. Quality Gate Assessment

### Minimum improvement (must beat CPU baseline)

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| defect_f1 > 0.4175 | > 0.4175 | *(fill in)* | ⬜ PENDING |
| defect_recall ≥ 0.488 | ≥ 0.488 | *(fill in)* | ⬜ PENDING |

### Target (project goal)

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| defect_f1 ≥ 0.85 | ≥ 0.85 | *(fill in)* | ⬜ PENDING |
| defect_recall ≥ 0.80 | ≥ 0.80 | *(fill in)* | ⬜ PENDING |

### Overall Quality Gate: ⬜ PENDING

*(Replace PENDING with one of:)*
- ✅ PASSED — both defect_f1 ≥ 0.85 and defect_recall ≥ 0.80
- ⚠️ PARTIAL — improved over CPU but did not reach target
- ❌ FAILED — no improvement over CPU baseline

---

## 9. Explanation (if defect_f1 < 0.85)

> Fill in if the target is not reached.

Potential reasons to check:
- [ ] Was the backbone fully frozen (training_mode should be `layer4`, not `frozen`)?
- [ ] Was the GPU actually used (device should be `cuda`)?
- [ ] Did the training run for enough epochs before early stopping?
- [ ] Was the defect_recall unusually low (possible threshold issue)?
- [ ] Was oversample set high enough (>=10)?

Recommendations if still below target:
1. Try `--no-freeze` (full backbone fine-tuning) at `--lr 1e-5` for all layers
2. Try `--oversample 15` for more defect exposure
3. Lower threshold to 0.35–0.45 to trade precision for recall
4. Label more defect images (target 3,000+)
5. Try EfficientNet-B3 or ConvNeXt-Tiny backbone

---

## 10. Output Artifacts

| Artifact | Path | Status |
|----------|------|--------|
| GPU checkpoint | `ai_engine/models/resnet50_defect_gpu_layer4.pth` | ⬜ PENDING |
| CPU backup | `ai_engine/models/resnet50_defect_cpu_backup.pth` | ⬜ PENDING |
| Confusion matrix | `reports/figures/confusion_matrix_resnet50_(defect_detection).png` | ⬜ PENDING |
| Training instructions | `docs/kaggle_gpu_training_instructions.md` | ✅ |
| This report template | `reports/resnet50_gpu_training_report_template.md` | ✅ |

---

> **Instructions**: Replace all *(fill in)* and ⬜ PENDING values with actual results from Kaggle.  
> **Do not claim success until training, threshold tuning, and final evaluation are all complete.**  
> Every metric must be supported by Kaggle logs or checkpoint metadata — if unverified, write UNVERIFIED.
