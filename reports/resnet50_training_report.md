# ResNet50 Defect Detection — Final Training Report

> Generated: 2026-06-07  
> Branch: `feature/label-image-v2`  
> Latest pulled commit (origin/main): `714f224` — Merge pull request #47 from ThanhChuong12/feat/optimize-shopee-scraper  
> Status: **FINAL** — training, threshold tuning, and evaluation all complete.

---

## 1. Executive Summary

This report documents the results of migrating the ResNet50 defect detection pipeline from a local CPU environment to a Kaggle GPU environment (Tesla T4), allowing for backbone fine-tuning.

The GPU model demonstrates a **strong improvement** over the CPU baseline, successfully passing the defect recall target of 0.80, while the defect F1 score improved significantly but did not quite reach the ambitious 0.85 target.

### Quality Gate Conclusion
* **Status**: **PARTIAL PASS / STRONG IMPROVEMENT**
* **Recall Target**: **PASSED** (0.8042 achieved vs. ≥ 0.80 target)
* **Defect F1 Target**: **NOT FULLY REACHED** (0.8042 achieved vs. ≥ 0.85 target)
* **Performance Gain**: The model F1 score for the defect class improved from **0.4175** (CPU baseline) to **0.8042** (GPU fine-tuned) — an absolute increase of **+38.67%**.

---

## 2. Git Workflow & Files Modified

### Git Workflow Summary

| Step | Result |
|------|--------|
| Branch | `feature/label-image-v2` |
| Pull strategy | `git stash` → `git pull origin main` → `git stash pop` |
| Conflicts resolved | `.gitignore` (kept upstream), `plan.md` (merged both sides) |
| Latest commit after pull | `714f224` |

### Files Modified

| File | Change |
|------|--------|
| `scripts/prepare_dataset.py` | Full rewrite — correct paths, CLI args, `wrong_item` → `defect`, balance check |
| `scripts/train_defect_model.py` | Full rewrite — support GPU/CPU, differential learning rates, unfreezing options |
| `scripts/tune_threshold.py` | Fixed default `--data-dir` from `data/processed` → `data/image_dataset` |
| `docs/resnet50_training_flow.md` | Created — full workflow documentation |
| `reports/resnet50_training_report.md` | Updated with final GPU results and comparison (this file) |

---

## 3. Dataset & Class Balance

### Dataset Distribution

**Command:**
```bash
python scripts/prepare_dataset.py \
    --images-root image_labeling \
    --output-dir data/image_dataset \
    --label-map binary
```

| Source Label | Binary Class | Count |
|-------------|-------------|-------|
| `intact` | `no-defect` | 17,851 |
| `damaged` | `defect` | 1,297 |
| `wrong_item` | `defect` | 363 |
| `irrelevant` | EXCLUDED | 8,232 |
| **Total defect** | — | **1,660** |
| **Total no-defect** | — | **17,851** |

* **Class Imbalance**: Severe (10.75:1 ratio).
* **Oversampling**: 10x oversampling was applied to the minority class (`defect`) in training to achieve an effective 1:1 balance in the training batches.
* **Validation Split**: 20% stratified (3,903 total samples: 332 defect, 3,571 no-defect).

---

## 4. CPU Baseline Training (Frozen Backbone)

The CPU baseline model was trained with a completely frozen ResNet50 backbone (only training the newly attached fully-connected head).

* **Trainable Parameters**: 1,051,138 / 24,559,170 (4.3%)
* **Device**: CPU (~25–35 min/epoch)
* **Best Epoch**: 8 (with early stopping triggered at epoch 20)
* **Best Threshold**: 0.500

### CPU Baseline Metrics
* **Defect Precision**: 0.3649
* **Defect Recall**: 0.4880
* **Defect F1**: 0.4175
* **Macro F1**: 0.6766
* **ROC-AUC**: 0.8230

---

## 5. GPU Training (Fine-tuned Backbone)

With GPU acceleration, the backbone's `layer4` was unfrozen and trained using a differential learning rate.

### Environment & Configuration

| Parameter | Value |
|-----------|-------|
| **GPU Device** | Tesla T4 |
| **Training Mode** | ResNet50 `layer4` fine-tuning |
| **Trainable Parameters** | 6,407,682 / 24,559,170 (26.1%) |
| **Epochs** | 25 |
| **Batch Size** | 64 |
| **Optimizer** | Adam (differential learning rates) |
| **Learning Rate (FC head)** | 5e-4 |
| **Learning Rate (layer4)** | 1e-5 |
| **Patience** | 8 |
| **Loss Function** | FocalLoss (gamma=2.0) + dynamic class weights |

### Training History & Best Checkpoint

* **Best Checkpoint Epoch**: 9
* **Recommended Inference Threshold**: 0.525
* **Saved Path**: `ai_engine/models/resnet50_defect_gpu_best.pth`

---

## 6. Model Comparison & Metrics

### Confusion Matrix (GPU Model)

| | Predicted: no-defect | Predicted: defect |
|---|---|---|
| **Actual: no-defect** | 3,506 (TN) | 65 (FP) |
| **Actual: defect** | 65 (FN) | 267 (TP) |

* **Confusion Matrix Plot**: `reports/figures/confusion_matrix_resnet50_(defect_detection).png`

### Detailed Comparison Table

The table below contrasts the final GPU results with the CPU baseline:

| Metric | CPU Baseline (Frozen) | GPU Model (layer4 Fine-tuned) | Absolute Change | Status |
|--------|----------------------|------------------------------|-----------------|--------|
| **Device** | CPU | Tesla T4 | — | — |
| **Training Mode** | Frozen Backbone | `layer4` Fine-tuning | — | — |
| **Best Epoch** | 8 | 9 | — | — |
| **Best Threshold** | 0.500 | **0.525** | — | — |
| **Defect Precision** | 0.3649 | **0.8042** | **+43.93%** | ✅ Improved |
| **Defect Recall** | 0.4880 | **0.8042** | **+31.62%** | ✅ Passed Target (≥ 0.80) |
| **Defect F1-Score** | 0.4175 | **0.8042** | **+38.67%** | ⚠️ Missed Target (≥ 0.85) |
| **Macro F1-Score** | 0.6766 | **0.8930** | **+21.64%** | ✅ Passed Target (≥ 0.85) |
| **ROC-AUC** | 0.8230 | **0.9619** | **+13.89%** | ✅ Improved |

---

## 7. Quality Gate Assessment

### Project Metric Targets

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| **Defect Recall** | ≥ 0.80 | **0.8042** | ✅ **PASSED** |
| **Defect F1** | ≥ 0.85 | **0.8042** | ❌ **NOT FULLY REACHED** (Missed by 0.0458) |
| **Macro F1** | ≥ 0.85 | **0.8930** | ✅ **PASSED** |

### Findings & Analysis
* **Why F1 target 0.85 was not fully reached**: The dataset contains 1,660 defect samples compared to 17,851 non-defect samples. Even with 10x oversampling and Focal Loss, the severe class imbalance presents a major challenge for maximizing precision without hurting recall.
* **Why recall passed**: The combination of `layer4` fine-tuning on GPU, Focal Loss, and oversampling successfully adapted the high-level features of the ResNet50 model to the product defect domain, making it highly sensitive to defects while maintaining high specificity.

---

## 8. Recommendations & Inference Configuration

1. **Inference Threshold**: We recommend using the threshold of **0.525** for production deployments. This threshold balances defect detection sensitivity (recall) and false alarm rate (precision) at exactly 80.42% each.
2. **Next Steps to Reach F1 ≥ 0.85**:
   * **Full Backbone Fine-tuning**: Run training with `training_mode="all"` (completely unfreezing ResNet50) at a very low learning rate (e.g., `1e-5`) on GPU.
   * **More Labeled Data**: Collect and label additional defect images (aiming for 3,000+ total defect images) to reduce class imbalance.
   * **Alternative Architectures**: Experiment with newer backbones such as `EfficientNet-B3` or `ConvNeXt-Tiny`, which have shown higher representational power on small datasets.

---

## 9. Output Artifacts

The CPU baseline checkpoint is kept intact and not overwritten to allow for rollback or further analysis.

| Artifact | Path | Status / Description |
|----------|------|----------------------|
| **CPU Checkpoint (Baseline)** | `ai_engine/models/resnet50_defect.pth` | ✅ **Preserved** (F1 = 0.4175) |
| **GPU Checkpoint (Best)** | `ai_engine/models/resnet50_defect_gpu_best.pth` | 📦 **External Artifact** (F1 = 0.8042) |
| **Confusion Matrix Plot** | `reports/figures/confusion_matrix_resnet50_(defect_detection).png` | ✅ **Updated** (GPU confusion matrix) |

---

## 10. External Artifact Policy & Restoration Instructions

To keep the GitHub repository lightweight, large training outputs and binary model checkpoints are stored externally. The repository contains source code, configuration, documentation, reports, and scripts only.

### Inference Configuration
* **Required Inference Threshold**: `DEFECT_THRESHOLD=0.525`
* **Default Checkpoint Path**: `ai_engine/models/resnet50_defect_gpu_best.pth`

### Restoring the Model Locally
1. Download `resnet50_defect_gpu_best.pth` from the external Drive/artifact storage.
2. Place the file under `ai_engine/models/` in your local workspace.
3. The inference pipeline `ai_engine/image_processing/defect_detection.py` will load this file by default, or you can override it via the environment variable `RESNET_WEIGHTS_PATH`.

### Git LFS Safeguard
While `.gitattributes` is configured to track `*.pth` files using Git LFS as a safeguard, the current project policy is to **not commit any checkpoint binaries directly to GitHub**. Keep all `.pth` files listed in `.gitignore` to prevent accidental staging.
