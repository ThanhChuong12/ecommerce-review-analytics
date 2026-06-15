# ResNet50 Defect Detection Experimental Report (Train/Val/Test Split)

> **Status:** `COMPLETE` (GPU training executed on Kaggle; final results evaluated on unseen Test split)  
> **Dataset Split Structure:** Stratified 70% Train / 15% Validation / 15% Test  
> **Random Seed:** 42  

---

## 1. Data Split Rationale

A clean evaluation methodology is essential for developing a production-ready machine learning model, particularly in high-imbalance settings such as defect detection. Here, the dataset consists of **1,660 defect** images and **17,851 no-defect** images (approx. 10.75:1 imbalance). 

To ensure unbiased evaluation, we utilize a stratified **70% Train / 15% Validation / 15% Test** split:

| Split | Defect Count | No-Defect Count | Total Count | Class Ratio | Percentage |
|---|---|---|---|---|---|
| **Train** | 1,162 | 12,495 | 13,657 | ~8.51% defect | 70% |
| **Validation** | 249 | 2,678 | 2,927 | ~8.51% defect | 15% |
| **Test** | 249 | 2,678 | 2,927 | ~8.51% defect | 15% |
| **Total** | **1,660** | **17,851** | **19,511** | **~8.51% defect** | **100%** |

### Why Stratification is Appropriate
With severe imbalance, a standard random split risks partitioning the minority class unevenly, potentially leaving the validation or test sets with too few defect images to construct stable metrics. Stratified splitting enforces that the ~8.5% defect class proportion is strictly maintained across all splits, ensuring stable evaluation metrics.

### Split Roles & Information Leakage Prevention
1. **Train Set (70%):** Used exclusively for weight optimization.
2. **Validation Set (15%):** Used for monitoring validation loss, triggering early stopping, selecting the best epoch checkpoint, and tuning the post-training decision threshold.
3. **Test Set (15%):** Left entirely untouched until the model, weights, and decision threshold are fully frozen. Evaluating on this test set guarantees that the final metrics represent true generalizability without lookahead bias.

---

## 2. Preprocessing & Augmentation Pipeline

To prevent data leakage, different preprocessing configurations are applied to each split:

```
               [ Labeled Images ]
                       |
               (70/15/15 Split)
                       |
     ┌─────────────────┼─────────────────┐
     ▼                 ▼                 ▼
[ Train set ]     [ Val set ]      [ Test set ]
     │                 │                 │
(Augmentation)         │                 │
(Oversampling 10x)     │                 │
     │                 │                 │
     ▼                 ▼                 ▼
[Resize 224x224]  [Resize 224x224]  [Resize 224x224]
[Normalize]       [Normalize]       [Normalize]
(Deterministic)   (Deterministic)   (Deterministic)
```

### 1. Training Set Preprocessing
* **Oversampling:** Defect class images are oversampled by a factor of 10x (yielding 11,620 defect instances and 12,495 no-defect instances) to bring the class distribution close to 1:1.
* **Random Augmentations:** Applied dynamically to oversampled defect samples to increase variance:
  * Random horizontal flip
  * Random rotation (up to 15 degrees)
  * Random color jitter (brightness, contrast, saturation, and hue)
* **Standard Preprocessing:** Resize to $224 \times 224$ pixels, conversion to tensor, and ImageNet channel-wise normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 2. Validation & Test Set Preprocessing
* **Clean & Deterministic:** No oversampling or random augmentation is applied. Images are subjected only to resizing ($224 \times 224$), tensor conversion, and normalization. This guarantees that validation and test metrics accurately reflect the true unbalanced distribution of the production environment.

---

## 3. Model Parameters & Fine-Tuning Strategy

The network architecture is based on the ResNet50 model (He et al. 2016).

### 1. Freeze/Unfreeze Backbone Strategy
* **CPU Baseline (Frozen Backbone):** All convolutional layers of ResNet50 are frozen. Only the newly attached classification head is trained.
  * **Trainable Parameters:** 1,051,138 / 24,559,170 (approx. 4.3%).
* **GPU Fine-Tuning (layer4 Unfrozen):** To adapt the high-level convolutional features to domain-specific product details, the `layer4` block of the ResNet50 backbone is unfrozen along with the FC head.
  * **Trainable Parameters:** 6,407,682 / 24,559,170 (approx. 26.1%).

### 2. Hyperparameters & Loss Function
* **Optimizer:** Adam.
* **Learning Rates (Differential):**
  * Fully Connected (FC) head: $5 \times 10^{-4}$ (to quickly adapt the new classification layers).
  * `layer4` backbone: $1 \times 10^{-5}$ (a small rate to adjust features without corrupting pre-trained weights).
* **Patience:** 8 epochs (early stopping monitored on Validation Defect F1).
* **Loss Function:** Focal Loss (Lin et al. 2017) to prioritize hard-to-classify examples:
  $$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  * $\gamma = 2.0$
  * $\alpha_t$ dynamically computed from the inverse class frequencies in the training batch to balance class gradients.

---

## 4. Experimental Results

The following tables represent the comparison between the CPU baseline (frozen backbone) and the GPU fine-tuned `improve` experiment evaluated on the unseen Test split.

### 1. Training Statistics

| Parameter | CPU Baseline (Frozen) | GPU Model (layer4 Unfrozen - Improve) |
|---|---|---|
| **Best Epoch** | 8 | 9 |
| **Best Threshold (Val)** | 0.500 | 0.550 |

### 2. Official Evaluation on unseen Test Set

| Metric | CPU Baseline (Frozen) | GPU Model (layer4 Unfrozen - Improve) | Status / Target |
|---|---|---|---|
| **Test Accuracy** | 0.890000 | 0.883157 | — |
| **Test Precision (no-defect)** | 0.950000 | 0.948196 | — |
| **Test Recall (no-defect)** | 0.930000 | 0.922704 | — |
| **Test F1 (no-defect)** | 0.940000 | 0.935276 | — |
| **Test Precision (defect)** | 0.364900 | 0.355140 | — |
| **Test Recall (defect)** | 0.488000 | 0.457831 | **Target ≥ 0.80** (FAIL ❌) |
| **Test F1 (defect)** | 0.417500 | 0.400000 | **Target ≥ 0.85** (FAIL ❌) |
| **Macro F1-Score** | 0.676600 | 0.667638 | **Target ≥ 0.85** (FAIL ❌) |
| **ROC-AUC** | 0.823000 | 0.827362 | — |

### 3. Confusion Matrix (Test Set)

| Actual \ Predicted | Predicted: no-defect | Predicted: defect |
|---|---|---|
| **Actual: no-defect** | 2471 (TN) | 207 (FP) |
| **Actual: defect** | 135 (FN) | 114 (TP) |

---

## 5. Experimental Figures

The final report figures are saved in the `reports/figures_resnet50_improve/` folder:

1. **Loss Curves (`reports/figures_resnet50_improve/resnet50_improve_loss_curve.png`):** Shows Train vs Validation loss across epochs.
2. **Accuracy Curves (`reports/figures_resnet50_improve/resnet50_improve_accuracy_curve.png`):** Visualizes Train vs Validation accuracy.
3. **Validation Defect F1 Curve (`reports/figures_resnet50_improve/resnet50_improve_defect_f1_curve.png`):** Monitored for early stopping and checkpoint selection.
4. **Threshold Tuning Curve (`reports/figures_resnet50_improve/resnet50_improve_threshold_tuning.png`):** Illustrates the precision-recall trade-off on the validation split.
5. **Confusion Matrix (`reports/figures_resnet50_improve/confusion_matrix_resnet50_improve_test.png`):** Summarizes correct and incorrect predictions on the Test set.
6. **Misclassified Grid (`reports/figures_resnet50_improve/resnet50_improve_test_misclassified_examples.png`):** Visual grid of test images misclassified by the model.

---

## 6. Quality Gate Validation

| Criterion | Target | Achieved | Status |
|---|---|---|---|
| **Defect Recall (Test)** | ≥ 0.80 | 0.457831 | FAIL ❌ |
| **Defect F1 (Test)** | ≥ 0.85 | 0.400000 | FAIL ❌ |
| **Macro F1 (Test)** | ≥ 0.85 | 0.667638 | FAIL ❌ |

**Conclusion:** Although the `resnet50_improve` model provides valuable learning parameters, evaluating on the clean unseen Test set shows that it fails all target quality gates due to the extreme class imbalance and domain shift.

---

## 7. Future Work & Recommendations

1. **Full Backbone Fine-tuning:** If unfreezing `layer4` is insufficient to reach the 0.85 F1 target, unfreeze all layers of ResNet50 using a very low learning rate (e.g. $1 \times 10^{-6}$) to fine-tune shallower layers.
2. **Alternative Architectures:** Evaluate `EfficientNet-B3` or `ConvNeXt-Tiny` backbones to compare against ResNet50.
3. **Data Expansion:** Collect and label additional defect instances to reduce the reliance on extreme oversampling.

---

## 8. Citations

* He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778).
* Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In *Proceedings of the IEEE international conference on computer vision* (pp. 2980-2988).
