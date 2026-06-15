# MobileNetV3 Model 2 Experiment Results Summary

This document summarizes the performance of the MobileNetV3-Large Model 2 experiments on the fixed physical dataset split for defect detection.

## Experiment Context

- **Model Backbone:** MobileNetV3-Large
- **Dataset Split Protocol:** Fixed physical split (no random split, no tuning on Test set)
- **Classes:** `defect` (class 0 or target) and `no-defect` (class 1)
- **Dataset Split Counts:**
  - **Train:** 1,162 defect, 12,495 no-defect (total: 13,657)
  - **Validation:** 249 defect, 2,678 no-defect (total: 2,927)
  - **Test:** 249 defect, 2,678 no-defect (total: 2,927)
- **Training Environment:** Kaggle Tesla T4 GPU, Internet OFF
- **Training Hyperparameters:** Batch size = 32, Learning rate = 1e-3, Epochs = 15

---

## Experiment Results

### A. MobileNetV3 Baseline (Selected Main Result)

This experiment uses the standard threshold (`0.5`) and is selected as our final MobileNetV3 result due to superior accuracy, Macro-F1, and Defect F1.

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.7947 |
| **Macro-F1** | 0.6290 |
| **Defect Precision** | 0.26 |
| **Defect Recall** | 0.74 |
| **Defect F1** | 0.38 |
| **Status** | **Selected Main Result** |

### B. MobileNetV3 Threshold 0.12 Experiment (Auxiliary Run)

This experiment sweeps thresholding down to `0.12` to increase recall, but degrades other metrics significantly due to excessive false positives.

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.6382 |
| **Macro-F1** | 0.5214 |
| **Defect Precision** | 0.1713 |
| **Defect Recall** | 0.8474 |
| **Defect F1** | 0.2849 |
| **Threshold Used** | 0.12 |
| **Status** | **Not Selected** |

---

## Comparison and Conclusion

1. **Precision-Recall Trade-off:** 
   Adjusting the decision threshold to `0.12` successfully improved the **Defect Recall** from `0.74` to `0.8474`. However, it led to a significant increase in false positives (predicting "no-defect" samples as "defect"), lowering **Defect Precision** to `0.1713` and overall **Accuracy** to `0.6382`.
2. **Final Model Choice:** 
   The **MobileNetV3 Baseline** remains the superior model choice for reporting, showing significantly better **Macro-F1 (0.6290 vs 0.5214)**, **Defect F1 (0.38 vs 0.2849)**, and **Accuracy (0.7947 vs 0.6382)**.
3. **Recommendation:** 
   The threshold 0.12 run should be treated purely as an auxiliary experiment illustrating model sensitivity, rather than the primary model for deployment.

---

## Important Evaluation Note

- The path `labeled/test` was used only as a local Windows junction alias to the official physical split directory `data/image_dataset_split/test`.
- **No new Test set was created** and no data leakage occurred (the Test set was strictly isolated from training/tuning).
