# ResNet50 Split Test Evaluation Results (Improve)

## Model Information
* **Best Epoch:** 9
* **Selected Threshold from Validation:** 0.550

## Final Test Metrics
| Metric | Value |
|---|---|
| Test Accuracy | 0.883157 |
| Test Precision (no-defect) | 0.948196 |
| Test Recall (no-defect) | 0.922704 |
| Test F1-Score (no-defect) | 0.935276 |
| Test Precision (defect) | 0.355140 |
| Test Recall (defect) | 0.457831 |
| Test F1-Score (defect) | 0.400000 |
| Macro Precision | 0.651668 |
| Macro Recall | 0.690267 |
| Macro F1-Score | 0.667638 |
| Weighted F1-Score | 0.889740 |
| ROC-AUC | 0.827362 |

## Confusion Matrix (Test Set)
* **True Negative (TN):** 2471 (no-defect correctly classified)
* **False Positive (FP):** 207 (no-defect misclassified as defect)
* **False Negative (FN):** 135 (defect misclassified as no-defect)
* **True Positive (TP):** 114 (defect correctly classified)

| Actual \\ Predicted | Predicted: no-defect | Predicted: defect |
|---|---|---|
| **Actual: no-defect** | 2471 | 207 |
| **Actual: defect** | 135 | 114 |

## Quality Gate Status
* **Defect Recall (Target >= 0.80):** FAIL ❌ (0.4578)
* **Defect F1 (Target >= 0.85):** FAIL ❌ (0.4000)
* **Macro F1 (Target >= 0.85):** FAIL ❌ (0.6676)
