# Text Baseline Model — Architecture & Reference Guide

> **Current version**: Weighted Soft-Voting Ensemble (May 2026)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Reference](#2-component-reference)
3. [Automatic Weight Algorithm](#3-automatic-weight-algorithm)
4. [Benchmark Experiments](#4-benchmark-experiments)
5. [Usage](#5-usage)
6. [Changelog](#6-changelog)

---

## 1. Architecture Overview

```
Raw Text Input
      │
      ▼
┌─────────────────────────────┐
│   TF-IDF Vectorizer         │  max_features=15 000, ngram=(1,2), sublinear_tf=True
│   (Sparse Feature Matrix)   │
└──────────────┬──────────────┘
               │
        use_smote?
       ┌───────┴───────┐
      YES              NO
       │               │
       ▼               │
┌─────────────┐        │
│    SMOTE    │        │   Over-samples minority classes in TF-IDF space
└──────┬──────┘        │
       └───────────────┤
                       ▼
       ┌───────────────────────────────────────────────────┐
       │           Soft-Voting Ensemble                    │
       │                                                   │
       │  ┌────────────────┐  w₁ (auto / manual)          │
       │  │ Logistic Reg.  │ ──────────────────────────┐  │
       │  │ (LR)           │                           │  │
       │  └────────────────┘                           │  │
       │                                               ▼  │
       │  ┌────────────────┐  w₂  ┌──────────────────────┐│
       │  │ LinearSVC      │ ────▶│  Weighted Average of  ││
       │  │ + Calibration  │      │  Predicted Proba       ││ ── argmax ──▶ Label
       │  └────────────────┘  w₃  └──────────────────────┘│
       │                                               ▲  │
       │  ┌────────────────┐                           │  │
       │  │ Random Forest  │ ──────────────────────────┘  │
       │  │ (RF)           │                               │
       │  └────────────────┘                               │
       └───────────────────────────────────────────────────┘
```

---

## 2. Component Reference

### 2.1 TF-IDF Vectorizer

| Parameter       | Value        | Rationale                                               |
|-----------------|--------------|---------------------------------------------------------|
| `max_features`  | `15 000`     | Wider vocabulary improves RF diversity                  |
| `ngram_range`   | `(1, 2)`     | Captures phrase-level sentiment cues                    |
| `sublinear_tf`  | `True`       | `log(1+tf)` — reduces dominance of high-frequency terms |
| `min_df`        | `3`          | Prunes hapax legomena                                   |
| `max_df`        | `0.85`       | Removes corpus-wide stop words                          |

### 2.2 Base Estimators

| Estimator               | Class                     | Key Params                                     |
|-------------------------|---------------------------|------------------------------------------------|
| Logistic Regression     | `LogisticRegression`      | `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=1000` |
| Calibrated LinearSVC    | `CalibratedClassifierCV(LinearSVC(...), cv=3)` | `class_weight='balanced'`, `dual='auto'` — calibration adds `predict_proba` |
| Random Forest           | `RandomForestClassifier`  | `n_estimators=200`, `class_weight='balanced'`  |

> **Why `CalibratedClassifierCV` for SVM?**  
> `LinearSVC` is 10–100× faster than `SVC(kernel='linear')` on sparse TF-IDF matrices
> but does not natively expose `predict_proba`. `CalibratedClassifierCV` wraps it with
> isotonic regression calibration, adding the probability outputs required for soft voting.

### 2.3 SMOTE (Optional)

- Applies **Synthetic Minority Over-sampling Technique** to the TF-IDF output matrix.
- Positioned *after* TF-IDF inside the `imblearn.pipeline.Pipeline`, ensuring it is
  only applied to training folds (no data leakage).
- Default strategy: all minority classes are up-sampled to match the majority class count.

---

## 3. Automatic Weight Algorithm

When `weights=None` (default), `TextEnsembleModel.fit()` invokes
`compute_auto_weights()` before training the full ensemble.

### Algorithm Steps

```
for each estimator E in [LR, Calibrated SVM, RF]:
    X_tfidf ← TF-IDF.fit_transform(X_train)
    scores  ← cross_val_score(E, X_tfidf, y_train, cv=5, scoring='f1_macro')
    w_E     ← mean(scores)

weights ← [w_LR, w_SVM, w_RF]
VotingClassifier(weights=weights)
```

### Rationale

| Property                | Effect                                                  |
|-------------------------|---------------------------------------------------------|
| F1-macro as weight      | Naturally emphasises recall for minority classes        |
| Proportional (not fixed)| The best-performing model receives a larger vote share  |
| CV before full fit      | Weights reflect generalisation, not just training fit   |

---

## 4. Benchmark Experiments

The training script (`train_text_baseline.py`) runs four experiments:

| Exp ID | SMOTE | Weights       | Purpose                              |
|--------|-------|---------------|--------------------------------------|
| EXP-1  | No    | Auto (F1-CV)  | **Primary** — balanced cost-sensitive |
| EXP-2  | Yes   | Auto (F1-CV)  | **Primary** — SMOTE + smart weights  |
| EXP-3  | No    | Equal [1,1,1] | Control — no weighting               |
| EXP-4  | Yes   | Equal [1,1,1] | Control — SMOTE without weighting    |

Artifacts are saved to `artifacts/models/`:

```
artifacts/models/
├── ensemble_no_smote_auto_weights.pkl   ← EXP-1
├── ensemble_smote_auto_weights.pkl      ← EXP-2
├── ensemble_no_smote_equal_weights.pkl  ← EXP-3
└── ensemble_smote_equal_weights.pkl     ← EXP-4
```

---

## 5. Usage

### Training

```bash
# From project root
python ai_engine/scripts/train_text_baseline.py
```

### Python API

```python
from ai_engine.models.text_baseline import TextEnsembleModel

# --- Training ---
model = TextEnsembleModel(use_smote=False)   # weights=None → auto-computed
model.fit(X_train, y_train)
model.save("artifacts/models/my_ensemble.pkl")

# --- Inference ---
loaded = TextEnsembleModel.load("artifacts/models/my_ensemble.pkl")
labels = loaded.predict(X_test)           # array of class labels
probas = loaded.predict_proba(X_test)     # shape (n_samples, n_classes)

# --- Manual weights (override auto-computation) ---
model_manual = TextEnsembleModel(use_smote=True, weights=[0.85, 0.80, 0.78])
model_manual.fit(X_train, y_train)
```

---

## 6. Changelog

### May 2026 — v3.0: Weighted Soft-Voting Ensemble

- **New class**: `TextEnsembleModel` replaces `TextBaselineModel`.
- **New**: Automatic F1-proportional weight computation (`compute_auto_weights()`).
- **New**: `CalibratedClassifierCV(LinearSVC)` for probability-calibrated SVM.
- **New**: Four-experiment SMOTE vs. No-SMOTE benchmark in training script.
- **New**: Formatted comparison table printed at end of training run.
- **Improved**: `save()` / `load()` serialise the full `TextEnsembleModel` instance
  (weights + pipeline), not just the raw pipeline.
- **Improved**: `sublinear_tf=True` and `max_features=15 000` in TF-IDF.
- **Docs**: This file rewritten in English with architecture diagram and algorithm table.

### May 2026 — v2.0: Performance & Comparative Training

- `LinearSVC` upgraded with `dual="auto"` for sparse-matrix optimisation.
- `LogisticRegression` switched to multi-threaded `n_jobs=-1` with `lbfgs` solver.
- Automated multi-model evaluation loop across 3 configurations.

### May 2026 — v1.0: OOP Refactor & Code Standards

- Initial `TextBaselineModel` class with type hints and Google-style docstrings.
- `imblearn.pipeline.Pipeline` adopted; SMOTE made optional via flag.
- `logging` replaces `print()` throughout; `random_state` enforced for reproducibility.