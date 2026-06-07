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

---

## 7. Quy trình Chia Dữ liệu (Data Splitting Strategy)

### 7.1 Tỷ lệ chia dữ liệu (Train / Validation / Test)
Dữ liệu phân tích cảm xúc văn bản được chia theo tỷ lệ **70 / 15 / 15**:
- **Train (Tập huấn luyện)**: Chiếm 70% tổng số mẫu.
- **Validation (Tập kiểm định)**: Chiếm 15% tổng số mẫu.
- **Test (Tập thử nghiệm)**: Chiếm 15% tổng số mẫu.

### 7.2 Chi tiết thực hiện & Rationale (Lý do lựa chọn)
Trong file [split_text_data.py](file:///d:/3rdY_HCMUS/Machine_Learning/PROJECT_LT/ecommerce-review-analytics/scripts/split_text_data.py), quy trình chia dữ liệu được thực hiện như sau:
1. Đầu tiên, dữ liệu được tách thành tập Train (70%) và tập tạm thời Temp (30%) sử dụng hàm `train_test_split` với tham số `stratify` dựa trên nhãn lớp `sentiment_label`.
2. Tiếp theo, tập Temp (30%) tiếp tục được tách đôi thành tập Validation (15%) và tập Test (15%), cũng áp dụng cơ chế `stratify`.

**Lý do lựa chọn tỷ lệ chia và phương thức này:**
- **Kiểm soát tính mất cân bằng nhãn (Class Imbalance)**: Dữ liệu review thương mại điện tử thường có xu hướng cực kỳ mất cân bằng (lớp tích cực chiếm tỷ lệ vượt trội). Việc sử dụng **Phân tầng (Stratified Split)** đảm bảo tỷ lệ phân phối nhãn giữa 3 tập Train, Validation và Test luôn đồng nhất với tập dữ liệu gốc, ngăn ngừa tình trạng một lớp thiểu số biến mất hoặc bị thiếu hụt nghiêm trọng ở một trong các tập.
- **Lượng dữ liệu tối ưu cho huấn luyện**: Tỷ lệ 70% dành cho huấn luyện cung cấp đủ số lượng mẫu để bộ TF-IDF Vectorizer xây dựng được từ điển đặc trưng phong phú (cấu hình tối đa lên tới 15,000 features) và giúp các mô hình học máy (tuyến tính lẫn phi tuyến) học được ranh giới phân loại rõ ràng.
- **Kiểm định và đánh giá độc lập (Unbiased Evaluation)**: Việc tách riêng biệt tập Validation (15%) dùng để tối ưu hóa siêu tham số (Hyperparameter Tuning) và tập Test (15%) dùng để kiểm tra hiệu năng cuối cùng giúp tránh tình trạng rò rỉ thông tin (data leakage) hoặc ước lượng quá lạc quan (overoptimistic performance estimation) về mô hình.

---

## 8. Chi tiết Kỹ thuật Các Mô hình (Technical Details of Base Models)

Nhằm tối ưu hóa hiệu năng và tăng tính đa dạng cho mô hình Soft-Voting Ensemble, dự án sử dụng **3 mô hình** nền tảng có bản chất thuật toán và giả định phân phối khác nhau:

### 8.1 Logistic Regression (Hồi quy Logistic)
- **Lý do lựa chọn**: Là mô hình tuyến tính phân loại cổ điển, tốc độ xử lý nhanh, khả năng diễn giải (interpretability) cao. Nó đặc biệt hiệu quả trên các không gian đặc trưng thưa, số chiều lớn (sparse high-dimensional) được tạo ra từ TF-IDF.
- **Kiến trúc chi tiết**:
  - **Sơ đồ kiến trúc**:
    ```
    Input (15,000 TF-IDF Features) ──► [ Linear Function: z = W^T * X + b ] ──► [ Softmax ] ──► Probabilities (3 Classes)
    ```
  - **Số lượng tham số**:
    $$\text{Tham số} = (15,000 \text{ đặc trưng} \times 3 \text{ lớp}) + 3 \text{ biases} = 45,003 \text{ tham số}.$$
  - **Hàm kích hoạt**: Hàm **Softmax** được áp dụng ở đầu ra để chuẩn hóa các giá trị logits thành phân bố xác suất hợp lệ (tổng bằng 1.0).

### 8.2 Linear Support Vector Machine (LinearSVC) kết hợp Hiệu chuẩn (CalibratedClassifierCV)
- **Lý do lựa chọn**: SVM tìm kiếm siêu phẳng có lề phân tách lớn nhất (maximum-margin hyperplane). `LinearSVC` chạy nhanh hơn từ 10 đến 100 lần so với mô hình sử dụng kernel tuyến tính thông thường (`SVC(kernel='linear')`) trên ma trận thưa TF-IDF.
- **Kiến trúc chi tiết**:
  - **Sơ đồ kiến trúc**:
    ```
    Input (15,000 TF-IDF Features) ──► [ LinearSVC: decision_function ] ──► [ CalibratedClassifierCV (Isotonic) ] ──► Probabilities
    ```
  - **Số lượng tham số**:
    - Mô hình SVM cơ sở: $15,000 \times 3 + 3 = 45,003$ tham số.
    - Bộ hiệu chuẩn: Sử dụng phương pháp hồi quy đơn điệu (Isotonic Regression) trên kết quả kiểm định chéo để map khoảng cách lề (decision values) thành xác suất. Số lượng tham số của bộ hiệu chuẩn phụ thuộc vào số điểm phân đoạn đơn điệu được tối ưu từ tập dữ liệu.
  - **Hàm kích hoạt**: Base LinearSVC không sử dụng hàm kích hoạt. Bước hiệu chuẩn xác suất sử dụng hàm cắt mảnh đơn điệu (Isotonic step function) để chuyển kết quả về khoảng $[0, 1]$.

### 8.3 Random Forest Classifier (Phân loại Rừng Ngẫu nhiên)
- **Lý do lựa chọn**: Là mô hình phi tuyến tính dạng Bagging dựa trên tập hợp nhiều cây quyết định độc lập. Random Forest có khả năng mô hình hóa các tương tác phi tuyến phức tạp giữa các cụm từ (n-grams) và mang lại sự đa dạng về thuật toán cho Ensemble so với hai mô hình tuyến tính trên.
- **Kiến trúc chi tiết**:
  - **Sơ đồ kiến trúc**:
    ```
    Input (15,000 TF-IDF Features)
             │
      ┌──────┼──────┐ (Bootstrap Samples & Random Feature Selection)
      ▼      ▼      ▼
    [Tree1][Tree2][Tree200]
      │      │      │ (Recursive Binary Splitting)
      ▼      ▼      ▼
    [Proba][Proba][Proba] ──► [ Average Probability ] ──► Output Probabilities
    ```
  - **Số lượng tham số**: Do các cây quyết định được huấn luyện không giới hạn độ sâu tối đa (`max_depth=None`), số lượng tham số thực tế phụ thuộc hoàn toàn vào cấu trúc cây khi fit (số nút phân nhánh và nút lá). Số lượng tham số (bao gồm chỉ số đặc trưng phân tách và ngưỡng phân tách tại mỗi nút quyết định) có thể lên tới hàng triệu.
  - **Hàm kích hoạt**: Không sử dụng hàm kích hoạt (sử dụng các logic rẽ nhánh có điều kiện dựa trên đặc trưng).

---

## 9. Cấu hình Huấn luyện & Tinh chỉnh Siêu tham số (Training & Hyperparameter Tuning)

### 9.1 Cấu hình Huấn luyện Tổng quát
- **Hàm mất mát (Loss Function)**:
  - **Logistic Regression**: Tối ưu hóa hàm **Multinomial Cross-Entropy Loss** (entropy chéo đa lớp).
  - **LinearSVC**: Tối ưu hóa hàm **Squared Hinge Loss** giúp phạt nặng hơn các điểm dữ liệu vi phạm khoảng cách lề.
  - **Random Forest**: Sử dụng chỉ số **Gini Impurity** (Độ vẩn đục Gini) làm tiêu chí đánh giá chất lượng phân nhánh cây quyết định.
- **Cơ chế xử lý mất cân bằng nhãn**:
  - Áp dụng **Cost-Sensitive Learning** bằng cách thiết lập tham số `class_weight='balanced'` trên cả ba mô hình nền tảng. Trọng số của mỗi lớp được tính toán tự động tỷ lệ nghịch với tần suất xuất hiện của lớp đó trong tập dữ liệu:
    $$w_c = \frac{N_{\text{mẫu}}}{N_{\text{lớp}} \times N_c}.$$
    Giúp điều chỉnh hàm mất mát để phạt nặng hơn khi mô hình dự đoán sai các mẫu thuộc lớp thiểu số.
  - Áp dụng **SMOTE** (Synthetic Minority Over-sampling Technique) để sinh thêm mẫu ảo cho các lớp thiểu số trong không gian đặc trưng TF-IDF (chỉ áp dụng trên các fold huấn luyện của pipeline để tránh rò rỉ thông tin).
- **Thuật toán tối ưu (Optimizer)**:
  - **Logistic Regression**: Sử dụng bộ giải thuật **L-BFGS** (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) - một thuật toán tối ưu thuộc nhóm Quasi-Newton hội tụ rất nhanh đối với các bài toán lồi. Tốc độ học (Learning rate) được điều chỉnh động bởi thuật toán tìm kiếm đường đi (line search) đi kèm solver.
  - **LinearSVC**: Sử dụng thuật toán **Coordinate Descent** (Xuống thang tọa độ) giải quyết bài toán đối ngẫu tối ưu hóa SVM một cách nhanh chóng.
  - **Ensemble Integration**: Tích hợp các bộ phân loại theo cơ chế **Soft Voting** với trọng số biểu quyết tỷ lệ thuận với điểm F1-macro của từng mô hình qua 5-fold cross-validation trên tập huấn luyện.

### 9.2 Phương pháp Tinh chỉnh Siêu tham số (Hyperparameter Tuning Method)
- **Phương pháp thực hiện**: Sử dụng **GridSearchCV** thực hiện tìm kiếm dạng lưới vét cạn trên các khoảng giá trị siêu tham số định sẵn.
- **Kiểm định chéo (Cross-Validation)**: Sử dụng **5-fold Stratified Cross-Validation** (`StratifiedKFold(n_splits=5)`) giúp đánh giá khách quan độ tổng quát hóa của mô hình và bảo toàn tỷ lệ phân bố nhãn.
- **Chỉ số tối ưu mục tiêu**: `f1_macro` (Macro F1-Score) được lựa chọn để điều phối quá trình tìm kiếm, đảm bảo mô hình tối ưu hóa đồng đều hiệu năng trên cả 3 lớp cảm xúc (tích cực, tiêu cực, trung lập).
- **Danh sách các tham số chính được tinh chỉnh trong Grid Search**:
  - **TF-IDF**: `max_features` (`[10,000, 20,000]`), `ngram_range` (`[(1, 1), (1, 2)]`).
  - **Logistic Regression**: Tham số điều hòa $C$ (`[0.1, 0.5, 1.0, 5.0, 10.0]`), solver (`['lbfgs', 'saga']`), penalty (`['l1', 'l2']`).
  - **LinearSVC**: Lực lượng điều hòa $C$ (`[0.01, 0.1, 0.5, 1.0, 5.0, 10.0]`), `max_iter` (`[2000, 5000]`).
  - **Random Forest**: Số lượng cây `n_estimators` (`[100, 200, 300]`), độ sâu tối đa `max_depth` (`[None, 20, 40]`), mẫu tối thiểu ở nút lá `min_samples_leaf` (`[1, 2, 5]`), số đặc trưng tối đa `max_features` (`['sqrt', 'log2']`).
  - **SMOTE**: Số láng giềng gần nhất `k_neighbors` (`[3, 5]`).