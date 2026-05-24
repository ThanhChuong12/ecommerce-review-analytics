# Hướng dẫn sử dụng: `tune_text_hyperparams.py`

## Tổng quan
Script `tune_text_hyperparams.py` thực hiện tinh chỉnh siêu tham số toàn diện cho các mô hình phân loại cảm xúc văn bản (Text Sentiment Analysis) đóng vai trò Baseline, bao gồm: `Logistic Regression`, `LinearSVC (Calibrated)`, và `RandomForest`.

Điểm nổi bật của script này là việc tích hợp **SMOTE (Synthetic Minority Over-sampling Technique)** thông qua `ImbPipeline` của thư viện `imbalanced-learn`. Điều này cho phép SMOTE tự động sinh mẫu cho các class thiểu số một cách an toàn trong lúc thực hiện Cross-Validation (CV) mà không gây rò rỉ dữ liệu (data leakage) sang tập validation.

## Chi tiết Grid Search

Hệ thống đánh giá song song 6 nhóm mô hình:
- 3 Mô hình không dùng SMOTE (Baseline tiêu chuẩn kết hợp Class Weights)
- 3 Mô hình dùng SMOTE (Thêm SMOTE vào Pipeline trước bộ phân loại)

Các tham số được quét bao gồm:
- **TF-IDF Vectorizer**: `max_features` (10k, 20k), `ngram_range` ((1,1), (1,2)).
- **SMOTE**: `k_neighbors` (3, 5).
- **Logistic Regression**: `C`, `solver` (lbfgs, saga), `penalty`.
- **LinearSVC**: `C`, `max_iter`.
- **RandomForest**: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.

Tiêu chí đánh giá chính là **Macro-F1 Score** do tập dữ liệu bị lệch rất nghiêm trọng (Imbalanced: 94% Pos, 5% Neg, 1% Neu).

## Voting Ensemble
Sau khi chạy xong GridSearchCV cho toàn bộ mô hình, script sẽ tự động trích xuất các biến thể tốt nhất (Best Variant) cho từng thuật toán cơ sở (ví dụ: nó sẽ tự quyết định xem nên dùng LR thường hay LR có SMOTE dựa trên Test F1 Score). Sau đó, nó tự động khởi tạo và huấn luyện một **Tuned Voting Ensemble** sử dụng trọng số đánh giá Soft-Voting được cung cấp bởi điểm Macro F1 của các thuật toán Base.

## Usage (Cách chạy)
Chạy từ thư mục gốc của project:

```bash
# Chạy đầy đủ lưới tham số với 5-fold CV
py scripts/tune_text_hyperparams.py

# Chạy nhanh cho mục đích test/dev với 3-fold CV
py scripts/tune_text_hyperparams.py --quick --cv 3
```

## Kết quả đầu ra (Outputs)
1. **JSON Report**: `artifacts/metrics/text_hyperparameter_tuning_results.json`
2. **Model Files**: Các file mô hình `.pkl` của từng thuật toán được tune tốt nhất, lưu tại `artifacts/models/tuned/` (ví dụ: `tuned_logisticregression_smote.pkl`, `tuned_voting_ensemble.pkl`).
3. **Console Report**: Bảng xếp hạng Macro-F1, Weighted-F1 và báo cáo phân loại (Classification Report) chi tiết cho từng cấu hình trên tập hold-out test.
