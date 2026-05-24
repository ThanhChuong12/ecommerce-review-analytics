# Hướng dẫn sử dụng: `compare_text_models.py`

## Tổng quan
`compare_text_models.py` là một khung đánh giá (Evaluation Framework) chuẩn hóa được dùng để so sánh hiệu năng thực tế giữa:
1. **Mô hình Baseline (Sklearn)**: Thường là Soft-Voting Ensemble (Logistic Regression, SVM, Random Forest) kết hợp TF-IDF.
2. **Mô hình Deep Learning (PhoBERT)**: Dựa trên kiến trúc `vinai/phobert-base-v2` đã được fine-tune cho bài toán phân loại cảm xúc.

Script này đảm bảo cả hai hệ thống (với luồng vector hóa, tokenization hoàn toàn khác nhau) được đánh giá một cách công bằng trên **cùng một tập hold-out test set** đã được chia phân tầng (stratified split).

## Các Metrics Đánh Giá
Hệ thống không chỉ đo lường độ chính xác mà còn đo lường tính khả thi khi triển khai thực tế:
- **Macro-F1 & Weighted-F1**: Đánh giá trên tập dữ liệu mất cân bằng.
- **Accuracy**: Độ chính xác tổng thể.
- **ROC-AUC (One-vs-Rest)**: Đánh giá khả năng phân tách ranh giới của xác suất phân loại (Predict Proba).
- **Inference Latency**: Tốc độ suy luận (ms/sample). Script tự động đo lường độ trễ trung bình (mean) và độ trễ P95 (percentile 95).
- **Confusion Matrix**: Xuất ra biểu đồ heatmap trực quan để xem mô hình hay bị nhầm lẫn ở lớp nào.

## Usage (Cách chạy)
Chạy script từ thư mục gốc. Bạn có thể cung cấp file baseline, thư mục PhoBERT, hoặc cả hai.

```bash
# So sánh song song cả Baseline và PhoBERT
python scripts/compare_text_models.py \
    --baseline-path artifacts/models/baselines/ensemble_smote_auto_weights.pkl \
    --phobert-path ai_engine/models/weights/phobert_best

# Chỉ đánh giá riêng Baseline (Bỏ qua PhoBERT)
python scripts/compare_text_models.py \
    --baseline-path artifacts/models/tuned/tuned_voting_ensemble.pkl \
    --no-phobert

# Kiểm tra sức khỏe của framework bằng dữ liệu giả (Sanity Check)
python scripts/compare_text_models.py --sanity
```

### Các tham số cấu hình bổ sung
- `--data-path`: Trỏ đến file CSV chuẩn bị sẵn (`processed_labeled_reviews.csv`).
- `--phobert-batch-size`: Batch size khi thực hiện inference với PhoBERT (mặc định: 32).
- `--test-size`: Tỷ lệ hold-out test set (mặc định: 0.20).

## Kết quả đầu ra (Outputs)
1. **JSON Report**: `artifacts/metrics/model_comparison_report.json` - chứa bảng so sánh chi tiết.
2. **Plots**: Các hình ảnh Confusion Matrix định dạng PNG được lưu tại `artifacts/plots/cm_*.png`.
3. **Console Report**: Bảng so sánh in ra Terminal, cung cấp thông tin "Mô hình tốt nhất", "Chênh lệch F1", và "Tỉ lệ chênh lệch tốc độ (Speed Ratio)".
