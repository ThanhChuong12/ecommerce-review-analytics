# Hướng dẫn sử dụng: `tune_spam_model.py`

## Tổng quan
Script `tune_spam_model.py` được thiết kế để tự động hóa quá trình tinh chỉnh siêu tham số (Hyperparameter Tuning) cho **SpamHybridModel** - mô hình kết hợp giữa logic dựa trên luật (Rule-based) và máy học không giám sát (Isolation Forest).

Vì thuật toán Isolation Forest là học không giám sát (unsupervised) và chúng ta không có sẵn nhãn ground-truth cho dữ liệu thực tế, script sử dụng một kỹ thuật đặc biệt: **Dùng nhãn được gán từ Rule-based làm proxy ground-truth** để đánh giá mức độ đồng thuận của Isolation Forest với tập luật.

## Chi tiết các tham số được Tuning

Script quét (grid search) qua tổ hợp của các tham số sau:
1. **Rule-based**:
   - `dup_threshold`: Ngưỡng cosine similarity để nhóm các bình luận trùng lặp/seeding (Ví dụ: `0.75, 0.80, 0.85, 0.90`).
2. **Isolation Forest**:
   - `contamination`: Tỷ lệ mẫu bất thường kỳ vọng trong tập dữ liệu (Ví dụ: `0.05, 0.08, 0.10, 0.12, 0.15, 0.20`).
   - `n_estimators`: Số lượng cây trong rừng (Ví dụ: `100, 150, 200, 300`).
   - `max_samples`: Số lượng mẫu dùng để huấn luyện từng cây (Ví dụ: `"auto", 0.7, 1.0`).

## Cách thức hoạt động
1. Tải dữ liệu từ file csv (`reviews_flagged.csv`).
2. Lặp qua tất cả các cấu hình tham số có thể có.
3. Chạy thuật toán `detect_spam()` của bộ lọc rule-based tương ứng với `dup_threshold`.
4. Huấn luyện `IsolationForest` trên đặc trưng cấu trúc (Structural Features).
5. Kết hợp dự đoán của cả hai và so sánh kết quả của IForest với nhãn Proxy (Rule-based) thông qua các metric: **Precision, Recall, F1-Score**.
6. Chọn ra cấu hình mang lại **IForest F1-Score** cao nhất. Cấu hình này sau đó được train lại trên toàn bộ dữ liệu.

## Usage (Cách chạy)
Chạy script từ thư mục gốc của project:

```bash
# Chạy đầy đủ toàn bộ lưới tham số (Có thể mất thời gian)
py scripts/tune_spam_model.py

# Chạy nhanh với lưới tham số rút gọn (Dùng để dev/debug)
py scripts/tune_spam_model.py --quick

# Tùy chỉnh đường dẫn dữ liệu đầu vào
py scripts/tune_spam_model.py --data-path data/processed/reviews.csv
```

## Kết quả đầu ra (Outputs)
Sau khi chạy thành công, script sẽ sinh ra:
1. **JSON Report**: `artifacts/metrics/spam_tuning_results.json` (chứa toàn bộ chỉ số F1, precision của từng cấu hình, và xếp hạng Top 10).
2. **Model File**: `artifacts/models/tuned/tuned_spam_iforest.pkl` (Mô hình tốt nhất đã được huấn luyện sẵn sàng cho production inference).
3. **Console Report**: Bảng kết quả in trực tiếp trên terminal với tỷ lệ spam bị chặn được phân tích chi tiết.
