# Text Model Documentation

## Cập nhật - Code Refactoring (Tháng 5/2026)

Mã nguồn cho pipeline xử lý ngôn ngữ tự nhiên (`text_baseline.py` và `train_text_baseline.py`) đã được refactor để tuân thủ các chuẩn mực kỹ thuật dành cho kĩ sư ML:

1. **Architecture & OOP**: Mã được bọc trong các Class với trách nhiệm duy nhất (SOLID). Tránh sử dụng global context.
2. **Readability & Typing**: Bổ sung đầy đủ type hinting (với `typing` module) cho toàn bộ arguments và return types. Các biến số được phân định rõ ràng.
3. **Documentation**: Thêm documentation chuẩn Google (Google-style docstrings) cho tất cả các Classes và Methods, mô tả chi tiết logic bên trong.
4. **Robustness**: Thay thế hoàn toàn hàm `print()` mặc định bằng `logging` để dễ dàng quản lý log và theo dõi (tracking) trong môi trường production. Bổ sung Error Handling với khối lượng code ngoại lệ (ex: `FileNotFoundError`).
5. **ML Best Practices**: 
   - Sử dụng `imblearn.pipeline.Pipeline` thay cho `sklearn.pipeline.Pipeline`.
   - Cố định `random_state` xuyên suốt pipeline để tạo tính lặp lại (reproducibility).
   - Đặt chiến lược ưu tiên **Cost-Sensitive Learning** (`class_weight='balanced'`) thay vì lạm dụng SMOTE ở dạng cấu hình mặc định nhằm xử lý vấn đề imbalanced classes một cách hiệu quả hơn mà không can thiệp vào phân bố tự nhiên trước khi chứng minh được sự cần thiết của SMOTE. (SMOTE trở thành một attribute dưới dạng cờ `--use_smote=True/False`).

### Files Changed:
- `ai_engine/models/text_baseline.py`
- `ai_engine/scripts/train_text_baseline.py`

## Cập nhật - Performance & Comparative Training (Tháng 5/2026)

Mã nguồn được thay đổi để tăng tốc độ huấn luyện cũng như đánh giá A/B Testing giữa Cost-sensitive Learning và Oversampling (SMOTE):

1. **Tối ưu tốc độ huấn luyện**: 
   - `LinearSVC`: Thêm tham số `dual="auto"` giúp tự động chuyển đổi logic tối ưu cho Sparse Matrix (TF-IDF matrix). Tăng độ hội tụ.
   - `LogisticRegression`: Kích hoạt multi-threading (`n_jobs=-1`) để chạy song song. Đi kèm với việc thay đổi sang thuật giải `solver='lbfgs'` đáp ứng tương thích cho n_jobs.
2. **Automated Multi-Model Evaluation**:
   - `train_text_baseline.py` được cải tiến với vòng lặp cho một mảng các `configurations`, huấn luyện liên tục 3 biến thể: "Logistic Regression /w Cost-sensitive", "Linear SVM /w Cost-sensitive", và "Logistic Regression /w SMOTE".
   - Tự động hóa đánh giá và sinh artifact cho 3 models vào thư mục mới: `artifacts/models`.