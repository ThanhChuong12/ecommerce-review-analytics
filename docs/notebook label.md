# Tổng kết Notebook 01: Data Preprocessing and Labeling

Phân hệ này thiết lập nền tảng dữ liệu cho hệ thống phân tích cảm xúc đa phương thức. Các tác vụ chính đã thực hiện bao gồm:

## 1. Khởi tạo môi trường và nạp dữ liệu
- Cấu hình thư viện phân tích toán học, học máy (Pandas, Scikit-learn, Underthesea, Transformers).
- Khởi tạo công cụ xử lý song song (Pandarallel). Nạp tập dữ liệu thô all_reviews.csv.

## 2. Kiểm định chất lượng (EDA) và Làm sạch
- Tính toán mô tả thống kê và tỷ lệ khuyết thiếu dữ liệu.
- Loại bỏ các đánh giá trùng lặp thông qua thuật toán băm (MD5).
- Loại bỏ các bộ dữ liệu nhiễu: rỗng (NaN) hoặc văn bản quá ngắn (chiều dài <= 2).
- Đồng nhất chuẩn dữ liệu toán học (DateTime, Rating 1-5).

## 3. Chuẩn hóa văn bản (Text Normalization)
- Triển khai text normalization song đa luồng (loại tag web, đưa về chữ thường, filter noise).
- Đối chiếu hiệu quả giảm thiểu kích thước không gian từ vựng (Vocabulary Reduction) trước và sau tinh chỉnh.
- Tối ưu xử lý emoji theo miền thương mại điện tử: ánh xạ trực tiếp emoji tần suất cao sang token tiếng Việt có ý nghĩa cảm xúc (ví dụ: `⭐️ -> tuyệt_vời`, `❤ -> yêu_thích`, `👍 -> tốt`) thay vì dịch sang chuỗi tiếng Anh dài.
- Loại bỏ hoàn toàn emoji ngoài từ điển ánh xạ để hạn chế phân mảnh subword và kiểm soát độ dài chuỗi đầu vào cho mô hình Transformer.
- Bảo toàn từ vựng đặc thù e-commerce như `shop`, `ok`, `okie`, `sp`, `ship`, `shipper` để giữ sắc thái ngôn ngữ thực tế của người dùng.

## 4. Phân tích & Thực thi Tokenize (Word Segmentation)
- So sánh định lượng 3 chiến lược: Đơn âm tiết (Syllable), Word-level (underthesea) và Subword-level (PhoBERT từ HuggingFace).
- Trích xuất ra cả hai biểu diễn: Word-level (phục vụ mô hình TF-IDF hoặc Naive Bayes) và Subword-level (phục vụ kiến trúc Transformer/Deep Learning).

## 5. Phân tích Từ dừng (Stopword Evaluation)
- Thử nghiệm lọc Stopword và đánh giá hiệu năng duy trì tín hiệu ngữ nghĩa qua hàm Mutual Information và mô hình Multinomial Naive Bayes.
- **Chiến lược được chọn:** Không loại bỏ từ dừng. Lý do: Giúp duy trì chặt chẽ cấu trúc phủ định (không, chưa) và chuỗi ngữ cảnh tuần tự thiết yếu cho mô hình Deep Learning.

## 6. Gán nhãn Tự động dạng Giám sát yếu (Weak Supervision Labeling)
- Xây dựng mạng lưới Heuristics siêu tốc áp dụng luật (Từ điển tiêu cực/tích cực) và tính chất thao túng của mức Rating (lực hấp dẫn từ số sao).
- Nâng cấp luật Heuristic theo hướng nhận biết phạm vi phủ định (negation-aware) bằng N-gram/Regex cho các cụm như `không tốt`, `chưa hài lòng`, `không thích` để tránh dương tính giả/âm tính giả.
- Áp dụng cơ chế phát hiện mâu thuẫn mạnh giữa rating và ngữ nghĩa bề mặt: các trường hợp xung đột lớn sẽ không gán nhãn cứng mà chuyển trực tiếp sang `LLM Fallback` với nhãn `ambiguous`.
- Phát hiện các mẫu mâu thuẫn (Ambiguous) và điều hướng phân loại sâu thông qua Zero-Shot (XLM-R) + LLM API Fallback.
- Báo cáo Metadata thống kê (labeling_metadata.json) chuẩn MLOps và xuất file processed_labeled_reviews.csv.
