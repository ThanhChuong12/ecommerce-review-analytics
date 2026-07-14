# Kiến trúc Hệ thống Phát hiện Rác (Spam Detection Pipeline)

Tài liệu này mô tả kiến trúc mới nhất của hệ thống phát hiện bình luận rác (Spam Reviews) trong dự án, chuyển đổi từ cơ chế phụ thuộc Luật sang **Trí Tuệ Nhân Tạo (Isolation Forest)** hoàn toàn độc lập.

## 1. Phương pháp Gán Nhãn & Xác Thực Ground Truth

Bài toán phát hiện rác đối mặt với thách thức thiếu hụt dữ liệu có nhãn sẵn. Để giải quyết triệt để, dự án áp dụng chiến lược gán nhãn chặt chẽ:

- **Khởi tạo nhãn bằng Bộ Luật (Rule-based):** Sử dụng các tập hợp quy tắc (như phát hiện lặp từ, template, icon, cấu trúc...) để quét qua toàn bộ 25.000 dòng dữ liệu và gán nhãn thô ban đầu.
- **Xác thực bằng Yếu tố Con người (Human Verification):** Nhãn do Bộ Luật tạo ra KHÔNG được sử dụng trực tiếp làm đáp án thi. Đội ngũ đã tiến hành trích xuất ngẫu nhiên các mẫu bình luận và **xác thực thủ công (kiểm duyệt bằng mắt người)**. Chỉ khi con người xác nhận Bộ Luật đã dán nhãn chuẩn xác, các nhãn này mới chính thức được công nhận là **Ground Truth** (Đáp án chuẩn).
- **Phân tách Dữ liệu (Train/Val/Test):** Dữ liệu được chia theo tỷ lệ 70-15-15.
  - **Tập Train (70%):** Bị xóa bỏ hoàn toàn nhãn Ground Truth. Mô hình bắt buộc phải học theo cơ chế Không Giám Sát (Unsupervised Learning) để tự tìm ra các cụm rác.
  - **Tập Val & Test (30%):** Sử dụng nhãn Ground Truth (đã qua con người xác thực) làm thang đo khắt khe để đánh giá và tinh chỉnh AI.

## 2. Trích Xuất Đặc Trưng (Feature Engineering)

Văn bản thô được số hóa thành một không gian vector 27 chiều thông qua:
- **5 Đặc trưng Cấu trúc:** Tỷ lệ in hoa, tỷ lệ số, tỷ lệ ký tự đặc biệt, tỷ lệ biểu tượng cảm xúc (emoji), và độ phong phú từ vựng (Type-Token Ratio).
- **22 Đặc trưng Hành vi (Dựa trên Luật):** Bộ Luật lúc này lùi về hậu phương. Thay vì ra quyết định, nó đóng vai trò là "máy quét" để tạo ra các cờ hiệu (flags) như `có_chứa_link`, `cày_xu`, `lặp_template`... Các cờ hiệu này trở thành các Input phong phú cho AI.

## 3. Cơ Chế Dự Đoán (Inference Pipeline)

> [!IMPORTANT]  
> **Chỉ sử dụng Mô hình AI để dự đoán (Pure AI).** Trong giai đoạn Inference (Dự đoán thực tế), thuật toán **Isolation Forest** là thực thể duy nhất đưa ra quyết định. 

- Cơ chế "Lai tạp" (Hybrid) trước đây — nơi Bộ Luật có quyền ghi đè quyết định của AI — đã bị loại bỏ hoàn toàn.
- Mô hình Isolation Forest (với `n_estimators=200`, `contamination=0.1`) phân tích 27 đặc trưng của bình luận và tự tính toán điểm bất thường (Anomaly Score). Dựa vào ranh giới cách ly, AI sẽ hoàn toàn tự chủ phán quyết đó có phải là Rác (Spam) hay không.

## 4. Hiệu Suất Thực Nghiệm

Các báo cáo đánh giá (Learning Curve, Confusion Matrix và Classification Report) chứng minh:
- AI tự chủ đạt mức Macro F1-Score **~0.88** và độ chính xác (Accuracy) **~95.7%** trên tập Test. 
- Mức độ tương đồng cao so với Ground Truth (đã xác thực) khẳng định AI có đủ năng lực phân tách rác tinh vi mà không cần "dựa dẫm" vào các phán quyết cứng nhắc của Bộ Luật.
