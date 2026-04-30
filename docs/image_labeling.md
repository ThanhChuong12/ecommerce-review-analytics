# Gán nhãn ảnh tự động

## Bối cảnh

Dữ liệu ảnh được cào từ review trên sàn thương mại điện tử (Shopee, Lazada). Mỗi review có thể kèm nhiều ảnh và video. Mục tiêu là gán nhãn cho từng ảnh theo 4 lớp phục vụ cho việc huấn luyện mô hình phân loại ảnh sau này.

Vì số lượng ảnh lớn (hàng nghìn), việc gán nhãn thủ công là không khả thi. Nhóm chọn hướng dùng **vision model** để auto-label, kết hợp với một bộ quy tắc prompt được tinh chỉnh qua nhiều vòng thử nghiệm.

---

## Các nhãn

| Nhãn | Ý nghĩa |
|------|---------|
| `intact` | Ảnh sản phẩm bình thường, không bị hỏng. Bao gồm cả ảnh mờ, ảnh cận cảnh chi tiết, ảnh đang mở hộp, hoặc ảnh sản phẩm cùng dòng nhưng khác variant |
| `damaged` | Sản phẩm hoặc bao bì bị hỏng rõ ràng — móp, vỡ, rách, rò rỉ |
| `wrong_item` | Ảnh là một món hàng hoàn toàn khác với sản phẩm đặt, và review text khiếu nại rõ ràng về việc giao sai |
| `irrelevant` | Ảnh không liên quan đến sản phẩm — selfie, ảnh phong cảnh, màn hình đen, screenshot app, hoặc ảnh spam kiếm xu |

---

## Pipeline

### Các bước thực hiện

```
download → extract → validate → build-images → label
```

**1. Download** — Tải ảnh và video từ URL trong CSV của scraper. Mỗi media được lưu vào `data/raw_media/` với tên file dạng `{review_id}_img{n}.jpg`.

**2. Extract** — Cắt ngẫu nhiên 3 frame từ mỗi video, lưu vào `data/frames/`.

**3. Validate** — Quét toàn bộ ảnh, xóa các file bị corrupted không mở được.

**4. Build-images** — Gộp ảnh tải về và frame video vào một manifest chung `images.csv`.

**5. Label** — Gọi vision model để gán nhãn từng ảnh. Kết quả ghi vào `labels.csv`, đồng thời copy ảnh vào thư mục `data/labeled/<label>/` để kiểm tra bằng mắt.

**6. Tạo training labels** — Sau khi auto-label xong, lọc và chuẩn hóa `labels.csv` để tạo ra `training_labels.csv` — file này chỉ giữ lại các cột cần thiết cho training và đảm bảo đường dẫn ảnh hợp lệ.

---

## Thiết kế prompt

Đây là phần tốn nhiều công nhất. Nhiều lớp ảnh trong thực tế có ranh giới mơ hồ, khiến model dễ phán sai nếu prompt không đủ rõ ràng. Nhóm xây dựng prompt theo cấu trúc:

- **Mô tả nhiệm vụ** — Xác định rõ model đang chuẩn bị dữ liệu cho computer vision, không phải đánh giá chất lượng sản phẩm.
- **4 quy tắc ghi đè tuyệt đối** — Xử lý các trường hợp dễ nhầm:
  - Variant vs Mismatch — cùng dòng sản phẩm nhưng khác size/màu thì vẫn là `intact`
  - Coin farming vs giao sai hàng — ảnh không liên quan kèm review tích cực là spam `irrelevant`, chỉ `wrong_item` khi review khiếu nại rõ ràng
  - Blur và phản chiếu không phải hư hỏng
  - Ảnh đóng gói/mở hộp là bình thường
- **15 ví dụ few-shot** — Bao phủ các trường hợp edge case thực tế từ dữ liệu cào được

Prompt được viết bằng tiếng Anh để tận dụng khả năng lý luận của model tốt hơn, nhưng các ví dụ few-shot có kèm tiếng Việt để sát với dữ liệu thực tế.

---

## Xử lý batch (Gemini)

Khi dùng `--provider google`, pipeline hỗ trợ gọi nhiều ảnh trong một lần API (`--batch-size 5` đến `10`). Model nhận toàn bộ ảnh và text cùng lúc, trả về JSON array. Cách này giảm đáng kể số lượt gọi API và thời gian chạy.

Với các provider khác (OpenAI, Groq), pipeline chạy từng ảnh một.

---

## Xác thực (Authentication)

Nhóm dùng Google Vertex AI thông qua **Application Default Credentials** — đăng nhập bằng `gcloud auth application-default login`, không cần tạo API key. Chỉ cần khai báo `PROJECT_ID` trong file `.env`.

Lý do chọn Vertex AI thay vì Google AI Studio API key: quota cao hơn và phù hợp hơn cho chạy batch lớn.

---

## Đánh giá chất lượng prompt

Để đo độ chính xác của prompt trước khi chạy toàn bộ, nhóm tạo một bộ **ground truth thủ công** gồm 17 ảnh đã được gán nhãn bằng tay, trải đều 4 lớp và bao gồm các trường hợp khó.

Script `eval_prompt.py` chạy model trên bộ này và in kết quả từng ảnh:

```
✅ [PASS] Ảnh: 2cdc63c500f2_img3.jpg | AI: intact == ĐÁP ÁN: intact
❌ [FAIL] Ảnh: 208688e9b2fd_img3.jpg | AI: irrelevant != ĐÁP ÁN: wrong_item
```

Quá trình này lặp lại nhiều lần — mỗi lần thấy model sai ở đâu thì bổ sung quy tắc hoặc ví dụ vào prompt tương ứng.

### Thành phần của bộ ground truth

| Nhãn | Số ảnh |
|------|--------|
| intact | 6 |
| irrelevant | 7 |
| damaged | 1 |
| wrong_item | 3 |
| **Tổng** | **17** |

---

## File đầu ra

| File | Mô tả |
|------|-------|
| `data/manifests/labels.csv` | Toàn bộ nhãn từ auto-label, kèm review text và metadata đầy đủ |
| `data/manifests/training_labels.csv` | Nhãn đã lọc và chuẩn hóa, sẵn sàng cho training pipeline |
| `data/manifests/ground_truth.csv` | 17 ảnh gán nhãn thủ công dùng để đánh giá prompt |
| `data/labeled/<label>/` | Ảnh đã phân loại theo thư mục, để kiểm tra bằng mắt |

---

## Từ labels.csv đến training_labels.csv

`labels.csv` là file thô đầu ra trực tiếp từ pipeline — chứa toàn bộ metadata gốc gồm review text, product URL, rating, ngày, source URL và nhãn. File này đầy đủ nhưng nặng và chứa nhiều cột không cần thiết cho training.

`training_labels.csv` được tạo ra bằng cách lọc và chuẩn hóa từ `labels.csv`:

- Chỉ giữ lại các cột cần thiết: `image_path` và `label`
- Lọc bỏ các dòng có đường dẫn ảnh không tồn tại trên disk
- Chuẩn hóa đường dẫn về dạng tương đối so với thư mục `image_labeling/`

File này là đầu vào trực tiếp cho notebook EDA và training pipeline.

---

## Những khó khăn gặp phải

**Ranh giới nhãn mơ hồ.** `irrelevant` và `wrong_item` dễ nhầm nhất. Người dùng trên sàn thương mại thường upload ảnh không liên quan (spam kiếm xu) nhưng review lại tích cực — nếu không có quy tắc rõ, model dễ gán `wrong_item` thay vì `irrelevant`.

**Variant sản phẩm.** Model ban đầu hay gán `wrong_item` cho các ảnh cùng dòng sản phẩm nhưng khác stage (ví dụ Abbott Grow 1+ và Abbott Grow 3). Phải thêm quy tắc phân biệt rõ Variant vs Mismatch mới sửa được lỗi này.

**Blur và phản chiếu.** Ảnh chụp lon sữa kim loại hay bị bóng phản chiếu hoặc bóng đổ trông giống vết móp. Model dễ gán `damaged` sai. Giải quyết bằng quy tắc override riêng kèm ví dụ cụ thể.

**Rate limit.** Chạy hàng nghìn ảnh liên tục dễ bị throttle. Giải quyết bằng `--sleep` và cơ chế resume tự động — chạy lại lệnh sẽ tự skip ảnh đã label.
