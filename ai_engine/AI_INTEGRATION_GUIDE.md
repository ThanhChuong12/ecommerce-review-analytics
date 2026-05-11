# Hướng Dẫn Tích Hợp AI (Dành cho Developer AI Engine)

Tài liệu này giải thích luồng hoạt động hiện tại giữa **Python (AI Engine)** và **Web Platform (Node.js Backend & Next.js Frontend)**, đồng thời đưa ra lộ trình rõ ràng để bạn thay thế code mô phỏng (mocking) bằng hệ thống thật.

---

## 1. Tình trạng luồng hoạt động hiện tại (Mock Simulation)

Hiện tại, file `main.py` chỉ đóng vai trò **giả lập (Mock API)** để hoàn thiện luồng giao tiếp với Web. 

Khi Node.js bắn một request POST sang `/process-job` (kèm theo `productId` và `url`), đây là những gì đang diễn ra trong hàm `heavy_ai_process`:
1. **Giả lập xử lý (Delay):** Dùng `time.sleep(2)` để bắt chước thời gian chờ cào dữ liệu, thời gian inference của mô hình Ảnh và chữ.
2. **Cập nhật giao diện (Realtime Progress):** Ở mỗi bước, AI Engine bắn HTTP POST request vào `WEBHOOK_PROGRESS` (Cổng 5000 của Node.js) mang theo biến `progress` (%). Node.js sẽ dùng WebSocket đẩy % này lên cho Frontend nhích thanh loading.
3. **Giả lập dữ liệu kết quả:** Sinh ra một cục JSON cứng (Hardcoded data) gồm 5 review mẫu nhân bản lên và thông tin `product_data` giả.
4. **Trả kết quả cuối:** Bắn cục JSON giả đó vào `WEBHOOK_FINISHED`. Node.js nhận được sẽ tự động lưu vào Database và báo cho Frontend mở màn hình Kết Quả.

---

## 2. Nhiệm vụ tiếp theo: Chuyển đổi thành AI thật (Dựa trên đồ án)

Để hoàn thiện đồ án, người phát triển AI Engine cần xoá bỏ các đoạn `time.sleep` và thay thế bằng logic chạy thật tuần tự theo các bước sau:

### Bước 1: Viết Script Cào Dữ Liệu (Scraping Agent)
- Nhận tham số `url` từ Request.
- Chạy Selenium / Playwright hoặc BeautifulSoup (từ thư mục `scraping_agent`) để truy cập link E-commerce.
- Bóc tách thông tin: Tên sản phẩm, giá, link ảnh Thumbnail, và **đặc biệt là danh sách các Reviews (Text + URL ảnh review)**.
- Gửi Webhook: `requests.post(WEBHOOK_PROGRESS, json={"productId": ..., "progress": 30, "message": "Đã cào xong dữ liệu..."})`

### Bước 2: Tích hợp Computer Vision (ResNet & CLIP)
- Đưa danh sách các link ảnh review vừa cào được đi tải về hoặc tải thẳng vào RAM.
- Đưa qua Pipeline mô hình ảnh (trong thư mục `image_processing`).
- Mô hình phải dự đoán nhãn (label) cho từng ảnh: `intact` (nguyên vẹn), `damaged` (móp méo), `wrong_item` (sai hàng), `irrelevant` (không liên quan).
- Gửi Webhook Progress tiếp tục lên mức `60%`.

### Bước 3: Tích hợp Natural Language Processing (PhoBERT)
- Lấy toàn bộ Text Review đưa qua Pipeline NLP (trong thư mục `text_processing`).
- PhoBERT dự đoán cảm xúc (sentiment) cho từng text: `positive` (Tích cực), `neutral` (Trung lập), `negative` (Tiêu cực).
- Gửi Webhook Progress lên mức `80%`.

### Bước 4: Tích hợp Large Language Model (Gemini / ChatGPT)
- Gộp các text review tiêu biểu (hoặc toàn bộ nếu token cho phép) để đẩy vào Prompt.
- Yêu cầu LLM (trong thư mục `llm_integration`) đọc qua và sinh ra 1 đoạn nhận xét ngắn gọn (Tóm tắt chất lượng sản phẩm & tình trạng vận chuyển).
- Đoạn này chính là biến `summary`.

### Bước 5: Đóng gói và Giao tiếp Database (Hoàn tất)
Gom tất cả dữ liệu thật thu được nãy giờ thành định dạng Dict y hệt như cấu trúc cũ:
```python
payload = {
    "productId": product_id,       # Lấy từ Request ban đầu
    "productData": {
        "name": scraped_name,
        "thumbnail": scraped_image_url,
        "price": scraped_price
    },
    "reviews": [
        {
            "review_text": "Hàng đẹp", 
            "rating": 5, 
            "image_path": "url_anh_neu_co", 
            "label": "intact",       # Kết quả từ CV model
            "sentiment": "positive"  # Kết quả từ NLP model
        },
        # ... các review khác
    ],
    "summary": llm_generated_summary # Kết quả từ LLM
}
```
Sau đó gọi:
```python
requests.post(WEBHOOK_FINISHED, json=payload)
```
> ⚠️ **Lưu ý quan trọng:** Bạn không cần viết code kết nối Postgres (SQLAlchemy, v.v.) bên trong AI Engine. Nhiệm vụ của bạn chỉ là bắn `WEBHOOK_FINISHED` đúng cục JSON như trên, Backend Node.js sẽ lo toàn bộ việc validation và Insert vào DB!
