# Multimodal Review Web - Technical Specification

## 1. Tech Stack (Yêu cầu sử dụng)
- **Frontend:** Next.js 14+ (App Router), Tailwind CSS.
- **Backend Node.js (Orchestrator):** Express.js, Sequelize ORM (PostgreSQL) (dùng Supabase).
- **Backend Python (AI Worker):** FastAPI (Xử lý Scraper, PhoBERT, ResNet, CLIP).
- **Database:** PostgreSQL.
- **Real-time:** Socket.io (Đẩy tiến độ phân tích từ Server lên Client).
- **Job Queue:** BullMQ + Redis (Quản lý hàng chờ tác vụ nặng) (Redis dùng Upstash).
- **Authentication:** Google OAuth 2.0 (tích hợp qua Supabase Auth, đã cấu hình Google Cloud kết nối Supabase xong hết rồi).

## 2. Core Features (Chức năng hệ thống)
### A. Trang chủ (Input & Auth)
- Đăng nhập bằng Google để sử dụng tính năng lưu trữ lịch sử.
- Nhận URL sản phẩm thương mại điện tử (Shopee, Tiki...).
- Gửi URL sang Node.js để khởi tạo tiến trình phân tích bất đồng bộ.

### B. Màng lọc tiến trình (Processing State)
- Theo dõi tiến độ thời gian thực (Real-time tracking) qua Socket.io.
- Hiển thị trạng thái cụ thể: `Scraping`, `Image Verification (CLIP)`, `Damage Detection (ResNet)`, `Sentiment Analysis (PhoBERT)`.

### C. Dashboard kết quả (Analytics)
- **Thông tin sản phẩm:** Tên, giá, hình ảnh gốc.
- **Phân tích Đa phương thức:** - Biểu đồ cảm xúc theo khía cạnh (Chất lượng, Giao hàng).
    - Biểu đồ phân loại tình trạng vật lý ảnh (Intact, Damaged, Wrong Item, Irrelevant).
- **AI Insight:** Đoạn tóm tắt và cảnh báo rủi ro sinh bởi LLM.

### D. Quản lý Lịch sử (User History)
- Xem danh sách các sản phẩm đã từng phân tích (chỉ dành cho user đã đăng nhập).
- Tra cứu lại chi tiết kết quả cũ mà không cần chạy lại AI Backend.
- Xóa lịch sử (Xóa Product và các Review/Report liên quan).

## 3. Workflow Logic & API Specification

1. **Khởi tạo (Frontend -> Node.js):**
    - **Frontend:** `POST /api/analyze` kèm `{ url: string, userId?: string }`.
    - **Node.js Controller:** - Khởi tạo record `Product` với `status: 'PENDING'` và gán `userId` (nếu có).
        - Đẩy Job vào **BullMQ** với dữ liệu: `{ productId: product.id, url: url }`.
        - Trả về `productId` ngay lập tức để Frontend join vào **Socket.io Room** có ID tương ứng.

2. **Điều phối (Node.js Worker -> Python FastAPI):**
    - **Node.js Worker:** Tự động bốc Job từ Redis, cập nhật `status: 'PROCESSING'`.
    - **Action:** Gọi `POST http://localhost:8000/process-job` (FastAPI) kèm `{ productId, url }`.
    - **Tracking:** Trong khi chờ, Python liên tục gọi API `POST /api/update-progress` của Node.js để phát tín hiệu Socket.io (progress %, message) về Client.

3. **Xử lý & Trả bài (Python FastAPI -> Node.js Webhook):**
    - **Python AI Engine:** Thực hiện Scrape và chạy 3 model AI (CLIP, ResNet, PhoBERT).
    - **Action:** Khi hoàn tất, Python gọi **Webhook** `POST /api/webhook/finished` của Node.js.
    - **Payload JSON:**
      ```json
      {
        "productId": "uuid",
        "productData": { "name": "string", "thumbnail": "string", "price": "string" },
        "reviews": [
          { "review_text": "string", "rating": 5, "image_path": "string", "label": "intact/damaged/wrong_item/irrelevant", "sentiment": "positive/neutral/negative" }
        ],
        "summary": "Đoạn tóm tắt từ Gemini"
      }
      ```

4. **Tích hợp & Kết thúc (Node.js -> Database -> Frontend):**
    - **Node.js Webhook:** - Cập nhật thông tin `Product` (name, thumbnail) và chuyển `status: 'COMPLETED'`.
        - Dùng **Sequelize** `Review.bulkCreate(reviews)` để lưu hàng ngàn bản ghi vào PostgreSQL trong một lần query duy nhất để tối ưu hiệu năng.
        - Lưu `Report` (summary_text) liên kết với `product_id`.
    - **Socket.io:** Phát sự kiện `finished` kèm theo Payload kết quả cuối cùng để Frontend render Dashboard.

5. **API Quản lý Lịch sử (Dành cho User):**
    - `GET /api/history`: Trả về danh sách `Product` của `userId` hiện tại.
    - `GET /api/history/:productId`: Trả về chi tiết `Product`, `Reviews`, và `Report`.
    - `DELETE /api/history/:productId`: Xóa `Product` (Cấu hình `ON DELETE CASCADE` trong Sequelize để xóa sạch Review/Report).

## 4. Database Schema (Sequelize Models)

### A. Model: User
- `id`: UUID (Primary Key).
- `email`: String (Unique).
- `name`: String.
- `avatar`: String.

### B. Model: Product
- `id`: UUID (Primary Key).
- `userId`: UUID (Foreign Key, Nullable nếu ẩn danh).
- `name`: String (Tên sản phẩm).
- `url`: String (Link gốc).
- `thumbnail`: String (Ảnh đại diện sản phẩm).
- `status`: Enum ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED').

### C. Model: Review
- `id`: UUID (Primary Key).
- `product_id`: UUID (Foreign Key).
- `review_text`: Text (Nội dung review).
- `rating`: Integer (Số sao).
- `image_path`: String (Đường dẫn ảnh).
- `label`: Enum ('intact', 'damaged', 'wrong_item', 'irrelevant').
- `sentiment`: String (Kết quả từ PhoBERT).

### D. Model: Report
- `id`: UUID (Primary Key).
- `product_id`: UUID (Foreign Key).
- `summary_text`: Text (Kết quả từ Gemini).
- `risk_level`: String (Cảnh báo rủi ro).