# Hướng Dẫn Chạy Toàn Bộ Hệ Thống (Frontend + Backend Node + Backend Python AI)

Dự án **Ecommerce Review Analytics** được chia thành 3 service riêng biệt chạy song song với nhau. Dưới đây là hướng dẫn chi tiết cách khởi động từng phần để hệ thống hoạt động hoàn chỉnh trên môi trường Windows.

---

## Yêu cầu môi trường (Prerequisites)
- **Node.js** (Phiên bản v18 trở lên)
- **Python** (Phiên bản 3.9 trở lên)
- Đảm bảo cơ sở dữ liệu (PostgreSQL/Supabase) đã được kết nối trong các file `.env` tương ứng.

---

## Bước 1: Khởi động Backend Node.js (Web Platform Backend)
Đảm nhiệm vai trò là cầu nối giữa Client, DB và AI Engine. Nó cũng cung cấp WebSockets (`socket.io`) để cập nhật trạng thái theo thời gian thực (Realtime Progress) lên màn hình người dùng.

1. Mở một cửa sổ Terminal/Command Prompt mới.
2. Di chuyển vào thư mục backend:
   ```bash
   cd web_platform\backend
   ```
3. Cài đặt các thư viện (chỉ cần chạy lần đầu):
   ```bash
   npm install
   ```
4. Khởi động server ở chế độ Development:
   ```bash
   npm run dev
   ```
> 💡 **Kết quả:** Nếu thành công, bạn sẽ thấy dòng thông báo Node.js Server đang chạy trên port **5000** (hoặc tương tự tuỳ vào cấu hình `.env`).

---

## Bước 2: Khởi động Backend Python (AI Engine)
Đây là trái tim phân tích của hệ thống, sử dụng FastAPI. Nó xử lý cào dữ liệu, gọi Model Computer Vision (ResNet/CLIP) và NLP Model (PhoBERT/Gemini).

1. Mở một cửa sổ Terminal/Command Prompt **thứ 2**.
2. Di chuyển vào thư mục chứa code AI:
   ```bash
   cd ai_engine
   ```
3. Kích hoạt môi trường ảo (Virtual Environment):
   ```bash
   # Dành cho Windows (PowerShell/CMD):
   .\venv\Scripts\activate
   ```
   *(Sau khi kích hoạt, bạn sẽ thấy chữ `(venv)` hiện ở đầu dòng lệnh)*
4. Cài đặt các package cần thiết (chỉ cần chạy lần đầu):
   ```bash
   pip install -r requirements.txt
   ```
5. Khởi động server FastAPI:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
> 💡 **Kết quả:** FastAPI sẽ lắng nghe ở port **8000**. Nó luôn sẵn sàng nhận Request từ Node.js Backend bắn sang mỗi khi có link sản phẩm mới.

---

## Bước 3: Khởi động Frontend (Next.js Web Platform)
Giao diện người dùng tuyệt đẹp mà bạn tương tác trực tiếp.

1. Mở một cửa sổ Terminal/Command Prompt **thứ 3**.
2. Di chuyển vào thư mục frontend:
   ```bash
   cd web_platform\frontend
   ```
3. Cài đặt các modules (chỉ cần chạy lần đầu):
   ```bash
   npm install
   ```
4. Chạy Web App ở chế độ Development:
   ```bash
   npm run dev
   ```
> 💡 **Kết quả:** Next.js sẽ build và mở cổng ở **3000**. Bây giờ bạn có thể mở trình duyệt và truy cập vào địa chỉ: **[http://localhost:3000](http://localhost:3000)**

---

## Tóm lược Luồng Hoạt Động Của Cả 3 🔄
1. **[Cổng 3000 - Frontend]**: Người dùng dán Link Shopee/Tiki và bấm "Phân Tích". Request được gửi xuống Node.js.
2. **[Cổng 5000 - Node Backend]**: Node.js tiếp nhận, lưu dữ liệu tạm, bắn WebSockets cập nhật trạng thái, đồng thời gửi Request phân tích sang FastAPI bằng HTTP.
3. **[Cổng 8000 - Python AI]**: FastAPI (AI Engine) nhận Link, tiến hành xử lý cào data, chạy mô hình AI phân tích ảnh và chữ. Quá trình chạy tới đâu, Python lại gọi `Webhook` bắn ngược tiến độ về Node.js (Cổng 5000).
4. Hệ thống Node.js lập tức đẩy Realtime WebSockets lên màn hình Client (Cổng 3000) hiển thị thanh Progress Bar tuyệt đẹp.
5. Khi hoàn tất, kết quả cuối cùng được trả về và trang Web hiển thị các biểu đồ.

🎉 **Hoàn tất! Hệ thống đã sẵn sàng cho bạn test phân tích đa phương thức!**
