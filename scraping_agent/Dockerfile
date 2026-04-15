# Sử dụng Python bản nhẹ làm nền tảng
FROM python:3.9-slim

# Cài đặt các thư viện hệ thống cơ bản cần thiết cho môi trường
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Copy và cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- QUAN TRỌNG: NẾU BẠN DÙNG PLAYWRIGHT ---
# Hãy bỏ comment (xóa dấu #) ở dòng dưới đây để hệ thống tự động tải trình duyệt Chromium về
# RUN playwright install --with-deps chromium
# -------------------------------------------

# Copy toàn bộ mã nguồn của agent vào container
COPY . .

# Khởi chạy Agent. (Giả sử file main.py của bạn đang chạy một server FastAPI nhỏ ở port 8001 để chờ nhận URL từ Node.js)
CMD ["python", "main.py"]