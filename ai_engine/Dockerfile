FROM python:3.9-slim
WORKDIR /app
# Cài đặt tạm các thư viện nhẹ để chạy giao diện web (Mock Data)
# TODO: Mở comment dòng dưới và xóa dòng cài fastapi khi tích hợp AI Models thật
# RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --default-timeout=100 --no-cache-dir fastapi uvicorn requests pydantic
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]