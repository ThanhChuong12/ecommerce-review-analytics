from fastapi import FastAPI, BackgroundTasks, Request
import requests
import time
import os

app = FastAPI()

WEBHOOK_FINISHED = os.getenv("NODE_WEBHOOK_FINISHED", "http://localhost:5000/api/webhook/finished")
WEBHOOK_PROGRESS = os.getenv("NODE_WEBHOOK_PROGRESS", "http://localhost:5000/api/webhook/update-progress")

def heavy_ai_process(product_id: int, url: str):
    print(f"\n[Python AI] Bắt đầu xử lý Product ID: {product_id}")
    
    # 1. Phát tín hiệu đang cào
    requests.post(WEBHOOK_PROGRESS, json={"productId": product_id, "progress": 20, "message": "Đang cào dữ liệu E-commerce..."})
    time.sleep(2)
    
    # 2. Phát tín hiệu chạy model ảnh
    requests.post(WEBHOOK_PROGRESS, json={"productId": product_id, "progress": 55, "message": "Đang phân tích hình ảnh (ResNet & CLIP)..."})
    time.sleep(2)

    # 3. Phát tín hiệu chạy model text
    requests.post(WEBHOOK_PROGRESS, json={"productId": product_id, "progress": 85, "message": "Đang đánh giá cảm xúc ngôn ngữ (PhoBERT)..."})
    time.sleep(2)
    
    # Giả lập Data
    product_data = {
        "name": "Sữa Bột Abbott Grow 4",
        "thumbnail": "https://cf.shopee.vn/file/dad9a07e8346b1ba21b39f615ff16b63",
        "price": "350,000đ"
    }
    
    # Tạo 2000 review bằng cách nhân bản để vẽ biểu đồ
    sample_reviews = [
        {"review_text": "Sữa thơm", "rating": 5, "image_path": "", "label": "intact", "sentiment": "positive"},
        {"review_text": "Hộp móp", "rating": 3, "image_path": "", "label": "damaged", "sentiment": "negative"},
        {"review_text": "Bình thường", "rating": 4, "image_path": "", "label": "intact", "sentiment": "neutral"},
        {"review_text": "Giao nhầm", "rating": 1, "image_path": "", "label": "wrong_item", "sentiment": "negative"},
        {"review_text": "Tuyệt vời", "rating": 5, "image_path": "", "label": "intact", "sentiment": "positive"}
    ]
    processed_reviews = sample_reviews * 10
    summary = "Sản phẩm tốt, khách ưng ý chất lượng. Nhưng khâu vận chuyển còn kém làm móp hộp khá nhiều."

    # Gửi kết quả cuối cùng
    payload = {
        "productId": product_id,
        "productData": product_data,
        "reviews": processed_reviews,
        "summary": summary
    }
    
    requests.post(WEBHOOK_PROGRESS, json={"productId": product_id, "progress": 100, "message": "Hoàn tất! Đang lưu báo cáo..."})
    requests.post(WEBHOOK_FINISHED, json=payload, timeout=120)

@app.post("/process-job")
async def receive_job(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    background_tasks.add_task(heavy_ai_process, payload.get("productId"), payload.get("url"))
    return {"message": "Python đã nhận job!"}

if __name__ == "__main__":
    import uvicorn
    # Kích hoạt server FastAPI chạy ở port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
