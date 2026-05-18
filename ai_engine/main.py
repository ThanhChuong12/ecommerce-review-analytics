# pyrefly: ignore [missing-import]
from fastapi import FastAPI, BackgroundTasks, Request
import requests
import requests.exceptions
import time
import os

# detect_defect sẽ được dùng khi weights đã train xong
# from ai_engine.image_processing.defect_detection import detect_defect

app = FastAPI()

WEBHOOK_FINISHED = os.getenv("NODE_WEBHOOK_FINISHED", "http://localhost:5000/api/webhook/finished")
WEBHOOK_PROGRESS = os.getenv("NODE_WEBHOOK_PROGRESS", "http://localhost:5000/api/webhook/update-progress")

# Backbone mặc định — đổi thành "resnet50" để dùng ResNet (chính xác hơn, chậm hơn)
IMAGE_BACKBONE = os.getenv("IMAGE_BACKBONE", "mobilenet_v3")


def _report_progress(product_id: int, progress: int, message: str):
    """Gửi tiến độ về Node.js để emit Socket.IO cho frontend."""
    try:
        requests.post(WEBHOOK_PROGRESS, json={
            "productId": product_id,
            "progress": progress,
            "message": message,
        }, timeout=5)
    except requests.exceptions.RequestException as e:
        # Không crash pipeline nếu Node.js chưa sẵn sàng nhận webhook
        print(f"[Warning] Progress webhook failed: {e}")


def heavy_ai_process(product_id: int, url: str) -> None:
    """Pipeline AI chạy nền: scrape → lọc ảnh → defect detection → gửi kết quả.

    Args:
        product_id: ID sản phẩm trong DB để gửi kèm kết quả.
        url: URL trang sản phẩm E-commerce cần scrape (dùng ở scraping_agent).
    """
    print(f"[Python AI] Bat dau xu ly Product ID: {product_id} | URL: {url}")

    # Bước 1: Scrape review (TODO: tích hợp scraping_agent thật)
    _report_progress(product_id, 15, "Đang cào dữ liệu E-commerce...")
    time.sleep(2)

    # Bước 2: Lọc ảnh rác bằng CLIP (TODO: tích hợp zero_shot_clip.py)
    _report_progress(product_id, 35, "Đang lọc ảnh không liên quan (CLIP)...")
    time.sleep(1)

    # Bước 3: Defect detection bằng ResNet50 / MobileNetV3
    _report_progress(product_id, 55, f"Đang nhận diện tình trạng hộp ({IMAGE_BACKBONE})...")
    time.sleep(1)

    # Bước 4: Phân tích cảm xúc văn bản (TODO: tích hợp PhoBERT)
    _report_progress(product_id, 80, "Đang đánh giá cảm xúc ngôn ngữ (PhoBERT)...")
    time.sleep(2)

    # --- Dữ liệu giả lập (sẽ thay bằng kết quả thật khi scraper + model hoàn thiện) ---
    product_data = {
        "name": "Sữa Bột Abbott Grow 4",
        "thumbnail": "https://cf.shopee.vn/file/dad9a07e8346b1ba21b39f615ff16b63",
        "price": "350,000đ",
    }

    # Mẫu review với label từ defect_detection (hiện là mock — thay bằng kết quả thật)
    sample_reviews = [
        {"review_text": "Sữa thơm", "rating": 5, "image_path": "", "label": "intact", "sentiment": "positive"},
        {"review_text": "Hộp móp", "rating": 3, "image_path": "", "label": "damaged", "sentiment": "negative"},
        {"review_text": "Bình thường", "rating": 4, "image_path": "", "label": "intact", "sentiment": "neutral"},
        {"review_text": "Giao nhầm", "rating": 1, "image_path": "", "label": "wrong_item", "sentiment": "negative"},
        {"review_text": "Tuyệt vời", "rating": 5, "image_path": "", "label": "intact", "sentiment": "positive"},
    ]
    processed_reviews = sample_reviews * 10
    summary = "Sản phẩm tốt, khách ưng ý chất lượng. Nhưng khâu vận chuyển còn kém làm móp hộp khá nhiều."

    # Gửi kết quả cuối về Node.js qua webhook
    _report_progress(product_id, 100, "Hoàn tất! Đang lưu báo cáo...")
    requests.post(WEBHOOK_FINISHED, json={
        "productId": product_id,
        "productData": product_data,
        "reviews": processed_reviews,
        "summary": summary,
    }, timeout=120)


@app.post("/process-job")
async def receive_job(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    background_tasks.add_task(heavy_ai_process, payload.get("productId"), payload.get("url"))
    return {"message": "Python đã nhận job!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

