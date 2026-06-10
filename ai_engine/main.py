# pyrefly: ignore [missing-import]
from fastapi import FastAPI, BackgroundTasks, Request
import asyncio
import requests
import requests.exceptions
import time
import os
import random
from datetime import datetime, timedelta

# detect_defect sẽ được dùng khi weights đã train xong
# from ai_engine.image_processing.defect_detection import detect_defect

app = FastAPI()

WEBHOOK_PROGRESS = os.getenv("NODE_WEBHOOK_PROGRESS", "http://localhost:5000/api/webhook/update-progress")
WEBHOOK_FINISHED = os.getenv("NODE_WEBHOOK_FINISHED", "http://localhost:5000/api/webhook/finished")

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
    """Pipeline AI chạy nền: scrape → lọc ảnh → defect detection → gửi kết quả."""
    print(f"[Python AI] Bat dau xu ly Product ID: {product_id} | URL: {url}")

    _report_progress(product_id, 15, "Đang cào dữ liệu E-commerce...")
    time.sleep(1)

    _report_progress(product_id, 35, "Đang lọc ảnh không liên quan (CLIP)...")
    time.sleep(1)

    _report_progress(product_id, 55, f"Đang nhận diện tình trạng hộp ({IMAGE_BACKBONE})...")
    time.sleep(1)

    _report_progress(product_id, 80, "Đang đánh giá cảm xúc ngôn ngữ (PhoBERT)...")
    time.sleep(1)

    # --- Tùy biến mock data ngẫu nhiên ---
    mock_products = [
        {"name": "Áo thun levent", "thumbnail": "https://down-vn.img.susercontent.com/file/vn-11134207-7r98o-lwqqeti8ig9l26", "url": "https://shopee.vn/product/123/456"},
        {"name": "Quần jean baggy nam ống rộng", "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTIR25xMN6cFcwvem2QzU4Ckc_0Pm7QG1BR-A&s", "url": "https://shopee.vn/product/123/457"},
        {"name": "Album Blackpink", "thumbnail": "https://i5.walmartimages.com/seo/Blackpink-The-Album-Version-4-CD_ef501910-ec67-48b5-aa7c-3473a753765f.58ef4adb6a4c7b69ff24a416c2f8ee0c.jpeg", "url": "https://shopee.vn/product/123/458"},
        {"name": "Truyện tranh Doraemon", "thumbnail": "https://product.hstatic.net/1000376556/product/xhljijuw_9de22abba6a2407d87e202d773acda07_1024x1024.png", "url": "https://shopee.vn/product/123/459"},
        {"name": "Ốp lưng iPhone", "thumbnail": "https://lucas.vn/wp-content/uploads/2023/09/Op-Lung-iPhone-15-Series-WIWU-Magsafe-Mat-Lung-Nham-Chong-Van-Tay-8.png", "url": "https://shopee.vn/product/123/460"},
        {"name": "Tai nghe Bluetooth", "thumbnail": "https://cf.shopee.vn/file/dad9a07e8346b1ba21b39f615ff16b63", "url": "https://shopee.vn/product/123/461"},
        {"name": "Sách Mắt Biếc", "thumbnail": "https://www.nxbtre.com.vn/Images/Book/nxbtre_full_01372019_043734.jpg", "url": "https://shopee.vn/product/123/462"}
    ]

    selected_product = random.choice(mock_products)
    product_name = selected_product["name"]
    thumbnail = selected_product["thumbnail"]
    mock_img = thumbnail

    product_data = {
        "name": product_name,
        "thumbnail": thumbnail
    }

    # Generate mock reviews
    base_reviews = [
        {"review_text": "Sản phẩm xài rất thích, đáng tiền.", "rating": 5, "image_path": mock_img, "label": "intact", "sentiment": "positive"},
        {"review_text": "Hộp bị móp méo tơi tả, thất vọng quá.", "rating": 1, "image_path": mock_img, "label": "damaged", "sentiment": "negative"},
        {"review_text": "Cũng tạm được, không có gì đặc sắc.", "rating": 3, "image_path": mock_img, "label": "intact", "sentiment": "neutral"},
        {"review_text": "Shop giao sai màu, làm ăn chán thật.", "rating": 1, "image_path": mock_img, "label": "wrong_item", "sentiment": "negative"},
        {"review_text": "Đóng gói kỹ, hàng mới tinh.", "rating": 5, "image_path": mock_img, "label": "intact", "sentiment": "positive"},
        {"review_text": "Chụp cái ảnh không liên quan nhận xu.", "rating": 5, "image_path": "https://img.freepik.com/free-photo/beautiful-scenery-road-forest-with-lot-colorful-autumn-trees_181624-30942.jpg", "label": "irrelevant", "sentiment": "positive"},
        {"review_text": "Tuyệt vời nhưng mà vỏ bị trầy nhẹ.", "rating": 4, "image_path": mock_img, "label": "damaged", "sentiment": "positive"}, # Cross-modal case
    ]
    
    users = ["Nguyễn Văn A", "Trần Thị B", "Lê Văn C", "Phạm Thị D", "Hoàng Văn E", "Vũ Thị F", "Đặng Văn G"]
    processed_reviews = []
    
    for i in range(50):
        base = random.choice(base_reviews)
        rev = base.copy()
        rev["user_name"] = random.choice(users) + f"_{i}"
        rev["date"] = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
        processed_reviews.append(rev)

    summary = "Sản phẩm được đánh giá khá tốt về chất lượng, tuy nhiên có nhiều phản ánh về việc đóng gói kém dẫn đến hỏng hóc trong quá trình vận chuyển."
    
    # --- Similar products: thử lấy thật từ scraper, fallback về mock nếu thất bại ---
    real_similar: list[dict] = []
    try:
        import sys
        from pathlib import Path
        _agent_root = str(Path(__file__).resolve().parent.parent / "scraping_agent")
        if _agent_root not in sys.path:
            sys.path.insert(0, _agent_root)
        # pyrefly: ignore [missing-import]
        from similar_products_fetcher import scrape_similar_products

        # heavy_ai_process là sync function → chạy coroutine bằng asyncio.run()
        similar_items = asyncio.run(scrape_similar_products(url, limit=5))
        real_similar = [
            {
                "name":       p.name,
                "thumbnail":  p.image_url,
                "url":        p.url,
                "trustScore": random.randint(80, 98),  # placeholder cho đến khi có real score
            }
            for p in similar_items if p.name
        ]
        if real_similar:
            print(f"[Python AI] Similar products: đã lấy được {len(real_similar)} sản phẩm từ scraper")
    except Exception as _e:
        print(f"[Python AI] Similar products scraper thất bại, dùng mock: {_e}")

    # Nếu scraper không trả về gì (URL không hỗ trợ hoặc bị block) → fallback mock
    alternative_products = real_similar or [
        {
            "name": p["name"],
            "thumbnail": p["thumbnail"],
            "url": p["url"],
            "trustScore": random.randint(80, 98)
        }
        for p in random.sample([p for p in mock_products if p["name"] != product_name], min(5, len(mock_products)-1))
    ]

    # --- Dữ liệu AI nâng cao (Metadata) ---
    metadata = {
        "spamPercentage": random.randint(10, 45),
        "trustScore": random.randint(40, 95),
        "aspectSentiment": {
            "Product": round(random.uniform(3.5, 4.8), 1),
            "Packaging": round(random.uniform(2.0, 4.5), 1),
            "Shipping": round(random.uniform(2.5, 4.9), 1)
        },
        "sentimentTimeSeries": [],
        "keywords": {
            "positive": [
                {"text": "tuyệt vời", "value": 60}, {"text": "đẹp", "value": 55}, {"text": "chất lượng", "value": 45}, 
                {"text": "thơm", "value": 40}, {"text": "nhanh", "value": 38}, {"text": "chính hãng", "value": 35},
                {"text": "giá rẻ", "value": 30}, {"text": "đóng gói kỹ", "value": 32}, {"text": "hài lòng", "value": 48},
                {"text": "ủng hộ", "value": 25}, {"text": "sang trọng", "value": 20}, {"text": "xịn", "value": 28},
                {"text": "tốt", "value": 50}, {"text": "chuẩn", "value": 22}, {"text": "đáng tiền", "value": 42},
                {"text": "mịn", "value": 15}, {"text": "giao nhanh", "value": 36}, {"text": "nhiệt tình", "value": 18}
            ],
            "negative": [
                {"text": "móp méo", "value": 50}, {"text": "trầy xước", "value": 45}, {"text": "chậm", "value": 40}, 
                {"text": "thái độ", "value": 35}, {"text": "sai hàng", "value": 60}, {"text": "tệ", "value": 55},
                {"text": "thất vọng", "value": 42}, {"text": "cũ", "value": 30}, {"text": "hư hỏng", "value": 48},
                {"text": "không đáng", "value": 25}, {"text": "chất lượng kém", "value": 38}, {"text": "fake", "value": 28},
                {"text": "lừa đảo", "value": 20}, {"text": "dỏm", "value": 18}, {"text": "vỡ", "value": 33},
                {"text": "xấu", "value": 22}, {"text": "bong tróc", "value": 15}, {"text": "rách", "value": 27}
            ]
        },
        "smartAdvice": "💡 Gợi ý mua hàng: Sản phẩm có chất lượng tốt nhưng rủi ro hư hỏng bao bì cao. Khuyến nghị bạn NHẮN TIN CHO SHOP yêu cầu bọc thêm màng xốp nổ (Bubble Wrap) trước khi đặt hàng.",
        "alternativeProducts": alternative_products,
    }

    # Sinh Time Series data
    for i in range(14, -1, -1):
        d = (datetime.now() - timedelta(days=i)).strftime("%m/%d")
        metadata["sentimentTimeSeries"].append({
            "date": d,
            "positive": random.randint(10, 50),
            "neutral": random.randint(5, 20),
            "negative": random.randint(2, 15) if i != 5 else random.randint(40, 60) # Simulate a spike
        })

    # Gửi kết quả cuối về Node.js qua webhook
    _report_progress(product_id, 100, "Hoàn tất! Đang lưu báo cáo...")
    requests.post(WEBHOOK_FINISHED, json={
        "productId": product_id,
        "productData": product_data,
        "reviews": processed_reviews,
        "summary": summary,
        "metadata": metadata
    }, timeout=120)


@app.post("/process-job")
async def receive_job(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    background_tasks.add_task(heavy_ai_process, payload.get("productId"), payload.get("url"))
    return {"message": "Python đã nhận job!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

