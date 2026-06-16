# Walkthrough: Tích Hợp Real AI Pipeline Vào Web Platform

## Tổng Quan

Đã thay thế hoàn toàn **mock data** trong `ai_engine/main.py` bằng **pipeline AI thực tế** kết nối tất cả các module đã train.

---

## Các File Đã Sửa

### 1. [`ai_engine/main.py`](file:///e:/Nhập môn học máy/Project/ecommerce-review-analytics/ai_engine/main.py) — THAY THẾ HOÀN TOÀN

Pipeline mới thực hiện đúng thứ tự:

| Bước | Progress | Mô tả |
|------|----------|-------|
| 1 | 10% | **Scraping**: `scraping_agent/scraper/dispatcher.py` → scraped_reviews |
| 2 | 25% | **Spam Filter**: `spam_filter.detect_spam()` → is_spam flags |
| 3 | 40% | **Sentiment**: Heuristic lexicon + `assign_heuristic_label()` → positive/negative/neutral |
| 4 | 55% | **Download ảnh**: urllib → local temp files |
| 5 | 70% | **MobileNetV3**: `detect_defect_mobilenet_batch()` → intact/damaged/wrong_item/irrelevant |
| 6 | 80% | **Fusion Engine**: `TrustScoreCalculator.calculate()` → Trust Score |
| 7 | 88% | **LLM Summary**: `LLMRecommendationClient` → tóm tắt AI |
| 8 | 93% | **Similar Products**: `scrape_similar_products()` → alternatives |
| 9 | 99% | **Webhook** → Node.js → DB → Socket.IO → Frontend |

**Graceful fallbacks** tại mỗi bước — pipeline không crash ngay cả khi một module thất bại.

### 2. [`ai_engine/.env`](file:///e:/Nhập môn học máy/Project/ecommerce-review-analytics/ai_engine/.env) — CẬP NHẬT

Thêm đủ biến môi trường cần thiết:
- `NODE_WEBHOOK_PROGRESS` / `NODE_WEBHOOK_FINISHED`
- `IMAGE_BACKBONE=mobilenet_v3`
- `MAX_REVIEWS_SCRAPE=200`
- `MAX_IMAGES_PROCESS=40`

### 3. [`ai_engine/requirements.txt`](file:///e:/Nhập môn học máy/Project/ecommerce-review-analytics/ai_engine/requirements.txt) — THÊM DEPENDENCY

Thêm `python-dotenv>=1.0.0` để load `.env` file.

---

## Cách Chạy

### Terminal 1: Python AI Engine
```bash
cd e:\Nhập môn học máy\Project\ecommerce-review-analytics
# Cài dotenv nếu chưa có
pip install python-dotenv

cd ai_engine
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2: Node.js Backend
```bash
cd e:\Nhập môn học máy\Project\ecommerce-review-analytics\web_platform\backend
node index.mjs
```

### Terminal 3: Next.js Frontend
```bash
cd e:\Nhập môn học máy\Project\ecommerce-review-analytics\web_platform\frontend
npm run dev
```

Sau đó vào `http://localhost:3000`, dán URL Shopee/Tiki/Lazada → nhấn **Phân Tích**.

---

## Kiến Trúc Dữ Liệu (Payload từ Python về Node.js)

```json
{
  "productId": 123,
  "productData": {
    "name": "Tên sản phẩm thực từ scraper",
    "thumbnail": "https://cdn.shopee.vn/..."
  },
  "reviews": [
    {
      "review_text": "Nội dung đánh giá thực",
      "rating": 4,
      "image_path": "https://cdn.shopee.vn/review/...",
      "label": "intact",
      "sentiment": "positive",
      "date": "2024-01-15"
    }
  ],
  "summary": "Tóm tắt AI (LLM hoặc heuristic)...",
  "metadata": {
    "spamPercentage": 15,
    "trustScore": 78.5,
    "aspectSentiment": {
      "Product": 4.2,
      "Packaging": 3.8,
      "Shipping": 4.0
    },
    "sentimentTimeSeries": [
      {"date": "06/01", "positive": 12, "negative": 3, "neutral": 5}
    ],
    "keywords": {
      "positive": [{"text": "tốt", "value": 45}],
      "negative": [{"text": "chậm", "value": 12}]
    },
    "smartAdvice": "✅ Sản phẩm có chất lượng đáng tin cậy...",
    "alternativeProducts": [...]
  }
}
```

---

## Troubleshooting

### "Scraping thất bại"
- Kiểm tra `scraping_agent/` có Playwright chưa: `playwright install chromium`
- URL phải là Shopee/Tiki/Lazada/TGDĐ

### "MobileNetV3 không load được"
- File weights phải tồn tại tại `ai_engine/models/weights/mobilenet_v3_defect.pt` ✅ (đã có)
- Nếu fail, pipeline fallback về label `intact` cho tất cả ảnh (vẫn chạy được)

### "Import ai_engine.xxx failed"
- Đảm bảo chạy uvicorn từ thư mục `ai_engine/` (không phải project root)
- Hoặc: `cd <project_root> && uvicorn ai_engine.main:app --port 8000`

### "LLM Summary fail"
- Kiểm tra `GEMINI_API_KEY` trong root `.env` đã có giá trị chưa ✅
- Nếu fail, pipeline dùng heuristic summary (vẫn hiển thị được)

### "Port 5000 / 3000 bị conflict"
- Backend dùng port 5000 (`.env`: `PORT=5000`)
- Frontend dùng port 3000 (Next.js default)
- AI Engine dùng port 8000

---

## Lưu Ý Về Performance

- Scraping 200 reviews: ~1-3 phút tuỳ site
- Download 40 ảnh: ~30-60 giây
- MobileNetV3 inference 40 ảnh (CPU): ~20-40 giây
- **Tổng cộng**: ~2-5 phút một lần phân tích
- Progress bar cập nhật liên tục qua Socket.IO

> [!TIP]
> Để demo nhanh hơn: giảm `MAX_REVIEWS_SCRAPE=50` và `MAX_IMAGES_PROCESS=10` trong `ai_engine/.env`
