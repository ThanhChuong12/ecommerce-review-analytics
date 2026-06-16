# Kế Hoạch Tích Hợp Real AI Engine Vào Web Platform

## Tổng Quan

Project đã có đủ các thành phần hoạt động: scraping agent, text processing (spam + sentiment), image processing (MobileNetV3 + CLIP), fusion engine, LLM integration, và web platform (Next.js frontend + Node.js backend). Vấn đề hiện tại là `ai_engine/main.py` hoàn toàn dùng **mock data hardcoded**, chưa gọi các module AI thực sự.

**Pipeline hiện tại (mock):**
```
Frontend → Node.js (BullMQ queue) → Python FastAPI (/process-job)
→ [sleep + random data] → webhook gửi về Node.js → Socket.IO → Frontend
```

**Pipeline sau khi fix (real AI):**
```
Frontend → Node.js → Python FastAPI (/process-job)
→ [Scraper → Spam Filter → Sentiment → CLIP → MobileNetV3 → Fusion → LLM Summary]
→ webhook → Node.js DB → Socket.IO → Frontend
```

---

## Kiến Trúc Hiện Tại — Phân Tích

### Vị Trí Scraping Agent
Hiện tại `scraping_agent/` là module độc lập (CLI). `ai_engine/main.py` đã thử import `similar_products_fetcher` từ đó. Ta sẽ **giữ nguyên vị trí** nhưng import trực tiếp vào `ai_engine/main.py` — không cần copy vào `ai_engine/` vì:
1. Scraper dùng Playwright và có môi trường `.venv` riêng ở `scraping_agent/`
2. `ai_engine/main.py` đã có code làm điều này (`sys.path.insert`)
3. Tách biệt concerns: scraping ≠ AI inference

### Các Model Weights Có Sẵn
- `ai_engine/models/weights/mobilenet_v3_defect.pt` ✅ (14MB, đã train)
- `ai_engine/models/resnet50_defect_gpu_best.pth` — **cần kiểm tra**

---

## Proposed Changes

### 1. `ai_engine/main.py` — CORE CHANGE (thay toàn bộ mock bằng real pipeline)

Thay thế hàm `heavy_ai_process()` bằng real pipeline:

```
Step 1 (15%): Scrape reviews từ URL → scraping_agent/scraper/dispatcher.py
Step 2 (35%): Spam Detection → text_processing/spam_filter.detect_spam()
Step 3 (55%): Sentiment Analysis → text_processing/sentiment_analysis.assign_heuristic_label()
Step 4 (70%): Tải ảnh review xuống temp → download image URLs
Step 5 (80%): Image Classification → CLIP filter + MobileNetV3 detect_defect_mobilenet_batch()
Step 6 (90%): Fusion → fusion/fusion_engine.TrustScoreCalculator.calculate()
Step 7 (95%): LLM Summary → llm_client.LLMRecommendationClient
Step 8 (100%): Similar Products + send webhook
```

**Schema data gửi về Node.js (đảm bảo khớp với DB + Frontend):**
```json
{
  "productId": int,
  "productData": {"name": str, "thumbnail": str},
  "reviews": [
    {
      "review_text": str,
      "rating": int,
      "image_path": str | null,
      "label": "intact"|"damaged"|"wrong_item"|"irrelevant",
      "sentiment": "positive"|"neutral"|"negative",
      "user_name": str,
      "date": str
    }
  ],
  "summary": str,
  "metadata": {
    "spamPercentage": int,
    "trustScore": float,
    "aspectSentiment": {"Product": float, "Packaging": float, "Shipping": float},
    "sentimentTimeSeries": [...],
    "keywords": {"positive": [...], "negative": [...]},
    "smartAdvice": str,
    "alternativeProducts": [...]
  }
}
```

### 2. `ai_engine/main.py` — Thêm `/health` endpoint
Để web có thể kiểm tra AI engine sẵn sàng chưa.

### 3. `ai_engine/main.py` — Xử Lý Graceful Fallback
- Nếu MobileNetV3 weights không load được → fallback rule-based label
- Nếu CLIP fail → skip filter, giữ tất cả ảnh
- Nếu scraping timeout/fail → return lỗi rõ ràng về Node.js

### 4. `ai_engine/.env` — Cập nhật biến môi trường
Thêm các key còn thiếu: `GEMINI_API_KEY`, `NODE_WEBHOOK_PROGRESS`, `NODE_WEBHOOK_FINISHED`

### 5. Scraping Integration
Scraper xuất ra CSV với columns: `text`, `rating`, `image_urls`, `product_name`, `date`

Ta cần:
- Download ảnh từ `image_urls` (list) về temp folder
- Map sang format `image_path` (string path)
- Với reviews không có ảnh: `image_path = null`, `label = null`

### 6. Sentiment Mapping
Scraper/AI dùng: `"tích cực"`, `"tiêu cực"`, `"trung lập"`  
Frontend/DB expect: `"positive"`, `"negative"`, `"neutral"`

→ Cần mapping trong `main.py`

### 7. Keyword Extraction (từ reviews thực)
Thay mock keywords bằng extract thực từ reviews dùng `Counter` + `text_processing/preprocessor.py`

### 8. Time Series (từ dates thực)
Nhóm reviews theo ngày (từ cột `date` trong scraped data) → sentiment per day

### 9. `web_platform/backend/.env` — Không cần sửa
Backend đã config đúng: `PYTHON_API_URL=http://localhost:8000`

### 10. `web_platform/frontend/.env.local` — Không cần sửa
Frontend đã config đúng: API URL + Socket URL

---

## Vấn Đề Kỹ Thuật Cần Xử Lý

### A. Import Path cho ai_engine khi chạy từ thư mục `ai_engine/`
`sentiment_analysis.py` import: `from ai_engine.llm_integration.llm_client import ...`  
→ Khi chạy `uvicorn` từ `ai_engine/`, path `ai_engine.*` sẽ fail  
→ Fix: thêm `sys.path` hoặc chạy uvicorn từ project root

### B. Ảnh từ Scraper là URLs, không phải local files
CLIP và MobileNetV3 cần local file paths  
→ Download ảnh về `tempfile` trước khi inference  
→ Sau khi inference xong: upload lên Supabase Storage hoặc giữ URL gốc  

**Quyết định**: Giữ nguyên URL gốc từ Shopee/Tiki vì:
1. Frontend render ảnh qua URL trực tiếp (đã hoạt động với mock)
2. Không cần storage riêng
3. Chỉ download tạm về local để chạy model, sau đó dùng URL gốc cho DB

### C. Performance
Inference 50-200 ảnh có thể mất 5-30 phút → cần progress report liên tục

### D. Sentiment labels mapping
| AI Output | DB/Frontend |
|-----------|-------------|
| tích cực | positive |
| tiêu cực | negative |
| trung lập | neutral |

---

## Verification Plan

### Sau khi sửa xong:
1. Khởi động Python AI Engine: `cd ai_engine && uvicorn main:app --reload --port 8000`
2. Khởi động Node.js Backend: `cd web_platform/backend && node index.mjs`
3. Khởi động Next.js Frontend: `cd web_platform/frontend && npm run dev`
4. Dán URL Shopee/Tiki → nhấn Phân Tích
5. Kiểm tra progress bar cập nhật
6. Kiểm tra kết quả hiển thị đúng (reviews thực, sentiment thực, labels thực)

### Test Cases:
- URL Shopee hợp lệ → reviews thực
- URL không hợp lệ → error message rõ ràng
- Sản phẩm có ảnh → image grid hiển thị
- Sản phẩm không có ảnh → hiển thị "Không có ảnh"

---

## Open Questions

> [!IMPORTANT]
> **API Keys cho AI Engine**: `ai_engine/.env` hiện chỉ có `NODE_WEBHOOK_URL`. Cần biết các key này đã có chưa:
> - `GEMINI_API_KEY` — dùng cho LLM summary
> - `GROQ_API_KEY` — fallback
> Nếu không có, LLM summary sẽ fallback về heuristic (vẫn chạy được).

> [!IMPORTANT]  
> **ResNet weights**: File `ai_engine/models/resnet50_defect_gpu_best.pth` có tồn tại không? Chỉ có MobileNetV3 weights được confirm. Nếu không có ResNet → dùng MobileNetV3 (đã có, 14MB).

> [!NOTE]
> **Scraping Agent dependencies**: `scraping_agent/.venv` có Playwright cài chưa? Nếu chưa cần `playwright install chromium`. 
> MobileNetV3 inference cần: `torch`, `torchvision`, `Pillow`, `albumentations` — đã trong `ai_engine/requirements.txt`.

> [!NOTE]
> **Timeout**: Scraping 3000 reviews có thể mất 5-15 phút. Đề xuất giới hạn `max_reviews=200` cho web demo (có thể config qua env).
