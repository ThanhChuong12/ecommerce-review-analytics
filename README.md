# Multimodal Review Analytics Dashboard

> Hệ thống phân tích đánh giá sản phẩm đa phương thức (văn bản + hình ảnh) từ các sàn thương mại điện tử Việt Nam.

**Nhóm thực hiện:** Thanh Chương · Văn Sỹ · Đức Thịnh · Trung Hiếu · Công Phúc

---

## Tổng quan hệ thống

```
scraping_agent  ──►  data/raw  ──►  backend_ai  ──►  data/processed  ──►  web_app
    (Crawler)          (Raw)       (AI Pipeline)       (Analyzed)        (Dashboard)
```

| Module | Công nghệ | Người phụ trách |
|---|---|---|
| `scraping_agent` | Playwright, httpx, browser-use | — |
| `backend_ai/text_processing` | Scikit-learn, PhoBERT, Transformers | — |
| `backend_ai/image_processing` | CLIP, ResNet, MobileNet | — |
| `backend_ai/llm_integration` | Gemini API, OpenAI API | — |
| `web_app/frontend` | React.js, Plotly | — |
| `web_app/backend_server` | Node.js, Express.js | — |

---

## Cấu trúc thư mục

```
multimodal-review-analytics/
├── data/
│   ├── raw/                    # Dữ liệu thô từ scraper và Kaggle
│   └── processed/              # Dữ liệu sau khi lọc spam và xử lý ảnh
│
├── scraping_agent/             # Module thu thập dữ liệu tự động
│   ├── main.py                 # CLI entry point
│   ├── requirements.txt
│   ├── scraper/
│   │   ├── dispatcher.py       # Routing URL → đúng scraper
│   │   ├── agent.py            # LLM Agent (browser-use) cho Shopee, v.v.
│   │   ├── exporter.py         # Ghi CSV/JSON incremental
│   │   ├── models.py           # Pydantic model: Review
│   │   └── direct/             # Scraper trực tiếp (không cần LLM)
│   │       ├── base.py         # BaseScraper (retry, dedup, pagination)
│   │       ├── tiki.py         # TikiScraper
│   │       ├── tgdd.py         # TGDDScraper
│   │       └── lazada.py       # LazadaScraper (Playwright)
│   └── output/                 # Kết quả cào (không commit)
│
├── backend_ai/                 # AI Pipeline (Python)
│   ├── text_processing/
│   │   ├── spam_filter.py      # Isolation Forest + SVM
│   │   └── sentiment_analysis.py  # TF-IDF vs PhoBERT
│   ├── image_processing/
│   │   ├── zero_shot_clip.py   # Phát hiện ảnh rác bằng CLIP
│   │   └── defect_detection.py # Nhận diện hàng lỗi (ResNet/MobileNet)
│   ├── llm_integration/
│   │   └── llm_client.py       # Gemini/GPT API: tổng hợp insight
│   └── models/                 # File trọng số mô hình đã train
│
├── web_app/                    # Dashboard React + API Server
│   ├── frontend/               # React.js + Plotly
│   └── backend_server/         # Node.js/Express API Gateway
│
├── notebooks/                  # Jupyter Notebooks thử nghiệm & so sánh mô hình
├── docs/                       # Đề cương, báo cáo PDF
├── .env.example                # Template cấu hình API keys
├── .gitignore
└── requirements.txt            # Thư viện Python cho backend_ai
```

---

## Cài đặt

### 1. Backend AI (Python)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Scraping Agent
```bash
cd scraping_agent

# Cài uv — package manager nhanh hơn pip
pip install uv

# Tạo môi trường ảo Python 3.11
uv venv --python 3.11
.venv\Scripts\activate          # Windows

# Cài thư viện:
#   browser-use  : LLM agent điều khiển browser (Shopee, v.v.)
#   playwright   : engine tự động hóa trình duyệt
#   httpx        : HTTP client async cho Tiki / TGDD API
#   pydantic     : validate và serialize dữ liệu review
#   python-dotenv: đọc API keys từ file .env
uv pip install -r requirements.txt

# Tải Chromium cho Playwright & browser-use
uvx browser-use install
```

### 3. Web App
```bash
# Frontend
cd web_app/frontend
npm install && npm run dev

# Backend Server
cd web_app/backend_server
npm install && npm run dev
```

---

## Cấu hình `.env`

Sao chép `.env.example` thành `.env` và điền API keys:

```bash
cp .env.example .env   # Linux/Mac
copy .env.example .env  # Windows
```

```env
GOOGLE_API_KEY=...        # Gemini 2.0 Flash (khuyến nghị)
GROQ_API_KEY=...          # Llama-4 Scout (miễn phí, rate-limited)
OPENAI_API_KEY=...        # GPT-4.1-mini
BROWSER_USE_API_KEY=...   # ChatBrowserUse (nhanh nhất)
```

> ⚠️ **KHÔNG commit file `.env` lên GitHub.** File `.gitignore` đã cấu hình sẵn.

---

## Sử dụng Scraping Agent

```bash
cd scraping_agent

# Tiki / TGDD — scraper trực tiếp qua API, không cần LLM
python main.py "https://tiki.vn/..."
python main.py "https://www.thegioididong.com/..."

# Lazada — Playwright network interception, không cần LLM
python main.py "https://www.lazada.vn/..."

# Bất kỳ trang nào khác — dùng LLM agent (browser-use)
python main.py "https://shopee.vn/..." --llm google
python main.py "https://shopee.vn/..." --llm groq

# Tuỳ chỉnh
python main.py "URL" --max-reviews 5000 --format json --output myfile.csv
python main.py "URL" --headless
python main.py --help
```

### Các tham số CLI

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `url` | *(bắt buộc)* | URL sản phẩm cần cào |
| `--output`, `-o` | `reviews_YYYYMMDD_HHMMSS.csv` | Đường dẫn file đầu ra |
| `--format`, `-f` | `csv` | Định dạng: `csv` hoặc `json` |
| `--max-reviews`, `-n` | `3000` | Số review tối đa cần lấy |
| `--llm` | `auto` | LLM provider: `auto` \| `browseruse` \| `openai` \| `google` \| `groq` |
| `--headless` | `False` | Chạy browser ẩn (không khuyến nghị với Shopee) |

### Kiến trúc Scraping Agent

```
URL
 │
 ▼
Dispatcher (scraper/dispatcher.py)
 │
 ├── tiki.vn           ──► TikiScraper    (httpx, Tiki internal API v2)
 │
 ├── thegioididong.com ──► TGDDScraper    (httpx, webapi.thegioididong.com)
 │
 ├── lazada.vn         ──► LazadaScraper  (Playwright, network interception)
 │
 └── bất kỳ trang nào  ──► scrape_reviews (browser-use Agent + LLM)
                                           (Playwright + LLM, --llm chọn provider)
 │
 ▼
ReviewExporter (scraper/exporter.py)
 └── Ghi incremental sau mỗi trang → không mất dữ liệu nếu crash
```

### Schema output

Mỗi review được lưu với các cột:

| Cột | Mô tả |
|---|---|
| `text` | Nội dung đánh giá |
| `rating` | Số sao (1–5) |
| `date` | Ngày đăng |
| `image_urls` | URL ảnh đính kèm (nhiều URL cách nhau bằng `\|`) |
| `product_url` | URL sản phẩm gốc |
| `scraped_at` | Thời điểm cào (ISO 8601) |

---

## Ghi chú phát triển

- **Thử nghiệm mô hình**: Lưu notebook và bảng so sánh Accuracy/F1 vào `notebooks/`
- **Môi trường**: Dùng `venv` hoặc `Conda` để tránh xung đột thư viện
- **API Keys**: Quản lý tập trung qua `.env`, không hardcode trong source code
- **Dữ liệu thô**: Không commit vào Git — dùng `.gitignore` cho `data/raw/` và `data/processed/`
