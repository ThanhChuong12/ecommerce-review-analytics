# E-commerce Review Scraper

Cào đánh giá sản phẩm từ các sàn TMĐT Việt Nam: **Tiki**, **Thế Giới Di Động**, **Lazada**, và bất kỳ trang nào khác (via LLM agent).

---

## Cài đặt

```bash
# 1. Cài uv — package manager nhanh hơn pip, dùng để tạo venv và cài thư viện
pip install uv

# 2. Tạo môi trường ảo Python 3.11 (tránh xung đột với các project khác)
uv venv --python 3.11

# 3. Kích hoạt môi trường ảo (mọi lệnh sau chạy trong môi trường này)
.venv\Scripts\activate        # Windows

# 4. Cài các thư viện trong requirements.txt:
#    - browser-use  : thư viện LLM agent điều khiển browser (Approach 2)
#    - playwright   : engine tự động hóa trình duyệt, dùng cho Lazada + agent
#    - httpx        : HTTP client async, dùng gọi API Tiki / TGDD (Approach 1)
#    - pydantic     : validate và serialize dữ liệu review
#    - python-dotenv: đọc API keys từ file .env
uv pip install -r requirements.txt

# 5. Tải Chromium về máy để Playwright có thể mở browser
#    (browser-use và LazadaScraper đều dùng Chromium qua Playwright)
uvx browser-use install
```

---

## Cấu hình `.env`

Tạo file `.env` ở thư mục gốc, điền ít nhất một API key:

```env
GOOGLE_API_KEY=...        # Gemini 2.0 Flash (khuyến nghị)
GROQ_API_KEY=...          # Llama-4 Scout (miễn phí, rate-limited)
OPENAI_API_KEY=...        # GPT-4.1-mini
BROWSER_USE_API_KEY=...   # ChatBrowserUse (nhanh nhất)
```

---

## Sử dụng

```bash
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

---

## Kiến trúc

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

---

## Cấu trúc thư mục

```
scraping_agent/
├── main.py               # CLI entry point
├── requirements.txt
├── .env                  # API keys (không commit)
├── scraper/
│   ├── dispatcher.py     # Routing URL → scraper đúng
│   ├── models.py         # Pydantic model: Review
│   ├── exporter.py       # Ghi CSV / JSON incremental
│   ├── agent.py          # browser-use Agent cho trang bất kỳ
│   └── direct/
│       ├── base.py       # BaseScraper
│       ├── tiki.py       # TikiScraper
│       ├── tgdd.py       # TGDDScraper
│       └── lazada.py     # LazadaScraper (Playwright)
└── output/               # File kết quả (tự động tạo, không commit)
```

---

## Output

Mỗi review được lưu với các cột:

| Cột | Mô tả |
|---|---|
| `text` | Nội dung đánh giá |
| `rating` | Số sao (1–5) |
| `date` | Ngày đăng |
| `image_urls` | URL ảnh đính kèm (nhiều URL cách nhau bằng `\|`) |
| `product_url` | URL sản phẩm gốc |
| `scraped_at` | Thời điểm cào (ISO 8601) |
