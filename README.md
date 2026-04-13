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
│   ├── scraper/
│   │   ├── dispatcher.py       # Routing URL → đúng scraper
│   │   ├── agent.py            # LLM Agent (browser-use) cho Shopee, v.v.
│   │   ├── exporter.py         # Ghi CSV/JSON incremental
│   │   ├── models.py           # Pydantic model: Review
│   │   └── direct/             # Scraper trực tiếp (không cần LLM)
│   │       ├── tiki.py
│   │       ├── tgdd.py
│   │       └── lazada.py
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

## Cài đặt nhanh

### 1. Backend AI (Python)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Scraping Agent
```bash
cd scraping_agent
pip install uv
uv venv --python 3.11
uv pip install -r requirements.txt
uvx browser-use install         # Tải Chromium cho Playwright
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
cp .env.example .env
```

> ⚠️ **KHÔNG commit file `.env` lên GitHub.** File `.gitignore` đã cấu hình sẵn.

---

## Ghi chú phát triển

- **Thử nghiệm mô hình**: Lưu notebook và bảng so sánh Accuracy/F1 vào `notebooks/`
- **Môi trường**: Dùng `venv` hoặc `Conda` để tránh xung đột thư viện
- **API Keys**: Quản lý tập trung qua `.env`, không hardcode trong source code
- **Dữ liệu thô**: Không commit vào Git — dùng `.gitignore` cho `data/raw/` và `data/processed/`
