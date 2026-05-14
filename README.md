# Multimodal Review Analytics Dashboard

> Hệ thống phân tích đánh giá sản phẩm đa phương thức (văn bản + hình ảnh) từ các sàn thương mại điện tử Việt Nam.

**Nhóm thực hiện:** Thanh Chương · Văn Sỹ · Đức Thịnh · Trung Hiếu · Công Phúc

---

## Tổng quan hệ thống

```
scraping_agent  ──►  data/raw  ──►  ai_engine   ──►  data/processed  ──►  web_app
    (Crawler)          (Raw)       (AI Pipeline)       (Analyzed)        (Dashboard)
```

| Module | Công nghệ | Người phụ trách |
|---|---|---|
| `scraping_agent` | Playwright, httpx, browser-use | — |
| `ai_engine/text_processing` | Scikit-learn, PhoBERT, Transformers | — |
| `ai_engine/image_processing` | CLIP, ResNet, MobileNet, Albumentations | — |
| `ai_engine/llm_integration` | Gemini API, OpenAI API | — |
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
├── ai_engine/                  # AI Pipeline (Python)
│   ├── text_processing/
│   │   ├── spam_filter.py      # Isolation Forest + SVM
│   │   └── sentiment_analysis.py  # TF-IDF vs PhoBERT
│   ├── image_processing/
│   │   ├── zero_shot_clip.py   # Phát hiện ảnh rác bằng CLIP
│   │   ├── defect_detection.py # Nhận diện hàng lỗi (ResNet/MobileNet)
│   │   └── augmentation/       # Data augmentation cho defect detection
│   │       └── transforms.py   # Albumentations pipeline
│   ├── llm_integration/
│   │   └── llm_client.py       # Gemini/GPT API: tổng hợp insight
│   └── models/                 # File trọng số mô hình đã train
│
├── web_app/                    # Dashboard React + API Server
│   ├── frontend/               # React.js + Plotly
│   └── backend_server/         # Node.js/Express API Gateway
│
├── notebooks/                  # Jupyter Notebooks thử nghiệm & so sánh mô hình
├── scripts/                    # Scripts tiện ích
│   └── validate_augmentation.py # Script kiểm tra augmentation trực quan
├── docs/                       # Đề cương, báo cáo PDF
├── .env.example                # Template cấu hình API keys
├── .gitignore
└── requirements.txt            # Thư viện Python cho ai_engine
```

---

## Cài đặt

### 1. AI Engine (Python)
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

## Image Augmentation Pipeline

Hệ thống cung cấp module Image Augmentation tại `ai_engine/image_processing/augmentation/transforms.py` sử dụng thư viện `Albumentations` để xử lý mất cân bằng dữ liệu cho lớp `defect` (chỉ định cấu hình riêng biệt cho 2 class, sau đó tích hợp vào `PyTorch Dataset` và `DataLoader`).

- Pipeline cho `defect`: áp dụng biến đổi vừa phải (Flip, Rotate, GaussNoise, Brightness/Contrast) để giữ đặc trưng dị tật, kết thúc bằng Resize(224, 224) và ImageNet Normalize.
- Pipeline cho `no-defect`: không thay đổi (chỉ Resize và Normalize).

**Tạo ảnh mẫu để kiểm tra trực quan:**
```bash
python scripts/validate_augmentation.py
```
Kết quả ảnh được sinh ra sẽ lưu ở `data/processed/aug_samples/`.

### Chuẩn bị dữ liệu training

Chuyển ảnh đã gán nhãn từ `image_labeling/` sang cấu trúc thư mục training:

```bash
python scripts/prepare_dataset.py
```

Mapping nhãn: `damaged` → `defect/`, `intact` → `no-defect/`, `irrelevant` & `wrong_item` → bỏ qua.

### Huấn luyện mô hình ResNet50

```bash
# Chạy mặc định (10 epochs, batch_size=16, lr=1e-4)
python scripts/train_defect_model.py

# Tùy chỉnh
python scripts/train_defect_model.py --epochs 20 --batch-size 32 --lr 0.0001
```

Mô hình tốt nhất sẽ được lưu tại `ai_engine/models/resnet50_defect.pth`.

---

## Text Sentiment Analysis — Weighted Soft-Voting Ensemble

> Full technical reference: [`docs/TEXT_MODEL.md`](docs/TEXT_MODEL.md)

### Architecture

```
Raw Text
  │
  ▼
TF-IDF (max_features=15 000, ngram=(1,2), sublinear_tf=True)
  │
  ├─ [optional] SMOTE over-sampling
  │
  ▼
Soft-Voting Ensemble
  ├── LogisticRegression          (weight: w₁ — auto-computed from macro-F1 CV)
  ├── CalibratedClassifierCV      (weight: w₂ — wraps LinearSVC for predict_proba)
  └── RandomForestClassifier      (weight: w₃ — cognitive diversity via tree-based model)
  │
  ▼
Predicted Sentiment Label
```

### Base Estimators

| Estimator | Class | Imbalance Strategy |
|---|---|---|
| LR | `LogisticRegression` | `class_weight='balanced'` |
| SVM | `CalibratedClassifierCV(LinearSVC)` | `class_weight='balanced'` + isotonic calibration |
| RF | `RandomForestClassifier` | `class_weight='balanced'` |

### Automatic Weight Computation

When `weights=None`, the model runs **5-fold cross-validation** on each base estimator independently
over TF-IDF features, uses the resulting **macro-F1 scores as ensemble weights**, and then trains
the full ensemble. Models that generalise better receive proportionally larger votes.

### Quick Start

```bash
# Install ML dependencies
pip install -r ai_engine/requirements.txt

# Run all 4 benchmark experiments (SMOTE × Weights)
python ai_engine/scripts/train_text_baseline.py
```

Artifacts are saved to `artifacts/models/`:

```
artifacts/models/
├── ensemble_no_smote_auto_weights.pkl    # EXP-1 — primary
├── ensemble_smote_auto_weights.pkl       # EXP-2 — primary (SMOTE)
├── ensemble_no_smote_equal_weights.pkl   # EXP-3 — control
└── ensemble_smote_equal_weights.pkl      # EXP-4 — control (SMOTE)
```

### Python API

```python
from ai_engine.models.text_baseline import TextEnsembleModel

model = TextEnsembleModel(use_smote=False)   # weights computed automatically
model.fit(X_train, y_train)
labels = model.predict(X_test)
probas = model.predict_proba(X_test)

model.save("artifacts/models/my_ensemble.pkl")
loaded = TextEnsembleModel.load("artifacts/models/my_ensemble.pkl")
```

---

## Ghi chú phát triển

- **Thử nghiệm mô hình**: Lưu notebook và bảng so sánh Accuracy/F1 vào `notebooks/`
- **Môi trường**: Dùng `venv` hoặc `Conda` để tránh xung đột thư viện
- **API Keys**: Quản lý tập trung qua `.env`, không hardcode trong source code
- **Dữ liệu thô**: Không commit vào Git — dùng `.gitignore` cho `data/raw/` và `data/processed/`
