# 🗺️ Roadmap: Các giai đoạn sau Image Labeling

> **Context:** Dataset ảnh đã được label xong (~7,988 ảnh, 4 class: `intact / damaged / wrong_item / irrelevant`).  
> Workflow tuân theo: `.agents/skills/senior-ml-engineer/SKILL.md` và `plan.md`.  
> Cập nhật: 05/06/2026

---

## 📊 Trạng thái hiện tại (sau Image Labeling)

| Component | Trạng thái | Ghi chú |
|---|---|---|
| Scraping (Shopee/Lazada/Tiki/TGDD) | ✅ Done | 162 rev/s Shopee |
| Image Labeling (Gemini Vision) | ✅ Done | ~7,988 ảnh có label |
| Spam Detection (Rule + IsolationForest) | ✅ Done | Cần fix circular eval |
| Sentiment Analysis | 🔄 Partial | xlm-roberta hoạt động, chưa cache |
| **Defect Detection (ResNet50)** | ❌ Chưa train | `train_defect_model.py` chưa chạy với data thật |
| FastAPI `main.py` | ❌ Mock | Toàn bộ là fake data |
| Web Platform | 🔄 Partial | Backend/Frontend có, chưa kết nối AI thật |

---

## 🔴 GIAI ĐOẠN 1 — Train Image Defect Detection Model (P1 — Critical)

> **Mục tiêu:** Có model `resnet50_defect.pth` chạy được inference thật.  
> **Target quality gate:** F1 macro ≥ 0.85, Latency < 500ms/image (CPU)

### 1.1 Chuẩn bị dataset từ labeled images

Dataset đã label nằm ở `image_labeling/data/manifests/labels.csv`:

```
review_id, product_url, product_name, review_text, rating, date, source_url, image_path, label
```

**Mapping strategy** (Binary — recommended cho MVP):

```python
label_map = {
    "intact":     "no-defect",   # 0
    "damaged":    "defect",      # 1  — hàng hỏng
    "wrong_item": "defect",      # 1  — giao sai cũng là vấn đề
    "irrelevant": None,          # Bỏ qua — ảnh spam, không dùng train
}
```

**Lệnh tổ chức dataset:**

```bash
python scripts/prepare_dataset.py \
    --labels-csv image_labeling/data/manifests/labels.csv \
    --images-root image_labeling \
    --output-dir data/image_dataset \
    --label-map binary
```

Cấu trúc thư mục sau khi chạy:

```
data/image_dataset/
├── defect/        ← damaged + wrong_item (~10-20%)
│   └── *.jpg
└── no-defect/     ← intact (~70-80%)
    └── *.jpg
```

### 1.2 Kiểm tra class balance

```bash
python scripts/prepare_dataset.py --check-balance --data-dir data/image_dataset
```

Dự kiến phân phối (~7,988 ảnh từ labels.csv):

| Label gốc | Mapping | Tỷ lệ ước tính |
|---|---|---|
| `intact` | `no-defect` | ~70-80% |
| `damaged` | `defect` | ~5-10% |
| `wrong_item` | `defect` | ~5-10% |
| `irrelevant` | bỏ qua | ~10-15% |

> ⚠️ **Nếu imbalance > 10:1** → tăng `--oversample-defect` lên 20-30x.

### 1.3 Train model

```bash
python scripts/train_defect_model.py \
    --data-dir data/image_dataset \
    --batch-size 32 \
    --epochs 30 \
    --lr 1e-4 \
    --freeze-backbone True \
    --oversample-defect 15 \
    --dropout-rate 0.5 \
    --save-path ai_engine/models/resnet50_defect.pth
```

**Kiến trúc model** (`ai_engine/image_processing/defect_detection.py`):

| Thành phần | Chi tiết |
|---|---|
| Backbone | ResNet50, pretrained ImageNet, **frozen** |
| Classification Head | `Linear(2048→512) → BN → ReLU → Dropout(0.5) → Linear(512→2)` |
| Loss | `FocalLoss(gamma=2.0)` — focus vào hard samples (defect bị nhầm) |
| Oversampling | 15x cho class defect (bù imbalance) |
| Early stopping | Theo Defect F1, patience=5 |

**Checkpoint format đầu ra:**

```python
{
    "epoch": int,
    "model_state_dict": dict,
    "defect_f1": float,
    "defect_recall": float,
    "val_loss": float,
}
```

### 1.4 Tune threshold

```bash
python scripts/tune_threshold.py \
    --model-path ai_engine/models/resnet50_defect.pth \
    --data-dir data/image_dataset \
    --output-path artifacts/metrics/threshold_tuning.json
```

Kết quả tham khảo từ code hiện tại:

| Threshold | F1 | Recall | Precision | FP |
|---|---|---|---|---|
| 0.20 | 0.431 | 0.737 | 0.304 | 32 |
| **0.45** | **0.619** | **0.684** | **0.565** | **10** ← Default |
| 0.50 | 0.615 | 0.632 | 0.600 | 8 |

→ Lưu threshold vào `.env`: `DEFECT_THRESHOLD=0.45`

### 1.5 Evaluate & Quality Gate

```bash
# Evaluate
python scripts/evaluate_models.py image \
    --model-path ai_engine/models/resnet50_defect.pth \
    --data-dir data/image_dataset

# Quality Gate
python .agents/skills/senior-ml-engineer/scripts/quality_gate.py \
    --task defect_detection \
    --model-path ai_engine/models/resnet50_defect.pth
```

**Acceptance criteria:**

- ✅ F1 macro ≥ 0.85
- ✅ Latency < 500ms/image trên CPU
- ✅ `detect_defect_resnet("test.jpg")` → `{"label": "defect", "confidence": 0.82, "defect_probability": 0.82, "threshold_used": 0.45, "model_path": "..."}`

### 1.6 Update plan.md

Sau khi pass quality gate, cập nhật `plan.md`:

```markdown
### [x] [IMAGE] Implement production inference for Defect Detection

**Completed:** DD/MM/2026 HH:MM
**Results:**
- Defect F1: 0.87 (target ≥ 0.85) ✅
- Latency: 380ms/image CPU (target < 500ms) ✅
- Threshold tuned: 0.45 (best F1 + Recall balance)
```

---

## 🟡 GIAI ĐOẠN 2 — Hoàn thiện Text Pipeline (P2 — Important)

### 2.1 Add Singleton Cache cho Sentiment Analyzer

**File:** `ai_engine/text_processing/sentiment_analysis.py`  
**Vấn đề:** `xlm-roberta-large-xnli` (1.2GB) load lại mỗi lần request → latency ~2s

**Fix:**

```python
# Thêm vào cuối file sentiment_analysis.py

_analyzer_instance: NextGenReviewAnalyzer | None = None

def get_analyzer() -> NextGenReviewAnalyzer:
    """Singleton — load model một lần duy nhất, tránh reload mỗi request."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = NextGenReviewAnalyzer()
    return _analyzer_instance
```

**Kết quả sau fix:**

| | Trước fix | Sau fix |
|---|---|---|
| Lần đầu tiên | ~2s | ~2s (load model) |
| Từ lần 2 trở đi | ~2s | ~50ms ✅ |

### 2.2 Fix Circular Evaluation cho Spam Model

**Vấn đề:** `final_spam` (được sinh bởi chính model) đang dùng làm ground-truth → metrics vô nghĩa.

**Steps:**

```bash
# 1. Export 300-500 review cần label thủ công
python -c "
import pandas as pd
df = pd.read_csv('data/processed/processed_labeled_all.csv')
sample = df.sample(n=400, random_state=42)[['text', 'rating', 'final_spam']]
sample['is_spam_manual'] = ''  # cột cần điền tay
sample.to_csv('data/processed/reviews_manual_label_task.csv', index=False)
print('File tạo xong. Mở và điền cột is_spam_manual (0/1).')
"

# 2. Sau khi label xong → re-evaluate
python scripts/evaluate_models.py spam \
    --model-path ai_engine/models/spam_iforest.pkl \
    --data-path data/processed/reviews_manual_labeled.csv \
    --ground-truth-col is_spam_manual
```

**Acceptance criteria:**

- ✅ ≥ 300 review được label thủ công
- ✅ Spam F1 (real ground-truth) ≥ 0.90

### 2.3 Fine-tune PhoBERT cho Sentiment (P3 — Optional)

**Mục tiêu:** Thay `xlm-roberta-large-xnli` (1.2GB, ~2s/req) bằng `vinai/phobert-base` fine-tuned (~350MB, <200ms).

**Data đã sẵn có** (từ main branch):

```
data/processed/processed_labeled_text_train.csv  ← 17,705 rows
data/processed/processed_labeled_text_val.csv    ← 3,795 rows
data/processed/processed_labeled_text_test.csv   ← 3,795 rows
```

**Train:**

```bash
python scripts/train_phobert.py \
    --train-path data/processed/processed_labeled_text_train.csv \
    --val-path data/processed/processed_labeled_text_val.csv \
    --output-dir ai_engine/models/weights/phobert_sentiment \
    --epochs 5 \
    --batch-size 16 \
    --lr 2e-5
```

**Inference wrapper đã có** (`ai_engine/models/phobert_model.py`):

```python
from ai_engine.models.phobert_model import PhoBertSentimentModel

model = PhoBertSentimentModel("ai_engine/models/weights/phobert_sentiment")
label = model.predict("Sản phẩm rất tốt, giao hàng nhanh")  # → "tích cực"
probs = model.predict_proba(["text 1", "text 2"])             # → np.ndarray (2, 3)
```

**Quality target:** F1 ≥ 0.82, latency < 200ms.

---

## 🟡 GIAI ĐOẠN 3 — Hardening Backend (P2 — Important)

### 3.1 Add Timeout & Stall Detection cho BullMQ Worker

**File:** `web_platform/backend/queue/worker.mjs`

**Vấn đề hiện tại:** Nếu FastAPI treo → job không bao giờ fail → queue bị block vĩnh viễn.

**Fix:**

```javascript
// Thêm timeout cho HTTP call sang Python
await axios.post(PYTHON_AI_URL, jobData, { 
    timeout: 300_000  // 5 phút — đủ cho scrape + AI processing
});

// Thêm stall detection vào Worker config
const worker = new Worker('AnalysisQueue', processorFn, {
    connection: redisConnection,
    settings: {
        maxStalledCount: 2,           // tối đa 2 lần stall rồi fail
        stalledInterval: 30_000,      // check stall mỗi 30 giây
    },
    defaultJobOptions: {
        attempts: 2,                  // retry 2 lần khi fail
        backoff: { type: 'fixed', delay: 5000 },
    },
});
```

### 3.2 Add Rate Limiting trên /api/analyze

**File:** `web_platform/backend/routes/analyzeRoutes.mjs`

```bash
cd web_platform/backend
npm install express-rate-limit
```

```javascript
import rateLimit from 'express-rate-limit';

const analyzeLimiter = rateLimit({
    windowMs: 60 * 1000,   // window 1 phút
    max: 5,                 // tối đa 5 requests/phút/IP
    standardHeaders: true,
    legacyHeaders: false,
    message: { 
        error: 'Quá nhiều yêu cầu. Vui lòng thử lại sau 1 phút.' 
    },
});

router.post('/analyze', analyzeLimiter, analyzeController);
```

---

## 🔴 GIAI ĐOẠN 4 — Integrate Real AI Pipeline vào FastAPI (Sprint 2 — P1)

**File:** `ai_engine/main.py` (hiện tại 167 dòng, toàn bộ `heavy_ai_process` là mock)

### 4.1 Flow thật

```
POST /process-job
     │
     ├─ [10%] Scrape URL → reviews_raw (scraping_agent/scraper/dispatcher.py)
     ├─ [30%] Spam filter → df_clean (ai_engine/text_processing/spam_filter.py)
     ├─ [55%] Sentiment analysis → label + aspects (sentiment_analysis.py)
     ├─ [75%] Defect detection → label + confidence/image (defect_detection.py)
     ├─ [90%] LLM CoT recommendation → DUYỆT/CẢNH BÁO/XÓA (llm_client.py)
     └─ [100%] Webhook → Node.js → Socket.io → Frontend
```

### 4.2 `heavy_ai_process()` sau khi rewrite

```python
async def heavy_ai_process(product_id: int, url: str) -> None:
    """Real AI pipeline — thay toàn bộ mock data."""

    # ── STEP 1: Scrape ────────────────────────────────────────
    _report_progress(product_id, 10, "Đang cào dữ liệu...")
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scraping_agent"))
    from scraper.dispatcher import scrape
    reviews_raw = await scrape(url=url, max_reviews=200, fmt="dict")

    # ── STEP 2: Spam filter ───────────────────────────────────
    _report_progress(product_id, 30, "Đang lọc spam...")
    import pandas as pd
    from ai_engine.text_processing.spam_filter import detect_spam
    df = pd.DataFrame(reviews_raw)
    df = detect_spam(df)
    clean_df = df[df["is_spam"] == 0].reset_index(drop=True)
    spam_pct = int(len(df[df["is_spam"] == 1]) / len(df) * 100) if len(df) else 0

    # ── STEP 3: Sentiment ─────────────────────────────────────
    _report_progress(product_id, 55, "Đang phân tích cảm xúc...")
    from ai_engine.text_processing.sentiment_analysis import get_analyzer
    analyzer = get_analyzer()
    sentiments = []
    for _, row in clean_df.iterrows():
        result = analyzer.analyze_review(
            text=str(row.get("text", "")),
            rating=int(row.get("rating", 0)) or None,
        )
        sentiments.append(result["sentiment"])
    clean_df["sentiment"] = sentiments

    # ── STEP 4: Defect detection ──────────────────────────────
    _report_progress(product_id, 75, "Đang kiểm tra ảnh sản phẩm...")
    from ai_engine.image_processing.defect_detection import detect_defect_resnet
    for _, row in clean_df.iterrows():
        for img_path in (row.get("local_image_paths") or []):
            try:
                defect_result = detect_defect_resnet(img_path)
                # defect_result = {"label": "defect", "confidence": 0.82, ...}
            except (FileNotFoundError, ValueError):
                pass

    # ── STEP 5: LLM CoT summary ───────────────────────────────
    _report_progress(product_id, 90, "Đang tổng hợp báo cáo...")
    from ai_engine.llm_integration.llm_client import LLMRecommendationClient
    llm = LLMRecommendationClient()
    fusion_result = {
        "final_score": 100 - spam_pct,
        "is_conflict": False,
        "reason_code": "SPAM_RATE",
    }
    sample_text = clean_df["text"].iloc[0] if len(clean_df) else ""
    recommendation = llm.analyze_review(sample_text, fusion_result)

    # ── STEP 6: Webhook ───────────────────────────────────────
    _report_progress(product_id, 100, "Hoàn tất!")
    requests.post(WEBHOOK_FINISHED, json={
        "productId": product_id,
        "productData": {"name": ..., "thumbnail": ...},
        "reviews": clean_df.to_dict(orient="records"),
        "summary": recommendation.get("step_2", ""),
        "metadata": {
            "spamPercentage": spam_pct,
            "trustScore": fusion_result["final_score"],
            "recommendation": recommendation.get("recommendation_action", "CẢNH BÁO"),
            # ... aspect sentiment, keywords, time-series ...
        },
    }, timeout=120)
```

---

## 🟢 GIAI ĐOẠN 5 — MLflow Tracking & Docker Deployment (Sprint 2)

### 5.1 Add MLflow tracking cho Image Model

```bash
pip install mlflow
```

Thêm vào `scripts/train_defect_model.py`:

```python
import mlflow
import mlflow.pytorch

with mlflow.start_run(run_name="resnet50_defect_v2"):
    mlflow.log_params({
        "backbone": "resnet50",
        "freeze_backbone": True,
        "oversample_defect": oversample_defect,
        "focal_loss_gamma": 2.0,
        "dropout_rate": dropout_rate,
        "lr": lr,
        "epochs": epochs,
    })
    # ... training loop ...
    mlflow.log_metrics({
        "best_defect_f1":     best_f1,
        "best_defect_recall": best_recall,
        "best_val_loss":      best_val_loss,
    })
    mlflow.pytorch.log_model(model, artifact_path="defect_model")

# Xem kết quả
# mlflow ui --port 5001
```

### 5.2 Complete Docker Compose

Cập nhật `docker-compose.yml` với volume cho model weights:

```yaml
version: "3.9"

services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    volumes:
      - redis_data:/data

  ai-engine:
    build: ./ai_engine
    ports: ["8000:8000"]
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - GROQ_API_KEY=${GROQ_API_KEY}
      - RESNET_WEIGHTS_PATH=/app/models/resnet50_defect.pth
      - DEFECT_THRESHOLD=0.45
    volumes:
      - ./ai_engine/models:/app/models:ro  # read-only — không cho container ghi đè

  backend:
    build: ./web_platform/backend
    ports: ["5000:5000"]
    environment:
      - REDIS_URL=redis://redis:6379
      - PYTHON_AI_URL=http://ai-engine:8000
    depends_on:
      - redis
      - ai-engine

  frontend:
    build: ./web_platform/frontend
    ports: ["3000:3000"]
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:5000
    depends_on:
      - backend

volumes:
  redis_data:
```

---

## 📋 Checklist tổng thể (đồng bộ với `plan.md`)

```
SPRINT 1 — Data & AI Foundation
├── [x] Scraping pipeline (Shopee/Lazada/Tiki/TGDD)       ← Done 23/05/2026
├── [x] Image Labeling (Gemini Vision, ~7,988 ảnh)         ← Done
├── [x] Spam Detection (Rule-based + IsolationForest)       ← Done
├── [x] Text Augmentation (back-translation)                ← Done
├── [x] Evaluation Framework                                ← Done
│
├── [ ] ★ Train Image Defect Model (ResNet50)              ← NEXT STEP
├── [ ]   Tune threshold + quality gate defect model
├── [ ] Fix circular evaluation Spam model                  ← P1
├── [ ] Sentiment Analyzer singleton cache                  ← P2
├── [ ] PhoBERT fine-tune sentiment                        ← P3 optional
├── [ ] BullMQ timeout + stall detection                   ← P2
└── [ ] Rate limiting /api/analyze                         ← P2

SPRINT 2 — Integration & Deployment
├── [ ] Replace FastAPI mock → real AI pipeline            ← P1
├── [ ] MLflow tracking cho tất cả models                  ← P2
└── [ ] Docker Compose production-ready                    ← P2
```

---

## ⏱️ Ước tính thời gian

| Giai đoạn | Công việc chính | Thời gian ước tính |
|---|---|---|
| 1. Image Defect Train | Chuẩn bị data + train + tune | 2–4 giờ (GPU) / 6–8 giờ (CPU) |
| 2. Text Pipeline | Cache singleton + eval fix | 2–3 giờ |
| 3. Backend Hardening | Timeout + rate limit | 1 giờ |
| 4. FastAPI Integration | Swap mock → real pipeline | 4–6 giờ |
| 5. MLflow + Docker | Tracking + finalize deployment | 2–3 giờ |
| **Tổng** | | **~11–20 giờ** |

---

## 🚦 Quality Gates (từ `SKILL.md`)

| Model | Metric | ✅ Pass | ❌ Fail |
|---|---|---|---|
| Spam detection | F1 macro | ≥ 0.90 | < 0.80 |
| Sentiment analysis | F1 macro | ≥ 0.82 | < 0.70 |
| Defect detection | F1 macro | ≥ 0.85 | < 0.75 |
| Inference | ms/request | < 200ms | > 500ms |
| Scraping | success rate | > 90% | < 70% |

---

> **▶ Bước tiếp theo ngay bây giờ:**  
> ```bash
> # Bước 1: Tổ chức ảnh từ labels.csv
> python scripts/prepare_dataset.py \
>     --labels-csv image_labeling/data/manifests/labels.csv \
>     --images-root image_labeling \
>     --output-dir data/image_dataset \
>     --label-map binary
>
> # Bước 2: Train ResNet50
> python scripts/train_defect_model.py \
>     --data-dir data/image_dataset \
>     --epochs 30 --oversample-defect 15
> ```
