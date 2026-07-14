# E-Commerce Review Analytics — Multimodal AI System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=next.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-20%2B-339933?style=for-the-badge&logo=node.js&logoColor=white)

![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supabase-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Upstash-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</div>

> **Final Project — Introduction to Artificial Intelligence**
>
> _Faculty of Information Technology, VNU-HCM University of Science_

---

## Table of Contents

- [1. About The Project](#1-about-the-project)
- [2. System Architecture](#2-system-architecture)
- [3. AI Models & Modules](#3-ai-models--modules)
  - [3.1 Spam & Seeding Detection](#31-spam--seeding-detection)
  - [3.2 Sentiment Analysis (PhoBERT)](#32-sentiment-analysis-phobert)
  - [3.3 Image Defect Detection (ResNet50 + MobileNetV3)](#33-image-defect-detection-resnet50--mobilenetv3)
  - [3.4 CLIP Zero-Shot Image Filter](#34-clip-zero-shot-image-filter)
  - [3.5 Cross-Modal Fusion Engine](#35-cross-modal-fusion-engine)
  - [3.6 LLM Summary & Recommendation](#36-llm-summary--recommendation)
- [4. Scraping Pipeline](#4-scraping-pipeline)
- [5. Web Platform](#5-web-platform)
- [6. Repository Structure](#6-repository-structure)
- [7. Getting Started](#7-getting-started)
  - [Prerequisites](#prerequisites)
  - [Run with Docker Compose (Recommended)](#run-with-docker-compose-recommended)
  - [Manual Setup](#manual-setup)
  - [Environment Variables](#environment-variables)
  - [Training & Evaluation](#training--evaluation)
- [8. Contributors](#8-contributors)
- [9. License & Acknowledgments](#9-license--acknowledgments)

---

## 1. About The Project

**E-Commerce Review Analytics** is a full-stack, production-grade **Multimodal AI System** that accepts a product URL from Vietnamese e-commerce platforms (Shopee, Tiki, Lazada, The Gioi Di Dong), automatically scrapes all customer reviews and attached images, and runs them through a multi-stage AI pipeline to produce a comprehensive trust and quality analysis dashboard — delivered to the user in real time.

The system addresses three core problems observed in Vietnamese e-commerce:

| Problem | Solution |
| :--- | :--- |
| **Spam & fake reviews** flood product pages, distorting star ratings | Rule-based 5-axis spam filter (21 flags) + Isolation Forest hybrid model |
| **Sentiment** in Vietnamese is context-dependent and hard to classify | PhoBERT fine-tuned on Vietnamese reviews + multilayer fallback (lexicon → zero-shot → LLM) |
| **Image quality** — reviews contain irrelevant photos (faces, memes) mixed with real product/damage shots | CLIP zero-shot binary filter → ResNet50 / MobileNetV3 4-class defect classifier |

The final output per product is a **Trust Score (0–100)** computed by a cross-modal fusion engine, accompanied by an LLM-generated summary, aspect-based sentiment breakdown, sentiment time-series, keyword cloud, and alternative product recommendations.

**Technology stack at a glance:**

| Layer | Technology |
| :--- | :--- |
| Scraping | Playwright (browser interception) + httpx (direct API) — 3-tier dispatcher |
| Spam Detection | Rule-based filter (21 rules) + Isolation Forest hybrid (`SpamHybridModel`) |
| Sentiment Analysis | `vinai/phobert-base` fine-tuned + TF-IDF Ensemble baseline + LLM fallback |
| Image Classification | ResNet50 + MobileNetV3-Large (Transfer Learning, FocalLoss, SMOTE oversampling) |
| Image Filtering | OpenAI CLIP ViT-B/32 zero-shot (product vs. irrelevant) |
| Feature Denoising | Diffusion-based `FeatureDenoiser` (adapted from MDSBR, RecSys'25) |
| Fusion | `TrustScoreCalculator` — weighted multimodal fusion with Pydantic-typed I/O |
| LLM | Gemini 1.5 Flash / GPT-4o-mini / Grok-2 (cascading provider chain) |
| AI Worker | Python FastAPI (async background tasks) |
| Orchestrator | Node.js (Express) + BullMQ job queue |
| Real-time | Socket.IO (progress streaming to frontend) |
| Database | PostgreSQL via Supabase (Sequelize ORM) |
| Auth | Google OAuth 2.0 via Supabase Auth |
| Frontend | Next.js 16 (App Router) + Tailwind CSS + Recharts + Framer Motion |
| Infra | Docker Compose · Redis (Upstash) · GitHub Actions CI |

---

## 2. System Architecture

The system follows an **event-driven, queue-based microservice** pattern. All heavy AI work runs asynchronously in Python, completely decoupled from the Node.js orchestrator.

```
+------------------+   POST /api/analyze    +------------------------+
|   Next.js 16     | ---------------------->|   Node.js (Express)    |
|   Frontend       | <-- Socket.IO events --|   Orchestrator         |
|                  |                        |   + BullMQ Worker      |
+------------------+                        +-----------+------------+
                                                        | POST /process-job
                                                        v
                                            +------------------------+
                                            |   Redis Queue          |
                                            |   (Upstash)            |
                                            +-----------+------------+
                                                        |
                                                        v
                                            +------------------------+
                                            |   Python FastAPI       |
                                            |   AI Engine            |
                                            |                        |
                                            |   Step 1 - Scraping    |
                                            |   Step 2 - Spam Filter |
                                            |   Step 3 - Sentiment   |
                                            |   Step 4 - CLIP Filter |
                                            |   Step 5 - Defect Det. |
                                            |   Step 6 - Fusion      |
                                            |   Step 7 - LLM Summary |
                                            |   Step 8 - Similar     |
                                            +-----------+------------+
                                                        | Webhook (finished)
                                                        v
                                            +------------------------+
                                            |   PostgreSQL           |
                                            |   (Supabase)           |
                                            |                        |
                                            |   products             |
                                            |   reviews              |
                                            |   reports              |
                                            +------------------------+
```

**Real-time progress flow:** The Python AI engine calls `POST /api/webhook/update-progress` at each step (10% → 25% → 42% → 55% → 63% → 72% → 82% → 90% → 95% → 99%), and Node.js relays these updates to the frontend via Socket.IO room `room-{productId}`.

---

## 3. AI Models & Modules

### 3.1 Spam & Seeding Detection

**Module:** `ai_engine/text_processing/spam_filter.py`

The rule-based spam filter is organized along **five detection axes**, manually calibrated from 1,000 real Vietnamese reviews:

| Axis | Description | Example signals |
| :--- | :--- | :--- |
| **Axis 1 — AI-generated / Templated** | Bot-generated reviews with repeated marketing phrases | Known template phrases in `_TEMPLATE_PHRASES` |
| **Axis 2 — Coin/voucher farming** | Reviews padded with random text to meet minimum character thresholds for reward coins | "hình ảnh chỉ mang tính chất nhận xu" |
| **Axis 3 — Structural noise** | Pure emoji, keyboard mash, character repetition, digit-only text | Emoji ratio > 60%, character entropy < threshold |
| **Axis 4 — Off-topic / contact** | URLs, phone numbers, Zalo/Facebook handles, competitor ads | Phone regex, URL regex, platform handle patterns |
| **Axis 5 — Rating-text mismatch** | 5-star rating with clearly negative sentiment, or vice versa | Negative lexicon match on 4–5 star reviews |

A **cross-review seeding check** uses TF-IDF + cosine similarity with 500-record batching to detect groups of near-identical reviews (organised seeding). The hybrid model (`SpamHybridModel`) combines rule flags into a 31-dimensional feature matrix and trains Isolation Forest on top.

**Public API:**

```python
from ai_engine.text_processing.spam_filter import detect_spam, summarize_spam

df_result = detect_spam(df)            # adds 'is_spam' column
summary   = summarize_spam(df_result)  # per-rule counts & percentages
```

---

### 3.2 Sentiment Analysis (PhoBERT)

**Module:** `ai_engine/models/phobert_model.py` · `ai_engine/text_processing/sentiment_analysis.py`

The sentiment pipeline runs three layers of analysis with a fallback chain:

```
Text Input
    │
    ▼
[Layer 1] PhoBERT fine-tuned (vinai/phobert-base) + MLP Classification Head
    │  if model unavailable or low confidence
    ▼
[Layer 2] Text Ensemble Baseline (TF-IDF + LR + Calibrated SVM + RF, soft-voting)
    │  if text is empty or rating is clear
    ▼
[Layer 3] Rating prior heuristic (≥4 → positive, ≤2 → negative, else neutral)
```

**Aspect extraction** uses a `DeepEmbedder` (dense sentence embeddings) with cosine similarity against four anchor phrases: `shipping`, `product quality`, `price`, `service` — mapped to the display names `Packaging`, `Product`, `Price`, `Shipping` in the frontend dashboard.

**Output per review:** `{positive: float, negative: float, neutral: float}` probability distribution, fed directly into the Fusion Engine.

---

### 3.3 Image Defect Detection (ResNet50 + MobileNetV3)

**Module:** `ai_engine/image_processing/defect_detection.py` · `ai_engine/models/image_baseline.py`

Two transfer learning models are trained on a custom dataset of Vietnamese e-commerce product images, labeled across **4 classes**: `intact`, `damaged`, `wrong_item`, `irrelevant`.

| Model | Architecture | Training strategy | Output |
| :--- | :--- | :--- | :--- |
| **ResNet50** | ImageNet pretrained backbone + MLP head (Linear→BN→ReLU→Dropout→Linear) | FocalLoss (γ=2.0) + class weights; defect oversampling 15× to correct 1:37 imbalance → 1:2.5; early stopping on Defect F1 (patience=5) | 2-class binary: `defect` / `no-defect`, threshold=0.525 |
| **MobileNetV3-Large** | MobileNetV3-Large pretrained + custom head | Same training strategy; 4-class output; ~50 ms/image inference | 4-class: `intact` / `damaged` / `wrong_item` / `irrelevant` |

**Advanced pipeline (MDSBR-adapted):** The production pipeline loads ResNet50 as a backbone (stripping the FC head to extract 2048-dim features), passes them through a diffusion-based `FeatureDenoiser`, and feeds the denoised embeddings into a lightweight MLP classification head — significantly improving robustness to label noise and domain mismatch.

**Image labeling pipeline:** `image_labeling/media_pipeline.py` — a standalone offline tool that downloads, deduplicates (MD5 hash), and builds training manifests from scraped review media.

---

### 3.4 CLIP Zero-Shot Image Filter

**Module:** `ai_engine/image_processing/zero_shot_clip.py`

**Model:** `openai/clip-vit-base-patch32` (ViT-B/32) — selected over ViT-B/16 for superior accuracy on this dataset (68% vs. 60.5%) at lower latency (~100–180 ms/image on CPU).

CLIP is used as a **pre-filter before ResNet50**, performing binary zero-shot classification:

```
Review image
    │
    ▼
CLIP ViT-B/32 zero-shot
    ├── "product"    → [PASS] forward to ResNet50 for defect detection
    └── "irrelevant" → [SKIP] label as irrelevant, bypass ResNet50
```

**Prompt design v4 (HYBRID):**

- **PRODUCT prompts** — broad coverage (product packaging, unboxing photos, actual product items, delivery photos)
- **IRRELEVANT prompts** — targeted on the 35 most-common false-negative categories observed in the dataset (human faces, memes, screenshots, food photos, etc.)

This design eliminates the core bug where faces and memes were classified as `no-defect` by ResNet50 (which only knew `defect` vs. `no-defect` and had no concept of "irrelevant").

---

### 3.5 Cross-Modal Fusion Engine

**Module:** `ai_engine/fusion/fusion_engine.py`

The `TrustScoreCalculator` fuses three typed input signals into a single **Trust Score (0–100)**:

```python
FusionInput(
    text_probs  = TextProbs(positive, negative, neutral),                      # from PhoBERT
    image_probs = ImageProbs(intact, damaged, wrong_item, irrelevant),         # from ResNet50
    auth_meta   = AuthMeta(is_spam, spam_score),                               # from SpamHybridModel
)
```

Fusion logic (simplified):

- **Spam reviews** → hard-penalized to ~9.5/100 regardless of text/image content
- **No image** → text-only trust score with `MISSING_IMAGE` penalty
- **Multimodal conflict** (positive text + damaged image, or vice versa) → `MULTIMODAL_CONFLICT` flag, reduced score
- **High agreement** → score boosted, `MULTIMODAL_OK` reason code

The overall product Trust Score is computed as the **mean of non-spam review scores** (spam reviews are excluded from the aggregate to prevent artificially pulling down legitimate products).

---

### 3.6 LLM Summary & Recommendation

**Module:** `ai_engine/llm_integration/llm_client.py`

A **multi-provider LLM client** with cascading fallback routing:

```
Provider chain: Gemini 1.5 Flash → OpenAI GPT-4o-mini → Grok-2
```

Features:

- **`LLMBudget`** singleton — thread-safe global call cap (default: 300 calls/session) to prevent quota exhaustion during large batch labeling runs
- **`LLMFallbackClient`** — sentiment label prediction for the zero-shot fallback tier
- **`LLMProductSummaryClient`** — Chain-of-Thought product summary generation given statistics and representative reviews
- Produces a final action label: `DUYET` (approve) / `CANH BAO` (caution) / `XOA` (reject) based on Trust Score and sentiment distribution

---

## 4. Scraping Pipeline

**Module:** `scraping_agent/`

The dispatcher routes each URL through three tiers in order of speed and cost:

```
URL Input
    │
    ▼  Tier 1 — Direct API (fastest, no browser, no LLM)
    ├── tiki.vn           → TikiScraper    (Tiki internal API v2, httpx)
    ├── thegioididong.com → TGDDScraper    (webapi.thegioididong.com)
    │
    ▼  Tier 2 — Playwright browser interception (browser, no LLM)
    ├── lazada.vn         → LazadaScraper  (dynamic tokens, Playwright)
    ├── shopee.vn         → ShopeeParallelScraper v6 (CloakBrowser + JS fetch)
    ├── any other site    → GenericPlaywrightScraper (auto-detect review API)
    │
    ▼  Tier 3 — LLM browser agent (slowest, most expensive — last resort)
    └── any site          → scraper/agent.py (browser_use.Agent)
```

**Shopee scraper architecture (v6):**

- **Phase 1 — CloakBrowser warm-up (~10 s):** Open product page, intercept `get_ratings`, extract total and per-star counts, **keep browser open**
- **Phase 2 — Browser JS parallel fetch:** Call Shopee API from inside the browser context using `page.evaluate(fetch())` — bypasses IP bans because requests share the browser's authenticated session; `Promise.allSettled()` batches 5 concurrent requests per evaluate call; fetches each star rating (1–5) independently to bypass Shopee's 3K review cap on `type=0`

**Similar products:** `scraping_agent/similar_products_fetcher.py` — scrapes up to 5 similar products from the same platform to populate the "Alternative Products" recommendation section when Trust Score is low.

---

## 5. Web Platform

### Backend (Node.js / Express)

**Directory:** `web_platform/backend/`

| File / Module | Responsibility |
| :--- | :--- |
| `index.mjs` | Express app entry point; registers all routes; syncs Sequelize models |
| `queue/analysisQueue.mjs` | BullMQ queue definition (`Queue` named queue, Upstash Redis connection) |
| `queue/worker.mjs` | BullMQ worker — picks jobs, updates `Product.status`, POSTs to FastAPI `/process-job` |
| `controllers/analyzeController.mjs` | Creates `Product` record (status=PENDING), enqueues job, returns `productId` immediately |
| `controllers/webhookController.mjs` | Receives AI Engine webhook; bulk-inserts reviews; saves report; emits `finished` via Socket.IO |
| `controllers/historyController.mjs` | CRUD for analysis history; PDF export via Puppeteer |
| `middleware/auth.mjs` | JWT verification via Supabase `auth.getUser()` |
| `config/database.mjs` | Sequelize + PostgreSQL (SSL, Supabase connection string) |
| `config/redis.mjs` | ioredis connection to Upstash (TLS, `maxRetriesPerRequest: null` for BullMQ) |
| `socket.mjs` | Socket.IO server initialization; `join-room` event handler |

**Database schema (Sequelize):**

```
User (id UUID PK, email, name, avatar)
  └── Product (id UUID PK, userId FK, name, url, thumbnail, status ENUM, timestamps)
        ├── Review[] (id UUID PK, product_id FK, review_text, rating, image_path,
        │              label ENUM[intact/damaged/wrong_item/irrelevant], sentiment)
        └── Report  (id UUID PK, product_id FK, summary_text, risk_level, metadata JSON)
```

All foreign keys use `ON DELETE CASCADE`.

### Frontend (Next.js 16)

**Directory:** `web_platform/frontend/`

| Page / Component | Description |
| :--- | :--- |
| `app/page.tsx` | Landing page — animated URL input form with floating icon decorations |
| `app/analyze/page.tsx` | Main analysis dashboard — real-time progress bar, 3 main tabs (Overview / Details / Recommendations), sentiment pie/bar charts, image defect distribution, aspect radar chart, sentiment time-series line chart, word cloud (custom SVG spiral placement algorithm), review data table with filters (sentiment / label / rating / pagination), image gallery lightbox, PDF export |
| `app/history/page.tsx` | Analysis history list — search, status filter, pagination, delete with cascade |
| `components/Header.tsx` | Navigation header — dark/light theme toggle, Google OAuth login/logout, avatar display |

**Key frontend dependencies:** `recharts` (charts), `framer-motion` (animations), `socket.io-client` (real-time), `@supabase/supabase-js` (auth), `lucide-react` (icons).

---

## 6. Repository Structure

```text
ecommerce-review-analytics/
│
├── ai_engine/                          # Python AI Worker (FastAPI)
│   ├── main.py                         # FastAPI entry point — 9-step AI pipeline
│   ├── requirements.txt                # AI engine dependencies
│   ├── Dockerfile
│   │
│   ├── text_processing/
│   │   ├── spam_filter.py              # Rule-based spam detection (5 axes, 21 flags)
│   │   ├── spam_model.py               # SpamHybridModel (Isolation Forest)
│   │   ├── sentiment_analysis.py       # NextGenReviewAnalyzer (lexicon + zero-shot + LLM)
│   │   ├── augmentation.py             # Back-translation data augmentation (GoogleFree / DeepL)
│   │   ├── preprocessor.py             # Vietnamese text normalization pipeline
│   │   ├── embeddings.py               # DeepEmbedder (dense sentence embeddings)
│   │   ├── vectorizers.py              # TF-IDF vectorizer wrappers
│   │   └── config.py                   # Text processing configuration
│   │
│   ├── image_processing/
│   │   ├── defect_detection.py         # ResNet50 + MobileNetV3 (training + inference)
│   │   ├── zero_shot_clip.py           # CLIP ViT-B/32 zero-shot binary filter
│   │   ├── defect_dataloader.py        # ProductDefectDataset (oversampling, augmentation)
│   │   ├── onnx_inference.py           # ONNX runtime inference wrapper
│   │   └── augmentation/               # Albumentations transforms for defect class
│   │
│   ├── models/
│   │   ├── phobert_model.py            # PhoBertSentimentModel (inference wrapper)
│   │   ├── phobert_trainer.py          # PhoBERT fine-tuning trainer
│   │   ├── text_baseline.py            # TextEnsembleModel (LR + SVM + RF, TF-IDF)
│   │   ├── image_baseline.py           # MobileNetV3 4-class model definition
│   │   ├── evaluate_models.py          # Unified evaluation (confusion matrix, F1, ROC-AUC)
│   │   ├── weights/                    # Saved model checkpoints (.pkl, .pth)
│   │   └── results/                    # Evaluation output (PNG heatmaps, JSON)
│   │
│   ├── denoising/
│   │   └── feature_denoiser.py         # Diffusion-based FeatureDenoiser (adapted from MDSBR)
│   │
│   ├── fusion/
│   │   └── fusion_engine.py            # TrustScoreCalculator (Pydantic-typed, cross-modal)
│   │
│   ├── llm_integration/
│   │   └── llm_client.py               # Multi-provider LLM client (Gemini / OpenAI / Grok)
│   │
│   ├── explainability/
│   │   ├── lime_explainer.py           # LIME explanations for text model
│   │   └── phobert_explainer.py        # PhoBERT attention-based explainability
│   │
│   └── api/
│       └── routes.py                   # Additional FastAPI route definitions
│
├── scraping_agent/                     # Multi-tier scraping pipeline
│   ├── main.py                         # CLI entry point for standalone scraping
│   ├── crawl.py                        # Legacy crawler (Playwright)
│   ├── crawl_shopee_bad_reviews.py     # Shopee 1-2 star review harvester
│   ├── similar_products_fetcher.py     # Public API: scrape_similar_products()
│   ├── requirements.txt
│   ├── Dockerfile
│   │
│   └── scraper/
│       ├── dispatcher.py               # 3-tier URL router
│       ├── agent.py                    # LLM browser-use fallback agent
│       ├── exporter.py                 # CSV/JSON review exporter
│       ├── models.py                   # Review dataclass
│       ├── stealth_browser.py          # Anti-detection Playwright wrapper
│       └── direct/
│           ├── base.py                 # BaseScraper abstract class
│           ├── tiki.py                 # Tiki internal API v2
│           ├── shopee_fast.py          # ShopeeParallelScraper v6 (CloakBrowser + JS fetch)
│           ├── lazada.py               # Lazada Playwright interception
│           ├── tgdd.py                 # The Gioi Di Dong API
│           ├── similar_products.py     # Similar products fetcher (Tiki/Lazada/Shopee)
│           └── generic_playwright.py   # Auto-detect review API for unknown sites
│
├── web_platform/
│   ├── backend/                        # Node.js Express orchestrator
│   │   ├── index.mjs                   # App entry point
│   │   ├── socket.mjs                  # Socket.IO server
│   │   ├── package.json
│   │   ├── Dockerfile
│   │   ├── config/
│   │   │   ├── database.mjs            # Sequelize + PostgreSQL (Supabase SSL)
│   │   │   └── redis.mjs               # ioredis → Upstash (BullMQ compatible)
│   │   ├── models/
│   │   │   ├── User.mjs
│   │   │   ├── Product.mjs
│   │   │   ├── Review.mjs
│   │   │   ├── Report.mjs
│   │   │   └── index.mjs               # Associations (hasMany, belongsTo, CASCADE)
│   │   ├── queue/
│   │   │   ├── analysisQueue.mjs       # BullMQ Queue
│   │   │   └── worker.mjs              # BullMQ Worker
│   │   ├── controllers/
│   │   │   ├── analyzeController.mjs
│   │   │   ├── webhookController.mjs
│   │   │   ├── historyController.mjs   # + PDF export via Puppeteer
│   │   │   └── authController.mjs
│   │   ├── routes/
│   │   │   ├── analyzeRoutes.mjs
│   │   │   ├── webhookRoutes.mjs
│   │   │   ├── historyRoutes.mjs
│   │   │   └── authRoutes.mjs
│   │   └── middleware/
│   │       └── auth.mjs                # Supabase JWT verification
│   │
│   └── frontend/                       # Next.js 16 (App Router)
│       ├── src/
│       │   ├── app/
│       │   │   ├── layout.tsx          # Root layout (dark/light theme, Header)
│       │   │   ├── page.tsx            # Landing page (URL input form)
│       │   │   ├── analyze/page.tsx    # Main dashboard (all charts + filters)
│       │   │   └── history/page.tsx    # Analysis history list
│       │   ├── components/
│       │   │   └── Header.tsx          # Navigation (auth + theme toggle)
│       │   └── lib/
│       │       └── supabase.ts         # Supabase client singleton
│       ├── package.json
│       ├── next.config.ts
│       └── Dockerfile
│
├── scripts/                            # Training & Evaluation (35 scripts)
│   ├── train_spam_model.py             # SpamHybridModel training
│   ├── train_phobert.py                # PhoBERT fine-tuning
│   ├── train_defect_model.py           # ResNet50 defect model training
│   ├── train_image_baseline.py         # MobileNetV3 training
│   ├── train_text_baseline.py          # TF-IDF ensemble training
│   ├── train_denoiser.py               # FeatureDenoiser training (MDSBR)
│   ├── evaluate_models.py              # Unified CLI evaluator (text/image/spam/sanity)
│   ├── compare_text_models.py          # PhoBERT vs. baselines comparison
│   ├── tune_spam_model.py              # Hyperparameter tuning for Isolation Forest
│   ├── tune_phobert.py                 # PhoBERT learning rate / batch size tuning
│   ├── prepare_dataset.py              # Dataset preparation & splitting
│   ├── build_paired_dataset.py         # Text-image paired dataset builder
│   ├── extract_embeddings.py           # Offline embedding extraction for training denoiser
│   ├── export_onnx.py                  # Export models to ONNX format
│   └── ...                             # 21 additional training / evaluation / analysis scripts
│
├── image_labeling/                     # Offline image dataset builder
│   └── media_pipeline.py               # Download, deduplicate, manifest, label images
│
├── notebooks/                          # Jupyter notebooks for EDA & experiments
│   ├── 01_text_preprocessing_and_labeling.ipynb
│   ├── 02_text_eda_and_feature_space.ipynb
│   ├── 03_image_preprocessing_and_eda.ipynb
│   ├── 04_rule_based_spam_detection.ipynb
│   ├── 05_data_overview_for_report.ipynb
│   └── error_analysis_resnet50_clip.ipynb
│
├── tests/
│   ├── test_feature_denoiser.py
│   ├── test_fusion_engine.py
│   ├── test_image_baseline.py
│   └── test_llm_client.py
│
├── docs/                               # Technical documentation
│   ├── project_specification.md        # System tech spec & API contract
│   ├── spam_detection_pipeline.md
│   ├── phobert_pipeline.md
│   ├── image_pipeline.md
│   └── nlp.md
│
├── artifacts/                          # Model weights & outputs (gitignored)
│   └── models/
│       ├── mobilenet/                  # mobilenet_v3_model2_defect.pt
│       ├── resnet50/                   # resnet50_defect_gpu_best.pth
│       ├── phobert/                    # Fine-tuned PhoBERT checkpoint
│       ├── baselines/                  # ensemble_smote_auto_weights.pkl
│       └── tuned_spam_iforest/         # tuned_spam_iforest.pkl
│
├── data/                               # Datasets (gitignored)
│   └── processed/                      # Cleaned CSVs (reviews, labels)
│
├── .agents/                            # AI assistant customization
│   └── skills/senior-ml-engineer/
│
├── .github/workflows/ci.yml            # GitHub Actions CI (Node.js + Python checks)
├── docker-compose.yml                  # 3-service compose: ai_engine, web_backend, frontend
├── requirements.txt                    # Root Python dependencies
├── .env.example                        # Environment variable template
└── README.md
```

---

## 7. Getting Started

### Prerequisites

| Requirement | Version |
| :--- | :--- |
| Python | ≥ 3.11 |
| Node.js | ≥ 20 |
| Redis | Upstash (cloud) or local |
| PostgreSQL | Supabase (cloud) or local |
| Docker & Docker Compose | Optional but recommended |

### Run with Docker Compose (Recommended)

The entire system (Frontend, Backend, AI Engine) starts with a single command:

```bash
git clone https://github.com/ThanhChuong12/ecommerce-review-analytics.git
cd ecommerce-review-analytics

cp .env.example .env
cp ai_engine/.env.example ai_engine/.env
cp web_platform/backend/.env.example web_platform/backend/.env
# Fill in all environment variables (see section below)

docker-compose up --build
```

Services will be available at:

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:5000
- **AI Engine:** http://localhost:8000

### Manual Setup

#### 1. Python AI Engine

```bash
# Lightweight mode (no heavy ML models — for UI testing only)
cd ai_engine
pip install fastapi uvicorn requests pydantic python-dotenv

# Full mode (all AI models — requires ~8 GB RAM)
cd ai_engine
pip install -r requirements.txt

# Start the AI engine
uvicorn main:app --reload --port 8000
```

#### 2. Node.js Backend

```bash
cd web_platform/backend
npm install
npm run dev          # starts on port 5000
```

#### 3. Next.js Frontend

```bash
cd web_platform/frontend
npm install
npm run dev          # starts on port 3000
```

### Environment Variables

Copy `.env.example` to `.env` and fill in the values:

```env
# LLM Providers (for AI summary & sentiment fallback)
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
GROK_API_KEY=your_grok_api_key

# LLM Settings
LLM_PROVIDER=gemini           # gemini | openai | grok
GEMINI_MODEL=gemini-1.5-flash
OPENAI_MODEL=gpt-4o-mini

# Backend Connection
PORT=5000
PYTHON_AI_SERVICE_URL=http://localhost:8000

# Hugging Face (for downloading PhoBERT)
HF_TOKEN=your_huggingface_token
```

For `web_platform/backend/.env`, additionally configure:

```env
# Supabase (PostgreSQL + Auth)
DATABASE_URL=postgresql://...
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key

# Upstash Redis (BullMQ)
UPSTASH_REDIS_REST_URL=https://your-redis.upstash.io
UPSTASH_REDIS_REST_TOKEN=your_token
```

For `web_platform/frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:5000/api
NEXT_PUBLIC_SOCKET_URL=http://localhost:5000
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
```

### Training & Evaluation

```bash
# --- Spam Model ---
python scripts/train_spam_model.py --data-path data/processed/reviews.csv
python scripts/tune_spam_model.py

# --- Sentiment Analysis ---
python scripts/train_text_baseline.py --data-path data/processed/reviews.csv
python scripts/train_phobert.py          # GPU recommended (Kaggle / Colab)
python scripts/compare_text_models.py

# --- Image Models ---
python scripts/train_image_baseline.py   # MobileNetV3
python scripts/train_defect_model.py     # ResNet50

# --- Evaluation ---
python scripts/evaluate_models.py sanity
python scripts/evaluate_models.py text  --model-path artifacts/models/baselines/ensemble.pkl
python scripts/evaluate_models.py spam  --data-path data/processed/reviews.csv

# --- Notebooks ---
jupyter notebook notebooks/01_text_preprocessing_and_labeling.ipynb
jupyter notebook notebooks/03_image_preprocessing_and_eda.ipynb
```

---

## 8. Contributors

This project was developed by a team of 5 students from the _Faculty of Information Technology, VNU-HCM University of Science_.

| Contributor | Student ID | Role | Main Contributions |
| :--- | :---: | :--- | :--- |
| **Lê Hà Thanh Chương** | `23120195` | **Project Lead & Full-Stack Engineer** | Managed the overall project timeline, coordinated between all subteams, and authored the technical specification documents. Designed and implemented the complete **Web Platform** — Next.js frontend (analysis dashboard with all charts, word cloud, filters, and real-time Socket.IO integration), Node.js backend (BullMQ queue orchestration, Sequelize models, webhook handlers, PDF export), and the FastAPI AI engine main pipeline (`main.py`) that connects all AI modules end-to-end. |
| **Trà Văn Sỹ** | `23120197` | **NLP Engineer** | Designed and implemented the full **Text AI pipeline**: PhoBERT fine-tuning pipeline (`train_phobert.py`, `phobert_trainer.py`), TF-IDF ensemble baseline model (Logistic Regression + Calibrated SVM + Random Forest with soft-voting), text data augmentation via back-translation, and Vietnamese text preprocessing & normalization. Ran all NLP training experiments and authored the NLP technical documentation. |
| **Huỳnh Đức Thịnh** | `23120199` | **Scraping Agent & Computer Vision Engineer** | Built the complete **3-tier scraping pipeline** (`dispatcher.py`, `TikiScraper`, `ShopeeParallelScraper v6` with CloakBrowser JS-fetch architecture, `LazadaScraper`, `GenericPlaywrightScraper`) and the similar products fetcher. Developed and trained the **MobileNetV3-Large** 4-class defect detection model — including the custom dataset loader (`ProductDefectDataset`) with 15× defect oversampling, FocalLoss training, early stopping on Defect F1, and the full inference API. |
| **Bùi Trung Hiếu** | `23120257` | **Spam & ML Research Engineer** | Designed and implemented the **Spam Detection system** — the 5-axis rule-based filter (`spam_filter.py`, 21 flags, seeding detection via TF-IDF cosine similarity), `SpamHybridModel` (31-dimensional feature matrix + Isolation Forest), and the unified evaluation framework (`evaluate_models.py` with confusion matrix, Macro F1, ROC-AUC). Also contributed to data labeling strategy and authored the spam detection technical documentation. |
| **Lê Công Phúc** | `23120330` | **Computer Vision & Fusion Engineer** | Led the **Image AI subsystem**: built and trained the ResNet50 defect model with MLP head and FocalLoss, implemented the CLIP ViT-B/32 zero-shot binary filter (`zero_shot_clip.py`) with v4 prompt engineering, and designed the diffusion-based `FeatureDenoiser` (adapted from MDSBR RecSys'25). Implemented the **Cross-Modal Fusion Engine** (`TrustScoreCalculator`) and the multi-provider LLM client with cascading fallback for product summary generation. |

---

## 9. License & Acknowledgments

### Academic Acknowledgments

This project is the **Final Project** for the _Introduction to Artificial Intelligence_ course at _VNU-HCM University of Science_.

The team sincerely thanks our instructor for guidance throughout the project.

### Pre-trained Models & External Resources

- **PhoBERT:** _PhoBERT: Pre-trained language models for Vietnamese_ — Dat Quoc Nguyen & Anh Tuan Nguyen. EMNLP 2020. ([GitHub](https://github.com/VinAIResearch/PhoBERT))

- **CLIP:** _Learning Transferable Visual Models From Natural Language Supervision_ — Radford et al. ICML 2021. ([GitHub](https://github.com/openai/CLIP))

- **ResNet:** _Deep Residual Learning for Image Recognition_ — He et al. CVPR 2016.

- **MobileNetV3:** _Searching for MobileNetV3_ — Howard et al. ICCV 2019.

- **MDSBR (Feature Denoiser):** _MDSBR: Multimodal Denoising for Session-based Recommendation_ — RecSys 2025. ([GitHub](https://github.com/YutongLi2024/MDSBR))

- **Isolation Forest:** _Isolation Forest_ — Liu, Ting & Zhou. ICDM 2008.

### Frameworks & Tools

[FastAPI](https://fastapi.tiangolo.com/) · [Next.js](https://nextjs.org/) · [PyTorch](https://pytorch.org/) · [Transformers (HuggingFace)](https://huggingface.co/docs/transformers) · [BullMQ](https://docs.bullmq.io/) · [Socket.IO](https://socket.io/) · [Supabase](https://supabase.com/) · [Playwright](https://playwright.dev/) · [Recharts](https://recharts.org/) · [Framer Motion](https://www.framer.com/motion/) · [underthesea](https://github.com/undertheseanlp/underthesea)

### License

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

The source code of this project is distributed under the **MIT License**.
See the `LICENSE` file for full details.

<br>
<p align="center">
  <i>Built by the AI Team | University of Science, VNU-HCM | 2026</i>
</p>

