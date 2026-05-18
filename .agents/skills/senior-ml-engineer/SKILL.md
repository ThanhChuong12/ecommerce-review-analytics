---
name: senior-ml-engineer
description: >
  Production-grade ML engineering for this e-commerce review analytics project.
  Use when training spam/sentiment models, building scraping pipelines, deploying
  AI inference endpoints, implementing MLOps workflows, or integrating LLMs/RAG
  into the platform. Covers PyTorch, Scikit-learn, MLflow, feature engineering,
  model monitoring, and automated retraining.
---

# Senior ML Engineer Skill

Skill chuyên biệt cho dự án **ecommerce-review-analytics**: xây dựng và vận hành các
hệ thống ML phân tích review sản phẩm từ Tiki, Lazada, Shopee.

## Project Context

```
ecommerce-review-analytics/
├── ai_engine/                  # Core ML: training, inference, text/image processing
│   ├── text_processing/        # Augmentation, spam detection, sentiment
│   ├── image_processing/       # Defect detection (baseline model đang dev)
│   └── *.py                    # Model training & evaluation scripts
├── scraping_agent/             # Data collection: Tiki / Lazada / Shopee
│   └── scraper/direct/         # Site-specific scrapers (httpx + Playwright)
├── scripts/                    # Training & evaluation entry points
│   ├── train_spam_model.py
│   └── evaluate_models.py
├── web_platform/               # Backend API (Node.js) + frontend
└── docs/                       # Experiment docs & evaluation reports
```

## When to Use This Skill

- Train hoặc evaluate spam/sentiment/image defect models
- Thêm data augmentation pipeline hoặc feature engineering
- Tích hợp model inference vào backend API
- Thiết lập MLflow tracking, model versioning
- Tối ưu scraping pipeline để thu thập training data
- Implement RAG hoặc LLM integration cho review analysis
- Debug model performance, xử lý class imbalance

## Quick Start

```bash
# Train spam detection model
python scripts/train_spam_model.py --config configs/spam.yaml

# Evaluate models
python scripts/evaluate_models.py --model-path models/spam_v2.pkl

# Deploy model pipeline
python scripts/model_deployment_pipeline.py --input data/ --output results/

# Build RAG system
python scripts/rag_system_builder.py --target ai_engine/ --analyze

# Monitor ML metrics
python scripts/ml_monitoring_suite.py --config config.yaml --deploy
```

## Tech Stack

| Category | Tools |
|---|---|
| **ML Frameworks** | Scikit-learn, PyTorch, XGBoost, Transformers |
| **Text Processing** | PhoBERT, underthesea, TextAugmentation |
| **Data** | Pandas, NumPy, Playwright, httpx |
| **Experiment Tracking** | MLflow, Weights & Biases |
| **Deployment** | FastAPI, Docker, Redis (queue) |
| **Monitoring** | Prometheus, custom drift detection |

## Development Workflow

### 1. Data Collection & Validation

- Scrape via `scraping_agent/` → validate schema → check class distribution
- Log data stats (n_samples, label_ratio, source_breakdown) trước khi train

### 2. Feature Engineering & Augmentation

- Text: tokenize, normalize Vietnamese, apply augmentation (`ai_engine/text_processing/augmentation.py`)
- Image: resize, normalize, augment với albumentations (`ai_engine/image_processing/`)
- Luôn fit transform trên train set, transform test set riêng biệt

### 3. Training

- Dùng MLflow để track experiments
- Log: accuracy, F1, confusion matrix, training time
- Lưu model artifact + preprocessing pipeline cùng nhau

### 4. Evaluation

- Báo cáo: accuracy, precision, recall, F1, ROC-AUC
- So sánh với baseline (simple heuristics)
- Test trên real scraped data (out-of-distribution check)

### 5. Deployment

- Expose qua FastAPI endpoint trong `web_platform/backend/`
- Queue inference nặng qua Redis worker (`web_platform/backend/queue/worker.mjs`)
- Health check + fallback khi model unavailable

## ML Engineering Checklist

- [ ] Accuracy target đạt (spam: ≥ 90% F1)
- [ ] Inference latency < 200ms/request
- [ ] Model artifact versioned + reproducible
- [ ] Data leakage không xảy ra (train/val/test split đúng)
- [ ] Class imbalance được xử lý (oversampling / class weights)
- [ ] Experiment logged đầy đủ trong MLflow
- [ ] Model có thể load lại từ artifact (test pickle/joblib)
- [ ] Monitoring alert khi accuracy drop > 5%

## Reference Documentation

| File | Nội dung |
|---|---|
| `references/mlops_production_patterns.md` | MLOps patterns: versioning, CI/CD cho model |
| `references/llm_integration_guide.md` | Tích hợp LLM vào pipeline phân tích review |
| `references/rag_system_architecture.md` | RAG architecture cho Q&A về sản phẩm |

## Decision Tree: Chọn Approach

```
Task mới?
├── Có data labeled?
│   ├── Có → Train supervised model (spam/sentiment)
│   └── Không → Dùng LLM zero-shot / thu thập label trước
│
├── Text hay Image?
│   ├── Text → ai_engine/text_processing/
│   └── Image → ai_engine/image_processing/
│
└── Scale data lớn (>100k)?
    ├── Có → Batch processing qua Redis queue
    └── Không → Inference trực tiếp qua API
```

## Common Pitfalls

- **Vietnamese text**: Dùng `underthesea` để tokenize, không dùng whitespace split
- **Shopee price**: Giá Shopee × 100000, cần chia trước khi dùng làm feature
- **Class imbalance spam**: Thường 80-90% không spam → dùng `class_weight='balanced'`
- **Playwright timeout**: Tăng timeout lên 30s+ cho Lazada/Shopee (anti-bot chậm)
