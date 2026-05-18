---
name: senior-ml-engineer
description: >
  Production-grade ML engineering for ecommerce-review-analytics.
  Use when: train/evaluate spam/sentiment/defect models, scraping pipeline,
  MLOps, LLM integration, data augmentation, or deploy AI endpoints.
  This skill auto-updates plan.md after each completed task.
---

# Senior ML Engineer — Ecommerce Review Analytics

## MANDATORY WORKFLOW — Read before doing anything

```
STEP 1: Read plan.md
        └── Identify which task the user is requesting
        └── Check dependencies (what must be done first?)

STEP 2: Execute the task
        └── Follow patterns in this file
        └── Do not skip checklist items

STEP 3: Run quality gate
        └── python .agents/skills/senior-ml-engineer/scripts/quality_gate.py --task <name>
        └── If FAIL → auto-improve and re-run

STEP 4: Update plan.md
        └── Change [ ] → [x] for completed task
        └── Record actual metrics
        └── Write improvement suggestions

STEP 5: Report to user (in Vietnamese)
        └── Summarize what was done, results
        └── Highlight next improvement opportunities
```

> CRITICAL: After every task, MUST update plan.md and run quality gate. No exceptions.

---

## Project Structure

```
ecommerce-review-analytics/
├── ai_engine/
│   ├── text_processing/
│   │   ├── spam_filter.py        ← 21 rule flags
│   │   ├── augmentation.py       ← Back-translation vi→en→vi
│   │   ├── sentiment_analysis.py ← Lexicon + Zero-shot + LLM fallback
│   │   ├── preprocessor.py
│   │   ├── embeddings.py
│   │   └── vectorizers.py
│   ├── image_processing/
│   │   ├── defect_detection.py   ← ResNet50 + MobileNetV3
│   │   └── augmentation/
│   ├── llm_integration/
│   │   └── llm_client.py         ← Gemini + OpenAI fallback
│   └── models/                   ← .pkl, .pth artifacts
├── scraping_agent/
│   ├── scraper/dispatcher.py     ← 3-tier: Direct→Playwright→LLM
│   └── similar_products_fetcher.py
├── scripts/
│   ├── train_spam_model.py
│   ├── train_defect_model.py
│   ├── evaluate_models.py
│   └── prepare_dataset.py
├── data/processed/
└── .agents/skills/senior-ml-engineer/
    ├── SKILL.md
    ├── plan.md                   ← Task tracker (update after each task)
    ├── scripts/
    │   ├── quality_gate.py
    │   ├── auto_train_pipeline.py
    │   └── deploy_model.py
    └── references/
```

---

## Quality Targets

| Model | Metric | Pass | Fail |
|-------|--------|------|------|
| Spam detection | F1 macro | ≥ 0.90 | < 0.80 |
| Sentiment | F1 macro | ≥ 0.82 | < 0.70 |
| Defect detection | F1 macro | ≥ 0.85 | < 0.75 |
| Inference | ms/req | < 200 | > 500 |
| Scraping | success% | > 90% | < 70% |

---

## Tech Stack

| Category | Tools |
|---|---|
| ML | Scikit-learn, PyTorch, XGBoost, Transformers |
| Vietnamese NLP | underthesea, PhoBERT (vinai/phobert-base) |
| Augmentation | Back-translation (googletrans/deepl), Albumentations |
| Tracking | MLflow, joblib |
| Deployment | FastAPI:8000, Redis+BullMQ |
| Scraping | Playwright, httpx, LLM fallback |

---

## Task Workflows

### Train / Retrain Model

```bash
# 1. Check data balance
python scripts/prepare_dataset.py --check-balance

# 2. Augment if imbalance > 3x
python ai_engine/text_processing/augmentation.py \
    --data-path data/processed/reviews_labeled.csv \
    --multiply 2 --output-path data/processed/reviews_augmented.csv

# 3. Train
python scripts/train_spam_model.py \
    --data-path data/processed/reviews_augmented.csv \
    --contamination 0.1 \
    --save-path ai_engine/models/spam_iforest.pkl

# 4. Evaluate
python scripts/evaluate_models.py spam \
    --model-path ai_engine/models/spam_iforest.pkl \
    --data-path data/processed/reviews_test.csv

# 5. Quality gate
python .agents/skills/senior-ml-engineer/scripts/quality_gate.py \
    --task spam_detection --model-path ai_engine/models/spam_iforest.pkl
```

### Deploy Model

```bash
python .agents/skills/senior-ml-engineer/scripts/deploy_model.py \
    --model-path ai_engine/models/spam_iforest.pkl --model-type spam

curl -X POST http://localhost:8000/analyze-text \
    -H "Content-Type: application/json" \
    -d '{"text": "San pham tot", "rating": 5}'
```

---

## Common Pitfalls

| Pitfall | Correct | Wrong |
|---------|---------|-------|
| Vietnamese tokenize | `underthesea.word_tokenize(text)` | `text.split()` |
| Shopee price | `price // 100_000` if > 10B | raw price |
| Train/val split | `StratifiedShuffleSplit` | `train_test_split` |
| Class imbalance | `class_weight='balanced'` | default |
| CSV encoding | `encoding='utf-8-sig'` (Windows) | `utf-8` |

---

## plan.md Update Format (after each task)

```markdown
### [x] Task name

**Completed:** DD/MM/YYYY HH:MM
**Results:**
- F1 Score: 0.921 (target ≥ 0.90) ✅
- Latency: 145ms (target < 200ms) ✅

**Notes:** Brief description of issues and solutions.

**Improvement suggestions:**
- [ ] Fine-tune PhoBERT to push F1 to 0.95+
- [ ] Cache model to reduce latency below 50ms
```

---

## Auto-Improvement Loop (when quality gate FAILS)

```
1. Diagnose:
   - Low F1? → Check class imbalance → Increase augmentation multiply
   - High latency? → Add model cache / batch inference
   - Scrape fail? → Check anti-bot / add retry with backoff

2. Auto-fix if possible:
   - Increase n_estimators (Isolation Forest)
   - Increase augmentation multiply
   - Add request delay for scraper

3. Retrain and re-evaluate

4. If still FAIL after 3 attempts → Report to user with explanation
```

---

## Quick Reference

```bash
# Full spam pipeline
python scripts/train_spam_model.py \
    --data-path data/processed/reviews.csv \
    --output-csv data/processed/reviews_spam_labeled.csv

# Sanity check all models
python scripts/evaluate_models.py sanity

# Augmentation demo
python ai_engine/text_processing/augmentation.py --demo

# Quality gate all
python .agents/skills/senior-ml-engineer/scripts/quality_gate.py --task all

# Deploy all models
python .agents/skills/senior-ml-engineer/scripts/deploy_model.py --all
```
