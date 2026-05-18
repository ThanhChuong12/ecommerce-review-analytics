# Tài liệu kỹ thuật — Tuần 1 (11/05 – 17/05): Data & AI Tasks

> **Thành viên:** Hiệu | **Lĩnh vực:** Data & AI  
> **Cập nhật lần cuối:** 16/05/2025

---

## Tổng quan 3 Task

| # | Task | File chính | Trạng thái |
|---|------|-----------|-----------|
| 1 | Evaluation Framework (Confusion Matrix, Macro F1, ROC-AUC) | `scripts/evaluate_models.py` | ✅ Hoàn thành |
| 2 | Huấn luyện mô hình phát hiện Spam/Seeding (Rule-based + Isolation Forest) | `scripts/train_spam_model.py` | ✅ Hoàn thành |
| 3 | Text Data Augmentation bằng Back-translation | `ai_engine/text_processing/augmentation.py` | ✅ Hoàn thành |

---

## Task 1 — Evaluation Framework

### Mô tả
Xây dựng **khung đánh giá mô hình chuẩn** (`scripts/evaluate_models.py`) phục vụ việc đánh giá tất cả các mô hình AI trong dự án một cách nhất quán và tái sử dụng được.

### Các chỉ số được tính toán

| Chỉ số | Mô tả | Chiến lược |
|--------|-------|-----------|
| **Confusion Matrix** | Ma trận nhầm lẫn, hiển thị bằng DataFrame + heatmap PNG | Sklearn `confusion_matrix` |
| **Macro F1-Score** | F1 trung bình không trọng số qua các lớp (phù hợp với dữ liệu mất cân bằng) | `f1_score(average="macro")` |
| **ROC-AUC** | Diện tích dưới đường cong ROC | One-vs-Rest (OvR), hỗ trợ cả nhị phân và đa lớp |

### Kiến trúc

```
scripts/evaluate_models.py
├── SECTION 1 — Core metric functions
│   ├── compute_confusion_matrix()    # numpy array + DataFrame
│   ├── compute_macro_f1()            # điểm tổng + report per-class
│   └── compute_roc_auc()             # macro AUC + dict AUC từng lớp
│
├── SECTION 2 — Report printer
│   ├── print_full_report()           # in báo cáo đầy đủ, trả về dict kết quả
│   └── _save_confusion_matrix_plot() # heatmap PNG (matplotlib + seaborn)
│
├── SECTION 3 — Text model evaluator
│   └── evaluate_text_model()         # load .pkl, predict, in report
│
├── SECTION 4 — Image model evaluator
│   └── evaluate_image_model()        # load .pth, chạy validation loader, in report
│
├── SECTION 4b — Spam model evaluator
│   └── evaluate_spam_model()         # load .pkl SpamHybridModel, in report
│
├── SECTION 5 — Sanity check
│   └── run_sanity_check()            # demo trên dữ liệu ngẫu nhiên
│
└── SECTION 6 — CLI
    └── main()                        # sub-commands: text | image | spam | sanity
```

### Cách sử dụng

```powershell
# Kiểm tra framework hoạt động đúng
py scripts/evaluate_models.py sanity

# Đánh giá mô hình Text Ensemble (sentiment)
py scripts/evaluate_models.py text `
    --model-path ai_engine/models/ensemble_no_smote.pkl `
    --data-path data/processed/reviews_labeled.csv `
    --text-col cleaned_text `
    --label-col sentiment

# Đánh giá mô hình ResNet50 (defect detection)
py scripts/evaluate_models.py image `
    --model-path ai_engine/models/resnet50_defect.pth `
    --data-path data/processed `
    --batch-size 16

# Đánh giá SpamHybridModel (cần CSV có cột ground-truth spam)
py scripts/evaluate_models.py spam `
    --model-path ai_engine/models/spam_iforest.pkl `
    --data-path data/processed/reviews_spam_labeled.csv `
    --label-col final_spam

# Không lưu biểu đồ
py scripts/evaluate_models.py text --model-path ... --data-path ... --no-plot
```

### Output mẫu (sanity check)

```
======================================================================
  EVALUATION REPORT — [SANITY] BINARY MODEL
======================================================================

[1] CONFUSION MATRIX
           Predicted: 0  Predicted: 1
Actual: 0            51            48
Actual: 1            57            44

[2] CLASSIFICATION REPORT (Macro F1-Score = 0.4744)
              precision  recall  f1-score  support
0                0.4722  0.5152    0.4928  99.0000
1                0.4783  0.4356    0.4560 101.0000

[3] ROC-AUC (One-vs-Rest, Macro = 0.4894)
    0                    AUC = N/A (negative class in binary task)
    1                    AUC = 0.4894
```

### Dependency
- `scikit-learn` (đã có trong `requirements.txt`)
- `matplotlib`, `seaborn` (tùy chọn, để lưu biểu đồ)

---

## Task 2 — Spam/Seeding Detection Model (Rule-based + Isolation Forest)

### Mô tả
Huấn luyện mô hình phát hiện spam và seeding bằng chiến lược **Hybrid** kết hợp:
1. **Rule-based detection** (từ `spam_filter.py` — 21 rule flags)
2. **Isolation Forest** — phát hiện bất thường dựa trên đặc trưng cấu trúc văn bản

### Lý do dùng Isolation Forest

Rule-based tốt với spam rõ ràng (coin farming, template AI...) nhưng bỏ sót các loại seeding tinh vi hơn (review ngắn, không vi phạm rule đơn lẻ nhưng có pattern bất thường). Isolation Forest học phân phối tổng thể và phát hiện outlier mà không cần nhãn spam.

### Kiến trúc

```
scripts/train_spam_model.py
├── SECTION 1 — Feature extraction
│   ├── extract_structural_features()   # 9 đặc trưng cấu trúc văn bản
│   └── build_feature_matrix()          # ghép 21 rule flags + 1 aggregate + 9 structural
│
├── SECTION 2 — Model (SpamHybridModel class)
│   ├── fit()                           # huấn luyện IsolationForest + StandardScaler
│   ├── predict_anomaly()               # 1 = bình thường, -1 = bất thường
│   ├── anomaly_score()                 # score liên tục (càng âm càng bất thường)
│   ├── predict_final_spam()            # union của rule_spam và iforest_spam
│   ├── save() / load()                 # joblib serialization
│
├── SECTION 3 — Report printer
│   └── print_spam_report()             # breakdown per-rule
│
└── SECTION 4 — CLI
    └── main()
```

### Feature Matrix (31 features tổng)

| Nhóm | Features | Số chiều |
|------|---------|---------|
| **Rule flags** | 21 binary flags từ `spam_filter.py` | 21 |
| **Rule aggregate** | Tổng số rule bị vi phạm (0–21) | 1 |
| **Structural** | word_count, char_count, emoji_ratio, special_char_ratio, uppercase_ratio, digit_ratio, type_token_ratio, avg_word_len, rating_norm | 9 |

### Final Label Logic

```
final_spam = rule_based_spam  OR  iforest_anomaly
```

Nghĩa là: **1 review bị đánh dấu spam nếu vi phạm ÍT NHẤT 1 trong 2 điều kiện**.

### Cách sử dụng

```powershell
# Chạy trên CSV reviews (cần cột 'text' và 'rating')
py scripts/train_spam_model.py `
    --data-path data/processed/reviews.csv `
    --contamination 0.1 `
    --save-path ai_engine/models/spam_iforest.pkl `
    --output-csv data/processed/reviews_spam_labeled.csv

# Nếu cột tên khác
py scripts/train_spam_model.py `
    --data-path data/processed/reviews.csv `
    --text-col review_text `
    --rating-col star `
    --contamination 0.15
```

### Tham số quan trọng

| Tham số | Mặc định | Ghi chú |
|---------|---------|--------|
| `--contamination` | `0.1` | Tỉ lệ spam ước tính trong dataset. Chỉnh lên nếu dataset nhiều spam |
| `--n-estimators` | `200` | Số cây Isolation Forest. Tăng để chính xác hơn |
| `--dup-threshold` | `0.85` | Ngưỡng cosine similarity để phát hiện duplicate seeding |

### Output mẫu

```
=================================================================
  SPAM DETECTION REPORT
=================================================================
  Total reviews        :   10,000
  Rule-based spam      :    1,234  (12.3%)
  IForest anomalies    :    1,456  (14.6%)
  Final spam (union)   :    1,891  (18.9%)
  Clean reviews        :    8,109  (81.1%)

  Rule                           Count      %
  ------------------------------ --------  ------
  ai_template                      456      4.6%
  duplicate_seeding                312      3.1%
  xu_farming                       198      2.0%
  ...
```

### Dependency
- `scikit-learn` — IsolationForest, StandardScaler
- `joblib` — model serialization

---

## Task 3 — Text Data Augmentation (Back-translation)

### Mô tả
Nghiên cứu và cài đặt kỹ thuật **Back-translation** để tạo dữ liệu tổng hợp cho các lớp nhãn thiếu mẫu (Tiêu cực / Trung lập), giải quyết vấn đề mất cân bằng lớp (class imbalance).

### Cơ chế Back-translation

```
Văn bản gốc (vi)
      │
      ▼  Bước 1: Dịch Việt → Anh
"Sản phẩm rất tệ, vỡ ngay lần đầu."
      │
      ▼
"The product is very bad, broke on the first use."
      │
      ▼  Bước 2: Dịch Anh → Việt
"Sản phẩm rất tệ, vỡ ngay khi sử dụng lần đầu."
      │
      ▼
Văn bản mới — cùng ý nghĩa, khác cấu trúc câu
```

**Ưu điểm so với các kỹ thuật augment khác:**
- Giữ nguyên ngữ nghĩa (không làm mất nhãn)
- Đa dạng hóa từ vựng và cấu trúc câu một cách tự nhiên
- Không cần model riêng, dùng API dịch có sẵn

### Kiến trúc

```
ai_engine/text_processing/augmentation.py
├── SECTION 1 — Translation backends
│   ├── GoogleFreeTranslator          # googletrans (free, rate-limited)
│   └── DeepLTranslator               # DeepL API (chính xác hơn, có giới hạn)
│
├── SECTION 2 — Back-translation core
│   ├── back_translate_text()         # augment 1 văn bản
│   └── back_translate_batch()        # augment danh sách (có fallback)
│
├── SECTION 3 — Augmentation pipeline
│   ├── augment_minority_classes()    # auto-detect & augment lớp thiếu mẫu
│   └── print_augmentation_report()   # báo cáo trước/sau augmentation
│
├── SECTION 4 — Demo mode
│   └── run_demo()                    # test nhanh với 5 câu mẫu
│
└── SECTION 5 — CLI
    └── main()
```

### Cách sử dụng

```powershell
# Demo nhanh (không cần data)
py ai_engine/text_processing/augmentation.py --demo

# Augment CSV, nhân đôi lớp tiêu cực và trung lập
py ai_engine/text_processing/augmentation.py `
    --data-path data/processed/reviews_labeled.csv `
    --label-col sentiment `
    --target-labels "tieu cuc" "trung lap" `
    --multiply 2 `
    --output-path data/processed/reviews_augmented.csv

# Dùng DeepL (cần DEEPL_API_KEY trong .env)
py ai_engine/text_processing/augmentation.py `
    --data-path data/processed/reviews_labeled.csv `
    --backend deep_l `
    --multiply 3
```

### Output mẫu

```
============================================================
  AUGMENTATION REPORT
============================================================
  Label                Before     After    Added
  -------------------- --------  --------  --------
  tich cuc              5,000     5,000         0
  tieu cuc              1,200     2,400     1,200
  trung lap               800     1,600       800
  TOTAL                 7,000     9,000     2,000
============================================================
```

### Tham số quan trọng

| Tham số | Mặc định | Ghi chú |
|---------|---------|--------|
| `--multiply` | `2` | Nhân x2 số mẫu cho lớp thiếu. Tăng cẩn thận, quá nhiều có thể gây overfitting |
| `--backend` | `google_free` | `google_free` = không cần key; `deep_l` = cần API key, chất lượng cao hơn |
| `--pivot-lang` | `en` | Ngôn ngữ trung gian. Thử `fr` hoặc `zh` để tạo đa dạng hơn |
| `--delay` | `1.0` | Thời gian chờ (giây) giữa các request với `google_free` |

### Dependency bổ sung

```bash
# Backend google_free (miễn phí, không cần API key)
pip install googletrans==4.0.0rc1

# Backend DeepL (chất lượng cao hơn, cần API key)
pip install deepl
```

> **Lưu ý:** `googletrans` có thể bị rate-limit khi chạy số lượng lớn. Nếu augment > 1,000 mẫu, nên dùng `--delay 2.0` hoặc chuyển sang `deep_l`.

---

## Luồng dữ liệu kết hợp 3 Task

```
data/processed/reviews.csv
        │
        ▼  Task 2: train_spam_model.py
data/processed/reviews_spam_labeled.csv   (thêm cột: final_spam, anomaly_score)
        │
        │  Lọc clean reviews (final_spam == 0)
        ▼
data/processed/reviews_clean.csv
        │
        ▼  Task 3: augmentation.py
data/processed/reviews_augmented.csv      (thêm cột: is_augmented)
        │
        │  Chia train/val/test
        ▼
Huấn luyện TextEnsembleModel / PhoBERT
        │
        ▼  Task 1: evaluate_models.py
Báo cáo: Confusion Matrix, Macro F1, ROC-AUC
```

---

## Cấu trúc file liên quan

```
ecommerce-review-analytics/
├── scripts/
│   ├── evaluate_models.py        # Task 1 — đánh giá text/image/spam model
│   ├── train_spam_model.py       # Task 2 — huấn luyện SpamHybridModel
│   └── train_defect_model.py     # (có sẵn từ trước)
├── ai_engine/
│   ├── models/
│   │   └── text_baseline.py      # TextEnsembleModel (sentiment)
│   └── text_processing/
│       ├── spam_filter.py        # Rule-based spam detection (21 rules)
│       ├── augmentation.py       # Task 3: Back-translation
│       ├── sentiment_analysis.py
│       └── vectorizers.py
└── docs/
    └── evaluate_spam_textaugmentation.md    # File này
```
