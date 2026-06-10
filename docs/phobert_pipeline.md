# Hướng Dẫn Pipeline Huấn Luyện Mô Hình PhoBERT

## Tổng quan

Pipeline huấn luyện PhoBERT gồm **2 bước chính**:

```
[Dữ liệu đã xử lý] → [Tune siêu tham số] → [Huấn luyện chính thức] → [Kết quả]
  data/processed/      tune_phobert.py       train_phobert.py       artifacts/
```

- **Bước 1 – Tune:** Dùng Optuna tìm siêu tham số tốt nhất → lưu vào `phobert_best_params.json`
- **Bước 2 – Train:** Dùng siêu tham số vừa tìm để huấn luyện model chính thức → lưu model, metrics, biểu đồ

> ⚠️ **Lưu ý:** Pipeline này được thiết kế để chạy trên **Kaggle** (GPU T4) vì tập dữ liệu lớn và model PhoBERT (~500MB). Chạy trên CPU sẽ rất chậm.

---

## Yêu cầu

### Thư viện

```bash
pip install transformers datasets optuna sentencepiece
```

### Dữ liệu đầu vào

Các file CSV đã được split sẵn phải nằm ở:

```
data/processed/
  ├── processed_labeled_text_train.csv   # ~17,700 mẫu
  ├── processed_labeled_text_val.csv     # ~3,800 mẫu
  └── processed_labeled_text_test.csv    # ~3,800 mẫu
```

Mỗi file phải có 2 cột:
- `cleaned_text`: văn bản đã làm sạch
- `sentiment_label`: nhãn cảm xúc (`tích cực` / `tiêu cực` / `trung lập`)

---

## Phân bố lớp

Dataset có mất cân bằng lớp nghiêm trọng:

| Lớp | Tỉ lệ |
|---|---|
| tích cực | ~94% |
| tiêu cực | ~5% |
| trung lập | ~1% |

→ Pipeline xử lý bằng **Focal Loss + Alpha Weighting** (không dùng SMOTE/oversampling vì PhoBERT đã đủ mạnh).

---

## Bước 1: Tune siêu tham số

**Script:** `scripts/tune_phobert.py`

### Mục đích

Dùng **Optuna** để tìm bộ siêu tham số tốt nhất, tránh việc phải thử thủ công.

### Không gian tìm kiếm

| Tham số | Khoảng |
|---|---|
| `learning_rate` | `1e-5` → `5e-5` (log scale) |
| `weight_decay` | `0.01` → `0.1` (log scale) |
| `warmup_ratio` | `0.0` → `0.2` |
| `num_train_epochs` | `2` → `4` (số nguyên) |
| `per_device_train_batch_size` | `16` hoặc `32` |
| `gamma` (Focal Loss) | `1.0` → `3.0` |

### Cấu hình đặc biệt khi tune

Để tránh hết dung lượng đĩa Kaggle (~20GB):

```python
save_strategy="no"          # Không lưu checkpoint giữa chừng
load_best_model_at_end=False
```

### Chạy

```bash
python scripts/tune_phobert.py --n_trials 10
```

### Output

```
artifacts/metrics/phobert_best_params.json
```

Ví dụ nội dung:

```json
{
  "gamma": 1.3549,
  "learning_rate": 4.76e-05,
  "weight_decay": 0.0508,
  "warmup_ratio": 0.0162,
  "num_train_epochs": 2,
  "per_device_train_batch_size": 16
}
```

---

## Bước 2: Huấn luyện chính thức

**Script:** `scripts/train_phobert.py`

### Mục đích

Huấn luyện model PhoBERT với siêu tham số tối ưu từ Bước 1. Script tự động đọc `phobert_best_params.json` nếu file tồn tại.

### Các tính năng chính

| Tính năng | Chi tiết |
|---|---|
| **Focal Loss** | Tập trung vào mẫu khó, giảm ảnh hưởng của lớp đa số |
| **Alpha Weighting** | Trọng số nghịch đảo tần số theo căn bậc 2 |
| **Early Stopping** | `patience=2`, theo dõi `eval_val_f1_macro` |
| **Best Model** | Tự động load checkpoint tốt nhất sau train |
| **Metrics đôi** | Đánh giá cả Train và Val mỗi epoch |
| **FP16** | Tự động bật khi có GPU (tăng tốc ~2x) |

### Optimizer & Scheduler

- **Optimizer:** AdamW
- **Scheduler:** Cosine LR với warmup
- **Metric chọn model tốt nhất:** Macro F1 trên Val set (`eval_val_f1_macro`)

### Chạy

```bash
python scripts/train_phobert.py
```

Hoặc ghi đè tham số thủ công (khi không có file JSON):

```bash
python scripts/train_phobert.py \
    --lr 4e-5 \
    --epochs 3 \
    --batch_size 16 \
    --gamma 1.5
```

### Output

```
artifacts/
  metrics/
    phobert_train_metrics.json    # Loss, epoch trong quá trình train
    phobert_val_metrics.json      # Accuracy, F1, Precision, Recall trên Val
    phobert_test_metrics.json     # Kết quả cuối cùng trên Test (dùng báo cáo)
  plots/
    phobert_loss_curve.png        # Biểu đồ Loss (train + val)
    phobert_f1_curve.png          # Biểu đồ F1 (train + val)
    phobert_confusion_matrix.png  # Ma trận nhầm lẫn trên Test
  model/tuned/phobert/
    model.safetensors             # Trọng số model (~500MB)
    config.json
    tokenizer_config.json
    vocab.txt
    bpe.codes
    ...
```

---

## Kết quả

### Metrics cuối cùng trên tập Test

| Chỉ số | Giá trị |
|---|---|
| Accuracy | 0.878 |
| F1 Macro | 0.754 |
| Precision Macro | 0.746 |
| Recall Macro | 0.765 |
| F1 tích cực | 0.953 |
| F1 tiêu cực | 0.831 |
| F1 trung lập | 0.478 |

### Biểu đồ

Biểu đồ Loss và F1 có **2 đường** (train + val) để quan sát overfitting.

Ma trận nhầm lẫn có thứ tự:
- Cột (trái → phải): tích cực, trung lập, tiêu cực
- Hàng (trên → dưới): tiêu cực, trung lập, tích cực *(đọc từ dưới lên: tích cực, trung lập, tiêu cực)*

---

## Chạy trên Kaggle (khuyến nghị)

### Cell 1: Cài đặt & Clone repo

```python
!pip install transformers datasets optuna sentencepiece -q
!git clone https://ThanhChuong12:<TOKEN>@github.com/ThanhChuong12/ecommerce-review-analytics.git
%cd ecommerce-review-analytics
!git checkout feat/split_text
```

### ⚠️ Trước Cell 2: Thêm dataset vào Kaggle Notebook

Trước khi copy dữ liệu, cần đảm bảo dataset đã được thêm vào notebook:

1. Trong Kaggle Notebook, nhìn bên phải → mục **Input** → click **+ Add Input**
2. Tìm kiếm dataset `processed_labeled_text` (của user `buinhan`)
3. Click **Add** → dataset sẽ xuất hiện trong tab Input với 3 file:
   - `processed_labeled_text_train.csv`
   - `processed_labeled_text_val.csv`
   - `processed_labeled_text_test.csv`
4. Đường dẫn trên Kaggle sẽ là `/kaggle/input/datasets/<kaggle-username>/processed-labeled-text/`

### Cell 2: Copy dữ liệu

```python
import shutil, os
os.makedirs("data/processed", exist_ok=True)
base = "/kaggle/input/datasets/<kaggle-username>/processed-labeled-text/"
for name in ["train", "val", "test"]:
    shutil.copy(f"{base}processed_labeled_text_{name}.csv", "data/processed/")
print("Done!")
```

### Cell 3 (tuỳ chọn): Tune siêu tham số

Bỏ qua nếu đã có `phobert_best_params.json` trong repo.

```python
!python scripts/tune_phobert.py --n_trials 10
```

### Cell 4: Train chính thức

```python
!python scripts/train_phobert.py
```

### Cell 5: Tải kết quả về máy

```python
!zip -r results.zip artifacts/metrics/ artifacts/plots/
# → Tải file results.zip trong tab Output bên phải
```

> **Lưu ý Kaggle:** Sau khi chạy xong, nhớ bấm **Save Version** trước khi tắt session, nếu không tất cả file trong `/kaggle/working/` sẽ bị xóa!

---

## Cấu trúc code liên quan

```
scripts/
  tune_phobert.py           # Bước 1: Optuna tuning
  train_phobert.py          # Bước 2: Huấn luyện chính thức

ai_engine/
  data/
    phobert_dataset.py      # Dataset class cho PhoBERT
  models/
    phobert_trainer.py      # FocalLossTrainer (custom Trainer)

artifacts/
  metrics/
    phobert_best_params.json    # Output của tune, input của train
    phobert_train_metrics.json
    phobert_val_metrics.json
    phobert_test_metrics.json
  plots/
    phobert_loss_curve.png
    phobert_f1_curve.png
    phobert_confusion_matrix.png
```

---

## Câu hỏi thường gặp

**Q: Tại sao không dùng SMOTE hay Back-translation?**
PhoBERT đã pre-train trên 20GB văn bản tiếng Việt. Focal Loss + Alpha weighting xử lý imbalance hiệu quả hơn mà không cần augment.

**Q: Tại sao tune chạy chậm?**
Mỗi trial Optuna train 1 epoch trên ~17.000 mẫu. 10 trials × ~10 phút = ~100 phút. Dùng GPU T4 trên Kaggle sẽ nhanh hơn CPU ~10-20 lần.

**Q: Có cần chạy lại tune không nếu đã có `phobert_best_params.json`?**
Không. `train_phobert.py` tự đọc file JSON này. Chỉ cần chạy `python scripts/train_phobert.py`.

**Q: Model lưu ở đâu để deploy?**
`artifacts/model/tuned/phobert/` — chứa đủ `model.safetensors`, tokenizer, config để load và inference.
