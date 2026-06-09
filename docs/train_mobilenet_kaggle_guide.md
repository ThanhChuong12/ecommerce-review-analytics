# Hướng dẫn train MobileNetV3 trên Kaggle GPU

Tài liệu này mô tả từng bước để chạy training MobileNetV3 trên GPU miễn phí của Kaggle, download kết quả và tích hợp về project local.

Ước tính thời gian: **30–45 phút** trên GPU P100/T4, so với 2+ giờ trên CPU.

---

## 1. Chuẩn bị trước khi lên Kaggle

### 1.1 Đảm bảo dữ liệu đã được split

Trên máy local, kiểm tra xem `split_manifest.json` đã tồn tại chưa:

```
labeled/split_manifest.json
```

Nếu chưa có, chạy lệnh dưới đây một lần duy nhất (không chạy lại sau khi đã train):

```bash
python scripts/split_image_dataset.py
```

Kết quả sau khi split:

```
labeled/
├── split_manifest.json       ← file lock, chứng minh đã split
├── labeled/                  ← nguồn gốc, giữ nguyên
├── train/   → 19,419 ảnh (70%)
├── val/     → 4,162 ảnh  (15%)
└── test/    → 4,162 ảnh  (15%)
```

### 1.2 Nén thư mục ảnh để upload

Kaggle yêu cầu upload dưới dạng dataset. Nén thư mục `labeled/labeled/` (thư mục gốc chứa 4 class folders):

```bash
# Windows PowerShell
Compress-Archive -Path "labeled\labeled" -DestinationPath "kaggle_dataset.zip"
```

Cấu trúc bên trong zip phải là:

```
labeled/
├── intact/
├── damaged/
├── wrong_item/
└── irrelevant/
```

> **Lưu ý:** Upload `labeled/labeled/` chứ không phải `labeled/train/`. Script Kaggle sẽ tự nhận diện 4 class folders và sử dụng.

---

## 2. Tạo Kaggle Dataset

1. Vào [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Đặt tên: `ecommerce-labeled-images`
3. Upload file `kaggle_dataset.zip`
4. Chọn **Private** để không ai khác thấy dữ liệu
5. Nhấn **Create** và chờ xử lý xong

---

## 3. (Tuỳ chọn) Tạo Dataset chứa `image_baseline.py` đã fix

Nếu code trong repo đã ổn định, bỏ qua bước này. Nếu cần áp bug fix lên file `image_baseline.py` trước khi train:

1. Vào [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Đặt tên: `mobilenet-fixed`
3. Upload file: `ai_engine/models/image_baseline.py`
4. Nhấn **Create**

---

## 4. Tạo Kaggle Notebook

1. Vào [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Đổi type sang **Script** (không phải Notebook)
3. Vào **Settings** (biểu tượng bánh răng bên phải):
   - **Accelerator:** GPU P100 *(khuyến nghị)* hoặc T4 x2
   - **Internet:** ON *(cần để clone repo GitHub)*
4. Vào **Add Input** → tìm dataset `ecommerce-labeled-images` vừa upload → Add
5. (Nếu có) Vào **Add Input** → tìm `mobilenet-fixed` → Add

---

## 5. Paste script vào Notebook

Copy toàn bộ nội dung file `notebooks/train_mobilenet_kaggle.py` vào phần editor của notebook.

Mặc định script sẽ chạy lệnh:

```bash
python scripts/train_image_baseline.py \
  --backbone   mobilenet_v3 \
  --data-dir   <auto-detected> \
  --epochs     20 \
  --lr         1e-3 \
  --batch-size 64 \
  --val-split  0.2 \
  --patience   5 \
  --weights-dir /kaggle/working/project/ai_engine/models/weights
```

### Tuỳ chỉnh nếu cần

Tìm đoạn `cmd = [...]` trong script (khoảng dòng 124) và sửa:

| Tham số | GPU P100 | GPU T4 | Ghi chú |
|---|---|---|---|
| `--batch-size` | `64` | `32` | P100 có 16GB VRAM |
| `--epochs` | `20` | `15` | Tăng nếu muốn train lâu hơn |
| `--subset-ratio` | `1.0` | `1.0` | Dùng toàn bộ data trên GPU |
| `--patience` | `5` | `5` | Early stopping |

---

## 6. Chạy Notebook

Nhấn **Save Version** → **Save & Run All** → chờ kết quả.

Luồng thực hiện của script:

```
[Step 1] Kiểm tra GPU
[Step 2] Clone repo từ GitHub
[Step 3] Apply bug fix image_baseline.py (nếu có dataset fix)
[Step 4] Cài dependencies (albumentations, scikit-learn, Pillow)
[Step 5] Tìm thư mục data trong /kaggle/input
[Step 6] Chạy training (~30-45 phút)
[Step 7] Copy output ra /kaggle/working/outputs/
```

Output được lưu tại:

```
/kaggle/working/outputs/
├── mobilenet_v3_defect.pt          ← file weights (14MB)
└── image_baseline_results.json     ← accuracy, F1-macro
```

---

## 7. Download kết quả

Sau khi notebook chạy xong:

1. Vào tab **Output** của notebook
2. Download 2 file:
   - `mobilenet_v3_defect.pt`
   - `image_baseline_results.json`

Ngoài ra, để download cả learning curves và confusion matrix, thêm đoạn sau vào cuối script (trước dòng print cuối):

```python
# Thêm vào cuối script để copy thêm artifacts
RESULTS_SRC_DIR = WORK_DIR / "ai_engine/models/results"
for fname in [
    "mobilenet_v3_learning_curves.png",
    "mobilenet_v3_confusion_matrix.png",
    "mobilenet_v3_training_history.json",
]:
    src = RESULTS_SRC_DIR / fname
    if src.exists():
        shutil.copy(src, OUTPUT_DIR / fname)
        print(f"{OK}Saved: {fname}")
```

---

## 8. Tích hợp kết quả về project local

Sau khi download, copy các file vào đúng thư mục:

```bash
# Weights
copy mobilenet_v3_defect.pt "ai_engine\models\weights\mobilenet_v3_defect.pt"

# Results
copy image_baseline_results.json     "ai_engine\models\results\"
copy mobilenet_v3_learning_curves.png "ai_engine\models\results\"
copy mobilenet_v3_confusion_matrix.png "ai_engine\models\results\"
copy mobilenet_v3_training_history.json "ai_engine\models\results\"
```

---

## 9. Đánh giá trên Test set (sau khi có weights)

Sau khi copy weights về, chạy lệnh dưới để lấy metrics chính thức trên tập test (chưa bao giờ dùng trong training):

```bash
python scripts/train_image_baseline.py \
  --backbone   mobilenet_v3 \
  --eval-test
```

Kết quả được lưu tại:

```
ai_engine/models/results/test_set_results.json
```

---

## 10. Chuẩn bị ảnh cho báo cáo LaTeX

Copy 2 file sau vào thư mục `ML_Final_Models/graphics/MobileNetV3/`:

```bash
mkdir "ML_Final_Models\graphics\MobileNetV3"
copy "ai_engine\models\results\mobilenet_v3_learning_curves.png"  "ML_Final_Models\graphics\MobileNetV3\"
copy "ai_engine\models\results\mobilenet_v3_confusion_matrix.png" "ML_Final_Models\graphics\MobileNetV3\"
```

---

## Troubleshooting

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `GPU không khả dụng` | Quên bật GPU trong Settings | Vào Settings → Accelerator → GPU P100 |
| `Không tìm thấy 4 class folders` | Upload sai thư mục | Phải có `intact/`, `damaged/`, `wrong_item/`, `irrelevant/` trong zip |
| `git clone thất bại` | Internet OFF | Settings → Internet → ON |
| `CUDA out of memory` | Batch size quá lớn | Giảm `--batch-size 64` xuống `32` |
| `image_baseline.py không tìm thấy` | Dataset fix chưa được add | Add Input dataset `mobilenet-fixed` vào notebook |

---

## Thông tin kỹ thuật tham khảo

| Thông số | Giá trị |
|---|---|
| Backbone | MobileNetV3-Large (ImageNet1K_V2) |
| Tổng tham số | 3,466,036 |
| Tham số huấn luyện (giai đoạn 1) | 494,084 (14.3%) |
| Tham số đóng băng | 2,971,952 (85.7%) |
| Epoch mở backbone | Epoch 4 (features[13:]) |
| Hàm mất mát | CrossEntropyLoss + class weights + label_smoothing=0.05 |
| Optimizer | Adam, differential LR (head: 1e-3, backbone: 1e-4) |
| Lịch trình LR | ReduceLROnPlateau (factor=0.5, patience=2) |
| Gradient clipping | max_norm=1.0 |
| Augmentation đặc biệt | RandomErasing(p=0.15), RandomVerticalFlip(p=0.3) |
