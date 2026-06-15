# Hướng dẫn chi tiết Huấn luyện Model 2 (MobileNetV3) trên Kaggle GPU (Offline)

Tài liệu này tổng hợp toàn bộ các bước từ chuẩn bị dữ liệu và mã nguồn tại local, thiết lập môi trường Kaggle ở chế độ ngoại tuyến (**Internet OFF**), chạy huấn luyện, tải kết quả và đánh giá mô hình MobileNetV3 (Model 2).

---

## A. Chuẩn bị các gói dữ liệu tại máy Local

### 1. Nén tập dữ liệu phân tách cố định (Fixed Split)
Kiểm tra thư mục `data/image_dataset_split/` tại local, đảm bảo chứa cấu trúc phân tách cố định:
- `train/` (defect: 1,162 | no-defect: 12,495)
- `val/` (defect: 249 | no-defect: 2,678)
- `test/` (defect: 249 | no-defect: 2,678)

Nén thư mục `image_dataset_split` bằng PowerShell:
```powershell
Compress-Archive -Path "data\image_dataset_split" -DestinationPath "image_dataset_split.zip"
```

### 2. Nén mã nguồn dự án ngoại tuyến (Offline Code Package)
Nén các thư mục và file cần thiết từ nhánh `feature/mobilenetv3-model2-sync` (Model 2) để chạy trên Kaggle:
```powershell
Compress-Archive `
  -Path "ai_engine","scripts","notebooks","docs",".gitignore","requirements.txt" `
  -DestinationPath "mobilenetv3_model2_code_offline.zip" `
  -Force
```

---

## B. Tạo hai Kaggle Datasets (Private)

### 1. Dataset chứa mã nguồn: `mobilenetv3-model2-code-offline`
1. Vào [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**.
2. **Dataset Title:** `mobilenetv3-model2-code-offline`.
3. Kéo thả file `mobilenetv3_model2_code_offline.zip` vào upload.
4. Chọn chế độ **Private** → Bấm **Create**.

### 2. Dataset chứa ảnh: `ecommerce-fixed-split-defect-dataset`
1. Chọn **New Dataset**.
2. **Dataset Title:** `ecommerce-fixed-split-defect-dataset`.
3. Kéo thả file `image_dataset_split.zip` vào upload.
4. Chọn chế độ **Private** → Bấm **Create**.

---

## C. Thiết lập Kaggle Notebook (Internet OFF)

1. Vào [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**.
2. Chuyển đổi Notebook sang dạng **Script**:
   * Nhấp biểu tượng ba chấm dọc ở góc trên bên phải → chọn **Convert to Script**.
3. Cấu hình bảng **Settings** ở bên phải màn hình:
   * **Accelerator:** Chọn **GPU T4** (hoặc GPU P100 nếu có).
   * **Internet:** Gạt sang **OFF** (Chạy hoàn toàn ngoại tuyến).
4. Thêm đầu vào (**Add Input**):
   * Tìm kiếm và add dataset `mobilenetv3-model2-code-offline`.
   * Tìm kiếm và add dataset `ecommerce-fixed-split-defect-dataset`.
5. Mở file `notebooks/train_mobilenet_kaggle.py` tại máy local, copy toàn bộ nội dung và dán đè hoàn toàn vào trình soạn thảo script của Kaggle.

---

## D. Lệnh Huấn luyện (Chạy tự động bởi Script)

Script Kaggle sẽ tự động giải nén mã nguồn ngoại tuyến vào `/kaggle/working/project`, định vị thư mục ảnh huấn luyện và chạy lệnh:

```bash
python scripts/train_image_baseline.py \
  --backbone mobilenet_v3 \
  --data-dir /kaggle/input/ecommerce-fixed-split-defect-dataset/image_dataset_split/train \
  --val-dir /kaggle/input/ecommerce-fixed-split-defect-dataset/image_dataset_split/val \
  --epochs 15 \
  --lr 1e-3 \
  --batch-size 32 \
  --patience 5 \
  --weights-dir /kaggle/working/project/ai_engine/models/weights
```

---

## E. Tải Kết quả huấn luyện từ Kaggle

Khi script hoàn tất, truy cập tab **Output** của Notebook và tải về các file:
- `mobilenet_v3_model2_defect.pt`
- `mobilenet_v3_model2_results.json`
- `mobilenet_v3_model2_learning_curves.png`
- `mobilenet_v3_model2_confusion_matrix.png`
- `mobilenet_v3_model2_training_history.json`

---

## F. Tích hợp và Đánh giá tại Local

Di chuyển các file đã tải về vào các thư mục tương ứng trong project Model 2 local:
- Weights: `ai_engine/models/weights/mobilenet_v3_model2_defect.pt`
- Results: `ai_engine/models/results/`

Chạy đánh giá hiệu năng chính thức trên tập Test set vật lý độc lập tại local:
```bash
python scripts/train_image_baseline.py \
  --backbone mobilenet_v3 \
  --weights-dir ai_engine/models/weights \
  --test-dir data/image_dataset_split/test \
  --eval-test
```
Kết quả kiểm thử cuối cùng sẽ được ghi nhận tại file `ai_engine/models/results/test_set_results.json`.
