# Hướng dẫn train MobileNetV3 trên Kaggle GPU (Fixed Split)

Tài liệu này mô tả chi tiết quy trình chạy training MobileNetV3 trên GPU của Kaggle sử dụng tập dữ liệu phân tách cố định (**fixed physical split**), tương thích hoàn toàn với cấu hình Model 1 (ResNet50).

Ước tính thời gian: **30–45 phút** trên GPU P100/T4.

---

## 1. Chuẩn bị trước khi lên Kaggle

### 1.1 Kiểm tra tập dữ liệu phân tách cố định
Đảm bảo tập dữ liệu phân tách cố định đã có sẵn tại máy local ở đường dẫn:
```text
data/image_dataset_split/
├── train/
│   ├── defect/          (1,162 ảnh)
│   └── no-defect/       (12,495 ảnh)
├── val/
│   ├── defect/          (249 ảnh)
│   └── no-defect/       (2,678 ảnh)
└── test/
    ├── defect/          (249 ảnh)
    └── no-defect/       (2,678 ảnh)
```

> [!IMPORTANT]
> * **Không chạy lại** script `scripts/split_image_dataset.py` để tránh làm xáo trộn tập Test set.
> * **Không sử dụng** dataset `labeled/labeled/` (4 class) cũ cho đợt training chính thức này.

### 1.2 Nén thư mục dữ liệu để upload
Nén toàn bộ thư mục `image_dataset_split` thành tệp tin zip:
```powershell
# Windows PowerShell (chạy từ thư mục gốc của project)
Compress-Archive -Path "data\image_dataset_split" -DestinationPath "image_dataset_split.zip"
```

Cấu trúc bên trong `image_dataset_split.zip` phải là:
```text
image_dataset_split/
├── train/
│   ├── defect/
│   └── no-defect/
├── val/
│   ├── defect/
│   └── no-defect/
└── test/
    ├── defect/
    └── no-defect/
```

---

## 2. Tạo Kaggle Dataset

1. Truy cập [kaggle.com/datasets](https://www.kaggle.com/datasets) → Chọn **New Dataset**.
2. Đặt tên Dataset: `ecommerce-image-split` (hoặc tên tuỳ chọn của bạn).
3. Kéo thả file `image_dataset_split.zip` vào để upload.
4. Thiết lập quyền riêng tư là **Private**.
5. Bấm **Create** và chờ quá trình xử lý zip hoàn tất.

---

## 3. Tạo Kaggle Notebook

1. Truy cập [kaggle.com/code](https://www.kaggle.com/code) → Chọn **New Notebook**.
2. Chuyển đổi định dạng Notebook sang **Script** (vào menu File hoặc dấu ba chấm ở góc trên bên phải → chọn *Convert to Script*).
3. Mở khung **Settings** bên phải màn hình:
   * **Accelerator:** Chọn **GPU P100** *(Khuyến nghị)* hoặc **GPU T4**.
   * **Internet:** Gạt nút sang **ON** *(Bắt buộc để clone mã nguồn từ GitHub)*.
4. Bấm **Add Input** ở góc trên bên phải → Tìm kiếm dataset `ecommerce-image-split` vừa tạo ở Bước 2 → Bấm **Add**.

---

## 4. Paste mã nguồn huấn luyện

Sao chép toàn bộ nội dung file `notebooks/train_mobilenet_kaggle.py` từ nhánh `feature/mobilenetv3-model2-sync` của Model 2 và dán vào trình soạn thảo script của Kaggle.

Script này sẽ tự động tìm kiếm đường dẫn tập dữ liệu split trong `/kaggle/input`, sau đó kích hoạt lệnh train:
```bash
python scripts/train_image_baseline.py \
  --backbone mobilenet_v3 \
  --data-dir <đường-dẫn-train-split> \
  --val-dir <đường-dẫn-val-split> \
  --epochs 20 \
  --lr 1e-3 \
  --batch-size 64 \
  --patience 5 \
  --weights-dir /kaggle/working/project/ai_engine/models/weights
```

> [!TIP]
> * Nếu bạn chạy trên GPU T4 có VRAM thấp hơn, hãy chỉnh sửa dòng `--batch-size 64` thành `--batch-size 32` trong biến `cmd` của script để tránh lỗi Out-Of-Memory (OOM).

---

## 5. Chạy Notebook & Tải kết quả

1. Nhấp vào **Save Version** ở góc trên bên phải.
2. Chọn cấu hình **Save & Run All (Commit)** và bấm **Save**.
3. Khi tiến trình huấn luyện hoàn tất (khoảng 30-45 phút):
   * Truy cập vào tab **Output** của Notebook.
   * Tải về các file kết quả huấn luyện Model 2:
     * `mobilenet_v3_model2_defect.pt` (weights của model)
     * `mobilenet_v3_model2_results.json` (chỉ số đánh giá chung)
     * `mobilenet_v3_model2_learning_curves.png` (biểu đồ loss/accuracy)
     * `mobilenet_v3_model2_confusion_matrix.png` (ma trận nhầm lẫn)
     * `mobilenet_v3_model2_training_history.json` (lịch sử train chi tiết)

---

## 6. Tích hợp kết quả về Local

Sau khi tải về thành công các file output từ Kaggle, di chuyển chúng vào đúng các thư mục trong repository Model 2:

```powershell
# Chạy từ thư mục gốc của project Model 2 local

# 1. Di chuyển weights vào weights/
Move-Item "path/to/download/mobilenet_v3_model2_defect.pt" "ai_engine/models/weights/mobilenet_v3_model2_defect.pt"

# 2. Di chuyển các file kết quả vào results/
Move-Item "path/to/download/mobilenet_v3_model2_results.json" "ai_engine/models/results/mobilenet_v3_model2_results.json"
Move-Item "path/to/download/mobilenet_v3_model2_learning_curves.png" "ai_engine/models/results/mobilenet_v3_model2_learning_curves.png"
Move-Item "path/to/download/mobilenet_v3_model2_confusion_matrix.png" "ai_engine/models/results/mobilenet_v3_model2_confusion_matrix.png"
Move-Item "path/to/download/mobilenet_v3_model2_training_history.json" "ai_engine/models/results/mobilenet_v3_model2_training_history.json"
```

---

## 7. Đánh giá chính thức trên Test Set vật lý tại Local

Tập Test set vật lý hoàn toàn bị cô lập khỏi quá trình huấn luyện và chọn lọc checkpoint. Để đưa ra báo cáo hiệu năng cuối cùng của Model 2 trên tập Test set này, hãy chạy lệnh sau tại local:

```bash
# Đổi tên file model tạm thời sang tên gốc để evaluate
# Hoặc sử dụng trực tiếp cờ --weights-dir nếu cần thiết.
python scripts/train_image_baseline.py \
  --backbone mobilenet_v3 \
  --test-dir data/image_dataset_split/test \
  --eval-test
```

---

## Troubleshooting

* **Lỗi `GPU không khả dụng`**: Bạn hãy kiểm tra lại mục **Accelerator** ở cấu hình bên phải của Kaggle, đảm bảo đã chọn GPU P100 hoặc T4.
* **Lỗi `Không tìm thấy cấu trúc split train/val/test`**: Hãy kiểm tra lại file `.zip` tải lên. Phải chứa chính xác thư mục con `train`, `val`, và `test`, bên trong mỗi thư mục con là các nhãn `defect` và `no-defect`.
* **Lỗi `CUDA out of memory`**: Hãy giảm `--batch-size` trong script huấn luyện từ `64` xuống `32`.
