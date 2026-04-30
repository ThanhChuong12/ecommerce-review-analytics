# Gán nhãn ảnh từ dữ liệu cào

Thư mục này chứa pipeline offline để tạo dữ liệu train từ media trong review
(ảnh/video). Đây **không** phải backend runtime.

## Pipeline làm gì

1) Đọc CSV đầu ra từ scraper.
2) Tải ảnh từ URL.
3) Tải video và cắt ngẫu nhiên 3 frame lưu thành JPG.
4) Kiểm tra ảnh lỗi/corrupted và xóa.
5) Auto-label bằng vision model (Google Vertex AI / OpenAI / Groq / Custom), có kèm review text + product name.
6) Xuất `labels.csv` (toàn bộ nhãn) và `training_labels.csv` (nhãn sẵn sàng cho training).
7) (Tuỳ chọn) Copy ảnh vào thư mục theo nhãn để kiểm tra bằng mắt.

## Cấu trúc thư mục

```
image_labeling/
  media_pipeline.py
  requirements.txt
  data/
    raw_media/        # ảnh và video đã tải về
    frames/           # frame cắt từ video (JPG)
    manifests/
      media.csv           # tất cả media đã tải (ảnh + video)
      images.csv          # tất cả ảnh (tải về + frame)
      labels.csv          # nhãn đầu ra từ auto-label
      training_labels.csv # nhãn đã lọc, sẵn sàng cho training
      ground_truth.csv    # nhãn kiểm tra thủ công (dùng để đánh giá model)
    labeled/
      intact/
      damaged/
      irrelevant/
      wrong_item/
```

## Cài đặt

Tạo môi trường ảo riêng cho thư mục này:

```bash
cd image_labeling
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Xác thực (Authentication)

Pipeline hỗ trợ nhiều provider. Tùy provider cần cấu hình khác nhau trong file `.env` ở root repo.

### Google (Vertex AI) 

Không cần API key. Xác thực qua Google Application Default Credentials (ADC):

```bash
gcloud auth application-default login
```

Sau đó cần khai báo `PROJECT_ID` trong `.env`:

```
PROJECT_ID=your-gcp-project-id
```

Lệnh label sẽ dùng project GCP này để gọi Vertex AI.

### Các provider khác (OpenAI / Groq / Custom)

Khai báo API key tương ứng trong `.env`:

```
OPENAI_API_KEY=...
GROQ_API_KEY=...
CUSTOM_API_KEY=...         # optional
CUSTOM_BASE_URL=...        # optional (OpenAI-compatible endpoint)
```

## Cách dùng

### 1) Tải media từ CSV của scraper

CSV đầu vào cần có cột `product_name`.

```bash
python media_pipeline.py download --csv "..\scraping_agent\data"
```

Hoặc chỉ định file cụ thể:

```bash
python media_pipeline.py download --csv "..\scraping_agent\data\all_reviews.csv"
```

### 2) Cắt frame từ video

```bash
python media_pipeline.py extract
```

### 3) Kiểm tra ảnh lỗi (xóa corrupted)

```bash
python media_pipeline.py validate
```

### 4) Tạo manifest ảnh (ảnh gốc + ảnh từ frame video)

```bash
python media_pipeline.py build-images
```

### 5) Auto-label

```bash
python media_pipeline.py label --provider <google|openai|groq|custom> --model <MODEL_NAME> --max-images 200 --sleep 0.8 --batch-size 1
```

Ví dụ:

```bash
# Google Vertex AI (xác thực qua gcloud, không cần API key)
python media_pipeline.py label --provider google --model gemini-3.1-flash-lite-preview --batch-size 10 --sleep 1

# OpenAI
python media_pipeline.py label --provider openai --model "gpt-4.1" --max-images 1000 --sleep 0.8
```

Lệnh này sẽ ghi `data/manifests/labels.csv` và (mặc định) copy ảnh vào
`data/labeled/<label>/` để kiểm tra nhanh bằng mắt.

## File đầu ra

| File | Mô tả |
|------|-------|
| `labels.csv` | Toàn bộ nhãn từ auto-label, kèm review text và metadata |
| `training_labels.csv` | Nhãn đã lọc và chuẩn hóa, sẵn sàng đưa vào training pipeline |
| `ground_truth.csv` | Nhãn kiểm tra thủ công dùng để đánh giá độ chính xác của auto-label |

## Ghi chú

- Việc gán nhãn dùng cả **ảnh** và **review text + product name**.
- Nếu model không chắc chắn hoặc ảnh quá mờ/khó nhận dạng, nên label `irrelevant`.
- Muốn tái lập kết quả random, dùng `--seed` khi download/extract.
- Nếu dùng provider có rate-limit, thêm `--sleep 0.6` đến `1.0` và chạy theo batch. Chạy lại lệnh sẽ tự skip ảnh đã label rồi.
- Gemini hỗ trợ batch đa ảnh: dùng `--batch-size 5` đến `10` để giảm số lượt gọi (chỉ áp dụng khi `--provider google`).
- Để tắt việc copy ảnh vào thư mục `labeled/`, thêm flag `--no-copy`.
