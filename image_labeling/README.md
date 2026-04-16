# Gán nhãn ảnh từ dữ liệu cào

Thư mục này chứa pipeline offline để tạo dữ liệu train từ media trong review
(ảnh/video). Đây **không** phải backend runtime.

## Pipeline làm gì

1) Đọc CSV đầu ra từ scraper.
2) Tải ảnh từ URL.
3) Tải video và cắt ngẫu nhiên 3 frame lưu thành JPG.
4) Kiểm tra ảnh lỗi/corrupted và xóa.
5) Auto-label bằng vision model (Google/OpenAI/Groq/Custom), có kèm review text + product name.
6) Xuất labels CSV và (tuỳ chọn) copy ảnh vào thư mục theo nhãn.

## Cấu trúc thư mục

```
image_labeling/
  media_pipeline.py
  requirements.txt
  data/
    raw_media/        # downloaded images and videos
    frames/           # extracted video frames (JPG)
    manifests/
      media.csv       # all downloaded media items
      images.csv      # all image items (downloaded + frames)
      labels.csv      # auto-label output
    labeled/
      intact/
      damaged/
      irrelevant/
```

## Cài đặt (khuyến nghị)

Tạo môi trường ảo riêng cho thư mục này:

```bash
cd image_labeling
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Đảm bảo API key đã có trong `.env` ở root repo (tùy provider):

```
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
GROQ_API_KEY=...
CUSTOM_API_KEY=...         # optional
CUSTOM_BASE_URL=...        # optional (OpenAI-compatible)
```

## Cách dùng

### 1) Tải media từ CSV của scraper

CSV đầu vào cần có cột `product_name` (bạn có thể bổ sung sau khi chạy scraper).

```bash
python media_pipeline.py download --csv "..\scraping_agent\data"
```
Nếu muốn tải file cụ thể thay vì toàn bộ CSV, dùng `--csv` với đường dẫn đến file đó.
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

### 4) Tạo manifest ảnh (ảnh tải + frame video)

```bash
python media_pipeline.py build-images
```

### 5) Auto-label (multi-provider)

```bash
python media_pipeline.py label --provider <google|openai|groq|custom> --model <MODEL_NAME> --max-images 200 --sleep 0.8
```

Ví dụ:

```bash
python media_pipeline.py label --provider google --model "gemini-2.0-flash" --max-images 1000 --sleep 0.8
python media_pipeline.py label --provider openai --model "gpt-4.1" --max-images 1000 --sleep 0.8
```

Lệnh này sẽ ghi `data/manifests/labels.csv` và (mặc định) copy ảnh vào
`data/labeled/<label>/` để kiểm tra nhanh bằng mắt.

## Ghi chú

- Việc gán nhãn dùng cả **ảnh** và **review text + product name**.
- Nếu model không chắc chắn hoặc ảnh quá mờ/khó nhận dạng, nên label `irrelevant`.
- Muốn tái lập kết quả random, dùng `--seed` khi download/extract.
- Nếu dùng API key free dễ bị rate-limit, nên chạy theo batch 1000-2000 ảnh/lần và thêm `--sleep 0.6` đến `1.0` giây. Chạy lại lệnh sẽ tự skip ảnh đã label rồi.
