# Multimodal Feature Denoiser — Tài liệu tích hợp

Tài liệu này mô tả pipeline denoising mới (dựa trên MDSBR), các vấn đề cần lưu ý và hướng dẫn tích hợp vào web.

---

## 1. Kiến trúc pipeline mới

### 1.1 Text Sentiment

```
URL Review
    └─► Scrape Text
            └─► PhoBERT (frozen encoder)
                    └─► Embedding [768-dim]
                            └─► MLP Head  (768 → 256 → 128 → 3)
                                    └─► tích cực / trung lập / tiêu cực
```

- PhoBERT giữ nguyên (không fine-tune lại), chỉ dùng để trích xuất embedding 768 chiều.
- MLP head nhẹ (~2MB) được train độc lập trên embedding đó.
- Không đưa qua Denoiser vì Denoiser cần paired image — nếu chỉ có text, embedding sẽ bị lệch phân phối.

---

### 1.2 Image Defect Detection

```
URL Review
    └─► Download ảnh
            └─► ResNet50 (frozen encoder)
                    └─► Embedding [2048-dim]
                            └─► FeatureDenoiser  (Gaussian Diffusion, hidden=1024)
                                    └─► Embedding đã lọc nhiễu [2048-dim]
                                            └─► MLP Head  (2048 → 256 → 128 → 2)
                                                    └─► no-defect / defect
```

- ResNet50 giữ nguyên (frozen), dùng làm feature extractor, không phải classifier.
- FeatureDenoiser lọc nhiễu trong không gian embedding trước khi đưa vào head phân loại.

---

## 2. Vấn đề cần lưu ý

### 2.1 Pipeline mới chưa tích hợp vào web

`ai_engine/main.py` hiện vẫn dùng pipeline cũ (NextGenReviewAnalyzer cho text, MobileNetV3/ResNet50 fine-tuned cho ảnh).

**Tham chiếu khi implement:**
- `scripts/evaluate_pipeline_comparison.py` — pipeline mới đã chạy đầy đủ tại đây. Các hàm `evaluate_new_text_pipeline()` và `evaluate_new_image_pipeline()` chính là logic cần port vào `main.py`.
- `ai_engine/denoising/feature_denoiser.py` — định nghĩa `FeatureDenoiser` và `ClassificationHead`.

**Cách implement:**

**Text sentiment** — chạy PhoBERT fine-tuned (lấy embedding từ `[CLS]` token ở `last_hidden_state[:, 0, :]`), sau đó đưa thẳng vào MLP head. Không qua Denoiser.

> Lưu ý: phải dùng **chính model PhoBERT đã fine-tune** (đặt tại `artifacts/models/phobert/` — xem mục 3 để biết cách tạo lại), không phải base model từ HuggingFace. Lấy `roberta` sub-module từ `AutoModelForSequenceClassification` để bỏ classification head gốc.

**Image defect** — load ResNet50 fine-tuned, set `model.fc = Identity()` sau khi load state dict để lấy embedding 2048-dim thay vì predict class. Truyền embedding đó vào FeatureDenoiser (với `text_input = zeros(1, 768)`), lấy `image_clean` từ output, rồi đưa vào MLP head.

> Lưu ý: phải dùng **chính ResNet50 đã fine-tune** (`resnet50_defect_gpu_best.pth`, đặt tại `ai_engine/models/`) — vì Denoiser và image head được train trên embeddings extract từ model đó. Dùng ResNet50 khác sẽ ra embedding space khác và predict sai.

> Lưu ý: image transform phải giống hệt lúc extract embeddings:
> `Resize((224, 224)) → ToTensor() → Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

---



### 2.2 Trust Score sản phẩm tương tự = random

`ai_engine/main.py` dòng 675 trả về `random.randint(70, 92)` thay vì giá trị thật từ pipeline AI.

Gợi ý: bỏ hiển thị hoặc ghi "Chưa phân tích".

---

### 2.3 IForest Spam Model bị tắt

`ai_engine/main.py` dòng 306-317 đang comment out model IForest vì nhận nhầm comment ngắn bình thường là spam.

Gợi ý: retrain với `contamination=0.03-0.05` hoặc thêm whitelist cho comment ngắn.

---

## 3. File model — không push lên GitHub (trong .gitignore)

Các file model weights bị gitignore (`*.pt`, `*.pth`, `artifacts/models/**`).
Sau khi clone repo, cần train lại hoặc copy thủ công vào đúng đường dẫn:

| File model | Đường dẫn trong repo | Ghi chú |
|---|---|---|
| `feature_denoiser.pt` | `ai_engine/models/feature_denoiser.pt` | Train bằng `scripts/train_denoiser.py` |
| `text_sentiment_head.pt` | `artifacts/models/text_sentiment_head.pt` | Train bằng `scripts/train_classification_heads.py` |
| `image_defect_head.pt` | `artifacts/models/image_defect_head.pt` | Train bằng `scripts/train_classification_heads.py` |
| PhoBERT fine-tuned | `artifacts/models/phobert/` | **Chưa có trong repo** — copy thủ công, phải là model đã dùng để extract embeddings |
| ResNet50 fine-tuned | `ai_engine/models/resnet50_defect_gpu_best.pth` | **Chưa có trong repo** — copy thủ công, phải là model đã dùng để extract embeddings |

---

*Cập nhật: Tháng 7/2026*
