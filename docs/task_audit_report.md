# ✅ Audit Report — Kiểm tra 4 Tasks

> **Ngày kiểm tra:** 05/06/2026  
> **Workflow chuẩn:** `.agents/skills/senior-ml-engineer/SKILL.md`  
> **Phương pháp:** Đọc toàn bộ source code, kiểm tra file weights, results JSON, và các TODO marker.

---

## TASK 1 — Huấn luyện Baseline Image Model (ResNet50 / MobileNetV3)

### Yêu cầu đặt ra
> Áp dụng Transfer Learning với ResNet50 / MobileNetV3 trên tập dữ liệu Kaggle để nhận diện ảnh hộp hư hỏng.

### Bằng chứng kiểm tra

| Artifact | Tồn tại? | Nội dung |
|---|---|---|
| `ai_engine/models/image_baseline.py` | ✅ Có | 490 dòng — Full implementation |
| `scripts/train_image_baseline.py` | ✅ Có | 184 dòng — CLI đầy đủ |
| `ai_engine/models/weights/mobilenet_v3_defect.pt` | ✅ Có | **17 MB** — Weights đã train |
| `ai_engine/models/weights/resnet50_defect.pt` | ❌ Không có | Chỉ có MobileNetV3 |
| `ai_engine/models/results/image_baseline_results.json` | ✅ Có | Metrics đã ghi lại |

### Kết quả train đã có (từ `image_baseline_results.json`)

```json
{
  "overall_accuracy": 0.9806  ← 98.06%
  "macro_f1":         0.9473  ← 94.73% ✅ (target ≥ 0.85)
  "training_minutes": 49.4
  "evaluated_at": "2026-05-15T01:40:16"

  "per_class": {
    "damaged":    { "f1": 0.912, "support": 101 }   ← 91.2%
    "intact":     { "f1": 0.982, "support": 4241 }  ← 98.2%
    "irrelevant": { "f1": 0.981, "support": 3628 }  ← 98.1%
    "wrong_item": { "f1": 0.914, "support": 17 }    ← 91.4%
  }
}
```

### Chi tiết code kiểm tra

**`ai_engine/models/image_baseline.py`** đã implement đầy đủ:
- ✅ `ImageBaselineModel(backbone="resnet50" | "mobilenet_v3")`
- ✅ `fit(data_dir, epochs, batch_size, lr, val_split, patience)` — Train với WeightedRandomSampler + Label Smoothing + Early Stopping
- ✅ `predict(image_path)` → `{"label", "confidence", "probabilities", "inference_ms"}`
- ✅ `predict_batch(image_paths, batch_size)` — Batch inference hiệu quả
- ✅ `evaluate(data_dir)` → classification_report đầy đủ
- ✅ `save(filepath)` / `load(filepath)` — Serialize/deserialize model

**Kiến trúc fine-tuning:**
```
ResNet50 (ImageNet pretrained, frozen)
  └── Dropout(0.4) → Linear(2048→256) → ReLU → Dropout(0.2) → Linear(256→4)

MobileNetV3-Large (ImageNet pretrained, frozen)
  └── Dropout(0.3) → Linear(1280→4)

2-stage unfreezing: Freeze → 3 epoch → Unfreeze last block (differential LR = lr/10)
4 classes: intact | damaged | wrong_item | irrelevant
```

### ⚠️ Vấn đề tìm thấy

1. **Thiếu ResNet50 weights**: Chỉ có `mobilenet_v3_defect.pt` (17MB). File `resnet50_defect.pt` **không tồn tại** trong `ai_engine/models/weights/`. Results JSON cũng chỉ ghi 1 entry (không có backbone field → không rõ là ResNet hay MobileNet).
2. **Data source là `image_labeling/data/labeled/`** (từ labeled images của dự án), **không phải Kaggle**. Script comment nói "Dữ liệu đầu vào: image_labeling/data/labeled/ (ImageFolder format)" — không có evidence dùng Kaggle dataset.
3. **Results JSON thiếu `backbone` field** → không xác định được đây là kết quả của backbone nào.

### 🏁 Verdict

```
✅ PARTIALLY DONE — MobileNetV3 đã train thành công (Macro-F1=0.947, ≥ 0.85 target)
❌ ResNet50 weights KHÔNG TỒN TẠI → chưa train hoặc chưa lưu
❌ Không có bằng chứng dùng Kaggle dataset — dùng data nội bộ thay thế
⚠️  Results JSON thiếu backbone field → cần kiểm tra lại
```

---

## TASK 2 — Cập nhật Scraper: Cào 5 Sản phẩm tương tự (Shopee/Tiki/Lazada)

### Yêu cầu đặt ra
> Viết hàm cào 5 "Sản phẩm tương tự" trên Shopee/Tiki/Lazada làm mồi cho Web.

### Bằng chứng kiểm tra

| Artifact | Tồn tại? | Nội dung |
|---|---|---|
| `scraping_agent/similar_products_fetcher.py` | ✅ Có | 114 dòng — Public API |
| `scraping_agent/scraper/direct/similar_products.py` | ✅ Có | 362 dòng — Implementations |
| `scraping_agent/scraper/models.py` | ✅ Có | `SimilarProduct` dataclass |

### Chi tiết implementation

**`similar_products_fetcher.py`** — Entry point công khai:
```python
from scraping_agent.similar_products_fetcher import scrape_similar_products

products = asyncio.run(scrape_similar_products(
    url="https://tiki.vn/...",
    limit=5          # ← đúng yêu cầu
))
# → list[SimilarProduct(name, url, price, rating, sold, image_url, source)]
```

**3 implementaions trong `similar_products.py`:**

| Platform | Class | Method | Ghi chú |
|---|---|---|---|
| Tiki | `TikiSimilar` | Direct API (httpx) | 3 endpoint fallback, nhanh nhất |
| Lazada | `LazadaSimilar` | Playwright + network intercept | Bắt API `api/v1/recommend` |
| Shopee | `ShopeeSimilar` | Playwright + network intercept | Bắt API `api/v4/item/get_related` |

**Shopee normalization** (xử lý đúng Shopee price ×100,000):
```python
price = int(price_raw)
if price > 10_000_000_000:
    price //= 100_000  # ← đúng pattern từ SKILL.md
```

**Lazy import dispatcher:**
```python
_SITE_MAP = {
    "tiki.vn":   "scraper.direct.similar_products.TikiSimilar",
    "lazada.vn": "scraper.direct.similar_products.LazadaSimilar",
    "shopee.vn": "scraper.direct.similar_products.ShopeeSimilar",
}
```

### ⚠️ Vấn đề tìm thấy

1. **Chưa có integration test** — Không có file test verify hoạt động thực tế với URL thật.
2. **LazadaSimilar** dùng Playwright headless — có thể bị block giống như scraper review nếu không có session/cookie.
3. **Kết quả trả về** chưa được dùng ở đâu trong `ai_engine/main.py` (vẫn là mock data).
4. **TGDD** không có similar products fetcher (chỉ hỗ trợ 3/4 platform).

### 🏁 Verdict

```
✅ DONE — Code đầy đủ, implementation đúng yêu cầu (limit=5, cả 3 platform)
✅ Tiki: Direct API (nhanh, không cần browser)
✅ Lazada: Playwright intercept (đúng pattern)
✅ Shopee: Playwright intercept + price normalization đúng
⚠️  Chưa integrate vào FastAPI main pipeline (vẫn mock)
⚠️  Chưa có test với URL thật để verify
```

---

## TASK 3 — Pipeline Zero-shot Classification bằng CLIP

### Yêu cầu đặt ra
> Xây dựng Pipeline Zero-shot Classification bằng CLIP (OpenAI) để nhận diện các ảnh rác/ảnh không liên quan.

### Bằng chứng kiểm tra

| Artifact | Tồn tại? | Nội dung |
|---|---|---|
| `ai_engine/image_processing/zero_shot_clip.py` | ✅ Có | **45 dòng** — Stub implementation |
| TODO comments trong file | ❌ | 4 TODO chưa complete |
| Integration với pipeline | ❌ | Không dùng ở đâu |

### Nội dung đầy đủ của file (45 dòng)

```python
"""
zero_shot_clip.py
-----------------
TODO:
  - [ ] Load CLIP model (openai/clip-vit-base-patch32)   ← CHƯA DONE
  - [ ] Định nghĩa các label phân loại                   ← CHƯA DONE
  - [ ] Classify ảnh từ link thu thập bởi scraping_agent ← CHƯA DONE
  - [ ] Filter ảnh không liên quan                        ← CHƯA DONE
"""
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

_clip_model = None
_clip_processor = None

def detect_irrelevant_image(image_path: str) -> bool:
    """Trả về True nếu ảnh KHÔNG liên quan đến sản phẩm."""
    global _clip_model, _clip_processor
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _clip_model is None or _clip_processor is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        
    try:
        image = Image.open(image_path).convert("RGB")
        prompts = [
            "a clean product review photo", 
            "an irrelevant photo or spam image..."
        ]
        inputs = _clip_processor(text=prompts, images=image, ...)
        with torch.no_grad():
            outputs = _clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
        return probs[1].item() > probs[0].item()  # True = irrelevant
    except Exception:
        return False
```

### Phân tích chi tiết

**Những gì đã có:**
- ✅ Import đúng: `CLIPModel`, `CLIPProcessor` từ `transformers`
- ✅ Singleton pattern (`_clip_model`, `_clip_processor`)
- ✅ Auto device detection (CUDA/CPU)
- ✅ Hàm `detect_irrelevant_image(image_path)` trả về `bool`
- ✅ Text prompts phân biệt `relevant` vs `irrelevant`

**Những gì còn thiếu nghiêm trọng:**
- ❌ **Chỉ 2 prompts** — Quá đơn giản, không phân biệt được các loại ảnh rác khác nhau
- ❌ **Không có threshold tuning** — So sánh `probs[1] > probs[0]` là 50/50, không tối ưu
- ❌ **Không có batch processing** — Chỉ xử lý 1 ảnh tại một thời điểm
- ❌ **Chưa có `filter_irrelevant_batch(image_paths)`** — Không thể dùng trong pipeline
- ❌ **4 TODO comments** — Tự nhận là chưa hoàn thành
- ❌ **Không có test** — Không có evidence model chạy đúng
- ❌ **Không integrate** vào `ai_engine/main.py`

### 🏁 Verdict

```
❌ STUB ONLY — Code là placeholder, chưa đủ để dùng trong production
- Hàm detect_irrelevant_image() có logic cơ bản nhưng:
  * Prompt quá đơn giản (chỉ 2 text labels)
  * Không có threshold calibration
  * Không có batch processing
  * 4 TODO chưa hoàn thành
  * Không integrate vào pipeline
- Về mặt kỹ thuật: code sẽ CHẠY ĐƯỢC nếu có CLIP installed,
  nhưng chưa được validate và chưa đủ production-grade
```

---

## TASK 4 — Inference Pipeline ONNX (PhoBERT / ResNet / Denoising)

### Yêu cầu đặt ra
> Xây dựng Inference Pipeline: Đóng gói và chuyển đổi toàn bộ mô hình (PhoBERT, ResNet, Denoising) sang định dạng ONNX để tối ưu tốc độ suy luận.

### Bằng chứng kiểm tra

```bash
# Tìm kiếm toàn bộ project với query "onnx"
grep -ri "onnx" . --include="*.py" --include="*.md" --include="*.json"
# → Không có kết quả
```

| Artifact | Tồn tại? |
|---|---|
| Script export ONNX (bất kỳ file nào) | ❌ **Không** |
| File `.onnx` trong bất kỳ thư mục nào | ❌ **Không** |
| `requirements.txt` chứa `onnx` hoặc `onnxruntime` | ❌ **Không** |
| Mention "onnx" trong bất kỳ file code/doc nào | ❌ **Không** |
| `plan.md` có task ONNX | ❌ **Không có** |

### Kiểm tra requirements.txt

```
# ai_engine/requirements.txt — KHÔNG có onnx/onnxruntime
fastapi, uvicorn, torch, torchvision, transformers, scikit-learn,
pandas, numpy, pillow, httpx, requests, googletrans, deepl,
underthesea, opencv-python, albumentations ...
```

### 🏁 Verdict

```
❌ KHÔNG ĐƯỢC THỰC HIỆN — Không có bất kỳ dấu vết nào
- Không có file script export ONNX
- Không có file .onnx nào trong project
- Không có onnxruntime trong requirements
- Không được mention trong plan.md
- Task này chưa được bắt đầu
```

---

## 📊 Tóm tắt Audit Tổng thể

| Task | Trạng thái | Chi tiết |
|---|---|---|
| **1. Baseline Image Model** | 🟡 **Partial** | MobileNetV3 ✅ done (F1=0.947), ResNet50 ❌ weights không tồn tại |
| **2. Similar Products Scraper** | ✅ **Done** | Tiki/Lazada/Shopee đầy đủ, chưa integrate vào pipeline |
| **3. CLIP Zero-shot** | 🔴 **Stub** | Code skeleton có, 4 TODO chưa done, không integrate |
| **4. ONNX Inference Pipeline** | ❌ **Not Started** | Không có bất kỳ code/file nào |

---

## 🎯 Recommended Actions (theo thứ tự ưu tiên)

### P1 — Ngay bây giờ

#### [IMAGE] Train ResNet50 weights còn thiếu
```bash
python scripts/train_image_baseline.py \
    --backbone resnet50 \
    --data-dir image_labeling/data/labeled \
    --epochs 15 \
    --lr 5e-4
# → Lưu vào ai_engine/models/weights/resnet50_defect.pt
```

#### [CLIP] Hoàn thiện zero_shot_clip.py

File cần thêm:
1. Mở rộng prompts (5-7 labels thay vì 2)
2. Thêm `filter_irrelevant_batch(image_paths, threshold=0.6)`
3. Calibrate threshold trên validation set
4. Xóa 4 TODO comments sau khi hoàn thành

```python
# Prompts mạnh hơn
RELEVANT_PROMPTS = [
    "a photo of a product package or box",
    "a product review image showing an item",
    "a photo showing product condition or damage",
]
IRRELEVANT_PROMPTS = [
    "a screenshot, selfie, pet, or random photo",
    "a receipt, invoice, or document",
    "a blank screen or black image",
    "a food photo or scenery unrelated to product",
]
```

### P2 — Sprint tiếp theo

#### [ONNX] Bắt đầu từ đầu

Cần tạo script `scripts/export_onnx.py`:

```python
# Export MobileNetV3 → ONNX (nhanh nhất, ưu tiên)
import torch
from ai_engine.models.image_baseline import ImageBaselineModel

model = ImageBaselineModel.load("ai_engine/models/weights/mobilenet_v3_defect.pt")
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model.model,
    dummy_input,
    "ai_engine/models/weights/mobilenet_v3_defect.onnx",
    opset_version=17,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch_size"}},
)
```

```bash
# Cài dependencies
pip install onnx onnxruntime

# Chạy export
python scripts/export_onnx.py --model mobilenet_v3

# Benchmark
python -c "
import onnxruntime as ort, numpy as np, time
sess = ort.InferenceSession('ai_engine/models/weights/mobilenet_v3_defect.onnx')
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
t = time.perf_counter()
for _ in range(100):
    sess.run(None, {'image': dummy})
print(f'Avg: {(time.perf_counter()-t)/100*1000:.1f}ms')
"
```

**Thứ tự export:**
1. MobileNetV3 → `.onnx` (dễ nhất, ưu tiên)
2. ResNet50 → `.onnx` (sau khi train xong)
3. PhoBERT → `.onnx` (phức tạp nhất — cần `optimum[onnxruntime]`)

---

## 📝 Update plan.md sau audit này

```markdown
### [ ] [IMAGE] Train ResNet50 weights (còn thiếu)
**Vấn đề:** Chỉ có mobilenet_v3_defect.pt, thiếu resnet50_defect.pt
**Action:** Chạy train_image_baseline.py --backbone resnet50

### [ ] [CLIP] Hoàn thiện zero_shot_clip.py (Stub → Production)
**Vấn đề:** 4 TODO chưa done, 2 prompts quá đơn giản, không có batch
**Action:** Mở rộng prompts + thêm filter_irrelevant_batch() + calibrate threshold

### [ ] [ONNX] Export models sang ONNX format (Not Started)
**Vấn đề:** Chưa có bất kỳ ONNX artifact nào
**Action:** Tạo scripts/export_onnx.py, bắt đầu với MobileNetV3
```
