# Pipeline Xử Lý Ảnh — CLIP + ResNet50

## 1. Tổng Quan Pipeline

Hệ thống xử lý ảnh review thương mại điện tử gồm **2 tầng AI** xếp nối tiếp nhau, mỗi tầng giải quyết một bài toán riêng:

```
Ảnh review từ Shopee/Tiki/Lazada
        │
        ▼
┌─── Tầng 1: CLIP (Bộ lọc) ────────────────┐
│                                            │
│  Câu hỏi: "Ảnh này có phải sản phẩm?"     │
│                                            │
│  ✅ Product  → chuyển xuống tầng 2         │
│  🗑️ Irrelevant → loại bỏ, không xử lý    │
│                                            │
└────────────────┬──────────────────────────┘
                 │ (chỉ ảnh sản phẩm)
                 ▼
┌─── Tầng 2: ResNet50 (Phân loại) ─────────┐
│                                            │
│  Câu hỏi: "Hộp hàng có bị hỏng không?"    │
│                                            │
│  ✅ Intact   → sản phẩm nguyên vẹn        │
│  ❌ Damaged  → sản phẩm bị hỏng/móp       │
│                                            │
└────────────────┬──────────────────────────┘
                 │
                 ▼
         Fusion Engine
    (kết hợp kết quả ảnh + text)
                 │
                 ▼
          Trust Score
```

---

## 2. Tầng 1 — CLIP (Zero-shot Binary Filter)

### 2.1 Vai trò

CLIP đóng vai trò **bộ lọc đầu vào** — loại bỏ ảnh không liên quan trước khi đưa vào ResNet50. Điều này giúp:

- ResNet50 chỉ phải xử lý ảnh sản phẩm thật sự
- Tránh gán nhãn sai cho ảnh selfie, screenshot, đồ ăn...
- Tiết kiệm thời gian inference

### 2.2 Cách hoạt động

CLIP **không cần train** — sử dụng model OpenAI đã học sẵn trên 400 triệu cặp ảnh-text từ internet.

**Nguyên lý:** So sánh ảnh với 16 câu mô tả (prompts) chia 2 nhóm:

| Nhóm | Số câu | Ví dụ prompt |
|---|---|---|
| **Product** (8 câu) | Mô tả rộng, cover mọi loại sản phẩm | "a photo of a product package", "an unboxing photo showing a delivered parcel" |
| **Irrelevant** (8 câu) | Mô tả cụ thể theo pattern rác thực tế | "a selfie or group photo of people", "a screenshot of a phone app" |

**Quá trình phân loại 1 ảnh:**

1. CLIP tính độ tương đồng giữa ảnh và 16 câu → 16 điểm số
2. Lấy trung bình 8 điểm product → 1 số
3. Lấy trung bình 8 điểm irrelevant → 1 số
4. So sánh 2 số → nhóm nào cao hơn thì chọn

### 2.3 Thông số kỹ thuật

| Thông số | Giá trị |
|---|---|
| Model | `openai/clip-vit-base-patch32` (ViT-B/32) |
| Kích thước | ~350 MB |
| Tốc độ | ~150ms/ảnh (CPU) |
| Training | Không cần — zero-shot |
| Prompt version | v3 Hybrid (8 product + 8 irrelevant) |

### 2.4 Hiệu năng đo được (trên 200 ảnh test)

| Metric | Kết quả | Đánh giá |
|---|---|---|
| **Product Recall** | **97%** | ✅ Gần như không mất ảnh sản phẩm nào |
| **False Alarm** (loại nhầm sản phẩm) | **3%** | ✅ Rất ít loại nhầm |
| **Irrelevant Recall** (lọc đúng rác) | **32%** | ⚠️ Chỉ lọc được ~1/3 ảnh rác |
| **Damaged Recall** | **100%** | ✅ Tuyệt vời — không mất ảnh hỏng nào |

### 2.5 CLIP lọc được gì

| Loại ảnh rác | CLIP lọc được? | Ví dụ |
|---|---|---|
| Selfie, ảnh chân dung | ✅ Lọc tốt | Ảnh tự chụp, ảnh nhóm |
| Screenshot điện thoại | ✅ Lọc tốt | Giao diện app, chat |
| Hoa, trang trí Tết | ✅ Lọc được | Bó hoa, đồ trang trí |
| Đồ ăn rõ ràng | ✅ Lọc được | Bát phở, ly trà sữa |
| Quần áo đang mặc | ❌ Khó lọc | Người mặc áo → CLIP nghĩ là sản phẩm |
| Chai nước sốt, sách | ❌ Khó lọc | Vật thể thật → CLIP nghĩ là sản phẩm |
| Xe máy, nội thất | ❌ Khó lọc | Vật thể lớn → CLIP nghĩ là sản phẩm |

> **Lưu ý:** Ảnh rác lọt qua CLIP → ResNet50 gán "intact" (nguyên vẹn) → **vô hại** vì intact = "không hỏng", không ảnh hưởng trust score.

---

## 3. Tầng 2 — ResNet50 (Supervised Defect Detection)

### 3.1 Vai trò

ResNet50 là **bộ phân loại tình trạng hộp hàng** — xác định ảnh cho thấy hộp nguyên vẹn hay bị hỏng.

### 3.2 Cách hoạt động

ResNet50 **đã được train** trên dataset ảnh có gán nhãn thủ công:

- Train trên ~19,400 ảnh labeled (intact + damaged)
- Học pattern: móp méo, rách, ướt, bẹp → **defect**
- Sản phẩm bình thường, hộp nguyên → **no-defect**

### 3.3 Thông số kỹ thuật

| Thông số | Giá trị |
|---|---|
| Model | ResNet50 (pre-trained ImageNet → fine-tuned) |
| Classes | 2: `defect` / `no-defect` |
| Weights | `resnet50_defect_gpu_best.pth` |
| Training | Supervised — đã train trên dataset labeled |
| Threshold | 0.85 confidence |

### 3.4 ResNet50 phân loại được gì

| Tình trạng | Phát hiện được? | Ví dụ |
|---|---|---|
| Hộp móp, bẹp | ✅ Phát hiện tốt | Góc hộp bị lõm, carton bị gãy |
| Hộp rách, thủng | ✅ Phát hiện tốt | Hộp bị xé, lỗ thủng |
| Hộp ướt, bẩn | ✅ Phát hiện được | Vết nước, bẩn trên hộp |
| Hàng nguyên vẹn | ✅ Xác nhận đúng | Hộp đẹp, sản phẩm tốt |
| Sai sản phẩm (wrong_item) | ❌ **Không thể** | Đặt áo đỏ nhận áo xanh |
| Thiếu phụ kiện | ❌ **Không thể** | Thiếu sạc, thiếu dây |
| Chất lượng kém (chất liệu) | ❌ **Không thể** | Vải mỏng, nhựa rẻ |

---

## 4. Kết Hợp CLIP + ResNet50 — Xử Lý Được Gì?

### 4.1 Bảng tổng hợp

| Trường hợp | CLIP | ResNet50 | Kết quả cuối cùng |
|---|---|---|---|
| Hộp hàng nguyên vẹn | ✅ Product | ✅ Intact | **"Nguyên vẹn"** |
| Hộp hàng bị móp/rách | ✅ Product | ✅ Damaged | **"Bị hỏng"** |
| Selfie người dùng | ✅ Irrelevant | — (bỏ qua) | **"Không liên quan"** |
| Screenshot đơn hàng | ✅ Irrelevant | — (bỏ qua) | **"Không liên quan"** |
| Ảnh đồ ăn rõ ràng | ✅ Irrelevant | — (bỏ qua) | **"Không liên quan"** |
| Sai sản phẩm | ✅ Product | ⚠️ Intact | ❌ **Không phát hiện** |
| Ảnh rác giống sản phẩm | ❌ Product | ⚠️ Intact | ❌ **Lọt qua (vô hại)** |

### 4.2 Ước tính coverage trên dataset thực tế

Dựa trên phân bố dataset 27,743 ảnh:

| Nhóm | Tỷ lệ | CLIP + ResNet50 xử lý | Ghi chú |
|---|---|---|---|
| **Intact** | ~45% | ✅ Xử lý tốt | ResNet50 phán intact chính xác |
| **Damaged** | ~30% | ✅ Xử lý tốt | ResNet50 phát hiện defect |
| **Irrelevant** (lọc được) | ~8% | ✅ CLIP lọc đúng | Selfie, screenshot, hoa |
| **Irrelevant** (lọt qua) | ~17% | ⚠️ Gán intact | Vô hại — intact = "không hỏng" |

**Kết luận: ~75% dataset được xử lý chính xác bằng ảnh. ~25% còn lại cần text hỗ trợ.**

---

## 5. Các Vấn Đề KHÔNG Xử Lý Được Bằng Ảnh — Cần Kết Hợp Text

### 5.1 Sai sản phẩm (Wrong Item)

**Vấn đề:** Khách đặt sản phẩm A nhưng nhận sản phẩm B. Cả A và B đều là vật thể thật, CLIP phán "product", ResNet50 phán "intact".

```
Ví dụ:
  - Đặt mua áo đỏ → nhận áo xanh
  - Đặt mua iPhone → nhận Samsung
  - Đặt mua kem chống nắng → nhận sữa rửa mặt

  Ảnh: sản phẩm trông nguyên vẹn
  CLIP: "product" ✅
  ResNet50: "intact" ✅
  → Hệ thống nghĩ mọi thứ ổn
  → NHƯNG thực tế khách hàng rất bực mình!
```

**Giải pháp bằng text:** Phân tích text review tìm dấu hiệu sai hàng:
- "Đặt A nhưng nhận B"
- "Giao sai sản phẩm"
- "Không đúng mẫu", "khác hình"
- "Sai màu", "sai size"

---

### 5.2 Chất lượng sản phẩm kém

**Vấn đề:** Sản phẩm trông bình thường trong ảnh nhưng chất lượng thực tế kém (vải mỏng, nhựa rẻ, pin yếu...). CLIP và ResNet50 chỉ nhìn bề ngoài.

```
Ví dụ:
  - Áo mặc 1 lần thì rách
  - Tai nghe dùng 2 ngày thì hỏng
  - Mỹ phẩm bị dị ứng
  - Đồ ăn không ngon

  Ảnh: sản phẩm nhìn bình thường
  CLIP: "product" ✅
  ResNet50: "intact" ✅
  → Ảnh không cho thấy vấn đề
  → CHỈ CÓ TEXT mới biết chất lượng kém
```

**Giải pháp bằng text:** Phân tích sentiment và keyword:
- "Chất liệu kém", "mỏng quá"
- "Dùng được 1 tuần thì hỏng"
- "Không giống mô tả"
- "Hàng fake", "hàng nhái"

---

### 5.3 Thiếu phụ kiện / Không đủ bộ

**Vấn đề:** Sản phẩm chính có thể nguyên vẹn nhưng thiếu phụ kiện đi kèm. Ảnh chỉ chụp cái có, không chụp cái thiếu.

```
Ví dụ:
  - Mua điện thoại nhưng thiếu sạc
  - Mua giày nhưng thiếu 1 bên
  - Mua combo 5 cái nhưng chỉ nhận 3

  Ảnh: sản phẩm chính trông OK
  CLIP: "product" ✅
  ResNet50: "intact" ✅
  → Không biết thiếu gì
```

**Giải pháp bằng text:**
- "Thiếu sạc", "không có cáp"
- "Chỉ nhận được 3/5"
- "Thiếu phụ kiện"

---

### 5.4 Ảnh rác giống sản phẩm

**Vấn đề:** ~70% ảnh irrelevant trong dataset chứa vật thể thật (quần áo, sách, chai lọ) mà CLIP không phân biệt được với sản phẩm review.

```
Ví dụ:
  - Ảnh chai nước sốt (irrelevant nhưng giống sản phẩm)
  - Ảnh người mặc áo (irrelevant nhưng có quần áo)
  - Ảnh nội thất phòng (irrelevant nhưng có đồ vật)

  CLIP: "product" (sai — nhưng vô hại)
  ResNet50: "intact"
  → Gán "nguyên vẹn" cho ảnh rác
  → Không ảnh hưởng trust score (intact = tốt)
```

**Không cần fix** — gán "intact" cho ảnh rác là **vô hại**. Trust score không bị giảm vì "nguyên vẹn" là kết quả tích cực.

---

### 5.5 Hàng giả / Hàng nhái

**Vấn đề:** Ảnh không thể cho biết sản phẩm là hàng giả hay hàng thật. Cần text review để phát hiện.

```
Ví dụ:
  - "Hàng fake rõ ràng, logo bị lệch"
  - "Không phải hàng chính hãng"
  - "So với hàng auth thì khác xa"
```

**Giải pháp bằng text:** NLP phát hiện keyword hàng giả.

---

## 6. Tóm Tắt: Ảnh vs Text — Ai Xử Lý Gì

| Vấn đề | Ảnh (CLIP + ResNet50) | Text (NLP/Sentiment) | Fusion |
|---|---|---|---|
| Hộp hỏng / móp | ✅ **Ảnh xử lý** | Hỗ trợ | Khớp nhau → trust ↓ |
| Hàng nguyên vẹn | ✅ **Ảnh xử lý** | Hỗ trợ | Khớp nhau → trust ↑ |
| Ảnh rác (selfie, screenshot) | ✅ **Ảnh lọc được** | — | — |
| Sai sản phẩm | ❌ Không thể | ✅ **Text xử lý** | Text phát hiện |
| Chất lượng kém | ❌ Không thể | ✅ **Text xử lý** | Text phát hiện |
| Thiếu phụ kiện | ❌ Không thể | ✅ **Text xử lý** | Text phát hiện |
| Hàng giả/nhái | ❌ Không thể | ✅ **Text xử lý** | Text phát hiện |
| Mâu thuẫn text↔ảnh | ⚠️ Phát hiện 1 phần | ⚠️ Phát hiện 1 phần | ✅ **Fusion phát hiện** |

---

## 7. Fusion Engine — Kết Hợp Ảnh + Text

Fusion Engine là nơi kết hợp kết quả từ cả 2 nguồn để tính Trust Score:

| Text nói | Ảnh cho thấy | → Kết luận |
|---|---|---|
| "Hàng đẹp lắm" (tích cực) | Hộp nguyên (intact) | ✅ Đáng tin → Trust **cao** |
| "Hộp bị móp" (tiêu cực) | Hộp hỏng (damaged) | ✅ Đáng tin → Trust **thấp** |
| "Hàng đẹp" (tích cực) | Hộp hỏng (damaged) | ⚠️ **Mâu thuẫn** → Cần xem lại |
| "Giao sai hàng" (tiêu cực) | Hộp nguyên (intact) | 🔍 Text phát hiện wrong_item |
| "Chất liệu kém" (tiêu cực) | Hộp nguyên (intact) | 🔍 Text phát hiện quality issue |

> **Ảnh phát hiện vấn đề VẬT LÝ. Text phát hiện vấn đề LOGIC. Fusion kết hợp cả hai để đưa ra Trust Score chính xác nhất.**
