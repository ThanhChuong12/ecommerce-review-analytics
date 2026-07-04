# 🎓 TOÀN BỘ LUỒNG XỬ LÝ HỆ THỐNG — TỪ LÚC DÁN LINK ĐẾN LÚC CÓ KẾT QUẢ

> **Dành riêng cho:** Người làm Web — hiểu rõ phần mình làm, nhưng cần giải thích được phần AI/ML của bạn nhóm khi thầy hỏi.

---

## MỤC LỤC

1. [Kiến trúc tổng quan hệ thống](#1-kiến-trúc-tổng-quan)
2. [PHẦN WEB: Luồng từ Frontend → Backend → Queue](#2-phần-web-luồng-từ-frontend--backend--queue)
3. [PHẦN AI: Luồng 9 bước xử lý của ai_engine](#3-phần-ai-luồng-9-bước-trong-ai_engine)
   - [Bước 1: Scraping — Cào dữ liệu](#bước-1-scraping--cào-dữ-liệu)
   - [Bước 2: Spam Detection — Phát hiện đánh giá rác](#bước-2-spam-detection--phát-hiện-đánh-giá-rác)
   - [Bước 3: Sentiment & Aspect Analysis — Phân tích cảm xúc](#bước-3-sentiment--aspect-analysis--phân-tích-cảm-xúc)
   - [Bước 4: Download ảnh](#bước-4-download-ảnh)
   - [Bước 4.5: CLIP Filter — Lọc ảnh không liên quan](#bước-45-clip-filter--lọc-ảnh-không-liên-quan)
   - [Bước 5: ResNet50 Defect Detection — Nhận diện hỏng hóc](#bước-5-resnet50-defect-detection--nhận-diện-hỏng-hóc)
   - [Bước 6: Cross-Modal Fusion — Tính Trust Score](#bước-6-cross-modal-fusion--tính-trust-score)
   - [Bước 7: LLM Summarization — AI viết lời khuyên](#bước-7-llm-summarization--ai-viết-lời-khuyên)
   - [Bước 8: Similar Products — Tìm sản phẩm thay thế](#bước-8-similar-products--tìm-sản-phẩm-thay-thế)
   - [Bước 9: Webhook — Gửi kết quả về Web](#bước-9-webhook--gửi-kết-quả-về-web)
4. [Sơ đồ toàn bộ luồng (ASCII Diagram)](#4-sơ-đồ-toàn-bộ-luồng)
5. [Bảng Q&A Phòng thủ vấn đáp](#5-bảng-qa-phòng-thủ-vấn-đáp)

---

## 1. KIẾN TRÚC TỔNG QUAN

Hệ thống được xây dựng theo kiến trúc **Microservices** (tách thành nhiều dịch vụ độc lập liên lạc với nhau), gồm **3 service chính**:

| Service | Công nghệ | Vai trò |
|---|---|---|
| **Frontend** | Next.js (React) | Giao diện người dùng, hiển thị kết quả |
| **Backend** | Node.js + Express | Nhận request từ Web, quản lý DB, điều phối queue, WebSocket |
| **AI Engine** | Python + FastAPI | Chạy toàn bộ các mô hình Machine Learning |

**Tại sao lại tách ra như vậy?**

Vì **Node.js** rất giỏi xử lý I/O bất đồng bộ (nhiều kết nối cùng lúc, WebSocket thời gian thực), còn **Python** là vua của thư viện Machine Learning (PyTorch, Transformers, scikit-learn). Nếu nhét hết vào một chỗ thì code sẽ cực kỳ lộn xộn và chậm. Kiến trúc này cho phép 2 team làm song song và nếu AI cần server mạnh hơn thì chỉ cần nâng cấp server Python mà không ảnh hưởng phần Web.

---

## 2. PHẦN WEB: LUỒNG TỪ FRONTEND → BACKEND → QUEUE

### Bước W1: Người dùng dán URL và bấm "Phân tích"

Ở file `web_platform/frontend/src/app/analyze/page.tsx`, hàm `handleAnalyze(url)` được gọi. Nó gửi một request HTTP `POST` lên Backend Node.js:

```
POST http://localhost:5000/api/analyze
Body: { url: "https://tiki.vn/san-pham/..." }
```

### Bước W2: Backend Node.js nhận request (analyzeController.mjs)

File `web_platform/backend/controllers/analyzeController.mjs` được kích hoạt.

**Nếu người dùng đã đăng nhập (`userId` tồn tại):**
- Node.js tạo một bản ghi mới vào bảng `Product` trong **MySQL Database** với trạng thái `status: 'PENDING'`. Đây là để lưu lịch sử phân tích cho người dùng.
- Bản ghi này có `productId` (ví dụ: `42`).

**Nếu khách không đăng nhập:**
- Node.js **KHÔNG** tạo bản ghi DB. Thay vào đó, nó tạo một ID ảo kiểu `temp-1720065234512`. Việc này tránh làm rác database với dữ liệu của người dùng ẩn danh.

Sau đó, dù đăng nhập hay không, Node.js đều đẩy `{ productId, url }` vào **BullMQ Queue (Hàng đợi)** trong Redis và trả về ngay cho Frontend:

```json
{ "success": true, "productId": 42, "jobId": "xyz" }
```

### Bước W3: Tại sao cần Queue (Hàng đợi)?

**Vấn đề:** Xử lý AI có thể mất 1-5 phút. Nếu không có Queue, mà người dùng thứ 2 dán link vào trong lúc người dùng thứ 1 đang xử lý thì server có thể chạy 2 pipeline AI cùng lúc → hết RAM → server sập.

**Giải pháp — BullMQ + Redis:**
- **Redis** là một loại database cực nhanh lưu trữ trong RAM, dùng để lưu "danh sách chờ".
- **BullMQ** là thư viện Node.js cho phép tạo hàng đợi (Queue). Mỗi lần có người yêu cầu phân tích, BullMQ thêm một `job` vào danh sách chờ trong Redis.
- **Worker** (`web_platform/backend/queue/worker.mjs`) là một tiến trình Node.js chạy ngầm, cứ có job trong Queue thì lấy ra xử lý từng cái một.

> **Ví dụ dễ hiểu:** Hàng đợi này giống như hàng chờ tại ngân hàng. Dù có 100 người đến cùng lúc, nhân viên (Worker) vẫn chỉ phục vụ 1 người 1 lần, người còn lại ngồi chờ. Không ai bị mất lượt, server không bị sập.

### Bước W4: Worker gọi AI Engine (worker.mjs)

File `web_platform/backend/queue/worker.mjs` lấy job ra khỏi Queue, sau đó gọi HTTP sang Python:

```
POST http://localhost:8000/process-job
Body: { productId: 42, url: "https://tiki.vn/..." }
```

Lưu ý: Worker cũng check nếu `productId` bắt đầu bằng `temp-` thì **bỏ qua** bước update DB (vì không có bản ghi nào để update).

### Bước W5: Realtime Progress qua Socket.IO

Frontend đã mở sẵn một kết nối **WebSocket** với Backend thông qua thư viện **Socket.IO**.

- WebSocket khác với HTTP thông thường: HTTP thì request → response rồi thôi. WebSocket thì giữ kết nối liên tục, server có thể "đẩy" dữ liệu về client bất kỳ lúc nào mà không cần client phải hỏi.
- Trong lúc Python đang xử lý, cứ xong mỗi bước, Python sẽ gọi một API nhỏ về Node.js (gọi là `webhook/update-progress`). Node.js nhận xong liền đẩy ngay % tiến độ qua Socket.IO về Frontend. Nhờ đó bạn thấy thanh loading cứ tăng dần dần.

### Bước W6: Nhận kết quả cuối (webhookController.mjs)

Khi Python xử lý xong 100%, nó gọi `POST /api/webhook/finished` với toàn bộ kết quả (reviews, trust score, sentiment, ảnh...) gói trong JSON.

File `web_platform/backend/controllers/webhookController.mjs` nhận data:

1. **Nếu không phải temp:** Lưu vào DB (cập nhật bảng `Product`, thêm hàng loạt vào bảng `Review` và `Report`).
2. **Dù có hay không:** Dùng `Socket.IO` đẩy toàn bộ kết quả JSON về Frontend ngay lập tức.

### Bước W7: Frontend render kết quả

Frontend nhận dữ liệu qua Socket, gọi `setState(result)` và React tự động render toàn bộ giao diện: biểu đồ, danh sách review, Trust Score, sản phẩm đề xuất...

---

## 3. PHẦN AI: LUỒNG 9 BƯỚC TRONG AI_ENGINE

Toàn bộ logic nằm trong hàm `heavy_ai_process()` trong file `ai_engine/main.py`. Python nhận `productId` và `url`, chạy 9 bước sau:

---

### BƯỚC 1: SCRAPING — CÀO DỮ LIỆU

**Tiến độ:** 0% → 10%  
**File:** `scraping_agent/scraper/dispatcher.py`

**Mục tiêu:** Lấy toàn bộ review (text, số sao, ảnh, ngày đăng) từ trang sản phẩm về máy chủ.

**Vấn đề cốt lõi:** Các sàn TMĐT (Tiki, Shopee, Lazada...) **không cung cấp API công khai** để lấy review. Để lấy được dữ liệu, nhóm phải dùng kỹ thuật **Web Scraping** — giả lập trình duyệt hoặc gọi trực tiếp API ẩn của sàn.

**Giải pháp — Dispatcher 3 lớp (ưu tiên từ nhanh đến chậm):**

```
URL đến → Dispatcher kiểm tra domain → Chọn Scraper phù hợp
```

**Lớp 1 — Direct API (Nhanh nhất, 5-10 giây):**
- Áp dụng cho: `tiki.vn`, `thegioididong.com`
- Cách hoạt động: Các sàn này tải dữ liệu review từ API nội bộ của họ (ví dụ `https://api.tiki.vn/product-detail/api/v1/review...`). Dù họ không công bố API này, nhưng bằng cách dùng trình duyệt bấm F12 → tab Network → lọc request, nhóm đã tìm ra URL của API đó và gọi thẳng bằng Python `httpx`. **Không cần mở trình duyệt thật, chỉ cần gửi request HTTP thuần túy.**
- Đây giống như bạn biết địa chỉ nhà bếp của nhà hàng và đặt thức ăn thẳng từ đó, thay vì phải ra quầy gọi bồi bàn.

**Lớp 2 — Playwright (Chậm hơn, 20-60 giây, cần giả lập trình duyệt):**
- Áp dụng cho: `shopee.vn`, `lazada.vn` — những sàn có cơ chế chống bot phức tạp.
- **Playwright** là thư viện cho phép code Python điều khiển một trình duyệt Chrome thật (hoặc giả lập không hiển thị màn hình - headless). Nó click, cuộn trang, chờ JavaScript load... y hệt người dùng thật. Vì bot thật sự đang giả làm người dùng nên sàn khó phát hiện hơn.

**Lớp 3 — LLM Browser Agent (Chậm nhất, 1-3 phút, dùng AI điều khiển):**
- Dùng cho các site lạ mà cả 2 lớp trên không nhận diện được.
- Đây là kỹ thuật tiên tiến nhất: AI (Language Model) tự đọc HTML của trang, tự "hiểu" cần click vào đâu để lấy review, tự điều khiển Playwright làm theo.

**Kết quả Bước 1:** Một file CSV tạm thời chứa tất cả review với các cột: `text`, `rating`, `image_urls`, `date`, `product_name`.

---

### BƯỚC 2: SPAM DETECTION — PHÁT HIỆN ĐÁNH GIÁ RÁC

**Tiến độ:** 10% → 25%  
**File:** `ai_engine/text_processing/spam_filter.py`  
**Tình trạng trong code:** Hiện tại chỉ dùng phần **Rule-based**. Phần **IForest** đã được train nhưng đang tắt (comment out) vì độ chính xác chưa đủ tốt.

**Mục tiêu:** Xác định xem review nào là thật, review nào là rác (cày xu, seeding, copy-paste, spam bàn phím).

**Tại sao quan trọng?** Nếu không lọc spam, Trust Score sẽ sai hoàn toàn. Một shop có thể thuê người gõ 5000 bình luận 5 sao giả để lừa khách hàng. Nhiệm vụ của bước này là vạch mặt bọn đó.

**Thuật toán đang dùng: Rule-based (Luật heuristic) + TF-IDF Cosine Similarity**

Đây là cách hoạt động của từng nhóm luật:

#### 🔴 Nhóm Luật 1: AI/Template — Phát hiện bình luận được viết sẵn

Nhóm này bắt những bình luận copy từ một "kịch bản" marketing có sẵn, rất phổ biến trên Lazada.

**Cách hoạt động:**
- Code có một danh sách ~60 câu mẫu marketing (gọi là `_TEMPLATE_PHRASES`):
  - `"công thức sữa tốt nhất cho bé"`, `"hương vị mượt mà và thỏa mãn"`, `"phù hợp với tất cả các loại máy giặt"`, v.v.
- Với mỗi review, code đếm xem có bao nhiêu câu trong danh sách đó xuất hiện.
- Nếu **≥ 2 câu mẫu** xuất hiện trong 1 review → Đánh dấu SPAM.
- Ví dụ: `"Sản phẩm chất lượng cao, hương thơm lâu dài, phù hợp cho quần áo trẻ em"` → chứa 3 câu mẫu → Spam!

#### 🔴 Nhóm Luật 2: Cày Xu (Xu Farming)

**Cách hoạt động:**
- Code có danh sách các từ khóa bán lộ: `"nhận xu"`, `"lấy xu"`, `"đủ ký tự"`, `"viết cho đủ"`, `"hình ảnh mang tính chất nhận xu"`.
- Hễ review chứa bất kỳ từ nào trong danh sách → Spam ngay.
- Đây là trường hợp "người dùng tự thú nhận" — họ ghi thẳng vào review rằng họ viết để lấy xu, không phải để đánh giá sản phẩm.

#### 🔴 Nhóm Luật 3: Nhiễu cấu trúc (Structural Noise)

Phát hiện các review không có nội dung thật:

| Kiểm tra | Ví dụ bị bắt |
|---|---|
| `is_too_short`: < 3 từ | `"ok"`, `"tốt"` |
| `is_emoji_only`: chỉ emoji, không có chữ | `"😍😍😍👌👌"` |
| `is_keyboard_spam`: gõ lặp ký tự | `"aaaaaa"`, `"hahahaha"`, `"tốt tốt tốt tốt"` |
| `is_too_long`: > 500 từ | Bài văn dài 2 trang — có thể copy từ chỗ khác |
| `has_too_many_uppercase`: quá nhiều chữ HOA | `"HÀNG CHẤT LƯỢNG CAO!!! MUA NGAY!!!"` |

#### 🔴 Nhóm Luật 4: Ngoài chủ đề / Quảng cáo

Bắt các review không liên quan đến sản phẩm:
- Chứa **link bên ngoài** (URL): `"mua tại đây: https://..."` → Spam quảng cáo.
- Chứa **số điện thoại / Zalo / Facebook**: `"liên hệ Zalo 0901..."` → Spam liên hệ ngoài sàn.
- **Copy từ bài báo / email / nhà thờ**: Bắt bằng các keyword đặc biệt như `"Subject:", "Dear IT Support"`, `"sư thích minh tuệ"` — đây là những bài copy-paste thực tế tìm thấy trong dataset!

#### 🔴 Nhóm Luật 5: Mâu thuẫn giữa Sao và Nội dung (Rating-Text Mismatch)

**Cách hoạt động:**
- Cho 5 sao nhưng text chứa nhiều từ tiêu cực (tệ, thất vọng, hỏng, fake...) → Nghi ngờ bị ép rating.
- Cho 1-2 sao nhưng text chứa nhiều từ tích cực (tuyệt, hoàn hảo, hài lòng...) → Nghi ngờ review bẩn.
- Chỉ kiểm tra review dài ≥ 15 từ để tránh false positive với review ngắn bình thường.

#### 🔴 Nhóm Cuối: TF-IDF + Cosine Similarity — Bắt Seeding (Buff review copy-paste hàng loạt)

Đây là phần ML thực sự trong bước Spam Detection.

**Vấn đề:** Một số shop thuê người viết cùng một câu khen hàng trăm lần với các biến thể nhỏ. Ví dụ:
- Review 1: `"Hàng đẹp, giao nhanh, đóng gói cẩn thận, rất hài lòng"`
- Review 2: `"Hàng đẹp, giao rất nhanh, đóng gói cẩn thận, hài lòng lắm"`
- Review 3: `"Hàng đẹp lắm, giao nhanh lắm, đóng gói cẩn thận, rất ưng"`

**Giải pháp — TF-IDF:**

**TF-IDF** là viết tắt của **Term Frequency - Inverse Document Frequency**. Đây là kỹ thuật biến văn bản thành vector số học.

- **TF (Term Frequency):** Đếm xem một từ xuất hiện bao nhiêu lần trong review này. Từ xuất hiện nhiều → quan trọng hơn.
- **IDF (Inverse Document Frequency):** Giảm trọng số những từ xuất hiện trong NHIỀU review (vì những từ phổ biến như "và", "là", "của" thì ít có giá trị phân biệt). Tăng trọng số những từ hiếm và đặc trưng.
- **Kết quả:** Mỗi review được biểu diễn thành một vector (danh sách số) dài, mỗi chiều tương ứng với một từ trong toàn bộ tập từ vựng.

**Cosine Similarity:**

Sau khi có vector, code tính **Cosine Similarity** giữa tất cả các cặp review. Cosine Similarity đo góc giữa 2 vector:
- Nếu 2 vector cùng hướng (góc = 0°): `similarity = 1.0` → Hai review giống hệt nhau.
- Nếu 2 vector vuông góc (góc = 90°): `similarity = 0.0` → Hai review hoàn toàn khác nhau.

**Kết luận:** Nếu 2 review có Cosine Similarity ≥ 0.85 (giống nhau ≥ 85%) → Gom vào cùng 1 cụm (cluster) và đánh dấu là **Seeding/Duplicate**. Toàn bộ cluster đó bị gắn `is_spam = 1`.

> **Giải thích cho thầy:** *"Thưa thầy, TF-IDF là phương pháp vector hóa văn bản cổ điển trong NLP. Em dùng nó thay vì dùng embedding deep learning vì nó không cần GPU, chạy nhanh hơn 10-50 lần, và hiệu quả tương đương cho bài toán phát hiện bản sao gần giống. Cosine Similarity là chỉ số đo độ tương đồng ngữ nghĩa giữa 2 văn bản sau khi biểu diễn dưới dạng vector."*

**Kết quả Bước 2:** Mỗi review có thêm cờ `is_spam = 0` (sạch) hoặc `1` (spam). Biến `spam_pct` chứa % spam tổng.

---

### BƯỚC 3: SENTIMENT & ASPECT ANALYSIS — PHÂN TÍCH CẢM XÚC

**Tiến độ:** 25% → 42%  
**Files:** `ai_engine/models/phobert_model.py`, `ai_engine/models/text_baseline.py`, `ai_engine/text_processing/embeddings.py`

**Mục tiêu:** Với mỗi review, xác định:
1. **Sentiment (Cảm xúc):** `positive` (tích cực), `negative` (tiêu cực), `neutral` (trung lập).
2. **Aspect (Khía cạnh):** Review đang nói về `product` (chất lượng), `shipping` (giao hàng/đóng gói), `price` (giá cả) hay `service` (dịch vụ)?

**Tại sao không chỉ nhìn vào số sao?** Vì người dùng Việt Nam hay đánh giá không nhất quán: cho 5 sao nhưng chê hàng giả, cho 1 sao vì ship chậm dù hàng tốt... Phân tích ngữ nghĩa của chữ mới phản ánh thực tế.

#### Phân tích Sentiment — Fallback Pipeline

Code dùng chiến lược "thử từ tốt nhất → nếu thất bại thì dùng cái tiếp theo":

**Cấp 1 — PhoBERT (Tốt nhất):**

**PhoBERT** là một mô hình AI được nhóm nghiên cứu VinAI (Việt Nam) phát triển riêng cho **tiếng Việt**, dựa trên kiến trúc **BERT** (Bidirectional Encoder Representations from Transformers).

Để hiểu BERT, hãy nghĩ thế này: Trước khi có BERT, các mô hình NLP đọc câu từ trái sang phải (như người đọc bình thường). Nhưng BERT đọc **cả hai chiều cùng lúc** (Bidirectional), nên nó hiểu ngữ cảnh của một từ dựa trên cả những từ trước **và** sau nó.

Ví dụ câu: `"sản phẩm không tốt giao hàng lại rất nhanh"`
- Mô hình 1 chiều: Thấy "không tốt" → tiêu cực, xong.
- BERT: Thấy "không tốt" nhưng cũng thấy "giao hàng lại rất nhanh" → hiểu đây là review khen dịch vụ giao hàng nhưng chê chất lượng hàng → phân tích phức tạp hơn.

**PhoBERT được Fine-tuned:** Nhóm bạn đã lấy mô hình PhoBERT gốc và **fine-tune** (huấn luyện thêm) trên dataset review TMĐT Việt Nam đã được gán nhãn. Quá trình này dạy PhoBERT nhận ra ngôn ngữ đặc thù của người mua hàng online Việt (tiếng lóng, viết tắt, emoji...).

**Output của PhoBERT:** Với mỗi review, model trả về **3 xác suất**: `[P(tiêu cực), P(tích cực), P(trung lập)]` tổng bằng 1.0. Ví dụ: `[0.05, 0.90, 0.05]` → Review này tích cực với xác suất 90%.

**Cấp 2 — Text Ensemble (Fallback nếu không có PhoBERT):**

Đây là một mô hình Ensemble truyền thống hơn (kết hợp nhiều mô hình sklearn như Logistic Regression, SVM...) train trên cùng dataset. Kém chính xác hơn PhoBERT nhưng nhẹ hơn nhiều.

**Cấp 3 — Rating Prior (Fallback cuối cùng nếu cả 2 thất bại):**

Đây là cách đơn giản nhất: dùng số sao để suy ra cảm xúc.
- Rating 4-5 sao → `positive`
- Rating 1-2 sao → `negative`
- Rating 3 sao → `neutral`

#### Phân tích Aspect — Sentence Embedding + Cosine Similarity

**Sentence Embedding (DeepEmbedder):** Là kỹ thuật biến một **câu văn** thành một vector số học sao cho các câu có **nghĩa giống nhau** thì có vector **gần nhau** trong không gian toán học.

Ví dụ trong không gian vector:
- `"giao hàng siêu chậm"` ↔ `"vận chuyển mất cả tuần"` → 2 vector rất gần nhau (vì cùng nói về giao hàng chậm)
- `"hàng bị móp"` ↔ `"giao hàng siêu chậm"` → 2 vector xa nhau (khác chủ đề)

**Cách dùng:**
1. Code định nghĩa 4 "neo ngữ nghĩa" (anchor) cho 4 khía cạnh:
   - `shipping: "giao hàng đóng gói thời gian vận chuyển nhanh chậm"`
   - `product: "chất lượng sản phẩm chính hãng hàng giả hàng nhái"`
   - `price: "giá cả đắt rẻ khuyến mãi voucher"`
   - `service: "dịch vụ chăm sóc khách hàng tư vấn thái độ"`

2. Với mỗi review, encode nó thành vector, rồi tính Cosine Similarity với 4 anchor vector.
3. Nếu similarity với một anchor > 0.65 → Review này đang nói về aspect đó.

**Kết quả Bước 3:** Mảng `sentiments[]` chứa `["positive", "negative", "neutral", ...]` và `text_probs[]` chứa `[{positive: 0.9, negative: 0.05, neutral: 0.05}, ...]` cho từng review.

---

### BƯỚC 4: DOWNLOAD ẢNH

**Tiến độ:** 42% → 55%

**Mục tiêu:** Tải ảnh mà khách hàng đã đăng kèm review về server để phân tích.

Bước này đơn giản: Dùng `urllib` của Python để download từng URL ảnh về thư mục tạm trong máy chủ. Giới hạn tối đa `MAX_IMAGES_PROCESS` ảnh (mặc định 40-50 ảnh) để tránh mất quá nhiều thời gian và storage.

**Vấn đề:** Nhiều khách hàng đăng ảnh **KHÔNG liên quan** đến sản phẩm (selfie, ảnh nhà, ảnh cái cây...) để platform nghĩ họ có đính kèm ảnh và tính xu cho họ. Nếu đưa những ảnh này vào model nhận diện hỏng hóc thì kết quả sẽ vô nghĩa. Đây là lý do cần Bước 4.5.

---

### BƯỚC 4.5: CLIP FILTER — LỌC ẢNH KHÔNG LIÊN QUAN

**Tiến độ:** 55% → 63%  
**File:** `ai_engine/image_processing/zero_shot_clip.py`  
**Model:** `openai/clip-vit-base-patch32`

**Mục tiêu:** Với mỗi ảnh vừa tải về, phân loại xem đây là ảnh **product** (sản phẩm/hộp hàng) hay **irrelevant** (không liên quan). Chỉ ảnh `product` mới được đưa vào Bước 5.

#### CLIP là gì?

**CLIP** (Contrastive Language-Image Pre-Training) là mô hình của OpenAI, được huấn luyện trên **400 triệu cặp (ảnh, text) từ internet**.

Ý tưởng đột phá của CLIP: **Học chung không gian vector cho cả ảnh và văn bản.** Nghĩa là sau khi huấn luyện, một bức ảnh con chó và câu text `"a photo of a dog"` sẽ có **vector rất gần nhau** trong cùng một không gian toán học.

Trước CLIP, muốn phân loại ảnh thì phải: Thu thập dataset, gán nhãn thủ công hàng nghìn ảnh, huấn luyện model mới. Với CLIP, chỉ cần viết text mô tả là phân loại được ngay — gọi là **Zero-shot Classification**.

#### Cách hoạt động trong code:

1. Code định nghĩa 2 nhóm prompt:
   - **PRODUCT_PROMPTS (9 câu):** Mô tả ảnh sản phẩm hợp lệ: `"a photo of a product package or cardboard shipping box"`, `"a consumer product sitting on a table for inspection"`, v.v.
   - **IRRELEVANT_PROMPTS (8 câu):** Mô tả ảnh không liên quan: `"a selfie, portrait, or group photo of people without any product"`, `"cooked food, a meal plated on a dish"`, `"flowers, bouquets, gift baskets"`, v.v.

2. CLIP model encode bức ảnh thành vector ảnh, và encode tất cả 17 câu prompt thành vector text.

3. Tính Cosine Similarity giữa vector ảnh và **từng** vector text:
   - Kết quả: 9 điểm similarity với PRODUCT_PROMPTS, 8 điểm với IRRELEVANT_PROMPTS.

4. **Tính điểm trung bình** của 2 nhóm, nhóm nào có điểm cao hơn thì ảnh thuộc nhóm đó.

5. Nếu label = `irrelevant` → Đánh dấu index này, **không** đưa vào ResNet50.

> **Giải thích cho thầy:** *"Thưa thầy, kỹ thuật Zero-shot với CLIP là một trong những đột phá lớn nhất của AI trong 3 năm qua. Truyền thống phải dán nhãn hàng nghìn ảnh để train model phân loại. Với CLIP, em chỉ cần viết mô tả bằng tiếng Anh, model tự phân loại được ngay vì nó đã học sự liên kết giữa ngôn ngữ và hình ảnh từ hàng triệu ví dụ trên internet."*

**Kết quả Bước 4.5:** Tập hợp `clip_irrelevant_indices` chứa index của các ảnh bị loại. Chỉ ảnh `product` mới tiếp tục.

---

### BƯỚC 5: RESNET50 DEFECT DETECTION — NHẬN DIỆN HỎNG HÓC

**Tiến độ:** 63% → 72%  
**File:** `ai_engine/image_processing/defect_detection.py`  
**Model:** ResNet50 (đã được fine-tune)

**Mục tiêu:** Với các ảnh đã qua lọc CLIP (ảnh sản phẩm thật), xác định tình trạng: hộp hàng có nguyên vẹn hay bị móp méo/hỏng?

**Output có 2 nhãn:**
- `intact` (no-defect): Sản phẩm/hộp nguyên vẹn, không có vấn đề.
- `damaged` (defect): Sản phẩm/hộp bị móp, xước, vỡ, giao sai hàng.

#### ResNet50 là gì?

**ResNet50** (Residual Network, 50 tầng) là một trong những kiến trúc mạng nơ-ron tích chập (**CNN — Convolutional Neural Network**) nổi tiếng nhất trong Computer Vision (thị giác máy tính).

**CNN học như thế nào?**

Hãy tưởng tượng não người nhìn ảnh: đầu tiên nhận ra các đường thẳng cơ bản, rồi ghép lại thành góc cạnh, rồi ghép thành hình khối, rồi nhận ra vật thể. CNN làm y hệt vậy theo các lớp (layers):

- **Lớp đầu (Early layers):** Học phát hiện các đặc trưng đơn giản như cạnh nằm ngang, cạnh thẳng đứng, góc 45°.
- **Lớp giữa (Middle layers):** Kết hợp các đặc trưng đơn giản để nhận ra texture (bề mặt), pattern (hoa văn).
- **Lớp cuối (Deep layers):** Kết hợp tất cả để nhận ra vật thể hoàn chỉnh: "đây là hộp carton bị móp".

**Vấn đề của mạng sâu:** Khi tăng số lớp lên 50+, thông tin gradient trong quá trình huấn luyện có thể biến mất (Vanishing Gradient) khiến model không học được. ResNet giải quyết vấn đề này bằng **Residual Connection (Skip Connection)** — tạo đường tắt cho thông tin bypass qua nhiều lớp, đảm bảo gradient luôn chạy được.

**Fine-tuning (Huấn luyện thêm):**

ResNet50 ban đầu được pre-train để nhận dạng 1000 loại vật thể thông thường (chó, mèo, xe hơi...). Để nó nhận dạng được hộp hàng móp, nhóm bạn đã **fine-tune**: Thay lớp output cuối cùng từ 1000 nhãn thành 2 nhãn (`intact`/`defect`), rồi train thêm trên dataset ảnh hộp hàng đã gán nhãn thủ công.

Quá trình fine-tune giống như: Bạn học đại học về Cơ khí (kiến thức nền = pre-training), rồi đi học thêm khóa "sửa điện thoại" 1 tháng (fine-tuning) để chuyên sâu vào một lĩnh vực hẹp.

**Inference (Dự đoán):**

Model nhận ảnh đầu vào (resized về 224x224 pixel), xử lý qua 50 lớp tích chập, và output ra **xác suất**: `{"intact": 0.85, "damaged": 0.15}` → Ảnh này có 85% khả năng là hàng nguyên vẹn.

**Batch Inference:** Thay vì xử lý từng ảnh một (chậm), code dùng `batch_size=16` — xử lý 16 ảnh cùng lúc trên GPU/CPU để tận dụng tối đa phần cứng.

**Kết quả Bước 5:** Mảng `image_labels[]` chứa `["intact", "damaged", "irrelevant", ...]` và `image_probs_dict[]` chứa xác suất cho từng ảnh.

---

### BƯỚC 6: CROSS-MODAL FUSION — TÍNH TRUST SCORE

**Tiến độ:** 72% → 82%  
**File:** `ai_engine/fusion/fusion_engine.py`  
**Class:** `TrustScoreCalculator`

**Mục tiêu:** Tổng hợp **3 luồng thông tin** từ các bước trước thành **1 điểm Trust Score duy nhất (0-100)** cho mỗi review, sau đó lấy trung bình để ra Trust Score tổng của sản phẩm.

**"Đa phương thức" (Multimodal) có nghĩa là gì?**

Thay vì chỉ dựa vào 1 loại dữ liệu (chỉ text HOẶC chỉ ảnh), hệ thống kết hợp **nhiều loại dữ liệu cùng lúc** (text + ảnh + metadata spam). Đây chính là điểm đặc biệt và độc đáo nhất của đề tài này.

#### 3 tín hiệu đầu vào:

| Tín hiệu | Nguồn | Ví dụ |
|---|---|---|
| `TextProbs` | PhoBERT (Bước 3) | `{positive: 0.9, negative: 0.05, neutral: 0.05}` |
| `ImageProbs` | ResNet50 (Bước 5) | `{intact: 0.85, damaged: 0.10, wrong_item: 0.03, irrelevant: 0.02}` |
| `AuthMeta` | Spam Detection (Bước 2) | `{is_spam: False, spam_score: 0.0}` |

#### Công thức tính Trust Score:

**Bước F1 — Gatekeeper (Bộ lọc tuyệt đối):**

Nếu review bị đánh dấu `is_spam = True` → **Dừng ngay**, trả về điểm thấp (~5-20 điểm) tùy mức độ spam. Review spam không được tính điểm bình thường. Code:

```python
SPAM_PENALTY_SCORE = 5.0
SPAM_MILD_CEILING = 20.0
penalty_score = SPAM_PENALTY_SCORE + (1.0 - severity) * (SPAM_MILD_CEILING - SPAM_PENALTY_SCORE)
# severity cao → penalty_score gần 5 (rất nặng)
# severity thấp → penalty_score gần 20 (nhẹ hơn)
```

**Bước F2 — Định tuyến ảnh (Image Routing):**

- Không có ảnh → Phân bổ trọng số của ảnh sang cho text (text_weight tăng từ 0.4 lên 0.8).
- Ảnh không liên quan (irrelevant > 0.4) → Tương tự, bỏ qua ảnh, dùng text nhiều hơn.
- Ảnh hợp lệ → Dùng đủ cả text (0.4) + ảnh (0.4) + auth (0.2).

**Tại sao phải tái phân bổ trọng số?** Vì nếu không có ảnh mà vẫn tính `image_score = 0` với weight 40%, thì mọi review không có ảnh đều bị phạt oan. Thay vào đó, nếu không có ảnh thì tín hiệu text được tin tưởng hơn (trọng số tăng lên).

**Bước F3 — Confidence-Aware Weighting:**

Không phải lúc nào model cũng "chắc chắn" với dự đoán của mình. Một review khó (nhiều mặt tích cực lẫn tiêu cực) thì PhoBERT có thể cho ra `[0.4, 0.35, 0.25]` — không rõ ràng. Một review dễ thì ra `[0.95, 0.03, 0.02]` — rất chắc chắn.

Code đo độ chắc chắn bằng **Top-2 Margin**: Hiệu số giữa xác suất cao nhất và xác suất cao thứ 2. Margin cao = model chắc, margin thấp = model không chắc.

- Review chắc chắn: `[0.95, 0.03, 0.02]` → Margin = 0.95 - 0.03 = 0.92 → Trọng số cao hơn.
- Review mơ hồ: `[0.40, 0.35, 0.25]` → Margin = 0.40 - 0.35 = 0.05 → Trọng số thấp hơn.

**Bước F4 — Tính điểm cơ bản:**

```python
score_text = 50.0 + 50.0 * (positive - negative)
# → Nếu positive=0.9, negative=0.05: score_text = 50 + 50*(0.85) = 92.5
# → Nếu positive=0.1, negative=0.8: score_text = 50 + 50*(-0.7) = 15

image_score = intact * 100 - (damaged + wrong_item) * 50
# → Nếu intact=0.9, damaged=0.1: image_score = 90 - 5 = 85
# → Nếu intact=0.1, damaged=0.9: image_score = 10 - 45 = -35 → clamp về 0

score_auth = 100.0  # Không spam → điểm auth tối đa

final_score = score_text * eff_text_w + image_score * eff_img_w + score_auth * eff_auth_w
```

**Bước F5 — Phát hiện Conflict (Mâu thuẫn đa phương thức):**

Đây là điểm thông minh nhất của Fusion Engine:

- **Conflict 1 (Nghiêm trọng):** Text khen (positive > 0.6) NHƯNG ảnh hỏng hóc (damaged > 0.6) → Nghi ngờ review khen giả / bị ép review. Phạt nặng: nhân điểm với 0.5 (giảm một nửa).
- **Conflict 2 (Nhẹ hơn):** Text chê (negative > 0.6) NHƯNG ảnh hoàn hảo (intact > 0.8) → Có thể khách chê vì lý do khác (ship chậm) chứ không phải hàng hỏng. Phạt nhẹ: nhân với 0.8.

**Trust Score tổng của sản phẩm:**

```python
# CHỈ lấy trung bình của non-spam reviews
non_spam_scores = [score[i] for i in range(N) if not is_spam[i]]
overall_trust = mean(non_spam_scores)
```

Spam reviews bị penalize về ~5-20 điểm. Nếu tính trung bình cả spam vào thì Trust Score sẽ bị kéo xuống quá thấp dù hàng thực sự tốt. Nên chỉ tính từ non-spam reviews để phản ánh đúng chất lượng thực.

> **Giải thích cho thầy:** *"Thưa thầy, điểm Trust Score của em không phải là điểm sao trung bình đơn giản. Đây là một chỉ số tổng hợp (composite index) được tính từ xác suất thực của 2 mô hình AI độc lập (PhoBERT cho văn bản, ResNet50 cho hình ảnh) được kết hợp thông qua thuật toán Fusion có trọng số động, có phát hiện mâu thuẫn, có lọc spam. Điều này làm cho Trust Score của em robust (bền vững) hơn nhiều so với chỉ dựa vào rating."*

---

### BƯỚC 7: LLM SUMMARIZATION — AI VIẾT LỜI KHUYÊN

**Tiến độ:** 82% → 90%  
**File:** `ai_engine/llm_integration/llm_client.py`

**Mục tiêu:** Tạo ra đoạn tóm tắt ngắn gọn, dễ đọc bằng tiếng Việt cho người dùng (hiển thị ở đầu trang kết quả).

**LLM là gì?**

**LLM** (Large Language Model) là các mô hình ngôn ngữ khổng lồ như GPT-4, Llama, Gemini... Chúng được huấn luyện trên hàng nghìn tỷ từ văn bản từ internet, sách, bài báo... nên có khả năng viết, tóm tắt, phân tích văn bản cực kỳ tốt.

**Cách code dùng LLM:**

1. Code gom thống kê lại:
   - Số lượng review tích cực/tiêu cực/trung lập.
   - Trust Score.
   - Lấy 3 review khen hay nhất và 3 review chê gay gắt nhất làm ví dụ.

2. Gửi tất cả lên LLM API (Groq/OpenAI) kèm **System Prompt** (chỉ dẫn cho AI):
   ```
   "Bạn là AI chuyên phân tích đánh giá sản phẩm. 
   Dựa vào dữ liệu thống kê và các review tiêu biểu, 
   hãy viết MỘT đoạn văn ngắn (3-4 câu) tóm tắt chất lượng sản phẩm 
   và đưa ra lời khuyên cho người mua (Nên mua hay Cẩn thận)."
   ```

3. LLM viết ra đoạn tóm tắt, code ghép thêm prefix (✅ / ❌ / ⚠️) tùy vào Trust Score và tỉ lệ review.

**Fallback (nếu API thất bại):** Code tự tạo câu tóm tắt cứng từ template có sẵn — không cần LLM.

> **Giải thích cho thầy:** *"Thưa thầy, em không dùng LLM để phân tích review vì 2 lý do: 1) Chi phí API rất đắt khi xử lý hàng nghìn review. 2) LLM có thể hallucinate (bịa đặt). Em chỉ dùng LLM ở bước cuối để viết tóm tắt dựa trên số liệu đã được tính toán chính xác bởi các model chuyên dụng (PhoBERT, ResNet50). Dữ liệu đầu vào cho LLM là số liệu thực → LLM chỉ làm nhiệm vụ diễn đạt lại thành ngôn ngữ tự nhiên."*

---

### BƯỚC 8: SIMILAR PRODUCTS — TÌM SẢN PHẨM THAY THẾ

**Tiến độ:** 90% → 95%  
**File:** `scraping_agent/similar_products_fetcher.py`

**Mục tiêu:** Tìm các sản phẩm tương tự trên cùng sàn để gợi ý cho người dùng (đặc biệt hữu ích khi sản phẩm đang xem có Trust Score thấp).

**Cách hoạt động:** Gọi API gợi ý của sàn (Tiki, Shopee...) với keyword lấy từ tên sản phẩm, lấy về danh sách 5 sản phẩm tương tự kèm link, ảnh, tên.

---

### BƯỚC 9: WEBHOOK — GỬI KẾT QUẢ VỀ WEB

**Tiến độ:** 95% → 100%

**Mục tiêu:** Đóng gói toàn bộ kết quả từ 8 bước trên thành 1 JSON khổng lồ và gửi về Node.js.

**JSON payload bao gồm:**

```json
{
  "productId": 42,
  "productData": { "name": "Tên sản phẩm", "thumbnail": "url_ảnh" },
  "reviews": [
    {
      "review_text": "Hàng đẹp lắm...",
      "rating": 5,
      "image_path": "https://...",
      "label": "intact",
      "sentiment": "positive",
      "date": "2024-01-15"
    }
  ],
  "summary": "✅ AI đánh giá đây là sản phẩm đáng tin cậy. ...",
  "metadata": {
    "spamPercentage": 23,
    "trustScore": 82.5,
    "aspectSentiment": { "Product": 4.2, "Packaging": 3.8, "Shipping": 4.0 },
    "sentimentTimeSeries": [...],
    "keywords": {
      "positive": [{"text": "tốt", "value": 45}, ...],
      "negative": [{"text": "chậm", "value": 12}, ...]
    },
    "smartAdvice": "23% đánh giá bị nghi ngờ spam. Sản phẩm đáng tin cậy.",
    "alternativeProducts": [...]
  }
}
```

Python gửi JSON này về `POST /api/webhook/finished` của Node.js. Node.js nhận, lưu DB (nếu cần), và đẩy qua Socket.IO về Frontend.

---

## 4. SƠ ĐỒ TOÀN BỘ LUỒNG

```
NGƯỜI DÙNG (Browser)
        │
        │  Dán URL, bấm "Phân tích"
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js)                       │
│                                                             │
│  handleAnalyze(url)                                         │
│  → POST /api/analyze { url }                                │
│  → Mở kết nối WebSocket để lắng nghe tiến độ               │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP POST
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND (Node.js)                        │
│                                                             │
│  analyzeController.mjs                                      │
│  ├── Có userId? → Tạo Product record trong MySQL DB         │
│  └── Không có? → Tạo productId ảo (temp-xxxx)              │
│                                                             │
│  Đẩy { productId, url } vào BullMQ Queue ──→ Redis          │
│  Return ngay: { success: true, productId }                  │
│                                                             │
│  worker.mjs (chạy ngầm)                                     │
│  ├── Nhặt job từ Queue                                      │
│  └── POST /process-job → AI Engine Python                   │
│                                                             │
│  Đang chờ webhook từ Python...                              │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP POST
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI ENGINE (Python/FastAPI)                │
│                                                             │
│  heavy_ai_process(productId, url) — chạy background        │
│                                                             │
│  STEP 1: Scraping (Dispatcher 3 lớp)          → 10%        │
│    └── Direct API / Playwright / LLM Agent                  │
│         ↓ CSV với tất cả review                             │
│                                                             │
│  STEP 2: Spam Detection                        → 25%        │
│    ├── Rule-based (5 trục: Template/Xu/Noise/OT/Mismatch)   │
│    └── TF-IDF + Cosine Similarity (Seeding Cluster)         │
│         ↓ is_spam[] array                                   │
│                                                             │
│  STEP 3: Sentiment + Aspect                    → 42%        │
│    ├── PhoBERT → xác suất [pos, neg, neu]                   │
│    ├── TextEnsemble (fallback)                              │
│    ├── Rating Prior (fallback cuối)                         │
│    └── DeepEmbedder + Cosine → 4 aspects                    │
│         ↓ sentiments[], text_probs[], aspects[]             │
│                                                             │
│  STEP 4: Download images                       → 55%        │
│         ↓ image_local_paths[]                               │
│                                                             │
│  STEP 4.5: CLIP Zero-shot Filter               → 63%        │
│    └── CLIP-ViT-B/32: ảnh vs 17 text prompts               │
│         ↓ clip_irrelevant_indices (set)                     │
│                                                             │
│  STEP 5: ResNet50 Defect Detection             → 72%        │
│    └── CNN 50 lớp: intact vs damaged                        │
│         ↓ image_labels[], image_probs_dict[]                │
│                                                             │
│  STEP 6: Cross-Modal Fusion                    → 82%        │
│    └── TrustScoreCalculator:                                │
│         Spam Gatekeeper → Image Routing →                   │
│         Confidence Weighting → Base Score →                 │
│         Conflict Detection → Trust Score 0-100              │
│         ↓ overall_trust (float)                             │
│                                                             │
│  STEP 7: LLM CoT Summary                       → 90%        │
│    └── Groq/OpenAI API → đoạn tóm tắt tiếng Việt           │
│         ↓ llm_summary (string)                              │
│                                                             │
│  STEP 8: Similar Products                      → 95%        │
│    └── Scrape API sàn → 5 sản phẩm tương tự               │
│         ↓ alternative_products[]                            │
│                                                             │
│  STEP 9: Webhook → Node.js                     → 99%        │
│    └── POST /api/webhook/finished { JSON lớn }              │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP POST (webhook)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND (Node.js)                        │
│                                                             │
│  webhookController.mjs                                      │
│  ├── Có DB? → Update Product, BulkCreate Review, Report     │
│  └── Dù thế nào → Socket.IO emit "result" → Frontend        │
└───────────────────────────┬─────────────────────────────────┘
                            │ WebSocket
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js)                       │
│                                                             │
│  setState(result) → React render toàn bộ UI                 │
│  ✅ Biểu đồ, danh sách review, Trust Score, bản đồ cảm xúc │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. BẢNG Q&A PHÒNG THỦ VẤN ĐÁP

### Câu hỏi về Kiến trúc

**Q: Tại sao tách Node.js và Python, không viết hết vào 1 chỗ?**

> *"Thưa thầy, đây là kiến trúc Microservices — phân chia trách nhiệm rõ ràng. Node.js tối ưu cho I/O bất đồng bộ: quản lý hàng nghìn kết nối WebSocket đồng thời, truy vấn database, xử lý API — những thứ cần tốc độ phản hồi nhanh. Python tối ưu cho tính toán nặng: PyTorch, Transformers, scikit-learn đều là thư viện Python native. Nếu nhét hết vào 1 server thì code lộn xộn, và khi AI cần scale lên server mạnh hơn sẽ phải di chuyển cả hệ thống. Kiến trúc này cho phép scale độc lập từng phần."*

**Q: BullMQ và Redis dùng để làm gì?**

> *"Thưa thầy, BullMQ là thư viện hàng đợi message. Redis là database in-memory cực nhanh dùng để lưu danh sách chờ. Khi nhiều người dùng yêu cầu phân tích cùng lúc, thay vì chạy song song nhiều pipeline AI tốn RAM có thể làm sập server, BullMQ xếp các request vào queue, Worker xử lý từng cái theo thứ tự. Đây là pattern Producer-Consumer (hay Message Queue) rất phổ biến trong hệ thống phân tán."*

**Q: Socket.IO dùng để làm gì?**

> *"Socket.IO implement WebSocket — một giao thức kết nối 2 chiều liên tục giữa browser và server. Khác với HTTP thông thường (browser phải hỏi server mới được trả lời), WebSocket cho phép server CHỦ ĐỘNG đẩy dữ liệu về browser bất kỳ lúc nào. Dùng để hiển thị tiến độ 10%, 25%, 42%... theo thời gian thực mà không cần người dùng F5."*

---

### Câu hỏi về Spam Detection

**Q: TF-IDF là gì?**

> *"TF-IDF là kỹ thuật biến văn bản thành vector số học, thường dùng trong bài toán truy xuất thông tin (Information Retrieval). TF (Term Frequency) đo tần suất của một từ trong document. IDF (Inverse Document Frequency) giảm trọng số của các từ quá phổ biến như 'và', 'là', 'của' (xuất hiện ở mọi review nên không có giá trị phân biệt), tăng trọng số các từ hiếm và đặc trưng. Kết hợp lại, TF-IDF tạo ra vector đại diện cho từng review trong không gian nhiều chiều."*

**Q: Cosine Similarity là gì?**

> *"Cosine Similarity đo góc giữa 2 vector — cụ thể là cosine của góc đó. Nếu 2 vector cùng hướng (góc 0°), cosine = 1 → 2 review giống hệt nhau. Nếu vuông góc (90°), cosine = 0 → hoàn toàn khác nhau. Thầy có thể hiểu đơn giản: nó đo xem 2 review 'nói cùng một chủ đề với cùng từ ngữ' hay không. Threshold 0.85 có nghĩa là 2 review giống nhau ít nhất 85% → coi là seeding."*

**Q: Tại sao không dùng Deep Learning model để detect spam?**

> *"Thưa thầy, 2 lý do chính. Thứ nhất: chi phí training. Để train Deep Learning phát hiện spam tiếng Việt, cần dataset hàng chục nghìn review được gán nhãn thủ công — rất tốn công. Thứ hai: tính giải thích (Explainability). Khi demo, nếu thầy hỏi 'tại sao review này bị spam?' em có thể nói ngay 'vì nó chứa câu cày xu X' hoặc 'vì giống 85% với 20 review khác'. Với Deep Learning (Black-box), em không thể giải thích được."*

---

### Câu hỏi về Sentiment Analysis

**Q: PhoBERT là gì? BERT là gì?**

> *"BERT (Bidirectional Encoder Representations from Transformers) là mô hình ngôn ngữ của Google, được train để hiểu ngữ cảnh của từ theo cả 2 chiều (trái-phải và phải-trái đồng thời). Điều này giúp BERT xử lý tốt các câu mà nghĩa phụ thuộc vào ngữ cảnh toàn câu, không chỉ từ đứng trước. PhoBERT là phiên bản BERT được pre-train đặc biệt cho tiếng Việt bởi nhóm VinAI, sử dụng corpus văn bản tiếng Việt lớn. Nhóm em đã fine-tune PhoBERT trên dataset review TMĐT để tối ưu cho bài toán cụ thể này."*

**Q: Sentence Embedding là gì? Dùng để làm gì?**

> *"Sentence Embedding là kỹ thuật mã hóa một câu văn thành một vector số học trong không gian nhiều chiều, sao cho các câu có nghĩa tương đồng thì vector của chúng gần nhau trong không gian đó. Ví dụ: 'giao hàng siêu chậm' và 'vận chuyển mất cả tuần' sẽ có vector gần nhau dù không dùng cùng từ. Em dùng kỹ thuật này để phát hiện review đang nói về khía cạnh nào của sản phẩm (shipping, product, price, service) bằng cách so sánh vector review với vector 'neo ngữ nghĩa' của từng aspect."*

---

### Câu hỏi về Image Processing

**Q: CNN là gì? Tại sao dùng ResNet50?**

> *"CNN (Convolutional Neural Network) là kiến trúc mạng nơ-ron được thiết kế đặc biệt để xử lý ảnh. Nó học phát hiện đặc trưng thị giác từ đơn giản đến phức tạp: lớp đầu nhận ra cạnh, lớp giữa nhận ra texture và hình dạng, lớp sâu nhận ra vật thể hoàn chỉnh. ResNet50 giải quyết vấn đề 'Vanishing Gradient' khi mạng quá sâu bằng Skip Connection — tạo đường tắt cho gradient đi thẳng qua nhiều lớp. Nhóm em dùng ResNet50 vì nó cân bằng tốt giữa độ chính xác và tốc độ inference cho bài toán phân loại binary (intact/damaged)."*

**Q: CLIP và Zero-shot là gì?**

> *"CLIP là mô hình của OpenAI được train trên 400 triệu cặp (ảnh, text mô tả ảnh đó) từ internet. Nó học cách ánh xạ cả ảnh và text vào chung một không gian vector, sao cho ảnh và caption mô tả ảnh đó nằm gần nhau. Zero-shot Classification là khả năng phân loại vào các nhóm mà model chưa thấy trong quá trình training, chỉ bằng cách đưa ra mô tả bằng ngôn ngữ tự nhiên. Em dùng CLIP để lọc ảnh không liên quan bằng cách cho model so sánh ảnh với các prompt mô tả ảnh sản phẩm và ảnh không liên quan."*

---

### Câu hỏi về Fusion Engine

**Q: Cross-Modal Fusion là gì?**

> *"Cross-Modal Fusion là kỹ thuật kết hợp thông tin từ nhiều loại dữ liệu khác nhau (text và ảnh trong trường hợp này) để ra quyết định chính xác hơn so với chỉ dùng 1 loại. Giống như bác sĩ khi chuẩn đoán bệnh không chỉ hỏi triệu chứng mà còn xét nghiệm máu, chụp X-quang — kết hợp nhiều nguồn thông tin cho kết quả tin cậy hơn. Trust Score của em kết hợp xác suất từ PhoBERT (text) và ResNet50 (ảnh) với trọng số động theo confidence, và có cơ chế phát hiện mâu thuẫn giữa 2 nguồn."*

**Q: Tại sao Trust Score không phải là điểm sao trung bình?**

> *"Điểm sao trung bình rất dễ bị thao túng: 1000 review 5 sao giả là điểm 5/5 ngay. Trust Score của em khác ở 3 điểm: 1) Lọc spam trước, review giả không được tính. 2) Dùng ngữ nghĩa text thực sự (PhoBERT) chứ không tin vào số sao người dùng tự chấm. 3) Có thêm tín hiệu ảnh từ ResNet50. 4) Phát hiện mâu thuẫn — nếu ai đó khen nhưng ảnh hàng bị móp, Trust Score sẽ bị phạt. Kết hợp lại, Trust Score phản ánh thực tế trung thực hơn rating đơn thuần."*

---

*Chúc bạn vấn đáp thành công! 🎉*

---

### Bổ sung cực mạnh (Dựa trên code thực tế): Tại sao có MobileNetV3 nhưng không dùng?

**Q: Trong code file `defect_detection.py` em có viết hàm chạy model `MobileNetV3`, nhưng tại sao thực tế luồng chính `main.py` lại gọi `ResNet50`? Em viết code vào cho có tụ hay sao?**

> *"Dạ thưa thầy, không phải code thừa đâu ạ. Đây là quá trình nghiên cứu và tối ưu hóa (model iteration) của nhóm em qua 2 phiên bản:*
> 
> *- **Ở Phiên bản V1 (Baseline):** Nhóm em dùng **MobileNetV3**. Ưu điểm của nó là kiến trúc cực nhẹ, suy luận (inference) rất nhanh (~50ms/ảnh trên CPU), phân loại được 4 nhãn. Nhưng nhược điểm là mạng khá nông, khi gặp các vết móp hộp carton mờ hoặc nhỏ, nó hay nhận diện sai.*
> 
> *- **Ở Phiên bản V2 (Production - Đang dùng):** Nhóm quyết định nâng cấp lên **ResNet50** (mạng sâu 50 tầng, trích xuất đặc trưng mạnh hơn rất nhiều). Hơn nữa, thay vì chỉ đổi model, nhóm còn áp dụng kỹ thuật xịn hơn: dùng hàm **Focal Loss** (để ép model học các ca móp méo khó nhìn thay vì chỉ học ca dễ) và kỹ thuật **Oversampling** (nhân bản ảnh lỗi để xử lý việc data bị mất cân bằng trầm trọng 1:37). Ngoài ra tụi em thêm **MLP head** tùy biến với Dropout để chống Overfitting.*
> 
> *Kết quả là ResNet50 có chỉ số **F1-Score cao hơn hẳn**. Nên hiện tại hệ thống Web đang gọi ResNet50 (V2) làm model chính thức. Đoạn code MobileNetV3 (V1) nhóm em giữ lại đóng vai trò là Baseline Model để so sánh (benchmark), hoặc dùng làm Fallback khi server quá tải không chạy nổi ResNet ạ."*
