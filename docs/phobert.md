# Tài Liệu Kỹ Thuật: Tối ưu hóa & Nâng cấp Pipeline PhoBERT

Tài liệu này ghi lại chi tiết các module đã được hiện thực hóa dựa trên kế hoạch tổng quát, bao gồm những nâng cấp, cải tiến (Upgrades) vượt bậc so với baseline ban đầu để đảm bảo hệ thống vận hành trơn tru trong môi trường thực tế (Production-ready).

## TASK 1: Xây dựng Bộ nạp dữ liệu (Data Pipeline) cho PhoBERT
**File:** `ai_engine/data/phobert_dataset.py`

**Chi tiết hiện thực & Nâng cấp:**
- **Dynamic Padding (Đệm động):** Thay vì đệm tất cả các chuỗi về `max_length=256` cố định (gây lãng phí VRAM), hệ thống chỉ thực hiện truncation. Việc padding được nhường lại cho `DataCollatorWithPadding` trong lúc tạo batch, giúp tiết kiệm từ 30% - 70% VRAM so với cơ chế tĩnh.
- **Tiền xử lý văn bản (Robust Preprocessing):** Xây dựng hàm `_clean_text()` và `_resolve_text()` để xử lý các giá trị rác như `NaN`, chuỗi rỗng từ Pandas. Tự động chuẩn hóa khoảng trắng nhiều lớp (tabs, newlines) mà vẫn giữ nguyên casing để tương thích hoàn hảo với tokenizer của PhoBERT.
- **Class Factory `from_dataframe`:** Đóng gói quá trình mapping nhãn (tích cực/tiêu cực/trung lập -> 0/1/2) và khởi tạo, đảm bảo an toàn kiểu dữ liệu trước khi đưa vào PyTorch Tensor. Kiểm tra chặt chẽ các nhãn lạ từ dataset.

## TASK 2: Viết đè hàm Mất mát (Custom Trainer với Focal Loss)
**File:** `ai_engine/models/phobert_trainer.py`

**Chi tiết hiện thực & Nâng cấp:**
- **Focal Loss Native PyTorch:** Viết đè hàm `compute_loss` trong HuggingFace Trainer bằng thuật toán Focal Loss thay vì CrossEntropy mặc định để khắc phục tình trạng dữ liệu lệch nghiêm trọng (94% Tích cực).
- **Tính toán Alpha Động (Dynamic Alpha):** Không hardcode trọng số alpha. Hệ thống nhận vào `class_counts` từ tập Train và tự động tính toán trọng số nghịch đảo (inverse-frequency weighting) cho các lớp thiểu số, giúp linh hoạt khi dữ liệu thay đổi.
- **Đồng bộ hóa Device (Device Synchronization):** Mọi tensor tự tạo (`alpha`, `labels`) đều được map tự động sang `logits.device` (GPU/CPU/MPS) ngay bên trong vòng lặp forward, loại bỏ hoàn toàn rủi ro crash do lệch thiết bị tính toán.
- **Numerical Stability:** Phân bố xác suất (Probabilities) được kẹp bằng `.clamp(min=1e-7, max=1.0-1e-7)` trước khi đưa vào hàm `log()` để tránh lỗi `-inf` làm hỏng toàn bộ quá trình huấn luyện.

## TASK 3: Lập trình Kịch bản Huấn luyện (Training Script)
**File:** `scripts/train_phobert.py`

**Chi tiết hiện thực & Nâng cấp:**
- **Kiểm soát Reproducibility:** Viết hàm `set_seed` khóa seed trên toàn bộ hệ thống (Python, Numpy, PyTorch CPU/CUDA) và thiết lập CUDNN determinism, đảm bảo kết quả train có thể tái lập 100%.
- **Metrics chuyên sâu:** Độ chính xác (Accuracy) sẽ bị nhiễu do 94% dữ liệu tích cực, script sử dụng hàm đánh giá tùy chỉnh tính toán **Macro-F1, Precision, và Recall**.
- **Hỗ trợ MLOps (EarlyStopping):** Tích hợp `EarlyStoppingCallback` giám sát trực tiếp trên `eval_loss` (patience=2) kết hợp `load_best_model_at_end=True` để ngăn ngừa Overfitting và tự động lưu mô hình tối ưu nhất.
- **Siêu tham số khắt khe:** Learning rate 2e-5, Weight Decay 0.01, Cosine scheduler với 6% steps warmup. Sử dụng Automatic Mixed Precision (AMP `fp16=True`) tăng tốc huấn luyện trên GPU.

## TASK 4: Xây dựng Module Explainable AI (XAI)
**File:** `ai_engine/explainability/phobert_explainer.py`

**Chi tiết hiện thực & Nâng cấp:**
- **Sử dụng Captum (Integrated Gradients):** Nhắm thẳng vào layer `embeddings.word_embeddings` của kiến trúc RoBERTa để chấm điểm đóng góp của từng token đối với kết quả phán đoán cuối cùng.
- **Xử lý Sub-word phức tạp (Sub-word Aggregation):** PhoBERT sử dụng Byte-Level BPE với ký hiệu `@@` cho các từ ghép (sub-words) và `_` cho khoảng trắng. Pipeline XAI tự động hợp nhất các sub-words (VD: `tuyệt_@@` và `vời`) lại thành từ hoàn chỉnh (`tuyệt vời`) và cộng dồn điểm Attribution. Trả về kết quả sạch cho Frontend.
- **API-ready:** Trả kết quả dưới dạng `List[ExplanationResult]` (với `word` và `score`), có kiểu dữ liệu mạnh thông qua dataclass, giúp Web Frontend dễ dàng map để bôi đậm văn bản theo heatmap.

## TASK 5: Khớp nối Đa phương thức (Cross-Modal Fusion Engine)
**File:** `ai_engine/fusion/fusion_engine.py`

**Chi tiết hiện thực & Nâng cấp:**
- **Tính toán Trust Score Đa chiều:** Tính điểm tin cậy (0-100) theo trọng số chuẩn (Text 40%, Ảnh 40%, Authenticity 20%).
- **Động hóa Trọng số (Dynamic Weighting):** Xử lý hoàn hảo trường hợp đánh giá không kèm ảnh (`image_probs = None`). Toàn bộ 40% trọng số của ảnh sẽ được cộng bù sang cho Text (Text chiếm 80%), tránh việc điểm bị rớt oan uổng do lỗi tính toán.
- **Hình phạt Spam Tức thì (Spam Penalty):** Nếu flag `is_spam=True`, điểm Trust Score lập tức bị giới hạn ở mức trừng phạt cứng là `5.0/100`, đè lên mọi thông số tích cực của chữ/ảnh.
- **Phát hiện Xung đột (Multimodal Conflict):** Logic chéo phát hiện review giả mạo / buff đơn: Nếu xác suất Text khen ngợi > 60% NHƯNG mô hình nhận diện ảnh lại dự đoán Ảnh Hư hỏng/Vỡ > 60%, Trust Score lập tức bị chia đôi (`*= 0.5`) và gắn mã lỗi `MULTIMODAL_CONFLICT`. Trả về `FusionResult` có format chặt chẽ.
