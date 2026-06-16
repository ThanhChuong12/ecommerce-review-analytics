"""
AI Engine — FastAPI Service
===========================
Pipeline thực tế kết nối đúng các module AI đã train:
  1. Scrape reviews từ URL (scraping_agent/scraper/dispatcher.py)
  2. Spam Detection (text_processing/spam_filter.detect_spam — rule-based)
  3. Sentiment + Aspects (NextGenReviewAnalyzer: heuristic → zero-shot xlm-roberta → LLM)
  4. Download ảnh review về local temp
  5. MobileNetV3 batch inference (image_processing/defect_detection.detect_defect_mobilenet_batch)
  6. Cross-Modal Fusion (fusion/fusion_engine.TrustScoreCalculator — dùng xác suất thực từ model)
  7. LLM CoT Summary (llm_integration/llm_client.LLMRecommendationClient)
  8. Similar Products (scraping_agent/similar_products_fetcher.scrape_similar_products)
  9. Webhook → Node.js → DB → Socket.IO → Frontend
"""

from fastapi import FastAPI, BackgroundTasks, Request
import asyncio
import requests
import requests.exceptions
import tempfile
import shutil
import os
import sys
import random
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Optional, List
import pandas as pd

# ─── Path setup ──────────────────────────────────────────────────────────────
# Đảm bảo import được:
#   - ai_engine.* (khi gọi từ project root)
#   - text_processing.*, fusion.*, etc. (khi gọi từ ai_engine/)
_THIS_DIR    = Path(__file__).resolve().parent        # .../ai_engine
_PROJECT_ROOT = _THIS_DIR.parent                      # .../ecommerce-review-analytics
_SCRAPING_DIR = _PROJECT_ROOT / "scraping_agent"      # .../scraping_agent

for _p in [str(_PROJECT_ROOT), str(_THIS_DIR), str(_SCRAPING_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─── Load env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(_THIS_DIR / ".env", override=False)
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass  # dotenv optional — env vars có thể set trực tiếp

# ─── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="AI Engine", version="3.0.0")

WEBHOOK_PROGRESS    = os.getenv("NODE_WEBHOOK_PROGRESS", "http://localhost:5000/api/webhook/update-progress")
WEBHOOK_FINISHED    = os.getenv("NODE_WEBHOOK_FINISHED", "http://localhost:5000/api/webhook/finished")
MAX_REVIEWS_SCRAPE  = int(os.getenv("MAX_REVIEWS_SCRAPE", "200"))
MAX_IMAGES_PROCESS  = int(os.getenv("MAX_IMAGES_PROCESS", "50"))
MOBILENET_WEIGHTS   = os.getenv(
    "MOBILENET_WEIGHTS_PATH",
    "e:\\Nhập môn học máy\\Project\\ecommerce-review-analytics\\artifacts\\models\\mobilenet\\mobilenet_v3_model2_defect.pt"
)
RESNET_WEIGHTS = os.getenv(
    "RESNET_WEIGHTS_PATH",
    "e:\\Nhập môn học máy\\Project\\ecommerce-review-analytics\\artifacts\\models\\resnet50\\resnet50_defect_gpu_best.pth"
)
SPAM_WEIGHTS        = os.getenv(
    "SPAM_WEIGHTS_PATH",
    str(_PROJECT_ROOT / "artifacts" / "models" / "tuned_spam_iforest" / "tuned_spam_iforest.pkl")
)
PHOBERT_WEIGHTS     = os.getenv(
    "PHOBERT_WEIGHTS_PATH",
    str(_PROJECT_ROOT / "artifacts" / "models" / "phobert" / "phobert")
)
TEXT_BASELINE_WEIGHTS = os.getenv(
    "TEXT_BASELINE_WEIGHTS_PATH",
    str(_PROJECT_ROOT / "artifacts" / "models" / "baselines" / "ensemble_smote_auto_weights.pkl")
)

# Mapping nhãn tiếng Việt → English (DB + Frontend)
_SENTIMENT_VI_EN = {
    "tích cực": "positive",
    "tiêu cực": "negative",
    "trung lập": "neutral",
    "positive":  "positive",
    "negative":  "negative",
    "neutral":   "neutral",
}


# ─── Utils ────────────────────────────────────────────────────────────────────

def _report_progress(product_id: int, progress: int, message: str):
    """Emit progress về Node.js → Socket.IO → Frontend."""
    try:
        requests.post(WEBHOOK_PROGRESS, json={
            "productId": product_id,
            "progress":  progress,
            "message":   message,
        }, timeout=5)
    except Exception as e:
        print(f"[Progress] Webhook thất bại (không critical): {e}")


def _download_image(url: str, dest_dir: str) -> Optional[str]:
    """Download ảnh về local. Trả về path hoặc None nếu thất bại."""
    if not url or not str(url).startswith("http"):
        return None
    try:
        fname = f"img_{abs(hash(url)) % 10**9}.jpg"
        dest  = os.path.join(dest_dir, fname)
        if os.path.exists(dest):
            return dest
        req = urllib.request.Request(url, headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            )
        })
        with urllib.request.urlopen(req, timeout=12) as r:
            data = r.read()
        with open(dest, "wb") as f:
            f.write(data)
        return dest
    except Exception as e:
        print(f"[IMG] Download thất bại {url[:60]}: {e}")
        return None


def _build_time_series(reviews: List[dict]) -> List[dict]:
    """Nhóm reviews theo ngày → sentiment per day (15 ngày gần nhất)."""
    daily: dict = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
    for r in reviews:
        date_str  = str(r.get("date", ""))
        sentiment = r.get("sentiment", "neutral")
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                dt  = datetime.strptime(date_str[:10], fmt)
                key = dt.strftime("%m/%d")
                if sentiment in daily[key]:
                    daily[key][sentiment] += 1
                break
            except ValueError:
                continue

    if not daily:
        return [
            {
                "date":     (datetime.now() - timedelta(days=i)).strftime("%m/%d"),
                "positive": 0, "neutral": 0, "negative": 0
            }
            for i in range(14, -1, -1)
        ]
    return [
        {"date": d, **v}
        for d, v in sorted(daily.items())[-15:]
    ]


def _build_smart_advice(trust_score: float, spam_pct: float, reviews: List[dict]) -> str:
    """Tạo gợi ý thông minh dựa trên kết quả AI thực."""
    damaged_pct = sum(1 for r in reviews if r.get("label") in ("damaged", "wrong_item")) / max(len(reviews), 1) * 100
    neg_pct     = sum(1 for r in reviews if r.get("sentiment") == "negative")            / max(len(reviews), 1) * 100

    parts = []
    if spam_pct > 30:
        parts.append(f"⚠️ {spam_pct:.0f}% đánh giá bị nghi ngờ spam/seeding.")
    if damaged_pct > 20:
        parts.append(f"📦 {damaged_pct:.0f}% ảnh có dấu hiệu hộp bị móp méo/lỗi hàng.")
    if neg_pct > 40:
        parts.append(f"😞 Tỷ lệ đánh giá tiêu cực cao ({neg_pct:.0f}%).")
    if trust_score < 50:
        parts.append("🛡️ Trust Score thấp — xem xét sản phẩm thay thế bên dưới.")
    if not parts:
        if trust_score >= 80:
            parts.append("✅ Sản phẩm đáng tin cậy, đa số đánh giá tích cực.")
        else:
            parts.append("💡 Sản phẩm ổn, đọc kỹ đánh giá 1-2 sao trước khi mua.")
    return " ".join(parts)


# ─── Main AI Pipeline ─────────────────────────────────────────────────────────

def heavy_ai_process(product_id: int, url: str) -> None:
    """
    Pipeline AI chạy nền. Thứ tự gọi đúng các module đã train:

    Step 1    (10%): Scraping          → scraping_agent
    Step 2    (25%): Spam Detection    → spam_filter.detect_spam (rule-based, tất cả reviews)
    Step 3    (42%): Sentiment+Aspects → NextGenReviewAnalyzer (heuristic→zero-shot→LLM)
    Step 4    (55%): Download images   → urllib (parallel by URL list)
    Step 4.5  (63%): CLIP Filter       → zero-shot binary (lọc ảnh irrelevant)
    Step 5    (72%): Image Defect      → ResNet50 (chỉ ảnh product từ CLIP)
    Step 6    (82%): Fusion            → TrustScoreCalculator (dùng xác suất thực từ model)
    Step 7    (90%): LLM Summary       → LLMRecommendationClient CoT
    Step 8    (95%): Similar products  → scrape_similar_products
    Step 9    (99%): Webhook           → finishedWebhook → DB → Socket.IO
    """
    print(f"\n[AI Engine] ===== START | productId={product_id} | url={url} =====")

    def _fail(reason: str):
        """Gửi lỗi về webhook và thoát."""
        print(f"[AI Engine] FATAL: {reason}")
        try:
            requests.post(WEBHOOK_FINISHED, json={
                "productId":   product_id,
                "productData": {"name": "Không xác định", "thumbnail": ""},
                "reviews":     [],
                "summary":     f"Lỗi xử lý: {reason}",
                "metadata": {
                    "spamPercentage": 0, "trustScore": 0,
                    "aspectSentiment": {}, "sentimentTimeSeries": [],
                    "keywords": {"positive": [], "negative": []},
                    "smartAdvice": f"⚠️ Không thể phân tích URL này: {reason}",
                    "alternativeProducts": [],
                }
            }, timeout=30)
        except Exception:
            pass

    # ── STEP 1: Scraping ─────────────────────────────────────────────────────
    _report_progress(product_id, 10, "Đang cào dữ liệu đánh giá từ sàn TMĐT...")

    scraped_rows: List[dict] = []   # [{text, rating, image_urls, date}, ...]
    product_name  = "Sản phẩm"
    thumbnail_url = ""

    try:
        from scraper.dispatcher import scrape as _scrape

        import uuid
        tmp_csv = os.path.join(tempfile.gettempdir(), f"scrape_{uuid.uuid4().hex[:8]}.csv")

        total = asyncio.run(_scrape(
            url         = url,
            output_path = tmp_csv,
            fmt         = "csv",
            max_reviews = MAX_REVIEWS_SCRAPE,
        ))
        print(f"[Scraper] {total} reviews → {tmp_csv}")

        if total > 0 and os.path.exists(tmp_csv):
            df_raw = pd.read_csv(tmp_csv, encoding="utf-8-sig")

            # Tìm tên cột động (scraper khác nhau có thể đặt tên khác nhau)
            text_col   = next((c for c in df_raw.columns if c in ("text", "review_text", "content")), None)
            rating_col = next((c for c in df_raw.columns if c in ("rating", "stars", "star")), None)
            img_col    = next((c for c in df_raw.columns if c in ("image_urls", "images", "image_url")), None)
            name_col   = next((c for c in df_raw.columns if c in ("product_name", "name")), None)
            date_col   = next((c for c in df_raw.columns if c in ("date", "created_at", "reviewed_at")), None)

            if text_col:
                for _, row in df_raw.iterrows():
                    img_urls: List[str] = []
                    if img_col and pd.notna(row.get(img_col)):
                        raw = str(row[img_col]).strip()
                        if raw.startswith("["):
                            import json as _json
                            try:
                                img_urls = [u for u in _json.loads(raw) if u]
                            except Exception:
                                img_urls = raw.split("|") if raw.startswith("http") else []
                        elif raw.startswith("http"):
                            img_urls = raw.split("|")

                    _t = row.get(text_col, "")
                    text_val = "" if pd.isna(_t) else str(_t).strip()

                    scraped_rows.append({
                        "text":       text_val,
                        "rating":     int(row.get(rating_col, 3)) if rating_col else 3,
                        "image_urls": img_urls,
                        "date":       str(row.get(date_col, "")) if date_col else "",
                    })

                if name_col and not df_raw[name_col].dropna().empty:
                    product_name = str(df_raw[name_col].dropna().iloc[0])

            os.unlink(tmp_csv)

    except Exception as e:
        _fail(f"Scraping thất bại: {e}")
        return

    if not scraped_rows:
        _fail("Không tìm thấy đánh giá nào cho sản phẩm này.")
        return

    print(f"[Scraper] Parsed {len(scraped_rows)} reviews")

    # ── STEP 2: Spam Detection ────────────────────────────────────────────────
    _report_progress(product_id, 25, "Đang lọc đánh giá spam/seeding (Hybrid IForest)...")

    is_spam_flags: List[int] = [0] * len(scraped_rows)
    spam_pct = 0.0

    try:
        from ai_engine.text_processing.spam_filter import detect_spam
        from ai_engine.text_processing.spam_model import SpamHybridModel, build_feature_matrix

        df_spam = pd.DataFrame([{"text": r["text"], "rating": r["rating"]} for r in scraped_rows])
        # 1. Rule-based
        df_result = detect_spam(df_spam)
        rule_is_spam = df_result["is_spam"].values.astype(int)

        # 2. IForest Hybrid
        if os.path.exists(SPAM_WEIGHTS):
            import __main__
            __main__.SpamHybridModel = SpamHybridModel
            spam_model = SpamHybridModel.load(SPAM_WEIGHTS)
            texts = df_result["text"].tolist()
            ratings = df_result["rating"].tolist()
            X = build_feature_matrix(df_result, texts, ratings)
            final_spam = spam_model.predict_final_spam(X, rule_is_spam)
            is_spam_flags = final_spam.tolist()
        else:
            print(f"[SpamFilter] Warning: {SPAM_WEIGHTS} not found. Using rule-based only.")
            is_spam_flags = rule_is_spam.tolist()

        spam_count = int(sum(is_spam_flags))
        spam_pct   = round(spam_count / max(len(is_spam_flags), 1) * 100, 1)
        print(f"[SpamFilter] {spam_count}/{len(is_spam_flags)} spam ({spam_pct}%)")

    except Exception as e:
        print(f"[SpamFilter] Thất bại (bỏ qua, tất cả non-spam): {e}")

    # ── STEP 3: Sentiment + Aspect Analysis (PhoBERT + TextEnsemble) ──────────
    _report_progress(product_id, 38, "Đang phân tích cảm xúc & khía cạnh (PhoBERT)...")

    sentiments:   List[str]            = []
    text_probs:   List[dict]           = []
    aspects_list: List[List[str]]      = []
    aspect_scores: dict = defaultdict(list)

    try:
        from ai_engine.models.phobert_model import PhoBertSentimentModel
        from ai_engine.models.text_baseline import TextEnsembleModel
        from ai_engine.text_processing.embeddings import DeepEmbedder

        # Load Sentiment Models
        phobert_model = None
        if os.path.exists(PHOBERT_WEIGHTS):
            phobert_model = PhoBertSentimentModel(PHOBERT_WEIGHTS)
        
        text_baseline_model = None
        if os.path.exists(TEXT_BASELINE_WEIGHTS):
            text_baseline_model = TextEnsembleModel.load(TEXT_BASELINE_WEIGHTS)

        # Aspect extraction setup
        embedder = DeepEmbedder()
        aspect_anchors = {
            "shipping": "giao hàng đóng gói thời gian vận chuyển nhanh chậm",
            "product": "chất lượng sản phẩm chính hãng hàng giả hàng nhái",
            "price": "giá cả đắt rẻ khuyến mãi voucher",
            "service": "dịch vụ chăm sóc khách hàng tư vấn thái độ",
        }
        anchor_texts = list(aspect_anchors.values())
        anchor_vectors = embedder.encode(anchor_texts)

        for i, r in enumerate(scraped_rows):
            text = r["text"]
            
            vi_label = "trung lập"
            probs = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

            try:
                if phobert_model and text.strip():
                    probs_array = phobert_model.predict_proba(text)
                    vi_label = phobert_model.predict(text)
                    probs = {
                        "positive": float(probs_array[phobert_model.label2id.get("tích cực", 1)]),
                        "negative": float(probs_array[phobert_model.label2id.get("tiêu cực", 0)]),
                        "neutral":  float(probs_array[phobert_model.label2id.get("trung lập", 2)]),
                    }
                elif text_baseline_model and text.strip():
                    probs_array = text_baseline_model.predict_proba(pd.Series([text]))[0]
                    vi_label = text_baseline_model.predict(pd.Series([text]))[0]
                    classes = list(text_baseline_model.pipeline.classes_)
                    probs = {
                        "positive": float(probs_array[classes.index("tích cực")]) if "tích cực" in classes else 0.33,
                        "negative": float(probs_array[classes.index("tiêu cực")]) if "tiêu cực" in classes else 0.33,
                        "neutral":  float(probs_array[classes.index("trung lập")]) if "trung lập" in classes else 0.34,
                    }
                else:
                    if r["rating"] >= 4:
                        vi_label = "tích cực"; probs = {"positive": 0.8, "negative": 0.05, "neutral": 0.15}
                    elif r["rating"] <= 2:
                        vi_label = "tiêu cực"; probs = {"positive": 0.05, "negative": 0.8, "neutral": 0.15}
            except Exception as e:
                print(f"[Sentiment] Inference failed for row {i}: {e}")

            en_label = _SENTIMENT_VI_EN.get(vi_label, "neutral")
            sentiments.append(en_label)
            text_probs.append(probs)

            # Aspect extraction
            extracted_aspects = []
            if text.strip():
                vector = embedder.encode(text)
                if vector.size > 0 and anchor_vectors.size > 0:
                    sim_scores = embedder.compute_cosine_similarity(vector, anchor_vectors)[0]
                    extracted_aspects = [
                        list(aspect_anchors.keys())[idx]
                        for idx, score in enumerate(sim_scores)
                        if score > 0.65
                    ]

            aspects_list.append(extracted_aspects)
            for asp in extracted_aspects:
                aspect_scores[asp].append(r["rating"])

        print(f"[Sentiment] {Counter(sentiments)} | Aspects gom được: {dict((k, len(v)) for k,v in aspect_scores.items())}")

    except Exception as e:
        print(f"[Sentiment] Pipeline thất bại → fallback rating-prior: {e}")
        for r in scraped_rows:
            if r["rating"] >= 4:
                en = "positive";  tp = {"positive": 0.80, "negative": 0.05, "neutral": 0.15}
            elif r["rating"] <= 2:
                en = "negative";  tp = {"positive": 0.05, "negative": 0.80, "neutral": 0.15}
            else:
                en = "neutral";   tp = {"positive": 0.20, "negative": 0.20, "neutral": 0.60}
            sentiments.append(en)
            text_probs.append(tp)
            aspects_list.append([])

    # ── STEP 4: Download ảnh ─────────────────────────────────────────────────
    _report_progress(product_id, 55, f"Đang tải ảnh đánh giá (tối đa {MAX_IMAGES_PROCESS})...")

    img_temp_dir = tempfile.mkdtemp(prefix="ai_imgs_")
    image_local_paths: List[Optional[str]] = [None] * len(scraped_rows)
    image_orig_urls:   List[Optional[str]] = [None] * len(scraped_rows)

    download_targets = [
        (i, r["image_urls"][0])
        for i, r in enumerate(scraped_rows)
        if r.get("image_urls")
    ][:MAX_IMAGES_PROCESS]

    for idx, img_url in download_targets:
        local = _download_image(img_url, img_temp_dir)
        image_local_paths[idx] = local
        image_orig_urls[idx]   = img_url

    valid_count = sum(1 for p in image_local_paths if p and os.path.exists(str(p)))
    print(f"[Images] Tải thành công {valid_count}/{len(download_targets)} ảnh")

    # ── STEP 4.5: CLIP Filter (lọc ảnh không liên quan) ───────────────────
    _report_progress(product_id, 63, "Đang lọc ảnh không liên quan (CLIP)...")

    clip_irrelevant_indices: set = set()  # indices bị CLIP loại

    try:
        from ai_engine.image_processing.zero_shot_clip import classify_image as clip_classify

        for i, path in enumerate(image_local_paths):
            if path and os.path.exists(str(path)):
                clip_result = clip_classify(str(path))
                if clip_result and clip_result["label"] == "irrelevant":
                    clip_irrelevant_indices.add(i)

        clip_product_count = valid_count - len(clip_irrelevant_indices)
        print(f"[CLIP] {clip_product_count} product, {len(clip_irrelevant_indices)} irrelevant (tổng {valid_count})")

    except Exception as e:
        print(f"[CLIP] Filter thất bại (bỏ qua, giữ tất cả ảnh): {e}")

    # ── STEP 5: ResNet50 Defect Detection ─────────────────────────────────
    _report_progress(product_id, 72, "Đang nhận diện tình trạng hộp (ResNet50)...")

    image_labels:     List[str]           = ["intact"] * len(scraped_rows)
    image_probs_dict: List[Optional[dict]] = [None]    * len(scraped_rows)

    # Gán nhãn "irrelevant" cho ảnh bị CLIP loại (không cần qua ResNet50)
    for i in clip_irrelevant_indices:
        image_labels[i] = "irrelevant"

    if os.path.exists(RESNET_WEIGHTS):
        try:
            from ai_engine.image_processing.defect_detection import detect_defect_resnet_batch

            # Chỉ lấy ảnh PRODUCT (bỏ qua ảnh CLIP đã loại)
            valid_indices = [
                i for i, p in enumerate(image_local_paths)
                if p and os.path.exists(str(p))
                and i not in clip_irrelevant_indices
            ]
            valid_paths = [image_local_paths[i] for i in valid_indices]

            if valid_paths:
                batch_results = detect_defect_resnet_batch(
                    image_paths = valid_paths,
                    model_path  = RESNET_WEIGHTS,
                    threshold   = 0.85,
                    batch_size  = 16,
                )
                # batch_results[j] = {"label": str, "confidence": float, "probabilities": {class:float}}
                for j, res in enumerate(batch_results):
                    orig_i = valid_indices[j]
                    
                    # Map MobileNetV3 'defect'/'no-defect' to Web DB 'damaged'/'intact'
                    lbl = res["label"]
                    if lbl == "defect": lbl = "damaged"
                    elif lbl == "no-defect": lbl = "intact"
                    image_labels[orig_i] = lbl
                    
                    raw_probs = res.get("probabilities", {})
                    mapped_probs = {}
                    if "defect" in raw_probs:
                        mapped_probs["damaged"] = raw_probs["defect"]
                    if "no-defect" in raw_probs:
                        mapped_probs["intact"] = raw_probs["no-defect"]
                    if not mapped_probs:
                        mapped_probs = raw_probs
                        
                    image_probs_dict[orig_i] = mapped_probs

                label_dist = Counter(image_labels[i] for i in valid_indices)
                print(f"[MobileNetV3] Label distribution: {dict(label_dist)}")

        except Exception as e:
            print(f"[MobileNetV3] Inference thất bại (fallback intact): {e}")
    else:
        print(f"[MobileNetV3] Weights không tìm thấy tại {MOBILENET_WEIGHTS} → skip image classification")

    # ── STEP 6: Fusion Engine ─────────────────────────────────────────────────
    # Dùng xác suất THỰC từ model: TextProbs từ sentiment, ImageProbs từ MobileNetV3
    _report_progress(product_id, 82, "Đang tính Trust Score (Cross-Modal Fusion Engine)...")

    per_review_scores: List[float] = []

    try:
        from ai_engine.fusion.fusion_engine import (
            TrustScoreCalculator, FusionInput, TextProbs, ImageProbs, AuthMeta
        )
        calculator = TrustScoreCalculator()

        for i in range(len(scraped_rows)):
            tp = text_probs[i]
            tp_sum = tp["positive"] + tp["negative"] + tp["neutral"]

            text_p = TextProbs(
                positive = round(tp["positive"] / tp_sum, 4),
                negative = round(tp["negative"] / tp_sum, 4),
                neutral  = round(tp["neutral"]  / tp_sum, 4),
            )

            img_p = None
            raw_probs = image_probs_dict[i]
            if raw_probs and isinstance(raw_probs, dict):
                try:
                    _s = sum(raw_probs.values()) or 1.0
                    img_p = ImageProbs(
                        intact     = round(raw_probs.get("intact",     0.0) / _s, 4),
                        damaged    = round(raw_probs.get("damaged",    0.0) / _s, 4),
                        wrong_item = round(raw_probs.get("wrong_item", 0.0) / _s, 4),
                        irrelevant = round(raw_probs.get("irrelevant", 0.0) / _s, 4),
                    )
                except Exception:
                    img_p = None

            spam_flag = bool(is_spam_flags[i] if i < len(is_spam_flags) else 0)
            auth_m    = AuthMeta(is_spam=spam_flag, spam_score=0.7 if spam_flag else 0.0)

            fusion_in  = FusionInput(text_probs=text_p, image_probs=img_p, auth_meta=auth_m)
            fusion_out = calculator.calculate(fusion_in)
            per_review_scores.append(fusion_out.final_score)

        overall_trust = round(sum(per_review_scores) / max(len(per_review_scores), 1), 1)
        print(f"[Fusion] Trust Score trung bình: {overall_trust}/100")

    except Exception as e:
        print(f"[Fusion] Thất bại (fallback 60.0): {e}")
        overall_trust = 60.0
        per_review_scores = [60.0] * len(scraped_rows)

    # ── STEP 7: LLM Summary (CoT) ─────────────────────────────────────────────
    _report_progress(product_id, 90, "Đang tổng hợp AI summary (LLM CoT)...")

    llm_summary = ""
    pos_count = sentiments.count("positive")
    neg_count = sentiments.count("negative")
    neu_count = sentiments.count("neutral")
    total_rev = len(sentiments)

    try:
        from ai_engine.llm_integration.llm_client import BaseLLMClient

        # Lấy vài review tiêu biểu để AI có ngữ cảnh
        pos_revs = [r["text"] for r in scraped_rows if r.get("rating", 3) >= 4 and r["text"].strip()][:3]
        neg_revs = [r["text"] for r in scraped_rows if r.get("rating", 3) <= 2 and r["text"].strip()][:3]
        
        stats_text = (
            f"Sản phẩm có {total_rev} đánh giá ({pos_count} khen, {neg_count} chê).\n"
            f"Điểm tin cậy (Trust Score): {overall_trust}/100.\n"
            f"Khen: {pos_revs}\n"
            f"Chê: {neg_revs}"
        )

        class LLMProductSummaryClient(BaseLLMClient):
            def __init__(self):
                super().__init__(timeout=20.0)
                self.temperature = 0.3
                self.max_tokens = 800
                self.response_format = None
                self.system_prompt = (
                    "Bạn là AI chuyên phân tích đánh giá sản phẩm. "
                    "Dựa vào dữ liệu thống kê và các review tiêu biểu, hãy viết MỘT đoạn văn ngắn (3-4 câu) "
                    "tóm tắt chất lượng sản phẩm và đưa ra lời khuyên cho người mua (Nên mua hay Cẩn thận). "
                    "Chỉ trả về đoạn văn tóm tắt, không giải thích gì thêm."
                )
            def summarize(self, text: str) -> str:
                for provider in self.provider_chain:
                    try:
                        ans = self._call_provider(provider, text)
                        if ans: return ans.strip()
                    except Exception:
                        pass
                return ""

        step2 = LLMProductSummaryClient().summarize(stats_text)

        action = "DUYỆT" if overall_trust >= 70 and pos_count >= neg_count else "XÓA" if overall_trust < 40 or neg_count > pos_count * 2 else "CẢNH BÁO"
        
        if action == "DUYỆT":
            prefix = "✅ AI đánh giá đây là sản phẩm đáng tin cậy."
        elif action == "XÓA":
            prefix = "❌ AI phát hiện nhiều dấu hiệu bất thường, nên thận trọng."
        else:
            prefix = "⚠️ AI ghi nhận một số điểm cần lưu ý trước khi mua."

        llm_summary = (
            f"{prefix} "
            f"Đã phân tích {total_rev} đánh giá: {pos_count} tích cực, {neg_count} tiêu cực, {neu_count} trung lập. "
            f"Trust Score: {overall_trust}/100. "
            f"{step2}"
        ).strip()

        print(f"[LLM] Summary ({action}): {llm_summary[:100]}...")

    except Exception as e:
        print(f"[LLM] CoT thất bại → heuristic summary: {e}")
        if pos_count > neg_count * 2:
            llm_summary = f"Sản phẩm được đánh giá tốt ({pos_count}/{total_rev} tích cực, Trust Score: {overall_trust}/100). Người mua hài lòng về chất lượng."
        elif neg_count > pos_count:
            llm_summary = f"Sản phẩm có nhiều đánh giá tiêu cực ({neg_count}/{total_rev}, Trust Score: {overall_trust}/100). Cân nhắc thận trọng trước khi mua."
        else:
            llm_summary = f"Sản phẩm đánh giá trung bình ({pos_count} tích cực, {neg_count} tiêu cực, Trust Score: {overall_trust}/100)."

    # ── STEP 8: Similar Products ──────────────────────────────────────────────
    _report_progress(product_id, 95, "Đang tìm sản phẩm thay thế tương tự...")

    alternative_products: List[dict] = []
    try:
        from similar_products_fetcher import scrape_similar_products
        similar_items = asyncio.run(scrape_similar_products(url, limit=5))
        alternative_products = [
            {
                "name":       p.name,
                "thumbnail":  p.image_url,
                "url":        p.url,
                "trustScore": random.randint(70, 92),   # placeholder trust score cho sản phẩm tương tự
            }
            for p in similar_items if p.name
        ]
        print(f"[Similar] {len(alternative_products)} sản phẩm tương tự")
    except Exception as e:
        print(f"[Similar] Thất bại (bỏ qua): {e}")

    # ── Xây dựng aspectSentiment từ kết quả embedding thực ───────────────────
    # NextGenReviewAnalyzer.extract_aspects trả về keys: "shipping", "product", "price", "service"
    # Map sang tên hiển thị frontend: Product, Packaging (shipping), Shipping (service)
    _aspect_map = {
        "product":  "Product",
        "shipping": "Packaging",   # "shipping" aspect = đóng gói + vận chuyển
        "service":  "Shipping",    # "service" aspect = dịch vụ giao hàng
        "price":    "Price",
    }
    aspect_sentiment_result: dict = {}
    for asp_key, asp_scores in aspect_scores.items():
        display_name = _aspect_map.get(asp_key, asp_key.title())
        aspect_sentiment_result[display_name] = round(sum(asp_scores) / len(asp_scores), 1)

    # Ensure mặc định nếu không detect được aspect nào
    if not aspect_sentiment_result:
        avg_rating = sum(r["rating"] for r in scraped_rows) / max(len(scraped_rows), 1)
        aspect_sentiment_result = {
            "Product":   round(avg_rating, 1),
            "Packaging": round(avg_rating - 0.3, 1),
            "Shipping":  round(avg_rating - 0.2, 1),
        }

    # ── Keyword extraction từ lexicon thực trong sentiment_analysis.py ────────
    try:
        from ai_engine.text_processing.sentiment_analysis import POSITIVE_LEXICON, NEGATIVE_LEXICON

        pos_kw_counter: Counter = Counter()
        neg_kw_counter: Counter = Counter()
        tmp_reviews = [
            {"review_text": scraped_rows[i]["text"], "sentiment": sentiments[i]}
            for i in range(len(scraped_rows))
        ]
        for r in tmp_reviews:
            text_lower = r["review_text"].lower()
            if r["sentiment"] == "positive":
                for kw in POSITIVE_LEXICON:
                    if kw in text_lower:
                        pos_kw_counter[kw] += 1
            elif r["sentiment"] == "negative":
                for kw in NEGATIVE_LEXICON:
                    if kw in text_lower:
                        neg_kw_counter[kw] += 1
        keywords_result = {
            "positive": [{"text": k, "value": int(v)} for k, v in pos_kw_counter.most_common(18) if v > 0],
            "negative": [{"text": k, "value": int(v)} for k, v in neg_kw_counter.most_common(18) if v > 0],
        }
    except Exception as e:
        print(f"[Keywords] Import LEXICON thất bại: {e}")
        keywords_result = {"positive": [], "negative": []}

    # ── Build processed_reviews list cho DB ───────────────────────────────────
    processed_reviews = [
        {
            "review_text": scraped_rows[i]["text"],
            "rating":      scraped_rows[i]["rating"],
            "image_path":  image_orig_urls[i],          # URL gốc, không phải local path
            "label":       image_labels[i] if image_orig_urls[i] else None,
            "sentiment":   sentiments[i],
            "date":        scraped_rows[i].get("date", ""),
        }
        for i in range(len(scraped_rows))
    ]

    # Thumbnail = ảnh đầu tiên trong reviews nếu không có riêng
    if not thumbnail_url:
        thumbnail_url = next((url for url in image_orig_urls if url), "")

    # ── STEP 9: Webhook → Node.js ─────────────────────────────────────────────
    _report_progress(product_id, 99, "Đang lưu kết quả vào database...")

    metadata = {
        "spamPercentage":      int(spam_pct),
        "trustScore":          round(overall_trust, 1),
        "aspectSentiment":     aspect_sentiment_result,
        "sentimentTimeSeries": _build_time_series(processed_reviews),
        "keywords":            keywords_result,
        "smartAdvice":         _build_smart_advice(overall_trust, spam_pct, processed_reviews),
        "alternativeProducts": alternative_products,
    }

    payload = {
        "productId":   product_id,
        "productData": {"name": product_name, "thumbnail": thumbnail_url},
        "reviews":     processed_reviews,
        "summary":     llm_summary,
        "metadata":    metadata,
    }

    try:
        resp = requests.post(WEBHOOK_FINISHED, json=payload, timeout=120)
        print(f"[Webhook] finishedWebhook → {resp.status_code}")
    except Exception as e:
        print(f"[Webhook] FAILED: {e}")

    # Cleanup temp images
    shutil.rmtree(img_temp_dir, ignore_errors=True)

    print(f"[AI Engine] ===== DONE | productId={product_id} | trust={overall_trust} =====\n")


# ─── FastAPI Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":    "ok",
        "version":   "3.0.0",
        "mobilenet": os.path.exists(MOBILENET_WEIGHTS),
    }


@app.post("/process-job")
async def receive_job(request: Request, background_tasks: BackgroundTasks):
    """Nhận job từ Node.js BullMQ worker và chạy AI pipeline trong background."""
    body = await request.json()
    product_id = body.get("productId")
    url        = body.get("url")

    if not product_id or not url:
        return {"error": "Thiếu productId hoặc url"}, 400

    background_tasks.add_task(heavy_ai_process, product_id, url)
    return {"message": "AI Engine đã nhận job!", "productId": product_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
