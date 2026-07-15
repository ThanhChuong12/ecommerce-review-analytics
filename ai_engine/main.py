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
import traceback
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Optional, List
import pandas as pd

# ─── Fix Unicode stdout/stderr trên Windows CP1252 ───────────────────────────
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

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

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ─── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="AI Engine", version="3.0.0")

WEBHOOK_PROGRESS    = os.getenv("NODE_WEBHOOK_PROGRESS", "http://localhost:5000/api/webhook/update-progress")
WEBHOOK_FINISHED    = os.getenv("NODE_WEBHOOK_FINISHED", "http://localhost:5000/api/webhook/finished")
MAX_REVIEWS_SCRAPE  = int(os.getenv("MAX_REVIEWS_SCRAPE", "0"))  # 0 = không giới hạn, cào toàn bộ
MAX_IMAGES_PROCESS  = int(os.getenv("MAX_IMAGES_PROCESS", "50"))
MOBILENET_WEIGHTS   = os.getenv(
    "MOBILENET_WEIGHTS_PATH",
    str(_PROJECT_ROOT / "artifacts" / "models" / "mobilenet" / "mobilenet_v3_model2_defect.pt")
)
RESNET_WEIGHTS = os.getenv(
    "RESNET_WEIGHTS_PATH",
    str(_PROJECT_ROOT / "artifacts" / "models" / "resnet50" / "resnet50_defect_gpu_best.pth")
)
SPAM_WEIGHTS        = os.getenv(
    "SPAM_WEIGHTS_PATH",
    str(_PROJECT_ROOT / "artifacts" / "models" / "spam_iforest" / "spam_iforest.pkl")
)
PHOBERT_WEIGHTS     = os.getenv(
    "PHOBERT_WEIGHTS_PATH",
    str(_PROJECT_ROOT / "artifacts" / "models" / "phobert")
)
TEXT_BASELINE_WEIGHTS = os.getenv(
    "TEXT_BASELINE_WEIGHTS_PATH",
    str(_PROJECT_ROOT / "artifacts" / "models" / "baselines" / "ensemble_smote_auto_weights.pkl")
)

DENOISER_DIR = os.getenv("DENOISER_DIR", str(_PROJECT_ROOT / "artifacts" / "models" / "denoiser"))
FEATURE_DENOISER_PATH = os.getenv("FEATURE_DENOISER_PATH", os.path.join(DENOISER_DIR, "feature_denoiser.pt"))
TEXT_HEAD_PATH = os.getenv("TEXT_HEAD_PATH", os.path.join(DENOISER_DIR, "text_sentiment_head.pt"))
IMAGE_HEAD_PATH = os.getenv("IMAGE_HEAD_PATH", os.path.join(DENOISER_DIR, "image_defect_head.pt"))

# Mapping nhãn tiếng Việt → English (DB + Frontend)
_SENTIMENT_VI_EN = {
    "tích cực": "positive",
    "tiêu cực": "negative",
    "trung lập": "neutral",
    "positive":  "positive",
    "negative":  "negative",
    "neutral":   "neutral",
}

# ─── Custom MLP Classification Head & Denoiser Models (MDSBR Pipeline) ────────

import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """MLP head for classification on top of embeddings."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Cache for new denoiser and classification head models
_denoiser_cache = None
_text_head_cache = None
_image_head_cache = None
_resnet_backbone_cache = None
_phobert_backbone_cache = None
_phobert_tokenizer_cache = None

def _load_new_models(device: torch.device):
    global _denoiser_cache, _text_head_cache, _image_head_cache
    global _resnet_backbone_cache, _phobert_backbone_cache, _phobert_tokenizer_cache
    
    # 1. Load PhoBERT Tokenizer and Backbone if not cached
    if _phobert_tokenizer_cache is None or _phobert_backbone_cache is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(f"[AI Engine] Loading PhoBERT...")
        _phobert_tokenizer_cache = AutoTokenizer.from_pretrained(PHOBERT_WEIGHTS)
        full_model = AutoModelForSequenceClassification.from_pretrained(PHOBERT_WEIGHTS)
        if hasattr(full_model, "roberta"):
            _phobert_backbone_cache = full_model.roberta
        else:
            _phobert_backbone_cache = full_model.base_model
        _phobert_backbone_cache.to(device)
        _phobert_backbone_cache.eval()

    # 2. Load ResNet50 Backbone if not cached
    if _resnet_backbone_cache is None:
        from torchvision import models
        print(f"[AI Engine] Loading ResNet50...")
        resnet = models.resnet50(weights=None)
        
        # Determine num_classes from checkpoint
        checkpoint = torch.load(RESNET_WEIGHTS, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        fc_key = "fc.weight"
        if fc_key in state_dict:
            num_classes = state_dict[fc_key].shape[0]
            resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
            
        resnet.load_state_dict(state_dict, strict=False)
        resnet.fc = torch.nn.Identity()  # strip the classification head to get 2048-dim features
        resnet.to(device)
        resnet.eval()
        _resnet_backbone_cache = resnet

    # 3. Load Feature Denoiser
    if _denoiser_cache is None and os.path.exists(FEATURE_DENOISER_PATH):
        from ai_engine.denoising.feature_denoiser import FeatureDenoiser
        print(f"[AI Engine] Loading Feature Denoiser...")
        ckpt = torch.load(FEATURE_DENOISER_PATH, map_location=device, weights_only=False)
        config = ckpt["config"]
        denoiser = FeatureDenoiser(
            text_dim=config["text_dim"],
            image_dim=config["image_dim"],
            hidden_dim=config["hidden_dim"],
            noise_steps=config["noise_steps"],
            noise_schedule=config.get("noise_schedule", "cosine"),
        )
        denoiser.load_state_dict(ckpt["model_state_dict"])
        denoiser.to(device)
        denoiser.eval()
        _denoiser_cache = denoiser

    # 4. Load Text Classification Head
    if _text_head_cache is None and os.path.exists(TEXT_HEAD_PATH):
        print(f"[AI Engine] Loading Text MLP Head...")
        ckpt = torch.load(TEXT_HEAD_PATH, map_location=device, weights_only=False)
        head = ClassificationHead(input_dim=ckpt["input_dim"], num_classes=ckpt["num_classes"])
        head.load_state_dict(ckpt["model_state_dict"])
        head.to(device)
        head.eval()
        _text_head_cache = (head, ckpt.get("class_names", ["tiêu cực", "trung lập", "tích cực"]))

    # 5. Load Image Classification Head
    if _image_head_cache is None and os.path.exists(IMAGE_HEAD_PATH):
        print(f"[AI Engine] Loading Image MLP Head...")
        ckpt = torch.load(IMAGE_HEAD_PATH, map_location=device, weights_only=False)
        head = ClassificationHead(input_dim=ckpt["input_dim"], num_classes=ckpt["num_classes"])
        head.load_state_dict(ckpt["model_state_dict"])
        head.to(device)
        head.eval()
        _image_head_cache = (head, ckpt.get("class_names", ["no-defect", "defect"]))


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
        import requests
        r = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/webp,image/apng,image/jpeg,image/png,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.lazada.vn/",
        }, timeout=12)
        r.raise_for_status()
        
        # Ngăn chặn việc tải nhầm trang báo lỗi HTML (403/Captcha)
        if "text/html" in r.headers.get("Content-Type", ""):
            print(f"[IMG] Lỗi ảnh trả về HTML/Captcha: {url[:80]}")
            return None

        with open(dest, "wb") as f:
            f.write(r.content)
        return dest
    except Exception as e:
        print(f"[IMG] Download thất bại {url[:60]}: {e}")
        return None


def _build_time_series(reviews: List[dict]) -> List[dict]:
    """Nhóm reviews theo thời gian (ngày/tuần/tháng) tùy thuộc vào độ trải dài của dữ liệu."""
    import pandas as pd
    if not reviews:
        return []
    
    valid_records = []
    for r in reviews:
        date_str = str(r.get("date", ""))
        sentiment = r.get("sentiment", "neutral")
        parsed_dt = None
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                parsed_dt = datetime.strptime(date_str[:10], fmt)
                break
            except ValueError:
                continue
        if parsed_dt:
            valid_records.append({"date": parsed_dt, "sentiment": sentiment})
            
    if not valid_records:
        return [
            {
                "date":     (datetime.now() - timedelta(days=i)).strftime("%m/%d"),
                "positive": 0, "neutral": 0, "negative": 0
            }
            for i in range(14, -1, -1)
        ]
        
    df = pd.DataFrame(valid_records)
    min_date = df["date"].min()
    max_date = df["date"].max()
    days_diff = (max_date - min_date).days
    
    if days_diff <= 30:
        # Nhóm theo ngày
        df["period"] = df["date"].dt.strftime("%d/%m")
    elif days_diff <= 180:
        # Nhóm theo tuần (hiển thị ngày đầu tuần)
        df["period"] = df["date"].dt.to_period('W').apply(lambda r: r.start_time.strftime("%d/%m"))
    else:
        # Nhóm theo tháng
        df["period"] = df["date"].dt.strftime("%m/%Y")
        
    grouped = df.groupby(["period", "sentiment"]).size().unstack(fill_value=0)
    
    for col in ["positive", "negative", "neutral"]:
        if col not in grouped.columns:
            grouped[col] = 0
            
    period_to_date = df.groupby("period")["date"].min()
    grouped = grouped.loc[period_to_date.sort_values().index]
    
    grouped = grouped.tail(15)
    
    result = []
    for period, row in grouped.iterrows():
        result.append({
            "date": period,
            "positive": int(row["positive"]),
            "negative": int(row["negative"]),
            "neutral": int(row["neutral"])
        })
        
    return result


def _build_smart_advice(trust_score: float, spam_pct: float, reviews: List[dict]) -> str:
    """Tạo gợi ý thông minh dựa trên kết quả AI thực."""
    damaged_pct = sum(1 for r in reviews if r.get("label") in ("damaged", "wrong_item")) / max(len(reviews), 1) * 100
    neg_pct     = sum(1 for r in reviews if r.get("sentiment") == "negative")            / max(len(reviews), 1) * 100

    parts = []
    if spam_pct > 30:
        parts.append(f"{spam_pct:.0f}% đánh giá bị nghi ngờ spam/seeding.")
    if damaged_pct > 20:
        parts.append(f"{damaged_pct:.0f}% ảnh có dấu hiệu hộp bị móp méo/lỗi hàng.")
    if neg_pct > 40:
        parts.append(f"Tỷ lệ đánh giá tiêu cực cao ({neg_pct:.0f}%).")
    if trust_score < 50:
        parts.append("Trust Score thấp — xem xét sản phẩm thay thế bên dưới.")
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
    print(f"\n[AI Engine] ===== START")

    def _fail(reason: str):
        """Gửi lỗi về webhook và thoát."""
        safe_reason = reason.encode('ascii', errors='replace').decode('ascii')
        print(f"[AI Engine] FATAL: {safe_reason}")
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
                    "smartAdvice": f"Không thể phân tích URL này: {reason}",
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
        import logging
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        # === 1) Setup Constants & Paths === 
        def _scrape_progress_cb(lines: int):
            pct = 10
            if MAX_REVIEWS_SCRAPE > 0:
                pct = min(10 + int((lines / MAX_REVIEWS_SCRAPE) * 14), 24)
            _report_progress(product_id, pct, f"Đang cào dữ liệu... ({lines} đánh giá)")

        total = asyncio.run(_scrape(
            url         = url,
            output_path = tmp_csv,
            fmt         = "csv",
            max_reviews = MAX_REVIEWS_SCRAPE,
            progress_callback = _scrape_progress_cb
        ))
        
        print(f"[Scraper] Đã lấy thành công {total} reviews.")

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
        tb = traceback.format_exc()
        print(f"[Scraper] EXCEPTION TRACEBACK:\n{tb}")
        _fail(f"Scraping that bai: {type(e).__name__}: {str(e)[:200]}")
        return

    if not scraped_rows:
        _fail("Khong tim thay danh gia nao cho san pham nay.")
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

        # 2. IForest Hybrid — bắt spam nâng cao (train lại, version mới)
        if os.path.exists(SPAM_WEIGHTS):
            import __main__
            __main__.SpamHybridModel = SpamHybridModel
            spam_model = SpamHybridModel.load(SPAM_WEIGHTS)
            texts = df_result["text"].tolist()
            ratings = df_result["rating"].tolist()
            X = build_feature_matrix(df_result, texts, ratings)
            final_spam = spam_model.predict_final_spam(X, rule_is_spam)
            is_spam_flags = final_spam.tolist()
            print("[SpamFilter] IForest hỗ trợ được khởi tạo thành công.")
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_new_text_pipeline = False
        try:
            _load_new_models(device)
            if _phobert_backbone_cache is not None and _text_head_cache is not None:
                use_new_text_pipeline = True
                print("[AI Engine] Using retrained PhoBERT + MLP Head sentiment pipeline.")
        except Exception as e:
            print(f"[AI Engine] Error loading new text pipeline: {e}. Falling back to old sentiment model.")

        phobert_model = None
        if not use_new_text_pipeline and os.path.exists(PHOBERT_WEIGHTS):
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

        total_rows = len(scraped_rows)
        for i, r in enumerate(scraped_rows):
            if i % max(1, total_rows // 10) == 0 or i == total_rows - 1:
                _report_progress(
                    product_id, 
                    38 + int((i / total_rows) * 15), 
                    f"Đang phân tích cảm xúc & khía cạnh ({i}/{total_rows})..."
                )
                
            text = r["text"]
            
            vi_label = "trung lập"
            probs = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

            try:
                if use_new_text_pipeline and text.strip():
                    cleaned = " ".join(str(text).strip().split())
                    encodings = _phobert_tokenizer_cache(
                        [cleaned], padding=True, truncation=True,
                        max_length=256, return_tensors="pt"
                    )
                    input_ids = encodings["input_ids"].to(device)
                    attention_mask = encodings["attention_mask"].to(device)
                    
                    with torch.no_grad():
                        outputs = _phobert_backbone_cache(input_ids=input_ids, attention_mask=attention_mask)
                        if hasattr(outputs, "last_hidden_state"):
                            cls_emb = outputs.last_hidden_state[:, 0, :]
                        else:
                            cls_emb = outputs[0][:, 0, :]
                        
                        text_head, text_class_names = _text_head_cache
                        logits = text_head(cls_emb)
                        probs_tensor = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                        
                    pred_idx = int(probs_tensor.argmax())
                    vi_label = text_class_names[pred_idx]
                    
                    probs = {}
                    for cls_name, prob_val in zip(text_class_names, probs_tensor):
                        en_name = _SENTIMENT_VI_EN.get(cls_name, "neutral")
                        probs[en_name] = float(prob_val)
                    
                    for k in ["positive", "negative", "neutral"]:
                        if k not in probs:
                            probs[k] = 0.0
                elif phobert_model and text.strip():
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
            
            # Map sentiment to 1-5 score for the aspect
            if en_label == "positive":
                asp_val = 5
            elif en_label == "negative":
                asp_val = 1
            else:
                asp_val = 3

            for asp in extracted_aspects:
                aspect_scores[asp].append(asp_val)

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

    _report_progress(product_id, 55, f"Đang tải ảnh đánh giá (tất cả ảnh)...")

    img_temp_dir = tempfile.mkdtemp(prefix="ai_imgs_")
    image_local_paths: List[Optional[str]] = [None] * len(scraped_rows)
    image_orig_urls:   List[Optional[str]] = [None] * len(scraped_rows)

    all_img_targets = [
        (i, r["image_urls"][0])
        for i, r in enumerate(scraped_rows)
        if r.get("image_urls")
    ]
    # MAX_IMAGES_PROCESS=0 → không giới hạn, tải toàn bộ
    download_targets = all_img_targets if MAX_IMAGES_PROCESS == 0 else all_img_targets[:MAX_IMAGES_PROCESS]

    total_imgs = len(download_targets)
    for enum_idx, (idx, img_url) in enumerate(download_targets):
        if enum_idx % max(1, total_imgs // 10) == 0 or enum_idx == total_imgs - 1:
            _report_progress(product_id, 55 + int((enum_idx / total_imgs) * 7), f"Đang tải ảnh ({enum_idx}/{total_imgs})...")
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

        total_clip = len(image_local_paths)
        for i, path in enumerate(image_local_paths):
            if i % max(1, total_clip // 10) == 0 or i == total_clip - 1:
                _report_progress(product_id, 63 + int((i / total_clip) * 8), f"Đang lọc ảnh không liên quan ({i}/{total_clip})...")
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
        image_probs_dict[i] = {"irrelevant": 1.0, "intact": 0.0, "damaged": 0.0, "wrong_item": 0.0}

    use_new_image_pipeline = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        _load_new_models(device)
        if _resnet_backbone_cache is not None and _denoiser_cache is not None and _image_head_cache is not None:
            use_new_image_pipeline = True
            print("[AI Engine] Using new ResNet50 + FeatureDenoiser + MLP Head image pipeline.")
    except Exception as e:
        print(f"[AI Engine] Error loading new image pipeline: {e}. Falling back to old model.")

    if use_new_image_pipeline:
        try:
            valid_indices = [
                i for i, p in enumerate(image_local_paths)
                if p and os.path.exists(str(p))
                and i not in clip_irrelevant_indices
            ]
            valid_paths = [image_local_paths[i] for i in valid_indices]

            if valid_paths:
                import albumentations as A
                from albumentations.pytorch import ToTensorV2
                import cv2
                
                preprocess = A.Compose([
                    A.Resize(224, 224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
                
                image_head, image_class_names = _image_head_cache

                total_resnet = len(valid_paths)
                for enum_idx, (idx, path) in enumerate(zip(valid_indices, valid_paths)):
                    if enum_idx % max(1, total_resnet // 10) == 0 or enum_idx == total_resnet - 1:
                        _report_progress(product_id, 72 + int((enum_idx / total_resnet) * 12), f"Đang nhận diện hộp ({enum_idx}/{total_resnet})...")
                    image_bgr = cv2.imread(str(path))
                    if image_bgr is None:
                        continue
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    transformed = preprocess(image=image_rgb)
                    img_tensor = transformed["image"].unsqueeze(0).to(device) # [1, 3, 224, 224]
                    
                    with torch.no_grad():
                        raw_emb = _resnet_backbone_cache(img_tensor) # [1, 2048]
                        dummy_text = torch.zeros(1, _denoiser_cache.text_dim, device=device)
                        _, image_clean = _denoiser_cache(dummy_text, raw_emb)
                        logits = image_head(image_clean)
                        probs_tensor = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                        
                    pred_idx = int(probs_tensor.argmax())
                    lbl = image_class_names[pred_idx] # "defect" or "no-defect"
                    
                    if lbl == "defect":
                        lbl_mapped = "damaged"
                    else:
                        lbl_mapped = "intact"
                        
                    image_labels[idx] = lbl_mapped
                    
                    mapped_probs = {
                        "damaged": float(probs_tensor[1]) if len(probs_tensor) > 1 else 0.0,
                        "intact": float(probs_tensor[0]) if len(probs_tensor) > 0 else 0.0,
                    }
                    image_probs_dict[idx] = mapped_probs

                label_dist = Counter(image_labels[i] for i in valid_indices)
                print(f"[New Image Pipeline] Label distribution: {dict(label_dist)}")
        except Exception as e:
            print(f"[New Image Pipeline] Failed: {e}. Falling back to old model.")
            use_new_image_pipeline = False

    if not use_new_image_pipeline:
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
                    print(f"[ResNet50] Label distribution: {dict(label_dist)}")

            except Exception as e:
                print(f"[ResNet50] Inference thất bại (fallback intact): {e}")
        else:
            print(f"[ResNet50] Weights không tìm thấy tại {RESNET_WEIGHTS} → skip image classification")

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

        # Tính trust score dựa trên NON-SPAM reviews
        non_spam_scores = [
            per_review_scores[i]
            for i in range(len(per_review_scores))
            if i < len(is_spam_flags) and not is_spam_flags[i]
        ]
        if non_spam_scores:
            overall_trust = round(sum(non_spam_scores) / len(non_spam_scores), 1)
        else:
            overall_trust = round(sum(per_review_scores) / max(len(per_review_scores), 1), 1)
        print(f"[Fusion] Trust Score: {overall_trust}/100 (từ {len(non_spam_scores)} non-spam reviews)")

    except Exception as e:
        print(f"[Fusion] Thất bại (fallback 60.0): {e}")
        overall_trust = 60.0
        per_review_scores = [60.0] * len(scraped_rows)

    # ── STEP 7: LLM Summary ────────────────────────────────────────────────────
    _report_progress(product_id, 90, "Đang tổng hợp AI summary (LLM CoT)...")

    llm_summary = ""
    pos_count = sentiments.count("positive")
    neg_count = sentiments.count("negative")
    neu_count = sentiments.count("neutral")
    total_rev = len(sentiments)

    try:
        from ai_engine.llm_integration.llm_client import BaseLLMClient

        # ── Phần 1: Keyword frequency từ TOÀN BỘ reviews ──────────────────────
        try:
            from ai_engine.text_processing.sentiment_analysis import POSITIVE_LEXICON, NEGATIVE_LEXICON
            _pos_kw: Counter = Counter()
            _neg_kw: Counter = Counter()
            for i, r in enumerate(scraped_rows):
                text_lower = r["text"].lower()
                snt = sentiments[i] if i < len(sentiments) else "neutral"
                if snt == "positive":
                    for kw in POSITIVE_LEXICON:
                        if kw in text_lower:
                            _pos_kw[kw] += 1
                elif snt == "negative":
                    for kw in NEGATIVE_LEXICON:
                        if kw in text_lower:
                            _neg_kw[kw] += 1
            top_pos_kw = [f"{k} ({v} lần)" for k, v in _pos_kw.most_common(15) if v > 0]
            top_neg_kw = [f"{k} ({v} lần)" for k, v in _neg_kw.most_common(10) if v > 0]
        except Exception:
            top_pos_kw = []
            top_neg_kw = []

        # ── Phần 2: Aspect summary ─────────────────────────────────────────────
        aspect_summary_parts = []
        for asp_key, asp_sc in aspect_scores.items():
            if asp_sc:
                avg_sc = round(sum(asp_sc) / len(asp_sc), 1)
                aspect_summary_parts.append(f"{asp_key}: TB {avg_sc}/5 ({len(asp_sc)} đề cập)")

        # ── Phần 3: Review mẫu stratified — mix dài + ngẫu nhiên ──────────────
        import random as _rnd
        _by_rating: dict = {5: [], 4: [], 3: [], 2: [], 1: []}
        for r in scraped_rows:
            rating = int(r.get("rating", 3))
            txt = r["text"].strip()
            if txt and len(txt) > 10:
                _by_rating.get(rating, _by_rating[3]).append(txt)

        total_with_text = sum(len(v) for v in _by_rating.values())
        sample_reviews: List[str] = []
        for stars in [5, 4, 3, 2, 1]:
            pool = _by_rating.get(stars, [])
            if not pool:
                continue
            ratio = len(pool) / max(total_with_text, 1)
            n = max(2, min(15, round(ratio * 40)))
            long_picks = sorted(pool, key=len, reverse=True)[:5]
            rest = [x for x in pool if x not in long_picks]
            _rnd.shuffle(rest)
            sample_reviews.extend((long_picks + rest)[:n])

        stats_text = (
            f"=== THỐNG KÊ ({total_rev} đánh giá | Trust Score: {overall_trust}/100) ===\n"
            f"{pos_count} tích cực ({round(pos_count/max(total_rev,1)*100)}%) | "
            f"{neg_count} tiêu cực ({round(neg_count/max(total_rev,1)*100)}%) | {neu_count} trung lập\n\n"
            f"=== TỪ KHÓA NỔI BẬT (tần suất từ {total_rev} đánh giá) ===\n"
            f"KHEN: {', '.join(top_pos_kw) if top_pos_kw else '(không có)'}\n"
            f"CHÊ: {', '.join(top_neg_kw) if top_neg_kw else '(không có)'}\n\n"
            f"=== KHÍA CẠNH ===\n"
            f"{chr(10).join(aspect_summary_parts) if aspect_summary_parts else 'chưa đủ dữ liệu'}\n\n"
            f"=== {len(sample_reviews)} REVIEW MẪU ===\n"
            + "\n".join(f"• {r[:400]}" for r in sample_reviews)
        )

        class LLMProductSummaryClient(BaseLLMClient):
            def __init__(self):
                super().__init__(timeout=30.0)
                self.temperature = 0.1
                self.max_tokens = 800
                self.response_format = None
                self.system_prompt = (
                    "Bạn là trợ lý AI tổng hợp đánh giá sản phẩm thương mại điện tử Việt Nam.\n"
                    "Dữ liệu bạn nhận được gồm: (1) Tần suất từ khóa từ TOÀN BỘ đánh giá, (2) Khía cạnh, (3) Review mẫu nhiều mức sao.\n\n"
                    "NHIỆM VỤ: Đọc hiểu toàn bộ và viết tóm tắt CHÂN THỰC, TỰ NHIÊN như một người đã đọc hàng trăm bình luận thật.\n\n"
                    "CẤU TRÚC BẮT BUỘC (copy y chang, chỉ điền nội dung vào):\n"
                    "Về sản phẩm:\n"
                    "+ [nhận xét tích cực phổ biến nhất]\n"
                    "+ [nhận xét tích cực khác nếu thực sự khác biệt]\n"
                    "- [điểm trừ nếu nhiều người đề cập]\n\n"
                    "Về dịch vụ:\n"
                    "+ [nhận xét giao hàng/đóng gói tốt]\n"
                    "- [điểm trừ dịch vụ nếu có]\n\n"
                    "LUẬT TUYỆT ĐỐI:\n"
                    "1. KHÔNG đếm ('1 đánh giá', 'N lần', 'nhiều người'...) — viết thành câu chủ động tự nhiên.\n"
                    "2. KHÔNG dùng **, markdown, hay in đậm bất kỳ thứ gì.\n"
                    "3. Dấu `+` CHỈ DÀNH CHO KHEN (ưu điểm). Dấu `-` CHỈ DÀNH CHO CHÊ (nhược điểm). Tuyệt đối không để ý chê vào phần dấu `+`.\n"
                    "4. Mỗi dòng + hay - là 1 câu 10-25 từ, viết như người thật.\n"
                    "5. Dùng TẦN SUẤT từ khóa để biết điểm nào phổ biến — từ khóa xuất hiện nhiều = ý nhiều người nói.\n"
                    "6. GOM Ý TRIỆT ĐỂ (VD: 'sách đẹp' + 'bìa xinh' + 'thiết kế bắt mắt' → gộp thành 1 dòng duy nhất).\n"
                    "7. Tối đa 4 dòng cho phần về sản phẩm, 3 dòng cho phần về dịch vụ. Không viết dòng chỉ để cho đủ số."
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
            prefix = "✅ AI đánh giá đây là sản phẩm đáng tin cậy"
        elif action == "XÓA":
            prefix = "❌ AI phát hiện nhiều dấu hiệu bất thường, nên thận trọng"
        else:
            prefix = "⚠️ AI ghi nhận một số điểm cần lưu ý trước khi mua"

        llm_summary = (
            f"Trợ lý AI tổng hợp từ {total_rev} đánh giá (Trust Score: {overall_trust}/100 - {prefix}). Sản phẩm đã nhận được {pos_count} đánh giá tích cực, {neg_count} đánh giá tiêu cực và {neu_count} đánh giá trung lập.\n\n"
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
    _report_progress(product_id, 95, "Đang tìm đề xuất sản phẩm tương tự...")

    alternative_products: List[dict] = []
    try:
        from similar_products_fetcher import scrape_similar_products
        similar_items = asyncio.run(scrape_similar_products(url, limit=5))
        alternative_products = [
            {
                "name":       p.name,
                "thumbnail":  p.image_url,
                "url":        p.url,
                "rating":     p.rating,
                "price":      p.price,
                "sold":       p.sold,
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

    # Ensure mặc định cho từng aspect nếu không detect được
    # Fallback to average sentiment score (5, 1, 3) instead of rating, to match the pie chart
    sentiment_scores = [5 if s == "positive" else (1 if s == "negative" else 3) for s in sentiments]
    avg_sentiment = sum(sentiment_scores) / max(len(sentiment_scores), 1)
    
    if "Product" not in aspect_sentiment_result:
        aspect_sentiment_result["Product"] = round(avg_sentiment, 1)
    if "Packaging" not in aspect_sentiment_result:
        aspect_sentiment_result["Packaging"] = round(max(1.0, avg_sentiment - 0.3), 1)
    if "Shipping" not in aspect_sentiment_result:
        aspect_sentiment_result["Shipping"] = round(max(1.0, avg_sentiment - 0.2), 1)

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

    # ── STEP 8.5: Lấy ảnh gốc sản phẩm (Thumbnail) ────────────────────────────
    if not thumbnail_url:
        def _get_thumbnail_sync(p_url: str) -> str:
            import httpx, re
            from bs4 import BeautifulSoup
            try:
                # Nếu là Shopee, dùng API lấy ảnh
                if "shopee." in p_url:
                    m = re.search(r"i\.(\d+)\.(\d+)", p_url)
                    if m:
                        shopid, itemid = m.groups()
                        api = f"https://shopee.vn/api/v4/item/get?itemid={itemid}&shopid={shopid}"
                        r = httpx.get(api, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                        if r.status_code == 200:
                            img_id = r.json().get("data", {}).get("image")
                            if img_id: return f"https://cf.shopee.vn/file/{img_id}"
                # Tiki, Lazada, TGDD: lấy og:image bằng HTTPX (không cần JS)
                r = httpx.get(p_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10, follow_redirects=True)
                soup = BeautifulSoup(r.text, 'html.parser')
                meta = soup.find('meta', property='og:image') or soup.find('meta', itemprop='image')
                if meta and meta.get('content'):
                    return str(meta['content'])
            except Exception:
                pass
            return ""

        real_thumb = _get_thumbnail_sync(url)
        if real_thumb:
            thumbnail_url = real_thumb
        else:
            thumbnail_url = next((u for u in image_orig_urls if u), "")

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
