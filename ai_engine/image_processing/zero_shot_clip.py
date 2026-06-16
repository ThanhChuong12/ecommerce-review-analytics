"""
zero_shot_clip.py
-----------------
Pipeline Zero-shot Classification bằng CLIP (OpenAI) để phân loại
ảnh review sản phẩm TMĐT thành 2 nhóm:

  1. product    → Ảnh sản phẩm / hộp hàng → đưa vào ResNet50
  2. irrelevant → Ảnh không liên quan       → loại bỏ

Pipeline tổng thể:
    Ảnh review → CLIP (binary) → [product] → ResNet50 → defect / no-defect
                                → [irrelevant] → loại bỏ

Model: openai/clip-vit-base-patch32 (ViT-B/32)
  - Benchmark: thắng ViT-B/16 trên dataset thực tế (68% vs 60.5%)
  - Nhẹ nhất (350MB), nhanh nhất (~100-180ms/img CPU)
  - ~82 phút cho toàn bộ 27K ảnh

Prompt Design v3 (HYBRID):
  - PRODUCT: prompts RỘNG (cover mọi loại sản phẩm) → tối ưu product recall
  - IRRELEVANT: prompts CỤ THỂ (dựa trên phân tích 35 ảnh bị lọt) → tối ưu irrelevant recall
"""

import logging
import os
from typing import Optional, List

import torch
from PIL import Image
# pyrefly: ignore [missing-import]
from transformers import CLIPProcessor, CLIPModel

_logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────

_CLIP_MODEL_NAME = os.getenv(
    "CLIP_MODEL_NAME",
    "openai/clip-vit-base-patch32"   # Thắng B/16 trên dataset thực tế
)

# ── Prompt Sets v3 — HYBRID (broad product + specific irrelevant) ──────────
#
# Design rationale:
#
# v1 (old):  product=broad, irrelevant=basic
#            → Product Recall 95% ✅, Irrelevant Recall 30% ❌
#
# v2 (new):  product=shipping-only, irrelevant=targeted
#            → Product Recall 71% ❌ (quá hẹp!), Irrelevant Recall 60% ✅
#
# v3 (hybrid): product=broad (giữ v1), irrelevant=targeted (giữ v2)
#              → Mục tiêu: Product Recall ≥90% VÀ Irrelevant Recall ≥50%
#
# CÂN BẰNG: 8 prompts mỗi nhóm (quan trọng vì dùng mean logit)

# ── Nhóm 1: PRODUCT — Rộng, cover mọi loại sản phẩm ──
# Giữ nguyên tinh thần v1 (đã đạt 95% product recall)
# Mô tả: bất kỳ vật thể sản phẩm, hộp hàng, đóng gói nào
PRODUCT_PROMPTS = [
    # Hộp hàng & đóng gói (quan trọng nhất — đây là input cho ResNet50)
    "a photo of a product package or cardboard shipping box",
    "an unboxing photo showing a delivered parcel or package",
    "a close-up photo of product packaging condition or damage",
    # Sản phẩm thực tế (quần áo, điện tử, mỹ phẩm...)
    "a sealed or wrapped product ready for delivery",
    "a consumer product sitting on a table for inspection",
    "a product photographed for an online shopping review",
    # Phụ kiện đi kèm & bao bì
    "a photo of a shipping container with labels or barcodes",
    "a product being examined or held up for quality check",
]

# ── Nhóm 2: IRRELEVANT — Cụ thể theo pattern dataset TMĐT Việt Nam ──
# Dựa trên phân tích trực tiếp 35/50 ảnh irrelevant bị lọt:
#   - Quần áo / tag quần áo → "person wearing/holding everyday item"
#   - Chai nước sốt / đồ ăn → "food, drinks, cooked meal"
#   - Sách / sticker / hình vẽ → "books, stickers, artwork"
#   - Hoa / trang trí Tết → "flowers, festive decorations"
#   - Thiệp / hóa đơn → "thank you card, receipt"
#   - Xe máy / nội thất → "vehicle, furniture, room"
#   - Selfie / đám đông → "selfie, group photo"
#   - Screenshot / video frame → "screenshot, phone app"
IRRELEVANT_PROMPTS = [
    # Người (selfie, đám đông, tay cầm đồ random)
    "a selfie, portrait, or group photo of people without any product",
    # Người đang dùng đồ (không phải review sản phẩm)
    "a person wearing, holding, or using an everyday household item",
    # Thực phẩm & đồ uống (chai nước sốt, đồ ăn, hoa quả)
    "a photo of food, drinks, fruits, a cooked meal, or sauce bottle",
    # Screenshot / giao diện / video frame
    "a screenshot of a phone app, chat message, or video thumbnail",
    # Hoa / quà tặng / trang trí / lễ hội Tết
    "flowers, bouquets, gift baskets, festive decorations, or ornaments",
    # Sách vở / sticker / hình vẽ / đồ học tập
    "books, notebooks, stickers, drawings, or school supplies",
    # Thiệp cảm ơn / hóa đơn / tờ rơi
    "a thank you card, printed receipt, invoice, or promotional flyer",
    # Xe / nội thất / cảnh ngoài trời / phòng ốc
    "a room interior, furniture, vehicle, motorcycle, or outdoor scene",
]

_N_PRODUCT = len(PRODUCT_PROMPTS)
_N_IRRELEVANT = len(IRRELEVANT_PROMPTS)
_ALL_PROMPTS = PRODUCT_PROMPTS + IRRELEVANT_PROMPTS

_GROUP_LABELS = ["product", "irrelevant"]

# ── Singleton Cache ─────────────────────────────────────────────────────────

_clip_model: Optional[CLIPModel] = None
_clip_processor: Optional[CLIPProcessor] = None
_clip_device: Optional[torch.device] = None


def _load_clip_model() -> tuple:
    """Load CLIP model với singleton cache. Chỉ load 1 lần duy nhất."""
    global _clip_model, _clip_processor, _clip_device

    if _clip_model is not None and _clip_processor is not None:
        return _clip_model, _clip_processor, _clip_device

    _clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info(
        "Loading CLIP model '%s' on %s...", _CLIP_MODEL_NAME, _clip_device
    )

    _clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME).to(_clip_device)
    _clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)
    _clip_model.eval()

    _logger.info("CLIP model loaded successfully (%d params).",
                 sum(p.numel() for p in _clip_model.parameters()))
    return _clip_model, _clip_processor, _clip_device


def reload_model(model_name: str = None) -> None:
    """Force reload CLIP model (dùng khi muốn đổi model variant).

    Args:
        model_name: Tên model trên HuggingFace. Nếu None, reload model hiện tại.
    """
    global _clip_model, _clip_processor, _clip_device, _CLIP_MODEL_NAME
    _clip_model = None
    _clip_processor = None
    _clip_device = None
    if model_name:
        _CLIP_MODEL_NAME = model_name
    _load_clip_model()


# ── Core Classification ────────────────────────────────────────────────────

def _classify_single(image: Image.Image) -> dict:
    """
    Core classification logic cho 1 ảnh PIL đã mở sẵn.

    Thuật toán:
    1. Tính CLIP similarity giữa ảnh và tất cả 16 prompts
    2. Lấy MEAN LOGIT của mỗi nhóm (2 scalar)
    3. Softmax trên 2 nhóm → xác suất binary thực sự
    4. Chọn nhóm có xác suất cao nhất
    """
    model, processor, device = _load_clip_model()

    inputs = processor(
        text=_ALL_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]  # [16]

    # ── Mean logit cho mỗi nhóm → Softmax binary ──
    product_logit    = logits[:_N_PRODUCT].mean()
    irrelevant_logit = logits[_N_PRODUCT:].mean()

    group_logits = torch.stack([product_logit, irrelevant_logit])
    group_probs = torch.softmax(group_logits, dim=0)  # [2], tổng = 1.0

    probs = {
        "product":    round(group_probs[0].item(), 4),
        "irrelevant": round(group_probs[1].item(), 4),
    }

    best_idx = group_probs.argmax().item()
    label = _GROUP_LABELS[best_idx]
    confidence = group_probs[best_idx].item()

    # Debug: prompt khớp nhất mỗi nhóm
    product_top = PRODUCT_PROMPTS[logits[:_N_PRODUCT].argmax().item()]
    irrelevant_top = IRRELEVANT_PROMPTS[
        logits[_N_PRODUCT:].argmax().item()
    ]

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probs": probs,
        "top_prompts": {
            "product": product_top,
            "irrelevant": irrelevant_top,
        },
    }


# ── Public API ──────────────────────────────────────────────────────────────

def classify_image(image_path: str) -> Optional[dict]:
    """Phân loại 1 ảnh thành product / irrelevant.

    Args:
        image_path (str): Đường dẫn tới file ảnh.

    Returns:
        dict: {
            "image_path": str,
            "label": "product" | "irrelevant",
            "confidence": float (0.0 – 1.0),
            "probs": {"product": float, "irrelevant": float},
            "top_prompts": {"product": str, "irrelevant": str},
        }
        None nếu ảnh không đọc được.
    """
    if not os.path.exists(image_path):
        _logger.warning("Ảnh không tồn tại: %s", image_path)
        return None

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        _logger.warning("Không thể đọc ảnh '%s': %s", image_path, exc)
        return None

    result = _classify_single(image)
    result["image_path"] = image_path
    return result


def classify_batch(
    image_paths: List[str],
    return_details: bool = False,
) -> dict:
    """Phân loại batch ảnh và nhóm theo label.

    Args:
        image_paths (list[str]): Danh sách đường dẫn ảnh.
        return_details (bool): Nếu True, trả thêm chi tiết mỗi ảnh.

    Returns:
        dict: {
            "product":    list[str],  — ảnh sản phẩm (đưa vào ResNet50)
            "irrelevant": list[str],  — ảnh rác (loại bỏ)
            "counts": {"product": int, "irrelevant": int},
            "total": int,
            "details": list[dict],    — chỉ có khi return_details=True
        }
    """
    _load_clip_model()

    groups = {"product": [], "irrelevant": []}
    details = []

    for path in image_paths:
        result = classify_image(path)
        if result is None:
            # Ảnh lỗi → giữ lại cho ResNet50 (safe default)
            groups["product"].append(path)
            continue

        groups[result["label"]].append(path)

        if return_details:
            details.append(result)

    counts = {k: len(v) for k, v in groups.items()}
    total = sum(counts.values())

    _logger.info(
        "CLIP binary filter: %d product, %d irrelevant (total=%d)",
        counts["product"], counts["irrelevant"], total,
    )

    output = {
        "product": groups["product"],
        "irrelevant": groups["irrelevant"],
        "counts": counts,
        "total": total,
    }
    if return_details:
        output["details"] = details

    return output


def filter_for_resnet(image_paths: List[str]) -> dict:
    """Lọc batch ảnh, chỉ giữ product images cho ResNet50.

    Returns:
        dict: {
            "product":    list[str],  — paths cho ResNet50
            "irrelevant": list[str],  — paths bị loại
            "counts": dict,
            "total": int,
        }
    """
    return classify_batch(image_paths, return_details=False)


# ── Backward Compatibility ─────────────────────────────────────────────────

def detect_irrelevant_image(image_path: str, threshold: float = None) -> bool:
    """[DEPRECATED] Dùng classify_image() thay thế.
    Trả về True nếu ảnh KHÔNG liên quan đến sản phẩm.
    """
    result = classify_image(image_path)
    if result is None:
        return False
    return result["label"] == "irrelevant"


def classify_image_detail(image_path: str) -> Optional[dict]:
    """[DEPRECATED] Dùng classify_image() thay thế."""
    result = classify_image(image_path)
    if result is None:
        return None

    return {
        "image_path": image_path,
        "relevant_score": result["probs"]["product"],
        "irrelevant_score": result["probs"]["irrelevant"],
        "is_irrelevant": result["label"] == "irrelevant",
        "top_relevant_prompt": result["top_prompts"]["product"],
        "top_irrelevant_prompt": result["top_prompts"]["irrelevant"],
        "label": result["label"],
        "probs": result["probs"],
    }


def filter_irrelevant_batch(
    image_paths: list,
    threshold: float = None,
    return_details: bool = False,
) -> dict:
    """[DEPRECATED] Dùng classify_batch() thay thế."""
    batch_result = classify_batch(image_paths, return_details=return_details)

    output = {
        "relevant": batch_result["product"],
        "irrelevant": batch_result["irrelevant"],
        "total": batch_result["total"],
        "relevant_count": len(batch_result["product"]),
        "irrelevant_count": len(batch_result["irrelevant"]),
    }
    if return_details and "details" in batch_result:
        output["details"] = batch_result["details"]
    return output
