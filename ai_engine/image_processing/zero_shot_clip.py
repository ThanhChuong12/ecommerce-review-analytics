"""
zero_shot_clip.py
-----------------
Pipeline Zero-shot Classification bằng CLIP (OpenAI) để nhận diện
và lọc ảnh rác/ảnh không liên quan từ reviews sản phẩm TMĐT.

CLIP (Contrastive Language-Image Pretraining) so sánh ảnh với mô tả văn bản
để phân loại mà KHÔNG cần training — chỉ dùng pretrained model.

Chức năng:
  - detect_irrelevant_image(): Phân loại 1 ảnh (relevant / irrelevant)
  - filter_irrelevant_batch(): Lọc batch ảnh, trả về danh sách ảnh relevant
  - classify_image_detail(): Phân loại chi tiết với confidence scores

Model: openai/clip-vit-base-patch32 (~350MB, auto-download từ HuggingFace)

Usage:
    >>> from ai_engine.image_processing.zero_shot_clip import detect_irrelevant_image
    >>> is_spam = detect_irrelevant_image("review_photo.jpg")  # True = ảnh rác

    >>> from ai_engine.image_processing.zero_shot_clip import filter_irrelevant_batch
    >>> relevant_paths = filter_irrelevant_batch(["img1.jpg", "img2.jpg", ...])
"""

import logging
import os
from typing import Optional

import torch
from PIL import Image
# pyrefly: ignore [missing-import]
from transformers import CLIPProcessor, CLIPModel

_logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────

# Model CLIP pretrained — nhẹ nhất, phù hợp zero-shot pre-filter
_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Ngưỡng mặc định: ảnh có P(irrelevant) > threshold → coi là rác
# 0.55 = hơi thiên về "giữ lại" — tránh false positive (loại nhầm ảnh tốt)
# Có thể tuỳ chỉnh qua biến môi trường CLIP_IRRELEVANT_THRESHOLD
_DEFAULT_THRESHOLD = float(os.getenv("CLIP_IRRELEVANT_THRESHOLD", "0.55"))

# ── Prompt Sets ─────────────────────────────────────────────────────────────
# Nhiều prompts → CLIP tính trung bình similarity → phân loại chính xác hơn

RELEVANT_PROMPTS = [
    "a photo of a product package or box",
    "a product review image showing an item",
    "a photo showing product condition or damage",
    "an unboxing photo of a delivered product",
    "a close-up photo of a physical product",
]

IRRELEVANT_PROMPTS = [
    "a screenshot of a phone or computer screen",
    "a selfie, pet photo, or random personal photo",
    "a receipt, invoice, or document scan",
    "a blank screen, black image, or solid color image",
    "a food photo, scenery, or image unrelated to shopping",
]

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

    _logger.info("CLIP model loaded successfully.")
    return _clip_model, _clip_processor, _clip_device


# ── Public API ──────────────────────────────────────────────────────────────

def detect_irrelevant_image(
    image_path: str,
    threshold: float = None,
) -> bool:
    """Trả về True nếu ảnh KHÔNG liên quan đến sản phẩm.

    Sử dụng CLIP zero-shot: so sánh ảnh với 2 nhóm text prompts
    (relevant vs irrelevant) rồi tính xác suất trung bình.

    Args:
        image_path (str): Đường dẫn tới file ảnh.
        threshold (float): Ngưỡng P(irrelevant) để phân loại.
                           Mặc định: 0.55 (từ env CLIP_IRRELEVANT_THRESHOLD).

    Returns:
        bool: True nếu ảnh là rác/không liên quan, False nếu ảnh hợp lệ.
    """
    if threshold is None:
        threshold = _DEFAULT_THRESHOLD

    result = classify_image_detail(image_path)
    if result is None:
        return False  # Lỗi → giữ lại ảnh (safe default)

    return result["irrelevant_score"] >= threshold


def classify_image_detail(image_path: str) -> Optional[dict]:
    """Phân loại chi tiết 1 ảnh bằng CLIP — trả về confidence scores.

    Args:
        image_path (str): Đường dẫn tới file ảnh.

    Returns:
        dict: {
            "image_path": str,
            "relevant_score": float (0.0 – 1.0),
            "irrelevant_score": float (0.0 – 1.0),
            "is_irrelevant": bool,
            "top_relevant_prompt": str,
            "top_irrelevant_prompt": str,
        }
        None nếu ảnh không đọc được.
    """
    if not os.path.exists(image_path):
        _logger.warning("Ảnh không tồn tại: %s", image_path)
        return None

    model, processor, device = _load_clip_model()

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        _logger.warning("Không thể đọc ảnh '%s': %s", image_path, exc)
        return None

    # Kết hợp cả 2 nhóm prompts
    all_prompts = RELEVANT_PROMPTS + IRRELEVANT_PROMPTS
    n_relevant = len(RELEVANT_PROMPTS)

    inputs = processor(
        text=all_prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # logits_per_image: [1, n_prompts] — similarity score ảnh vs mỗi prompt
        logits = outputs.logits_per_image[0]  # [n_prompts]
        probs = logits.softmax(dim=0)         # normalize thành xác suất

    # Tính trung bình xác suất cho mỗi nhóm
    relevant_probs = probs[:n_relevant]
    irrelevant_probs = probs[n_relevant:]

    relevant_score = relevant_probs.sum().item()
    irrelevant_score = irrelevant_probs.sum().item()

    # Tìm prompt có similarity cao nhất mỗi nhóm (để debug/explain)
    top_rel_idx = relevant_probs.argmax().item()
    top_irr_idx = irrelevant_probs.argmax().item()

    return {
        "image_path": image_path,
        "relevant_score": round(relevant_score, 4),
        "irrelevant_score": round(irrelevant_score, 4),
        "is_irrelevant": irrelevant_score >= _DEFAULT_THRESHOLD,
        "top_relevant_prompt": RELEVANT_PROMPTS[top_rel_idx],
        "top_irrelevant_prompt": IRRELEVANT_PROMPTS[top_irr_idx],
    }


def filter_irrelevant_batch(
    image_paths: list,
    threshold: float = None,
    return_details: bool = False,
) -> dict:
    """Lọc batch ảnh: tách relevant và irrelevant.

    Hiệu quả hơn gọi detect_irrelevant_image() từng ảnh vì
    chỉ load model 1 lần cho toàn bộ batch.

    Args:
        image_paths (list[str]): Danh sách đường dẫn ảnh.
        threshold (float): Ngưỡng P(irrelevant). Mặc định: 0.55.
        return_details (bool): Nếu True, trả thêm chi tiết classify cho mỗi ảnh.

    Returns:
        dict: {
            "relevant": list[str],       — đường dẫn ảnh hợp lệ
            "irrelevant": list[str],     — đường dẫn ảnh rác
            "total": int,
            "relevant_count": int,
            "irrelevant_count": int,
            "details": list[dict],       — chỉ có khi return_details=True
        }
    """
    if threshold is None:
        threshold = _DEFAULT_THRESHOLD

    # Load model trước (singleton — chỉ load lần đầu)
    _load_clip_model()

    relevant = []
    irrelevant = []
    details = []

    for path in image_paths:
        result = classify_image_detail(path)
        if result is None:
            # Ảnh lỗi → giữ lại (safe default)
            relevant.append(path)
            continue

        if result["irrelevant_score"] >= threshold:
            irrelevant.append(path)
        else:
            relevant.append(path)

        if return_details:
            details.append(result)

    _logger.info(
        "CLIP filter: %d/%d relevant, %d irrelevant (threshold=%.2f)",
        len(relevant), len(image_paths), len(irrelevant), threshold,
    )

    output = {
        "relevant": relevant,
        "irrelevant": irrelevant,
        "total": len(image_paths),
        "relevant_count": len(relevant),
        "irrelevant_count": len(irrelevant),
    }
    if return_details:
        output["details"] = details

    return output
