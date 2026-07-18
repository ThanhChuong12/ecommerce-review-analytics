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

Prompt Design v4 (HYBRID + Book-fix):
  - PRODUCT: prompts RỘNG (cover mọi loại sản phẩm) → tối ưu product recall
  - IRRELEVANT: prompts CỤ THỂ (dựa trên phân tích 35 ảnh bỏ lọt) → tối ưu irrelevant recall
  - Sửa: Xóa “sach/sach/notebooks” khỏi irrelevant (sách là sản phẩm TMĐT hợp lệ)
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
PRODUCT_PROMPTS = [
    # Hộp hàng & đóng gói (quan trọng nhất — đây là input cho ResNet50)
    "a photo of a product package or cardboard shipping box clearly visible",
    "an unboxing photo showing a delivered parcel or package with clear product",
    "a close-up of a cardboard shipping box or product packaging with visible labels",
    # Sản phẩm thực tế (quần áo, điện tử, mỹ phẩm...)
    "a sealed or wrapped product ready for delivery",
    "a consumer product sitting on a table or surface for inspection",
    "a product clearly photographed for an online shopping review",
    # Phụ kiện đi kèm & bao bì
    "a shipping container or box with labels or barcodes showing the product",
    "a product being examined or held up showing its condition clearly",
    # Sách và văn phòng phẩm là sản phẩm TMĐT hợp lệ (không phải irrelevant)
    "a book, notebook, or stationery product clearly shown for sale or review",
    "a book lying flat on a desk, table, or floor clearly shown for review",
]

# ── Nhóm 2: IRRELEVANT — Cụ thể theo pattern dataset TMĐT Việt Nam ──
# Dựa trên phân tích trực tiếp 35/50 ảnh irrelevant bị lọt:
#   - Quần áo / tag quần áo → "person wearing/holding everyday item"
#   - Thực phẩm / đồ uống không có bao bì sản phẩm → "cooked food, drinks in glass"
#   - Hoa / trang trí Tết → "flowers, festive decorations"
#   - Thiếp / hóa đơn → "thank you card, receipt"
#   - Xe máy / nội thất → "vehicle, furniture, room"
#   - Selfie / đám đông → "selfie, group photo"
#   - Screenshot / video frame → "screenshot, phone app"
# GHI CHÚ: Xóa “sach/notebooks” khỏi irrelevant vì sách là sản phẩm TMDT hợp lệ
IRRELEVANT_PROMPTS = [
    # Người (selfie, đám đông, tay cầm đồ random, idol kpop)
    "a selfie, portrait, or group photo of people without any product",
    "a photo of a K-pop idol, celebrity, or singer on stage",
    "a photo of a child or baby playing or smiling without any product",
    # Thú cưng — reinforced with multiple phrasings
    "a photo of a dog, cat, or pet animal with no product present",
    "an animal or pet resting or sitting with no product in frame",
    "a close-up photo of a dog or cat with no shipping box or product",
    # Người đang dùng đồ (đồ gia dụng random, không phải review)
    "a person wearing, holding, or using an everyday household item not for sale",
    "a person trying on clothes or wearing an outfit without clear product focus",
    # Thực phẩm & đồ uống đã được phục vụ — reinforced
    "cooked food or a meal served in a bowl, plate, or dish with no product packaging",
    "a photo of shrimp, seafood, or cooked dish in a bowl with no product box",
    "a photo of street food, restaurant food, or home-cooked meal on a table",
    "food or drinks without any e-commerce product packaging visible",
    # Screenshot / giao diện / video frame — reinforced
    "a screenshot of a TikTok, Instagram, or social media video with interface elements",
    "a digital screenshot showing a phone screen, UI elements, status bar, or social media feed",
    "a screenshot of a phone app, chat message, or video thumbnail",
    "a picture containing only text, song lyrics, quotes, or typography",
    "a screenshot of a website, social media post, or online order confirmation",
    "a screenshot of a shopping cart page, order list, or mobile checkout screen",
    "a mobile screen capture showing product listings with price tags, checkboxes, and cart icon",
    "a screenshot of a shopee or tiki cart with product prices and quantity selectors",
    # Ảnh tối, nhòe, không rõ nội dung — reinforced
    "a very dark or dimly lit photo where nothing is clearly visible",
    "a completely black image, very dark photo, or fully white blank image",
    "a very blurry, out-of-focus, or corrupted image with no recognizable content",
    "a dark photo with no clear product or box visible",
    # Hoa / quà tặng / trang trí / lễ hội Tết
    "flowers, bouquets, gift baskets, festive decorations, or ornaments",
    # Thiếp cảm ơn / hóa đơn / tờ rơi
    "a thank you card, printed receipt, invoice, or promotional flyer",
    # Xe / nội thất / cảnh ngoài trời / phòng ốc
    "a room interior, furniture, vehicle, motorcycle, or outdoor scene",
    "an empty floor, ceiling, or empty table without any product",
    # Abstract, màu sắc, nền trơn
    "an abstract background, solid color wall, or black and white pattern",
    "a plain colored background or gradient with no product visible",
    # Thiên nhiên / phong cảnh / bầu trời
    "a landscape photo, nature scenery, sky, clouds, or outdoor view",
    # Meme / fanart / hình ảnh vẽ / hoạt hình
    "a cartoon, anime illustration, meme, drawing, or digital artwork",
    "a map, diagram, chart, infographic, or hand-drawn sketch",
    # Cảnh sinh hoạt thường ngày không liên quan sản phẩm
    "an everyday lifestyle photo unrelated to any product for sale",
    # Ảnh ghép nhiều cảnh không liên quan / collage
    "a collage of unrelated photos, screenshots, or random images without product focus",
]

_N_PRODUCT = len(PRODUCT_PROMPTS)
_N_IRRELEVANT = len(IRRELEVANT_PROMPTS)
_ALL_PROMPTS = PRODUCT_PROMPTS + IRRELEVANT_PROMPTS

_GROUP_LABELS = ["product", "irrelevant"]

# ── Defect Detection Prompts (DAMAGED vs INTACT) ────────────────────────────
#
# Engineered for Vietnamese e-commerce product review photos:
#   - Covers book, electronics, cosmetics, clothing packaging
#   - Focuses on physical deformation, not just cosmetic imperfections
#   - "Intact" prompts emphasize structural integrity to avoid false positives
#
# Design: mean-logit per group → softmax binary → threshold at 0.5

DEFECT_DAMAGED_PROMPTS = [
    # ─── Hộp / Bao Bì — Hư Hỏng ───
    "a product box that is visibly crushed, dented, or caved in from physical impact",
    "a shipping box with deeply collapsed corners or a completely caved-in side",
    "a cardboard package severely squeezed or compressed out of its original shape",
    "a product package with large permanent dents caused by heavy impact during shipping",
    "a product tube or cylindrical package that is severely dented or crushed",
    # ─── Lon Sữa / Hộp Thiếc — Hư Hỏng ───
    "a dented milk tin, metal can, or tin container showing a physical dent on its surface or rim",
    "a metal cylinder or tin can with a collapsed edge or dented metal body",
    "a milk can with a deformed rim, crushed metal lid, or dent on the side",
    "a tin container or milk powder can with visible denting or deformation from shipping",
    # ─── Rách / Gãy / Vỡ — Hư Hỏng ───
    "torn or ripped product packaging with clearly visible holes or large openings",
    "a product box where a corner or edge has been broken off or physically split open",
    # ─── Ẩm / Ướt — Hư Hỏng ───
    "product packaging with large visible wet stains or signs of severe water damage",
    # ─── Sách — Hư Hỏng (chỉ góc/gáy cụ thể) ───
    "a book corner that is visibly bent, folded, or crushed from shipping impact",
    "a book spine that has been compressed or crushed causing visible deformation at the edge",
    "a close-up of a book showing a clearly bent or deformed corner due to physical damage",
    # ─── Quần Áo / Thời Trang — Hư Hỏng ───
    "a clothing item with torn fabric, ripped seams, or holes clearly visible from damage",
    "a garment or bag with clearly torn, frayed edges, or broken zipper from damage",
    "a fashion item or bag with broken strap, snapped clasp, or ripped material",
    # ─── Đồ Gia Dụng / Cốc Bình — Hư Hỏng ───
    "a cup, mug, or bowl with a clearly chipped or cracked rim from impact",
    "a ceramic, glass, or porcelain product with visible cracks or broken pieces",
    "a plastic container or bottle that is cracked, broken, or has missing parts",
    # ─── Điện Tử / Ỏp Lưng / Phụ Kiện — Hư Hỏng ───
    "a phone case, electronic device, or accessory with cracks, breaks, or snapped parts",
    "an electronic product with a visibly cracked, shattered, or broken component",
    # ─── Mỹ Phẩm / Lọ / Hủ — Hư Hỏng ───
    "a cosmetic product with a shattered or broken glass jar or bottle",
    "a beauty product container that is cracked, leaked, or severely deformed",
    # ─── Giày Dép — Hư Hỏng ───
    "a shoe with a sole separating from the upper or with ripped, torn material",
    "footwear showing visible structural damage such as broken heel or torn stitching",
    # ─── Đồ Chơi / Nhựa — Hư Hỏng ───
    "a toy or plastic product with broken, snapped, or missing pieces",
    "a plastic item that is visibly cracked, shattered, or structurally broken",
    # ─── Tổng Quát — Hư Hỏng Nặng ───
    "a product that is clearly destroyed, broken apart, or in completely unacceptable condition",
    "a product severely crushed or deformed beyond its original intended shape",
]

DEFECT_INTACT_PROMPTS = [
    # ─── Phổ Quát: Bất kỳ sản phẩm nào — Nguyên Vẹn ───
    "a product in perfect mint condition with no visible physical damage whatsoever",
    "a consumer product that appears completely undamaged and in its normal expected condition",
    "a product showing no signs of crushing, tearing, denting, or any physical deformation",
    "a product in the exact shape and form it was manufactured in, no defects visible",
    "a product that is fully intact, structurally sound, and in ready-to-use condition",
    "any object or item that is undamaged, whole, and in good condition",
    # ─── Sản Phẩm Đang Trưng Bày / Được Review ───
    "a product photographed for an online review showing it is in good and intact condition",
    "a product being held in hand for a review photo, the item shows no damage at all",
    "a product placed on any surface for display, completely intact and undamaged",
    "a product after unboxing shown to the camera, item is in perfect expected condition",
    "a product next to its original box or packaging, both items are undamaged",
    "a product clearly shown for a shopping review with no visible defects or damage",
    # ─── Phân Biệt Đặc Điểm Thiết Kế vs Hư Hỏng ───
    "a product with normal design features such as handles, bumps, ridges, or protrusions that are part of its intended design",
    "an item with a complex shape, texture, or geometry that is its normal undamaged form",
    "a product where all visible surfaces are smooth, complete, and without any deformation",
    "any everyday object in its correct intact shape, no structural failure or breakage",
    # ─── Sách — Nguyên Vẹn ───
    "a book in pristine condition with a flat, clean cover and no bending or creases",
    "a book or notebook clearly intact with no spine damage or physical deformation",
    "an open book showing its interior pages, the book itself is not physically damaged",
    "a book lying open on a surface displaying its pages, no damage or deformation visible",
    "several books arranged together on a shelf or in a stack, all in good condition",
    "a book being held in hand clearly showing its undamaged cover for a review",
    "a book wrapped in transparent protective plastic film, undamaged and intact",
    "a book placed next to a Tiki or Shopee shipping box, book appears undamaged",
    # ─── Quần Áo / Túi Xách / Thời Trang — Nguyên Vẹn ───
    "a clothing item neatly folded, hung, or displayed with all seams and fabric intact",
    "a garment or outfit laid flat or worn showing it is undamaged and in good condition",
    "a bag, handbag, backpack, or wallet shown in perfect condition with no tears or defects",
    "a fashion accessory or clothing item with all zippers, buttons, and seams intact",
    # ─── Đồ Gia Dụng / Cốc Bình / Bát Dĩa — Nguyên Vẹn ───
    "a cup, mug, tumbler, or water bottle with a perfectly smooth and unchipped rim",
    "a glass, ceramic, or porcelain item in perfect condition with no cracks or chips",
    "household drinkware, kitchenware, or tableware shown intact and undamaged for a review",
    "a thermos, insulated bottle, or travel mug in mint condition with no dents or scratches",
    "a plastic or stainless steel container shown clearly intact for a product review",
    # ─── Điện Tử / Ỏp Lưng / Điện Thoại / Phụ Kiện — Nguyên Vẹn ───
    "a phone case, smartphone case, or mobile phone accessory in perfect undamaged condition",
    "a consumer electronics product shown in good intact condition for a review",
    "a smartphone, tablet, or electronic gadget with no visible damage, cracks, or dents",
    "an electronic device or accessory with all ports, buttons, and surfaces intact",
    "a speaker, earphone, charger, or electronic peripheral in mint undamaged condition",
    "a smartwatch, fitness tracker, or wearable device shown in good intact condition",
    # ─── Mỹ Phẩm / Làm Đẹp — Nguyên Vẹn ───
    "a cosmetic product, skincare cream, or beauty item in sealed intact condition",
    "a perfume bottle, lotion tube, or makeup product with its container fully intact",
    "beauty or personal care products displayed in good undamaged condition for review",
    # ─── Giày Dép — Nguyên Vẹn ───
    "a pair of shoes, sneakers, or sandals displayed in good condition with no damage",
    "footwear shown clearly for a review, both shoes are undamaged and in good shape",
    "a single shoe or sandal shown in intact condition with sole and upper fully attached",
    # ─── Đồ Chơi / Đồ Nhựa — Nguyên Vẹn ───
    "a toy, game set, or children's product in complete undamaged condition",
    "a plastic product or household item in mint condition with no cracks or broken parts",
    # ─── Thực Phẩm Đóng Gói / Thực Phẩm Chức Năng — Nguyên Vẹn ───
    "a packaged food product, snack, or supplement in sealed intact original packaging",
    "a canned, bottled, or boxed food or beverage product in undamaged condition",
    "a milk tin, powder milk can, or metal tin container in perfect round shape with no dents",
    "a cylindrical metal can or food tin with smooth, even edges and no physical deformation",
    # ─── Đồng Hồ / Trang Sức / Phụ Kiện — Nguyên Vẹn ───
    "a watch, bracelet, necklace, or jewelry item in undamaged good condition for review",
    "fashion accessories or jewelry displayed intact with no broken or missing parts",
    # ─── Hộp / Bao Bì — Nguyên Vẹn ───
    "product packaging or a shipping box that arrived in perfect undamaged condition",
    "a sealed or opened box with all edges and corners intact, no crushing or tearing",
    # ─── Tổng Quát — Nguyên Vẹn ───
    "a product delivered in excellent condition with no signs of shipping damage",
    "a product that arrived safely, undamaged, and in perfectly acceptable condition",
    "a product reviewed positively with no complaints about physical damage or defects",
    "an e-commerce product photo showing the item received is in good intact condition",
]




_N_DEFECT_DAMAGED = len(DEFECT_DAMAGED_PROMPTS)
_N_DEFECT_INTACT  = len(DEFECT_INTACT_PROMPTS)
_ALL_DEFECT_PROMPTS = DEFECT_DAMAGED_PROMPTS + DEFECT_INTACT_PROMPTS
_DEFECT_LABELS = ["damaged", "intact"]

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

    # Đặt threshold ở mức 0.35 (phù hợp với Mean Logit) để lọc chính xác screenshot giỏ hàng (43.4%)
    # mà không chạm tới ảnh sản phẩm thực tế như sách vở (chỉ chiếm 0.2% - 2%).
    _IRRELEVANT_THRESHOLD = 0.35
    if group_probs[1].item() > _IRRELEVANT_THRESHOLD:
        label = "irrelevant"
        confidence = group_probs[1].item()
    else:
        label = "product"
        confidence = group_probs[0].item()

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

def _classify_defect_single(image: Image.Image) -> dict:
    """Core CLIP defect classification for one PIL image.

    Returns probabilities for 'damaged' and 'intact' via mean-logit softmax.
    """
    model, processor, device = _load_clip_model()

    inputs = processor(
        text=_ALL_DEFECT_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]  # [N_damaged + N_intact]

    damaged_logit = logits[:_N_DEFECT_DAMAGED].mean()
    intact_logit  = logits[_N_DEFECT_DAMAGED:].mean()

    group_logits = torch.stack([damaged_logit, intact_logit])
    group_probs  = torch.softmax(group_logits, dim=0)  # [2]

    probs = {
        "damaged": round(group_probs[0].item(), 4),
        "intact":  round(group_probs[1].item(), 4),
    }

    # Đặt threshold móp méo xuống 0.61 (thay vì 0.72 cũ) để phát hiện nhạy hơn các vết móp trên lon sữa
    # mà không gây nhầm lẫn trên các ảnh sản phẩm nguyên vẹn bình thường.
    _DAMAGED_THRESHOLD = 0.61
    if group_probs[0].item() > _DAMAGED_THRESHOLD:
        label = "damaged"
        confidence = group_probs[0].item()
    else:
        label = "intact"
        confidence = group_probs[1].item()

    # Top matching prompt for debug
    top_damaged = DEFECT_DAMAGED_PROMPTS[logits[:_N_DEFECT_DAMAGED].argmax().item()]
    top_intact  = DEFECT_INTACT_PROMPTS[logits[_N_DEFECT_DAMAGED:].argmax().item()]

    return {
        "label":      label,
        "confidence": round(confidence, 4),
        "probs":      probs,
        "top_prompts": {
            "damaged": top_damaged,
            "intact":  top_intact,
        },
    }


def classify_defect_clip(image_path: str) -> Optional[dict]:
    """Zero-shot CLIP defect classification for a single image.

    Classifies a product review image as 'damaged' or 'intact' using
    CLIP zero-shot similarity against carefully engineered prompt sets.
    Reuses the singleton CLIP model cache (no extra memory cost).

    Args:
        image_path: Path to the image file.

    Returns:
        dict with keys:
          - label (str): 'damaged' | 'intact'
          - confidence (float): probability of the predicted label (0-1)
          - probs (dict): {'damaged': float, 'intact': float}
          - top_prompts (dict): best matching prompt per group (for debug)
          - image_path (str): echoed back input path
        None if the image cannot be opened.
    """
    if not os.path.exists(image_path):
        _logger.warning("Image not found: %s", image_path)
        return None

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        _logger.warning("Cannot read image '%s': %s", image_path, exc)
        return None

    result = _classify_defect_single(image)
    result["image_path"] = image_path
    return result


def classify_defect_clip_batch(
    image_paths: List[str],
) -> List[Optional[dict]]:
    """Zero-shot CLIP defect classification for a batch of images.

    Args:
        image_paths: List of image file paths.

    Returns:
        List of dicts in the same order as image_paths.
        None entries for images that cannot be opened.
    """
    _load_clip_model()
    return [classify_defect_clip(p) for p in image_paths]


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
