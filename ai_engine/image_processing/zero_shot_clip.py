"""
zero_shot_clip.py
-----------------
Zero-shot Classification Pipeline using CLIP (OpenAI) to classify
e-commerce product review images into 2 categories:

  1. product    → Product / shipping box image → pass to ResNet50
  2. irrelevant → Unrelated image              → discard

Overall Pipeline:
    Review image → CLIP (binary) → [product] → ResNet50 → defect / no-defect
                                 → [irrelevant] → discard

Model: openai/clip-vit-base-patch32 (ViT-B/32)
  - Benchmark: outperformed ViT-B/16 on the real dataset (68% vs 60.5%)
  - Lighter (350MB), faster (~100-180ms/img CPU)
  - ~82 minutes for the entire 27K images

Prompt Design v4 (HYBRID + Book-fix):
  - PRODUCT: BROAD prompts (covering all kinds of products) → optimize product recall
  - IRRELEVANT: SPECIFIC prompts (based on analysis of 35 leaked images) → optimize irrelevant recall
  - Fix: Removed "book/notebooks" from irrelevant (books are valid e-commerce products)
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
    "openai/clip-vit-base-patch32"   # Outperformed B/16 on real dataset
)

# ── Prompt Sets v3 — HYBRID (broad product + specific irrelevant) ──────────
#
# Design rationale:
#
# v1 (old):  product=broad, irrelevant=basic
#            → Product Recall 95% ✅, Irrelevant Recall 30% ❌
#
# v2 (new):  product=shipping-only, irrelevant=targeted
#            → Product Recall 71% ❌ (too narrow!), Irrelevant Recall 60% ✅
#
# v3 (hybrid): product=broad (keeps v1), irrelevant=targeted (keeps v2)
PRODUCT_PROMPTS = [
    # Shipping boxes & packaging (most important — this is the input for ResNet50)
    "a photo of a product package or cardboard shipping box clearly visible",
    "an unboxing photo showing a delivered parcel or package with clear product",
    "a close-up of a cardboard shipping box or product packaging with visible labels",
    # Actual products (clothing, electronics, cosmetics...)
    "a sealed or wrapped product ready for delivery",
    "a consumer product sitting on a table or surface for inspection",
    "a product clearly photographed for an online shopping review",
    # Accessories & packaging
    "a shipping container or box with labels or barcodes showing the product",
    "a product being examined or held up showing its condition clearly",
    # Books and stationery are valid e-commerce products (not irrelevant)
    "a book, notebook, or stationery product clearly shown for sale or review",
    "a book lying flat on a desk, table, or floor clearly shown for review",
]

# ── Group 2: IRRELEVANT — Specific to patterns in Vietnamese e-commerce dataset ──
# Based on analysis of 35/50 leaked irrelevant images:
#   - Clothing / clothing tag → "person wearing/holding everyday item"
#   - Food / drinks without packaging → "cooked food, drinks in glass"
#   - Flowers / Tet decorations → "flowers, festive decorations"
#   - Thank you card / receipt → "thank you card, receipt"
#   - Vehicle / furniture → "vehicle, furniture, room"
#   - Selfie / crowd → "selfie, group photo"
#   - Screenshot / video frame → "screenshot, phone app"
# NOTE: Removed "book/notebooks" from irrelevant because books are valid e-commerce products
IRRELEVANT_PROMPTS = [
    # People (selfie, crowd, holding random items, K-pop idols)
    "a selfie, portrait, or group photo of people without any product",
    "a photo of a K-pop idol, celebrity, or singer on stage",
    "a photo of a child or baby playing or smiling without any product",
    # Pets — reinforced with multiple phrasings
    "a photo of a dog, cat, or pet animal with no product present",
    "an animal or pet resting or sitting with no product in frame",
    "a close-up photo of a dog or cat with no shipping box or product",
    # People using items (random household items, not product reviews)
    "a person wearing, holding, or using an everyday household item not for sale",
    "a person trying on clothes or wearing an outfit without clear product focus",
    # Served food & drinks — reinforced
    "cooked food or a meal served in a bowl, plate, or dish with no product packaging",
    "a photo of shrimp, seafood, or cooked dish in a bowl with no product box",
    "a photo of street food, restaurant food, or home-cooked meal on a table",
    "food or drinks without any e-commerce product packaging visible",
    # Screenshot / user interface / video frame — reinforced
    "a screenshot of a TikTok, Instagram, or social media video with interface elements",
    "a digital screenshot showing a phone screen, UI elements, status bar, or social media feed",
    "a screenshot of a phone app, chat message, or video thumbnail",
    "a picture containing only text, song lyrics, quotes, or typography",
    "a screenshot of a website, social media post, or online order confirmation",
    "a screenshot of a shopping cart page, order list, or mobile checkout screen",
    "a mobile screen capture showing product listings with price tags, checkboxes, and cart icon",
    "a screenshot of a shopee or tiki cart with product prices and quantity selectors",
    # Dark, blurry, or unrecognizable images — reinforced
    "a very dark or dimly lit photo where nothing is clearly visible",
    "a completely black image, very dark photo, or fully white blank image",
    "a very blurry, out-of-focus, or corrupted image with no recognizable content",
    "a dark photo with no clear product or box visible",
    # Flowers / gifts / decorations / Tet festival
    "flowers, bouquets, gift baskets, festive decorations, or ornaments",
    # Thank you cards / receipts / flyers
    "a thank you card, printed receipt, invoice, or promotional flyer",
    # Vehicles / furniture / outdoor scenes / rooms
    "a room interior, furniture, vehicle, motorcycle, or outdoor scene",
    "an empty floor, ceiling, or empty table without any product",
    # Abstract, solid colors, plain backgrounds
    "an abstract background, solid color wall, or black and white pattern",
    "a plain colored background or gradient with no product visible",
    # Nature / landscapes / sky
    "a landscape photo, nature scenery, sky, clouds, or outdoor view",
    # Memes / fanart / drawings / cartoons
    "a cartoon, anime illustration, meme, drawing, or digital artwork",
    "a map, diagram, chart, infographic, or hand-drawn sketch",
    # Everyday life scenes unrelated to products
    "an everyday lifestyle photo unrelated to any product for sale",
    # Collages of unrelated scenes
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
    # ─── Boxes / Packaging — Damaged ───
    "a product box that is visibly crushed, dented, or caved in from physical impact",
    "a shipping box with deeply collapsed corners or a completely caved-in side",
    "a cardboard package severely squeezed or compressed out of its original shape",
    "a product package with large permanent dents caused by heavy impact during shipping",
    "a product tube or cylindrical package that is severely dented or crushed",
    # ─── Milk Cans / Tin Cans — Damaged ───
    "a dented milk tin, metal can, or tin container showing a physical dent on its surface or rim",
    "a metal cylinder or tin can with a collapsed edge or dented metal body",
    "a milk can with a deformed rim, crushed metal lid, or dent on the side",
    "a tin container or milk powder can with visible denting or deformation from shipping",
    # ─── Torn / Broken / Cracked — Damaged ───
    "torn or ripped product packaging with clearly visible holes or large openings",
    "a product box where a corner or edge has been broken off or physically split open",
    # ─── Damp / Wet — Damaged ───
    "product packaging with visible wet stains or signs of severe water damage",
    # ─── Books — Damaged (corner/spine focus) ───
    "a book corner that is visibly bent, folded, or crushed from shipping impact",
    "a book spine that has been compressed or crushed causing visible deformation at the edge",
    "a close-up of a book showing a clearly bent or deformed corner due to physical damage",
    # ─── Clothing / Fashion — Damaged ───
    "a clothing item with torn fabric, ripped seams, or holes clearly visible from damage",
    "a garment or bag with clearly torn, frayed edges, or broken zipper from damage",
    "a fashion item or bag with broken strap, snapped clasp, or ripped material",
    # ─── Household / Drinkware — Damaged ───
    "a cup, mug, or bowl with a clearly chipped or cracked rim from impact",
    "a ceramic, glass, or porcelain product with visible cracks or broken pieces",
    "a plastic container or bottle that is cracked, broken, or has missing parts",
    # ─── Electronics / Phone Cases / Accessories — Damaged ───
    "a phone case, electronic device, or accessory with cracks, breaks, or snapped parts",
    "an electronic product with a visibly cracked, shattered, or broken component",
    # ─── Cosmetics / Jars / Bottles — Damaged ───
    "a cosmetic product with a shattered or broken glass jar or bottle",
    "a beauty product container that is cracked, leaked, or severely deformed",
    # ─── Footwear — Damaged ───
    "a shoe with a sole separating from the upper or with ripped, torn material",
    "footwear showing visible structural damage such as broken heel or torn stitching",
    # ─── Toys / Plastic Items — Damaged ───
    "a toy or plastic product with broken, snapped, or missing pieces",
    "a plastic item that is visibly cracked, shattered, or structurally broken",
    # ─── General — Severely Damaged ───
    "a product that is clearly destroyed, broken apart, or in completely unacceptable condition",
    "a product severely crushed or deformed beyond its original intended shape",
]

DEFECT_INTACT_PROMPTS = [
    # ─── General: Any Product — Intact ───
    "a product in perfect mint condition with no visible physical damage whatsoever",
    "a consumer product that appears completely undamaged and in its normal expected condition",
    "a product showing no signs of crushing, tearing, denting, or any physical deformation",
    "a product in the exact shape and form it was manufactured in, no defects visible",
    "a product that is fully intact, structurally sound, and in ready-to-use condition",
    "any object or item that is undamaged, whole, and in good condition",
    # ─── Products on Display / Under Review ───
    "a product photographed for an online review showing it is in good and intact condition",
    "a product being held in hand for a review photo, the item shows no damage at all",
    "a product placed on any surface for display, completely intact and undamaged",
    "a product after unboxing shown to the camera, item is in perfect expected condition",
    "a product next to its original box or packaging, both items are undamaged",
    "a product clearly shown for a shopping review with no visible defects or damage",
    # ─── Distinguishing Design Features vs Damage ───
    "a product with normal design features such as handles, bumps, ridges, or protrusions that are part of its intended design",
    "an item with a complex shape, texture, or geometry that is its normal undamaged form",
    "a product where all visible surfaces are smooth, complete, and without any deformation",
    "any object in its correct intact shape, no structural failure or breakage",
    # ─── Books — Intact ───
    "a book in pristine condition with a flat, clean cover and no bending or creases",
    "a book or notebook clearly intact with no spine damage or physical deformation",
    "an open book showing its interior pages, the book itself is not physically damaged",
    "a book lying open on a surface displaying its pages, no damage or deformation visible",
    "several books arranged together on a shelf or in a stack, all in good condition",
    "a book being held in hand clearly showing its undamaged cover for a review",
    "a book wrapped in transparent protective plastic film, undamaged and intact",
    "a book placed next to a Tiki or Shopee shipping box, book appears undamaged",
    # ─── Clothing / Bags / Fashion — Intact ───
    "a clothing item neatly folded, hung, or displayed with all seams and fabric intact",
    "a garment or outfit laid flat or worn showing it is undamaged and in good condition",
    "a bag, handbag, backpack, or wallet shown in perfect condition with no tears or defects",
    "a fashion accessory or clothing item with all zippers, buttons, and seams intact",
    # ─── Household / Drinkware / Tableware — Intact ───
    "a cup, mug, tumbler, or water bottle with a perfectly smooth and unchipped rim",
    "a glass, ceramic, or porcelain item in perfect condition with no cracks or chips",
    "household drinkware, kitchenware, or tableware shown intact and undamaged for a review",
    "a thermos, insulated bottle, or travel mug in mint condition with no dents or scratches",
    "a plastic or stainless steel container shown clearly intact for a product review",
    # ─── Electronics / Phone Cases / Phones / Accessories — Intact ───
    "a phone case, smartphone case, or mobile phone accessory in perfect undamaged condition",
    "a consumer electronics product shown in good intact condition for a review",
    "a smartphone, tablet, or electronic gadget with no visible damage, cracks, or dents",
    "an electronic device or accessory with all ports, buttons, and surfaces intact",
    "a speaker, earphone, charger, or electronic peripheral in mint undamaged condition",
    "a smartwatch, fitness tracker, or wearable device shown in good intact condition",
    # ─── Cosmetics / Beauty Products — Intact ───
    "a cosmetic product, skincare cream, or beauty item in sealed intact condition",
    "a perfume bottle, lotion tube, or makeup product with its container fully intact",
    "beauty or personal care products displayed in good undamaged condition for review",
    # ─── Footwear — Intact ───
    "a pair of shoes, sneakers, or sandals displayed in good condition with no damage",
    "footwear shown clearly for a review, both shoes are undamaged and in good shape",
    "a single shoe or sandal shown in intact condition with sole and upper fully attached",
    # ─── Toys / Plastic Items — Intact ───
    "a toy, game set, or children's product in complete undamaged condition",
    "a plastic product or household item in mint condition with no cracks or broken parts",
    # ─── Packaged Food / Supplements — Intact ───
    "a packaged food product, snack, or supplement in sealed intact original packaging",
    "a canned, bottled, or boxed food or beverage product in undamaged condition",
    "a milk tin, powder milk can, or metal tin container in perfect round shape with no dents",
    "a cylindrical metal can or food tin with smooth, even edges and no physical deformation",
    # ─── Watches / Jewelry / Accessories — Intact ───
    "a watch, bracelet, necklace, or jewelry item in undamaged good condition for review",
    "fashion accessories or jewelry displayed intact with no broken or missing parts",
    # ─── Boxes / Packaging — Intact ───
    "product packaging or a shipping box that arrived in perfect undamaged condition",
    "a sealed or opened box with all edges and corners intact, no crushing or tearing",
    # ─── General — Intact ───
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
    """Load CLIP model with singleton cache. Loaded exactly once."""
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
    """Force reload CLIP model (used to switch model variants).

    Args:
        model_name: Model name on HuggingFace. If None, reloads the current model.
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
    Core classification logic for one pre-opened PIL image.

    Algorithm:
    1. Calculate CLIP similarity between the image and all prompts
    2. Compute the MEAN LOGIT of each group
    3. Softmax over the two groups → binary probabilities
    4. Choose the group with the highest probability
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

    # ── Mean logit for each group → Softmax binary ──
    product_logit    = logits[:_N_PRODUCT].mean()
    irrelevant_logit = logits[_N_PRODUCT:].mean()

    group_logits = torch.stack([product_logit, irrelevant_logit])
    group_probs = torch.softmax(group_logits, dim=0)  # [2], sum = 1.0

    probs = {
        "product":    round(group_probs[0].item(), 4),
        "irrelevant": round(group_probs[1].item(), 4),
    }

    # Set threshold to 0.35 (suitable for Mean Logit) to filter cart screenshots accurately
    # without touching valid product images like books.
    _IRRELEVANT_THRESHOLD = 0.35
    if group_probs[1].item() > _IRRELEVANT_THRESHOLD:
        label = "irrelevant"
        confidence = group_probs[1].item()
    else:
        label = "product"
        confidence = group_probs[0].item()

    # Debug: best matching prompt for each group
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

    # Lower damage threshold to 0.61 (instead of 0.72) to be more sensitive to dents on milk tins
    # without causing confusion on normal intact product images.
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
    """Classify a single image into product / irrelevant.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: {
            "image_path": str,
            "label": "product" | "irrelevant",
            "confidence": float (0.0 – 1.0),
            "probs": {"product": float, "irrelevant": float},
            "top_prompts": {"product": str, "irrelevant": str},
        }
        None if image cannot be read.
    """
    if not os.path.exists(image_path):
        _logger.warning("Image does not exist: %s", image_path)
        return None

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        _logger.warning("Cannot read image '%s': %s", image_path, exc)
        return None

    result = _classify_single(image)
    result["image_path"] = image_path
    return result


def classify_batch(
    image_paths: List[str],
    return_details: bool = False,
) -> dict:
    """Classify a batch of images and group them by label.

    Args:
        image_paths (list[str]): List of image paths.
        return_details (bool): If True, returns detailed predictions for each image.

    Returns:
        dict: {
            "product":    list[str],  — product images (passed to ResNet50)
            "irrelevant": list[str],  — irrelevant images (discarded)
            "counts": {"product": int, "irrelevant": int},
            "total": int,
            "details": list[dict],    — only included if return_details=True
        }
    """
    _load_clip_model()

    groups = {"product": [], "irrelevant": []}
    details = []

    for path in image_paths:
        result = classify_image(path)
        if result is None:
            # Corrupted image → keep for ResNet50 (safe default)
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
    """Filter a batch of images, keeping only product images for ResNet50.

    Returns:
        dict: {
            "product":    list[str],  — paths for ResNet50
            "irrelevant": list[str],  — discarded paths
            "counts": dict,
            "total": int,
        }
    """
    return classify_batch(image_paths, return_details=False)


# ── Backward Compatibility ─────────────────────────────────────────────────

def detect_irrelevant_image(image_path: str, threshold: float = None) -> bool:
    """[DEPRECATED] Use classify_image() instead.
    Returns True if the image is NOT related to products.
    """
    result = classify_image(image_path)
    if result is None:
        return False
    return result["label"] == "irrelevant"


def classify_image_detail(image_path: str) -> Optional[dict]:
    """[DEPRECATED] Use classify_image() instead."""
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
    """[DEPRECATED] Use classify_batch() instead."""
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
