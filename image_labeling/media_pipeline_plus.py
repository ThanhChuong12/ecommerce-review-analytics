"""
media_pipeline.py
-----------------
Builds a training dataset from scraped review media (images/videos).
This is a standalone offline pipeline, not a backend service.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import httpx
from dotenv import find_dotenv, load_dotenv
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "new_data"
RAW_MEDIA_DIR = DATA_DIR / "raw_media"
FRAMES_DIR = DATA_DIR / "frames"
MANIFEST_DIR = DATA_DIR / "manifests"
LABEL_DIR = DATA_DIR / "labeled"

MEDIA_MANIFEST = MANIFEST_DIR / "media.csv"
IMAGES_MANIFEST = MANIFEST_DIR / "images.csv"
LABELS_CSV = MANIFEST_DIR / "labels.csv"

IMAGE_LABELS = {"intact", "damaged", "wrong_item", "irrelevant"}


def _resolve_path(path_str: str) -> Path:
    if not path_str:
        return Path(path_str)
    path = Path(path_str)
    return path if path.is_absolute() else (ROOT / path)


def _normalize_manifest_path(path_str: str) -> str:
    if not path_str:
        return ""
    path = Path(path_str)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        parts = [p.lower() for p in path.parts]
        for idx, part in enumerate(parts):
            if part == "image_labeling":
                rel = Path(*path.parts[idx + 1 :])
                return str(rel)
        return str(path)


@dataclass
class ReviewRow:
    review_id: str
    product_url: str
    product_name: str
    review_text: str
    rating: str
    date: str


def _ensure_dirs() -> None:
    RAW_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    for label in IMAGE_LABELS:
        (LABEL_DIR / label).mkdir(parents=True, exist_ok=True)


def _iter_csv_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.csv")))
        elif path.is_file():
            files.append(path)
    return files


def _parse_image_urls(raw: str) -> list[str]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split("|")]
    return [p for p in parts if p]


def _make_review_id(row: dict, row_index: int) -> str:
    raw = "|".join(
        [
            str(row.get("product_url", "")),
            str(row.get("text", ""))[:200],
            str(row.get("date", "")),
            str(row.get("rating", "")),
            str(row_index),
        ]
    )
    return hashlib.md5(raw.encode("utf-8", errors="replace")).hexdigest()[:12]


def _is_video_url(url: str) -> bool:
    lowered = url.lower()
    if any(ext in lowered for ext in [".mp4", ".mov", ".webm", ".mkv"]):
        return True
    return "video" in lowered and "mp4" in lowered


def _safe_write_image(image: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    rgb.save(out_path, format="JPEG", quality=92)


def _download_image(url: str, out_path: Path, timeout: float) -> bool:
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content))
            _safe_write_image(image, out_path)
        return True
    except (httpx.HTTPError, UnidentifiedImageError, OSError):
        return False


def _download_video(url: str, out_path: Path, timeout: float) -> bool:
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
        return True
    except httpx.HTTPError:
        return False


def _read_csv_rows(csv_path: Path) -> Iterable[dict]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# Ảnh và video chứa trong folder data/raw_media và file media.csv
def download_media(csv_inputs: list[str], timeout: float, seed: int | None) -> None:
    _ensure_dirs()
    if seed is not None:
        random.seed(seed)

    csv_files = _iter_csv_files(csv_inputs)
    if not csv_files:
        raise ValueError("No CSV files found in inputs.")

    print(f"[download] CSV files: {len(csv_files)}")
    existing_rows: list[dict] = []
    existing_paths: set[str] = set()
    if MEDIA_MANIFEST.exists():
        with open(MEDIA_MANIFEST, newline="", encoding="utf-8-sig") as f:
            existing_rows = list(csv.DictReader(f))
        existing_paths = {
            _normalize_manifest_path(row.get("local_path", "")) for row in existing_rows
        }
        print(f"[download] Resume from manifest: {len(existing_rows)} items")

    new_rows: list[dict] = []

    for csv_path in csv_files:
        rows = list(_read_csv_rows(csv_path))
        print(f"[download] Reading: {csv_path} ({len(rows)} rows)")
        for idx, row in enumerate(tqdm(rows, desc=f"download:{csv_path.name}")):
            review_id = _make_review_id(row, idx)
            review_text = str(row.get("text", ""))
            product_url = str(row.get("product_url", ""))
            product_name = str(row.get("product_name", ""))
            rating = str(row.get("rating", ""))
            date = str(row.get("date", ""))
            urls = _parse_image_urls(str(row.get("image_urls", "")))

            for media_idx, url in enumerate(urls, start=1):
                is_video = _is_video_url(url)
                if is_video:
                    local_path = RAW_MEDIA_DIR / f"{review_id}_media{media_idx}.mp4"
                    media_type = "video"
                else:
                    local_path = RAW_MEDIA_DIR / f"{review_id}_img{media_idx}.jpg"
                    media_type = "image"

                local_path_rel = str(local_path.relative_to(ROOT))
                local_path_abs = _resolve_path(local_path_rel)

                if local_path_rel in existing_paths and local_path_abs.exists():
                    continue

                if local_path_abs.exists():
                    ok = True
                else:
                    if is_video:
                        ok = _download_video(url, local_path_abs, timeout)
                    else:
                        ok = _download_image(url, local_path_abs, timeout)

                if not ok:
                    continue

                new_rows.append(
                    {
                        "review_id": review_id,
                        "product_url": product_url,
                        "product_name": product_name,
                        "review_text": review_text,
                        "rating": rating,
                        "date": date,
                        "source_url": url,
                        "media_type": media_type,
                        "local_path": local_path_rel,
                    }
                )

    print(f"[download] Saved media manifest: {MEDIA_MANIFEST}")
    combined = existing_rows + new_rows
    _write_csv(
        MEDIA_MANIFEST,
        combined,
        [
            "review_id",
            "product_url",
            "product_name",
            "review_text",
            "rating",
            "date",
            "source_url",
            "media_type",
            "local_path",
        ],
    )


# Ảnh frame được trích từ các video bỏ vào trong folder data/frames, tạo file
# images.csv lúc này chỉ chứa các frame từ video
def extract_frames(frames_per_video: int, seed: int | None) -> None:
    _ensure_dirs()
    if seed is not None:
        random.seed(seed)

    if not MEDIA_MANIFEST.exists():
        raise ValueError("media.csv not found. Run download first.")

    import cv2

    images_rows: list[dict] = []

    with open(MEDIA_MANIFEST, newline="", encoding="utf-8-sig") as f:
        reader = list(csv.DictReader(f))
        print(f"[extract] Videos to process: {sum(1 for r in reader if r.get('media_type') == 'video')}")
        for row in tqdm(reader, desc="extract:videos"):
            media_type = row.get("media_type", "")
            local_path = _resolve_path(row.get("local_path", ""))
            if media_type != "video" or not local_path.exists():
                continue

            cap = cv2.VideoCapture(str(local_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames <= 0:
                cap.release()
                continue

            picks = sorted(
                random.sample(
                    range(total_frames),
                    k=min(frames_per_video, total_frames),
                )
            )

            review_id = row.get("review_id", "unknown")
            for idx, frame_index in enumerate(picks, start=1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                out_path = FRAMES_DIR / f"{review_id}_img{idx}.jpg"
                cv2.imwrite(str(out_path), frame)
                out_rel = str(out_path.relative_to(ROOT))

                images_rows.append(
                    {
                        "review_id": review_id,
                        "product_url": row.get("product_url", ""),
                        "product_name": row.get("product_name", ""),
                        "review_text": row.get("review_text", ""),
                        "rating": row.get("rating", ""),
                        "date": row.get("date", ""),
                        "source_url": row.get("source_url", ""),
                        "image_path": out_rel,
                        "origin": "frame",
                        "frame_index": str(frame_index),
                    }
                )

            cap.release()

    existing = []
    if IMAGES_MANIFEST.exists():
        with open(IMAGES_MANIFEST, newline="", encoding="utf-8-sig") as f:
            existing = list(csv.DictReader(f))

    seen_paths = {row.get("image_path", "") for row in existing}
    combined = existing + [r for r in images_rows if r.get("image_path") not in seen_paths]
    print(f"[extract] Saved images manifest: {IMAGES_MANIFEST}")
    _write_csv(
        IMAGES_MANIFEST,
        combined,
        [
            "review_id",
            "product_url",
            "product_name",
            "review_text",
            "rating",
            "date",
            "source_url",
            "image_path",
            "origin",
            "frame_index",
        ],
    )

# Xóa ảnh lỗi 
def validate_images() -> None:
    _ensure_dirs()

    if not IMAGES_MANIFEST.exists():
        raise ValueError("images.csv not found. Run build-images or extract first.")

    valid_rows: list[dict] = []
    with open(IMAGES_MANIFEST, newline="", encoding="utf-8-sig") as f:
        reader = list(csv.DictReader(f))
        print(f"[validate] Images to check: {len(reader)}")
        for row in tqdm(reader, desc="validate:images"):
            image_path = _resolve_path(row.get("image_path", ""))
            if not image_path.exists():
                continue
            try:
                with Image.open(image_path) as img:
                    img.verify()
                row["image_path"] = _normalize_manifest_path(row.get("image_path", ""))
                valid_rows.append(row)
            except (UnidentifiedImageError, OSError):
                try:
                    image_path.unlink()
                except OSError:
                    pass

    _write_csv(IMAGES_MANIFEST, valid_rows, list(valid_rows[0].keys()) if valid_rows else [])
    print(f"[validate] Valid images: {len(valid_rows)}")


# Đưa các ảnh gốc trong media.csv vào trong images.csv
def build_images_manifest() -> None:
    _ensure_dirs()

    if not MEDIA_MANIFEST.exists():
        raise ValueError("media.csv not found. Run download first.")

    image_rows: list[dict] = []
    with open(MEDIA_MANIFEST, newline="", encoding="utf-8-sig") as f:
        reader = list(csv.DictReader(f))
        print(f"[build-images] Total media items: {len(reader)}")
        for row in tqdm(reader, desc="build-images:downloaded"):
            if row.get("media_type") != "image":
                continue
            image_path = _normalize_manifest_path(row.get("local_path", ""))
            image_rows.append(
                {
                    "review_id": row.get("review_id", ""),
                    "product_url": row.get("product_url", ""),
                    "product_name": row.get("product_name", ""),
                    "review_text": row.get("review_text", ""),
                    "rating": row.get("rating", ""),
                    "date": row.get("date", ""),
                    "source_url": row.get("source_url", ""),
                    "image_path": image_path,
                    "origin": "download",
                    "frame_index": "",
                }
            )

    existing = []
    if IMAGES_MANIFEST.exists():
        with open(IMAGES_MANIFEST, newline="", encoding="utf-8-sig") as f:
            existing = list(csv.DictReader(f))

    seen_paths = {row.get("image_path", "") for row in existing}
    combined = existing + [r for r in image_rows if r.get("image_path") not in seen_paths]
    print(f"[build-images] Saved images manifest: {IMAGES_MANIFEST}")
    _write_csv(
        IMAGES_MANIFEST,
        combined,
        [
            "review_id",
            "product_url",
            "product_name",
            "review_text",
            "rating",
            "date",
            "source_url",
            "image_path",
            "origin",
            "frame_index",
        ],
    )


def _build_prompt(review_text: str, product_name: str) -> str:
    return (
        "You are an expert e-commerce media verifier. Analyze the relationship between the Image, Review Text, and Product Name.\n\n"
        "Your task is to classify the image into EXACTLY ONE label based on its physical condition and context. Remember: The ultimate goal is to prepare images for a Computer Vision model that only sees pixels, not text.\n\n"
        "Return ONLY a valid JSON object strictly formatted like this: {\"reasoning\": \"briefly explain the relationship between the image and the text, and justify your label choice\", \"label\": \"your_choice\"}.\n"
        "Valid labels: 'intact', 'damaged', 'wrong_item', 'irrelevant'.\n\n"
        
        "*** ABSOLUTE OVERRIDE RULE 1 (VARIANTS vs MISMATCH) ***:\n"
        "Understand the difference between a VARIANT and a MISMATCH:\n"
        "- VARIANT: Same brand/product line but different stage, size, or color (e.g., Product is Abbott Grow 1+, Image is Abbott Grow 3; Product is Black shirt, Image is Red shirt). Label variants as 'intact' (physically undamaged).\n"
        "- MISMATCH: Completely different brand or entirely different item (e.g., Product is Abbott Milk, Image is 'Imochild' supplement, or a shoe). If it's a MISMATCH, you MUST apply Rule 2.\n\n"
        
        "*** ABSOLUTE OVERRIDE RULE 2 (COIN FARMING vs SELLER MISTAKE) ***:\n"
        "When there is a MISMATCH (the image is NOT the exact product being sold):\n"
        "  - Step A: Read the Review Text.\n"
        "  - Step B: If text is POSITIVE, NEUTRAL, or GIBBERISH (e.g., 'good', 'sản phẩm chính hãng', 'tốt', 'mmm') -> It is 100% COIN FARMING SPAM. You MUST label it 'irrelevant' (even if the image is a real product like medicine or toys).\n"
        "  - Step C: If and ONLY IF text is NEGATIVE and EXPLICITLY COMPLAINS about a wrong delivery ('giao sai', 'lừa đảo') -> Label it 'wrong_item'.\n\n"
        
        "*** ABSOLUTE OVERRIDE RULE 3 (SHADOWS, REFLECTIONS & BLUR) ***:\n"
        "CAMERA BLUR IS NOT DAMAGE. Blurry, out-of-focus images, lighting shadows, reflections on metallic surfaces, or the natural structural design of the product (e.g., horizontal ridges/rings on milk cans) MUST NOT be mistaken for physical damage (dents/crushes). If it looks like a dent but could just be a shadow/reflection, and the Review Text DOES NOT explicitly complain about a dent or damage, you MUST assume it is 'intact'.\n\n"
        
        "*** ABSOLUTE OVERRIDE RULE 4 (PACKAGING & UNBOXING) ***:\n"
        "Images of brown shipping boxes, taped parcels, bubble wrap, or hands using scissors are NORMAL. DO NOT label them 'wrong_item' or 'irrelevant'. Unless the actual product inside is visibly broken, label the packaging/unboxing process as 'intact'.\n\n"
        
        "Label Definitions:\n"
        "- intact: CORRECT product or VARIANT (physically fine, including blur, shadows, reflections, packaging, inside contents).\n"
        "- damaged: ACTUAL PHYSICAL product/packaging is visibly broken, dented, crushed, torn, or leaking. Confirmed by visual evidence and usually text.\n"
        "- wrong_item: MISMATCH item + text EXPLICITLY complains about wrong delivery.\n"
        "- irrelevant: Random images, pets, selfies, black screens, video transition artifacts, OR coin farming spam.\n\n"
        
        "*** EXAMPLES TO LEARN FROM (FEW-SHOT) ***\n"
        "Ex 1: Image = Box of 'Imochild' supplement. Product = 'Abbott Grow Milk'. Text = 'Sản phẩm chính hãng, rất tốt'. -> Label = 'irrelevant' (Mismatched brand but positive review = coin farming).\n"
        "Ex 2: Image = Blurry can of Abbott. Text = 'Sữa ngon'. -> Label = 'intact' (Blurry is not damaged).\n"
        "Ex 3: Image = Red Jeans. Product = Black Jeans. Text = 'Giao sai màu'. -> Label = 'intact' (It is a VARIANT, physically fine for CV training).\n"
        "Ex 4: Image = Cute dog. Text = 'Giao hàng cực nhanh'. -> Label = 'irrelevant' (Coin farming spam).\n"
        "Ex 5: Image = Can of Abbott Grow 3. Product = Abbott Grow 1+. Text = 'mmmmmmm'. -> Label = 'intact' (It is a VARIANT, visually an intact milk can).\n"
        "Ex 6: Image = Can of milk with a huge dent. Text = 'Lon bị móp, thất vọng'. -> Label = 'damaged'.\n"
        "Ex 7: Image = A pair of shoes. Product = Abbott Grow Milk. Text = 'Shop làm ăn chán, đặt sữa giao giày'. -> Label = 'wrong_item'.\n"
        "Ex 8: Image = Screenshot of Shopee order page. Text = 'Sữa thơm ngon'. -> Label = 'irrelevant' (Not a photo of a physical product).\n"
        "Ex 9: Image = Zoomed in picture of an expiration date. Text = 'Date xa'. -> Label = 'intact' (Close-up detail of the product).\n"
        "Ex 10: Image = Half-torn bubble wrap revealing a perfectly fine milk can. Text = 'Bọc kỹ'. -> Label = 'intact' (Normal unboxing).\n"
        "Ex 11: Image = Metal milk can with a dark shadow or reflection that looks like a dent. Text = 'Sữa tốt'. -> Label = 'intact' (Shadows/reflections on structural ridges are not physical damage, especially when text is positive).\n"
        "Ex 12: Image = A glass of prepared milk OR open can showing powder inside. Text = 'Bé rất thích uống'. -> Label = 'intact' (Showing the inside contents or product in use is normal, not a wrong item).\n"
        "Ex 13: Image = Milk powder spilled everywhere inside the parcel. Text = 'Vỡ nắp đổ hết ra ngoài'. -> Label = 'damaged' (Clear physical destruction and leaking).\n"
        "Ex 14: Image = Can of 'Ensure' milk. Product = 'Abbott Grow'. Text = 'Shop gửi nhầm loại Ensure rồi, đổi trả sao đây'. -> Label = 'wrong_item' (MISMATCH brand + explicit complaint about wrong delivery).\n"
        "Ex 15: Image = Completely black or white frame (caught during video transition). Text = 'Sữa ok'. -> Label = 'irrelevant' (No product visible due to video artifact, useless for training).\n\n"
        
        f"Review text: {review_text}\n"
        f"Product name: {product_name}\n"
    )


def _build_batch_prompt(items: list[dict]) -> str:
    header = (
        "You are an expert e-commerce media verifier. "
        "You will receive multiple images, each with its review text and product name. "
        "Your task is to classify EACH image into EXACTLY ONE label based on its physical condition and context. Remember: The ultimate goal is to prepare images for a Computer Vision model that only sees pixels, not text.\n\n"
        "Return ONLY a valid JSON array strictly formatted like this: [{\"id\": 1, \"reasoning\": \"briefly explain...\", \"label\": \"intact\"}].\n"
        "Valid labels: 'intact', 'damaged', 'wrong_item', 'irrelevant'.\n\n"
        
        "*** ABSOLUTE OVERRIDE RULE 1 (VARIANTS vs MISMATCH) ***:\n"
        "Understand the difference between a VARIANT and a MISMATCH:\n"
        "- VARIANT: Same brand/product line but different stage, size, or color (e.g., Product is Abbott Grow 1+, Image is Abbott Grow 3; Product is Black shirt, Image is Red shirt). Label variants as 'intact' (physically undamaged).\n"
        "- MISMATCH: Completely different brand or entirely different item (e.g., Product is Abbott Milk, Image is 'Imochild' supplement, or a shoe). If it's a MISMATCH, you MUST apply Rule 2.\n\n"
        
        "*** ABSOLUTE OVERRIDE RULE 2 (COIN FARMING vs SELLER MISTAKE) ***:\n"
        "When there is a MISMATCH (the image is NOT the exact product being sold):\n"
        "  - Step A: Read the Review Text.\n"
        "  - Step B: If text is POSITIVE, NEUTRAL, or GIBBERISH (e.g., 'good', 'sản phẩm chính hãng', 'tốt', 'mmm') -> It is 100% COIN FARMING SPAM. You MUST label it 'irrelevant' (even if the image is a real product like medicine or toys).\n"
        "  - Step C: If and ONLY IF text is NEGATIVE and EXPLICITLY COMPLAINS about a wrong delivery ('giao sai', 'lừa đảo') -> Label it 'wrong_item'.\n\n"
        
        "*** ABSOLUTE OVERRIDE RULE 3 (SHADOWS, REFLECTIONS & BLUR) ***:\n"
        "CAMERA BLUR IS NOT DAMAGE. Blurry, out-of-focus images, lighting shadows, reflections on metallic surfaces, or the natural structural design of the product (e.g., horizontal ridges/rings on milk cans) MUST NOT be mistaken for physical damage (dents/crushes). If it looks like a dent but could just be a shadow/reflection, and the Review Text DOES NOT explicitly complain about a dent or damage, you MUST assume it is 'intact'.\n\n"
        
        "*** ABSOLUTE OVERRIDE RULE 4 (PACKAGING & UNBOXING) ***:\n"
        "Images of brown shipping boxes, taped parcels, bubble wrap, or hands using scissors are NORMAL. DO NOT label them 'wrong_item' or 'irrelevant'. Unless the actual product inside is visibly broken, label the packaging/unboxing process as 'intact'.\n\n"
        
        "Label Definitions:\n"
        "- intact: CORRECT product or VARIANT (physically fine, including blur, shadows, reflections, packaging, inside contents).\n"
        "- damaged: ACTUAL PHYSICAL product/packaging is visibly broken, dented, crushed, torn, or leaking. Confirmed by visual evidence and usually text.\n"
        "- wrong_item: MISMATCH item + text EXPLICITLY complains about wrong delivery.\n"
        "- irrelevant: Random images, pets, selfies, black screens, video transition artifacts, OR coin farming spam.\n\n"
        
        "*** EXAMPLES TO LEARN FROM (FEW-SHOT) ***\n"
        "Ex 1: Image = Box of 'Imochild' supplement. Product = 'Abbott Grow Milk'. Text = 'Sản phẩm chính hãng, rất tốt'. -> Label = 'irrelevant' (Mismatched brand but positive review = coin farming).\n"
        "Ex 2: Image = Blurry can of Abbott. Text = 'Sữa ngon'. -> Label = 'intact' (Blurry is not damaged).\n"
        "Ex 3: Image = Red Jeans. Product = Black Jeans. Text = 'Giao sai màu'. -> Label = 'intact' (It is a VARIANT, physically fine for CV training).\n"
        "Ex 4: Image = Cute dog. Text = 'Giao hàng cực nhanh'. -> Label = 'irrelevant' (Coin farming spam).\n"
        "Ex 5: Image = Can of Abbott Grow 3. Product = Abbott Grow 1+. Text = 'mmmmmmm'. -> Label = 'intact' (It is a VARIANT, visually an intact milk can).\n"
        "Ex 6: Image = Can of milk with a huge dent. Text = 'Lon bị móp, thất vọng'. -> Label = 'damaged'.\n"
        "Ex 7: Image = A pair of shoes. Product = Abbott Grow Milk. Text = 'Shop làm ăn chán, đặt sữa giao giày'. -> Label = 'wrong_item'.\n"
        "Ex 8: Image = Screenshot of Shopee order page. Text = 'Sữa thơm ngon'. -> Label = 'irrelevant' (Not a photo of a physical product).\n"
        "Ex 9: Image = Zoomed in picture of an expiration date. Text = 'Date xa'. -> Label = 'intact' (Close-up detail of the product).\n"
        "Ex 10: Image = Half-torn bubble wrap revealing a perfectly fine milk can. Text = 'Bọc kỹ'. -> Label = 'intact' (Normal unboxing).\n"
        "Ex 11: Image = Metal milk can with a dark shadow or reflection that looks like a dent. Text = 'Sữa tốt'. -> Label = 'intact' (Shadows/reflections on structural ridges are not physical damage, especially when text is positive).\n"
        "Ex 12: Image = A glass of prepared milk OR open can showing powder inside. Text = 'Bé rất thích uống'. -> Label = 'intact' (Showing the inside contents or product in use is normal, not a wrong item).\n"
        "Ex 13: Image = Milk powder spilled everywhere inside the parcel. Text = 'Vỡ nắp đổ hết ra ngoài'. -> Label = 'damaged' (Clear physical destruction and leaking).\n"
        "Ex 14: Image = Can of 'Ensure' milk. Product = 'Abbott Grow'. Text = 'Shop gửi nhầm loại Ensure rồi, đổi trả sao đây'. -> Label = 'wrong_item' (MISMATCH brand + explicit complaint about wrong delivery).\n"
        "Ex 15: Image = Completely black or white frame (caught during video transition). Text = 'Sữa ok'. -> Label = 'irrelevant' (No product visible due to video artifact, useless for training).\n\n"
        "Items:\n"
    )
    lines = [header]
    for item in items:
        lines.append(f"ID: {item['id']}")
        lines.append(f"Review text: {item['review_text']}")
        lines.append(f"Product name: {item['product_name']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _init_provider(provider: str):
    provider = provider.strip().lower()
    # if provider == "google":
    #     api_key = os.getenv("GOOGLE_API_KEY")
    #     if not api_key:
    #         raise ValueError("GOOGLE_API_KEY is not set in your .env file.")
    #     from google import genai

    #     return provider, genai.Client(api_key=api_key)
    if provider == "google":
        from google import genai
        
        PROJECT_ID = os.getenv("PROJECT_ID")
        
        client = genai.Client(
            vertexai=True, 
            project=PROJECT_ID,
            location="global"
        )
        return provider, client
    if provider in {"openai", "groq", "custom"}:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = None
        elif provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        else:
            api_key = os.getenv("CUSTOM_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("CUSTOM_BASE_URL") or os.getenv("OPENAI_BASE_URL")

        if not api_key:
            raise ValueError("API key is not set for the selected provider.")

        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        return provider, client

    raise ValueError(f"Unsupported provider: {provider}")


# Label ảnh
def label_images(
    model_name: str,
    provider: str,
    max_images: int | None,
    sleep_sec: float,
    copy_to_labels: bool,
    batch_size: int,
) -> None:
    _ensure_dirs()

    if not IMAGES_MANIFEST.exists():
        raise ValueError("images.csv not found. Run build-images first.")

    existing_labels: list[dict] = []
    already_labeled: set[str] = set()
    if LABELS_CSV.exists():
        with open(LABELS_CSV, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                existing_labels.append(row)
                already_labeled.add(row.get("image_path", ""))

    load_dotenv(find_dotenv(usecwd=True))

    provider, client = _init_provider(provider)

    with open(IMAGES_MANIFEST, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    labeled: list[dict] = []
    total = len(rows)
    limit = max_images if max_images is not None else total

    print(f"[label] Images to label: {min(limit, len(rows))}")
    rows_to_label = rows[:limit]
    batch_size = max(1, int(batch_size))

    if provider == "google" and batch_size > 1:
        from google import genai

        batch: list[dict] = []
        for row in tqdm(rows_to_label, desc="label:images"):
            image_path = _resolve_path(row.get("image_path", ""))
            if not image_path.exists():
                print(f"[label][skip] Missing file: {image_path}")
                continue
            if _normalize_manifest_path(row.get("image_path", "")) in already_labeled:
                print(f"[label][skip] Already labeled: {image_path.name}")
                continue

            batch.append(row)
            if len(batch) < batch_size:
                continue

            items = []
            image_parts = []
            for idx, item in enumerate(batch, start=1):
                image_path = _resolve_path(item.get("image_path", ""))
                try:
                    with Image.open(image_path) as img:
                        rgb = img.convert("RGB")
                        buf = BytesIO()
                        rgb.save(buf, format="JPEG")
                        image_bytes = buf.getvalue()
                    image_parts.append(
                        genai.types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/jpeg",
                        )
                    )
                except Exception as exc:
                    print(f"[label][error] {image_path.name}: {type(exc).__name__}: {exc}")
                    continue

                items.append(
                    {
                        "id": idx,
                        "row": item,
                        "image_path": image_path,
                        "review_text": item.get("review_text", ""),
                        "product_name": item.get("product_name", ""),
                    }
                )

            if not items:
                batch = []
                continue

            prompt = _build_batch_prompt(items)
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, *image_parts],
                )
                raw_text = response.text or ""
                print(f"[label][model] batch={len(items)} | raw={raw_text[:300]!r}")
                data = _extract_json(raw_text)
            except Exception as exc:
                data = None
                print(f"[label][error] batch: {type(exc).__name__}: {exc}")

            if not isinstance(data, list):
                print("[label][skip] Invalid batch response (not a JSON array)")
                batch = []
                continue

            labels_by_id = {}
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                label = entry.get("label")
                item_id = entry.get("id")
                if label in IMAGE_LABELS and isinstance(item_id, int):
                    labels_by_id[item_id] = label

            for item in items:
                label = labels_by_id.get(item["id"])
                if label not in IMAGE_LABELS:
                    print(
                        f"[label][skip] Invalid label: {item['image_path'].name} | id={item['id']}"
                    )
                    continue

                row = item["row"]
                labeled.append(
                    {
                        "review_id": row.get("review_id", ""),
                        "product_url": row.get("product_url", ""),
                        "product_name": row.get("product_name", ""),
                        "review_text": row.get("review_text", ""),
                        "rating": row.get("rating", ""),
                        "date": row.get("date", ""),
                        "source_url": row.get("source_url", ""),
                        "image_path": _normalize_manifest_path(row.get("image_path", "")),
                        "label": label,
                    }
                )

                if copy_to_labels:
                    target = LABEL_DIR / label / item["image_path"].name
                    try:
                        if not target.exists():
                            target.write_bytes(item["image_path"].read_bytes())
                    except OSError as exc:
                        print(f"[label][error] Copy failed: {item['image_path'].name}: {exc}")
                else:
                    print(f"[label][info] Copy disabled: {item['image_path'].name}")

            _write_csv(
                LABELS_CSV,
                existing_labels + labeled,
                ["review_id", "product_url", "product_name", "review_text", "rating", "date", "source_url", "image_path", "label"]
            )
            
            if sleep_sec > 0:
                time.sleep(sleep_sec)

            batch = []

        if batch:
            rows_to_label = batch
        else:
            rows_to_label = []

    for row in rows_to_label:
        image_path = _resolve_path(row.get("image_path", ""))
        if not image_path.exists():
            print(f"[label][skip] Missing file: {image_path}")
            continue
        if _normalize_manifest_path(row.get("image_path", "")) in already_labeled:
            print(f"[label][skip] Already labeled: {image_path.name}")
            continue

        review_text = row.get("review_text", "")
        product_url = row.get("product_url", "")
        product_name = row.get("product_name", "")

        prompt = _build_prompt(review_text, product_name)
        try:
            with Image.open(image_path) as img:
                rgb = img.convert("RGB")
                buf = BytesIO()
                rgb.save(buf, format="JPEG")
                image_bytes = buf.getvalue()

            if provider == "google":
                from google import genai

                image_part = genai.types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, image_part],
                )
                raw_text = response.text or ""
            else:
                image_b64 = base64.b64encode(image_bytes).decode("ascii")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}",
                                        "detail": "low",
                                    },
                                },
                            ],
                        }
                    ],
                )
                raw_text = (response.choices[0].message.content or "").strip()

            print(f"[label][model] {image_path.name}: {raw_text}")
            data = _extract_json(raw_text)
        except Exception as exc:
            raw_text = ""
            data = None
            print(f"[label][error] {image_path.name}: {type(exc).__name__}: {exc}")

        label = None
        if data and isinstance(data, dict):
            label = data.get("label")
        if label not in IMAGE_LABELS:
            print(
                f"[label][skip] Invalid label: {image_path.name} | raw={raw_text[:200]!r}"
            )
            continue

        labeled.append(
            {
                "review_id": row.get("review_id", ""),
                "product_url": product_url,
                "product_name": product_name,
                "review_text": review_text,
                "rating": row.get("rating", ""),
                "date": row.get("date", ""),
                "source_url": row.get("source_url", ""),
                "image_path": _normalize_manifest_path(row.get("image_path", "")),
                "label": label,
            }
        )

        if copy_to_labels:
            # target = LABEL_DIR / label / image_path.name
            target = LABEL_DIR / label / f"{image_path.parent.name}_{image_path.name}"
            try:
                if target.exists():
                    pass
                else:
                    target.write_bytes(image_path.read_bytes())
            except OSError as exc:
                print(f"[label][error] Copy failed: {image_path.name}: {exc}")
        else:
            print(f"[label][info] Copy disabled: {image_path.name}")

        _write_csv(
            LABELS_CSV,
            existing_labels + labeled,
            ["review_id", "product_url", "product_name", "review_text", "rating", "date", "source_url", "image_path", "label"]
        )
        
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    combined = existing_labels + labeled
    _write_csv(
        LABELS_CSV,
        combined,
        [
            "review_id",
            "product_url",
            "product_name",
            "review_text",
            "rating",
            "date",
            "source_url",
            "image_path",
            "label",
        ],
    )
    print(f"[label] Saved labels: {LABELS_CSV}")



def main() -> None:
    parser = argparse.ArgumentParser(
        prog="media_pipeline.py",
        description="Build a labeled training dataset from scraped review media.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    download_p = sub.add_parser("download", help="Download image/video media from CSVs")
    download_p.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="CSV file(s) or directory containing CSV files",
    )
    download_p.add_argument("--timeout", type=float, default=30.0)
    download_p.add_argument("--seed", type=int, default=None)

    extract_p = sub.add_parser("extract", help="Extract frames from downloaded videos")
    extract_p.add_argument("--frames", type=int, default=3)
    extract_p.add_argument("--seed", type=int, default=None)

    build_p = sub.add_parser("build-images", help="Build images.csv from downloaded images")

    validate_p = sub.add_parser("validate", help="Validate images and remove corrupted files")

    label_p = sub.add_parser("label", help="Auto-label images with a vision model")
    label_p.add_argument(
        "--provider",
        choices=["google", "openai", "groq", "custom"],
        default="openai",
        help="Vision provider: google | openai | groq | custom",
    )
    label_p.add_argument("--model", default="gpt-4.1")
    label_p.add_argument("--max-images", type=int, default=None)
    label_p.add_argument("--sleep", type=float, default=0.3)
    label_p.add_argument("--no-copy", action="store_true")
    label_p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for Gemini (google) only. Use 5-10 to reduce calls.",
    )


    args = parser.parse_args()

    if args.command == "download":
        download_media(args.csv, timeout=args.timeout, seed=args.seed)
    elif args.command == "extract":
        extract_frames(frames_per_video=args.frames, seed=args.seed)
    elif args.command == "build-images":
        build_images_manifest()
    elif args.command == "validate":
        validate_images()
    elif args.command == "label":
        copy_to_labels = not args.no_copy
        label_images(
            model_name=args.model,
            provider=args.provider,
            max_images=args.max_images,
            sleep_sec=args.sleep,
            copy_to_labels=copy_to_labels,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()