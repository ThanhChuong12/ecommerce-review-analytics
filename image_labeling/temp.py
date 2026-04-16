"""
media_pipeline.py
-----------------
Builds a training dataset from scraped review media (images/videos).
This is a standalone offline pipeline, not a backend service.
"""

from __future__ import annotations

import argparse
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
DATA_DIR = ROOT / "data"
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
            "review_text",
            "rating",
            "date",
            "source_url",
            "image_path",
            "origin",
            "frame_index",
        ],
    )


def _build_prompt(review_text: str, product_url: str) -> str:
    return (
        "You are an expert e-commerce media verifier. "
        "You will receive one image plus the review text and product URL/Name. "
        "Your task is to classify the image into EXACTLY ONE label based on its consistency with the review context.\n\n"
        "Return ONLY a valid JSON object like: {\"label\": \"your_choice\"}.\n"
        "Valid labels: 'intact', 'damaged', 'wrong_item', 'irrelevant'.\n\n"
        "Label rules (CRITICAL: Cross-reference image with Review text and Product URL):\n"
        
        "- intact: The image shows the CORRECT product and is in good condition. "
        "IMPORTANT: Close-up shots, zoomed-in details (like logos, text, or brand mascots), and opened products showing the inside contents (e.g., milk powder inside a can) MUST be classified as 'intact' if they reasonably belong to the product.\n"
        
        "- damaged: The image shows the CORRECT product, but it is clearly broken, dented, or flawed.\n"
        
        "- wrong_item: The image shows a real physical item, but it DOES NOT MATCH the product being sold. Use this for scam/mis-ship cases.\n"
        
        "- irrelevant: The image is NOT a real product photo (memes, internet screenshots, black images, random selfies). "
        "DO NOT use this label for close-up macro shots of the actual product.\n\n"
        
        "Decision Logic:\n"
        "1. Is it a meme or screenshot? -> 'irrelevant'.\n"
        "2. Is it a real item but clearly the WRONG product? -> 'wrong_item'.\n"
        "3. Is it the CORRECT product (even if it's a close-up or inside view) but broken/dented? -> 'damaged'.\n"
        "4. Is it the CORRECT product (even if it's a close-up or inside view) and looks fine? -> 'intact'.\n\n"
        
        f"Review text: {review_text}\n"
        f"Product Info/URL: {product_url}\n"
    )


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


# Label ảnh
def label_images(
    model_name: str,
    max_images: int | None,
    sleep_sec: float,
    copy_to_labels: bool,
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
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in your .env file.")

    from google import genai

    client = genai.Client(api_key=api_key)

    with open(IMAGES_MANIFEST, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    labeled: list[dict] = []
    total = len(rows)
    limit = max_images if max_images is not None else total

    print(f"[label] Images to label: {min(limit, len(rows))}")
    for row in tqdm(rows[:limit], desc="label:images"):
        image_path = _resolve_path(row.get("image_path", ""))
        if not image_path.exists():
            continue
        if _normalize_manifest_path(row.get("image_path", "")) in already_labeled:
            continue

        review_text = row.get("review_text", "")
        product_url = row.get("product_url", "")

        prompt = _build_prompt(review_text, product_url)
        try:
            with Image.open(image_path) as img:
                rgb = img.convert("RGB")
                buf = BytesIO()
                rgb.save(buf, format="JPEG")
                image_part = genai.types.Part.from_bytes(
                    data=buf.getvalue(),
                    mime_type="image/jpeg",
                )
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, image_part],
            )
            raw_text = response.text or ""
            data = _extract_json(raw_text)
        except Exception:
            raw_text = ""
            data = None

        label = None
        if data and isinstance(data, dict):
            label = data.get("label")
        if label not in IMAGE_LABELS:
            continue

        labeled.append(
            {
                "review_id": row.get("review_id", ""),
                "product_url": product_url,
                "review_text": review_text,
                "rating": row.get("rating", ""),
                "date": row.get("date", ""),
                "source_url": row.get("source_url", ""),
                "image_path": _normalize_manifest_path(row.get("image_path", "")),
                "label": label,
            }
        )

        if copy_to_labels:
            target = LABEL_DIR / label / image_path.name
            try:
                if target.exists():
                    pass
                else:
                    target.write_bytes(image_path.read_bytes())
            except OSError:
                pass

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    combined = existing_labels + labeled
    _write_csv(
        LABELS_CSV,
        combined,
        [
            "review_id",
            "product_url",
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

    label_p = sub.add_parser("label", help="Auto-label images with Gemini Vision")
    label_p.add_argument("--model", default="gemini-1.5-flash")
    label_p.add_argument("--max-images", type=int, default=None)
    label_p.add_argument("--sleep", type=float, default=0.3)
    label_p.add_argument("--no-copy", action="store_true")


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
            max_images=args.max_images,
            sleep_sec=args.sleep,
            copy_to_labels=copy_to_labels,
        )


if __name__ == "__main__":
    main()
