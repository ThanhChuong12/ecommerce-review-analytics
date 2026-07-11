"""
extract_embeddings.py
======================
Rut trich (extract) embeddings tu pre-trained PhoBERT hoac MobileNetV3
de phuc vu cho FeatureDenoiser training.

Usage:
    # Extract text embeddings tu PhoBERT
    py scripts/extract_embeddings.py text \
        --csv data/processed/processed_labeled_text_train.csv \
        --text-col text \
        --model-name vinai/phobert-base-v2 \
        --output data/processed/text_embeddings.pt \
        --batch-size 32

    # Extract text embeddings tu PhoBERT da fine-tune
    py scripts/extract_embeddings.py text \
        --csv data/processed/processed_labeled_text_train.csv \
        --model-dir ai_engine/models/weights/phobert_best \
        --output data/processed/text_embeddings.pt

    # Extract image embeddings tu MobileNetV3
    py scripts/extract_embeddings.py image \
        --image-dir data/images/ \
        --model-path ai_engine/models/weights/mobilenet_v3_defect.pt \
        --output data/processed/image_embeddings.pt
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


# ════════════════════════════════════════════════════════════════════════════
#  TEXT EMBEDDING EXTRACTION (PhoBERT)
# ════════════════════════════════════════════════════════════════════════════

def extract_text_embeddings(
    csv_path: str,
    text_col: str = "text",
    model_name: str = "vinai/phobert-base-v2",
    model_dir: str | None = None,
    batch_size: int = 32,
    max_length: int = 256,
    max_samples: int | None = None,
    output_path: str = "data/processed/text_embeddings.pt",
) -> None:
    """Extract [CLS] embeddings from PhoBERT for all reviews in a CSV.

    Uses the hidden state of the [CLS] token from the last layer
    as the sentence embedding (before the classification head).

    Args:
        csv_path: Path to CSV file with review text.
        text_col: Column name containing review text.
        model_name: HuggingFace model name (used if model_dir is None).
        model_dir: Path to fine-tuned PhoBERT directory (overrides model_name).
        batch_size: Batch size for inference.
        max_length: Max token sequence length.
        max_samples: Limit number of samples (None = all).
        output_path: Where to save the .pt file.
    """
    from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    texts = df[text_col].fillna("").astype(str).tolist()

    if max_samples:
        texts = texts[:max_samples]
    logger.info("Total texts: %d", len(texts))

    # ── Load model ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = model_dir if model_dir else model_name
    logger.info("Loading model from: %s (device: %s)", source, device)

    tokenizer = AutoTokenizer.from_pretrained(source)

    # Try loading as sequence classification model first (fine-tuned),
    # fall back to base model
    try:
        full_model = AutoModelForSequenceClassification.from_pretrained(source)
        # Extract the base model (without classification head)
        if hasattr(full_model, "roberta"):
            model = full_model.roberta  # PhoBERT/RoBERTa
        elif hasattr(full_model, "bert"):
            model = full_model.bert
        else:
            model = AutoModel.from_pretrained(source)
        logger.info("Loaded fine-tuned model, using base encoder for embeddings")
    except Exception:
        model = AutoModel.from_pretrained(source)
        logger.info("Loaded base model")

    model = model.to(device)
    model.eval()

    # ── Extract embeddings ─────────────────────────────────────────────────
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        cleaned = [" ".join(t.strip().split()) for t in batch_texts]

        encodings = tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # [CLS] token embedding = first token of last hidden state
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu())

        if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(texts):
            logger.info(
                "  Processed %d/%d texts (%.1f%%)",
                min(i + batch_size, len(texts)),
                len(texts),
                min(i + batch_size, len(texts)) / len(texts) * 100,
            )

    # ── Save ───────────────────────────────────────────────────────────────
    embeddings = torch.cat(all_embeddings, dim=0)
    logger.info("Embedding shape: %s", embeddings.shape)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "source": source}, out_path)
    logger.info("Saved → %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


# ════════════════════════════════════════════════════════════════════════════
#  IMAGE EMBEDDING EXTRACTION (MobileNetV3)
# ════════════════════════════════════════════════════════════════════════════

def extract_image_embeddings(
    image_dir: str,
    model_path: str = "ai_engine/models/weights/mobilenet_v3_defect.pt",
    batch_size: int = 32,
    max_samples: int | None = None,
    output_path: str = "data/processed/image_embeddings.pt",
) -> None:
    """Extract feature embeddings from MobileNetV3 for all images.

    Uses the feature vector from the penultimate layer (before classifier).

    Args:
        image_dir: Directory containing images.
        model_path: Path to MobileNetV3 checkpoint.
        batch_size: Batch size for inference.
        max_samples: Limit number of images.
        output_path: Where to save the .pt file.
    """
    from torchvision import transforms, models
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Collect image paths ────────────────────────────────────────────────
    img_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    img_dir = Path(image_dir)
    img_paths = sorted([
        p for p in img_dir.rglob("*")
        if p.suffix.lower() in img_extensions
    ])

    if max_samples:
        img_paths = img_paths[:max_samples]
    logger.info("Total images: %d", len(img_paths))

    if not img_paths:
        logger.error("No images found in %s", image_dir)
        return

    # ── Load model ─────────────────────────────────────────────────────────
    logger.info("Loading MobileNetV3 from: %s", model_path)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Build MobileNetV3-Large and load weights
    model = models.mobilenet_v3_large(weights=None)

    # Adjust classifier for the number of classes in checkpoint
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Detect number of classes from classifier weights
    classifier_key = "classifier.3.weight"
    if classifier_key in state_dict:
        num_classes = state_dict[classifier_key].shape[0]
        model.classifier[3] = torch.nn.Linear(
            model.classifier[3].in_features, num_classes
        )

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Hook to extract features before classifier
    features_store = []

    def hook_fn(module, input, output):
        features_store.append(input[0].detach().cpu())

    # Register hook on the classifier to capture its input
    hook = model.classifier.register_forward_hook(hook_fn)

    # ── Transform ──────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ── Extract embeddings ─────────────────────────────────────────────────
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i : i + batch_size]
        batch_tensors = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(transform(img))
            except Exception as e:
                logger.warning("Skipped %s: %s", p.name, e)
                # Use zero tensor as placeholder
                batch_tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            model(batch)  # features captured by hook

        if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(img_paths):
            logger.info(
                "  Processed %d/%d images (%.1f%%)",
                min(i + batch_size, len(img_paths)),
                len(img_paths),
                min(i + batch_size, len(img_paths)) / len(img_paths) * 100,
            )

    hook.remove()

    # ── Save ───────────────────────────────────────────────────────────────
    embeddings = torch.cat(features_store, dim=0)
    logger.info("Embedding shape: %s", embeddings.shape)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "source": model_path}, out_path)
    logger.info("Saved → %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


# ════════════════════════════════════════════════════════════════════════════
#  PAIRED TEXT+IMAGE EXTRACTION (for Denoiser training)
# ════════════════════════════════════════════════════════════════════════════

def extract_paired_embeddings(
    paired_csv: str = "data/processed/paired_text_image.csv",
    text_model_dir: str | None = None,
    text_model_name: str = "vinai/phobert-base-v2",
    image_model_path: str = "d:/ecommerce-backup/resnet50_improve_with_weights/ai_engine/models/resnet50_defect_gpu_best.pth",
    batch_size: int = 16,
    max_length: int = 256,
    output_dir: str = "data/processed",
) -> None:
    """Extract paired text+image embeddings for Denoiser training.

    Reads paired_text_image.csv and extracts embeddings in matching order
    so that text_embeddings[i] corresponds to image_embeddings[i].
    """
    from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
    from torchvision import transforms, models
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load paired CSV ────────────────────────────────────────────────────
    csv_path = ROOT / paired_csv
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    texts = df["text"].fillna("").astype(str).tolist()
    image_paths = df["image_path"].tolist()
    logger.info("Loaded %d pairs from %s", len(df), csv_path.name)

    # ══════════════════════════════════════════════════════════════════════
    #  TEXT EMBEDDINGS (PhoBERT)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("  EXTRACTING TEXT EMBEDDINGS (PhoBERT)")
    logger.info("=" * 60)

    source = text_model_dir if text_model_dir else text_model_name
    logger.info("Loading model from: %s", source)
    tokenizer = AutoTokenizer.from_pretrained(source)

    try:
        model_text = AutoModelForSequenceClassification.from_pretrained(source)
        backbone = model_text.roberta if hasattr(model_text, "roberta") else model_text.base_model
        logger.info("Loaded fine-tuned model, using backbone for [CLS] extraction")
    except Exception:
        backbone = AutoModel.from_pretrained(source)
        logger.info("Loaded base model")

    backbone = backbone.to(device)
    backbone.eval()

    all_text_emb = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        cleaned = [" ".join(str(t).strip().split()) for t in batch_texts]

        encodings = tokenizer(
            cleaned, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, "last_hidden_state"):
                cls_emb = outputs.last_hidden_state[:, 0, :]
            else:
                cls_emb = outputs[0][:, 0, :]

        all_text_emb.append(cls_emb.cpu())

        if (i // batch_size + 1) % 50 == 0 or i + batch_size >= len(texts):
            logger.info("  Text: %d/%d (%.1f%%)", min(i + batch_size, len(texts)), len(texts),
                        min(i + batch_size, len(texts)) / len(texts) * 100)

    text_embeddings = torch.cat(all_text_emb, dim=0)
    logger.info("Text embedding shape: %s", text_embeddings.shape)

    # Clean up text model
    del backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    #  IMAGE EMBEDDINGS (ResNet50)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("  EXTRACTING IMAGE EMBEDDINGS (ResNet50)")
    logger.info("=" * 60)

    logger.info("Loading ResNet50 from: %s", image_model_path)
    checkpoint = torch.load(image_model_path, map_location=device, weights_only=False)

    # Build ResNet50
    model_img = models.resnet50(weights=None)

    # Handle checkpoint format
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Detect num_classes from fc layer
    fc_key = "fc.weight"
    if fc_key in state_dict:
        num_classes = state_dict[fc_key].shape[0]
        model_img.fc = torch.nn.Linear(model_img.fc.in_features, num_classes)
        logger.info("ResNet50 num_classes: %d", num_classes)

    model_img.load_state_dict(state_dict, strict=False)
    model_img = model_img.to(device)
    model_img.eval()

    # Hook to extract features before fc layer
    features_store = []

    def hook_fn(module, input, output):
        features_store.append(input[0].detach().cpu())

    hook = model_img.fc.register_forward_hook(hook_fn)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(transform(img))
            except Exception as e:
                logger.warning("Skipped %s: %s", p, e)
                batch_tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            model_img(batch)

        if (i // batch_size + 1) % 50 == 0 or i + batch_size >= len(image_paths):
            logger.info("  Image: %d/%d (%.1f%%)", min(i + batch_size, len(image_paths)), len(image_paths),
                        min(i + batch_size, len(image_paths)) / len(image_paths) * 100)

    hook.remove()

    image_embeddings = torch.cat(features_store, dim=0)
    logger.info("Image embedding shape: %s", image_embeddings.shape)

    # ══════════════════════════════════════════════════════════════════════
    #  SAVE
    # ══════════════════════════════════════════════════════════════════════
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text_out = out_dir / "paired_text_embeddings.pt"
    image_out = out_dir / "paired_image_embeddings.pt"

    torch.save({
        "embeddings": text_embeddings,
        "source": source,
        "n_pairs": len(df),
    }, text_out)

    torch.save({
        "embeddings": image_embeddings,
        "source": image_model_path,
        "n_pairs": len(df),
    }, image_out)

    logger.info("Saved text  → %s (%.1f MB)", text_out, text_out.stat().st_size / 1e6)
    logger.info("Saved image → %s (%.1f MB)", image_out, image_out.stat().st_size / 1e6)
    logger.info("DONE! %d paired embeddings ready for Denoiser training.", len(df))


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Extract embeddings from PhoBERT, MobileNetV3, or paired datasets"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Extraction mode")

    # ── Text subcommand ────────────────────────────────────────────────────
    text_parser = subparsers.add_parser("text", help="Extract text embeddings")
    text_parser.add_argument("--csv", required=True, help="CSV file with reviews")
    text_parser.add_argument("--text-col", default="text", help="Text column name")
    text_parser.add_argument(
        "--model-name", default="vinai/phobert-base-v2",
        help="HuggingFace model name (default: vinai/phobert-base-v2)",
    )
    text_parser.add_argument(
        "--model-dir", default=None,
        help="Path to fine-tuned model directory (overrides --model-name)",
    )
    text_parser.add_argument("--batch-size", type=int, default=32)
    text_parser.add_argument("--max-length", type=int, default=256)
    text_parser.add_argument("--max-samples", type=int, default=None)
    text_parser.add_argument(
        "--output", default="data/processed/text_embeddings.pt",
        help="Output .pt file path",
    )

    # ── Image subcommand ───────────────────────────────────────────────────
    img_parser = subparsers.add_parser("image", help="Extract image embeddings")
    img_parser.add_argument("--image-dir", required=True, help="Image directory")
    img_parser.add_argument(
        "--model-path", default="ai_engine/models/weights/mobilenet_v3_defect.pt",
        help="MobileNetV3 checkpoint path",
    )
    img_parser.add_argument("--batch-size", type=int, default=32)
    img_parser.add_argument("--max-samples", type=int, default=None)
    img_parser.add_argument(
        "--output", default="data/processed/image_embeddings.pt",
        help="Output .pt file path",
    )

    # ── Paired subcommand ──────────────────────────────────────────────────
    paired_parser = subparsers.add_parser("paired", help="Extract paired text+image embeddings")
    paired_parser.add_argument(
        "--paired-csv", default="data/processed/paired_text_image.csv",
        help="Paired CSV from build_paired_dataset.py",
    )
    paired_parser.add_argument(
        "--text-model-dir", default=None,
        help="Fine-tuned PhoBERT directory",
    )
    paired_parser.add_argument(
        "--text-model-name", default="vinai/phobert-base-v2",
        help="HuggingFace model name (if no --text-model-dir)",
    )
    paired_parser.add_argument(
        "--image-model-path",
        default="d:/ecommerce-backup/resnet50_improve_with_weights/ai_engine/models/resnet50_defect_gpu_best.pth",
        help="ResNet50 checkpoint path",
    )
    paired_parser.add_argument("--batch-size", type=int, default=16)
    paired_parser.add_argument("--output-dir", default="data/processed")

    args = parser.parse_args()

    if args.mode == "text":
        extract_text_embeddings(
            csv_path=args.csv,
            text_col=args.text_col,
            model_name=args.model_name,
            model_dir=args.model_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_samples=args.max_samples,
            output_path=args.output,
        )
    elif args.mode == "image":
        extract_image_embeddings(
            image_dir=args.image_dir,
            model_path=args.model_path,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            output_path=args.output,
        )
    elif args.mode == "paired":
        extract_paired_embeddings(
            paired_csv=args.paired_csv,
            text_model_dir=args.text_model_dir,
            text_model_name=args.text_model_name,
            image_model_path=args.image_model_path,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

