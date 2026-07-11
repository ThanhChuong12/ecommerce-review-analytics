"""
evaluate_pipeline_comparison.py
================================
So sánh pipeline cũ vs pipeline mới (Denoiser) trên cùng test set.

Pipeline cũ text:  PhoBERT fine-tuned → predict trực tiếp (có head gốc)
Pipeline mới text: PhoBERT embedding → FeatureDenoiser → MLP head mới

Pipeline cũ ảnh:   ResNet50 fine-tuned (binary: intact vs defect)
Pipeline mới ảnh:  ResNet50 embedding → FeatureDenoiser → 4-class MLP head

Usage:
    py scripts/evaluate_pipeline_comparison.py
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_engine.denoising.feature_denoiser import FeatureDenoiser
from scripts.train_classification_heads import ClassificationHead


# ════════════════════════════════════════════════════════════════════════════
#  Metrics helper
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(preds: torch.Tensor, labels: torch.Tensor, class_names: list[str]) -> dict:
    correct = (preds == labels).float()
    accuracy = correct.mean().item()

    per_class = {}
    for i, name in enumerate(class_names):
        mask = labels == i
        if mask.sum() == 0:
            continue
        pred_mask = preds == i
        tp = (pred_mask & mask).sum().item()
        fp = (pred_mask & ~mask).sum().item()
        fn = (~pred_mask & mask).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(mask.sum()),
        }

    macro_f1 = np.mean([v["f1"] for v in per_class.values()])
    return {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4), "per_class": per_class}


def print_results(title: str, metrics: dict, class_names: list[str]):
    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)
    print(f"  Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro F1 : {metrics['macro_f1']:.4f}")
    print()
    for cls in class_names:
        if cls not in metrics["per_class"]:
            continue
        m = metrics["per_class"][cls]
        print(f"  {cls:<16} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  (n={m['support']})")
    print(SEP)


# ════════════════════════════════════════════════════════════════════════════
#  Pipeline cũ TEXT: PhoBERT fine-tuned direct inference
# ════════════════════════════════════════════════════════════════════════════

def evaluate_old_text_pipeline(
    texts: list[str],
    labels: torch.Tensor,
    class_names: list[str],
    phobert_dir: str,
    batch_size: int = 16,
) -> dict:
    """Run original fine-tuned PhoBERT directly on test texts."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading fine-tuned PhoBERT from: %s", phobert_dir)

    tokenizer = AutoTokenizer.from_pretrained(phobert_dir)
    model = AutoModelForSequenceClassification.from_pretrained(phobert_dir)
    model = model.to(device)
    model.eval()

    id2label = model.config.id2label if hasattr(model.config, "id2label") else None
    logger.info("Model id2label: %s", id2label)

    all_preds = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        cleaned = [" ".join(str(t).strip().split()) for t in batch_texts]

        enc = tokenizer(
            cleaned, padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=1).cpu()

        all_preds.append(preds)

        if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(texts):
            logger.info("  Old text pipeline: %d/%d", min(i + batch_size, len(texts)), len(texts))

    all_preds = torch.cat(all_preds)

    # Remap if model uses different label ordering
    if id2label:
        model_labels = [id2label[k].lower() for k in sorted(id2label.keys())]
        class_lower = [c.lower() for c in class_names]
        remap = {}
        for model_idx, mlabel in enumerate(model_labels):
            for our_idx, clabel in enumerate(class_lower):
                if mlabel in clabel or clabel in mlabel:
                    remap[model_idx] = our_idx
                    break
        if len(remap) == len(model_labels):
            all_preds = torch.tensor([remap.get(p.item(), p.item()) for p in all_preds])
            logger.info("Label remapping applied: %s", remap)

    return compute_metrics(all_preds, labels, class_names)


# ════════════════════════════════════════════════════════════════════════════
#  Pipeline mới TEXT: Embedding → Denoiser → MLP head
# ════════════════════════════════════════════════════════════════════════════

def evaluate_new_text_pipeline(
    text_embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    head_path: str,
) -> dict:
    """Run raw PhoBERT embeddings → MLP head (no denoiser for text).
    Bypasses denoiser to avoid distribution shift from dummy image input.
    """
    head_ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
    input_dim = head_ckpt["input_dim"]
    num_classes = head_ckpt["num_classes"]
    head_class_names = head_ckpt["class_names"]

    head = ClassificationHead(input_dim=input_dim, num_classes=num_classes)
    head.load_state_dict(head_ckpt["model_state_dict"])
    head.eval()

    logger.info("Loaded text head: input_dim=%d, classes=%s", input_dim, head_class_names)

    with torch.no_grad():
        logits = head(text_embeddings)
        preds = logits.argmax(dim=1)

    if head_class_names != class_names:
        logger.warning("Class name mismatch! head=%s, ours=%s", head_class_names, class_names)

    return compute_metrics(preds, labels, head_class_names)


# ════════════════════════════════════════════════════════════════════════════
#  Pipeline cũ ẢNH: ResNet50 binary (intact vs defect)
# ════════════════════════════════════════════════════════════════════════════

def evaluate_old_image_pipeline(
    image_paths: list[str],
    labels_4class: torch.Tensor,
    defect_classes: list[str],
    resnet_path: str,
    batch_size: int = 16,
) -> tuple[dict, list[str], torch.Tensor]:
    """Run original binary ResNet50 (no-defect vs defect).
    Maps 4-class ground-truth to binary for fair comparison.
    Returns (metrics, binary_class_names, binary_labels).
    """
    from torchvision import transforms, models
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading ResNet50 from: %s", resnet_path)

    checkpoint = torch.load(resnet_path, map_location=device, weights_only=False)
    model = models.resnet50(weights=None)

    ckpt_class_names = checkpoint.get("class_names", ["no-defect", "defect"])
    logger.info("ResNet50 trained classes: %s", ckpt_class_names)

    # Set fc layer size BEFORE loading state_dict
    model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt_class_names))

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    # Convert 4-class GT to binary: intact=0 (no-defect), others=1 (defect)
    binary_classes = ["no-defect", "defect"]
    intact_idx = defect_classes.index("intact")
    labels_binary = (labels_4class != intact_idx).long()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_preds = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []
        for p in batch_paths:
            try:
                img = __import__("PIL.Image", fromlist=["Image"]).Image.open(p).convert("RGB")
                batch_tensors.append(transform(img))
            except Exception:
                batch_tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            logits = model(batch)
            preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)

        if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(image_paths):
            logger.info("  Old image pipeline: %d/%d", min(i + batch_size, len(image_paths)), len(image_paths))

    all_preds = torch.cat(all_preds)
    metrics = compute_metrics(all_preds, labels_binary, binary_classes)
    return metrics, binary_classes, labels_binary


# ════════════════════════════════════════════════════════════════════════════
#  Pipeline mới ẢNH: Embedding → Denoiser → 4-class MLP head
# ════════════════════════════════════════════════════════════════════════════

def evaluate_new_image_pipeline(
    image_embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    denoiser: FeatureDenoiser,
    head_path: str,
) -> dict:
    """Run denoiser → binary MLP head on pre-extracted test image embeddings."""
    head_ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
    input_dim = head_ckpt["input_dim"]
    num_classes = head_ckpt["num_classes"]
    head_class_names = head_ckpt["class_names"]

    head = ClassificationHead(input_dim=input_dim, num_classes=num_classes)
    head.load_state_dict(head_ckpt["model_state_dict"])
    head.eval()

    denoiser.eval()
    with torch.no_grad():
        _, image_clean = denoiser(
            torch.zeros(len(image_embeddings), denoiser.text_dim),
            image_embeddings,
        )
        logits = head(image_clean)
        preds = logits.argmax(dim=1)

    return compute_metrics(preds, labels, head_class_names)


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Compare old vs new pipeline on test set")
    parser.add_argument("--phobert-dir", default=r"C:\Users\ADMIN\Downloads\phobert")
    parser.add_argument("--resnet-path", default=r"d:/ecommerce-backup/resnet50_improve_with_weights/ai_engine/models/resnet50_defect_gpu_best.pth")
    parser.add_argument("--denoiser-path", default="ai_engine/models/feature_denoiser.pt")
    parser.add_argument("--text-head-path", default="artifacts/models/text_sentiment_head.pt")
    parser.add_argument("--image-head-path", default="artifacts/models/image_defect_head.pt")
    parser.add_argument("--paired-csv", default="data/processed/paired_text_image.csv")
    parser.add_argument("--text-embeddings", default="data/processed/paired_text_embeddings.pt")
    parser.add_argument("--image-embeddings", default="data/processed/paired_image_embeddings.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────
    df = pd.read_csv(ROOT / args.paired_csv, encoding="utf-8-sig")
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    test_mask = (df["split"] == "test").values
    logger.info("Test set size: %d", len(test_df))

    sentiment_classes = sorted(df["sentiment_label"].dropna().unique().tolist())
    sentiment_map = {c: i for i, c in enumerate(sentiment_classes)}
    test_sentiment_labels = torch.tensor(
        test_df["sentiment_label"].map(sentiment_map).values, dtype=torch.long
    )

    defect_classes = sorted(df["image_label"].dropna().unique().tolist())
    defect_map = {c: i for i, c in enumerate(defect_classes)}
    test_defect_labels = torch.tensor(
        test_df["image_label"].map(defect_map).values, dtype=torch.long
    )

    all_text_emb = torch.load(ROOT / args.text_embeddings, weights_only=False)["embeddings"]
    all_image_emb = torch.load(ROOT / args.image_embeddings, weights_only=False)["embeddings"]
    test_text_emb = all_text_emb[test_mask]
    test_image_emb = all_image_emb[test_mask]

    logger.info("Sentiment classes: %s", sentiment_classes)
    logger.info("Defect classes: %s", defect_classes)

    # ── Load denoiser ──────────────────────────────────────────────────────
    ckpt = torch.load(ROOT / args.denoiser_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    denoiser = FeatureDenoiser(
        text_dim=config["text_dim"],
        image_dim=config["image_dim"],
        hidden_dim=config["hidden_dim"],
        noise_steps=config["noise_steps"],
        noise_schedule=config.get("noise_schedule", "cosine"),
    )
    denoiser.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded denoiser (hidden_dim=%d)", config["hidden_dim"])

    # ════════════════════════════════════════════════════════════════════════
    #  TEXT SENTIMENT COMPARISON
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("  TEXT SENTIMENT COMPARISON")
    print("█" * 60)

    logger.info("Evaluating OLD text pipeline...")
    old_text = evaluate_old_text_pipeline(
        texts=test_df["text"].tolist(),
        labels=test_sentiment_labels,
        class_names=sentiment_classes,
        phobert_dir=args.phobert_dir,
        batch_size=args.batch_size,
    )
    print_results("OLD: PhoBERT fine-tuned (direct)", old_text, sentiment_classes)

    logger.info("Evaluating NEW text pipeline (raw embedding → MLP head)...")
    new_text = evaluate_new_text_pipeline(
        text_embeddings=test_text_emb,
        labels=test_sentiment_labels,
        class_names=sentiment_classes,
        head_path=str(ROOT / args.text_head_path),
    )
    print_results("NEW: PhoBERT embedding → MLP head (no denoiser)", new_text, sentiment_classes)

    # ════════════════════════════════════════════════════════════════════════
    #  IMAGE DEFECT COMPARISON
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("  IMAGE DEFECT COMPARISON")
    print("█" * 60)

    logger.info("Evaluating OLD image pipeline (binary)...")
    old_image, binary_classes, labels_binary = evaluate_old_image_pipeline(
        image_paths=test_df["image_path"].tolist(),
        labels_4class=test_defect_labels,
        defect_classes=defect_classes,
        resnet_path=args.resnet_path,
        batch_size=args.batch_size,
    )
    print_results("OLD: ResNet50 fine-tuned (binary)", old_image, binary_classes)

    logger.info("Evaluating NEW image pipeline (binary)...")
    new_image = evaluate_new_image_pipeline(
        image_embeddings=test_image_emb,
        labels=labels_binary,
        class_names=binary_classes,
        denoiser=denoiser,
        head_path=str(ROOT / args.image_head_path),
    )
    print_results("NEW: ResNet50 + Denoiser + MLP head (binary)", new_image, binary_classes)

    # ════════════════════════════════════════════════════════════════════════
    #  TỔNG KẾT SO SÁNH
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  TỔNG KẾT SO SÁNH")
    print("=" * 65)
    print(f"  {'Task':<22} {'Old Acc':>8} {'New Acc':>8} {'Delta':>8} {'Old F1':>8} {'New F1':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    acc_d = new_text["accuracy"] - old_text["accuracy"]
    f1_d = new_text["macro_f1"] - old_text["macro_f1"]
    print(f"  {'Text Sentiment':<22} {old_text['accuracy']:>8.4f} {new_text['accuracy']:>8.4f} {acc_d:>+8.4f} {old_text['macro_f1']:>8.4f} {new_text['macro_f1']:>8.4f}")

    acc_d_img = new_image["accuracy"] - old_image["accuracy"]
    f1_d_img = new_image["macro_f1"] - old_image["macro_f1"]
    print(f"  {'Image Defect (binary)':<22} {old_image['accuracy']:>8.4f} {new_image['accuracy']:>8.4f} {acc_d_img:>+8.4f} {old_image['macro_f1']:>8.4f} {new_image['macro_f1']:>8.4f}")

    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
