"""
test_clip_quick.py
==================
Quick test for CLIP binary classifier (product vs irrelevant) with a few images per class.
Used to verify CLIP performs correctly BEFORE sending it to other members to train the full model.

Pipeline:
    CLIP (binary) → [product] → ResNet50 (defect/no-defect)
                   → [irrelevant] → discard

    wrong_item: handled at the text review layer (not CLIP's job)

Usage (from project root):
    python scripts/test_clip_quick.py
    python scripts/test_clip_quick.py --per-class 10
"""

from __future__ import annotations

import argparse
import io
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "labeled" / "test"
sys.path.insert(0, str(ROOT))

CLASSES = ["intact", "damaged", "wrong_item", "irrelevant"]

# CLIP binary: product images (product) vs clutter/non-product images (irrelevant)
# intact     → product   (normal product → ResNet50 classifies as intact)
# damaged    → product   (damaged box → ResNet50 classifies as defect)
# wrong_item → product   (still a product image, just wrong type → handled by text review)
# irrelevant → irrelevant (selfie, food, meme → discard)
EXPECTED_CLIP_LABEL = {
    "intact":     "product",
    "damaged":    "product",
    "wrong_item": "product",      # wrong_item is still a product image, CLIP keeps it
    "irrelevant": "irrelevant",
}


def collect_samples(per_class: int) -> list[dict]:
    """Collect random image samples from each class."""
    data = []
    for cls in CLASSES:
        cls_dir = TEST_DIR / cls
        if not cls_dir.exists():
            print(f"  ⚠️  Directory does not exist: {cls_dir}")
            continue
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        if not imgs:
            print(f"  ⚠️  No images in {cls_dir}")
            continue

        random.seed(42)
        sampled = random.sample(imgs, min(per_class, len(imgs)))
        for img in sampled:
            data.append({"path": str(img), "true_label": cls})
    return data


def run_test(per_class: int = 5) -> None:
    """Run CLIP binary test on a small sample subset."""

    print(f"\n{'='*70}")
    print("  🧪 CLIP BINARY QUICK TEST (product vs irrelevant)")
    print(f"{'='*70}")
    print(f"  Test dir   : {TEST_DIR}")
    print(f"  Per class  : {per_class} images")
    print(f"  Pipeline   : CLIP → [product] → ResNet50")
    print(f"               CLIP → [irrelevant] → discard")
    print(f"               wrong_item → handled at text review layer")

    data = collect_samples(per_class)
    total = len(data)
    print(f"  Total samples: {total}")

    # Distribution
    class_dist = defaultdict(int)
    for d in data:
        class_dist[d["true_label"]] += 1
    print(f"\n  Distribution:")
    for cls in CLASSES:
        expected = EXPECTED_CLIP_LABEL[cls]
        print(f"    {cls:<12} → expect '{expected}'  ({class_dist[cls]} images)")

    # Load model
    print(f"\n  Loading CLIP model...")
    t0 = time.perf_counter()
    from ai_engine.image_processing.zero_shot_clip import classify_image
    # Warm up model load
    load_time = time.perf_counter() - t0
    print(f"  Import done in {load_time:.1f}s")

    # Run inference
    print(f"\n{'='*70}")
    correct = 0
    wrong_examples = []
    per_class_stats = defaultdict(lambda: {"correct": 0, "total": 0, "preds": defaultdict(int)})

    for i, item in enumerate(data):
        t_start = time.perf_counter()
        result = classify_image(item["path"])
        inference_ms = (time.perf_counter() - t_start) * 1000

        true_cls = item["true_label"]
        expected_clip = EXPECTED_CLIP_LABEL[true_cls]

        if result is None:
            pred_label = "ERROR"
            pred_conf = 0.0
            probs_str = "N/A"
            is_correct = False
        else:
            pred_label = result["label"]
            pred_conf = result["confidence"]
            probs = result["probs"]
            probs_str = f"product={probs['product']:.2f}  irrel={probs['irrelevant']:.2f}"
            is_correct = (pred_label == expected_clip)

        if is_correct:
            correct += 1
            icon = "✅"
        else:
            icon = "❌"
            wrong_examples.append({
                "path": Path(item["path"]).name,
                "true": true_cls,
                "expected_clip": expected_clip,
                "predicted": pred_label,
                "probs": probs_str,
            })

        per_class_stats[true_cls]["total"] += 1
        per_class_stats[true_cls]["preds"][pred_label] += 1
        if is_correct:
            per_class_stats[true_cls]["correct"] += 1

        fname = Path(item["path"]).name[:35]
        print(
            f"  {icon} [{true_cls:<12}] → CLIP: {pred_label:<12} "
            f"(conf={pred_conf:.2f})  {probs_str}  "
            f"({inference_ms:.0f}ms)  {fname}"
        )

    # ── Summary ──
    acc = correct / total if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"  📊 SUMMARY RESULTS")
    print(f"{'='*70}")
    print(f"  Overall Accuracy: {correct}/{total} = {acc:.1%}")

    print(f"\n  ── Per-class Accuracy ──")
    for cls in CLASSES:
        stats = per_class_stats[cls]
        cls_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        status = "✅" if cls_acc >= 0.60 else "⚠️" if cls_acc >= 0.40 else "❌"
        preds_detail = ", ".join(f"{k}={v}" for k, v in stats["preds"].items())
        expected = EXPECTED_CLIP_LABEL[cls]
        print(
            f"  {status} {cls:<12} → expect '{expected:<12}' : "
            f"{stats['correct']}/{stats['total']} ({cls_acc:.0%})  "
            f"[{preds_detail}]"
        )

    if wrong_examples:
        print(f"\n  ── Error Details ({len(wrong_examples)} incorrect) ──")
        for ex in wrong_examples[:15]:
            print(
                f"  ❌ {ex['path'][:35]:<35}  "
                f"true={ex['true']:<12}  expected={ex['expected_clip']:<12}  "
                f"got={ex['predicted']:<12}  {ex['probs']}"
            )

    # ── Role Evaluation in Pipeline ──
    print(f"\n{'='*70}")
    print(f"  🔍 ROLE EVALUATION IN PIPELINE")
    print(f"{'='*70}")

    # 1. Keep product images for ResNet50 (intact + damaged + wrong_item → product)
    product_classes = ["intact", "damaged", "wrong_item"]
    product_keep = sum(per_class_stats[c]["correct"] for c in product_classes)
    product_total = sum(per_class_stats[c]["total"] for c in product_classes)
    product_recall = product_keep / product_total if product_total > 0 else 0
    product_lost = product_total - product_keep

    print(f"\n  1. Keep product images for ResNet50 (intact+damaged+wrong_item → 'product'):")
    print(f"     Recall = {product_recall:.0%} ({product_keep}/{product_total} product images kept)")
    if product_recall >= 0.85:
        print(f"     ✅ Excellent! CLIP keeps almost all product images")
    elif product_recall >= 0.70:
        print(f"     ✅ Good. CLIP keeps most product images for ResNet50")
    elif product_recall >= 0.50:
        print(f"     ⚠️  Average. CLIP misclassified and discarded {product_lost} product images")
    else:
        print(f"     ❌ Poor. CLIP misclassified and discarded too many product images ({product_lost}/{product_total})")

    # 2. Filter irrelevant images
    irr_stats = per_class_stats["irrelevant"]
    irr_recall = irr_stats["correct"] / irr_stats["total"] if irr_stats["total"] > 0 else 0

    print(f"\n  2. Filter irrelevant images:")
    print(f"     Recall = {irr_recall:.0%} ({irr_stats['correct']}/{irr_stats['total']} irrelevant images correctly filtered)")
    if irr_recall >= 0.70:
        print(f"     ✅ CLIP filters irrelevant images well")
    elif irr_recall >= 0.50:
        print(f"     ⚠️  CLIP caught half of irrelevant images — prompts might need refinement")
    else:
        print(f"     ❌ CLIP missed too many irrelevant images")

    # 3. False alarm rate (wrongly filtering good images)
    false_alarms = product_total - product_keep
    false_alarm_rate = false_alarms / product_total if product_total > 0 else 0
    print(f"\n  3. False alarm (wrongly filtering product images):")
    print(f"     Rate = {false_alarm_rate:.0%} ({false_alarms}/{product_total} product images wrongly discarded)")
    if false_alarm_rate <= 0.10:
        print(f"     ✅ Very low false alarm rate")
    elif false_alarm_rate <= 0.20:
        print(f"     ⚠️  Acceptable but could be improved")
    else:
        print(f"     ❌ Discarded too many product images")

    # ── Conclusion ──
    print(f"\n{'='*70}")
    if product_recall >= 0.70 and irr_recall >= 0.50:
        print(f"  ✅ CLIP PERFORMS WELL — Ready to be used as a pre-filter for ResNet50")
        print(f"     Pipeline: CLIP (filter irrelevant) → ResNet50 (defect/no-defect)")
    elif product_recall >= 0.50:
        print(f"  ⚠️  CLIP NEEDS IMPROVEMENT — Functional but not accurate enough")
        print(f"     → Try a larger model: clip-vit-large-patch14")
        print(f"     → Or refine prompts for the dataset")
    else:
        print(f"  ❌ CLIP NOT EFFECTIVE — Design needs to be re-evaluated")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Quick test CLIP binary classifier"
    )
    parser.add_argument(
        "--per-class", type=int, default=5,
        help="Number of test images per class (default: 5, total ~20 images)"
    )
    args = parser.parse_args()
    run_test(args.per_class)


if __name__ == "__main__":
    main()
