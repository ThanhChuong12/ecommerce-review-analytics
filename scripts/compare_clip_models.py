"""
compare_clip_models.py
======================
Compare CLIP model variants with new prompt sets.
Run on 50 images/class to determine the best model + prompt.

Usage (from project root):
    python scripts/compare_clip_models.py
    python scripts/compare_clip_models.py --per-class 20
    python scripts/compare_clip_models.py --models vit-b-32 vit-b-16
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
EXPECTED_CLIP_LABEL = {
    "intact":     "product",
    "damaged":    "product",
    "wrong_item": "product",
    "irrelevant": "irrelevant",
}

MODEL_MAP = {
    "vit-b-32": "openai/clip-vit-base-patch32",
    "vit-b-16": "openai/clip-vit-base-patch16",
    "vit-l-14": "openai/clip-vit-large-patch14",
}


def collect_samples(per_class: int) -> list[dict]:
    data = []
    for cls in CLASSES:
        cls_dir = TEST_DIR / cls
        if not cls_dir.exists():
            continue
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        if not imgs:
            continue
        random.seed(42)
        sampled = random.sample(imgs, min(per_class, len(imgs)))
        for img in sampled:
            data.append({"path": str(img), "true_label": cls})
    return data


def evaluate_model(model_name: str, data: list) -> dict:
    """Evaluate 1 model on data, return metrics."""
    from ai_engine.image_processing.zero_shot_clip import classify_image, reload_model

    print(f"\n  Loading {model_name}...")
    t0 = time.perf_counter()
    reload_model(model_name)
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s")

    correct = 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    total_ms = 0

    for item in data:
        t_start = time.perf_counter()
        result = classify_image(item["path"])
        ms = (time.perf_counter() - t_start) * 1000
        total_ms += ms

        true_cls = item["true_label"]
        expected = EXPECTED_CLIP_LABEL[true_cls]

        per_class[true_cls]["total"] += 1

        if result and result["label"] == expected:
            correct += 1
            per_class[true_cls]["correct"] += 1

    total = len(data)
    avg_ms = total_ms / total if total > 0 else 0

    # Compute important metrics
    product_classes = ["intact", "damaged", "wrong_item"]
    product_keep = sum(per_class[c]["correct"] for c in product_classes)
    product_total = sum(per_class[c]["total"] for c in product_classes)
    product_recall = product_keep / product_total if product_total > 0 else 0

    irr = per_class["irrelevant"]
    irr_recall = irr["correct"] / irr["total"] if irr["total"] > 0 else 0

    false_alarm = product_total - product_keep
    false_alarm_rate = false_alarm / product_total if product_total > 0 else 0

    return {
        "model": model_name,
        "accuracy": correct / total if total > 0 else 0,
        "product_recall": product_recall,
        "irrelevant_recall": irr_recall,
        "false_alarm_rate": false_alarm_rate,
        "avg_ms": avg_ms,
        "load_time": load_time,
        "per_class": {
            cls: {
                "acc": per_class[cls]["correct"] / per_class[cls]["total"]
                if per_class[cls]["total"] > 0 else 0,
                "correct": per_class[cls]["correct"],
                "total": per_class[cls]["total"],
            }
            for cls in CLASSES
        },
        "total_time_27k": avg_ms * 27743 / 1000 / 60,  # minutes for full dataset
    }


def main():
    parser = argparse.ArgumentParser(description="Compare CLIP model variants")
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--models", nargs="+", default=["vit-b-32", "vit-b-16"],
                        choices=list(MODEL_MAP.keys()))
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"  🔬 CLIP MODEL COMPARISON (improved prompts)")
    print(f"{'='*80}")
    print(f"  Per class  : {args.per_class}")
    print(f"  Models     : {args.models}")

    data = collect_samples(args.per_class)
    print(f"  Total imgs : {len(data)}")

    results = []
    for short_name in args.models:
        model_name = MODEL_MAP[short_name]
        print(f"\n{'─'*80}")
        print(f"  📦 Testing: {short_name} ({model_name})")
        print(f"{'─'*80}")

        r = evaluate_model(model_name, data)
        results.append(r)

        # Print results immediately
        print(f"\n  Results for {short_name}:")
        print(f"    Overall Accuracy   : {r['accuracy']:.1%}")
        print(f"    Product Recall     : {r['product_recall']:.1%}")
        print(f"    Irrelevant Recall  : {r['irrelevant_recall']:.1%}")
        print(f"    False Alarm Rate   : {r['false_alarm_rate']:.1%}")
        print(f"    Avg inference      : {r['avg_ms']:.0f}ms/img")
        print(f"    Est. 27K imgs      : {r['total_time_27k']:.0f} phút")
        for cls in CLASSES:
            c = r["per_class"][cls]
            print(f"    {cls:<12}: {c['correct']}/{c['total']} ({c['acc']:.0%})")

    # ── Comparison table ──
    print(f"\n{'='*80}")
    print(f"  📊 BẢNG SO SÁNH")
    print(f"{'='*80}")

    header = f"  {'Metric':<22}"
    for r in results:
        short = [k for k, v in MODEL_MAP.items() if v == r["model"]][0]
        header += f" | {short:>12}"
    print(header)
    print(f"  {'─'*22}" + "─┼──────────────" * len(results))

    metrics = [
        ("Overall Accuracy", "accuracy"),
        ("Product Recall", "product_recall"),
        ("Irrelevant Recall", "irrelevant_recall"),
        ("False Alarm Rate", "false_alarm_rate"),
    ]
    for label, key in metrics:
        line = f"  {label:<22}"
        best_val = max(r[key] for r in results) if key != "false_alarm_rate" else min(r[key] for r in results)
        for r in results:
            val = r[key]
            marker = " ✅" if val == best_val else "   "
            line += f" | {val:>9.1%}{marker}"
        print(line)

    # Speed
    line = f"  {'Avg ms/img':<22}"
    best_speed = min(r["avg_ms"] for r in results)
    for r in results:
        marker = " ✅" if r["avg_ms"] == best_speed else "   "
        line += f" | {r['avg_ms']:>8.0f}ms{marker}"
    print(line)

    line = f"  {'Est. 27K imgs (min)':<22}"
    for r in results:
        line += f" | {r['total_time_27k']:>8.0f}m    "
    print(line)

    # Per-class
    print(f"\n  Per-class detail:")
    for cls in CLASSES:
        line = f"  {cls:<22}"
        best_acc = max(r["per_class"][cls]["acc"] for r in results)
        for r in results:
            c = r["per_class"][cls]
            marker = " ✅" if c["acc"] == best_acc else "   "
            line += f" | {c['correct']:>3}/{c['total']:<3} ({c['acc']:.0%}){marker}"
        print(line)

    # ── Recommendation ──
    print(f"\n{'='*80}")
    print(f"  💡 RECOMMENDATION")
    print(f"{'='*80}")

    # Score: product_recall * 0.3 + irrelevant_recall * 0.4 + (1-false_alarm_rate) * 0.3
    for r in results:
        r["score"] = (
            r["product_recall"] * 0.3 +
            r["irrelevant_recall"] * 0.4 +
            (1 - r["false_alarm_rate"]) * 0.3
        )

    best = max(results, key=lambda r: r["score"])
    best_short = [k for k, v in MODEL_MAP.items() if v == best["model"]][0]

    print(f"\n  Best model: {best_short} ({best['model']})")
    print(f"  Score: {best['score']:.3f}")
    print(f"  Product Recall={best['product_recall']:.0%}, "
          f"Irrelevant Recall={best['irrelevant_recall']:.0%}, "
          f"FalseAlarm={best['false_alarm_rate']:.0%}")
    print(f"  Speed: {best['avg_ms']:.0f}ms/img → {best['total_time_27k']:.0f} phút cho 27K ảnh")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
