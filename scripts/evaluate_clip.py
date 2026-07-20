"""
evaluate_clip.py
================
Evaluate CLIP zero-shot on the test split of the labeled dataset.
Read the folder structure test/{class_name}/*.jpg and compare
with predictions from CLIP.

Usage (from project root):
    python scripts/evaluate_clip.py
    python scripts/evaluate_clip.py --sample 200   # only test 200 images (faster)
    python scripts/evaluate_clip.py --threshold 0.45
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import io
from collections import defaultdict
from pathlib import Path

# Fix UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "labeled" / "test"

# Add ai_engine to path
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ai_engine"))

CLASSES = ["intact", "damaged", "wrong_item", "irrelevant"]


def collect_test_images(sample: int | None = None) -> list[dict]:
    """Collect all images from test/ with ground-truth label from folder name."""
    data = []
    for cls in CLASSES:
        cls_dir = TEST_DIR / cls
        if not cls_dir.exists():
            print(f"  [WARN] Folder không tồn tại: {cls_dir}")
            continue
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        for img in imgs:
            data.append({"path": str(img), "true_label": cls})

    if sample and sample < len(data):
        import random
        random.seed(42)
        random.shuffle(data)
        data = data[:sample]

    return data


def clip_predict(image_path: str, threshold: float) -> str:
    """
    Use CLIP to classify an image into one of 4 classes:
    intact, damaged, wrong_item, irrelevant.

    Logic:
    1. CLIP computes P(irrelevant) vs P(relevant)
    2. If P(irrelevant) >= threshold → 'irrelevant'
    3. If relevant → return 'relevant' (CLIP does not distinguish damaged/wrong_item/intact)
    """
    from ai_engine.image_processing.zero_shot_clip import classify_image_detail
    result = classify_image_detail(image_path)
    if result is None:
        return "intact"  # safe default

    if result["irrelevant_score"] >= threshold:
        return "irrelevant"
    else:
        # CLIP only distinguishes between relevant and irrelevant
        # Cannot predict damaged/wrong_item/intact from a 2-class CLIP model
        return "relevant"


def run_evaluation(sample: int | None, threshold: float) -> dict:
    """Run the entire evaluation and compute metrics."""

    print(f"\n{'='*60}")
    print("  CLIP ZERO-SHOT EVALUATION")
    print(f"{'='*60}")
    print(f"  Test dir : {TEST_DIR}")
    print(f"  Threshold: {threshold}")
    print(f"  Sample   : {sample or 'ALL'}")

    # Collect data
    data = collect_test_images(sample)
    print(f"\n  Tổng ảnh sẽ test: {len(data)}")

    # Count class distribution
    class_dist = defaultdict(int)
    for d in data:
        class_dist[d["true_label"]] += 1
    print("\n  Phân bố class:")
    for cls, cnt in sorted(class_dist.items()):
        print(f"    {cls:<15} {cnt:>5} ảnh")

    # Run inference
    print("\n  Đang chạy CLIP inference...")
    t0 = time.perf_counter()

    # Metrics for BINARY task: relevant vs irrelevant
    binary_tp = binary_fp = binary_tn = binary_fn = 0

    # Confusion matrix 4-class (but CLIP only provides binary output)
    # relevant_classes = intact + damaged + wrong_item
    # irrelevant_classes = irrelevant
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    results = []
    for i, item in enumerate(data):
        pred = clip_predict(item["path"], threshold)
        true = item["true_label"]

        per_class_total[true] += 1

        # Binary: relevant vs irrelevant
        is_irrelevant_true = (true == "irrelevant")
        is_irrelevant_pred = (pred == "irrelevant")

        if is_irrelevant_true and is_irrelevant_pred:
            binary_tp += 1
            per_class_correct[true] += 1
        elif not is_irrelevant_true and not is_irrelevant_pred:
            binary_tn += 1
            per_class_correct[true] += 1
        elif not is_irrelevant_true and is_irrelevant_pred:
            binary_fp += 1  # False alarm: misclassified product image
        elif is_irrelevant_true and not is_irrelevant_pred:
            binary_fn += 1  # Miss: missed irrelevant image

        results.append({
            "path": item["path"],
            "true": true,
            "pred": pred,
            "correct": (is_irrelevant_true == is_irrelevant_pred),
        })

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            speed = (i + 1) / elapsed
            remaining = (len(data) - i - 1) / speed
            print(f"    [{i+1}/{len(data)}] {speed:.1f} img/s | ETA: {remaining:.0f}s")

    elapsed = time.perf_counter() - t0

    # ── Compute metrics ───────────────────────────────────────────────────────
    total = len(data)
    overall_acc = (binary_tp + binary_tn) / total if total > 0 else 0

    precision_irr = binary_tp / (binary_tp + binary_fp) if (binary_tp + binary_fp) > 0 else 0
    recall_irr = binary_tp / (binary_tp + binary_fn) if (binary_tp + binary_fn) > 0 else 0
    f1_irr = 2 * precision_irr * recall_irr / (precision_irr + recall_irr) if (precision_irr + recall_irr) > 0 else 0

    precision_rel = binary_tn / (binary_tn + binary_fn) if (binary_tn + binary_fn) > 0 else 0
    recall_rel = binary_tn / (binary_tn + binary_fp) if (binary_tn + binary_fp) > 0 else 0
    f1_rel = 2 * precision_rel * recall_rel / (precision_rel + recall_rel) if (precision_rel + recall_rel) > 0 else 0

    f1_macro = (f1_irr + f1_rel) / 2

    # False Positive Rate: CLIP misclassifies valid images (intact/damaged/wrong_item)
    n_relevant_total = binary_tp + binary_fn + binary_tn + binary_fp - class_dist.get("irrelevant", 0)
    false_alarm_rate = binary_fp / (class_dist.get("intact", 0) + class_dist.get("damaged", 0) + class_dist.get("wrong_item", 0)) if (class_dist.get("intact", 0) + class_dist.get("damaged", 0) + class_dist.get("wrong_item", 0)) > 0 else 0

    # Per-class accuracy
    per_class_acc = {}
    for cls in CLASSES:
        total_cls = per_class_total[cls]
        correct_cls = per_class_correct[cls]
        per_class_acc[cls] = correct_cls / total_cls if total_cls > 0 else 0.0

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  KẾT QUẢ EVALUATION")
    print(f"{'='*60}")
    print(f"  Tổng ảnh test    : {total}")
    print(f"  Thời gian        : {elapsed:.1f}s ({total/elapsed:.1f} img/s)")
    print(f"  Threshold        : {threshold}")

    print(f"\n  ── BINARY METRICS (relevant vs irrelevant) ──")
    print(f"  Overall Accuracy : {overall_acc:.4f} ({overall_acc*100:.1f}%)")
    print(f"  F1 Macro         : {f1_macro:.4f}")

    print(f"\n  [Class: irrelevant]")
    print(f"    Precision : {precision_irr:.4f}  (trong số predicted irrelevant, bao nhiêu đúng)")
    print(f"    Recall    : {recall_irr:.4f}  (trong số actual irrelevant, bao nhiêu bắt được)")
    print(f"    F1        : {f1_irr:.4f}")

    print(f"\n  [Class: relevant (intact+damaged+wrong_item)]")
    print(f"    Precision : {precision_rel:.4f}")
    print(f"    Recall    : {recall_rel:.4f}  (trong số ảnh tốt, bao nhiêu giữ lại được)")
    print(f"    F1        : {f1_rel:.4f}")

    print(f"\n  ── LỖI QUAN TRỌNG ──")
    print(f"  False Positives (loại nhầm ảnh tốt) : {binary_fp}  ({false_alarm_rate*100:.1f}% của ảnh relevant)")
    print(f"  False Negatives (bỏ sót ảnh rác)    : {binary_fn}  ({(1-recall_irr)*100:.1f}% irrelevant bị miss)")

    print(f"\n  ── PER-CLASS ACCURACY ──")
    for cls in CLASSES:
        acc = per_class_acc[cls]
        total_cls = per_class_total[cls]
        status = "✅" if acc >= 0.75 else "⚠️" if acc >= 0.50 else "❌"
        print(f"  {cls:<15} : {acc:.4f} ({acc*100:.1f}%)  [{per_class_correct[cls]}/{total_cls}]  {status}")

    print(f"\n  ── CONFUSION MATRIX (binary) ──")
    print(f"                    Pred: relevant  Pred: irrelevant")
    print(f"  True: relevant        {binary_tn:<12}   {binary_fp}")
    print(f"  True: irrelevant      {binary_fn:<12}   {binary_tp}")

    # ── Error Analysis ───────────────────────────────────────────────────────
    fp_by_class = defaultdict(int)
    fn_by_class = defaultdict(int)
    for r in results:
        if not r["correct"]:
            if r["true"] != "irrelevant" and r["pred"] == "irrelevant":
                fp_by_class[r["true"]] += 1
            elif r["true"] == "irrelevant" and r["pred"] != "irrelevant":
                fn_by_class[r["true"]] += 1

    print(f"\n  ── FALSE POSITIVES (loại nhầm ảnh tốt) theo class ──")
    for cls, cnt in fp_by_class.items():
        pct = cnt / per_class_total[cls] * 100 if per_class_total[cls] > 0 else 0
        print(f"  {cls:<15}: {cnt} ảnh bị loại nhầm ({pct:.1f}%)")

    # ── Overall Evaluation ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ĐÁNH GIÁ TỔNG THỂ")
    print(f"{'='*60}")

    issues = []

    if recall_irr < 0.70:
        issues.append(f"❌ CRITICAL: Recall irrelevant={recall_irr:.2f} < 0.70 — CLIP bỏ sót quá nhiều ảnh rác")
    elif recall_irr < 0.80:
        issues.append(f"⚠️  Recall irrelevant={recall_irr:.2f} — cần cải thiện prompt để bắt nhiều ảnh rác hơn")
    else:
        print(f"  ✅ Recall irrelevant: {recall_irr:.2f} — CLIP bắt tốt ảnh không liên quan")

    if false_alarm_rate > 0.15:
        issues.append(f"❌ CRITICAL: False alarm rate={false_alarm_rate:.2f} — CLIP loại nhầm >15% ảnh tốt")
    elif false_alarm_rate > 0.08:
        issues.append(f"⚠️  False alarm rate={false_alarm_rate:.2f} — CLIP loại nhầm 8-15% ảnh tốt, cần hạ threshold")
    else:
        print(f"  ✅ False alarm rate: {false_alarm_rate:.2f} — ít loại nhầm ảnh tốt")

    if per_class_acc["damaged"] < 0.70:
        issues.append(f"⚠️  damaged accuracy={per_class_acc['damaged']:.2f} — CLIP hay loại nhầm ảnh hộp bị móp")
    if per_class_acc["wrong_item"] < 0.70:
        issues.append(f"⚠️  wrong_item accuracy={per_class_acc['wrong_item']:.2f} — CLIP hay loại nhầm ảnh sai hàng")

    if issues:
        print("\n  VẤN ĐỀ PHÁT HIỆN:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  ✅ CLIP hoạt động ổn ở vai trò pre-filter")

    # Recommended threshold
    print(f"\n  KHUYẾN NGHỊ:")
    if false_alarm_rate > 0.10:
        print(f"  → Giảm threshold (thử 0.45 hoặc 0.40) để giảm false positive")
    if recall_irr < 0.75:
        print(f"  → Tăng threshold (thử 0.60) hoặc cải thiện IRRELEVANT_PROMPTS")
    print(f"  → CLIP hiện tại chỉ làm được BINARY (relevant vs irrelevant)")
    print(f"  → Vẫn cần train model riêng để phân biệt intact/damaged/wrong_item")

    # ── Save results ──────────────────────────────────────────────────────────
    report = {
        "threshold": threshold,
        "total": total,
        "elapsed_s": round(elapsed, 2),
        "overall_accuracy": round(overall_acc, 4),
        "f1_macro": round(f1_macro, 4),
        "irrelevant_precision": round(precision_irr, 4),
        "irrelevant_recall": round(recall_irr, 4),
        "irrelevant_f1": round(f1_irr, 4),
        "relevant_precision": round(precision_rel, 4),
        "relevant_recall": round(recall_rel, 4),
        "relevant_f1": round(f1_rel, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
        "binary_matrix": {
            "TP": binary_tp, "FP": binary_fp,
            "TN": binary_tn, "FN": binary_fn
        },
        "per_class_accuracy": {k: round(v, 4) for k, v in per_class_acc.items()},
        "per_class_total": dict(per_class_total),
        "fp_by_class": dict(fp_by_class),
        "issues": issues,
    }

    report_path = ROOT / "reports" / "clip_evaluation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved → {report_path.relative_to(ROOT)}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Đánh giá CLIP trên tập test")
    parser.add_argument("--sample", type=int, default=None,
                        help="Số ảnh mẫu để test (mặc định: tất cả ~4162 ảnh)")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Ngưỡng P(irrelevant) của CLIP (mặc định: 0.55)")
    args = parser.parse_args()

    run_evaluation(args.sample, args.threshold)


if __name__ == "__main__":
    main()
