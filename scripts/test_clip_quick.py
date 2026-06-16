"""
test_clip_quick.py
==================
Test nhanh CLIP binary classifier (product vs irrelevant) với vài ảnh mỗi class.
Dùng để verify CLIP hoạt động đúng TRƯỚC KHI gửi thành viên khác train full model.

Pipeline:
    CLIP (binary) → [product] → ResNet50 (defect/no-defect)
                   → [irrelevant] → loại bỏ

    wrong_item: xử lý ở tầng text review (không phải việc của CLIP)

Usage (từ project root):
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

# CLIP binary: ảnh sản phẩm (product) vs ảnh rác (irrelevant)
# intact     → product   (sản phẩm nguyên → ResNet50 phán intact)
# damaged    → product   (hộp hỏng → ResNet50 phán defect)
# wrong_item → product   (vẫn là ảnh sản phẩm, chỉ sai loại → text xử lý)
# irrelevant → irrelevant (selfie, food, meme → loại bỏ)
EXPECTED_CLIP_LABEL = {
    "intact":     "product",
    "damaged":    "product",
    "wrong_item": "product",      # wrong_item vẫn là ảnh sản phẩm, CLIP giữ lại
    "irrelevant": "irrelevant",
}


def collect_samples(per_class: int) -> list[dict]:
    """Thu thập mẫu ảnh ngẫu nhiên từ mỗi class."""
    data = []
    for cls in CLASSES:
        cls_dir = TEST_DIR / cls
        if not cls_dir.exists():
            print(f"  ⚠️  Folder không tồn tại: {cls_dir}")
            continue
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        if not imgs:
            print(f"  ⚠️  Không có ảnh trong {cls_dir}")
            continue

        random.seed(42)
        sampled = random.sample(imgs, min(per_class, len(imgs)))
        for img in sampled:
            data.append({"path": str(img), "true_label": cls})
    return data


def run_test(per_class: int = 5) -> None:
    """Chạy test CLIP binary trên mẫu nhỏ."""

    print(f"\n{'='*70}")
    print("  🧪 CLIP BINARY QUICK TEST (product vs irrelevant)")
    print(f"{'='*70}")
    print(f"  Test dir   : {TEST_DIR}")
    print(f"  Per class  : {per_class} ảnh")
    print(f"  Pipeline   : CLIP → [product] → ResNet50")
    print(f"               CLIP → [irrelevant] → loại bỏ")
    print(f"               wrong_item → xử lý ở tầng text review")

    data = collect_samples(per_class)
    total = len(data)
    print(f"  Tổng mẫu   : {total}")

    # Phân bố
    class_dist = defaultdict(int)
    for d in data:
        class_dist[d["true_label"]] += 1
    print(f"\n  Phân bố:")
    for cls in CLASSES:
        expected = EXPECTED_CLIP_LABEL[cls]
        print(f"    {cls:<12} → expect '{expected}'  ({class_dist[cls]} ảnh)")

    # Load model
    print(f"\n  Loading CLIP model...")
    t0 = time.perf_counter()
    from ai_engine.image_processing.zero_shot_clip import classify_image
    # Warm up model load
    load_time = time.perf_counter() - t0
    print(f"  Import done in {load_time:.1f}s")

    # Chạy inference
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

    # ── Tổng kết ──
    acc = correct / total if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"  📊 KẾT QUẢ TỔNG HỢP")
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
        print(f"\n  ── Lỗi chi tiết ({len(wrong_examples)} sai) ──")
        for ex in wrong_examples[:15]:
            print(
                f"  ❌ {ex['path'][:35]:<35}  "
                f"true={ex['true']:<12}  expected={ex['expected_clip']:<12}  "
                f"got={ex['predicted']:<12}  {ex['probs']}"
            )

    # ── Đánh giá theo vai trò ──
    print(f"\n{'='*70}")
    print(f"  🔍 ĐÁNH GIÁ VAI TRÒ CLIP TRONG PIPELINE")
    print(f"{'='*70}")

    # 1. Giữ ảnh sản phẩm cho ResNet50 (intact + damaged + wrong_item → product)
    product_classes = ["intact", "damaged", "wrong_item"]
    product_keep = sum(per_class_stats[c]["correct"] for c in product_classes)
    product_total = sum(per_class_stats[c]["total"] for c in product_classes)
    product_recall = product_keep / product_total if product_total > 0 else 0
    product_lost = product_total - product_keep

    print(f"\n  1. Giữ ảnh sản phẩm cho ResNet50 (intact+damaged+wrong_item → 'product'):")
    print(f"     Recall = {product_recall:.0%} ({product_keep}/{product_total} ảnh sản phẩm được giữ)")
    if product_recall >= 0.85:
        print(f"     ✅ Xuất sắc! CLIP giữ lại gần như toàn bộ ảnh sản phẩm")
    elif product_recall >= 0.70:
        print(f"     ✅ Tốt. CLIP giữ lại phần lớn ảnh sản phẩm cho ResNet50")
    elif product_recall >= 0.50:
        print(f"     ⚠️  Trung bình. CLIP loại nhầm {product_lost} ảnh sản phẩm")
    else:
        print(f"     ❌ Kém. CLIP loại nhầm quá nhiều ảnh sản phẩm ({product_lost}/{product_total})")

    # 2. Lọc ảnh rác
    irr_stats = per_class_stats["irrelevant"]
    irr_recall = irr_stats["correct"] / irr_stats["total"] if irr_stats["total"] > 0 else 0

    print(f"\n  2. Lọc ảnh rác (irrelevant):")
    print(f"     Recall = {irr_recall:.0%} ({irr_stats['correct']}/{irr_stats['total']} ảnh rác bị loại đúng)")
    if irr_recall >= 0.70:
        print(f"     ✅ CLIP lọc tốt ảnh không liên quan")
    elif irr_recall >= 0.50:
        print(f"     ⚠️  CLIP bắt được một nửa ảnh rác — có thể cần cải thiện prompts")
    else:
        print(f"     ❌ CLIP bỏ sót quá nhiều ảnh rác")

    # 3. False alarm rate (loại nhầm ảnh tốt)
    false_alarms = product_total - product_keep
    false_alarm_rate = false_alarms / product_total if product_total > 0 else 0
    print(f"\n  3. False alarm (loại nhầm ảnh sản phẩm):")
    print(f"     Rate = {false_alarm_rate:.0%} ({false_alarms}/{product_total} ảnh sản phẩm bị loại nhầm)")
    if false_alarm_rate <= 0.10:
        print(f"     ✅ Rất ít loại nhầm")
    elif false_alarm_rate <= 0.20:
        print(f"     ⚠️  Chấp nhận được nhưng có thể cải thiện")
    else:
        print(f"     ❌ Loại nhầm quá nhiều ảnh tốt")

    # ── Kết luận ──
    print(f"\n{'='*70}")
    if product_recall >= 0.70 and irr_recall >= 0.50:
        print(f"  ✅ CLIP HOẠT ĐỘNG TỐT — Sẵn sàng dùng làm pre-filter cho ResNet50")
        print(f"     Pipeline: CLIP (lọc irrelevant) → ResNet50 (defect/no-defect)")
    elif product_recall >= 0.50:
        print(f"  ⚠️  CLIP CẦN CẢI THIỆN — Hoạt động nhưng chưa đủ chính xác")
        print(f"     → Thử model lớn hơn: clip-vit-large-patch14")
        print(f"     → Hoặc cải thiện prompts cho phù hợp dataset")
    else:
        print(f"  ❌ CLIP KHÔNG HIỆU QUẢ — Cần xem lại thiết kế")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Quick test CLIP binary classifier"
    )
    parser.add_argument(
        "--per-class", type=int, default=5,
        help="Số ảnh test mỗi class (mặc định: 5, tổng ~20 ảnh)"
    )
    args = parser.parse_args()
    run_test(args.per_class)


if __name__ == "__main__":
    main()
