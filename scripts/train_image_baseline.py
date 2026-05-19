"""train_image_baseline.py
------------------------
Script CLI để train ResNet50 và MobileNetV3 trên dữ liệu ảnh đã gán nhãn.

Dữ liệu đầu vào: image_labeling/data/labeled/ (ImageFolder format)
Artifacts đầu ra: ai_engine/models/weights/

Chạy:
  python ai_engine/scripts/train_image_baseline.py
  python ai_engine/scripts/train_image_baseline.py --backbone mobilenet_v3 --epochs 15
  python ai_engine/scripts/train_image_baseline.py --backbone resnet50 --lr 5e-4 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Thêm project root vào sys.path để import ai_engine đúng
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_engine.models.image_baseline import ImageBaselineModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Thư mục mặc định
DEFAULT_DATA_DIR = str(PROJECT_ROOT / "image_labeling" / "data" / "labeled")
DEFAULT_WEIGHTS_DIR = str(PROJECT_ROOT / "ai_engine" / "models" / "weights")
DEFAULT_RESULTS_DIR = str(PROJECT_ROOT / "ai_engine" / "models" / "results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Transfer Learning model nhận diện hộp hư hỏng"
    )
    parser.add_argument(
        "--backbone",
        choices=["resnet50", "mobilenet_v3", "both"],
        default="both",
        help="Backbone cần train. 'both' sẽ train cả 2 (mặc định: both)",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Thư mục ảnh gán nhãn (ImageFolder format)")
    parser.add_argument("--epochs", type=int, default=10, help="Số epoch tối đa (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Tỉ lệ validation split (default: 0.2)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (default: 3)")
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR, help="Thư mục lưu file .pt")
    parser.add_argument("--eval-only", action="store_true", help="Chỉ đánh giá model đã train, không train lại")
    return parser.parse_args()


def train_single(
    backbone: str,
    data_dir: str,
    weights_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    patience: int,
) -> dict:
    """Train 1 backbone và lưu weights. Trả về metrics summary."""
    logger.info("--- Bắt đầu train: %s ---", backbone.upper())
    t_start = time.time()

    model = ImageBaselineModel(backbone=backbone)

    model.fit(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_split=val_split,
        patience=patience,
    )

    # Lưu weights
    weights_path = os.path.join(weights_dir, f"{backbone}_defect.pt")
    model.save(weights_path)

    # Đánh giá trên toàn bộ val set (sử dụng lại data_dir vì ImageFolder)
    logger.info("Đang evaluate model %s...", backbone)
    eval_report = model.evaluate(data_dir=data_dir, batch_size=batch_size)

    elapsed = (time.time() - t_start) / 60
    summary = {
        "backbone": backbone,
        "overall_accuracy": round(eval_report["accuracy"], 4),
        "macro_f1": round(eval_report["macro avg"]["f1-score"], 4),
        "training_minutes": round(elapsed, 2),
        "weights_path": weights_path,
    }

    logger.info(
        "%s hoàn tất — Accuracy=%.4f | Macro-F1=%.4f | Time=%.1f phút",
        backbone, summary["overall_accuracy"], summary["macro_f1"], elapsed,
    )
    return summary


def eval_only(backbone: str, data_dir: str, weights_dir: str, batch_size: int) -> dict:
    """Load model đã train và chỉ đánh giá, không train lại."""
    weights_path = os.path.join(weights_dir, f"{backbone}_defect.pt")
    if not os.path.exists(weights_path):
        logger.error("Không tìm thấy weights: %s", weights_path)
        return {}

    model = ImageBaselineModel.load(weights_path)
    report = model.evaluate(data_dir=data_dir, batch_size=batch_size)
    return {
        "backbone": backbone,
        "overall_accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weights_path": weights_path,
    }


def main():
    args = parse_args()

    if not os.path.exists(args.data_dir):
        logger.error(
            "Không tìm thấy thư mục dữ liệu: %s\n"
            "Chạy media_pipeline.py để tải và gán nhãn ảnh trước.",
            args.data_dir,
        )
        sys.exit(1)

    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)

    # Xác định các backbone cần xử lý
    backbones = ["resnet50", "mobilenet_v3"] if args.backbone == "both" else [args.backbone]

    all_results = []
    for backbone in backbones:
        if args.eval_only:
            result = eval_only(backbone, args.data_dir, args.weights_dir, args.batch_size)
        else:
            result = train_single(
                backbone=backbone,
                data_dir=args.data_dir,
                weights_dir=args.weights_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                val_split=args.val_split,
                patience=args.patience,
            )
        if result:
            all_results.append(result)

    # Lưu bảng so sánh kết quả ra JSON
    results_path = os.path.join(DEFAULT_RESULTS_DIR, "image_baseline_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Kết quả so sánh lưu tại: %s", results_path)

    # In bảng tóm tắt
    print("\n" + "-" * 60)
    print(f"{'Backbone':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Time (min)':>12}")
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['backbone']:<20} {r['overall_accuracy']:>10.4f} "
            f"{r['macro_f1']:>10.4f} {r.get('training_minutes', '-'):>12}"
        )
    print("-" * 60)


if __name__ == "__main__":
    main()
