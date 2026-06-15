"""train_image_baseline.py
------------------------
Script CLI để train ResNet50, MobileNetV3, và EfficientNet-B0 trên dữ liệu ảnh đã gán nhãn.

Dữ liệu đầu vào: labeled/train/ (ImageFolder format, sau khi chạy split_image_dataset.py)
Artifacts đầu ra:
  - ai_engine/models/weights/<backbone>_defect.pt
  - ai_engine/models/results/<backbone>_learning_curves.png
  - ai_engine/models/results/<backbone>_confusion_matrix.png
  - ai_engine/models/results/<backbone>_training_history.json
  - ai_engine/models/results/<backbone>_error_analysis/

Chạy nhanh trên CPU (~1.5–2h với default settings):
  python scripts/train_image_baseline.py --backbone mobilenet_v3

Chạy full (Colab T4 GPU, ~40 phút):
  python scripts/train_image_baseline.py --backbone mobilenet_v3 --epochs 20 --subset-ratio 1.0

Train 3 mô hình để so sánh:
  python scripts/train_image_baseline.py --backbone all
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
DEFAULT_DATA_DIR = str(PROJECT_ROOT / "labeled" / "train")
DEFAULT_VAL_DIR  = str(PROJECT_ROOT / "labeled" / "val")
DEFAULT_TEST_DIR = str(PROJECT_ROOT / "labeled" / "test")
DEFAULT_WEIGHTS_DIR = str(PROJECT_ROOT / "ai_engine" / "models" / "weights")
DEFAULT_RESULTS_DIR = str(PROJECT_ROOT / "ai_engine" / "models" / "results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Transfer Learning model nhận diện hộp hư hỏng"
    )
    parser.add_argument(
        "--backbone",
        choices=["resnet50", "mobilenet_v3", "efficientnet_b0", "all"],
        default="mobilenet_v3",
        help="Backbone cần train. 'all' sẽ train cả 3 (mặc định: mobilenet_v3)",
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="Thư mục ảnh train (ImageFolder format). Mặc định: labeled/train/",
    )
    parser.add_argument(
        "--val-dir", default=None,
        help="Thư mục ảnh validation vật lý (ImageFolder format). Nếu cung cấp, fit() sẽ dùng thư mục này và bỏ qua val-split.",
    )
    parser.add_argument("--epochs", type=int, default=15,
        help="Số epoch tối đa (default: 15 — early stopping thường dừng epoch 8-12)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Tỉ lệ validation split NỘI BỘ. Khi dùng labeled/train/ nên đặt = 0.2 để fit() tự chia từ train set. (default: 0.2)",
    )
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (default: 3)")
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR, help="Thư mục lưu file .pt")
    parser.add_argument(
        "--results-dir", default=DEFAULT_RESULTS_DIR,
        help="Thư mục lưu plots, history, error analysis (mặc định: ai_engine/models/results)",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Chỉ đánh giá model đã lưu trên --data-dir, không train lại",
    )
    parser.add_argument(
        "--eval-test", action="store_true",
        help="Evaluate model đã lưu trên Test set vật lý (labeled/test/). Dùng SAU KHI train xong để báo cáo metrics chính thức.",
    )
    parser.add_argument(
        "--test-dir", default=DEFAULT_TEST_DIR,
        help="Thư mục Test set (mặc định: labeled/test/)",
    )
    parser.add_argument(
        "--subset-ratio", type=float, default=0.5,
        help="Tỉ lệ data train sử dụng (0 < x <= 1.0). Mặc định 0.5. Dùng 1.0 trên GPU.",
    )
    parser.add_argument(
        "--weights-name", default=None,
        help="Custom filename for model weights (e.g. mobilenet_v3_model2_improved_defect.pt)"
    )
    parser.add_argument(
        "--results-name", default=None,
        help="Custom filename for comparison results JSON (e.g. mobilenet_v3_model2_improved_results.json)"
    )
    parser.add_argument(
        "--learning-curves-name", default=None,
        help="Custom filename for learning curves plot"
    )
    parser.add_argument(
        "--confusion-matrix-name", default=None,
        help="Custom filename for confusion matrix plot"
    )
    parser.add_argument(
        "--training-history-name", default=None,
        help="Custom filename for training history JSON"
    )
    parser.add_argument(
        "--threshold-tuning-name", default=None,
        help="Custom filename for threshold tuning JSON"
    )
    parser.add_argument(
        "--class-weight-mode", choices=["none", "balanced", "sqrt"], default="sqrt",
        help="Class weighting mode for loss function (default: sqrt)"
    )
    parser.add_argument(
        "--use-sampler", action="store_true", default=False,
        help="Use WeightedRandomSampler for class balancing (default: False)"
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["maximize_defect_f1", "maximize_macro_f1", "maximize_macro_f1_subject_to_recall"],
        default="maximize_macro_f1_subject_to_recall",
        help="Mode for selecting the best threshold on validation set"
    )
    parser.add_argument(
        "--threshold-file", default=None,
        help="Path to saved threshold JSON file to use in evaluation (default: None)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Specific threshold value to use in evaluation (default: None)"
    )
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
    subset_ratio: float = 1.0,
    results_dir: str = "ai_engine/models/results",
    val_dir: str | None = None,
    class_weight_mode: str = "sqrt",
    use_sampler: bool = False,
    threshold_mode: str = "maximize_macro_f1_subject_to_recall",
    weights_name: str | None = None,
    results_name: str | None = None,
    learning_curves_name: str | None = None,
    confusion_matrix_name: str | None = None,
    training_history_name: str | None = None,
    threshold_tuning_name: str | None = None,
) -> dict:
    """Train 1 backbone và lưu weights. Trả về metrics summary."""
    logger.info("--- Bắt đầu train: %s ---", backbone.upper())
    t_start = time.time()

    model = ImageBaselineModel(backbone=backbone)

    model.fit(
        data_dir=data_dir,
        val_dir=val_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_split=val_split,
        patience=patience,
        subset_ratio=subset_ratio,
        results_dir=results_dir,
        class_weight_mode=class_weight_mode,
        use_sampler=use_sampler,
        threshold_mode=threshold_mode,
        learning_curves_name=learning_curves_name,
        confusion_matrix_name=confusion_matrix_name,
        training_history_name=training_history_name,
        threshold_tuning_name=threshold_tuning_name,
    )

    # Lưu weights
    filename = weights_name if weights_name else f"{backbone}_defect.pt"
    weights_path = os.path.join(weights_dir, filename)
    model.save(weights_path)

    # Lấy kết quả evaluate trên val set (đã chạy cuối training, không bị data leakage)
    logger.info("Đang lấy kết quả evaluate trên val set...")
    if hasattr(model, '_val_report') and model._val_report:
        eval_report = model._val_report
        logger.info("Sử dụng kết quả val set từ training (không bị data leakage).")
    else:
        logger.info("Fallback: evaluate model %s trên toàn bộ dataset...", backbone)
        eval_report = model.evaluate(data_dir=val_dir if val_dir else data_dir, batch_size=batch_size)

    elapsed = (time.time() - t_start) / 60
    
    defect_idx = 0
    defect_class = "defect"
    if "defect" in model.class_names:
        defect_class = "defect"
        defect_idx = model.class_names.index("defect")
    elif "damaged" in model.class_names:
        defect_class = "damaged"
        defect_idx = model.class_names.index("damaged")
        
    defect_precision = eval_report[defect_class]["precision"]
    defect_recall = eval_report[defect_class]["recall"]
    defect_f1 = eval_report[defect_class]["f1-score"]

    summary = {
        "backbone": backbone,
        "overall_accuracy": round(eval_report["accuracy"], 4),
        "macro_f1": round(eval_report["macro avg"]["f1-score"], 4),
        "defect_precision": round(defect_precision, 4),
        "defect_recall": round(defect_recall, 4),
        "defect_f1": round(defect_f1, 4),
        "training_minutes": round(elapsed, 2),
        "weights_path": weights_path,
        "threshold_used": model.threshold,
    }

    logger.info(
        "%s hoàn tất — Accuracy=%.4f | Macro-F1=%.4f | Defect F1=%.4f | Time=%.1f phút",
        backbone, summary["overall_accuracy"], summary["macro_f1"], summary["defect_f1"], elapsed,
    )
    return summary


def eval_only(
    backbone: str,
    data_dir: str,
    weights_dir: str,
    batch_size: int,
    threshold: float | None = None,
    weights_name: str | None = None,
) -> dict:
    """Load model đã train và chỉ đánh giá, không train lại."""
    filename = weights_name if weights_name else f"{backbone}_defect.pt"
    weights_path = os.path.join(weights_dir, filename)
    if not os.path.exists(weights_path):
        logger.error("Không tìm thấy weights: %s", weights_path)
        return {}

    model = ImageBaselineModel.load(weights_path)
    report = model.evaluate(data_dir=data_dir, batch_size=batch_size, threshold=threshold)
    
    defect_idx = 0
    defect_class = "defect"
    if "defect" in model.class_names:
        defect_class = "defect"
        defect_idx = model.class_names.index("defect")
    elif "damaged" in model.class_names:
        defect_class = "damaged"
        defect_idx = model.class_names.index("damaged")
        
    defect_precision = report[defect_class]["precision"]
    defect_recall = report[defect_class]["recall"]
    defect_f1 = report[defect_class]["f1-score"]

    return {
        "backbone": backbone,
        "overall_accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "defect_precision": round(defect_precision, 4),
        "defect_recall": round(defect_recall, 4),
        "defect_f1": round(defect_f1, 4),
        "weights_path": weights_path,
        "threshold_used": threshold if threshold is not None else getattr(model, "threshold", 0.5),
    }


def main():
    args = parse_args()

    # --- Evaluate trên Test set vật lý ---
    if args.eval_test:
        if not os.path.exists(args.test_dir):
            logger.error(
                "Không tìm thấy Test set: %s\n"
                "Chạy scripts/split_image_dataset.py trước để tạo labeled/test/",
                args.test_dir,
            )
            sys.exit(1)
        logger.info("=" * 60)
        logger.info("EVALUATE TRÊN TEST SET: %s", args.test_dir)
        logger.info("(Test set này KHÔNG được dùng trong bất kỳ bước training nào)")
        logger.info("=" * 60)
        
        eval_threshold = None
        if args.threshold_file and os.path.exists(args.threshold_file):
            try:
                with open(args.threshold_file, "r") as f:
                    tuning_data = json.load(f)
                eval_threshold = tuning_data.get("best_threshold")
                logger.info("Loaded best threshold %.3f from %s", eval_threshold, args.threshold_file)
            except Exception as e:
                logger.error("Lỗi khi đọc threshold file: %s", e)
        elif args.threshold is not None:
            eval_threshold = args.threshold
            logger.info("Using CLI-specified threshold %.3f", eval_threshold)
            
        backbones = (
            ["resnet50", "mobilenet_v3", "efficientnet_b0"]
            if args.backbone == "all"
            else [args.backbone]
        )
        test_results = []
        for backbone in backbones:
            result = eval_only(
                backbone, args.test_dir, args.weights_dir, args.batch_size,
                threshold=eval_threshold, weights_name=args.weights_name
            )
            if result:
                result["eval_set"] = "TEST"
                test_results.append(result)
                
        # Lưu kết quả test riêng
        res_name = args.results_name if args.results_name else "test_set_results.json"
        test_results_path = os.path.join(args.results_dir, res_name)
        os.makedirs(args.results_dir, exist_ok=True)
        with open(test_results_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        logger.info("Kết quả Test set lưu tại: %s", test_results_path)
        
        print("\n" + "=" * 80)
        print(f"  {'Backbone':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Defect F1':>10} {'Thresh':>10}  ← TEST SET")
        print("=" * 80)
        for r in test_results:
            print(
                f"  {r['backbone']:<20} {r['overall_accuracy']:>10.4f} "
                f"{r['macro_f1']:>10.4f} {r['defect_f1']:>10.4f} {r['threshold_used']:>10.3f}"
            )
        print("=" * 80)
        return

    # --- Train hoặc eval trên train/val set ---
    if not os.path.exists(args.data_dir):
        logger.error(
            "Không tìm thấy thư mục dữ liệu: %s\n"
            "Chạy scripts/split_image_dataset.py để tạo labeled/train/ trước.",
            args.data_dir,
        )
        sys.exit(1)

    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    backbones = ["resnet50", "mobilenet_v3"] if args.backbone == "both" else [args.backbone]

    # Load threshold for evaluation if in eval_only mode
    eval_threshold = None
    if args.eval_only:
        if args.threshold_file and os.path.exists(args.threshold_file):
            try:
                with open(args.threshold_file, "r") as f:
                    tuning_data = json.load(f)
                eval_threshold = tuning_data.get("best_threshold")
            except Exception as e:
                logger.error("Lỗi khi đọc threshold file: %s", e)
        elif args.threshold is not None:
            eval_threshold = args.threshold

    all_results = []
    for backbone in backbones:
        if args.eval_only:
            result = eval_only(
                backbone, args.data_dir, args.weights_dir, args.batch_size,
                threshold=eval_threshold, weights_name=args.weights_name
            )
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
                subset_ratio=args.subset_ratio,
                results_dir=args.results_dir,
                val_dir=args.val_dir,
                class_weight_mode=args.class_weight_mode,
                use_sampler=args.use_sampler,
                threshold_mode=args.threshold_mode,
                weights_name=args.weights_name,
                results_name=args.results_name,
                learning_curves_name=args.learning_curves_name,
                confusion_matrix_name=args.confusion_matrix_name,
                training_history_name=args.training_history_name,
                threshold_tuning_name=args.threshold_tuning_name,
            )
        if result:
            all_results.append(result)

    # Lưu bảng so sánh kết quả ra JSON
    res_name = args.results_name if args.results_name else "image_baseline_results.json"
    results_path = os.path.join(args.results_dir, res_name)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Kết quả so sánh lưu tại: %s", results_path)

    # In bảng tóm tắt
    print("\n" + "-" * 80)
    print(f"{'Backbone':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Defect F1':>10} {'Thresh':>10} {'Time (min)':>12}")
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['backbone']:<20} {r['overall_accuracy']:>10.4f} "
            f"{r['macro_f1']:>10.4f} {r.get('defect_f1', 0.0):>10.4f} "
            f"{r.get('threshold_used', 0.5):>10.3f} {r.get('training_minutes', '-'):>12}"
        )
    print("-" * 80)


if __name__ == "__main__":
    main()
