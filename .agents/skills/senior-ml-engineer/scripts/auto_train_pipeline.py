"""
auto_train_pipeline.py
======================
End-to-end training pipeline: data check → augment → train → evaluate → quality gate.
Called by the AI agent after the user requests a training task.

Usage:
    python auto_train_pipeline.py --model spam
    python auto_train_pipeline.py --model sentiment
    python auto_train_pipeline.py --model defect
    python auto_train_pipeline.py --model all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SKILL_DIR = Path(__file__).parent.parent


def run(cmd: list[str], label: str) -> bool:
    """Run a subprocess command, return True if successful."""
    print(f"\n[Pipeline] {label}")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        return False
    print(f"  OK")
    return True


def run_spam_pipeline(
    data_path: str = "data/processed/reviews.csv",
    contamination: float = 0.1,
    multiply: int = 2,
) -> bool:
    """Full spam detection training pipeline."""
    print("\n" + "=" * 60)
    print("  SPAM DETECTION PIPELINE")
    print("=" * 60)

    steps = [
        # Step 1: Check data
        (
            ["python", "scripts/prepare_dataset.py", "--check-balance"],
            "1/5 Check dataset balance",
        ),
        # Step 2: Augment minority classes
        (
            [
                "python", "ai_engine/text_processing/augmentation.py",
                "--data-path", data_path,
                "--multiply", str(multiply),
                "--output-path", "data/processed/reviews_augmented.csv",
            ],
            f"2/5 Back-translation augmentation (x{multiply})",
        ),
        # Step 3: Train
        (
            [
                "python", "scripts/train_spam_model.py",
                "--data-path", "data/processed/reviews_augmented.csv",
                "--contamination", str(contamination),
                "--save-path", "ai_engine/models/spam_iforest.pkl",
                "--output-csv", "data/processed/reviews_spam_labeled.csv",
            ],
            "3/5 Train SpamHybridModel",
        ),
        # Step 4: Evaluate
        (
            [
                "python", "scripts/evaluate_models.py", "spam",
                "--model-path", "ai_engine/models/spam_iforest.pkl",
                "--data-path", "data/processed/reviews_spam_labeled.csv",
            ],
            "4/5 Evaluate model",
        ),
        # Step 5: Quality gate
        (
            [
                "python",
                str(SKILL_DIR / "scripts" / "quality_gate.py"),
                "--task", "spam_detection",
                "--model-path", "ai_engine/models/spam_iforest.pkl",
            ],
            "5/5 Quality gate",
        ),
    ]

    for cmd, label in steps:
        if not run(cmd, label):
            print(f"\n  PIPELINE STOPPED at: {label}")
            print("  Check the error above and fix before retrying.")
            return False

    print("\n  SPAM PIPELINE COMPLETE ✅")
    return True


def run_sentiment_pipeline(
    data_path: str = "data/processed/reviews_labeled.csv",
) -> bool:
    """Sentiment model evaluation pipeline (training happens in notebook)."""
    print("\n" + "=" * 60)
    print("  SENTIMENT PIPELINE")
    print("=" * 60)

    steps = [
        (
            [
                "python", "scripts/evaluate_models.py", "text",
                "--model-path", "ai_engine/models/ensemble_no_smote.pkl",
                "--data-path", data_path,
                "--text-col", "cleaned_text",
                "--label-col", "sentiment",
            ],
            "1/2 Evaluate text model",
        ),
        (
            [
                "python",
                str(SKILL_DIR / "scripts" / "quality_gate.py"),
                "--task", "sentiment",
            ],
            "2/2 Quality gate",
        ),
    ]

    for cmd, label in steps:
        if not run(cmd, label):
            print(f"\n  PIPELINE STOPPED at: {label}")
            return False

    print("\n  SENTIMENT PIPELINE COMPLETE ✅")
    return True


def run_defect_pipeline(
    data_path: str = "data/processed",
    batch_size: int = 32,
    epochs: int = 10,
) -> bool:
    """Defect detection training pipeline."""
    print("\n" + "=" * 60)
    print("  DEFECT DETECTION PIPELINE")
    print("=" * 60)

    steps = [
        (
            [
                "python", "scripts/train_defect_model.py",
                "--data-path", data_path,
                "--batch-size", str(batch_size),
                "--epochs", str(epochs),
                "--save-path", "ai_engine/models/weights/resnet50_defect.pt",
            ],
            "1/3 Train ResNet50",
        ),
        (
            [
                "python", "scripts/evaluate_models.py", "image",
                "--model-path", "ai_engine/models/weights/resnet50_defect.pt",
                "--data-path", data_path,
                "--batch-size", "16",
            ],
            "2/3 Evaluate model",
        ),
        (
            [
                "python",
                str(SKILL_DIR / "scripts" / "quality_gate.py"),
                "--task", "defect_detection",
            ],
            "3/3 Quality gate",
        ),
    ]

    for cmd, label in steps:
        if not run(cmd, label):
            print(f"\n  PIPELINE STOPPED at: {label}")
            return False

    print("\n  DEFECT PIPELINE COMPLETE ✅")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Auto training pipeline — runs full train→evaluate→quality gate"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["spam", "sentiment", "defect", "all"],
        help="Which model pipeline to run",
    )
    parser.add_argument("--data-path", default=None, help="Override default data path")
    parser.add_argument("--contamination", type=float, default=0.1,
                        help="Spam model contamination ratio (default: 0.1)")
    parser.add_argument("--multiply", type=int, default=2,
                        help="Augmentation multiplier (default: 2)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs for image model (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for image model (default: 32)")
    args = parser.parse_args()

    success = True
    if args.model == "spam":
        success = run_spam_pipeline(
            data_path=args.data_path or "data/processed/reviews.csv",
            contamination=args.contamination,
            multiply=args.multiply,
        )
    elif args.model == "sentiment":
        success = run_sentiment_pipeline(
            data_path=args.data_path or "data/processed/reviews_labeled.csv",
        )
    elif args.model == "defect":
        success = run_defect_pipeline(
            data_path=args.data_path or "data/processed",
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif args.model == "all":
        results = {
            "spam": run_spam_pipeline(),
            "sentiment": run_sentiment_pipeline(),
            "defect": run_defect_pipeline(epochs=args.epochs),
        }
        print("\n" + "=" * 60)
        print("  ALL PIPELINES SUMMARY")
        print("=" * 60)
        for name, ok in results.items():
            print(f"  {name:<20} {'✅ PASS' if ok else '❌ FAIL'}")
        success = all(results.values())

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
