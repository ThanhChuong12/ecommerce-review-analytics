"""
quality_gate.py
===============
Automated quality gate for ML models in ecommerce-review-analytics.
Checks metric targets defined in SKILL.md and reports PASS/FAIL.

Usage:
    python quality_gate.py --task spam_detection
    python quality_gate.py --task sentiment
    python quality_gate.py --task defect_detection
    python quality_gate.py --task all
    python quality_gate.py --task scraping
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

# ── Quality targets (mirror SKILL.md) ──────────────────────────────────────
TARGETS = {
    "spam_detection": {
        "f1_macro": {"pass": 0.90, "fail": 0.80},
        "latency_ms": {"pass": 200, "fail": 500, "direction": "lower"},
    },
    "sentiment": {
        "f1_macro": {"pass": 0.82, "fail": 0.70},
        "latency_ms": {"pass": 200, "fail": 500, "direction": "lower"},
    },
    "defect_detection": {
        "f1_macro": {"pass": 0.85, "fail": 0.75},
        "latency_ms": {"pass": 500, "fail": 1000, "direction": "lower"},
    },
    "scraping": {
        "success_rate": {"pass": 0.90, "fail": 0.70},
    },
}


def check_metric(name: str, value: float, targets: dict) -> tuple[str, str]:
    """Return (status, message): 'PASS', 'WARN', or 'FAIL'."""
    direction = targets.get("direction", "higher")

    if direction == "lower":
        # Lower is better (latency)
        if value <= targets["pass"]:
            return "PASS", f"{name}: {value:.1f} ≤ {targets['pass']} ✅"
        elif value <= targets["fail"]:
            return "WARN", f"{name}: {value:.1f} (target ≤ {targets['pass']}, warn ≤ {targets['fail']}) ⚠️"
        else:
            return "FAIL", f"{name}: {value:.1f} > {targets['fail']} ❌"
    else:
        # Higher is better (F1, success rate)
        if value >= targets["pass"]:
            return "PASS", f"{name}: {value:.4f} ≥ {targets['pass']} ✅"
        elif value >= targets["fail"]:
            return "WARN", f"{name}: {value:.4f} (target ≥ {targets['pass']}, warn ≥ {targets['fail']}) ⚠️"
        else:
            return "FAIL", f"{name}: {value:.4f} < {targets['fail']} ❌"


def run_spam_gate(model_path: str | None = None) -> dict:
    """Run quality gate for spam detection model."""
    print("\n[Quality Gate] spam_detection")
    print("=" * 50)

    results = {}

    # Check model file exists
    default_path = ROOT / "ai_engine" / "models" / "spam_iforest.pkl"
    path = Path(model_path) if model_path else default_path

    if not path.exists():
        print(f"  SKIP: Model not found at {path}")
        print(f"  Run: python scripts/train_spam_model.py --data-path data/processed/reviews.csv")
        return {"status": "SKIP", "reason": "model not found"}

    # Try to load and run a quick inference test
    try:
        import joblib
        import numpy as np
        model = joblib.load(path)
        print(f"  Model loaded: {path.name}")

        # Latency test (10 dummy samples)
        dummy_X = np.random.randn(10, 31).astype(np.float32)
        start = time.perf_counter()
        for _ in range(10):
            model.predict_anomaly(dummy_X)
        elapsed = (time.perf_counter() - start) / 10 * 1000  # ms per batch

        status, msg = check_metric("latency_ms", elapsed, TARGETS["spam_detection"]["latency_ms"])
        print(f"  {msg}")
        results["latency_ms"] = {"value": elapsed, "status": status}

    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return {"status": "ERROR", "reason": str(e)}

    # F1 check (needs labeled test data)
    test_path = ROOT / "data" / "processed" / "reviews_manual_labeled.csv"
    if not test_path.exists():
        print(f"  WARN: No ground-truth test file at {test_path}")
        print(f"  ACTION: Manually label ≥ 300 reviews and save to {test_path}")
        results["f1_macro"] = {"value": None, "status": "SKIP", "reason": "no ground-truth data"}
    else:
        try:
            import pandas as pd
            from scripts.train_spam_model import SpamHybridModel, build_feature_matrix
            from ai_engine.text_processing.spam_filter import detect_spam
            from sklearn.metrics import f1_score

            df = pd.read_csv(test_path)
            df["text"] = df["text"].fillna("").astype(str)
            df_flagged = detect_spam(df[["text", "rating"]])
            X = build_feature_matrix(df_flagged, df["text"].tolist(), df["rating"].tolist())

            rule_spam = df_flagged["is_spam"].values.astype(int)
            y_pred = model.predict_final_spam(X, rule_spam)
            y_true = df["is_spam"].values.astype(int)

            f1 = f1_score(y_true, y_pred, average="macro")
            status, msg = check_metric("f1_macro", f1, TARGETS["spam_detection"]["f1_macro"])
            print(f"  {msg}")
            results["f1_macro"] = {"value": f1, "status": status}
        except Exception as e:
            print(f"  ERROR during F1 evaluation: {e}")
            results["f1_macro"] = {"value": None, "status": "ERROR", "reason": str(e)}

    # Overall status
    statuses = [v["status"] for v in results.values()]
    overall = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
    results["overall"] = overall
    print(f"\n  Overall: {overall}")

    if overall == "FAIL":
        print("\n  AUTO-FIX SUGGESTIONS:")
        if results.get("f1_macro", {}).get("value") and results["f1_macro"]["value"] < 0.80:
            print("  → Increase augmentation: --multiply 3")
            print("  → Increase n_estimators: --n-estimators 300")
            print("  → Check class imbalance in training data")
        if results.get("latency_ms", {}).get("value") and results["latency_ms"]["value"] > 500:
            print("  → Reduce n_estimators")
            print("  → Add model caching in FastAPI startup")

    return results


def run_sentiment_gate(model_path: str | None = None) -> dict:
    """Run quality gate for sentiment model."""
    print("\n[Quality Gate] sentiment")
    print("=" * 50)

    results = {}

    # Latency test
    try:
        import time
        from ai_engine.text_processing.sentiment_analysis import NextGenReviewAnalyzer

        print("  Loading analyzer (first load may take 30s)...")
        t0 = time.perf_counter()
        analyzer = NextGenReviewAnalyzer()
        load_time = (time.perf_counter() - t0) * 1000
        print(f"  Model load time: {load_time:.0f}ms")

        test_texts = [
            "Sản phẩm rất tốt, giao hàng nhanh",
            "Hàng bị lỗi, rất thất vọng",
            "Tạm ổn, không có gì đặc biệt",
        ]
        start = time.perf_counter()
        for text in test_texts:
            analyzer.predict_sentiment(text)
        elapsed = (time.perf_counter() - start) / len(test_texts) * 1000

        status, msg = check_metric("latency_ms", elapsed, TARGETS["sentiment"]["latency_ms"])
        print(f"  {msg}")
        results["latency_ms"] = {"value": elapsed, "status": status}

        if elapsed > 200:
            print("  ACTION: Consider switching to vinai/phobert-base fine-tuned (see plan.md P3)")

    except Exception as e:
        print(f"  ERROR: {e}")
        results["latency_ms"] = {"status": "ERROR", "reason": str(e)}

    # F1 check
    model_path = model_path or str(ROOT / "ai_engine" / "models" / "ensemble_no_smote.pkl")
    test_path = ROOT / "data" / "processed" / "reviews_labeled.csv"

    if not Path(model_path).exists():
        print(f"  SKIP F1: Model not found at {model_path}")
        results["f1_macro"] = {"status": "SKIP"}
    elif not test_path.exists():
        print(f"  SKIP F1: Test data not found at {test_path}")
        results["f1_macro"] = {"status": "SKIP"}
    else:
        try:
            import pandas as pd
            from sklearn.metrics import f1_score
            import joblib

            model = joblib.load(model_path)
            df = pd.read_csv(test_path).dropna(subset=["cleaned_text", "sentiment"])
            y_pred = model.predict(df["cleaned_text"])
            f1 = f1_score(df["sentiment"], y_pred, average="macro")

            status, msg = check_metric("f1_macro", f1, TARGETS["sentiment"]["f1_macro"])
            print(f"  {msg}")
            results["f1_macro"] = {"value": f1, "status": status}
        except Exception as e:
            print(f"  ERROR during F1 evaluation: {e}")
            results["f1_macro"] = {"status": "ERROR", "reason": str(e)}

    statuses = [v["status"] for v in results.values()]
    overall = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
    results["overall"] = overall
    print(f"\n  Overall: {overall}")
    return results


def run_defect_gate(model_path: str | None = None) -> dict:
    """Run quality gate for defect detection model."""
    print("\n[Quality Gate] defect_detection")
    print("=" * 50)

    default_path = ROOT / "ai_engine" / "models" / "weights" / "resnet50_defect.pt"
    path = Path(model_path) if model_path else default_path

    if not path.exists():
        print(f"  SKIP: Model checkpoint not found at {path}")
        print(f"  Run: python scripts/train_defect_model.py")
        return {"status": "SKIP", "reason": "checkpoint not found"}

    results = {}

    try:
        import torch
        import numpy as np
        from ai_engine.image_processing.defect_detection import get_resnet50_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = get_resnet50_model(num_classes=2)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device).eval()
        print(f"  Model loaded on {device}")

        # Latency test
        dummy = torch.randn(1, 3, 224, 224).to(device)
        times = []
        with torch.no_grad():
            for _ in range(10):
                t0 = time.perf_counter()
                model(dummy)
                times.append((time.perf_counter() - t0) * 1000)
        avg_latency = sum(times) / len(times)

        status, msg = check_metric("latency_ms", avg_latency, TARGETS["defect_detection"]["latency_ms"])
        print(f"  {msg}")
        results["latency_ms"] = {"value": avg_latency, "status": status}

    except Exception as e:
        print(f"  ERROR: {e}")
        return {"status": "ERROR", "reason": str(e)}

    # F1 check (needs test data)
    data_path = ROOT / "data" / "processed"
    if not (data_path / "defect").exists() and not (data_path / "no-defect").exists():
        print(f"  SKIP F1: Test data not found at {data_path}")
        results["f1_macro"] = {"status": "SKIP"}
    else:
        try:
            from scripts.evaluate_models import evaluate_image_model
            res = evaluate_image_model(
                model_path=str(path),
                data_path=str(data_path),
                save_plot=False,
            )
            f1 = res.get("macro_f1", 0)
            status, msg = check_metric("f1_macro", f1, TARGETS["defect_detection"]["f1_macro"])
            print(f"  {msg}")
            results["f1_macro"] = {"value": f1, "status": status}
        except Exception as e:
            print(f"  ERROR during F1 evaluation: {e}")
            results["f1_macro"] = {"status": "ERROR", "reason": str(e)}

    statuses = [v["status"] for v in results.values()]
    overall = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
    results["overall"] = overall
    print(f"\n  Overall: {overall}")
    return results


def run_all_gates() -> dict:
    """Run quality gates for all models."""
    print("\n" + "=" * 60)
    print("  QUALITY GATE REPORT — All Models")
    print("=" * 60)

    all_results = {
        "spam_detection": run_spam_gate(),
        "sentiment": run_sentiment_gate(),
        "defect_detection": run_defect_gate(),
    }

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for task, results in all_results.items():
        overall = results.get("overall", "SKIP")
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}.get(overall, "?")
        print(f"  {task:<25} {overall} {icon}")

    # Output JSON for CI/CD integration
    output_path = ROOT / "reports" / "quality_gate_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved → {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Quality gate for ML models — checks metrics against targets"
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["spam_detection", "sentiment", "defect_detection", "all"],
        help="Which task to run quality gate for",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override default model path",
    )
    args = parser.parse_args()

    if args.task == "all":
        results = run_all_gates()
        any_fail = any(r.get("overall") == "FAIL" for r in results.values())
        sys.exit(1 if any_fail else 0)
    elif args.task == "spam_detection":
        results = run_spam_gate(args.model_path)
    elif args.task == "sentiment":
        results = run_sentiment_gate(args.model_path)
    elif args.task == "defect_detection":
        results = run_defect_gate(args.model_path)

    sys.exit(1 if results.get("overall") == "FAIL" else 0)


if __name__ == "__main__":
    main()
