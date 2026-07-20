"""
export_onnx.py
--------------
Export PyTorch models (MobileNetV3 / ResNet50) to ONNX format
to optimize inference speed using ONNX Runtime.

Usage:
    # Export MobileNetV3 (default)
    python scripts/export_onnx.py

    # Export with custom options
    python scripts/export_onnx.py \
        --weights ai_engine/models/weights/mobilenet_v3_defect.pt \
        --output ai_engine/models/weights/mobilenet_v3_defect.onnx \
        --opset 17

    # Benchmark comparing PyTorch vs ONNX
    python scripts/export_onnx.py --benchmark

Dependencies:
    pip install onnx onnxruntime
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np


def export_to_onnx(weights_path: str, output_path: str, opset_version: int = 17) -> str:
    """Export PyTorch model to ONNX.

    Args:
        weights_path: Path to the .pt file (from ImageBaselineModel.save()).
        output_path: Path for output .onnx file.
        opset_version: ONNX opset version (17 recommended).

    Returns:
        str: Path of the generated ONNX file.
    """
    from ai_engine.models.image_baseline import ImageBaselineModel

    print(f"[1/4] Loading PyTorch model from: {weights_path}")
    baseline = ImageBaselineModel.load(weights_path)
    model = baseline.model
    model.eval()
    model.cpu()

    class_names = baseline.class_names
    backbone = baseline.backbone
    print(f"       Backbone: {backbone} | Classes: {class_names}")

    # Dummy input — must match inference size: [batch, 3, 224, 224]
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"[2/4] Exporting to ONNX (opset={opset_version})...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},     # support dynamic batch size
            "logits": {0: "batch_size"},
        },
    )

    # Save metadata (class_names, backbone) to accompanying JSON file
    import json
    meta_path = output_path.replace(".onnx", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "backbone": backbone,
            "class_names": class_names,
            "source_weights": weights_path,
            "opset_version": opset_version,
            "input_shape": [1, 3, 224, 224],
            "input_name": "image",
            "output_name": "logits",
        }, f, indent=2, ensure_ascii=False)

    onnx_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"       ONNX saved: {output_path} ({onnx_size:.1f} MB)")
    print(f"       Metadata saved: {meta_path}")

    # Validate ONNX
    print("[3/4] Validating ONNX model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("       ONNX model is valid!")

    # Quick inference test
    print("[4/4] Testing ONNX inference...")
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    dummy_np = dummy_input.numpy()
    outputs = session.run(None, {"image": dummy_np})
    logits = outputs[0]
    probs = _softmax(logits[0])
    pred_idx = np.argmax(probs)
    print(f"       Test inference OK → predicted: {class_names[pred_idx]} (conf={probs[pred_idx]:.4f})")

    print(f"\nExport complete!")
    return output_path


def benchmark(weights_path: str, onnx_path: str, n_runs: int = 100):
    """Benchmark speed comparison between PyTorch vs ONNX Runtime.

    Args:
        weights_path: Path to the original .pt file.
        onnx_path: Path to the exported .onnx file.
        n_runs: Number of inference runs to compute average.
    """
    import onnxruntime as ort
    from ai_engine.models.image_baseline import ImageBaselineModel

    print(f"\n{'='*60}")
    print(f"  Benchmark: PyTorch vs ONNX Runtime ({n_runs} runs)")
    print(f"{'='*60}\n")

    dummy_tensor = torch.randn(1, 3, 224, 224)
    dummy_np = dummy_tensor.numpy()

    # --- PyTorch ---
    baseline = ImageBaselineModel.load(weights_path)
    model = baseline.model
    model.eval()
    model.cpu()

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(dummy_tensor)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy_tensor)
    pytorch_ms = (time.perf_counter() - t0) / n_runs * 1000

    # --- ONNX Runtime ---
    session = ort.InferenceSession(onnx_path)

    # Warm-up
    for _ in range(10):
        session.run(None, {"image": dummy_np})

    t0 = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {"image": dummy_np})
    onnx_ms = (time.perf_counter() - t0) / n_runs * 1000

    # --- Accuracy check ---
    with torch.no_grad():
        pytorch_logits = model(dummy_tensor).numpy()[0]
    onnx_logits = session.run(None, {"image": dummy_np})[0][0]
    max_diff = np.max(np.abs(pytorch_logits - onnx_logits))

    # --- Results ---
    speedup = pytorch_ms / onnx_ms if onnx_ms > 0 else 0
    print(f"  PyTorch:      {pytorch_ms:.2f} ms/image")
    print(f"  ONNX Runtime: {onnx_ms:.2f} ms/image")
    print(f"  Speedup:      {speedup:.1f}x faster")
    print(f"  Max logit diff: {max_diff:.6f} (should be < 0.001)")
    print(f"\n  File sizes:")
    print(f"    PyTorch: {Path(weights_path).stat().st_size / (1024*1024):.1f} MB")
    print(f"    ONNX:    {Path(onnx_path).stat().st_size / (1024*1024):.1f} MB")
    print(f"{'='*60}")

    return {
        "pytorch_ms": round(pytorch_ms, 2),
        "onnx_ms": round(onnx_ms, 2),
        "speedup": round(speedup, 1),
        "max_logit_diff": round(float(max_diff), 6),
    }


def _softmax(x):
    """Numpy softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument(
        "--weights", type=str,
        default="ai_engine/models/weights/mobilenet_v3_defect.pt",
        help="Path to PyTorch weights (.pt)",
    )
    parser.add_argument(
        "--output", type=str,
        default=None,
        help="Path for output ONNX file (default: same name with .onnx extension)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after export")
    parser.add_argument("--benchmark-runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    # Default output: change .pt → .onnx
    if args.output is None:
        args.output = str(Path(args.weights).with_suffix(".onnx"))

    onnx_path = export_to_onnx(args.weights, args.output, args.opset)

    if args.benchmark:
        benchmark(args.weights, onnx_path, args.benchmark_runs)


if __name__ == "__main__":
    main()
