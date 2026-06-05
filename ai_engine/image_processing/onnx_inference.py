"""
onnx_inference.py
-----------------
ONNX Runtime inference wrapper cho các model đã export từ PyTorch.

Thay thế PyTorch inference khi deploy production:
  - Không cần cài PyTorch (~2GB) → chỉ cần onnxruntime (~50MB)
  - Inference nhanh hơn 2-3x trên CPU
  - Hỗ trợ batch inference

Usage:
    >>> from ai_engine.image_processing.onnx_inference import OnnxDefectDetector
    >>> detector = OnnxDefectDetector("ai_engine/models/weights/mobilenet_v3_defect.onnx")
    >>> result = detector.predict("path/to/image.jpg")
    >>> results = detector.predict_batch(["img1.jpg", "img2.jpg"])

Dependencies:
    pip install onnxruntime pillow numpy
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

_logger = logging.getLogger(__name__)

# ImageNet normalization — phải khớp với lúc train
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 224

# Default class names (khớp với ImageBaselineModel)
DEFAULT_CLASS_NAMES = ["intact", "damaged", "wrong_item", "irrelevant"]


def _preprocess_image(image_path: str) -> np.ndarray:
    """Tiền xử lý ảnh cho ONNX inference (khớp với _build_transforms(is_train=False)).

    Pipeline: Resize(256) → CenterCrop(224) → ToArray → Normalize(ImageNet)

    Args:
        image_path: Đường dẫn file ảnh.

    Returns:
        np.ndarray: shape [1, 3, 224, 224], float32, đã normalize.
    """
    img = Image.open(image_path).convert("RGB")

    # Resize sao cho cạnh ngắn = 256 (giữ tỷ lệ)
    w, h = img.size
    if w < h:
        new_w = 256
        new_h = int(h * 256 / w)
    else:
        new_h = 256
        new_w = int(w * 256 / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop 224×224
    left = (new_w - IMAGE_SIZE) // 2
    top = (new_h - IMAGE_SIZE) // 2
    img = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    # To numpy [H, W, C] → normalize → transpose to [C, H, W]
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # [3, 224, 224]

    return arr[np.newaxis, ...]  # [1, 3, 224, 224]


class OnnxDefectDetector:
    """ONNX Runtime inference wrapper cho model nhận diện tình trạng sản phẩm.

    Attributes:
        session: ONNX Runtime InferenceSession.
        class_names: Danh sách nhãn output.
        backbone: Tên backbone gốc (từ metadata).
    """

    def __init__(self, onnx_path: str, meta_path: str = None):
        """Khởi tạo ONNX detector.

        Args:
            onnx_path: Đường dẫn file .onnx.
            meta_path: Đường dẫn file metadata JSON (tự tìm nếu None).
        """
        import onnxruntime as ort

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model không tồn tại: '{onnx_path}'")

        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(onnx_path)

        # Load metadata (class_names, backbone)
        if meta_path is None:
            meta_path = onnx_path.replace(".onnx", "_meta.json")

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.class_names = meta.get("class_names", DEFAULT_CLASS_NAMES)
            self.backbone = meta.get("backbone", "unknown")
        else:
            self.class_names = DEFAULT_CLASS_NAMES
            self.backbone = "unknown"
            _logger.warning("Metadata not found at '%s', using defaults.", meta_path)

        _logger.info(
            "ONNX detector loaded: %s (backbone=%s, classes=%s)",
            onnx_path, self.backbone, self.class_names,
        )

    def predict(self, image_path: str) -> dict:
        """Dự đoán nhãn cho 1 ảnh.

        Args:
            image_path: Đường dẫn file ảnh.

        Returns:
            dict: {
                "label": str,
                "confidence": float,
                "probabilities": dict,
                "inference_ms": float,
                "model_path": str,
                "runtime": "onnxruntime",
            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Ảnh không tồn tại: '{image_path}'")

        input_array = _preprocess_image(image_path)

        t0 = time.perf_counter()
        outputs = self.session.run(None, {"image": input_array})
        inference_ms = (time.perf_counter() - t0) * 1000

        logits = outputs[0][0]  # [num_classes]
        probs = _softmax(logits)
        pred_idx = int(np.argmax(probs))

        return {
            "label": self.class_names[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": {
                name: round(float(probs[i]), 4)
                for i, name in enumerate(self.class_names)
            },
            "inference_ms": round(inference_ms, 2),
            "model_path": self.onnx_path,
            "runtime": "onnxruntime",
        }

    def predict_batch(self, image_paths: list, batch_size: int = 32) -> list:
        """Dự đoán cho nhiều ảnh (batch inference).

        Args:
            image_paths: Danh sách đường dẫn ảnh.
            batch_size: Số ảnh xử lý mỗi lần forward pass.

        Returns:
            list[dict]: Kết quả cho mỗi ảnh.
        """
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            arrays = []
            valid_paths = []

            for p in batch_paths:
                try:
                    arr = _preprocess_image(p)
                    arrays.append(arr[0])  # remove batch dim, will stack later
                    valid_paths.append(p)
                except Exception as exc:
                    _logger.warning("Bỏ qua ảnh lỗi %s: %s", p, exc)

            if not arrays:
                continue

            batch_input = np.stack(arrays, axis=0).astype(np.float32)  # [B, 3, 224, 224]

            t0 = time.perf_counter()
            outputs = self.session.run(None, {"image": batch_input})
            inference_ms = (time.perf_counter() - t0) * 1000 / len(arrays)

            batch_logits = outputs[0]  # [B, num_classes]
            for j, path in enumerate(valid_paths):
                probs = _softmax(batch_logits[j])
                pred_idx = int(np.argmax(probs))
                results.append({
                    "image_path": path,
                    "label": self.class_names[pred_idx],
                    "confidence": round(float(probs[pred_idx]), 4),
                    "probabilities": {
                        name: round(float(probs[k]), 4)
                        for k, name in enumerate(self.class_names)
                    },
                    "inference_ms": round(inference_ms, 2),
                    "model_path": self.onnx_path,
                    "runtime": "onnxruntime",
                })

        return results


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numpy softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ── Singleton factory ───────────────────────────────────────────────────────

_DEFAULT_ONNX_WEIGHTS = os.getenv(
    "ONNX_WEIGHTS_PATH",
    "ai_engine/models/weights/mobilenet_v3_defect.onnx",
)

_onnx_detector_cache: Optional[OnnxDefectDetector] = None
_onnx_detector_path_cache: Optional[str] = None


def get_onnx_detector(onnx_path: str = None) -> OnnxDefectDetector:
    """Singleton factory — load ONNX detector 1 lần duy nhất.

    Args:
        onnx_path: Đường dẫn file .onnx. Mặc định từ env ONNX_WEIGHTS_PATH.

    Returns:
        OnnxDefectDetector instance (cached).
    """
    global _onnx_detector_cache, _onnx_detector_path_cache

    if onnx_path is None:
        onnx_path = _DEFAULT_ONNX_WEIGHTS

    if _onnx_detector_cache is not None and _onnx_detector_path_cache == onnx_path:
        return _onnx_detector_cache

    _onnx_detector_cache = OnnxDefectDetector(onnx_path)
    _onnx_detector_path_cache = onnx_path
    return _onnx_detector_cache
