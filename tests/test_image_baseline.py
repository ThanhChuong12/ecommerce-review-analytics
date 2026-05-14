"""Unit tests cho ImageBaselineModel.

Chạy:
    python -m pytest tests/test_image_baseline.py -v

Không cần GPU hay data thật — dùng dummy tensors và temp dirs.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn


class TestBuildTransforms(unittest.TestCase):
    """Test pipeline transform ảnh."""

    def test_train_transform_output_shape(self):
        from ai_engine.models.image_baseline import _build_transforms
        from PIL import Image
        import numpy as np

        transform = _build_transforms(is_train=True)
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype="uint8"))
        tensor = transform(img)
        # Output phải là (3, 224, 224)
        self.assertEqual(tensor.shape, (3, 224, 224))

    def test_eval_transform_output_shape(self):
        from ai_engine.models.image_baseline import _build_transforms
        from PIL import Image
        import numpy as np

        transform = _build_transforms(is_train=False)
        img = Image.fromarray(np.random.randint(0, 255, (400, 400, 3), dtype="uint8"))
        tensor = transform(img)
        self.assertEqual(tensor.shape, (3, 224, 224))

    def test_train_transform_is_normalized(self):
        """Giá trị pixel sau normalize không còn nằm trong [0,1]."""
        from ai_engine.models.image_baseline import _build_transforms
        from PIL import Image
        import numpy as np

        transform = _build_transforms(is_train=False)
        img = Image.fromarray(np.zeros((300, 300, 3), dtype="uint8"))
        tensor = transform(img)
        # Pixel 0 sau normalize với ImageNet mean → âm
        self.assertTrue(tensor.min() < 0)


class TestBuildBackbone(unittest.TestCase):
    """Test việc khởi tạo backbone và freeze layers."""

    def test_resnet50_head_replaced(self):
        from ai_engine.models.image_baseline import _build_backbone, NUM_CLASSES

        net, in_features = _build_backbone("resnet50")
        # in_features phải là 2048
        self.assertEqual(in_features, 2048)
        # Head mới phải output NUM_CLASSES
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = net(dummy)
        self.assertEqual(out.shape, (1, NUM_CLASSES))

    def test_mobilenet_head_replaced(self):
        from ai_engine.models.image_baseline import _build_backbone, NUM_CLASSES

        net, in_features = _build_backbone("mobilenet_v3")
        self.assertEqual(in_features, 1280)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = net(dummy)
        self.assertEqual(out.shape, (1, NUM_CLASSES))

    def test_backbone_frozen_except_head(self):
        """Chỉ custom head mới có requires_grad=True."""
        from ai_engine.models.image_baseline import _build_backbone

        net, _ = _build_backbone("mobilenet_v3")
        trainable = [n for n, p in net.named_parameters() if p.requires_grad]
        # Các param trainable phải thuộc về classifier (head)
        self.assertTrue(all("classifier" in n for n in trainable))

    def test_invalid_backbone_raises(self):
        from ai_engine.models.image_baseline import _build_backbone

        with self.assertRaises(ValueError):
            _build_backbone("vgg16")


class TestImageBaselineModelPredict(unittest.TestCase):
    """Test predict() không cần train thật — dùng backbone mới khởi tạo."""

    def setUp(self):
        from ai_engine.models.image_baseline import ImageBaselineModel
        self.model = ImageBaselineModel(backbone="mobilenet_v3", device="cpu")
        # Force khởi tạo model (không cần train)
        self.model._get_model()

    def _make_temp_image(self) -> str:
        """Tạo ảnh JPEG giả vào tempdir."""
        from PIL import Image
        import numpy as np

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype="uint8"))
        img.save(tmp.name)
        return tmp.name

    def test_predict_returns_expected_keys(self):
        path = self._make_temp_image()
        try:
            result = self.model.predict(path)
            self.assertIn("label", result)
            self.assertIn("confidence", result)
            self.assertIn("probabilities", result)
            self.assertIn("inference_ms", result)
        finally:
            os.unlink(path)

    def test_predict_label_in_class_names(self):
        from ai_engine.models.image_baseline import CLASS_NAMES

        path = self._make_temp_image()
        try:
            result = self.model.predict(path)
            self.assertIn(result["label"], CLASS_NAMES)
        finally:
            os.unlink(path)

    def test_predict_confidence_range(self):
        path = self._make_temp_image()
        try:
            result = self.model.predict(path)
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)
        finally:
            os.unlink(path)

    def test_predict_probabilities_sum_to_one(self):
        path = self._make_temp_image()
        try:
            result = self.model.predict(path)
            total = sum(result["probabilities"].values())
            self.assertAlmostEqual(total, 1.0, places=3)
        finally:
            os.unlink(path)

    def test_predict_inference_ms_positive(self):
        path = self._make_temp_image()
        try:
            result = self.model.predict(path)
            self.assertGreater(result["inference_ms"], 0)
        finally:
            os.unlink(path)

    def test_predict_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.model.predict("non_existent_image.jpg")


class TestSaveLoad(unittest.TestCase):
    """Test lưu và load model weights."""

    def test_save_and_load_roundtrip(self):
        from ai_engine.models.image_baseline import ImageBaselineModel

        model = ImageBaselineModel(backbone="mobilenet_v3", device="cpu")
        model._get_model()  # khởi tạo weights

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model.pt")
            model.save(path)
            self.assertTrue(os.path.exists(path))

            # Load lại và kiểm tra metadata
            loaded = ImageBaselineModel.load(path)
            self.assertEqual(loaded.backbone, "mobilenet_v3")
            self.assertEqual(loaded.class_names, model.class_names)

    def test_save_without_fit_raises(self):
        from ai_engine.models.image_baseline import ImageBaselineModel

        model = ImageBaselineModel(backbone="mobilenet_v3")
        with self.assertRaises(RuntimeError):
            model.save("/tmp/should_fail.pt")

    def test_load_missing_file_raises(self):
        from ai_engine.models.image_baseline import ImageBaselineModel

        with self.assertRaises(FileNotFoundError):
            ImageBaselineModel.load("non_existent_weights.pt")


class TestPredictBatch(unittest.TestCase):
    """Test predict_batch() với nhiều ảnh."""

    def setUp(self):
        from ai_engine.models.image_baseline import ImageBaselineModel
        self.model = ImageBaselineModel(backbone="mobilenet_v3", device="cpu")
        self.model._get_model()

    def _make_images(self, n: int) -> list:
        from PIL import Image
        import numpy as np

        paths = []
        for _ in range(n):
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype="uint8"))
            img.save(tmp.name)
            paths.append(tmp.name)
        return paths

    def test_batch_returns_same_count(self):
        paths = self._make_images(5)
        try:
            results = self.model.predict_batch(paths)
            self.assertEqual(len(results), 5)
        finally:
            for p in paths:
                os.unlink(p)

    def test_batch_skips_invalid_image(self):
        """File không tồn tại bị bỏ qua, không raise."""
        paths = self._make_images(2) + ["invalid_path.jpg"]
        valid_paths = paths[:2]
        try:
            results = self.model.predict_batch(paths)
            # Chỉ 2 ảnh hợp lệ được trả về
            self.assertEqual(len(results), 2)
        finally:
            for p in valid_paths:
                os.unlink(p)


if __name__ == "__main__":
    unittest.main()
