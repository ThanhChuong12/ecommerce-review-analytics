"""Transfer Learning baseline for defect detection (damaged box classification).

Supports two backbone options:
  - ResNet50   — higher accuracy, slower inference
  - MobileNetV3-Large — lightweight, faster inference (< 50ms)

Both backbones are pretrained on ImageNet and fine-tuned by replacing
the final classification head to output 4 classes:
  intact | damaged | wrong_item | irrelevant

Label schema mirrors Review.label in the Node.js DB model.

Usage
-----
>>> from ai_engine.models.image_baseline import ImageBaselineModel
>>> model = ImageBaselineModel(backbone="resnet50")
>>> model.fit("image_labeling/data/labeled", epochs=10)
>>> result = model.predict("path/to/image.jpg")
>>> model.save("ai_engine/models/weights/resnet50_defect.pt")
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
# pyrefly: ignore [missing-import]
from torchvision import datasets, models, transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# 4 nhãn khớp với schema trong DB (Review.label)
CLASS_NAMES = ["intact", "damaged", "wrong_item", "irrelevant"]
NUM_CLASSES = len(CLASS_NAMES)

# ImageNet mean/std dùng để normalize — bắt buộc khi dùng pretrained backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Kích thước ảnh input chuẩn của cả 2 backbone
IMAGE_SIZE = 224


def _build_transforms(is_train: bool) -> transforms.Compose:
    """Tạo pipeline transform ảnh cho train hoặc inference.

    Train: augmentation mạnh để giảm overfitting.
    Eval/Inference: chỉ resize + center crop + normalize.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _build_backbone(backbone: str) -> Tuple[nn.Module, int]:
    """Load pretrained backbone và trả về (model, số feature của layer cuối).

    Chiến lược fine-tuning 2 giai đoạn:
      - Freeze toàn bộ backbone, chỉ train head (epoch đầu).
      - Unfreeze last block để backbone adapt domain (epoch sau).
    Caller tự unfreeze qua _unfreeze_last_block() sau vài epoch.
    """
    if backbone == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in net.parameters():
            param.requires_grad = False
        in_features = net.fc.in_features  # 2048
        net.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, NUM_CLASSES),
        )
        return net, in_features

    if backbone == "mobilenet_v3":
        net = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        # Freeze TOÀN BỘ features trước
        for param in net.features.parameters():
            param.requires_grad = False
        # FIX #1: Đọc in_features từ classifier[0].in_features (960 — output của features)
        # classifier[0] là Linear(960, 1280), [1] Hardswish, [2] Dropout, [3] Linear(1280,1000)
        in_features = net.classifier[0].in_features  # 960
        # FIX #2: Thay TOÀN BỘ classifier block (không chỉ [-1])
        # Thêm bottleneck 512 để có thêm capacity cho 4-class head
        net.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.4),
            nn.Linear(512, NUM_CLASSES),
        )
        # FIX #2 (cont): Đảm bảo classifier luôn trainable
        for param in net.classifier.parameters():
            param.requires_grad = True
        return net, in_features

    raise ValueError(f"Backbone khong hop le: '{backbone}'. Chon 'resnet50' hoac 'mobilenet_v3'.")


def _unfreeze_last_block(net: nn.Module, backbone: str) -> list:
    """Unfreeze last conv block de model adapt voi domain anh san pham.

    Returns danh sach params cua backbone block de dung differential LR.
    Backbone params dung LR nho hon head 10 lan.
    """
    backbone_params = []
    if backbone == "resnet50":
        # Unfreeze layer4 (last residual block)
        for param in net.layer4.parameters():  # type: ignore[attr-defined]
            param.requires_grad = True
            backbone_params.append(param)
        logger.info("Unfroze ResNet50 layer4 (%d param groups)", len(backbone_params))
    elif backbone == "mobilenet_v3":
        # FIX #3: Dùng index slice thay vì .children() để unfreeze đúng block
        # MobileNetV3-Large features có 17 blocks (index 0-16).
        # Block 16 là expansion Conv (1×1 → 960), 13-15 là InvertedResidual cuối.
        # Dùng net.features[13:] để unfreeze chính xác 4 blocks cuối.
        features = net.features  # type: ignore[attr-defined]
        for block in features[13:]:  # type: ignore[index]
            for param in block.parameters():
                param.requires_grad = True
                backbone_params.append(param)
        logger.info("Unfroze MobileNetV3 features[13:] (%d params)", len(backbone_params))
    return backbone_params


class ImageBaselineModel:
    """Transfer Learning model nhận diện tình trạng hộp sản phẩm.

    Attributes:
        backbone (str): 'resnet50' hoặc 'mobilenet_v3'.
        device (torch.device): CPU hoặc CUDA — tự phát hiện.
        model (nn.Module): Pretrained backbone với custom head.
        class_names (list[str]): Danh sách nhãn theo đúng thứ tự output.
    """

    def __init__(
        self,
        backbone: Literal["resnet50", "mobilenet_v3"] = "resnet50",
        device: Optional[str] = None,
    ) -> None:
        self.backbone = backbone
        # Tự chọn GPU nếu có, fallback về CPU
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = CLASS_NAMES
        self.model: Optional[nn.Module] = None
        logger.info("ImageBaselineModel khởi tạo — backbone=%s | device=%s", backbone, self.device)

    def _get_model(self) -> nn.Module:
        """Khởi tạo model nếu chưa có, load lên device."""
        if self.model is None:
            net, _ = _build_backbone(self.backbone)
            self.model = net.to(self.device)
        return self.model

    def fit(
        self,
        data_dir: str,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        val_split: float = 0.2,
        patience: int = 3,
        subset_ratio: float = 1.0,
    ) -> "ImageBaselineModel":
        """Fine-tune backbone trên dữ liệu ảnh đã gán nhãn.

        Cấu trúc thư mục data_dir phải theo ImageFolder format:
          data_dir/
            intact/      *.jpg ...
            damaged/     *.jpg ...
            wrong_item/  *.jpg ...
            irrelevant/  *.jpg ...

        Args:
            data_dir: Đường dẫn đến thư mục chứa các subfolder theo nhãn.
            epochs: Số epoch tối đa. Early stopping có thể dừng sớm hơn.
            batch_size: Kích thước mini-batch.
            lr: Learning rate cho optimizer Adam.
            val_split: Tỉ lệ dữ liệu dùng làm validation (0.2 = 20%).
            patience: Số epoch chờ nếu val_loss không cải thiện trước khi dừng.
            subset_ratio: Tỉ lệ data train dùng để train (0 < x <= 1.0).
                          Dùng < 1.0 để giảm thời gian train trên CPU.
                          Val set LUÔN dùng toàn bộ để evaluate công bằng.
                          Ví dụ: 0.35 → giảm ~65% thời gian, train ~2h thay vì ~6h.

        Returns:
            self (để chain method).
        """
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.metrics import classification_report as sk_report

        # --- FIX: 2 dataset RIÊNG BIỆT → tránh ghi đè transform ---
        train_dataset = datasets.ImageFolder(
            root=data_dir,
            transform=_build_transforms(is_train=True),
        )
        val_dataset = datasets.ImageFolder(
            root=data_dir,
            transform=_build_transforms(is_train=False),
        )

        self.class_names = train_dataset.classes
        logger.info("Dataset: %d images | Classes: %s", len(train_dataset), train_dataset.classes)

        # --- FIX: StratifiedShuffleSplit → đảm bảo tỉ lệ class giống nhau ---
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
        all_targets = train_dataset.targets
        train_idx, val_idx = next(sss.split(range(len(train_dataset)), all_targets))

        # --- subset_ratio: giảm train set để tiết kiệm thời gian trên CPU ---
        # Val set KHÔNG bị ảnh hưởng → evaluate vẫn công bằng
        if 0.0 < subset_ratio < 1.0:
            sss_sub = StratifiedShuffleSplit(
                n_splits=1, test_size=(1.0 - subset_ratio), random_state=42
            )
            sub_targets = [all_targets[i] for i in train_idx]
            keep_local, _ = next(sss_sub.split(range(len(train_idx)), sub_targets))
            train_idx = [train_idx[i] for i in keep_local]
            logger.info(
                "subset_ratio=%.2f → train set giảm còn %d ảnh (val vẫn %d ảnh)",
                subset_ratio, len(train_idx), len(val_idx),
            )

        train_set = Subset(train_dataset, train_idx)  # augmented transforms
        val_set = Subset(val_dataset, val_idx)         # eval transforms (riêng biệt!)

        n_train = len(train_idx)
        n_val = len(val_idx)

        # WeightedRandomSampler: cân bằng class imbalance khi train
        train_targets = [all_targets[i] for i in train_idx]
        class_counts = torch.bincount(torch.tensor(train_targets), minlength=NUM_CLASSES).float()
        sample_class_weights = 1.0 / class_counts.clamp(min=1)
        sample_weights = sample_class_weights[torch.tensor(train_targets)]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        logger.info("Class counts (train): %s", dict(zip(train_dataset.classes, class_counts.int().tolist())))

        # --- FIX: class weights cho loss → phạt nặng khi đoán sai class nhỏ ---
        loss_weights = 1.0 / class_counts.clamp(min=1)
        loss_weights = loss_weights / loss_weights.sum() * NUM_CLASSES
        logger.info("Loss weights: %s",
                    {c: round(w, 3) for c, w in zip(train_dataset.classes, loss_weights.tolist())})

        # --- FIX: pin_memory và num_workers theo platform ---
        use_pin = torch.cuda.is_available()
        n_workers = 0 if os.name == 'nt' else 2

        train_loader = DataLoader(
            train_set, batch_size=batch_size, sampler=sampler,
            num_workers=n_workers, pin_memory=use_pin,
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=n_workers, pin_memory=use_pin,
        )

        net = self._get_model()

        # Giai đoạn 1: chỉ train head (backbone vẫn frozen)
        head_params = list(filter(lambda p: p.requires_grad, net.parameters()))
        optimizer = torch.optim.Adam(head_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        # Train criterion: có class weights để penalize sai class thiểu số
        # label_smoothing=0.05 (giảm từ 0.1) — phù hợp hơn cho dataset nhỏ
        criterion = nn.CrossEntropyLoss(
            weight=loss_weights.to(self.device), label_smoothing=0.05
        )
        # FIX #4: Val criterion KHÔNG có class weights → early stopping công bằng
        # Val set có phân phối tự nhiên; weighted loss làm early stopping bias
        val_criterion = nn.CrossEntropyLoss()

        backbone_unfrozen = False
        best_val_loss = float("inf")
        best_val_f1 = 0.0
        epochs_no_improve = 0
        # Tăng patience: val_loss với imbalanced data không đơn điệu
        # Override patience nếu muốn dùng giá trị từ argument
        effective_patience = max(patience, 5)
        import tempfile
        best_weights_path = os.path.join(tempfile.gettempdir(), f"{self.backbone}_best.pt")

        for epoch in range(1, epochs + 1):
            # Giai đoạn 2: unfreeze last block sau epoch 3
            if not backbone_unfrozen and epoch > 3:
                backbone_params = _unfreeze_last_block(net, self.backbone)
                if backbone_params:
                    optimizer.add_param_group({"params": backbone_params, "lr": lr / 10})
                    backbone_unfrozen = True
                    logger.info("Epoch %d: backbone unfrozen, differential LR=%.2e", epoch, lr / 10)

            net.train()
            train_loss, train_correct = 0.0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = net(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                # FIX: Gradient clipping — tránh explosion khi unfreeze backbone
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()

            net.eval()
            val_loss, val_correct = 0.0, 0
            val_all_preds: list = []
            val_all_labels: list = []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    outputs = net(imgs)
                    # FIX #4: Dùng val_criterion (không weight) cho early stopping
                    loss = val_criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)
                    preds = outputs.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    val_all_preds.extend(preds.cpu().tolist())
                    val_all_labels.extend(labels.cpu().tolist())

            from sklearn.metrics import f1_score
            train_acc = train_correct / n_train
            val_acc = val_correct / n_val
            val_loss_avg = val_loss / n_val
            # Tính Macro-F1 để monitor class-imbalanced performance tốt hơn
            val_macro_f1 = f1_score(val_all_labels, val_all_preds, average="macro", zero_division=0)

            logger.info(
                "Epoch %02d/%02d — train_acc=%.4f | val_acc=%.4f | val_loss=%.4f | val_macro_f1=%.4f",
                epoch, epochs, train_acc, val_acc, val_loss_avg, val_macro_f1,
            )

            scheduler.step(val_loss_avg)

            # FIX: Early stopping theo val_loss (unweighted) và checkpoint khi F1 tốt nhất
            improved = False
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                improved = True
            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                improved = True

            if improved:
                epochs_no_improve = 0
                torch.save(net.state_dict(), best_weights_path)
                logger.info("  ✓ Checkpoint saved (val_loss=%.4f | val_f1=%.4f)",
                            best_val_loss, best_val_f1)
            else:
                epochs_no_improve += 1
                logger.info("  No improvement (%d/%d)", epochs_no_improve, effective_patience)

            if epochs_no_improve >= effective_patience:
                logger.info("Early stopping at epoch %d (patience=%d).", epoch, effective_patience)
                break

        # Restore best weights từ checkpoint
        net.load_state_dict(torch.load(best_weights_path, map_location=self.device))

        # --- Final evaluation trên VAL SET (không bị data leakage) ---
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION — val set only (%d images):", n_val)
        net.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(self.device)
                preds = net(imgs).argmax(1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        report_str = sk_report(all_labels, all_preds, target_names=self.class_names)
        report_dict = sk_report(all_labels, all_preds, target_names=self.class_names, output_dict=True)
        logger.info("\n%s", report_str)
        logger.info(
            "Val Accuracy: %.4f | Val Macro-F1: %.4f",
            report_dict["accuracy"], report_dict["macro avg"]["f1-score"],
        )
        logger.info("=" * 60)

        # Lưu val report để train script dùng (tránh evaluate lại trên full data)
        self._val_report = report_dict

        logger.info("Training hoàn tất. Best val_loss=%.4f", best_val_loss)
        return self

    def predict(self, image_path: str) -> Dict[str, object]:
        """Dự đoán nhãn cho một ảnh đơn lẻ.

        Args:
            image_path: Đường dẫn file ảnh (.jpg/.png).

        Returns:
            dict với keys:
              - label (str): Nhãn dự đoán, vd "damaged".
              - confidence (float): Xác suất của nhãn được chọn (0–1).
              - probabilities (dict): Xác suất đầy đủ cho cả 4 nhãn.
              - inference_ms (float): Thời gian inference tính bằng milli-giây.
        """
        from PIL import Image

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

        net = self._get_model()
        net.eval()

        transform = _build_transforms(is_train=False)
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = net(tensor)                           # (1, 4)
            probs = torch.softmax(logits, dim=1)[0]       # (4,)
        inference_ms = (time.perf_counter() - t0) * 1000

        pred_idx = probs.argmax().item()
        return {
            "label": self.class_names[pred_idx],
            "confidence": round(probs[pred_idx].item(), 4),
            "probabilities": {
                name: round(probs[i].item(), 4)
                for i, name in enumerate(self.class_names)
            },
            "inference_ms": round(inference_ms, 2),
        }

    def predict_batch(self, image_paths: list[str], batch_size: int = 32) -> list[Dict]:
        """Dự đoán cho nhiều ảnh cùng lúc — hiệu quả hơn gọi predict() từng cái.

        Args:
            image_paths: Danh sách đường dẫn ảnh.
            batch_size: Số ảnh xử lý mỗi lần forward pass.

        Returns:
            Danh sách dict kết quả, cùng thứ tự với image_paths.
        """
        from PIL import Image

        net = self._get_model()
        net.eval()
        transform = _build_transforms(is_train=False)

        results = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i: i + batch_size]
            tensors = []
            valid_paths = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(transform(img))
                    valid_paths.append(p)
                except Exception as e:
                    logger.warning("Bỏ qua ảnh lỗi %s: %s", p, e)

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors).to(self.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = net(batch_tensor)
                probs = torch.softmax(logits, dim=1)
            inference_ms = (time.perf_counter() - t0) * 1000 / len(tensors)

            for j, path in enumerate(valid_paths):
                pred_idx = probs[j].argmax().item()
                results.append({
                    "image_path": path,
                    "label": self.class_names[pred_idx],
                    "confidence": round(probs[j][pred_idx].item(), 4),
                    "probabilities": {
                        name: round(probs[j][k].item(), 4)
                        for k, name in enumerate(self.class_names)
                    },
                    "inference_ms": round(inference_ms, 2),
                })

        return results

    def evaluate(self, data_dir: str, batch_size: int = 32) -> Dict[str, float]:
        """Đánh giá model trên tập test (accuracy, per-class accuracy).

        Args:
            data_dir: Thư mục ảnh dạng ImageFolder (subfolder theo nhãn).
            batch_size: Batch size cho DataLoader.

        Returns:
            dict chứa overall_accuracy và per-class accuracy.
        """
        from sklearn.metrics import classification_report

        dataset = datasets.ImageFolder(
            root=data_dir,
            transform=_build_transforms(is_train=False),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        net = self._get_model()
        net.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                outputs = net(imgs)
                preds = outputs.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        report = classification_report(
            all_labels, all_preds,
            target_names=dataset.classes,
            output_dict=True,
        )
        accuracy = report["accuracy"]
        logger.info("Evaluation — Overall Accuracy: %.4f", accuracy)
        logger.info("\n%s", classification_report(all_labels, all_preds, target_names=dataset.classes))
        return report

    def save(self, filepath: str) -> None:
        """Lưu state dict + metadata ra file .pt.

        Lưu thêm backbone name và class_names để load lại đúng cấu hình.
        """
        if self.model is None:
            raise RuntimeError("Model chưa được train. Gọi .fit() trước.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            "backbone": self.backbone,
            "class_names": self.class_names,
            "state_dict": self.model.state_dict(),
        }, filepath)
        logger.info("Model saved → %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "ImageBaselineModel":
        """Tải model từ file .pt đã lưu bằng .save().

        Args:
            filepath: Đường dẫn file .pt.

        Returns:
            ImageBaselineModel đã sẵn sàng cho inference.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file model: {filepath}")

        checkpoint = torch.load(filepath, map_location="cpu")
        backbone = checkpoint["backbone"]

        instance = cls(backbone=backbone)
        net, _ = _build_backbone(backbone)
        net.load_state_dict(checkpoint["state_dict"])
        instance.model = net.to(instance.device)
        instance.class_names = checkpoint["class_names"]

        logger.info("Model loaded ← %s (backbone=%s)", filepath, backbone)
        return instance
