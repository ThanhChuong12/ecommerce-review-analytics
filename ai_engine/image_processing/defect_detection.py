"""
defect_detection.py
-------------------
Nhận diện hàng móp méo / lỗi bằng Transfer Learning:
  - ResNet50 (backbone mạnh hơn, chính xác hơn) — v2 với MLP head + FocalLoss
  - MobileNet (nhẹ hơn, phù hợp inference nhanh)

Training pipeline v2:
  - Oversampling lớp defect (15x) để cân bằng dữ liệu 1:37 → 1:2.5
  - FocalLoss (gamma=2.0) + class weights để focus vào hard samples
  - MLP head (Linear→BN→ReLU→Dropout→Linear) thay cho Linear đơn giản
  - Freeze backbone, chỉ train FC head (4.28% params)
  - Early stopping theo Defect F1 (patience=5)

Inference:
  - detect_defect_resnet(): Inference với model đã train, hỗ trợ threshold tuning
"""

import logging
import os
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split

try:
    # pyrefly: ignore [missing-import]
    import cv2
    from ai_engine.image_processing.augmentation.transforms import (
        get_defect_transforms,
        get_normal_transforms,
    )
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

_logger = logging.getLogger(__name__)

# Đường dẫn mặc định tới model checkpoint (v2)
_DEFAULT_RESNET_WEIGHTS = os.getenv(
    "RESNET_WEIGHTS_PATH",
    "ai_engine/models/resnet50_defect.pth",
)

# Threshold mac dinh de can bang Recall vs Precision.
# Tuning result (tune_threshold.py, val set):
#   threshold=0.45 -> F1=0.619, Recall=0.684, Precision=0.565, FP=10
#   threshold=0.50 -> F1=0.615, Recall=0.632, Precision=0.600, FP=8   (original)
#   threshold=0.20 -> F1=0.431, Recall=0.737, Precision=0.304, FP=32  (high recall)
# Default 0.45 = best F1 while keeping recall >= 0.68.
_DEFAULT_THRESHOLD = float(os.getenv("DEFECT_THRESHOLD", "0.45"))

# Cache model để tránh load lại mỗi lần inference
_resnet_model_cache = None
_resnet_model_path_cache = None


class ProductDefectDataset(Dataset):
    """
    Custom PyTorch Dataset cho bài toán phân loại hàng lỗi.
    Áp dụng Image Augmentation riêng cho lớp defect để khắc phục mất cân bằng dữ liệu.
    Hỗ trợ Oversampling lớp defect để cân bằng tỷ lệ giữa 2 class.
    """
    def __init__(self, data_dir: str, is_train: bool = True, oversample_defect: int = 1):
        """
        Args:
            data_dir (str): Đường dẫn đến thư mục chứa 2 thư mục con 'defect' và 'no-defect'.
            is_train (bool): Nếu True, áp dụng augmentation cho lớp defect.
                             Nếu False (Validation/Test), chỉ áp dụng resize và normalize.
            oversample_defect (int): Số lần lặp lại mỗi ảnh defect trong dataset.
                                     Ví dụ: oversample_defect=10 sẽ biến 81 ảnh thành 810 ảnh.
                                     Chỉ áp dụng khi is_train=True. Mặc định = 1 (không oversample).
        """
        self.data_dir = Path(data_dir)
        self.is_train = is_train

        self.image_paths = []
        self.labels = []

        # Load pipelines
        self.defect_transform = get_defect_transforms()
        self.normal_transform = get_normal_transforms()

        # Mapping labels: no-defect = 0, defect = 1
        class_mapping = {"no-defect": 0, "defect": 1}

        for class_name, label in class_mapping.items():
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for img_path in class_dir.glob(ext):
                        # Oversample: Lặp lại ảnh defect nhiều lần (mỗi lần augment sẽ cho ảnh khác nhau)
                        repeat = oversample_defect if (is_train and label == 1) else 1
                        for _ in range(repeat):
                            self.image_paths.append(img_path)
                            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = str(self.image_paths[idx])
        label = self.labels[idx]

        # Đọc ảnh bằng OpenCV thay vì PIL vì Albumentations nhận đầu vào Numpy Array
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {path}")

        # OpenCV mặc định đọc ảnh BGR -> Chuyển sang RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Áp dụng Augmentation dựa vào label
        # CHỈ áp dụng augmentation khi đang Train VÀ ảnh đó là ảnh Lỗi (label == 1)
        if self.is_train and label == 1:
            augmented = self.defect_transform(image=image)
        else:
            # Ngược lại (ảnh bình thường, hoặc lúc Validation/Test) chỉ Resize và Normalize
            augmented = self.normal_transform(image=image)

        image_tensor = augmented["image"]
        return image_tensor, torch.tensor(label, dtype=torch.long)


class FocalLoss(nn.Module):
    """
    Focal Loss cho bai toan mat can bang du lieu.
    Thay vi CrossEntropyLoss trao trong so nhu nhau cho moi mau,
    Focal Loss giam trong so cua cac mau "de" (model da du doan dung)
    va tap trung vao cac mau "kho" (defect bi nham thanh no-defect).

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    def __init__(self, alpha=None, gamma=2.0):
        """
        Args:
            alpha (Tensor): Trong so cho tung class. Vi du: [0.25, 0.75] cho [no-defect, defect].
            gamma (float): He so tap trung. gamma=0 tuong duong CrossEntropyLoss.
                           gamma cang lon, cac mau de cang bi giam trong so.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # Xac suat du doan dung
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def get_dataloaders(data_dir: str, batch_size: int = 32, val_split: float = 0.2,
                    seed: int = 42, oversample_defect: int = 1):
    """
    Tao DataLoaders cho Training va Validation.

    Dung Stratified Split de dam bao ty le defect/no-defect dong deu
    giua train va val — tranh truong hop val set co qua it anh defect.

    - train_dataset (is_train=True): ap dung augmentation + oversampling cho defect
    - val_dataset (is_train=False): chi resize + normalize, KHONG oversample
    """
    # Tao base dataset (khong oversample) de lay labels goc cho stratified split
    base_dataset = ProductDefectDataset(data_dir=data_dir, is_train=False, oversample_defect=1)
    base_size = len(base_dataset)

    # ✅ Stratified split: dam bao ty le class giong nhau o ca train va val
    train_indices, val_indices = train_test_split(
        list(range(base_size)),
        test_size=val_split,
        stratify=base_dataset.labels,
        random_state=seed,
    )

    # Val dataset: dung indices truc tiep, khong oversample
    val_dataset = ProductDefectDataset(data_dir=data_dir, is_train=False, oversample_defect=1)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    # Train dataset: tao moi voi oversampling
    train_dataset = ProductDefectDataset(data_dir=data_dir, is_train=True, oversample_defect=oversample_defect)

    if oversample_defect > 1:
        # Map lai indices: chi giu nhung anh thuoc train_indices goc (bao gom ban sao oversample)
        base_train_paths = set(str(base_dataset.image_paths[i]) for i in train_indices)
        oversampled_train_indices = [
            i for i in range(len(train_dataset))
            if str(train_dataset.image_paths[i]) in base_train_paths
        ]
        train_subset = torch.utils.data.Subset(train_dataset, oversampled_train_indices)
    else:
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def get_resnet50_model(num_classes: int = 2, freeze_backbone: bool = False,
                       dropout_rate: float = 0.5) -> nn.Module:
    """
    Tao mo hinh ResNet50 pretrained tren ImageNet va doi layer cuoi cho binary classification.

    Args:
        num_classes (int): So luong class dau ra.
        freeze_backbone (bool): Neu True, dong bang tat ca cac layer CNN (chi train FC layer).
                                Giup chong overfitting khi du lieu it (vi du: 81 anh defect).
        dropout_rate (float): Dropout rate trong classification head (default 0.5).
                              Dung de chong overfitting khi chi train head.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        # Buoc 1: Dong bang TOAN BO backbone
        for param in model.parameters():
            param.requires_grad = False

    # Buoc 2: Thay FC layer bang classification head manh hon
    # Linear -> BN -> ReLU -> Dropout -> Linear
    # Giup model hoc duoc nhieu hon tu ResNet features, dong thoi chong overfitting
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),       # Giam chieu tu 2048 -> 512
        nn.BatchNorm1d(512),            # On dinh hoa training
        nn.ReLU(inplace=True),          # Non-linearity
        nn.Dropout(p=dropout_rate),     # Regularization de chong overfitting
        nn.Linear(512, num_classes),    # Dau ra cuoi cung
    )

    # Buoc 3: Dam bao FC head LUON duoc train (ke ca khi freeze_backbone=True)
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def _load_resnet_model(model_path: str = None, device: torch.device = None) -> nn.Module:
    """
    Load ResNet50 model đã train từ checkpoint. Dùng cache để tránh load lại.

    Args:
        model_path (str): Đường dẫn tới file .pth. Mặc định dùng _DEFAULT_RESNET_WEIGHTS.
        device (torch.device): Device để load model. Mặc định tự detect.

    Returns:
        nn.Module: Model đã load weights, ở chế độ eval().
    """
    global _resnet_model_cache, _resnet_model_path_cache

    if model_path is None:
        model_path = _DEFAULT_RESNET_WEIGHTS

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Trả về cache nếu đã load cùng file
    if _resnet_model_cache is not None and _resnet_model_path_cache == model_path:
        return _resnet_model_cache

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint không tìm thấy tại '{model_path}'. "
            "Chạy: python scripts/train_defect_model.py"
        )

    # Khởi tạo kiến trúc (phải khớp với lúc train: freeze=True, dropout=0.5, num_classes=2)
    model = get_resnet50_model(num_classes=2, freeze_backbone=True, dropout_rate=0.5)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Ghi log thông tin checkpoint
    epoch = checkpoint.get("epoch", "?")
    f1 = checkpoint.get("defect_f1", "?")
    recall = checkpoint.get("defect_recall", "?")
    _logger.info(
        f"Loaded ResNet50 defect model from '{model_path}' "
        f"(epoch={epoch}, defect_f1={f1:.3f}, defect_recall={recall:.3f})"
    )

    # Cache lại
    _resnet_model_cache = model
    _resnet_model_path_cache = model_path

    return model


def detect_defect_resnet(
    image_path: str,
    model_path: str = None,
    threshold: float = None,
    device: torch.device = None,
) -> dict:
    """
    Nhận diện hàng lỗi bằng ResNet50 (v2 — MLP head + FocalLoss).

    Dùng threshold tuning để cân bằng Recall vs Precision:
    - threshold thấp hơn 0.5 → bắt thêm defect (Recall cao hơn, Precision thấp hơn)
    - threshold cao hơn 0.5 → ít false positive hơn (Precision cao hơn, Recall thấp hơn)

    Args:
        image_path (str): Đường dẫn tới ảnh cần phân loại.
        model_path (str): Đường dẫn tới checkpoint (.pth). Mặc định: ai_engine/models/resnet50_defect.pth.
        threshold (float): Ngưỡng xác suất để quyết định "defect". Mặc định: 0.35 (ưu tiên Recall).
        device (torch.device): Device để chạy inference. Mặc định tự detect.

    Returns:
        dict: {
            "label": "defect" | "no-defect",
            "confidence": float (0.0 – 1.0),  # xác suất của class được chọn
            "defect_probability": float,       # P(defect) raw từ softmax
            "threshold_used": float,
            "model_path": str,
        }

    Raises:
        FileNotFoundError: Nếu model_path hoặc image_path không tồn tại.
        ValueError: Nếu ảnh không thể đọc được.
    """
    if not _CV2_AVAILABLE:
        raise ImportError("cv2 (opencv-python) không được cài. Chạy: pip install opencv-python")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Ảnh không tồn tại: '{image_path}'")

    if threshold is None:
        threshold = _DEFAULT_THRESHOLD

    if model_path is None:
        model_path = _DEFAULT_RESNET_WEIGHTS

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model (dùng cache) ---
    model = _load_resnet_model(model_path=model_path, device=device)

    # --- Tiền xử lý ảnh ---
    # Dùng normal_transform (chỉ resize + normalize, không augment)
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    preprocess = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Không thể đọc ảnh: '{image_path}'. Kiểm tra lại file.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transformed = preprocess(image=image_rgb)
    image_tensor = transformed["image"].unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # --- Inference ---
    with torch.no_grad():
        logits = model(image_tensor)                     # [1, 2]
        probs = torch.softmax(logits, dim=1)[0]          # [2]
        defect_prob = probs[1].item()                    # P(defect)

    # --- Quyết định nhãn theo threshold ---
    is_defect = defect_prob >= threshold
    label = "defect" if is_defect else "no-defect"
    confidence = defect_prob if is_defect else (1.0 - defect_prob)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "defect_probability": round(defect_prob, 4),
        "threshold_used": threshold,
        "model_path": model_path,
    }


def detect_defect_mobilenet(image_path: str) -> dict:
    """Nhận diện hàng lỗi bằng MobileNet. Returns dict với label và confidence."""
    raise NotImplementedError("TODO: Implement MobileNet inference here")


# =============================================================================
# DEMO ONLY — Xóa section này khi tích hợp vào production pipeline chính
# Dùng bởi: demo_server.py
# =============================================================================
_MOBILENET_WEIGHTS = os.getenv(
    "MOBILENET_WEIGHTS_PATH",
    "ai_engine/models/weights/mobilenet_v3_defect.pt",
)

_mobilenet_model = None


def _load_mobilenet_demo():
    global _mobilenet_model
    if _mobilenet_model is None:
        from ai_engine.models.image_baseline import ImageBaselineModel
        _mobilenet_model = ImageBaselineModel.load(_MOBILENET_WEIGHTS)
        _logger.info("MobileNetV3 defect model loaded.")
    return _mobilenet_model


def detect_defect_mobilenet_demo(image_path: str) -> dict:
    """[DEMO] Inference voi MobileNetV3 da train. Dung trong demo_server.py."""
    if not os.path.exists(_MOBILENET_WEIGHTS):
        raise RuntimeError(
            f"Weights not found at '{_MOBILENET_WEIGHTS}'. "
            "Run: python ai_engine/scripts/train_image_baseline.py"
        )
    return _load_mobilenet_demo().predict(image_path)
# =============================================================================
# END DEMO
# =============================================================================
