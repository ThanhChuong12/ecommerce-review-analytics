"""
defect_detection.py
-------------------
Product defect/damage detection using Transfer Learning:
  - ResNet50 (stronger, more accurate backbone) — v2 with MLP head + FocalLoss
  - MobileNetV3-Large (lighter, suitable for fast inference, ~50ms/image)

Training pipeline v2:
  - Oversampling defect class (15x) to balance data from 1:37 to 1:2.5
  - FocalLoss (gamma=2.0) + class weights to focus on hard samples
  - MLP head (Linear→BN→ReLU→Dropout→Linear) replacing simple Linear layer
  - Freeze backbone, train only FC head (4.28% params)
  - Early stopping based on Defect F1 (patience=5)

Inference:
  - detect_defect_resnet(): ResNet50 — binary (defect/no-defect), with threshold tuning
  - detect_defect_mobilenet(): MobileNetV3 — 4 classes (intact/damaged/wrong_item/irrelevant)
  - detect_defect_mobilenet_batch(): MobileNetV3 batch inference
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

# Default path to the model checkpoint (v2)
_DEFAULT_RESNET_WEIGHTS = os.getenv(
    "RESNET_WEIGHTS_PATH",
    "ai_engine/models/resnet50_defect_gpu_best.pth",
)

# Default threshold to balance Recall vs Precision.
# Tuning result (tune_threshold.py, val set):
#   threshold=0.525 -> F1=0.8042, Recall=0.8042, Precision=0.8042
_DEFAULT_THRESHOLD = float(os.getenv("DEFECT_THRESHOLD", "0.525"))

# Cache the model to avoid reloading on each inference call
_resnet_model_cache = None
_resnet_model_path_cache = None


class ProductDefectDataset(Dataset):
    """
    Custom PyTorch Dataset for the product defect classification task.
    Applies separate Image Augmentation to the defect class to address data imbalance.
    Supports oversampling the defect class to balance class ratios.
    """
    def __init__(self, data_dir: str, is_train: bool = True, oversample_defect: int = 1):
        """
        Args:
            data_dir (str): Paths to data directories, separated by semicolons (;).
                            Example: 'image_labeling/data/labeled;image_labeling/new_data/labeled'
            is_train (bool): If True, applies augmentation to the defect class.
                             If False (Validation/Test), only applies resize and normalization.
            oversample_defect (int): Number of times to repeat each defect image in the dataset.
                                     Example: oversample_defect=10 will turn 81 images into 810.
                                     Only applied when is_train=True. Default = 1 (no oversampling).
        """
        self.data_dir = data_dir
        self.is_train = is_train

        self.image_paths = []
        self.labels = []

        # Load pipelines
        self.defect_transform = get_defect_transforms()
        self.normal_transform = get_normal_transforms()

        # Mapping labels: no-defect/intact = 0, defect/damaged = 1
        class_mapping = {
            "no-defect": 0, "defect": 1,
            "intact": 0, "damaged": 1
        }

        # Supports passing multiple directories separated by semicolons
        data_dirs = [Path(d.strip()) for d in data_dir.split(";") if d.strip()]

        for class_name, label in class_mapping.items():
            for d_dir in data_dirs:
                class_dir = d_dir / class_name
                if class_dir.exists():
                    for ext in ["*.jpg", "*.jpeg", "*.png"]:
                        for img_path in class_dir.glob(ext):
                            # Oversample: Repeat the defect image multiple times (different augmentations will be applied)
                            repeat = oversample_defect if (is_train and label == 1) else 1
                            for _ in range(repeat):
                                self.image_paths.append(img_path)
                                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = str(self.image_paths[idx])
        label = self.labels[idx]

        # Read image using OpenCV instead of PIL because Albumentations takes Numpy Array input
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Cannot read image: {path}")

        # OpenCV reads BGR by default -> Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation based on label
        # ONLY apply augmentation during training AND when the image is a defect (label == 1)
        if self.is_train and label == 1:
            augmented = self.defect_transform(image=image)
        else:
            # Otherwise (normal image, or during Validation/Test), only Resize and Normalize
            augmented = self.normal_transform(image=image)

        image_tensor = augmented["image"]
        return image_tensor, torch.tensor(label, dtype=torch.long)


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced datasets.
    Instead of CrossEntropyLoss which gives equal weight to all samples,
    Focal Loss reduces the weight of "easy" samples (already correctly predicted)
    and focuses on "hard" samples (defects misclassified as no-defect).

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    def __init__(self, alpha=None, gamma=2.0):
        """
        Args:
            alpha (Tensor): Weight for each class. E.g. [0.25, 0.75] for [no-defect, defect].
            gamma (float): Focusing parameter. gamma=0 is equivalent to CrossEntropyLoss.
                           Larger gamma down-weights easy samples more.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct prediction
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def get_dataloaders(data_dir: str, batch_size: int = 32, val_split: float = 0.2,
                    seed: int = 42, oversample_defect: int = 1,
                    train_dir: str = None, val_dir: str = None):
    """
    Create DataLoaders for Training and Validation.

    Uses Stratified Split to ensure even distribution of defect/no-defect classes
    between train and val sets — preventing val set from having too few defect items.

    If data_dir contains 'train' and 'val' subfolders, or train_dir and val_dir are provided,
    it loads directly from those directories without splitting.
    """
    path = Path(data_dir)
    has_sub_splits = (path / "train").exists() and (path / "val").exists()

    if has_sub_splits or (train_dir and val_dir):
        t_dir = train_dir or str(path / "train")
        v_dir = val_dir or str(path / "val")

        train_dataset = ProductDefectDataset(data_dir=t_dir, is_train=True, oversample_defect=oversample_defect)
        val_dataset = ProductDefectDataset(data_dir=v_dir, is_train=False, oversample_defect=1)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader

    # Create base dataset (no oversampling) to get raw labels for stratified split
    base_dataset = ProductDefectDataset(data_dir=data_dir, is_train=False, oversample_defect=1)
    base_size = len(base_dataset)

    # ✅ Stratified split: ensure identical class ratios in both train and val
    train_indices, val_indices = train_test_split(
        list(range(base_size)),
        test_size=val_split,
        stratify=base_dataset.labels,
        random_state=seed,
    )

    # Val dataset: use indices directly, no oversampling
    val_dataset = ProductDefectDataset(data_dir=data_dir, is_train=False, oversample_defect=1)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    # Train dataset: create new dataset with oversampling enabled
    train_dataset = ProductDefectDataset(data_dir=data_dir, is_train=True, oversample_defect=oversample_defect)

    if oversample_defect > 1:
        # Remap indices: keep only images belonging to original train_indices (including oversampled copies)
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
                       dropout_rate: float = 0.5, pretrained: bool = True) -> nn.Module:
    """
    Create ResNet50 model and replace the final layer for binary classification.

    Args:
        num_classes (int): Number of output classes.
        freeze_backbone (bool): If True, freezes all CNN layers (trains only the FC layer).
                                Helps prevent overfitting with small datasets (e.g., 81 defect images).
        dropout_rate (float): Dropout rate in classification head (default 0.5).
                              Used to prevent overfitting when only training the head.
        pretrained (bool): If True, loads ImageNet weights. If False, initializes with random weights.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        # Step 1: Freeze the ENTIRE backbone
        for param in model.parameters():
            param.requires_grad = False

    # Step 2: Replace the FC layer with a stronger classification head
    # Linear -> BN -> ReLU -> Dropout -> Linear
    # Helps model learn more from ResNet features, while preventing overfitting
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),       # Reduce dimension from 2048 -> 512
        nn.BatchNorm1d(512),            # Stabilize training
        nn.ReLU(inplace=True),          # Non-linearity
        nn.Dropout(p=dropout_rate),     # Regularization to prevent overfitting
        nn.Linear(512, num_classes),    # Final output
    )

    # Step 3: Ensure FC head is ALWAYS trainable (even when freeze_backbone=True)
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def _load_resnet_model(model_path: str = None, device: torch.device = None) -> nn.Module:
    """
    Load trained ResNet50 model from checkpoint. Uses caching to avoid reloading.

    Args:
        model_path (str): Path to the .pth file. Defaults to _DEFAULT_RESNET_WEIGHTS.
        device (torch.device): Device to load the model on. Autodetected if None.

    Returns:
        nn.Module: Loaded model in eval() mode.
    """
    global _resnet_model_cache, _resnet_model_path_cache

    if model_path is None:
        model_path = _DEFAULT_RESNET_WEIGHTS

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Return cached model if path matches
    if _resnet_model_cache is not None and _resnet_model_path_cache == model_path:
        return _resnet_model_cache

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{model_path}'. "
            "Run: python scripts/train_defect_model.py"
        )

    # Initialize architecture (must match training: freeze=True, dropout=0.5, num_classes=2)
    # Set pretrained=False to avoid downloading ImageNet weights from the internet
    model = get_resnet50_model(num_classes=2, freeze_backbone=True, dropout_rate=0.5, pretrained=False)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Log checkpoint information
    epoch = checkpoint.get("epoch", "?")
    f1 = checkpoint.get("defect_f1", "?")
    recall = checkpoint.get("defect_recall", "?")
    _logger.info(
        f"Loaded ResNet50 defect model from '{model_path}' "
        f"(epoch={epoch}, defect_f1={f1:.3f}, defect_recall={recall:.3f})"
    )

    # Save to cache
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
    Detect product defects using ResNet50 (v2 — MLP head + FocalLoss).

    Uses threshold tuning to balance Recall vs Precision:
    - Lower threshold (< 0.5) → catch more defects (higher Recall, lower Precision)
    - Higher threshold (> 0.5) → fewer false positives (higher Precision, lower Recall)

    Args:
        image_path (str): Path to the image to classify.
        model_path (str): Path to checkpoint (.pth). Defaults to _DEFAULT_RESNET_WEIGHTS.
        threshold (float): Probability threshold to decide "defect". Defaults to 0.525.
        device (torch.device): Device to run inference on. Autodetected if None.

    Returns:
        dict: {
            "label": "defect" | "no-defect",
            "confidence": float (0.0 – 1.0),  # probability of the selected class
            "defect_probability": float,       # raw P(defect) from softmax
            "threshold_used": float,
            "model_path": str,
        }

    Raises:
        FileNotFoundError: If model_path or image_path does not exist.
        ValueError: If the image cannot be read.
    """
    if not _CV2_AVAILABLE:
        raise ImportError("cv2 (opencv-python) is not installed. Run: pip install opencv-python")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image does not exist: '{image_path}'")

    if threshold is None:
        threshold = _DEFAULT_THRESHOLD

    if model_path is None:
        model_path = _DEFAULT_RESNET_WEIGHTS

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model (using cache) ---
    model = _load_resnet_model(model_path=model_path, device=device)

    # --- Image Preprocessing ---
    # Use normal_transform (only resize + normalize, no augmentations)
    # pyrefly: ignore [missing-import]
    import albumentations as A
    # pyrefly: ignore [missing-import]
    from albumentations.pytorch import ToTensorV2

    preprocess = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Cannot read image: '{image_path}'. Check the file path.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transformed = preprocess(image=image_rgb)
    image_tensor = transformed["image"].unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # --- Inference ---
    with torch.no_grad():
        logits = model(image_tensor)                     # [1, 2]
        probs = torch.softmax(logits, dim=1)[0]          # [2]
        defect_prob = probs[1].item()                    # P(defect)

    # --- Determine label based on threshold ---
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


def detect_defect_resnet_batch(
    image_paths: list,
    model_path: str = None,
    threshold: float = None,
    batch_size: int = 32,
    device: torch.device = None,
) -> list:
    if not _CV2_AVAILABLE:
        raise ImportError("cv2 (opencv-python) is not installed.")

    if threshold is None:
        threshold = _DEFAULT_THRESHOLD

    if model_path is None:
        model_path = _DEFAULT_RESNET_WEIGHTS

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_resnet_model(model_path=model_path, device=device)

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    preprocess = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        tensors = []
        valid_paths = []
        for p in batch_paths:
            if not os.path.exists(p):
                continue
            image_bgr = cv2.imread(p)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            transformed = preprocess(image=image_rgb)
            tensors.append(transformed["image"])
            valid_paths.append(p)

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)

        for j, p in enumerate(valid_paths):
            defect_prob = probs[j][1].item()
            is_defect = defect_prob >= threshold
            label = "defect" if is_defect else "no-defect"
            confidence = defect_prob if is_defect else (1.0 - defect_prob)
            results.append({
                "image_path": p,
                "label": label,
                "confidence": round(confidence, 4),
                "defect_probability": round(defect_prob, 4),
                "threshold_used": threshold,
                "model_path": model_path,
            })

    return results

# =============================================================================
# MobileNetV3 — Production Inference
# =============================================================================

_DEFAULT_MOBILENET_WEIGHTS = os.getenv(
    "MOBILENET_WEIGHTS_PATH",
    "ai_engine/models/weights/mobilenet_v3_defect.pt",
)

# Singleton cache — load model only once
_mobilenet_model_cache = None
_mobilenet_model_path_cache = None


def _load_mobilenet_model(model_path: str = None):
    """Load MobileNetV3 model with singleton cache.

    Model is only reloaded when model_path changes.
    """
    global _mobilenet_model_cache, _mobilenet_model_path_cache

    if model_path is None:
        model_path = _DEFAULT_MOBILENET_WEIGHTS

    # Return cache if loading the same path
    if _mobilenet_model_cache is not None and _mobilenet_model_path_cache == model_path:
        return _mobilenet_model_cache

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{model_path}'. "
            "Run: python scripts/train_image_baseline.py --backbone mobilenet_v3"
        )

    from ai_engine.models.image_baseline import ImageBaselineModel

    model = ImageBaselineModel.load(model_path)
    
    # Force threshold to 0.85 (default 0.5) to reduce False Positives
    # Avoids model labeling normal product images as 'defect' too easily
    model.threshold = 0.85
    
    _logger.info(
        "Loaded MobileNetV3 defect model from '%s' (classes=%s, forced_threshold=0.85)",
        model_path, model.class_names,
    )

    # Save to cache
    _mobilenet_model_cache = model
    _mobilenet_model_path_cache = model_path

    return model


def detect_defect_mobilenet(
    image_path: str,
    model_path: str = None,
) -> dict:
    """Detect product condition using MobileNetV3.

    Uses Transfer Learning (MobileNetV3-Large pretrained on ImageNet)
    fine-tuned on the e-commerce product review image dataset.

    Args:
        image_path (str): Path to the image to classify.
        model_path (str): Path to the weights file (.pt).
                          Defaults to: ai_engine/models/weights/mobilenet_v3_defect.pt.

    Returns:
        dict: {
            "label": "intact" | "damaged" | "wrong_item" | "irrelevant",
            "confidence": float (0.0 – 1.0),
            "probabilities": {"intact": float, "damaged": float, ...},
            "inference_ms": float,
            "model_path": str,
        }

    Raises:
        FileNotFoundError: If model_path or image_path does not exist.
        ValueError: If the image cannot be read.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image does not exist: '{image_path}'")

    model = _load_mobilenet_model(model_path=model_path)
    result = model.predict(image_path)

    # Add model_path to output for traceability
    result["model_path"] = model_path or _DEFAULT_MOBILENET_WEIGHTS
    return result


def detect_defect_mobilenet_batch(
    image_paths: list,
    model_path: str = None,
    batch_size: int = 32,
) -> list:
    """Detect product condition for multiple images (batch inference).

    More efficient than calling detect_defect_mobilenet() for each image
    by grouping them into a single forward pass on GPU/CPU.

    Args:
        image_paths (list[str]): List of image paths.
        model_path (str): Path to the weights file (.pt).
        batch_size (int): Number of images processed per forward pass.

    Returns:
        list[dict]: List of results, in the same order as image_paths.
    """
    model = _load_mobilenet_model(model_path=model_path)
    results = model.predict_batch(image_paths, batch_size=batch_size)

    # Add model_path to each result
    used_path = model_path or _DEFAULT_MOBILENET_WEIGHTS
    for r in results:
        r["model_path"] = used_path
    return results


# Alias for demo_server.py (backward compatibility)
detect_defect_mobilenet_demo = detect_defect_mobilenet
