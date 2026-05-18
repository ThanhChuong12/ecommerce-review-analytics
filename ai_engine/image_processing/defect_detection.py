"""
defect_detection.py
-------------------
Nhận diện hàng móp méo / lỗi bằng Transfer Learning:
  - ResNet (backbone mạnh hơn, chính xác hơn)
  - MobileNet (nhẹ hơn, phù hợp inference nhanh)

TODO:
  - [ ] Load pretrained ResNet / MobileNet từ ai_engine/models/
  - [ ] Fine-tune trên dataset hàng lỗi (data/processed/)
  - [ ] Đánh giá Accuracy, Precision trên tập test
  - [ ] Export mô hình tốt nhất vào ai_engine/models/
"""

import os
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from sklearn.model_selection import train_test_split

# Import Augmentation Pipeline đã viết ở Bước trước
from ai_engine.image_processing.augmentation.transforms import get_defect_transforms, get_normal_transforms

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


def detect_defect_resnet(image_path: str) -> dict:
    """Nhận diện hàng lỗi bằng ResNet. Returns dict với label và confidence."""
    raise NotImplementedError("TODO: Implement ResNet inference here")


def detect_defect_mobilenet(image_path: str) -> dict:
    """Nhận diện hàng lỗi bằng MobileNet. Returns dict với label và confidence."""
    raise NotImplementedError("TODO: Implement MobileNet inference here")