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

# Import Augmentation Pipeline đã viết ở Bước trước
from ai_engine.image_processing.augmentation.transforms import get_defect_transforms, get_normal_transforms

class ProductDefectDataset(Dataset):
    """
    Custom PyTorch Dataset cho bài toán phân loại hàng lỗi.
    Áp dụng Image Augmentation riêng cho lớp defect để khắc phục mất cân bằng dữ liệu.
    """
    def __init__(self, data_dir: str, is_train: bool = True):
        """
        Args:
            data_dir (str): Đường dẫn đến thư mục chứa 2 thư mục con 'defect' và 'no-defect'.
            is_train (bool): Nếu True, áp dụng augmentation cho lớp defect.
                             Nếu False (Validation/Test), chỉ áp dụng resize và normalize.
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


def get_dataloaders(data_dir: str, batch_size: int = 32, val_split: float = 0.2, seed: int = 42):
    """
    Tao DataLoaders cho Training va Validation.
    
    Tao 2 Dataset instances rieng biet de tranh loi shared-state:
    - train_dataset (is_train=True): ap dung augmentation cho defect
    - val_dataset (is_train=False): chi resize + normalize
    """
    # Tao 1 dataset tam de lay danh sach indices
    temp_dataset = ProductDefectDataset(data_dir=data_dir, is_train=True)
    total_size = len(temp_dataset)
    
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Tao indices va shuffle chung
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Tao 2 dataset instances RIENG BIET
    train_dataset = ProductDefectDataset(data_dir=data_dir, is_train=True)
    val_dataset = ProductDefectDataset(data_dir=data_dir, is_train=False)
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def get_resnet50_model(num_classes: int = 2) -> nn.Module:
    """Tạo mô hình ResNet50 pretrained trên ImageNet và đổi layer cuối cho binary classification."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze các layer dưới (optional, nhưng giúp train nhanh hơn với dữ liệu nhỏ)
    # for param in model.parameters():
    #     param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def detect_defect_resnet(image_path: str) -> dict:
    """Nhận diện hàng lỗi bằng ResNet. Returns dict với label và confidence."""
    raise NotImplementedError("TODO: Implement ResNet inference here")


def detect_defect_mobilenet(image_path: str) -> dict:
    """Nhận diện hàng lỗi bằng MobileNet. Returns dict với label và confidence."""
    raise NotImplementedError("TODO: Implement MobileNet inference here")
