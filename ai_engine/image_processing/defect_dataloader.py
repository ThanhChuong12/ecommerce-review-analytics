# -*- coding: utf-8 -*-
"""
defect_dataloader.py
--------------------
Module thiet lap DataLoader cho bai toan phan loai hang loi (Defect Detection).

Noi dung chinh:
  1. DefectDataset  — Custom Dataset ke thua torch.utils.data.Dataset.
     Ham __getitem__ load anh bang OpenCV, ap dung dung pipeline
     augmentation/preprocessing tuy theo nhan (defect / no-defect).

  2. WeightedRandomSampler — Xu ly mat can bang du lieu (defect << no-defect)
     bang cach lay mau ngau nhien co trong so thay vi oversampling tinh.

  3. get_defect_dataloaders() — Ham tien ich khoi tao train_loader va
     val_loader voi tham so toi uu (batch_size=32, num_workers=4, pin_memory=True).

Su dung:
    from ai_engine.image_processing.defect_dataloader import get_defect_dataloaders

    train_loader, val_loader = get_defect_dataloaders(
        data_dir="data/processed",
        batch_size=32,
    )
    for images, labels in train_loader:
        # images: Tensor [B, 3, 224, 224]  -- da normalize theo ImageNet
        # labels: Tensor [B]               -- 0 = no-defect, 1 = defect
        ...

Yeu cau moi truong:
    Python >= 3.11, PyTorch, OpenCV (cv2), Albumentations

Luu y Windows:
    Tren Windows, multiprocessing cua DataLoader co han che.
    num_workers=4 chi hoat dong dung khi script duoc chay duoi dang module
    (python -m ...) hoac ben trong if __name__ == "__main__".
    Neu gap loi freezing, set num_workers=0 de debug.
"""

import os
import platform
from pathlib import Path
from typing import Tuple, Optional

import cv2
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
    Subset,
)
from sklearn.model_selection import train_test_split

# Pipeline augmentation/preprocessing da dinh nghia o buoc Image Augmentation
from ai_engine.image_processing.augmentation.transforms import (
    get_defect_transforms,
    get_normal_transforms,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mapping nhan: thu muc "no-defect" -> 0, "defect" -> 1
CLASS_MAPPING = {"no-defect": 0, "defect": 1}

# Ten class tuong ung voi index (dung cho logging/report)
IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}


# ---------------------------------------------------------------------------
# 1. DefectDataset — Custom Dataset
# ---------------------------------------------------------------------------

class DefectDataset(Dataset):
    """
    Custom PyTorch Dataset cho bai toan phan loai hang loi.

    Cau truc thu muc du lieu mong doi:
        data_dir/
        +-- defect/        <- anh hang loi (thieu so)
        |   +-- img001.jpg
        |   +-- ...
        +-- no-defect/     <- anh hang binh thuong (da so)
            +-- img001.jpg
            +-- ...

    Logic ap dung augmentation:
        - Anh "defect" (label=1) khi train:
              Ap dung pipeline augmentation day du (xoay, lat, nhieu, chinh sang...)
              -> Giup model hoc duoc nhieu bien the cua hang loi.
        - Anh "no-defect" (label=0) khi train va TAT CA anh khi validate/test:
              Chi ap dung Resize + Normalize (khong augment)
              -> Dam bao validation phan anh phan phoi thuc te.

    Attributes:
        image_paths  (list[Path]): Danh sach duong dan toi tung anh.
        labels       (list[int]):  Nhan tuong ung (0 hoac 1).
        is_train     (bool):       True neu dang o che do train.
        class_counts (dict):       So luong anh moi class, dung de tinh class weights.
    """

    def __init__(self, data_dir: str, is_train: bool = True):
        """
        Khoi tao dataset bang cach quet thu muc va thu thap duong dan anh.

        Args:
            data_dir (str): Duong dan thu muc chua 2 thu muc con 'defect' va 'no-defect'.
            is_train (bool): Che do train (True) hay val/test (False).
                             Anh huong den pipeline augmentation duoc ap dung.
        """
        self.data_dir = Path(data_dir)
        self.is_train = is_train

        self.image_paths: list[Path] = []
        self.labels: list[int] = []

        # --- Load hai pipeline augmentation ---
        # Pipeline manh: dung cho anh defect trong train (xoay, lat, nhieu, v.v.)
        self._defect_transform = get_defect_transforms()
        # Pipeline nhe: chi resize + normalize, dung cho no-defect va toan bo val/test
        self._normal_transform = get_normal_transforms()

        # --- Quet thu muc va thu thap duong dan anh ---
        self.class_counts: dict[str, int] = {}
        for class_name, label in CLASS_MAPPING.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"[WARNING] Directory not found, skipping: {class_dir}")
                self.class_counts[class_name] = 0
                continue

            # Thu thap anh voi nhieu dinh dang pho bien
            found = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                found.extend(class_dir.glob(ext))

            self.image_paths.extend(found)
            self.labels.extend([label] * len(found))
            self.class_counts[class_name] = len(found)

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in '{data_dir}'. "
                "Check the directory path and folder structure."
            )

    def __len__(self) -> int:
        """Tra ve tong so mau trong dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load anh va ap dung pipeline augmentation/preprocessing phu hop.

        Args:
            idx (int): Chi so cua mau can lay.

        Returns:
            Tuple[Tensor, Tensor]:
                - image_tensor: FloatTensor kich thuoc [3, 224, 224],
                                da normalize theo chuan ImageNet.
                - label: LongTensor scalar (0 = no-defect, 1 = defect).

        Raises:
            ValueError: Neu khong the doc anh tai duong dan da cho.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # --- Doc anh bang OpenCV ---
        # OpenCV doc anh theo dinh dang BGR va numpy array (H, W, C)
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise ValueError(
                f"Cannot read image: '{img_path}'. "
                "File may be corrupted or not a valid image format."
            )

        # Chuyen BGR -> RGB vi Albumentations va PyTorch dung RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # --- Chon va ap dung pipeline tuong ung ---
        # Dieu kien: CHI augment khi train VA anh la defect (class thieu so)
        if self.is_train and label == CLASS_MAPPING["defect"]:
            # Pipeline day du: spatial transforms + pixel transforms + resize + normalize
            augmented = self._defect_transform(image=image_rgb)
        else:
            # Pipeline toi gian: chi resize + normalize (khong augment)
            # Ap dung cho: (1) anh no-defect trong train, (2) moi anh trong val/test
            augmented = self._normal_transform(image=image_rgb)

        # Albumentations tra ve dict, lay tensor tu key "image"
        # ToTensorV2() da chuyen HWC numpy -> CHW FloatTensor va normalize
        image_tensor: torch.Tensor = augmented["image"]

        # Label phai la LongTensor de dung voi CrossEntropyLoss / FocalLoss
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


# ---------------------------------------------------------------------------
# 2. Tinh class weights va tao WeightedRandomSampler
# ---------------------------------------------------------------------------

def build_weighted_sampler(dataset: DefectDataset, indices: list[int]) -> WeightedRandomSampler:
    """
    Tinh trong so cho tung mau va tao WeightedRandomSampler.

    Cach tiep can:
        - Moi class duoc gan trong so ty le nghich voi so luong mau cua no:
              weight_class = 1.0 / count_class
        - Moi mau trong train set duoc gan trong so cua class no thuoc ve.
        - WeightedRandomSampler lay mau ngau nhien co trong so voi replacement=True.
        => Ket qua: class defect (it anh) duoc lay mau thuong xuyen hon,
           can bang lai phan phoi ma khong can nhan ban anh (oversampling tinh).

    Args:
        dataset (DefectDataset): Dataset goc (toan bo, chua split).
        indices (list[int]):     Danh sach chi so cua tap train.

    Returns:
        WeightedRandomSampler: Sampler da cau hinh, dung truc tiep cho DataLoader.
    """
    # Lay nhan cua tat ca mau trong train subset
    train_labels = [dataset.labels[i] for i in indices]

    # Dem so luong mau moi class trong train set
    class_sample_counts = {}
    for label in set(train_labels):
        class_sample_counts[label] = train_labels.count(label)

    # Tinh trong so moi class: nghich dao so luong mau
    # Class cang it mau -> trong so cang cao -> duoc lay mau nhieu hon
    class_weights = {
        label: 1.0 / count
        for label, count in class_sample_counts.items()
    }

    # In thong tin de team de theo doi
    print("[WeightedSampler] Class distribution in train set:")
    for label, count in class_sample_counts.items():
        class_name = IDX_TO_CLASS.get(label, f"class_{label}")
        print(f"  {class_name:>12} (label={label}): {count:>6} samples | weight = {class_weights[label]:.6f}")

    # Gan trong so cho tung mau trong train subset
    sample_weights = [class_weights[label] for label in train_labels]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float64)

    # Tao sampler
    # num_samples = len(train set): moi epoch se lay du so mau bang train set goc
    # replacement=True: cho phep lay lai mau (can thiet de dam bao class hiem duoc lay du)
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(train_labels),
        replacement=True,
    )

    return sampler


# ---------------------------------------------------------------------------
# 3. get_defect_dataloaders() — Ham tien ich tao DataLoaders
# ---------------------------------------------------------------------------

def get_defect_dataloaders(
    data_dir: str = "data/processed",
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Khoi tao train_loader va val_loader toi uu cho bai toan Defect Detection.

    Pipeline:
        1. Tao DefectDataset goc (khong augment) de lay danh sach nhan cho stratified split.
        2. Stratified split: dam bao ty le defect/no-defect giong nhau o ca train va val.
        3. Tao 2 instance rieng biet: train_dataset (is_train=True) va val_dataset (is_train=False).
        4. Tinh class weights va tao WeightedRandomSampler cho train_loader.
        5. Khoi tao DataLoader voi cau hinh toi uu.

    Args:
        data_dir    (str):   Duong dan thu muc chua 'defect/' va 'no-defect/'.
        batch_size  (int):   So anh moi batch. Mac dinh: 32.
        val_split   (float): Ty le du lieu danh cho validation. Mac dinh: 0.2 (20%).
        seed        (int):   Random seed de ket qua split co the tai tao. Mac dinh: 42.
        num_workers (int):   So worker process cho DataLoader.
                             Mac dinh: 4 tren Linux/Mac, 0 tren Windows
                             (Windows multiprocessing co han che voi DataLoader).
        pin_memory  (bool):  Dung pinned memory de tang toc transfer CPU->GPU.
                             Mac dinh: True neu CUDA kha dung, False neu khong.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
            - train_loader: Su dung WeightedRandomSampler, shuffle=False
                            (sampler dieu khien thu tu thay the shuffle).
            - val_loader:   Shuffle=False, khong sampler, phan anh phan phoi thuc te.

    Example:
        >>> train_loader, val_loader = get_defect_dataloaders("data/processed")
        >>> images, labels = next(iter(train_loader))
        >>> print(images.shape)  # torch.Size([32, 3, 224, 224])
        >>> print(labels.shape)  # torch.Size([32])
    """

    # --- Xac dinh num_workers va pin_memory tu dong neu khong truyen vao ---
    if num_workers is None:
        # Windows: multiprocessing spawn context gay loi neu num_workers > 0
        # trong mot so truong hop (interactive session, Jupyter, v.v.)
        num_workers = 0 if platform.system() == "Windows" else 4

    if pin_memory is None:
        # pin_memory chi co loi khi dung GPU; tren CPU khong can thiet
        pin_memory = torch.cuda.is_available()

    print("\n[DataLoader Config]")
    print(f"  data_dir    = {data_dir}")
    print(f"  batch_size  = {batch_size}")
    print(f"  val_split   = {val_split} ({val_split*100:.0f}%)")
    print(f"  num_workers = {num_workers}  (auto: 0 on Windows, 4 on Linux/Mac)")
    print(f"  pin_memory  = {pin_memory}  (auto: True if CUDA available)")
    print(f"  seed        = {seed}")

    # --- Buoc 1: Tao dataset goc (is_train=False) de lay nhan cho stratified split ---
    # Dung is_train=False vi chung ta chi can danh sach nhan, khong can augmentation
    base_dataset = DefectDataset(data_dir=data_dir, is_train=False)
    total_size = len(base_dataset)

    print(f"\n[Dataset Summary] Total images: {total_size}")
    for class_name, count in base_dataset.class_counts.items():
        label = CLASS_MAPPING[class_name]
        ratio = count / total_size * 100 if total_size > 0 else 0
        print(f"  {class_name:>12} (label={label}): {count:>6} images ({ratio:.1f}%)")

    # --- Buoc 2: Stratified split ---
    # Stratified dam bao ty le defect/no-defect giong nhau o train va val.
    # Rat quan trong voi dataset mat can bang: tranh truong hop val set co qua it defect.
    all_indices = list(range(total_size))
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=val_split,
        stratify=base_dataset.labels,   # dam bao ty le class giong nhau
        random_state=seed,
        shuffle=True,
    )

    print(f"\n[Split] Train: {len(train_indices)} | Val: {len(val_indices)} images")

    # Kiem tra ty le class trong tung split
    train_defect = sum(1 for i in train_indices if base_dataset.labels[i] == 1)
    val_defect   = sum(1 for i in val_indices   if base_dataset.labels[i] == 1)
    print(f"  Train defect: {train_defect}/{len(train_indices)} ({train_defect/len(train_indices)*100:.1f}%)")
    print(f"  Val   defect: {val_defect}/{len(val_indices)} ({val_defect/len(val_indices)*100:.1f}%)")

    # --- Buoc 3: Tao 2 instance dataset rieng biet ---
    # QUAN TRONG: Phai tao 2 instance rieng biet thay vi dung chung 1 dataset.
    # Ly do: is_train anh huong den pipeline augmentation; neu dung chung,
    # augmentation cua train se bi ap dung len ca val khi lay batch cung luc.
    train_dataset = DefectDataset(data_dir=data_dir, is_train=True)    # augmentation bat
    val_dataset   = DefectDataset(data_dir=data_dir, is_train=False)   # chi resize + normalize

    # Tao Subset tu indices da split
    train_subset = Subset(train_dataset, train_indices)
    val_subset   = Subset(val_dataset,   val_indices)

    # --- Buoc 4: Tinh class weights va tao WeightedRandomSampler ---
    # Sampler xu ly mat can bang bang cach lay mau co trong so:
    # => Anh defect (hiem) duoc lay mau thuong xuyen hon trong moi epoch
    print("\n[WeightedRandomSampler] Computing weights...")
    sampler = build_weighted_sampler(train_dataset, train_indices)

    # --- Buoc 5: Khoi tao DataLoaders ---

    # Train loader:
    # - shuffle=False : PHAI set False khi dung sampler (sampler dieu khien thu tu)
    # - sampler       : WeightedRandomSampler da cau hinh
    # - drop_last=True: Bo batch cuoi neu khong du batch_size (tranh loi BatchNorm1d)
    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=False,          # <- BAT BUOC False khi co sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,         # Tranh batch cuoi < batch_size gay loi BatchNorm1d
    )

    # Val loader:
    # - shuffle=False  : Val set khong can xao tron (danh gia nhat quan moi epoch)
    # - Khong sampler  : Muon phan phoi tu nhien cua val set
    # - drop_last=False: Giu lai tat ca mau de danh gia day du
    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print("\n[DataLoader Ready]")
    print(f"  train_loader : {len(train_loader)} batches x {batch_size} = ~{len(train_loader)*batch_size} samples/epoch")
    print(f"  val_loader   : {len(val_loader)} batches x {batch_size}")
    print()

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick test (chay truc tiep file nay de kiem tra nhanh)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Them root project vao sys.path de import hoat dong
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

    print("=" * 60)
    print("  DefectDataLoader -- Quick Sanity Check")
    print("=" * 60)

    DATA_DIR = "data/processed"

    # Tao dataloaders (num_workers=0 khi chay truc tiep de tranh loi Windows)
    train_loader, val_loader = get_defect_dataloaders(
        data_dir=DATA_DIR,
        batch_size=32,
        num_workers=0,    # 0 khi test truc tiep
        pin_memory=False,
    )

    # Lay 1 batch tu train_loader va kiem tra
    print("[Test] Fetching 1 batch from train_loader...")
    images, labels = next(iter(train_loader))

    print(f"  images.shape : {images.shape}")   # [32, 3, 224, 224]
    print(f"  images.dtype : {images.dtype}")   # torch.float32
    print(f"  labels.shape : {labels.shape}")   # [32]
    print(f"  labels.dtype : {labels.dtype}")   # torch.int64
    print(f"  labels unique: {labels.unique().tolist()}")

    # Kiem tra normalize: gia tri pixel phai nam trong khoang hop ly (~[-2.5, 2.5])
    print(f"  pixel range  : [{images.min():.3f}, {images.max():.3f}]")

    # Kiem tra val_loader
    print("\n[Test] Fetching 1 batch from val_loader...")
    val_images, val_labels = next(iter(val_loader))
    print(f"  val_images.shape : {val_images.shape}")
    print(f"  val_labels unique: {val_labels.unique().tolist()}")

    print("\n[OK] DefectDataLoader working correctly!")
