# -*- coding: utf-8 -*-
"""
defect_dataloader.py
--------------------
Module setting up DataLoader for the defect classification task (Defect Detection).

Main content:
  1. DefectDataset  — Custom Dataset inheriting from torch.utils.data.Dataset.
     __getitem__ loads images using OpenCV, applying the appropriate
     augmentation/preprocessing pipeline depending on the label (defect / no-defect).

  2. WeightedRandomSampler — Handles data imbalance (defect << no-defect)
     by using random weighted sampling instead of static oversampling.

  3. get_defect_dataloaders() — Helper function to initialize train_loader and
     val_loader with optimal parameters (batch_size=32, num_workers=4, pin_memory=True).

Usage:
    from ai_engine.image_processing.defect_dataloader import get_defect_dataloaders

    train_loader, val_loader = get_defect_dataloaders(
        data_dir="data/processed",
        batch_size=32,
    )
    for images, labels in train_loader:
        # images: Tensor [B, 3, 224, 224]  -- normalized using ImageNet
        # labels: Tensor [B]               -- 0 = no-defect, 1 = defect
        ...

Environment requirements:
    Python >= 3.11, PyTorch, OpenCV (cv2), Albumentations

Windows note:
    On Windows, DataLoader multiprocessing has limitations.
    num_workers=4 only works correctly when the script is run as a module
    (python -m ...) or inside if __name__ == "__main__".
    If freezing issues occur, set num_workers=0 to debug.
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

# Augmentation/preprocessing pipeline defined in the Image Augmentation step
from ai_engine.image_processing.augmentation.transforms import (
    get_defect_transforms,
    get_normal_transforms,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Label mapping: "no-defect" folder -> 0, "defect" -> 1
CLASS_MAPPING = {"no-defect": 0, "defect": 1}

# Class name corresponding to index (used for logging/report)
IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}


# ---------------------------------------------------------------------------
# 1. DefectDataset — Custom Dataset
# ---------------------------------------------------------------------------

class DefectDataset(Dataset):
    """
    Custom PyTorch Dataset for defect classification.

    Expected directory structure:
        data_dir/
        +-- defect/        <- defect images (minority class)
        |   +-- img001.jpg
        |   +-- ...
        +-- no-defect/     <- normal images (majority class)
            +-- img001.jpg
            +-- ...

    Augmentation logic:
        - "defect" images (label=1) during training:
              Apply full augmentation pipeline (rotation, flips, noise, brightness adjustments, etc.)
              -> Helps model learn different variations of defects.
        - "no-defect" images (label=0) during training and ALL images during validation/test:
              Only apply Resize + Normalize (no augmentation)
              -> Ensures validation reflects actual distribution.

    Attributes:
        image_paths  (list[Path]): List of paths to each image.
        labels       (list[int]):  Corresponding label (0 or 1).
        is_train     (bool):       True if in training mode.
        class_counts (dict):       Image counts for each class, used to compute class weights.
    """

    def __init__(self, data_dir: str, is_train: bool = True):
        """
        Initialize dataset by scanning directory and collecting image paths.

        Args:
            data_dir (str): Path to folder containing 'defect' and 'no-defect' subfolders.
            is_train (bool): Train mode (True) or val/test mode (False).
                             Affects which augmentation pipeline is applied.
        """
        self.data_dir = Path(data_dir)
        self.is_train = is_train

        self.image_paths: list[Path] = []
        self.labels: list[int] = []

        # --- Load the two augmentation pipelines ---
        # Strong pipeline: used for defect images during training (rotation, flips, noise, etc.)
        self._defect_transform = get_defect_transforms()
        # Light pipeline: resize + normalize only, used for no-defect and all val/test
        self._normal_transform = get_normal_transforms()

        # --- Scan directory and collect image paths ---
        self.class_counts: dict[str, int] = {}
        for class_name, label in CLASS_MAPPING.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"[WARNING] Directory not found, skipping: {class_dir}")
                self.class_counts[class_name] = 0
                continue

            # Collect images with common formats
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
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load image and apply appropriate augmentation/preprocessing pipeline.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            Tuple[Tensor, Tensor]:
                - image_tensor: FloatTensor of size [3, 224, 224],
                                normalized using ImageNet standard.
                - label: LongTensor scalar (0 = no-defect, 1 = defect).

        Raises:
            ValueError: If the image cannot be read at the given path.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # --- Read image using OpenCV ---
        # OpenCV reads image in BGR format and returns a numpy array (H, W, C)
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise ValueError(
                f"Cannot read image: '{img_path}'. "
                "File may be corrupted or not a valid image format."
            )

        # Convert BGR -> RGB since Albumentations and PyTorch use RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # --- Select and apply corresponding pipeline ---
        # Condition: ONLY augment during training AND image is defect (minority class)
        if self.is_train and label == CLASS_MAPPING["defect"]:
            # Full pipeline: spatial transforms + pixel transforms + resize + normalize
            augmented = self._defect_transform(image=image_rgb)
        else:
            # Minimal pipeline: resize + normalize only (no augmentation)
            # Applied to: (1) no-defect images in train, (2) all images in val/test
            augmented = self._normal_transform(image=image_rgb)

        # Albumentations returns a dict, get tensor from key "image"
        # ToTensorV2() converts HWC numpy -> CHW FloatTensor and normalizes
        image_tensor: torch.Tensor = augmented["image"]

        # Label must be LongTensor to work with CrossEntropyLoss / FocalLoss
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


# ---------------------------------------------------------------------------
# 2. Compute class weights and create WeightedRandomSampler
# ---------------------------------------------------------------------------

def build_weighted_sampler(dataset: DefectDataset, indices: list[int]) -> WeightedRandomSampler:
    """
    Compute weights for each sample and create WeightedRandomSampler.

    Approach:
        - Each class is assigned a weight inversely proportional to its sample count:
              weight_class = 1.0 / count_class
        - Each sample in the train set is assigned the weight of the class it belongs to.
        - WeightedRandomSampler performs random weighted sampling with replacement=True.
        => Result: defect class (fewer images) is sampled more frequently,
           balancing the distribution without static oversampling.

    Args:
        dataset (DefectDataset): Base dataset (full, before split).
        indices (list[int]):     List of indices for the train set.

    Returns:
        WeightedRandomSampler: Configured sampler, used directly in DataLoader.
    """
    # Get labels of all samples in the train subset
    train_labels = [dataset.labels[i] for i in indices]

    # Count number of samples for each class in the train set
    class_sample_counts = {}
    for label in set(train_labels):
        class_sample_counts[label] = train_labels.count(label)

    # Compute weights for each class: inverse of sample count
    # Class with fewer samples -> higher weight -> sampled more frequently
    class_weights = {
        label: 1.0 / count
        for label, count in class_sample_counts.items()
    }

    # Print info for tracking
    print("[WeightedSampler] Class distribution in train set:")
    for label, count in class_sample_counts.items():
        class_name = IDX_TO_CLASS.get(label, f"class_{label}")
        print(f"  {class_name:>12} (label={label}): {count:>6} samples | weight = {class_weights[label]:.6f}")

    # Assign weights to each sample in the train subset
    sample_weights = [class_weights[label] for label in train_labels]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float64)

    # Create sampler
    # num_samples = len(train set): each epoch draws sample size equal to original train set
    # replacement=True: allows resampling (necessary to ensure rare classes are drawn sufficiently)
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(train_labels),
        replacement=True,
    )

    return sampler


# ---------------------------------------------------------------------------
# 3. get_defect_dataloaders() — Helper function to create DataLoaders
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
    Initialize train_loader and val_loader optimized for Defect Detection.

    Pipeline:
        1. Create base DefectDataset (no augmentation) to get label list for stratified split.
        2. Stratified split: ensures identical defect/no-defect ratios in both train and val.
        3. Create 2 separate instances: train_dataset (is_train=True) and val_dataset (is_train=False).
        4. Compute class weights and create WeightedRandomSampler for train_loader.
        5. Initialize DataLoader with optimal configuration.

    Args:
        data_dir    (str):   Path to folder containing 'defect/' and 'no-defect/'.
        batch_size  (int):   Batch size. Default: 32.
        val_split   (float): Validation split ratio. Default: 0.2 (20%).
        seed        (int):   Random seed for reproducible split. Default: 42.
        num_workers (int):   Number of worker processes for DataLoader.
                             Default: 4 on Linux/Mac, 0 on Windows
                             (Windows has multiprocessing limits with DataLoader).
        pin_memory  (bool):  Use pinned memory to speed up CPU->GPU transfer.
                             Default: True if CUDA is available, False otherwise.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
            - train_loader: Uses WeightedRandomSampler, shuffle=False
                            (sampler controls order instead of shuffle).
            - val_loader:   shuffle=False, no sampler, reflects natural distribution.

    Example:
        >>> train_loader, val_loader = get_defect_dataloaders("data/processed")
        >>> images, labels = next(iter(train_loader))
        >>> print(images.shape)  # torch.Size([32, 3, 224, 224])
        >>> print(labels.shape)  # torch.Size([32])
    """

    # --- Automatically determine num_workers and pin_memory if not provided ---
    if num_workers is None:
        # Windows: multiprocessing spawn context can cause errors if num_workers > 0
        # in some environments (interactive sessions, Jupyter, etc.)
        num_workers = 0 if platform.system() == "Windows" else 4

    if pin_memory is None:
        # pin_memory is only beneficial for GPU; not needed on CPU
        pin_memory = torch.cuda.is_available()

    print("\n[DataLoader Config]")
    print(f"  data_dir    = {data_dir}")
    print(f"  batch_size  = {batch_size}")
    print(f"  val_split   = {val_split} ({val_split*100:.0f}%)")
    print(f"  num_workers = {num_workers}  (auto: 0 on Windows, 4 on Linux/Mac)")
    print(f"  pin_memory  = {pin_memory}  (auto: True if CUDA available)")
    print(f"  seed        = {seed}")

    # --- Step 1: Create base dataset (is_train=False) to get labels for stratified split ---
    # Use is_train=False since we only need labels, no augmentation
    base_dataset = DefectDataset(data_dir=data_dir, is_train=False)
    total_size = len(base_dataset)

    print(f"\n[Dataset Summary] Total images: {total_size}")
    for class_name, count in base_dataset.class_counts.items():
        label = CLASS_MAPPING[class_name]
        ratio = count / total_size * 100 if total_size > 0 else 0
        print(f"  {class_name:>12} (label={label}): {count:>6} images ({ratio:.1f}%)")

    # --- Step 2: Stratified split ---
    # Stratified split ensures identical defect/no-defect ratios in train and val.
    # Crucial for imbalanced datasets: prevents val set from having too few defects.
    all_indices = list(range(total_size))
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=val_split,
        stratify=base_dataset.labels,   # ensures identical class ratios
        random_state=seed,
        shuffle=True,
    )

    print(f"\n[Split] Train: {len(train_indices)} | Val: {len(val_indices)} images")

    # Check class ratios in each split
    train_defect = sum(1 for i in train_indices if base_dataset.labels[i] == 1)
    val_defect   = sum(1 for i in val_indices   if base_dataset.labels[i] == 1)
    print(f"  Train defect: {train_defect}/{len(train_indices)} ({train_defect/len(train_indices)*100:.1f}%)")
    print(f"  Val   defect: {val_defect}/{len(val_indices)} ({val_defect/len(val_indices)*100:.1f}%)")

    # --- Step 3: Create 2 separate dataset instances ---
    # IMPORTANT: Must create 2 separate instances instead of sharing 1 dataset.
    # Reason: is_train affects the augmentation pipeline; if shared,
    # training augmentation will be applied to validation when fetching batch.
    train_dataset = DefectDataset(data_dir=data_dir, is_train=True)    # augmentation enabled
    val_dataset   = DefectDataset(data_dir=data_dir, is_train=False)   # resize + normalize only

    # Create Subset from split indices
    train_subset = Subset(train_dataset, train_indices)
    val_subset   = Subset(val_dataset,   val_indices)

    # --- Step 4: Compute class weights and create WeightedRandomSampler ---
    # Sampler handles imbalance by performing weighted sampling:
    # => Defect images (rare) are sampled more frequently in each epoch.
    print("\n[WeightedRandomSampler] Computing weights...")
    sampler = build_weighted_sampler(train_dataset, train_indices)

    # --- Step 5: Initialize DataLoaders ---

    # Train loader:
    # - shuffle=False : MUST set to False when using sampler (sampler controls ordering)
    # - sampler       : Configured WeightedRandomSampler
    # - drop_last=True: Drop the last batch if smaller than batch_size (prevents BatchNorm1d errors)
    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=False,          # <- MUST be False when sampler is present
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,         # Prevent last batch < batch_size from causing BatchNorm1d error
    )

    # Val loader:
    # - shuffle=False  : Validation set doesn't need shuffling (consistent evaluation every epoch)
    # - No sampler  : We want the natural distribution of the validation set
    # - drop_last=False: Keep all samples for full evaluation
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
# Quick test (run this file directly for a quick sanity check)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Add project root to sys.path for imports to work
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

    print("=" * 60)
    print("  DefectDataLoader -- Quick Sanity Check")
    print("=" * 60)

    DATA_DIR = "data/processed"

    # Create dataloaders (num_workers=0 when running directly to avoid Windows errors)
    train_loader, val_loader = get_defect_dataloaders(
        data_dir=DATA_DIR,
        batch_size=32,
        num_workers=0,    # 0 when testing directly
        pin_memory=False,
    )

    # Get 1 batch from train_loader and check
    print("[Test] Fetching 1 batch from train_loader...")
    images, labels = next(iter(train_loader))

    print(f"  images.shape : {images.shape}")   # [32, 3, 224, 224]
    print(f"  images.dtype : {images.dtype}")   # torch.float32
    print(f"  labels.shape : {labels.shape}")   # [32]
    print(f"  labels.dtype : {labels.dtype}")   # torch.int64
    print(f"  labels unique: {labels.unique().tolist()}")

    # Check normalization: pixel values should be within a reasonable range (~[-2.5, 2.5])
    print(f"  pixel range  : [{images.min():.3f}, {images.max():.3f}]")

    # Check val_loader
    print("\n[Test] Fetching 1 batch from val_loader...")
    val_images, val_labels = next(iter(val_loader))
    print(f"  val_images.shape : {val_images.shape}")
    print(f"  val_labels unique: {val_labels.unique().tolist()}")

    print("\n[OK] DefectDataLoader working correctly!")
