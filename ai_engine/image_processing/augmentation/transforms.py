import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_defect_transforms() -> A.Compose:
    """
    Returns the augmentation pipeline for the 'defect' class.
    
    This pipeline includes spatial and pixel-level transformations designed to 
    augment distorted/misaligned defect images while preserving semantic defect 
    characteristics. It ends with the required preprocessing for a ResNet50 backbone.
    
    Returns:
        A.Compose: Albumentations composition of transformations.
    """
    return A.Compose([
        # Spatial transformations
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        
        # Pixel-level transformations
        # Note: Use A.GaussNoise for Albumentations < 1.4.0. For >= 1.4.0, use A.GaussianNoise to avoid warnings.
        A.GaussNoise(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        
        # Mandatory ending transforms for ResNet50
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_normal_transforms() -> A.Compose:
    """
    Returns the preprocessing pipeline for the 'no-defect' class (and validation/test).
    
    This pipeline applies NO augmentations. It only resizes the image and normalizes it
    to match the expectations of an ImageNet-pretrained ResNet50 backbone.
    
    Returns:
        A.Compose: Albumentations composition of transformations.
    """
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
