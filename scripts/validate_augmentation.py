import os
import glob
import random
import cv2
import numpy as np
from pathlib import Path
import sys

# Ensure the root project directory is in the sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_engine.image_processing.augmentation.transforms import get_defect_transforms

def tensor_to_image(tensor):
    """
    Converts a normalized PyTorch tensor back to a valid image (numpy array).
    """
    # Convert tensor to numpy array and CHW to HWC
    image = tensor.numpy().transpose(1, 2, 0)
    
    # ImageNet statistics used in normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Un-normalize
    image = std * image + mean
    
    # Scale to [0, 255] and convert to uint8
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image

def main():
    # Define paths
    base_dir = Path(__file__).resolve().parent.parent
    input_dir = base_dir / "data" / "processed" / "defect"
    output_dir = base_dir / "data" / "processed" / "aug_samples"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for images
    image_paths = []
    if input_dir.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(input_dir.glob(ext))
    
    # If no images are found, we create 5 dummy images to demonstrate the functionality
    if not image_paths:
        print(f"No images found in {input_dir}. Generating dummy images for validation...")
        input_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            # Create a mock image with some "defect" patterns (e.g. circles, lines)
            dummy_img = np.ones((300, 300, 3), dtype=np.uint8) * 200
            # Add a mock defect (red line)
            cv2.line(dummy_img, (50, 50), (250, 250), (0, 0, 255), 5)
            # Add some noise
            noise = np.random.normal(0, 10, dummy_img.shape)
            dummy_img = np.clip(dummy_img + noise, 0, 255).astype(np.uint8)
            
            dummy_path = input_dir / f"mock_defect_{i+1:02d}.jpg"
            cv2.imwrite(str(dummy_path), dummy_img)
            image_paths.append(dummy_path)
            
    # Randomly select 5 images (or all if less than 5)
    selected_paths = random.sample(image_paths, min(5, len(image_paths)))
    
    print(f"Selected {len(selected_paths)} images for validation.")
    
    # Load transformation pipeline
    transform_pipeline = get_defect_transforms()
    
    saved_count = 0
    for img_path in selected_paths:
        # Load image with OpenCV
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to read image: {img_path}")
            continue
            
        # Convert BGR to RGB (Albumentations expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        filename = img_path.stem
        
        # Generate 4 augmented versions
        for i in range(1, 5):
            augmented = transform_pipeline(image=image)
            aug_tensor = augmented["image"]
            
            # Convert back to HWC, unnormalize, and to uint8
            aug_image_rgb = tensor_to_image(aug_tensor)
            
            # Convert RGB to BGR for saving with OpenCV
            aug_image_bgr = cv2.cvtColor(aug_image_rgb, cv2.COLOR_RGB2BGR)
            
            # Save the augmented image
            save_path = output_dir / f"{filename}_aug{i}.jpg"
            cv2.imwrite(str(save_path), aug_image_bgr)
            print(f"Saved augmented image to {save_path}")
            saved_count += 1
            
    print(f"\n[SUCCESS] Done. {saved_count} augmented images saved to {output_dir}")

if __name__ == "__main__":
    main()
