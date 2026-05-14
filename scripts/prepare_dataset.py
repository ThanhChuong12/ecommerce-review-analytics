import os
import shutil
import pandas as pd
from pathlib import Path

def main():
    # Định nghĩa các đường dẫn
    project_root = Path(__file__).resolve().parent.parent
    
    # File chứa nhãn đã gán
    labels_csv_path = project_root / "image_labeling" / "image_labeling" / "data" / "manifests" / "labels.csv"
    
    # Thư mục chứa ảnh gốc (từ kết quả labeling)
    source_image_base = project_root / "image_labeling" / "image_labeling"
    
    # Thư mục đích cho quá trình training AI
    target_defect_dir = project_root / "data" / "processed" / "defect"
    target_no_defect_dir = project_root / "data" / "processed" / "no-defect"
    
    # Tạo thư mục nếu chưa tồn tại
    target_defect_dir.mkdir(parents=True, exist_ok=True)
    target_no_defect_dir.mkdir(parents=True, exist_ok=True)
    
    # Xóa ảnh cũ (nếu có dummy images) để chuẩn bị data thật
    for f in target_defect_dir.glob("*.jpg"):
        f.unlink()
    for f in target_no_defect_dir.glob("*.jpg"):
        f.unlink()

    print(f"Reading data from {labels_csv_path}...")
    
    try:
        df = pd.read_csv(labels_csv_path)
    except FileNotFoundError:
        print(f"Cannot find file {labels_csv_path}")
        return

    # Map các label labeling sang label training
    # "damaged" -> defect
    # "intact" -> no-defect
    # "irrelevant" & "wrong_item" -> Bỏ qua (không đưa vào mô hình nhận diện lỗi hỏng hóc vật lý)
    
    copied_defect = 0
    copied_no_defect = 0
    missing_images = 0

    for index, row in df.iterrows():
        label = str(row['label']).strip().lower()
        
        # Lấy đường dẫn tương đối từ file CSV (ví dụ: data\frames\abc.jpg)
        relative_img_path = str(row['image_path'])
        
        # Đường dẫn tuyệt đối tới file ảnh gốc
        source_img_path = source_image_base / relative_img_path
        
        if not source_img_path.exists():
            missing_images += 1
            continue
            
        filename = source_img_path.name
        
        if label == "damaged":
            target_path = target_defect_dir / filename
            shutil.copy2(source_img_path, target_path)
            copied_defect += 1
        elif label == "intact":
            target_path = target_no_defect_dir / filename
            shutil.copy2(source_img_path, target_path)
            copied_no_defect += 1
            
    print("\n[SUCCESS] Dataset preparation complete!")
    print(f"- Copied Defect images: {copied_defect}")
    print(f"- Copied No-Defect images: {copied_no_defect}")
    if missing_images > 0:
        print(f"[WARNING] {missing_images} labeled images were not found in source dir.")

if __name__ == "__main__":
    main()
