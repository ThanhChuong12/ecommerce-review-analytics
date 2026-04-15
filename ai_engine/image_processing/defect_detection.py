"""
defect_detection.py
-------------------
Nhận diện hàng móp méo / lỗi bằng Transfer Learning:
  - ResNet (backbone mạnh hơn, chính xác hơn)
  - MobileNet (nhẹ hơn, phù hợp inference nhanh)

TODO:
  - [ ] Load pretrained ResNet / MobileNet từ backend_ai/models/
  - [ ] Fine-tune trên dataset hàng lỗi (data/raw/ hoặc Kaggle)
  - [ ] Đánh giá Accuracy, Precision trên tập test
  - [ ] Export mô hình tốt nhất vào backend_ai/models/
"""

def detect_defect_resnet(image_path: str) -> dict:
    """Nhận diện hàng lỗi bằng ResNet. Returns dict với label và confidence."""
    raise NotImplementedError("TODO: Implement ResNet defect detection")


def detect_defect_mobilenet(image_path: str) -> dict:
    """Nhận diện hàng lỗi bằng MobileNet. Returns dict với label và confidence."""
    raise NotImplementedError("TODO: Implement MobileNet defect detection")
