"""
zero_shot_clip.py
-----------------
Phát hiện hình ảnh không liên quan (rác) bằng CLIP zero-shot classification.

TODO:
  - [ ] Load CLIP model (openai/clip-vit-base-patch32)
  - [ ] Định nghĩa các label phân loại (relevant / irrelevant)
  - [ ] Classify ảnh từ link thu thập bởi scraping_agent
  - [ ] Filter ảnh không liên quan, chuyển sang data/processed/
"""

def detect_irrelevant_image(image_path: str) -> bool:
    """Trả về True nếu ảnh KHÔNG liên quan đến sản phẩm."""
    raise NotImplementedError("TODO: Implement CLIP zero-shot classification")
