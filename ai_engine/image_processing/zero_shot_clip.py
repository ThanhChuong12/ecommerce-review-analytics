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

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

_clip_model = None
_clip_processor = None

def detect_irrelevant_image(image_path: str) -> bool:
    """Trả về True nếu ảnh KHÔNG liên quan đến sản phẩm."""
    global _clip_model, _clip_processor
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _clip_model is None or _clip_processor is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        
    try:
        image = Image.open(image_path).convert("RGB")
        prompts = [
            "a clean product review photo", 
            "an irrelevant photo or spam image like a screenshot, receipt, blank image, or text only"
        ]
        inputs = _clip_processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = _clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
        # If the probability of being irrelevant (index 1) is higher than relevant (index 0)
        return probs[1].item() > probs[0].item()
    except Exception:
        return False

