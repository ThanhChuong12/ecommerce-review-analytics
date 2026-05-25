import csv
import json
from io import BytesIO
from pathlib import Path
from PIL import Image
from google import genai
import time

# --- CẤU HÌNH ---
PROJECT_ID = "project-6a7957c9-bb93-4361-a68"
MODEL_NAME = "gemini-3.1-flash-lite-preview"
GROUND_TRUTH_CSV = "data/manifests/ground_truth.csv"

def get_few_shot_prompt(review_text, product_name):
    return (
        "You are an expert e-commerce media verifier. Analyze the relationship between the Image, Review Text, and Product Name.\n\n"
        "Your task is to classify the image into EXACTLY ONE label based on its physical condition and context. Remember: The ultimate goal is to prepare images for a Computer Vision model that only sees pixels, not text.\n\n"
        "Return ONLY a valid JSON object strictly formatted like this: {\"reasoning\": \"briefly explain the relationship between the image and the text, and justify your label choice\", \"label\": \"your_choice\"}.\n"
        "Valid labels: 'intact', 'damaged', 'wrong_item', 'irrelevant'.\n\n"
        
        "*** ABSOLUTE OVERRIDE RULE 1 (VISUAL CONDITION & VARIANTS) ***:\n"
        "Your primary job is to assess PHYSICAL condition. 'Wrong Variant' is NOT 'Wrong Item'. If the image shows the correct product CATEGORY (e.g., product is Milk Grow 1+, image is Milk Grow 3; or product is Black Shirt, image is Red Shirt), label it 'intact' because the physical object is undamaged and visually represents the category perfectly.\n\n"
        
        "*** ABSOLUTE OVERRIDE RULE 2 (COIN FARMING vs SELLER MISTAKE) ***:\n"
        "On e-commerce platforms, users often upload random irrelevant photos (pets, random objects, screenshots) to earn coins.\n"
        "IF THE IMAGE IS AN ENTIRELY DIFFERENT CATEGORY FROM THE PRODUCT (e.g., Product=Milk, Image=Shoes):\n"
        "  - Step A: Read the Review Text.\n"
        "  - Step B: If text is POSITIVE, NEUTRAL, or GIBBERISH (e.g., 'good', '5 stars', 'mmm') -> Coin farming spam. Label MUST BE 'irrelevant'.\n"
        "  - Step C: If and ONLY IF text is NEGATIVE and EXPLICITLY COMPLAINS about receiving the wrong item -> Label MUST BE 'wrong_item'.\n\n"
        
        "*** ABSOLUTE OVERRIDE RULE 3 (BLURRINESS & ANGLES) ***:\n"
        "CAMERA BLUR IS NOT DAMAGE. Blurry, out-of-focus, low-resolution images, or extreme close-ups (e.g., zooming in on a barcode or logo) of the correct product MUST be labeled 'intact'.\n\n"
        
        "*** ABSOLUTE OVERRIDE RULE 4 (PACKAGING & UNBOXING) ***:\n"
        "Images of brown shipping boxes, taped parcels, bubble wrap, or hands using scissors are NORMAL. DO NOT label them 'wrong_item' or 'irrelevant'. Unless the actual product inside is visibly broken, label the packaging/unboxing process as 'intact'.\n\n"
        
        "Label Definitions:\n"
        "- intact: CORRECT product category (physically fine, including variants, blur, close-ups, packaging).\n"
        "- damaged: ACTUAL PHYSICAL product/packaging is visibly broken, dented, crushed, torn, or leaking.\n"
        "- wrong_item: ENTIRELY DIFFERENT item + text complains about wrong delivery.\n"
        "- irrelevant: Random images, pets, selfies, black screens, screenshots, OR coin farming spam.\n\n"
        
        "*** EXAMPLES TO LEARN FROM (FEW-SHOT) ***\n"
        "Ex 1: Image = Brown box. Text = 'Hộp móp méo'. -> Label = 'intact' (Product inside not seen damaged yet, box is just packaging).\n"
        "Ex 2: Image = Blurry can of Abbott. Text = 'Sữa ngon'. -> Label = 'intact' (Blurry is not damaged).\n"
        "Ex 3: Image = Red Jeans. Product = Black Jeans. Text = 'Giao sai màu'. -> Label = 'intact' (Visually it's a perfectly fine pair of jeans, good for CV training).\n"
        "Ex 4: Image = Cute dog. Text = 'Giao hàng cực nhanh, shop uy tín'. -> Label = 'irrelevant' (Coin farming spam).\n"
        "Ex 5: Image = Can of Abbott Grow 3. Product = Abbott Grow 1+. Text = 'mmmmmmm'. -> Label = 'intact' (Visually an intact milk can, text is gibberish).\n"
        "Ex 6: Image = Can of milk with a huge dent. Text = 'Lon bị móp, thất vọng'. -> Label = 'damaged'.\n"
        "Ex 7: Image = A pair of shoes. Product = Abbott Grow Milk. Text = 'Shop làm ăn chán, đặt sữa giao giày'. -> Label = 'wrong_item'.\n"
        "Ex 8: Image = Black screen or Screenshot of Shopee order page. Text = 'Sữa thơm ngon'. -> Label = 'irrelevant' (Not a photo of a physical product).\n"
        "Ex 9: Image = Zoomed in picture of an expiration date. Text = 'Date xa'. -> Label = 'intact' (Close-up detail of the product).\n"
        "Ex 10: Image = Half-torn bubble wrap revealing a perfectly fine milk can. Text = 'Bọc kỹ'. -> Label = 'intact' (Normal unboxing).\n\n"
        
        f"Review text: {review_text}\n"
        f"Product name: {product_name}\n"
    )

def main():
    client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
    
    with open(GROUND_TRUTH_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    correct_count = 0
    print(f"Bắt đầu test {len(rows)} ảnh Ground Truth...\n" + "="*50)
    
    for row in rows:
        img_path = Path(row["image_path"])
        if not img_path.exists():
            print(f"[LỖI] Không tìm thấy ảnh: {img_path}")
            continue
            
        try:
            # Xử lý ảnh
            with Image.open(img_path) as img:
                rgb = img.convert("RGB")
                buf = BytesIO()
                rgb.save(buf, format="JPEG")
                image_part = genai.types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
            
            # Gọi API
            prompt = get_few_shot_prompt(row["review_text"], row["product_name"])
            response = client.models.generate_content(model=MODEL_NAME, contents=[prompt, image_part])
            
            # Phân tích JSON
            raw_text = response.text.replace('```json', '').replace('```', '').strip()
            data = json.loads(raw_text)
            ai_label = data.get("label")
            ai_reason = data.get("reasoning")
            
            expected = row["expected_label"]
            
            # Đánh giá
            if ai_label == expected:
                correct_count += 1
                print(f"✅ [PASS] Ảnh: {img_path.name} | AI: {ai_label} == ĐÁP ÁN: {expected} | Reason: {ai_reason}")
            else:
                print(f"❌ [FAIL] Ảnh: {img_path.name} | AI: {ai_label} != ĐÁP ÁN: {expected} | Review: {row['review_text']} | Reason: {ai_reason}")
                
        except Exception as e:
            print(f"⚠️ [LỖI API] Ảnh {img_path.name}: {e}")
            
        time.sleep(1) # Né Rate Limit
        
    print("="*50)
    print(f"🎯 KẾT QUẢ: {correct_count}/{len(rows)} ảnh dán nhãn ĐÚNG ({(correct_count/len(rows))*100}%)")

if __name__ == "__main__":
    main()