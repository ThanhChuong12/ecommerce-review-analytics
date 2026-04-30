
# NLP Pipeline Summary

Tai lieu nay tong hop cac thanh phan NLP da trien khai trong du an va mo ta ro cach gan nhan (label) cho van ban dua tren cac module hien co.

## 1. Muc tieu tong quan

Pipeline NLP tap trung vao ba muc tieu chinh:
- Chuan hoa van ban tieng Viet tu du lieu thu thap (scraping) de giam nhieu.
- Trich xuat dac trung (sparse va dense) phuc vu phan tich ngu nghia va cam xuc.
- Gan nhan cam xuc va khia canh (aspect) theo co che quyet dinh ro rang, co fallback an toan.

## 2. Tien xu ly van ban (preprocessor.py)

**TextCleaner** cung cap chuoi buoc lam sach chuan hoa van ban, thiet ke de chay nhanh tren tap lon.

**Cac buoc chinh:**
1. Ep kieu input ve chuoi an toan, tranh loi None hoac kieu bat thuong.
2. Viet thuong toan bo van ban.
3. Loai bo URL, HTML, so dien thoai, email, mention.
4. Chuyen emoji thanh alias va map sang tu khoa cam xuc tieng Viet (giu lai tin hieu sentiment).
5. Giam lap ky tu va dau cau bi keo dai.
6. Loc ky tu dac biet, giu lai ky tu tieng Viet, so, va dau cau can thiet.
7. Chuan hoa teencode bang tu dien TEEN_CODE_DICT.
8. Tokenize tieng Viet bang underthesea.
9. Chuan hoa khoang trang.

**Ly do thiet ke:**
- Giu lai tin hieu cam xuc tu emoji thay vi loai bo.
- Chuan hoa teencode de giam tu vung gia va gop cac bien the chinh ta.
- Tokenize phu hop dac thu tieng Viet de tang chat luong vector hoa va mo hinh.

## 3. Vector hoa van ban (vectorizers.py)

**TextVectorizer** la lop TF-IDF chuan hoa cho van ban tieng Viet:
- Ho tro n-gram (mac dinh 1-2) de bat cum tu.
- Chuan hoa token pattern giu ca tu 1 ky tu va token co dau gach duoi.
- Co the luu va tai mo hinh vector hoa bang joblib.

**Muc dich:** tao dac trung dang sparse de phuc vu cac mo hinh hoc may truyen thong va phan tich tu vung.

## 4. Embedding ngu nghia (embeddings.py)

**DeepEmbedder** dung Sentence-Transformers:
- Tu dong chon thiet bi (CUDA/MPS/CPU).
- Tao embedding da chuan hoa (normalize) de tinh cosine similarity on dinh.

**Muc dich:** tao bieu dien ngu nghia da chieu cao phuc vu truy van gan nghia va so khop khia canh (aspect).

## 5. Phan tich cam xuc va khia canh (sentiment_analysis.py)

**NextGenReviewAnalyzer** gom 2 chuc nang chinh:

### 5.1. Trich xuat khia canh (Aspect Extraction)
- Dinh nghia tap anchor phrase theo khia canh: shipping, product, price, service.
- Sinh embedding cho anchor, tinh cosine similarity voi review.
- Neu do tuong dong vuot nguong (mac dinh 0.65), gan khia canh tuong ung.

**Ly do:** day la co che don gian, giai thich duoc, va phu hop khi chua co nhan khia canh.

### 5.2. Du doan cam xuc (Sentiment Prediction)
- Su dung zero-shot classification (XLM-RoBERTa XNLI) voi 3 nhan: tich cuc, tieu cuc, trung lap.
- Lay top label tu output model.
- Neu diem tin cay thap hoac khong cach biet ro voi nhan thu hai, goi fallback LLM.

## 6. Co che gan nhan label cho text (LLM fallback)

Gan nhan cam xuc duoc thuc hien theo luong quyet dinh sau:
1. **Neu text rong**: tra ve nhan mac dinh "trung lap" de tranh loi va giam noise.
2. **Zero-shot classification** (chon nhan co score cao nhat).
3. **Nguong tin cay**:
	- Neu score top < 0.45, hoac chenhlech so voi nhan thu hai < 0.05, coi la khong chac chan.
	- Trong truong hop khong chac chan, chuyen sang LLM fallback.
4. **LLM fallback** (llm_client.py):
	- Ho tro Gemini, OpenAI, Grok theo bien moi truong `LLM_PROVIDER`.
	- He thong prompt bat buoc tra ve JSON: {"sentiment": "tich cuc" | "tieu cuc" | "trung lap"}.
	- Neu response loi hoac sai dinh dang, tra ve "trung lap".

**Dac tinh quan trong:**
- Quy trinh giam rui ro nhan sai khi model khong tu tin.
- Co che fallback giai quyet cac review mo ho hoac can kien thuc ngu canh.
- LLM khong can thiet neu zero-shot du tu tin, giup tiet kiem chi phi.

## 7. Tom tat cong viec da hoan thanh

- Xay dung TextCleaner cho tieng Viet voi teencode, emoji mapping, va tokenization.
- Tich hop TF-IDF vectorizer cho bieu dien sparse.
- Tich hop embedding sentence-transformers cho bieu dien dense.
- Trien khai bo phan tich cam xuc va khia canh co fallback an toan.

## 8. De xuat mo rong

- Luu them thong tin tin cay (confidence) vao output de phuc vu thresholding.
- Mo rong tap anchor khia canh hoac tu dong hoc anchor tu du lieu.
- Ket hop luat dua tren rating neu can nhan yeu (weak supervision) cho tap huan luyen.
