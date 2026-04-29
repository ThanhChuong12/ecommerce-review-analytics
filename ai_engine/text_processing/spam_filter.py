"""
spam_filter.py
--------------
Module phát hiện đánh giá spam / seeding cho hệ thống phân tích review thương mại
điện tử Việt Nam (Shopee, Tiki, Lazada, Thế Giới Di Động).

Module này là phần việc của Hiếu trong dự án Multimodal Review Analytics.
Toàn bộ logic được đóng gói thành các hàm thuần (pure function) để dễ test
và tái sử dụng trong notebook hoặc trong pipeline huấn luyện mô hình ML.

Triết lý thiết kế:
    - Rule-based thay vì ML để có thể giải thích được tại sao một review bị
      gắn cờ spam (interpretability quan trọng khi ra mắt sản phẩm).
    - Các luật được nhóm theo 4 trục:
        (1) Trục cấu trúc văn bản (độ dài, ký tự, emoji, in hoa, lặp).
        (2) Trục mẫu khuôn (template generic, copy-paste của bot).
        (3) Trục đặc thù sàn thương mại điện tử (nhận xu, mã giảm giá, link
            chuyển hướng, số điện thoại, quảng cáo shop đối thủ).
        (4) Trục bất thường giữa rating và nội dung (5 sao kèm chê, 1 sao
            kèm khen, 5 sao kèm text rác).
    - Phát hiện trùng lặp giữa các review bằng TF-IDF + Cosine Similarity
      để bắt seeding hàng loạt.

Hàm chính được notebook gọi:
    - detect_spam(df)       : Trả về dataframe có thêm đúng 1 cột is_spam (0/1).
    - summarize_spam(df)    : Tổng hợp thống kê chi tiết theo từng loại luật.
    - normalize_pipeline(s) : Pipeline chuẩn hóa văn bản nhiều bước.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import List, Set, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ====================================================================
#  1. CÁC TIỆN ÍCH TIỀN XỬ LÝ VĂN BẢN
# ====================================================================

# Regex bắt emoji thuộc nhiều dải Unicode khác nhau.
# Đây là tập hợp các block phổ biến nhất, không cần đầy đủ 100%
# vì chỉ cần đủ để đo tỉ lệ emoji trong câu.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # Mặt cười, cảm xúc
    "\U0001F300-\U0001F5FF"   # Biểu tượng và pictographs
    "\U0001F680-\U0001F6FF"   # Phương tiện và bản đồ
    "\U0001F1E0-\U0001F1FF"   # Cờ quốc gia
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"   # Bổ sung
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"              # Zero width joiner
    "]+",
    flags=re.UNICODE,
)

# Regex bắt URL (http, https, www, hoặc tên miền .com/.vn/.net... ).
_URL_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+|"
    r"\b[\w\-]+\.(?:com|vn|net|org|info|biz|shop|store|me|co|io)\b/?\S*",
    flags=re.IGNORECASE,
)

# Regex bắt thẻ HTML còn sót lại từ scraping.
_HTML_PATTERN = re.compile(r"<[^>]+>")

# Regex bắt mention dạng @username và hashtag dạng #tag.
_MENTION_PATTERN = re.compile(r"@\w+")
_HASHTAG_PATTERN = re.compile(r"#\w+")

# Regex bắt số điện thoại Việt Nam, kể cả các kiểu chèn dấu hoặc khoảng trắng
# để né bộ lọc của sàn (ví dụ: "0 9 0 1 234 567" hoặc "090.123.4567").
_PHONE_PATTERN = re.compile(
    r"(?:(?:\+?84|0)[\s\.\-]?)(?:3|5|7|8|9)(?:[\s\.\-]?\d){8}",
)

# Các pattern đặc thù của sàn thương mại điện tử Việt Nam.
# Người seeding hay viết các cụm này để né bộ lọc tự động hoặc dụ người đọc.
_CONTACT_KEYWORDS_PATTERN = re.compile(
    r"\b(zalo|z\.a\.l\.o|fb|f\.b|facebook|messenger|"
    r"telegram|tele|whatsapp|wa|line|viber|skype|email)\b",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Chuẩn hóa cơ bản: NFC unicode, lowercase, gộp khoảng trắng.

    Đây là bước chuẩn hóa nhẹ nhất, dùng cho hầu hết các luật so khớp.
    NFC quan trọng với tiếng Việt vì cùng một chữ có dấu có thể được mã hóa
    theo nhiều cách khác nhau (composed vs decomposed).
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def remove_emojis(text: str) -> str:
    """Xóa toàn bộ emoji ra khỏi chuỗi để xử lý phần chữ cho dễ."""
    if not isinstance(text, str):
        return ""
    return _EMOJI_PATTERN.sub("", text)


def normalize_pipeline(text: str) -> Tuple[str, Dict[str, int]]:
    """Pipeline chuẩn hóa văn bản hoàn chỉnh, trả về cả văn bản đã chuẩn
    và số liệu thống kê về số ký tự bị loại bỏ ở mỗi bước.

    Pipeline gồm các bước (theo gợi ý trong đề bài):
        1. Chuẩn hóa Unicode NFC.
        2. Chuyển về chữ thường.
        3. Loại bỏ thẻ HTML còn sót.
        4. Loại bỏ URL.
        5. Loại bỏ mention và hashtag.
        6. Chuẩn hóa khoảng trắng.

    Return:
        (text_đã_chuẩn, stats_dict)
        Trong đó stats_dict ghi số ký tự đã loại ở mỗi bước, dùng để báo cáo
        "tỉ lệ từ vựng thay đổi" trong notebook EDA.
    """
    if not isinstance(text, str):
        return "", {"original_len": 0, "html_removed": 0, "url_removed": 0,
                    "mention_removed": 0, "final_len": 0}

    original = text
    stats = {"original_len": len(original)}

    # Bước 1: Chuẩn hóa NFC.
    text = unicodedata.normalize("NFC", text)

    # Bước 2: Lowercase.
    text = text.lower()

    # Bước 3: Loại bỏ HTML.
    before_html = len(text)
    text = _HTML_PATTERN.sub(" ", text)
    stats["html_removed"] = before_html - len(text)

    # Bước 4: Loại bỏ URL.
    before_url = len(text)
    text = _URL_PATTERN.sub(" ", text)
    stats["url_removed"] = before_url - len(text)

    # Bước 5: Loại bỏ mention và hashtag.
    before_mention = len(text)
    text = _MENTION_PATTERN.sub(" ", text)
    text = _HASHTAG_PATTERN.sub(" ", text)
    stats["mention_removed"] = before_mention - len(text)

    # Bước 6: Chuẩn hóa khoảng trắng.
    text = re.sub(r"\s+", " ", text).strip()

    stats["final_len"] = len(text)
    return text, stats


def count_words(text: str) -> int:
    """Đếm số từ thực sự trong văn bản, đã loại emoji trước khi đếm."""
    cleaned = remove_emojis(normalize_text(text))
    return len(cleaned.split())


def count_chars(text: str) -> int:
    """Đếm số ký tự trong văn bản, không tính khoảng trắng."""
    if not isinstance(text, str):
        return 0
    return len(re.sub(r"\s+", "", text))


def get_emoji_ratio(text: str) -> float:
    """Tỉ lệ ký tự emoji trên tổng độ dài chuỗi (0.0 đến 1.0)."""
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    emojis = _EMOJI_PATTERN.findall(text)
    total_emoji_len = sum(len(e) for e in emojis)
    return total_emoji_len / len(text)


def get_special_char_ratio(text: str) -> float:
    """Tỉ lệ ký tự đặc biệt (không phải chữ cái, chữ số, khoảng trắng).

    Review thật vẫn có dấu câu, nhưng tỉ lệ quá cao thường là spam ký hiệu.
    """
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    text_no_emoji = remove_emojis(text)
    if len(text_no_emoji) == 0:
        return 0.0
    special_count = sum(
        1 for c in text_no_emoji
        if not (c.isalnum() or c.isspace())
    )
    return special_count / len(text_no_emoji)


def get_uppercase_ratio(text: str) -> float:
    """Tỉ lệ chữ cái viết hoa trên tổng số chữ cái trong câu."""
    if not isinstance(text, str):
        return 0.0
    only_letters = [c for c in text if c.isalpha()]
    if len(only_letters) < 10:
        # Câu quá ngắn không xét tiêu chí này.
        return 0.0
    upper_count = sum(1 for c in only_letters if c.isupper())
    return upper_count / len(only_letters)


def get_digit_ratio(text: str) -> float:
    """Tỉ lệ ký tự là chữ số trên tổng độ dài.

    Spam mã giảm giá hoặc chuỗi số rác có tỉ lệ chữ số rất cao.
    """
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    digit_count = sum(1 for c in text if c.isdigit())
    return digit_count / len(text)


def get_type_token_ratio(text: str) -> float:
    """Tính Type-Token Ratio (TTR) = số từ duy nhất / tổng số từ.

    TTR đo độ phong phú từ vựng. Spam có TTR thấp (lặp lại nhiều) còn review
    chất lượng cao có TTR cao. Đây là một trong các kỹ thuật được gợi ý ở
    phần EDA bắt buộc của đề bài.
    """
    cleaned = remove_emojis(normalize_text(text))
    tokens = cleaned.split()
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)


# ====================================================================
#  2. CÁC LUẬT TRỤC CẤU TRÚC VĂN BẢN
# ====================================================================

def is_too_short(text: str, min_words: int = 3) -> bool:
    """Review có ít hơn min_words từ thì coi là không có thông tin.

    Ngưỡng 3 từ là chuẩn vì các câu kiểu "ok", "tốt", "ok shop" không cung
    cấp được gì cho phân tích cảm xúc theo khía cạnh.
    """
    return count_words(text) < min_words


def is_too_long(text: str, max_words: int = 500) -> bool:
    """Review dài bất thường thường là copy lời bài hát, thơ hoặc spam đoạn dài."""
    return count_words(text) > max_words


def is_emoji_only(text: str) -> bool:
    """Văn bản không có chữ cái nào, chỉ toàn emoji hoặc dấu câu vô nghĩa."""
    if not isinstance(text, str) or not text.strip():
        return True
    cleaned = remove_emojis(text)
    cleaned = re.sub(r"[^\w]", "", cleaned, flags=re.UNICODE)
    return len(cleaned.strip()) == 0


def is_keyboard_spam(text: str) -> bool:
    """Phát hiện gõ phím loạn hoặc lặp ký tự để câu chữ.

    Hai mẫu phổ biến:
        - Một ký tự lặp >= 5 lần liên tiếp: "aaaaaa", "kkkkk".
        - Một cụm 1-3 ký tự lặp >= 4 lần: "hahahaha", "lololololo".
    """
    norm = normalize_text(text)
    if re.search(r"(.)\1{4,}", norm):
        return True
    if re.search(r"(.{1,3})\1{3,}", norm):
        return True
    return False


def is_word_repetition(text: str, max_repeat: int = 4) -> bool:
    """Phát hiện một từ bị lặp liên tiếp >= max_repeat lần.

    Ví dụ: "tốt tốt tốt tốt tốt tốt" - kiểu spam để câu chữ cho đủ độ dài tối
    thiểu của sàn (Shopee yêu cầu 50 ký tự để được nhận xu).
    """
    cleaned = normalize_text(remove_emojis(text))
    tokens = cleaned.split()
    if len(tokens) < max_repeat:
        return False
    repeat_count = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            repeat_count += 1
            if repeat_count >= max_repeat:
                return True
        else:
            repeat_count = 1
    return False


def has_too_many_special_chars(text: str, threshold: float = 0.4) -> bool:
    """Tỉ lệ ký tự đặc biệt >= threshold (mặc định 40%) thì coi là spam ký hiệu."""
    if count_words(text) < 5:
        return False
    return get_special_char_ratio(text) >= threshold


def has_too_many_uppercase(text: str, threshold: float = 0.6) -> bool:
    """Câu dài mà tỉ lệ chữ in hoa >= 60% là dấu hiệu spam quảng cáo / hét lên."""
    return get_uppercase_ratio(text) >= threshold


def is_only_digits_or_punct(text: str) -> bool:
    """Văn bản chỉ toàn số hoặc dấu câu, không có chữ cái nào.

    Bắt các kiểu spam như "1234567890" hoặc "..........." hoặc "??????".
    """
    cleaned = remove_emojis(normalize_text(text))
    if not cleaned:
        return True
    has_letter = any(c.isalpha() for c in cleaned)
    return not has_letter


# ====================================================================
#  3. CÁC LUẬT TRỤC MẪU KHUÔN (TEMPLATE)
# ====================================================================

# Danh sách các mẫu câu khen chung chung mà bot và người seeding thường viết.
# Đã bao phủ cả có dấu và không dấu (vì người Việt hay gõ thiếu dấu).
_GENERIC_TEMPLATE_PATTERNS = [
    r"^s(ả|a)n ph(ẩ|a)m t(ố|o)t$",
    r"^h(à|a)ng t(ố|o)t$",
    r"^giao h(à|a)ng nhanh$",
    r"^(đ|d)(ó|o)ng g(ó|o)i c(ẩ|a)n th(ậ|a)n$",
    r"^t(ố|o)t$",
    r"^ok$",
    r"^oki$",
    r"^okela$",
    r"^good$",
    r"^nice$",
    r"^perfect$",
    r"^r(ấ|a)t t(ố|o)t$",
    r"^tuy(ệ|e)t v(ờ|o)i$",
    r"^h(à|a)i l(ò|o)ng$",
    r"^5 sao$",
    r"^r(ấ|a)t h(à|a)i l(ò|o)ng$",
    r"^ch(ấ|a)t l(ượ|uo)ng t(ố|o)t$",
    r"^(đ|d)(ú|u)ng m(ô|o) t(ả|a)$",
    r"^shop uy t(í|i)n$",
    r"^nh(ư|u) m(ô|o) t(ả|a)$",
    r"^\.*$",
    r"^\.+\s*$",
]
_GENERIC_TEMPLATE_COMPILED = [
    re.compile(p, re.IGNORECASE) for p in _GENERIC_TEMPLATE_PATTERNS
]


def is_generic_template(text: str) -> bool:
    """Khớp với một trong các mẫu câu generic do bot hoặc seeding tạo."""
    cleaned = normalize_text(remove_emojis(text)).strip()
    if not cleaned:
        return False
    for pattern in _GENERIC_TEMPLATE_COMPILED:
        if pattern.match(cleaned):
            return True
    return False


# ====================================================================
#  4. CÁC LUẬT TRỤC ĐẶC THÙ SÀN THƯƠNG MẠI ĐIỆN TỬ
# ====================================================================

# Cụm từ ám chỉ người viết chỉ đang nhận xu, nhận thưởng của sàn.
_XU_FARMING_PATTERNS = [
    r"nh(ậ|a)n xu",
    r"(đ|d)(ể|e) nh(ậ|a)n xu",
    r"comment nh(ậ|a)n xu",
    r"(đ|d)(á|a)nh gi(á|a) nh(ậ|a)n xu",
    r"review nh(ậ|a)n xu",
    r"(đ|d)(ủ|u) k(ý|y) t(ự|u)",
    r"cho (đ|d)(ủ|u) k(ý|y) t(ự|u)",
    r"vi(ế|e)t cho (đ|d)(ủ|u)",
    r"(đ|d)(ủ|u) ch(ữ|u)",
    r"g(õ|o) cho (đ|d)(ủ|u)",
    r"(đ|d)(á|a)nh gi(á|a) (đ|d)(ể|e) (đ|d)(ư|u)(ợ|o)c",
    r"(đ|d)(á|a)nh cho (đ|d)(ủ|u)",
    r"livestream",
    r"shopee xu",
    r"lazada xu",
    r"săn xu",
    r"hoàn xu",
    r"hoan xu",
]
_XU_FARMING_COMPILED = [re.compile(p, re.IGNORECASE) for p in _XU_FARMING_PATTERNS]


def is_xu_farming(text: str) -> bool:
    """Phát hiện bình luận có ý đồ chỉ để nhận xu / hoàn xu của sàn.

    Đây là dạng spam phổ biến nhất trên Shopee và Tiki khi sàn có chương
    trình thưởng coin cho việc đăng review. Người dùng gõ bừa cho đủ ký tự.
    """
    norm = normalize_text(text)
    for pattern in _XU_FARMING_COMPILED:
        if pattern.search(norm):
            return True
    return False


def contains_contact_info(text: str) -> bool:
    """Phát hiện review chứa thông tin liên hệ (sđt, zalo, fb, telegram).

    Đây là chiêu chuyển hướng giao dịch ra ngoài sàn để né phí và lừa đảo.
    Sàn TMĐT chính thống cấm hành vi này.
    """
    if not isinstance(text, str):
        return False
    if _PHONE_PATTERN.search(text):
        return True
    if _CONTACT_KEYWORDS_PATTERN.search(text):
        return True
    return False


def contains_external_link(text: str) -> bool:
    """Phát hiện review chứa URL hoặc link bên ngoài.

    Review thật hầu như không bao giờ chứa link. Spam thường chèn link
    sản phẩm khác hoặc link lừa đảo.
    """
    if not isinstance(text, str):
        return False
    return bool(_URL_PATTERN.search(text))


def contains_competitor_promotion(text: str) -> bool:
    """Phát hiện review quảng cáo / chuyển hướng sang sàn hoặc shop đối thủ.

    Chỉ flag khi có dấu hiệu rõ ràng của hành vi chuyển hướng (kiểu
    "qua bên kia rẻ hơn", "mua tại shop khác") để giảm false positive.
    """
    if not isinstance(text, str):
        return False
    norm = normalize_text(text)
    redirect_patterns = [
        "mua tại", "mua bên", "qua bên", "sang bên",
        "bên kia rẻ hơn", "chỗ khác rẻ", "shop khác rẻ hơn",
        "tìm shop khác", "không nên mua shop này",
    ]
    if any(rp in norm for rp in redirect_patterns):
        return True
    return False


# Pattern bắt mã giảm giá rác kiểu "MAGIAM50K", "FREESHIP100K", "DISCOUNT50".
_PROMO_CODE_PATTERN = re.compile(
    r"\b[A-Z]{4,}\d{2,}[A-Z0-9]*\b|"
    r"\b(magiam|freeship|free ship|voucher|coupon|"
    r"giảm giá \d+%|sale off \d+|mã \w+\d+)\b",
    flags=re.IGNORECASE,
)


def contains_promo_code_spam(text: str) -> bool:
    """Phát hiện review chèn mã giảm giá / voucher rác.

    Đây là chiêu của các shop khác đăng review giả để rải mã giảm giá của
    họ vào trang sản phẩm đối thủ.
    """
    if not isinstance(text, str):
        return False
    matches = _PROMO_CODE_PATTERN.findall(text)
    return len(matches) >= 1


# ====================================================================
#  5. CÁC LUẬT TRỤC BẤT THƯỜNG RATING - NỘI DUNG
# ====================================================================

# Từ khóa tiêu cực tiếng Việt, dùng để phát hiện mâu thuẫn rating cao - text chê.
_NEGATIVE_KEYWORDS = [
    "tệ", "tồi", "dở", "kém", "thất vọng", "chán", "hỏng", "vỡ",
    "lỗi", "gãy", "rách", "bẩn", "giả", "fake", "scam", "lừa",
    "không tốt", "không ok", "dở tệ", "không hài lòng", "không đáng",
    "không nên mua", "đừng mua", "không giống", "khác mô tả",
    "giao sai", "giao nhầm", "hư", "méo", "móp", "trầy", "xước",
    "mùi hôi", "hôi", "thối", "quá tệ", "rất tệ", "cực kỳ tệ",
    "không đúng", "không chất lượng", "đồ rác", "rác", "kinh khủng",
    "thảm họa", "chả ra gì", "vứt đi",
]

# Từ khóa tích cực tiếng Việt, dùng để phát hiện 1 sao kèm khen.
_POSITIVE_KEYWORDS = [
    "rất tốt", "tuyệt vời", "tuyệt", "xuất sắc", "hoàn hảo",
    "rất hài lòng", "hài lòng", "ưng ý", "đáng tiền", "đáng đồng tiền",
    "chất lượng tốt", "chất lượng cao", "đẹp", "siêu đẹp",
    "yêu shop", "yêu lắm", "perfect", "amazing", "good",
]


def has_rating_text_mismatch(text: str, rating) -> bool:
    """Rating cao (>= 4 sao) nhưng text chứa nhiều từ tiêu cực, hoặc ngược lại.

    Có thể do bot rải rating ngẫu nhiên hoặc người dùng nhấn nhầm sao.
    Yêu cầu >= 2 từ khóa cùng dấu để giảm báo nhầm.
    """
    try:
        stars = int(float(str(rating)))
    except (ValueError, TypeError):
        return False
    norm = normalize_text(text)
    # Trường hợp 1: 4-5 sao nhưng text tiêu cực.
    if stars >= 4:
        neg_count = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in norm)
        if neg_count >= 2:
            return True
    # Trường hợp 2: 1-2 sao nhưng text khen ngợi.
    if stars <= 2:
        pos_count = sum(1 for kw in _POSITIVE_KEYWORDS if kw in norm)
        if pos_count >= 2:
            return True
    return False


def is_five_star_junk(text: str, rating) -> bool:
    """5 sao nhưng nội dung không có giá trị thông tin.

    Bao gồm 3 trường hợp con: chỉ emoji, quá ngắn, hoặc spam bàn phím.
    """
    try:
        stars = int(float(str(rating)))
    except (ValueError, TypeError):
        return False
    if stars != 5:
        return False
    return is_emoji_only(text) or is_too_short(text) or is_keyboard_spam(text)


# ====================================================================
#  6. PHÁT HIỆN TRÙNG LẶP GIỮA CÁC REVIEW (SEEDING)
# ====================================================================

def find_duplicate_clusters(
    texts: List[str],
    threshold: float = 0.85,
    min_text_len: int = 10,
) -> List[Set[int]]:
    """Tìm các nhóm review có nội dung gần như giống nhau bằng TF-IDF + Cosine.

    Phương pháp:
        1. Lọc bỏ text quá ngắn (< min_text_len ký tự) để tránh false positive.
        2. Vector hóa bằng TF-IDF với unigram + bigram.
        3. Tính cosine similarity theo batch để tiết kiệm RAM.
        4. Gom các index có similarity >= threshold thành cluster.
        5. Merge các cluster có giao nhau.

    Return:
        Danh sách các set index, mỗi set là một cluster review trùng nhau.
    """
    valid_indices = [
        i for i, t in enumerate(texts)
        if len(str(t).strip()) >= min_text_len
    ]
    if len(valid_indices) < 2:
        return []

    valid_texts = [str(texts[i]) for i in valid_indices]

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(valid_texts)

    clusters: List[Set[int]] = []
    seen: Set[int] = set()
    batch_size = 500

    for start in range(0, len(valid_indices), batch_size):
        end = min(start + batch_size, len(valid_indices))
        sim_matrix = cosine_similarity(tfidf_matrix[start:end], tfidf_matrix)

        for local_i in range(end - start):
            global_i = start + local_i
            real_i = valid_indices[global_i]
            if real_i in seen:
                continue
            similar = set()
            for j in range(len(valid_indices)):
                if j == global_i:
                    continue
                if sim_matrix[local_i, j] >= threshold:
                    similar.add(valid_indices[j])
            if similar:
                new_cluster = {real_i} | similar
                merged = False
                for existing in clusters:
                    if existing & new_cluster:
                        existing |= new_cluster
                        merged = True
                        break
                if not merged:
                    clusters.append(new_cluster)
                seen |= new_cluster

    return clusters


def flag_duplicates(texts: List[str], threshold: float = 0.85) -> List[bool]:
    """Trả về list[bool] song song với input.

    Phần tử thứ i là True nếu review nằm trong một cluster trùng lặp.
    """
    clusters = find_duplicate_clusters(texts, threshold=threshold)
    dup_indices: Set[int] = set()
    for cluster in clusters:
        dup_indices |= cluster
    return [i in dup_indices for i in range(len(texts))]


# ====================================================================
#  7. PIPELINE TỔNG HỢP - HÀM CHÍNH ĐƯỢC GỌI TỪ NOTEBOOK
# ====================================================================

def detect_spam(df: pd.DataFrame, dup_threshold: float = 0.85) -> pd.DataFrame:
    """Hàm chính: nhận dataframe đầu vào, trả về dataframe có thêm cột is_spam.

    Đầu vào phải có ít nhất 2 cột: text và rating.
    Đầu ra: giữ nguyên các cột gốc và bổ sung đúng 1 cột mới là is_spam,
    nhận giá trị 1 nếu review bị nghi spam và 0 nếu sạch.

    Hàm còn đính kèm dataframe chi tiết các cờ thông qua attribute
    `flag_details` để notebook có thể dùng cho EDA, không ảnh hưởng tới
    file CSV xuất ra.
    """
    if "text" not in df.columns or "rating" not in df.columns:
        raise ValueError("Dataframe phải có 2 cột: text và rating")

    result = df.copy()
    texts = result["text"].fillna("").astype(str).tolist()
    ratings = result["rating"].tolist()

    print("[spam_filter] Đang chấm các luật ở mức từng review...")

    # Trục cấu trúc văn bản.
    flag_too_short = [is_too_short(t) for t in texts]
    flag_too_long = [is_too_long(t) for t in texts]
    flag_emoji_only = [is_emoji_only(t) for t in texts]
    flag_keyboard = [is_keyboard_spam(t) for t in texts]
    flag_word_repeat = [is_word_repetition(t) for t in texts]
    flag_special = [has_too_many_special_chars(t) for t in texts]
    flag_uppercase = [has_too_many_uppercase(t) for t in texts]
    flag_only_digits = [is_only_digits_or_punct(t) for t in texts]

    # Trục mẫu khuôn.
    flag_generic = [is_generic_template(t) for t in texts]

    # Trục đặc thù sàn TMĐT.
    flag_xu = [is_xu_farming(t) for t in texts]
    flag_contact = [contains_contact_info(t) for t in texts]
    flag_link = [contains_external_link(t) for t in texts]
    flag_competitor = [contains_competitor_promotion(t) for t in texts]
    flag_promo = [contains_promo_code_spam(t) for t in texts]

    # Trục bất thường rating - text.
    flag_5star_junk = [is_five_star_junk(t, r) for t, r in zip(texts, ratings)]
    flag_mismatch = [
        has_rating_text_mismatch(t, r) for t, r in zip(texts, ratings)
    ]

    # Trục trùng lặp giữa các review.
    print("[spam_filter] Đang tính cosine similarity để tìm review trùng lặp...")
    flag_duplicate = flag_duplicates(texts, threshold=dup_threshold)

    # Gộp tất cả thành quyết định cuối: review có spam hay không.
    all_flags = list(zip(
        flag_too_short, flag_too_long, flag_emoji_only, flag_keyboard,
        flag_word_repeat, flag_special, flag_uppercase, flag_only_digits,
        flag_generic, flag_xu, flag_contact, flag_link, flag_competitor,
        flag_promo, flag_5star_junk, flag_mismatch, flag_duplicate,
    ))
    is_spam = [1 if any(flags) else 0 for flags in all_flags]
    result["is_spam"] = is_spam

    # Lưu dataframe chi tiết các cờ vào attribute (không xuất ra file CSV).
    flag_details = pd.DataFrame({
        "too_short": flag_too_short,
        "too_long": flag_too_long,
        "emoji_only": flag_emoji_only,
        "keyboard_spam": flag_keyboard,
        "word_repetition": flag_word_repeat,
        "too_many_special_chars": flag_special,
        "too_many_uppercase": flag_uppercase,
        "only_digits_or_punct": flag_only_digits,
        "generic_template": flag_generic,
        "xu_farming": flag_xu,
        "contact_info": flag_contact,
        "external_link": flag_link,
        "competitor_promotion": flag_competitor,
        "promo_code_spam": flag_promo,
        "five_star_junk": flag_5star_junk,
        "rating_text_mismatch": flag_mismatch,
        "duplicate_seeding": flag_duplicate,
    }, index=result.index)
    result.attrs["flag_details"] = flag_details

    print(
        f"[spam_filter] Hoàn tất. Tổng review: {len(result):,}, "
        f"Nghi spam: {sum(is_spam):,}"
    )
    return result


def summarize_spam(df: pd.DataFrame) -> dict:
    """Tổng hợp các con số thống kê từ dataframe đã được phát hiện spam.

    Cần dataframe trả về từ hàm detect_spam (có cột is_spam và attribute
    flag_details).
    """
    if "is_spam" not in df.columns:
        raise ValueError("Dataframe phải có cột is_spam, hãy chạy detect_spam trước")

    total = len(df)
    spam_count = int(df["is_spam"].sum())
    clean_count = total - spam_count

    stats = {
        "total_reviews": total,
        "spam_count": spam_count,
        "clean_count": clean_count,
        "spam_pct": round(spam_count / total * 100, 2) if total else 0.0,
        "clean_pct": round(clean_count / total * 100, 2) if total else 0.0,
        "breakdown": {},
    }

    # Map tên cột nội bộ sang nhãn tiếng Việt cho dễ đọc khi vẽ biểu đồ.
    label_map = {
        "too_short": "Quá ngắn (<3 từ)",
        "too_long": "Quá dài (>500 từ)",
        "emoji_only": "Chỉ toàn emoji",
        "keyboard_spam": "Spam bàn phím",
        "word_repetition": "Lặp từ liên tục",
        "too_many_special_chars": "Quá nhiều ký tự đặc biệt",
        "too_many_uppercase": "Quá nhiều chữ in hoa",
        "only_digits_or_punct": "Chỉ số và dấu câu",
        "generic_template": "Mẫu generic / template",
        "xu_farming": "Bình luận nhận xu",
        "contact_info": "Chứa thông tin liên hệ",
        "external_link": "Chứa link bên ngoài",
        "competitor_promotion": "Quảng cáo shop đối thủ",
        "promo_code_spam": "Mã giảm giá rác",
        "five_star_junk": "5 sao kèm text rác",
        "rating_text_mismatch": "Mâu thuẫn rating - nội dung",
        "duplicate_seeding": "Trùng lặp (seeding)",
    }

    flag_details = df.attrs.get("flag_details")
    if flag_details is not None:
        for col, label in label_map.items():
            if col in flag_details.columns:
                count = int(flag_details[col].sum())
                stats["breakdown"][label] = {
                    "count": count,
                    "pct": round(count / total * 100, 2) if total else 0.0,
                }
    return stats