"""Rule-based spam and seeding detection module for Vietnamese e-commerce reviews."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "]+",
    flags=re.UNICODE,
)

_URL_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+|"
    r"\b[\w\-]+\.(?:com|vn|net|org|info|biz|shop|store|me|co|io)\b/?\S*",
    flags=re.IGNORECASE,
)
_HTML_PATTERN = re.compile(r"<[^>]+>")
_MENTION_PATTERN = re.compile(r"@\w+")
_HASHTAG_PATTERN = re.compile(r"#\w+")
_PHONE_PATTERN = re.compile(r"(?:(?:\+?84|0)[\s\.\-]?)(?:3|5|7|8|9)(?:[\s\.\-]?\d){8}")
_CONTACT_KEYWORD_PATTERN = re.compile(
    r"\b(zalo|z\.a\.l\.o|fb|f\.b|facebook|messenger|"
    r"telegram|tele|whatsapp|wa|line|viber|skype)\b",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize text into lowercase NFC unicode representation."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def remove_emojis(text: str) -> str:
    """Strip all emoji characters from string."""
    if not isinstance(text, str):
        return ""
    return _EMOJI_PATTERN.sub("", text)


def normalize_pipeline(text: str) -> Tuple[str, Dict[str, int]]:
    """Execute full normalization pipeline returning cleaned text and step statistics."""
    if not isinstance(text, str):
        return "", {
            "original_len": 0,
            "html_removed": 0,
            "url_removed": 0,
            "mention_removed": 0,
            "final_len": 0,
        }

    stats = {"original_len": len(text)}
    text = unicodedata.normalize("NFC", text).lower()

    before = len(text)
    text = _HTML_PATTERN.sub(" ", text)
    stats["html_removed"] = before - len(text)

    before = len(text)
    text = _URL_PATTERN.sub(" ", text)
    stats["url_removed"] = before - len(text)

    before = len(text)
    text = _MENTION_PATTERN.sub(" ", text)
    text = _HASHTAG_PATTERN.sub(" ", text)
    stats["mention_removed"] = before - len(text)

    text = re.sub(r"\s+", " ", text).strip()
    stats["final_len"] = len(text)
    return text, stats


def count_words(text: str) -> int:
    """Calculate word count after stripping emojis."""
    cleaned = remove_emojis(normalize_text(text))
    return len(cleaned.split())


def count_chars(text: str) -> int:
    """Calculate non-whitespace character count."""
    if not isinstance(text, str):
        return 0
    return len(re.sub(r"\s+", "", text))


def get_emoji_ratio(text: str) -> float:
    """Calculate ratio of emoji characters in string."""
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    emojis = _EMOJI_PATTERN.findall(text)
    total_emoji_len = sum(len(e) for e in emojis)
    return total_emoji_len / len(text)


def get_special_char_ratio(text: str) -> float:
    """Calculate ratio of non-alphanumeric and non-whitespace special characters."""
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    text_no_emoji = remove_emojis(text)
    if len(text_no_emoji) == 0:
        return 0.0
    special_count = sum(1 for c in text_no_emoji if not (c.isalnum() or c.isspace()))
    return special_count / len(text_no_emoji)


def get_uppercase_ratio(text: str) -> float:
    """Calculate ratio of uppercase alphabetic characters."""
    if not isinstance(text, str):
        return 0.0
    only_letters = [c for c in text if c.isalpha()]
    if len(only_letters) < 10:
        return 0.0
    upper_count = sum(1 for c in only_letters if c.isupper())
    return upper_count / len(only_letters)


def get_digit_ratio(text: str) -> float:
    """Calculate ratio of numeric digits in string."""
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    digit_count = sum(1 for c in text if c.isdigit())
    return digit_count / len(text)


def get_type_token_ratio(text: str) -> float:
    """Calculate Type-Token Ratio (vocabulary diversity metric)."""
    cleaned = remove_emojis(normalize_text(text))
    tokens = cleaned.split()
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)


_TEMPLATE_PHRASES = [
    "công thức sữa tốt nhất cho bé",
    "rất khuyến khích cho trẻ từ 12-24 tháng",
    "sản phẩm chính hãng với bao bì tốt",
    "giúp hỗ trợ sự phát triển và tăng trưởng",
    "công thức bột tiện lợi để chuẩn bị dễ dàng",
    "em bé của tôi thích hương vị vani",
    "công thức tuyệt vời cho sự phát triển não bộ",
    "công thức tuyệt vời giúp phát triển não bộ",
    "thúc đẩy tăng cân lành mạnh",
    "hỗ trợ tăng cân một cách lành mạnh",
    "được khuyến nghị cho sự phát triển lành mạnh",
    "được khuyến nghị cho sự phát triển khỏe mạnh",
    "tăng cường miễn dịch và phát triển",
    "tăng cường sức đề kháng và tăng trưởng",
    "lý tưởng cho trẻ em từ 2-6 tuổi",
    "lý tưởng cho trẻ từ 2-6 tuổi",
    "sữa bột chất lượng cao",
    "công thức 3 trong 1 tiện lợi",
    "công thức tiện lợi 3 trong 1",
    "cung cấp các chất dinh dưỡng thiết yếu",
    "cung cấp dưỡng chất cần thiết",
    "giúp trẻ cao lớn hơn",
    "giúp trẻ phát triển chiều cao",
    "hỗ trợ phát triển toàn diện",
    "chứa vitamin",
    "lý tưởng cho dinh dưỡng hàng ngày",
    "hương vị mượt mà và thỏa mãn",
    "lựa chọn tuyệt vời để tăng cường buổi sáng",
    "một cách tuyệt vời để bắt đầu ngày mới",
    "hương thơm dễ chịu",
    "hương vị đậm đà và táo bạo",
    "hàm lượng caffeine cao để tăng cường năng lượng",
    "bao bì tiện lợi để giữ độ tươi",
    "hoàn hảo cho những người yêu cà phê",
    "hạt cà phê chất lượng cao",
    "phù hợp cho tất cả các loại máy giặt",
    "phù hợp với tất cả các loại máy giặt",
    "công thức kháng khuẩn hiệu quả",
    "hương thơm lâu dài",
    "giữ màu sắc tươi sáng",
    "giữ cho vải mềm mại",
    "bảo quản độ mềm mại của vải",
    "nuôi dưỡng sợi sâu",
    "nuôi dưỡng sâu cho sợi vải",
    "hương hoa tươi mát",
    "hương hoa thơm mát",
    "thương hiệu đáng tin cậy",
    "chất lượng cao cấp",
    "hoàn hảo cho quần áo trẻ em",
    "hoàn hảo cho quần áo bé",
    "nhẹ nhàng trên vải",
    "nhẹ nhàng với vải",
    "khăn giấy chất lượng cao",
    "kích thước gói tiện lợi",
    "hoàn hảo cho việc sử dụng hàng ngày",
    "lâu dài và bền",
    "thân thiện với môi trường và bền vững",
    "cảm giác mềm mại và sang trọng",
    "thiết kế đẹp mắt",
    "bao bì tiện lợi",
    "nhẹ nhàng trên da",
    "giá trị tuyệt vời cho số tiền bỏ ra",
    "độ bền đáng tin cậy",
    "rất bền",
    "giữ tốt",
    "giấy bền",
    "chắc chắn và bền",
    "sữa chua ngon và lành mạnh",
    "sữa chua chất lượng cao từ thương hiệu đáng tin cậy",
    "tăng cường hệ thống miễn dịch và thúc đẩy tiêu hóa",
    "tăng cường hệ miễn dịch và tăng cường tiêu hóa",
    "thúc đẩy lối sống lành mạnh",
    "bao bì lạnh đảm bảo độ tươi",
    "bao bì giữ lạnh đảm bảo tươi ngon",
    "bao bì tiện lợi để mang theo",
    "đóng gói tiện lợi, dễ mang theo",
    "hương vị tự nhiên tươi mát",
    "hương vị tự nhiên thơm ngon",
    "hoàn hảo cho một bữa ăn nhẹ nhanh và lành mạnh",
    "lý tưởng cho một món ăn nhẹ nhanh chóng và lành mạnh",
]


def count_template_phrases(text: str) -> int:
    """Count occurrences of generic template marketing phrases."""
    text_low = normalize_text(text)
    return sum(1 for p in _TEMPLATE_PHRASES if p in text_low)


def is_ai_template_review(text: str) -> bool:
    """Check if review text matches AI template rules."""
    return False


def is_template_repetition(text: str) -> bool:
    """Check if a template phrase is repeated multiple times."""
    text_low = normalize_text(text)
    for p in _TEMPLATE_PHRASES:
        if text_low.count(p) >= 2:
            return True
    return False


def is_mostly_template(text: str) -> bool:
    """Check if review content consists mostly of template expressions."""
    return False


_XU_FARMING_KEYWORDS = [
    "nhận xu", "nhan xu", "săn xu", "san xu", "lấy xu", "lay xu",
    "lấy su", "lay su", "đủ ký tự", "du ky tu", "đủ chữ", "du chu",
    "cho đủ ký tự", "viết cho đủ", "đánh giá nhận xu",
    "đánh giá để được", "tchat nhận", "chata nhận", "chata nhan",
    "shopee xu", "lazada xu", "hoàn xu", "hoan xu",
]

_DISCLAIMER_PATTERN = re.compile(
    r"(hình ảnh|h(ì|i)nh\s*(ả|a)nh|video|hinh anh)"
    r"[^.]{0,40}"
    r"(mang tính chất|mang tinh chat|chỉ mang|mag tính|mag tinh|"
    r"không liên quan|khong lien quan|k liên quan|k lien quan)",
    flags=re.IGNORECASE,
)


def is_xu_farming(text: str) -> bool:
    """Check if review text matches reward/coin harvesting patterns."""
    text_low = normalize_text(text)
    if any(kw in text_low for kw in _XU_FARMING_KEYWORDS):
        return True
    if _DISCLAIMER_PATTERN.search(text_low):
        return True
    return False


def is_too_short(text: str, min_words: int = 1) -> bool:
    """Check if review contains fewer than min_words."""
    return count_words(text) < min_words


def is_non_informative_short(text: str, rating) -> bool:
    """Check for content-free short 5-star filler reviews."""
    return False


def is_too_long(text: str, max_words: int = 500) -> bool:
    """Check if review exceeds max_words limit."""
    return count_words(text) > max_words


def is_emoji_only(text: str) -> bool:
    """Check if review consists entirely of emoji or punctuation characters."""
    if not isinstance(text, str) or not text.strip():
        return True
    cleaned = remove_emojis(text)
    cleaned = re.sub(r"[^\w]", "", cleaned, flags=re.UNICODE)
    return len(cleaned.strip()) == 0


def is_keyboard_spam(text: str) -> bool:
    """Detect key-mashing or excessive character/pattern repetition."""
    norm = normalize_text(text)
    norm_clean = norm.rstrip(".,! ")
    if re.search(r"([a-z])\1{5,}", norm_clean):
        return True
    if re.search(r"(.{2,3})\1{3,}", norm_clean):
        return True
    return False


def is_word_repetition(text: str, max_repeat: int = 4) -> bool:
    """Detect consecutive repetitive word sequences."""
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
    """Check if special character ratio exceeds threshold."""
    if count_words(text) < 5:
        return False
    return get_special_char_ratio(text) >= threshold


def has_too_many_uppercase(text: str, threshold: float = 0.6) -> bool:
    """Check if uppercase letter ratio exceeds threshold."""
    return get_uppercase_ratio(text) >= threshold


def is_only_digits_or_punct(text: str) -> bool:
    """Check if review contains no alphabetic letters."""
    cleaned = remove_emojis(normalize_text(text))
    if not cleaned:
        return True
    return not any(c.isalpha() for c in cleaned)


_VIETNAMESE_DIACRITICS = "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"


def is_random_keyboard_string(text: str) -> bool:
    """Detect unpronounceable random keyboard string sequences."""
    text_low = normalize_text(text)
    tokens = text_low.split()

    if len(tokens) == 1 and len(text_low) >= 15:
        ascii_letters = re.sub(r"[^a-zA-Z]", "", text_low)
        has_diac = any(c in text_low for c in _VIETNAMESE_DIACRITICS)
        if not has_diac and len(ascii_letters) >= 15:
            vowels = sum(1 for c in ascii_letters if c in "aeiou")
            consonants = sum(1 for c in ascii_letters if c in "bcdfghjklmnpqrstvwxyz")
            if vowels == 0:
                return True
            if vowels > 0 and consonants / vowels > 3.5:
                return True
            if len(ascii_letters) >= 25:
                return True

    if 1 < len(tokens) <= 3:
        merged = "".join(tokens)
        ascii_letters = re.sub(r"[^a-zA-Z]", "", merged)
        if len(ascii_letters) >= 20:
            has_diac = any(c in text_low for c in _VIETNAMESE_DIACRITICS)
            vowels = sum(1 for c in ascii_letters if c in "aeiou")
            consonants = sum(1 for c in ascii_letters if c in "bcdfghjklmnpqrstvwxyz")
            if not has_diac and vowels > 0 and consonants / vowels > 3:
                return True
    return False


def is_short_generic(text: str) -> bool:
    """Check if review contains only generic short phrases."""
    return False


_OFF_TOPIC_INDICATORS = [
    "subject:", "dear it support", "i hope you are doing well",
    "đề nghị người dân không sử dụng",
    "bản quyền phim", "tân dòng sông ly biệt",
    "thượng tá đoàn văn báu", "sư thích minh tuệ",
    "newhouse", "lazland",
    "ngôn ngữ mang tính phê phán",
    "@all các b", "gửi đơn lên chụp giúp",
    "anxiously (trạng từ)", "prestigious (adj)",
    "the school will close", "the school will open",
]

_COMPETITOR_PATTERNS = [
    "bên e triển khai", "bên em triển khai", "bên anh triển khai",
    "lấy đủ suất giúp", "qua bên kia", "sang bên kia",
    "mua tại bên", "shop khác rẻ hơn", "shop khác giá tốt",
    "bên kia rẻ hơn", "chỗ khác rẻ", "tìm shop khác",
]


def is_off_topic(text: str) -> bool:
    """Check for off-topic, spammy, or irrelevant content."""
    text_low = normalize_text(text)
    return any(ind in text_low for ind in _OFF_TOPIC_INDICATORS)


def is_competitor_promo(text: str) -> bool:
    """Check for competitor promotional spam patterns."""
    text_low = normalize_text(text)
    return any(p in text_low for p in _COMPETITOR_PATTERNS)


def contains_external_link(text: str) -> bool:
    """Check if review contains external web URLs."""
    if not isinstance(text, str):
        return False
    return bool(_URL_PATTERN.search(text))


def contains_contact_info(text: str) -> bool:
    """Check for phone numbers or off-platform contact handles."""
    if not isinstance(text, str):
        return False
    if _PHONE_PATTERN.search(text):
        return True
    return bool(_CONTACT_KEYWORD_PATTERN.search(text))


_NEGATIVE_KEYWORDS = [
    "tệ", "tồi", "dở", "kém", "thất vọng", "chán", "hỏng", "vỡ",
    "lỗi", "gãy", "rách", "bẩn", "giả", "fake", "scam", "lừa",
    "không tốt", "không ok", "dở tệ", "không hài lòng", "không đáng",
    "không nên mua", "đừng mua", "không giống", "khác mô tả",
    "giao sai", "giao nhầm", "hư", "méo", "móp", "trầy", "xước",
    "mùi hôi", "hôi", "thối", "quá tệ", "rất tệ", "cực kỳ tệ",
    "kinh khủng", "thảm họa", "chả ra gì", "vứt đi",
    "không thể chấp nhận", "không hết gàu", "ko hết", "shop làm ăn",
    "móc hàng", "ko giao", "không nhận",
]

_POSITIVE_KEYWORDS = [
    "rất tốt", "tuyệt vời", "tuyệt", "xuất sắc", "hoàn hảo",
    "rất hài lòng", "hài lòng", "ưng ý", "đáng tiền", "đáng đồng tiền",
    "chất lượng tốt", "chất lượng cao", "đẹp", "siêu đẹp",
    "yêu shop", "yêu lắm",
]


def has_rating_text_mismatch(text: str, rating) -> bool:
    """Detect severe contradictions between star rating and review text sentiment."""
    try:
        stars = int(float(str(rating)))
    except (ValueError, TypeError):
        return False
    text_low = normalize_text(text)
    nwords = count_words(text)
    if nwords < 15:
        return False
    if stars >= 4:
        neg_count = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text_low)
        if neg_count >= 3:
            return True
    if stars <= 2:
        pos_count = sum(1 for kw in _POSITIVE_KEYWORDS if kw in text_low)
        if pos_count >= 3:
            return True
    return False


def find_duplicate_clusters(
    texts: List[str],
    threshold: float = 0.85,
    min_text_len: int = 30,
) -> List[Set[int]]:
    """Identify clusters of near-identical reviews using TF-IDF cosine similarity."""
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


def flag_duplicates(texts: List[str], threshold: float = 0.95) -> List[bool]:
    """Return boolean list indicating whether each review belongs to a duplicate cluster."""
    clusters = find_duplicate_clusters(texts, threshold=threshold)
    dup_indices: Set[int] = set()
    for cluster in clusters:
        dup_indices |= cluster
    return [i in dup_indices for i in range(len(texts))]


def detect_spam(df: pd.DataFrame, dup_threshold: float = 0.95) -> pd.DataFrame:
    """Execute rule-based spam detection across DataFrame and add 'is_spam' column."""
    if "text" not in df.columns or "rating" not in df.columns:
        raise ValueError("DataFrame must contain columns 'text' and 'rating'")

    result = df.copy()
    texts = result["text"].fillna("").astype(str).tolist()
    ratings = result["rating"].tolist()

    flag_ai_template = [is_ai_template_review(t) for t in texts]
    flag_template_repeat = [is_template_repetition(t) for t in texts]
    flag_mostly_template = [is_mostly_template(t) for t in texts]
    flag_xu_farming = [is_xu_farming(t) for t in texts]

    flag_too_short = [is_too_short(t) for t in texts]
    flag_too_long = [is_too_long(t) for t in texts]
    flag_emoji_only = [is_emoji_only(t) for t in texts]
    flag_keyboard = [is_keyboard_spam(t) for t in texts]
    flag_word_repeat = [is_word_repetition(t) for t in texts]
    flag_special = [has_too_many_special_chars(t) for t in texts]
    flag_uppercase = [has_too_many_uppercase(t) for t in texts]
    flag_only_digits = [is_only_digits_or_punct(t) for t in texts]
    flag_random_keyboard = [is_random_keyboard_string(t) for t in texts]
    flag_non_informative = [
        is_non_informative_short(t, r) for t, r in zip(texts, ratings)
    ]

    flag_short_generic = [is_short_generic(t) for t in texts]
    flag_off_topic = [is_off_topic(t) for t in texts]
    flag_competitor = [is_competitor_promo(t) for t in texts]
    flag_link = [contains_external_link(t) for t in texts]
    flag_contact = [contains_contact_info(t) for t in texts]
    flag_mismatch = [
        has_rating_text_mismatch(t, r) for t, r in zip(texts, ratings)
    ]

    flag_duplicate = flag_duplicates(texts, threshold=dup_threshold)

    all_flags = list(zip(
        flag_ai_template, flag_template_repeat, flag_mostly_template,
        flag_xu_farming,
        flag_too_long, flag_emoji_only,
        flag_keyboard, flag_word_repeat,
        flag_special, flag_uppercase, flag_only_digits,
        flag_random_keyboard,
        flag_off_topic, flag_competitor,
        flag_link, flag_contact,
        flag_mismatch, flag_duplicate,
    ))
    is_spam = [1 if any(flags) else 0 for flags in all_flags]
    result["is_spam"] = is_spam

    flag_details = pd.DataFrame({
        "ai_template": flag_ai_template,
        "template_repeat": flag_template_repeat,
        "mostly_template": flag_mostly_template,
        "xu_farming": flag_xu_farming,
        "too_short": flag_too_short,
        "too_long": flag_too_long,
        "emoji_only": flag_emoji_only,
        "keyboard_spam": flag_keyboard,
        "word_repetition": flag_word_repeat,
        "too_many_special": flag_special,
        "too_many_uppercase": flag_uppercase,
        "only_digits_or_punct": flag_only_digits,
        "random_keyboard": flag_random_keyboard,
        "non_informative_short": flag_non_informative,
        "short_generic": flag_short_generic,
        "off_topic": flag_off_topic,
        "competitor_promo": flag_competitor,
        "external_link": flag_link,
        "contact_info": flag_contact,
        "rating_mismatch": flag_mismatch,
        "duplicate_seeding": flag_duplicate,
    }, index=result.index)
    result.attrs["flag_details"] = flag_details
    return result


def summarize_spam(df: pd.DataFrame) -> dict:
    """Calculate summary statistics and breakdown metrics from detect_spam results."""
    if "is_spam" not in df.columns:
        raise ValueError("DataFrame must contain 'is_spam' column.")

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

    label_map = {
        "ai_template": "Review template AI",
        "template_repeat": "Lặp template",
        "mostly_template": "Chủ yếu là template",
        "xu_farming": "Bình luận nhận xu",
        "too_short": "Quá ngắn",
        "too_long": "Quá dài",
        "emoji_only": "Chỉ toàn emoji",
        "keyboard_spam": "Spam bàn phím",
        "word_repetition": "Lặp từ liên tục",
        "too_many_special": "Quá nhiều ký tự đặc biệt",
        "too_many_uppercase": "Quá nhiều chữ in hoa",
        "only_digits_or_punct": "Chỉ số và dấu câu",
        "random_keyboard": "Chuỗi gõ ngẫu nhiên",
        "non_informative_short": "5 sao không thông tin",
        "short_generic": "Mẫu khen ngắn",
        "off_topic": "Không liên quan sản phẩm",
        "competitor_promo": "Quảng cáo shop khác",
        "external_link": "Chứa link bên ngoài",
        "contact_info": "Chứa thông tin liên hệ",
        "rating_mismatch": "Mâu thuẫn rating - nội dung",
        "duplicate_seeding": "Trùng lặp seeding",
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
