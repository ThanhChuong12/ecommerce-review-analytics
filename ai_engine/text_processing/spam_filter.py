"""
spam_filter.py
==============

Rule-based spam and seeding detector for Vietnamese e-commerce reviews
(Shopee, Tiki, Lazada, The Gioi Di Dong).

This module is part of the "Multimodal Review Analytics" project, owned by
Hieu (member 4 of the team).

Design philosophy
-----------------
- Rules are explainable: every spam flag carries a human-readable reason.
- Rules are organised along five axes that match real-world spam behaviour
  observed when manually labelling 1,000 reviews of the dataset:

    Axis 1 - AI-generated / templated reviews
        Lazada in particular is flooded with bot-generated reviews of the
        form "Phrase A, Phrase B, Phrase C," where each phrase is a generic
        marketing-style claim about the product. We catch them by counting
        how many known template phrases appear in the text.

    Axis 2 - Coin/voucher farming
        Shopee, Tiki and Lazada offer coins for posting a review longer
        than N characters. Many users type random words or end their text
        with "hinh anh chi mang tinh chat nhan xu" (image is for coin
        collection only). These should be flagged.

    Axis 3 - Structural noise
        Pure emojis, keyboard mash, character/pattern repetition, and
        text that is only digits or punctuation.

    Axis 4 - Off-topic / contact / competitor content
        URLs, phone numbers, Zalo/Facebook handles, ads for other shops,
        copy-paste of song lyrics, news, religion, English emails, etc.

    Axis 5 - Rating-text mismatch
        Reviews where the star rating clearly contradicts the sentiment
        of the text (e.g. 5 stars + "very disappointed, broken").

- A separate cross-review check uses TF-IDF + cosine similarity to detect
  groups of nearly-identical reviews (organised seeding).

Public API
----------
- detect_spam(df, dup_threshold=0.85)
        Main entry point. Takes a DataFrame with at least the columns
        "text" and "rating" and returns a copy with a single new column
        "is_spam" (0 or 1). The detailed per-rule flags are attached to
        the result via .attrs["flag_details"] so the notebook can use
        them for EDA without polluting the saved CSV.

- summarize_spam(df)
        Aggregate counts and percentages by rule label.

- normalize_pipeline(text)
        Multi-step text normalisation pipeline that reports how many
        characters were stripped at each step (useful for reporting the
        "vocabulary change ratio" required by the project brief).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================================
#  1. TEXT NORMALISATION HELPERS
# =====================================================================

# Match the most common Unicode emoji blocks. Not 100% complete but enough
# to measure the emoji ratio of a sentence.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # Emoticons
    "\U0001F300-\U0001F5FF"   # Symbols and pictographs
    "\U0001F680-\U0001F6FF"   # Transport and map
    "\U0001F1E0-\U0001F1FF"   # Country flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"   # Supplemental
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"              # Zero-width joiner
    "]+",
    flags=re.UNICODE,
)

# Match URLs (http/https/www and bare domains with common TLDs).
_URL_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+|"
    r"\b[\w\-]+\.(?:com|vn|net|org|info|biz|shop|store|me|co|io)\b/?\S*",
    flags=re.IGNORECASE,
)

# Match leftover HTML tags from scraping.
_HTML_PATTERN = re.compile(r"<[^>]+>")

# Match @mentions and #hashtags.
_MENTION_PATTERN = re.compile(r"@\w+")
_HASHTAG_PATTERN = re.compile(r"#\w+")

# Match Vietnamese phone numbers, including obfuscated forms (spaces, dots,
# hyphens between digits) that users insert to bypass platform filters.
_PHONE_PATTERN = re.compile(
    r"(?:(?:\+?84|0)[\s\.\-]?)(?:3|5|7|8|9)(?:[\s\.\-]?\d){8}",
)

# Match keywords that hint at off-platform contact.
_CONTACT_KEYWORD_PATTERN = re.compile(
    r"\b(zalo|z\.a\.l\.o|fb|f\.b|facebook|messenger|"
    r"telegram|tele|whatsapp|wa|line|viber|skype)\b",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Light normalisation: NFC, lowercase, collapse whitespace.

    NFC matters for Vietnamese because the same diacritised letter can be
    encoded as a single code point or as a base letter plus a combining
    diacritic. Without NFC, two visually identical strings can mismatch.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def remove_emojis(text: str) -> str:
    """Strip all emoji characters from the string."""
    if not isinstance(text, str):
        return ""
    return _EMOJI_PATTERN.sub("", text)


def normalize_pipeline(text: str) -> Tuple[str, Dict[str, int]]:
    """Full normalisation pipeline with per-step character counts.

    Steps:
        1. NFC unicode normalisation
        2. Lowercase
        3. Remove HTML tags
        4. Remove URLs
        5. Remove mentions and hashtags
        6. Collapse whitespace

    Returns
    -------
    cleaned_text : str
    stats : dict with the number of characters removed at each step,
            useful for reporting how each step affects the vocabulary.
    """
    if not isinstance(text, str):
        return "", {"original_len": 0, "html_removed": 0, "url_removed": 0,
                    "mention_removed": 0, "final_len": 0}

    stats = {"original_len": len(text)}

    # Step 1 + 2.
    text = unicodedata.normalize("NFC", text)
    text = text.lower()

    # Step 3.
    before = len(text)
    text = _HTML_PATTERN.sub(" ", text)
    stats["html_removed"] = before - len(text)

    # Step 4.
    before = len(text)
    text = _URL_PATTERN.sub(" ", text)
    stats["url_removed"] = before - len(text)

    # Step 5.
    before = len(text)
    text = _MENTION_PATTERN.sub(" ", text)
    text = _HASHTAG_PATTERN.sub(" ", text)
    stats["mention_removed"] = before - len(text)

    # Step 6.
    text = re.sub(r"\s+", " ", text).strip()
    stats["final_len"] = len(text)
    return text, stats


def count_words(text: str) -> int:
    """Return the number of words after stripping emojis and lowercasing."""
    cleaned = remove_emojis(normalize_text(text))
    return len(cleaned.split())


def count_chars(text: str) -> int:
    """Return the number of non-whitespace characters."""
    if not isinstance(text, str):
        return 0
    return len(re.sub(r"\s+", "", text))


def get_emoji_ratio(text: str) -> float:
    """Return the fraction of characters that are emojis (0.0 - 1.0)."""
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    emojis = _EMOJI_PATTERN.findall(text)
    total_emoji_len = sum(len(e) for e in emojis)
    return total_emoji_len / len(text)


def get_special_char_ratio(text: str) -> float:
    """Return the fraction of non-alphanumeric, non-whitespace characters.

    Real reviews include some punctuation; very high ratios indicate
    symbol spam ("!!!@@@###$$$").
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
    """Fraction of letters that are uppercase. Sentences shorter than ten
    letters always return 0.0 to avoid noisy false positives on ALL-CAPS
    short tokens like "OK"."""
    if not isinstance(text, str):
        return 0.0
    only_letters = [c for c in text if c.isalpha()]
    if len(only_letters) < 10:
        return 0.0
    upper_count = sum(1 for c in only_letters if c.isupper())
    return upper_count / len(only_letters)


def get_digit_ratio(text: str) -> float:
    """Fraction of characters that are digits."""
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    digit_count = sum(1 for c in text if c.isdigit())
    return digit_count / len(text)


def get_type_token_ratio(text: str) -> float:
    """Type-Token Ratio: unique words divided by total words.

    A high TTR (close to 1) means the writer uses varied vocabulary.
    A low TTR (close to 0) signals heavy word repetition, common in
    spam reviews. The brief asks for this metric in the EDA section.
    """
    cleaned = remove_emojis(normalize_text(text))
    tokens = cleaned.split()
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)


# =====================================================================
#  2. AXIS 1 - AI / TEMPLATE GENERATED REVIEW DETECTION
# =====================================================================
# These phrases were extracted by manually inspecting 1,000 reviews and
# noting the "marketing copy" sentences that bots and seeders paste.
# They appear individually as single phrases or chained with commas.

_TEMPLATE_PHRASES = [
    # Baby formula category
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
    # Coffee category
    "hương vị mượt mà và thỏa mãn",
    "lựa chọn tuyệt vời để tăng cường buổi sáng",
    "một cách tuyệt vời để bắt đầu ngày mới",
    "hương thơm dễ chịu",
    "hương vị đậm đà và táo bạo",
    "hàm lượng caffeine cao để tăng cường năng lượng",
    "bao bì tiện lợi để giữ độ tươi",
    "hoàn hảo cho những người yêu cà phê",
    "hạt cà phê chất lượng cao",
    # Detergent / fabric care
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
    # Tissue / paper category
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
    # Yoghurt / dairy snack category
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
    """Count how many marketing-template phrases appear in the text."""
    text_low = normalize_text(text)
    return sum(1 for p in _TEMPLATE_PHRASES if p in text_low)


def is_ai_template_review(text: str) -> bool:
    """[UPDATED]: Disabled per user request."""
    return False


def is_template_repetition(text: str) -> bool:
    """Flag reviews that repeat the same template phrase more than once."""
    text_low = normalize_text(text)
    for p in _TEMPLATE_PHRASES:
        if text_low.count(p) >= 2:
            return True
    return False


def is_mostly_template(text: str) -> bool:
    """[UPDATED]: Disabled per user request."""
    return False


# =====================================================================
#  3. AXIS 2 - COIN / VOUCHER FARMING
# =====================================================================
# Phrases that explicitly admit the review is only there to claim coins
# from the platform. Some users even openly write "image is for coin
# collection only".

_XU_FARMING_KEYWORDS = [
    "nhận xu", "nhan xu", "săn xu", "san xu", "lấy xu", "lay xu",
    "lấy su", "lay su", "đủ ký tự", "du ky tu", "đủ chữ", "du chu",
    "cho đủ ký tự", "viết cho đủ", "đánh giá nhận xu",
    "đánh giá để được", "tchat nhận", "chata nhận", "chata nhan",
    "shopee xu", "lazada xu", "hoàn xu", "hoan xu",
]

# "Hình ảnh chỉ mang tính chất minh hoạ" / "Video không liên quan" are
# common boilerplate the user appends so that the platform thinks the
# review has media content.
_DISCLAIMER_PATTERN = re.compile(
    r"(hình ảnh|h(ì|i)nh\s*(ả|a)nh|video|hinh anh)"
    r"[^.]{0,40}"
    r"(mang tính chất|mang tinh chat|chỉ mang|mag tính|mag tinh|"
    r"không liên quan|khong lien quan|k liên quan|k lien quan)",
    flags=re.IGNORECASE,
)


def is_xu_farming(text: str) -> bool:
    """Flag reviews whose only purpose is to harvest coins/vouchers."""
    text_low = normalize_text(text)
    if any(kw in text_low for kw in _XU_FARMING_KEYWORDS):
        return True
    if _DISCLAIMER_PATTERN.search(text_low):
        return True
    return False


# =====================================================================
#  4. AXIS 3 - STRUCTURAL NOISE
# =====================================================================

# Short, content-free 5-star reviews. Each token, lowercased and stripped
# of punctuation, is checked against this set. We only fire on reviews
# with <= 6 non-whitespace characters to avoid catching short-but-real
# negative reviews.
_NON_INFORMATIVE_TOKENS = {
    "ok", "good", "nice", "tot", "tốt", "hay", "ngon", "dep", "đẹp",
    "nhanh", "đã", "da", "ben", "bền", "thom", "thơm", "yeh", "tuyet",
    "tuyệt", "5 sao", "5 *", "5*", "ok nha", "sp tot", "sp tốt", "gud",
    "perfect", "okela", "okie", "oki", "đã lắm", "da lam", "khá ony",
    "khá ổn", "kha on", "okkk", "okkkkllll", "tuyệt đỉnh", "siêu tuyệt vời",
    "rat tot", "rất tốt", "gh", "rất hay", "rat hay", "rât tuyệt",
    "10 luôn", "10 luon", "good good", "chất lượng", "chat luong",
    "rat hữu ích", "đã", "ngon nha", "ok lắm", "rất ưng ý",
    "đã nhận", "da nhan", "tuyệt", "tuyet", "ldl", "nice!!!", "good.",
}


def is_too_short(text: str, min_words: int = 1) -> bool:
    """Reviews shorter than min_words add no information."""
    return count_words(text) < min_words


def is_non_informative_short(text: str, rating) -> bool:
    """Flag five-star reviews that contain only a single empty filler word.

    Examples: "ok", "good", "tốt", "5 sao". 
    [UPDATED]: Disabled per user request to avoid catching real lazy users.
    """
    return False


def is_too_long(text: str, max_words: int = 500) -> bool:
    """Review pasted from somewhere else (book, song lyrics, etc.)."""
    return count_words(text) > max_words


def is_emoji_only(text: str) -> bool:
    """Text contains no letter at all - emojis or punctuation only."""
    if not isinstance(text, str) or not text.strip():
        return True
    cleaned = remove_emojis(text)
    cleaned = re.sub(r"[^\w]", "", cleaned, flags=re.UNICODE)
    return len(cleaned.strip()) == 0


def is_keyboard_spam(text: str) -> bool:
    """Detect random key-mashing or character/pattern repetition.

    Two patterns are matched:
        - the same character repeated >= 6 times in a row ("aaaaaa")
        - a 2-3 character group repeated >= 4 times ("hahahaha", "lololo")

    Threshold is conservative to avoid mislabelling reviews like
    "Sách hay" or short Vietnamese words with diacritics.
    """
    norm = normalize_text(text)
    # Don't fire on simple punctuation like "..." at sentence end.
    norm_clean = norm.rstrip(".,! ")
    if re.search(r"([a-z])\1{5,}", norm_clean):
        return True
    if re.search(r"(.{2,3})\1{3,}", norm_clean):
        return True
    return False


def is_word_repetition(text: str, max_repeat: int = 4) -> bool:
    """Flag reviews where the same word is repeated max_repeat times in a row.

    Common when users hammer the keyboard to reach the platform's character
    minimum: "tốt tốt tốt tốt tốt".
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
    """Symbol-heavy reviews like '!!!!@@@##$$$%%%'."""
    if count_words(text) < 5:
        return False
    return get_special_char_ratio(text) >= threshold


def has_too_many_uppercase(text: str, threshold: float = 0.6) -> bool:
    """ALL-CAPS shouting reviews, often used in promotional spam."""
    return get_uppercase_ratio(text) >= threshold


def is_only_digits_or_punct(text: str) -> bool:
    """Reviews that contain no letters at all."""
    cleaned = remove_emojis(normalize_text(text))
    if not cleaned:
        return True
    return not any(c.isalpha() for c in cleaned)


# Common Vietnamese diacritised characters - used to distinguish a real
# (possibly diacritic-less) Vietnamese review from a random key-mash.
_VIETNAMESE_DIACRITICS = "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"


def is_random_keyboard_string(text: str) -> bool:
    """Detect long random letter sequences typed to fill character minimums.

    The heuristic is conservative: we only fire on long single tokens with
    no Vietnamese diacritics, no vowels at all, or with a very high
    consonant-to-vowel ratio. This avoids mislabelling diacritic-less
    Vietnamese sentences (which are common).
    """
    text_low = normalize_text(text)
    tokens = text_low.split()

    # Single very long token
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

    # Two or three "tokens" that together look random
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


# =====================================================================
#  5. AXIS 4 - OFF-TOPIC, CONTACT, COMPETITOR
# =====================================================================
# Generic short clichés that look like seeding rather than feedback.
_SHORT_TEMPLATE_TOKENS = {
    "shop uy tín", "giao hàng nhanh", "đóng gói cẩn thận",
    "sản phẩm tốt", "sản phẩm chất lượng", "rất tốt", "hài lòng",
    "rất hài lòng", "chất lượng tốt", "đúng mô tả", "như mô tả",
}


def is_short_generic(text: str) -> bool:
    """Reviews whose entire content is a single platitude."""
    norm = normalize_text(remove_emojis(text)).strip(".,! ")
    return norm in _SHORT_TEMPLATE_TOKENS


# Substrings that strongly suggest the review is copied from somewhere
# unrelated to the product.
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

# Patterns that signal a competing seller hijacking the review section.
_COMPETITOR_PATTERNS = [
    "bên e triển khai", "bên em triển khai", "bên anh triển khai",
    "lấy đủ suất giúp", "qua bên kia", "sang bên kia",
    "mua tại bên", "shop khác rẻ hơn", "shop khác giá tốt",
    "bên kia rẻ hơn", "chỗ khác rẻ", "tìm shop khác",
]


def is_off_topic(text: str) -> bool:
    """Reviews containing text obviously unrelated to the product."""
    text_low = normalize_text(text)
    return any(ind in text_low for ind in _OFF_TOPIC_INDICATORS)


def is_competitor_promo(text: str) -> bool:
    """Reviews that try to redirect buyers to a different shop."""
    text_low = normalize_text(text)
    return any(p in text_low for p in _COMPETITOR_PATTERNS)


def contains_external_link(text: str) -> bool:
    """Real reviews almost never contain URLs."""
    if not isinstance(text, str):
        return False
    return bool(_URL_PATTERN.search(text))


def contains_contact_info(text: str) -> bool:
    """Phone numbers and off-platform contact handles."""
    if not isinstance(text, str):
        return False
    if _PHONE_PATTERN.search(text):
        return True
    if _CONTACT_KEYWORD_PATTERN.search(text):
        return True
    return False


# =====================================================================
#  6. AXIS 5 - RATING vs TEXT MISMATCH
# =====================================================================

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
    """Flag reviews where the rating clearly contradicts the text sentiment.

    Two cases (both require >= 3 keywords AND text length >= 15 words to
    avoid false positives on short legitimate complaints):
        - 5 stars + text contains many strong negative words
        - 1-2 stars + text contains many strong positive words
    """
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


# =====================================================================
#  7. CROSS-REVIEW DUPLICATE DETECTION (SEEDING)
# =====================================================================

def find_duplicate_clusters(
    texts: List[str],
    threshold: float = 0.85,
    min_text_len: int = 30,
) -> List[Set[int]]:
    """Find clusters of nearly-identical reviews using TF-IDF + cosine similarity.

    Reviews shorter than min_text_len characters are excluded. We use 30
    characters by default because short reviews like "giao hàng nhanh"
    appear independently many times and would create false-positive clusters.
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
    """Boolean list parallel to texts; True if the review is in a duplicate cluster."""
    clusters = find_duplicate_clusters(texts, threshold=threshold)
    dup_indices: Set[int] = set()
    for cluster in clusters:
        dup_indices |= cluster
    return [i in dup_indices for i in range(len(texts))]


# =====================================================================
#  8. PIPELINE
# =====================================================================

def detect_spam(df: pd.DataFrame, dup_threshold: float = 0.85) -> pd.DataFrame:
    """Run all rules and return df with one extra column 'is_spam' (0 or 1).

    The detailed per-rule flag matrix is attached as df.attrs['flag_details']
    so the EDA notebook can use it without changing the saved CSV.
    """
    if "text" not in df.columns or "rating" not in df.columns:
        raise ValueError("DataFrame must contain columns 'text' and 'rating'")

    result = df.copy()
    texts = result["text"].fillna("").astype(str).tolist()
    ratings = result["rating"].tolist()

    print("[spam_filter] Scoring per-review rules ...")

    # Axis 1: AI / template generated
    flag_ai_template = [is_ai_template_review(t) for t in texts]
    flag_template_repeat = [is_template_repetition(t) for t in texts]
    flag_mostly_template = [is_mostly_template(t) for t in texts]

    # Axis 2: Coin / voucher farming
    flag_xu_farming = [is_xu_farming(t) for t in texts]

    # Axis 3: Structural noise
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

    # Axis 4: Off-topic / contact / competitor
    flag_short_generic = [is_short_generic(t) for t in texts]
    flag_off_topic = [is_off_topic(t) for t in texts]
    flag_competitor = [is_competitor_promo(t) for t in texts]
    flag_link = [contains_external_link(t) for t in texts]
    flag_contact = [contains_contact_info(t) for t in texts]

    # Axis 5: Rating-text mismatch
    flag_mismatch = [
        has_rating_text_mismatch(t, r) for t, r in zip(texts, ratings)
    ]

    # Cross-review duplicates
    print("[spam_filter] Detecting duplicate review clusters with cosine similarity ...")
    flag_duplicate = flag_duplicates(texts, threshold=dup_threshold)

    # Combine into a single decision: a review is spam if ANY rule fires.
    # Note: too_short is computed for EDA reporting but NOT used as a spam
    # signal on its own. Short reviews with negative ratings are real
    # complaints; the only short reviews we want to catch are 5-star,
    # content-free ones, which the non_informative_short rule already covers.
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

    # Per-rule flag matrix for EDA (kept off the saved CSV)
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

    print(
        f"[spam_filter] Done. Total reviews: {len(result):,}, "
        f"flagged as spam: {sum(is_spam):,} "
        f"({sum(is_spam)/max(len(result),1)*100:.1f}%)"
    )
    return result


def summarize_spam(df: pd.DataFrame) -> dict:
    """Summary statistics from a DataFrame produced by detect_spam."""
    if "is_spam" not in df.columns:
        raise ValueError("DataFrame must have an 'is_spam' column - run detect_spam first")

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

    # Vietnamese labels for plotting (the CSV stays untouched)
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
