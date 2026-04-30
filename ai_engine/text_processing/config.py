"""
Configuration module for the NLP text preprocessing pipeline.
Contains regex patterns, mappings for text normalization, and system constants.
"""

import re

# ==============================================================================
# REGEX PATTERNS (Pre-compiled for performance)
# ==============================================================================

# Matches HTTP/HTTPS URLs and www.* links
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')

# Matches HTML tags
HTML_PATTERN = re.compile(r'<.*?>')

# Matches special characters, keeping only alphanumeric and whitespace (supports Unicode/Vietnamese)
# \w matches any word character (equivalent to [a-zA-Z0-9_] plus unicode characters)
SPECIAL_CHAR_PATTERN = re.compile(r'[^\w\s]')

# Matches multiple spaces to normalize whitespace
WHITESPACE_PATTERN = re.compile(r'\s+')


# ==============================================================================
# DICTIONARIES & SETS
# ==============================================================================

# Comprehensive mapping for teen code, social media abbreviations, and e-commerce typos
TEEN_CODE_DICT = {
    "bvs": "băng vệ sinh",
    "ok": "tốt",
    "okie": "tốt",
    "oki": "tốt",
    "okila": "tốt",
    "oke": "tốt",
    "sp": "sản phẩm",
    "chất": "chất lượng",
    "đc": "được",
    "dc": "được",
    "mik": "mình",
    "mh": "mình",
    "mk": "mình",
    "m": "mình",
    "nma": "nhưng mà",
    "ksao": "không sao",
    "r": "rồi",
    "rùi": "rồi",
    "roi": "rồi",
    "cx": "cũng",
    "ko": "không",
    "k": "không",
    "khg": "không",
    "kh": "không",
    "nx": "nữa",
    "đẹp": "đẹp",
    "chuẩn": "chuẩn",
    "auth": "chính hãng",
    "rep": "phản hồi",
    "ib": "nhắn tin",
    "sz": "kích cỡ",
    "size": "kích cỡ",
    "nv": "nhân viên",
    "nvbh": "nhân viên bán hàng",
    "ship": "giao hàng",
    "shipper": "người giao hàng",
    "shiper": "người giao hàng",
    "nhah": "nhanh",
    "thix": "thích",
    "sài": "dùng",
    "xài": "dùng",
    "hsd": "hạn sử dụng",
    "date": "hạn sử dụng",
    "mn": "mọi người",
    "mng": "mọi người",
    "sop": "cửa hàng",
    "shop": "cửa hàng",
    "xốp": "cửa hàng",
    "st": "siêu thị",
    "lun": "luôn",
    "vs": "với",
    "h": "giờ",
    "tgian": "thời gian",
    "km": "khuyến mãi",
    "kmai": "khuyến mãi",
    "bik": "biết",
    "bt": "biết",
    "trvia": "trộm vía",
    "trv": "trộm vía",
    "lzd": "lazada",
    "laz": "lazada",
    "bhx": "bách hóa xanh",
    "hnay": "hôm nay",
    "tr": "trời",
    "ms": "mới",
    "v": "vậy",
    "z": "vậy",
    "zậy": "vậy",
}

# Vietnamese filler words without emotional/sentiment weight
VI_STOPWORDS = {
    "là", "và", "thì", "mà", "được", "bị", "của", "có", "với", "cho", "trong",
    "để", "này", "đó", "đây", "kia", "sau", "khi", "tại", "tới", "từ", "những",
    "các", "một", "cái", "như", "nào", "bởi", "nên", "về", "nhưng", "tuy",
    "hay", "cũng", "đã", "đang", "sẽ", "rồi", "chứ", "ạ", "nhé", "nha", "nè",
    "ơi", "á", "hihi", "haha", "kkk",
}