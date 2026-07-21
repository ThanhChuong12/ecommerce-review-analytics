"""Configuration module for the NLP text preprocessing pipeline.

Defines regex patterns, dictionary mappings, and stopwords for text normalization.
"""

from __future__ import annotations

import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
SPECIAL_CHAR_PATTERN = re.compile(r"[^\w\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")

TEEN_CODE_DICT = {
    "bvs": "băng vệ sinh",
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
    "nhah": "nhanh",
    "thix": "thích",
    "sài": "dùng",
    "xài": "dùng",
    "hsd": "hạn sử dụng",
    "date": "hạn sử dụng",
    "mn": "mọi người",
    "mng": "mọi người",
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

EMOJI_DICT = {
    "⭐️": " tuyệt_vời ",
    "⭐": " tuyệt_vời ",
    "🌟": " tuyệt_vời ",
    "❤": " yêu_thích ",
    "❤️": " yêu_thích ",
    "👍": " tốt ",
    "😊": " vui_vẻ ",
    "🙄": " chán ",
    "😍": " tuyệt_vời ",
    "😘": " yêu_thích ",
    "😞": " tệ ",
    "😡": " tệ ",
    "😭": " buồn ",
}

VI_STOPWORDS = {
    "là", "và", "thì", "mà", "được", "bị", "của", "có", "với", "cho", "trong",
    "để", "này", "đó", "đây", "kia", "sau", "khi", "tại", "tới", "từ", "những",
    "các", "một", "cái", "như", "nào", "bởi", "nên", "về", "nhưng", "tuy",
    "hay", "cũng", "đã", "đang", "sẽ", "rồi", "chứ", "ạ", "nhé", "nha", "nè",
    "ơi", "á", "hihi", "haha", "kkk",
}