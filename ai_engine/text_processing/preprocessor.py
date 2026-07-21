"""Vietnamese text preprocessor for e-commerce review analytics."""

from __future__ import annotations

import re
from typing import Optional

import emoji
from underthesea import word_tokenize

from ai_engine.text_processing.config import (
    EMOJI_DICT,
    HTML_PATTERN,
    TEEN_CODE_DICT,
    URL_PATTERN,
    WHITESPACE_PATTERN,
)


class TextCleaner:
    """Production-grade Vietnamese text cleaner for noisy e-commerce reviews."""

    VOWELS = "aeiouyáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"

    def __init__(self) -> None:
        self.teen_code_dict = TEEN_CODE_DICT
        self.emoji_dict = EMOJI_DICT

        self._phone_pattern = re.compile(r"\b(?:\+?84|0)(?:[\s.\-]*\d){8,10}\b")
        self._email_pattern = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
        self._mention_pattern = re.compile(r"@\w+")

        if self.emoji_dict:
            escaped_emojis = [
                re.escape(e) for e in sorted(self.emoji_dict.keys(), key=len, reverse=True)
            ]
            self._emoji_pattern = re.compile(r"(" + "|".join(escaped_emojis) + r")")
        else:
            self._emoji_pattern = re.compile(r"$")

        self._repeat_char_pattern = re.compile(r"([a-zA-ZÀ-Ỹà-ỹ])\1{2,}")
        self._repeat_punct_pattern = re.compile(r"([.!?,])\1+")
        self._allowed_chars_pattern = re.compile(r"[^0-9a-zA-ZÀ-Ỹà-ỹ.,!?_\s]")
        self._teen_code_pattern = self._build_teen_code_pattern()

        consonants = "bcdfghjklmnpqrstvwxzđBCDFGHJKLMNPQRSTVWXZĐ"
        self._gibberish_consonants_pattern = re.compile(rf"[{consonants}]{{5,}}", re.IGNORECASE)
        self._gibberish_no_vowel_pattern = re.compile(
            rf"(?:^|\s)[^\s\d{self.VOWELS}]{{8,}}(?:$|\s)", re.IGNORECASE
        )

    def _remove_gibberish(self, text: str) -> str:
        """Filter out keyboard mashing and unpronounceable token sequences."""
        if self._gibberish_consonants_pattern.search(text) or self._gibberish_no_vowel_pattern.search(text):
            return ""
        return text

    def _build_teen_code_pattern(self) -> re.Pattern:
        """Compile regex matching teen code abbreviations using Unicode-safe boundaries."""
        if not self.teen_code_dict:
            return re.compile(r"$")

        escaped = [re.escape(k) for k in sorted(self.teen_code_dict, key=len, reverse=True)]
        boundary_left = r"(?:(?<=^)|(?<=\s)|(?<=[\.,!\?\(\)\{\}\[\]]))"
        boundary_right = r"(?:(?=$)|(?=\s)|(?=[\.,!\?\(\)\{\}\[\]]))"
        return re.compile(boundary_left + r"(" + "|".join(escaped) + r")" + boundary_right)

    def _coerce_text(self, text: Optional[object]) -> str:
        """Convert arbitrary input into a string safely."""
        if text is None:
            return ""
        if isinstance(text, str):
            return text
        try:
            return str(text)
        except Exception:
            return ""

    def _replace_teen_code_match(self, match: re.Match) -> str:
        """Replace regex matched teen code shortcut with standard Vietnamese expression."""
        token = match.group(1)
        return self.teen_code_dict.get(token, token)

    def _replace_custom_emoji(self, match: re.Match) -> str:
        """Replace emoji character with mapped Vietnamese textual representation."""
        return self.emoji_dict.get(match.group(1), match.group(1))

    def clean_text(self, text: str) -> str:
        """Clean and normalize a review text entry into standardized Vietnamese tokens."""
        raw_text = self._coerce_text(text)
        if not raw_text:
            return ""

        cleaned = raw_text.lower()
        cleaned = URL_PATTERN.sub(" ", cleaned)
        cleaned = HTML_PATTERN.sub(" ", cleaned)
        cleaned = self._phone_pattern.sub(" ", cleaned)
        cleaned = self._email_pattern.sub(" ", cleaned)
        cleaned = self._mention_pattern.sub(" ", cleaned)

        if self.emoji_dict:
            cleaned = self._emoji_pattern.sub(self._replace_custom_emoji, cleaned)

        cleaned = emoji.replace_emoji(cleaned, replace="")
        cleaned = self._repeat_char_pattern.sub(r"\1", cleaned)
        cleaned = self._repeat_punct_pattern.sub(r"\1", cleaned)
        cleaned = self._allowed_chars_pattern.sub(" ", cleaned)

        if cleaned:
            cleaned = self._teen_code_pattern.sub(self._replace_replace_teen_code_match if hasattr(self, "_replace_replace_teen_code_match") else self._replace_teen_code_match, cleaned)

        if cleaned:
            cleaned = self._remove_gibberish(cleaned)

        if cleaned:
            cleaned = word_tokenize(cleaned, format="text")

        return WHITESPACE_PATTERN.sub(" ", cleaned).strip()