"""
Robust Vietnamese text cleaning module for e-commerce review analytics.
"""

from typing import Optional
import re

import emoji
from underthesea import word_tokenize

from ai_engine.text_processing.config import HTML_PATTERN, TEEN_CODE_DICT, URL_PATTERN, WHITESPACE_PATTERN, EMOJI_DICT


class TextCleaner:
    """
    Production-grade Vietnamese text cleaner for noisy e-commerce reviews.

    The pipeline is optimized for large-scale DataFrame processing with
    precompiled regex patterns and stateless methods to support parallelism.
    """

    def __init__(self) -> None:
        """Initialize the cleaner and compile all regex patterns once."""
        self.teen_code_dict = TEEN_CODE_DICT

        # Precompile patterns to avoid re-allocation in massive batch runs.
        self._phone_pattern = re.compile(
            r"\b(?:\+?84|0)(?:[\s.\-]*\d){8,10}\b"
        )
        self._email_pattern = re.compile(
            r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"
        )
        self._mention_pattern = re.compile(r"@\w+")
        
        # Build regex pattern for custom emoji dictionary
        self.emoji_dict = EMOJI_DICT
        if self.emoji_dict:
            escaped_emojis = [re.escape(e) for e in sorted(self.emoji_dict.keys(), key=len, reverse=True)]
            self._emoji_pattern = re.compile(r"(" + "|".join(escaped_emojis) + r")")
        else:
            self._emoji_pattern = re.compile(r"$")

        self._repeat_char_pattern = re.compile(r"([a-zA-ZÀ-Ỹà-ỹ])\1{2,}")
        self._repeat_punct_pattern = re.compile(r"([.!?,])\1+")

        # Keep basic punctuation and underscores used by word_tokenize/emoji aliases.
        self._allowed_chars_pattern = re.compile(r"[^0-9a-zA-ZÀ-Ỹà-ỹ.,!?_\s]")

        # Unicode-safe boundary for Vietnamese tokens without \b pitfalls.
        self._teen_code_pattern = self._build_teen_code_pattern()

        # Gibberish detection patterns
        self.VOWELS = "aeiouyáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
        consonants = "bcdfghjklmnpqrstvwxzđBCDFGHJKLMNPQRSTVWXZĐ"
        self._gibberish_consonants_pattern = re.compile(rf"[{consonants}]{{5,}}", re.IGNORECASE)
        self._gibberish_no_vowel_pattern = re.compile(rf"(?:^|\s)[^\s\d{self.VOWELS}]{{8,}}(?:$|\s)", re.IGNORECASE)

    def _remove_gibberish(self, text: str) -> str:
        """
        Detect and remove gibberish text (e.g., random keys like "jfjjdks") to prevent
        vocabulary matrix pollution. If the text is overwhelmingly gibberish, returns "".
        """
        if self._gibberish_consonants_pattern.search(text) or self._gibberish_no_vowel_pattern.search(text):
            return ""
        return text

    def _build_teen_code_pattern(self) -> re.Pattern:
        """
        Build a regex pattern to replace teen code tokens using Unicode-safe boundaries.

        Returns:
            re.Pattern: Compiled regex matching any teen code token.
        """
        if not self.teen_code_dict:
            return re.compile(r"$")

        escaped = [re.escape(key) for key in sorted(self.teen_code_dict, key=len, reverse=True)]
        boundary_left = r"(?:(?<=^)|(?<=\s)|(?<=[\.,!\?\(\)\{\}\[\]]))"
        boundary_right = r"(?:(?=$)|(?=\s)|(?=[\.,!\?\(\)\{\}\[\]]))"
        return re.compile(boundary_left + r"(" + "|".join(escaped) + r")" + boundary_right)

    def _coerce_text(self, text: Optional[object]) -> str:
        """
        Convert input to a safe text string.

        Args:
            text (Optional[object]): Raw input that might be None or non-string.

        Returns:
            str: Sanitized string representation or empty string.
        """
        if text is None:
            return ""
        if isinstance(text, str):
            return text
        try:
            return str(text)
        except Exception:
            return ""

    def _replace_teen_code_match(self, match: re.Match) -> str:
        """
        Replace a teen code token using a dictionary lookup.

        This is a method (not a nested function) to avoid per-call allocations
        when processing millions of rows in a DataFrame.
        """
        token = match.group(1)
        return self.teen_code_dict.get(token, token)

    def _replace_custom_emoji(self, match: re.Match) -> str:
        """
        Map predefined e-commerce emojis directly into Vietnamese string tokens.
        """
        return self.emoji_dict.get(match.group(1), match.group(1))

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize a single text entry.

        Pipeline steps:
        1) Safeguard input
        2) Lowercase
        3) Remove URLs, HTML, phone numbers, emails, mentions
        4) Map high-signal emojis to Vietnamese text and clear remaining emojis
        5) Reduce repeated characters and punctuation
        6) Filter special characters (keep Vietnamese alnum + . , ! ? _)
        7) Apply teen code normalization with Unicode-safe boundaries
        8) Vietnamese word tokenization
        9) Normalize whitespace
        """
        raw_text = self._coerce_text(text)
        if not raw_text:
            return ""

        cleaned = raw_text.lower()

        cleaned = URL_PATTERN.sub(" ", cleaned)
        cleaned = HTML_PATTERN.sub(" ", cleaned)
        cleaned = self._phone_pattern.sub(" ", cleaned)
        cleaned = self._email_pattern.sub(" ", cleaned)
        cleaned = self._mention_pattern.sub(" ", cleaned)

        # Map high frequency emojis to text
        if self.emoji_dict:
            cleaned = self._emoji_pattern.sub(self._replace_custom_emoji, cleaned)
            
        # Remove remaining unknown emojis explicitly
        cleaned = emoji.replace_emoji(cleaned, replace="")

        cleaned = self._repeat_char_pattern.sub(r"\1", cleaned)
        cleaned = self._repeat_punct_pattern.sub(r"\1", cleaned)

        cleaned = self._allowed_chars_pattern.sub(" ", cleaned)

        if cleaned:
            cleaned = self._teen_code_pattern.sub(self._replace_teen_code_match, cleaned)

        if cleaned:
            cleaned = self._remove_gibberish(cleaned)

        if cleaned:
            cleaned = word_tokenize(cleaned, format="text")

        return WHITESPACE_PATTERN.sub(" ", cleaned).strip()


EXAMPLE_PARALLEL_USAGE = """
Example (parallel DataFrame processing):

from pandarallel import pandarallel
import pandas as pd

from ai_engine.text_processing.preprocessor import TextCleaner

pandarallel.initialize(progress_bar=True)
cleaner = TextCleaner()

df = pd.read_csv("data/raw/all_reviews.csv")
df["clean_text"] = df["text"].parallel_apply(cleaner.clean_text)

# Alternative with swifter:
import swifter
df["clean_text"] = df["text"].swifter.apply(cleaner.clean_text)
"""