"""
Text preprocessing module for Vietnamese e-commerce reviews.
Implements data cleaning, normalization, and tokenization.
"""

from typing import Optional
import re

import emoji
from underthesea import word_tokenize

from ai_engine.text_processing.config import (
    URL_PATTERN,
    HTML_PATTERN,
    SPECIAL_CHAR_PATTERN,
    WHITESPACE_PATTERN,
    TEEN_CODE_DICT,
    VI_STOPWORDS
)


class TextPreprocessor:
    """
    A comprehensive, production-ready pipeline for preprocessing Vietnamese text.
    Handles cleaning, abbreviation normalization, tokenization, and stopword removal.
    """

    def __init__(self) -> None:
        """Initialize the preprocessor with precompiled resources."""
        # Dictionaries could be dynamically re-loaded here if needed in the future
        self.teen_code_dict = TEEN_CODE_DICT
        self.stopwords = VI_STOPWORDS
        self._teen_code_pattern = self._build_teen_code_pattern()

    def _build_teen_code_pattern(self) -> re.Pattern:
        """
        Build a regex pattern to replace teen code tokens efficiently.

        Returns:
            re.Pattern: Compiled regex matching any teen code token.
        """
        if not self.teen_code_dict:
            return re.compile(r"$")
        escaped = [re.escape(key) for key in sorted(self.teen_code_dict, key=len, reverse=True)]
        return re.compile(r"\b(" + "|".join(escaped) + r")\b")

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

    def clean_text(self, text: str) -> str:
        """
        Applies basic cleaning operations: lowercasing, HTML/URL removal,
        emoji removal, and special character stripping.
        
        Args:
            text (str): The raw input string.
            
        Returns:
            str: The cleaned string.
        """
        # 1. Convert text to lowercase
        text = text.lower()

        # 2. Remove URLs and HTML tags
        text = URL_PATTERN.sub(' ', text)
        text = HTML_PATTERN.sub(' ', text)

        # 3. Remove emojis
        # emoji.replace_emoji replaces emojis with a chosen string (default is empty string)
        text = emoji.replace_emoji(text, replace='')

        # 4. Remove special characters (keeping alphanumeric and Vietnamese chars)
        text = SPECIAL_CHAR_PATTERN.sub(' ', text)

        # 5. Normalize multiple whitespaces into a single space
        text = WHITESPACE_PATTERN.sub(' ', text).strip()

        return text

    def normalize_teen_code(self, text: str) -> str:
        """
        Replaces social media abbreviations and 'teen code' with standard Vietnamese.
        
        Args:
            text (str): The text to normalize.
            
        Returns:
            str: The normalized text.
        """
        if not text:
            return ""

        def _replace(match: re.Match) -> str:
            token = match.group(0)
            return self.teen_code_dict.get(token, token)

        return self._teen_code_pattern.sub(_replace, text)

    def tokenize_vietnamese(self, text: str) -> str:
        """
        Tokenizes Vietnamese text using underthesea.
        Words will be grouped using underscores (e.g., "băng_vệ_sinh").
        
        Args:
            text (str): The text to tokenize.
            
        Returns:
            str: Tokenized Vietnamese string.
        """
        if not text:
            return ""
        # format="text" joins syllables of a word with an underscore "_"
        return word_tokenize(text, format="text")

    def remove_stopwords(self, text: str) -> str:
        """
        Filters out stopwords from the tokenized text.
        
        Args:
            text (str): The tokenized text containing underscores for compound words.
            
        Returns:
            str: Text with stopwords removed.
        """
        words = text.split()
        filtered_words = []
        for word in words:
            normalized = word.replace('_', ' ')
            if normalized not in self.stopwords:
                filtered_words.append(word)
        return ' '.join(filtered_words)

    def process(self, text: Optional[object], apply_stopwords: bool = True) -> str:
        """
        Executes the complete text processing pipeline in sequential order.
        
        Args:
            text (Optional[str]): The raw input text. Can be None.
            apply_stopwords (bool): Whether to filter out stopwords. Defaults to False.
            
        Returns:
            str: The fully processed, tokenized text ready for ML models.
        """
        raw_text = self._coerce_text(text)
        if not raw_text:
            return ""

        # Step 1-3: Basic cleaning (lowercase, links, html, emojis, specials)
        processed_text = self.clean_text(raw_text)

        # Step 4: Map teen code and abbreviations
        processed_text = self.normalize_teen_code(processed_text)

        # Step 5: Vietnamese Specific Tokenization
        processed_text = self.tokenize_vietnamese(processed_text)

        # Step 6: (Optional) Stopword removal
        if apply_stopwords:
            processed_text = self.remove_stopwords(processed_text)

        # Final cleanup of any lingering weird whitespaces
        return WHITESPACE_PATTERN.sub(' ', processed_text).strip()