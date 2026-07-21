"""
PhoBERT Review Dataset Module.

Provides the `PhoBertReviewDataset` PyTorch Dataset class for Vietnamese
e-commerce review sentiment classification using the vinai/phobert-base-v2
tokenizer.  Designed for use with HuggingFace's `DataCollatorWithPadding`
so that each batch is padded to the length of its *longest* sample (dynamic
padding) rather than a fixed global max_length, saving significant VRAM.
"""

import re
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Label mapping (Vietnamese → integer class index)
# ---------------------------------------------------------------------------
LABEL_MAP: Dict[str, int] = {
    "tích cực": 0,   # Positive  (~94 %)
    "tiêu cực": 1,   # Negative  (~5 %)
    "trung lập": 2,  # Neutral   (~1 %)
}

ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}


# ---------------------------------------------------------------------------
# Text pre-processing helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Apply lightweight cleaning suitable for PhoBERT input.

    PhoBERT was pre-trained on normalised Vietnamese text, so we:
      1. Collapse runs of whitespace (tabs, newlines, multiple spaces) to a
         single space.
      2. Strip leading / trailing whitespace.

    We intentionally *do not* strip punctuation or lowercase the text,
    because PhoBERT's SentencePiece tokeniser handles casing and the model
    was trained on cased text.

    Args:
        text: Raw review string.

    Returns:
        Lightly cleaned string ready for the tokeniser.
    """
    # Limit repeated punctuation (e.g., !!! -> !!)
    text = re.sub(r'([.?!])\1+', r'\1\1', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _resolve_text(value: object) -> str:
    """Safely convert any value to a clean string, handling NaN robustly.

    Pandas may pass ``float('nan')`` for missing cells.  Converting that
    directly to ``str`` yields ``"nan"``, which would be fed to the
    tokeniser.  We catch this case and return an empty string instead.

    Args:
        value: The raw cell value from a pandas DataFrame column.

    Returns:
        A clean, non-NaN string.
    """
    if value is None:
        return ""
    # float NaN check (works for numpy.float64 and built-in float)
    try:
        if float(value) != float(value):  # NaN != NaN is True
            return ""
    except (TypeError, ValueError):
        pass
    return _clean_text(str(value))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PhoBertReviewDataset(Dataset):
    """PyTorch Dataset for Vietnamese e-commerce review sentiment classification.

    Tokenises raw review texts with the PhoBERT tokeniser and returns
    tensors required by ``AutoModelForSequenceClassification``.

    **Dynamic padding strategy** (recommended):
        Do *not* set ``padding='max_length'`` here.  Instead, pass
        ``padding=True`` only at the *collation* stage via HuggingFace's
        ``DataCollatorWithPadding``.  This means each sample in this Dataset
        is stored **without** padding – the collator pads each mini-batch to
        the longest sequence in that batch, which can reduce VRAM usage by
        30–70 % compared to global max-length padding.

    **Fixed-length padding strategy** (fallback):
        Set ``pad_to_max_length=True`` to revert to the original behaviour of
        padding every sample to ``max_length`` tokens.  This trades VRAM
        efficiency for simplicity (no custom collator required).

    Args:
        texts: Sequence of raw review strings (may contain NaN / None).
        labels: Sequence of *integer* class labels (0, 1, or 2).
        tokenizer: An initialised HuggingFace ``PreTrainedTokenizer``.
        max_length: Maximum number of tokens.  Sequences longer than this
            are truncated.  Defaults to 256 (covers > 99 % of reviews).
        pad_to_max_length: If ``True``, pad every sample to ``max_length``
            with ``padding='max_length'``.  Defaults to ``False`` (dynamic
            padding – requires ``DataCollatorWithPadding`` in the Trainer).

    Example:
        >>> from transformers import AutoTokenizer, DataCollatorWithPadding
        >>> tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        >>> dataset = PhoBertReviewDataset(texts, labels, tokenizer)
        >>> collator = DataCollatorWithPadding(tokenizer)
        >>> # Pass collator as `data_collator` to HuggingFace Trainer.
    """

    def __init__(
        self,
        texts: List[object],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        pad_to_max_length: bool = False,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and labels must have the same length, "
                f"got {len(texts)} and {len(labels)}."
            )

        self.texts: List[str] = [_resolve_text(t) for t in texts]
        self.labels: List[int] = [int(lbl) for lbl in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Tokenise one sample and return model-ready tensors.

        Args:
            idx: Zero-based sample index.

        Returns:
            A dictionary with the following keys:

            * ``"input_ids"``     – ``LongTensor`` of shape ``(seq_len,)``.
            * ``"attention_mask"``– ``LongTensor`` of shape ``(seq_len,)``.
            * ``"labels"``        – scalar ``LongTensor`` (class index).
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenise – dynamic padding leaves padding to the collator.
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length" if self.pad_to_max_length else False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),        # (seq_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (seq_len,)
            "labels": torch.tensor(label, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df,
        text_column: str = "cleaned_text",
        label_column: str = "sentiment_label",
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 256,
        pad_to_max_length: bool = False,
    ) -> "PhoBertReviewDataset":
        """Construct a dataset directly from a pandas ``DataFrame``.

        Maps Vietnamese string labels (``"tích cực"``, ``"tiêu cực"``,
        ``"trung lập"``) to integer indices 0 / 1 / 2 automatically.

        Args:
            df: A ``pandas.DataFrame`` containing the text and label columns.
            text_column: Name of the column holding review text.
            label_column: Name of the column holding Vietnamese sentiment
                labels.  Unknown labels raise a ``ValueError``.
            tokenizer: An initialised HuggingFace tokeniser.  If ``None``,
                the model checkpoint ``"vinai/phobert-base-v2"`` is loaded
                automatically.
            max_length: Maximum token length (truncation only).
            pad_to_max_length: Enable fixed-length padding.

        Returns:
            A fully initialised ``PhoBertReviewDataset``.

        Raises:
            ValueError: If ``label_column`` contains unmapped label strings.
        """
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

        unknown = set(df[label_column].dropna().unique()) - set(LABEL_MAP.keys())
        if unknown:
            raise ValueError(
                f"Unknown label(s) found in '{label_column}': {unknown}. "
                f"Expected one of: {set(LABEL_MAP.keys())}."
            )

        int_labels = df[label_column].map(LABEL_MAP).tolist()
        texts = df[text_column].tolist()

        return cls(
            texts=texts,
            labels=int_labels,
            tokenizer=tokenizer,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
        )