"""Text data augmentation module using Back-Translation.

Enhances training datasets for minority sentiment classes (Negative / Neutral)
to mitigate class imbalance using back-translation (vi -> pivot -> vi).
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class GoogleFreeTranslator:
    """Free Google Translate backend client with rate-limiting support."""

    def __init__(self, delay: float = 1.0) -> None:
        try:
            from googletrans import Translator
            self._translator = Translator()
        except ImportError as e:
            raise ImportError(
                "googletrans is not installed. Install with: pip install googletrans==4.0.0rc1"
            ) from e
        self.delay = delay

    def translate(self, text: str, src: str, dest: str) -> str:
        """Translate text from source language to destination language."""
        if not text or not text.strip():
            return ""
        try:
            result = self._translator.translate(text, src=src, dest=dest)
            time.sleep(self.delay)
            return result.text if result and result.text else ""
        except Exception as e:
            logger.warning("Translation failed (%s->%s): %s | text: %.60s...", src, dest, e, text)
            return ""


class DeepLTranslator:
    """DeepL API translation client for high-accuracy back-translation."""

    def __init__(self) -> None:
        try:
            import deepl
        except ImportError as e:
            raise ImportError("deepl is not installed. Install with: pip install deepl") from e
        api_key = os.getenv("DEEPL_API_KEY")
        if not api_key:
            raise ValueError("DEEPL_API_KEY is not set in environment or .env file.")
        self._client = deepl.Translator(api_key)

    def translate(self, text: str, src: str, dest: str) -> str:
        """Translate text using DeepL API."""
        if not text or not text.strip():
            return ""
        src_code = src.upper() if src.upper() != "VI" else None
        dest_code = "VI" if dest.lower() == "vi" else dest.upper()
        try:
            result = self._client.translate_text(
                text,
                source_lang=src_code,
                target_lang=dest_code,
            )
            return str(result)
        except Exception as e:
            logger.warning("DeepL translation failed: %s", e)
            return ""


def get_translator(backend: str = "google_free", delay: float = 1.0):
    """Factory function creating a translator client based on backend name."""
    if backend == "google_free":
        return GoogleFreeTranslator(delay=delay)
    if backend == "deep_l":
        return DeepLTranslator()
    raise ValueError(f"Unknown backend: '{backend}'. Choose 'google_free' or 'deep_l'.")


def back_translate_text(
    text: str,
    translator,
    pivot_lang: str = "en",
) -> str:
    """Execute back-translation on a single text string (vi -> pivot -> vi)."""
    translated = translator.translate(text, src="vi", dest=pivot_lang)
    if not translated:
        return ""
    back_translated = translator.translate(translated, src=pivot_lang, dest="vi")
    return back_translated if back_translated else ""


def back_translate_batch(
    texts: List[str],
    translator,
    pivot_lang: str = "en",
    skip_short: int = 10,
) -> List[str]:
    """Execute back-translation across a batch of text items."""
    results = []
    total = len(texts)
    for i, text in enumerate(texts, 1):
        if i % 50 == 0 or i == 1:
            logger.info("Back-translating batch... %d/%d", i, total)

        if not text or len(text.strip()) < skip_short:
            results.append(text)
            continue

        aug = back_translate_text(text, translator, pivot_lang)
        if aug and aug.strip() and aug.strip().lower() != text.strip().lower():
            results.append(aug)
        else:
            results.append(text)
    return results


def augment_minority_classes(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "sentiment",
    target_labels: Optional[List[str]] = None,
    multiply: int = 2,
    translator=None,
    pivot_lang: str = "en",
    random_state: int = 42,
) -> pd.DataFrame:
    """Augment specified minority classes in a DataFrame using back-translation."""
    if translator is None:
        translator = get_translator("google_free")

    df = df.copy()
    df["is_augmented"] = False

    if target_labels is None:
        label_counts = df[label_col].value_counts()
        majority_count = label_counts.max()
        target_labels = label_counts[label_counts < majority_count * 0.8].index.tolist()
        logger.info("Auto-selected minority labels: %s", target_labels)

    rng = np.random.default_rng(random_state)
    augmented_dfs = [df]

    for label in target_labels:
        subset = df[df[label_col] == label]
        n_existing = len(subset)
        n_needed = n_existing * (multiply - 1)

        if n_needed <= 0:
            logger.info("Label '%s' has sufficient samples (%d), skipping.", label, n_existing)
            continue

        logger.info(
            "Augmenting label '%s': %d existing -> adding %d samples (x%d)...",
            label, n_existing, n_needed, multiply,
        )

        sample_indices = rng.choice(len(subset), size=n_needed, replace=(n_needed > n_existing))
        sample_texts = subset[text_col].iloc[sample_indices].tolist()
        aug_texts = back_translate_batch(sample_texts, translator, pivot_lang)

        aug_rows = subset.iloc[sample_indices].copy()
        aug_rows[text_col] = aug_texts
        aug_rows["is_augmented"] = True
        augmented_dfs.append(aug_rows)

        logger.info("Label '%s': added %d augmented samples.", label, len(aug_rows))

    result = pd.concat(augmented_dfs, ignore_index=True)
    return result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def print_augmentation_report(df_original: pd.DataFrame, df_augmented: pd.DataFrame, label_col: str) -> None:
    """Print class distribution comparison report before and after augmentation."""
    print("\n" + "=" * 60)
    print("  AUGMENTATION REPORT")
    print("=" * 60)
    print(f"  {'Label':<20} {'Before':>8}  {'After':>8}  {'Added':>8}")
    print(f"  {'-'*20} {'-'*8}  {'-'*8}  {'-'*8}")

    before_counts = df_original[label_col].value_counts()
    after_counts = df_augmented[label_col].value_counts()
    all_labels = sorted(set(before_counts.index) | set(after_counts.index))

    for label in all_labels:
        before = before_counts.get(label, 0)
        after = after_counts.get(label, 0)
        added = after - before
        print(f"  {str(label):<20} {before:>8,}  {after:>8,}  {added:>8,}")

    print(f"  {'TOTAL':<20} {len(df_original):>8,}  {len(df_augmented):>8,}  {len(df_augmented)-len(df_original):>8,}")
    print("=" * 60 + "\n")


def run_demo(backend: str = "google_free") -> None:
    """Execute quick demonstration of back-translation on sample review sentences."""
    demo_texts = [
        "Sản phẩm rất đẹp, chất lượng tốt, giao hàng nhanh.",
        "Hàng bị lỗi, hộp dập nát, rất thất vọng với shop.",
        "Tạm ổn, không có gì đặc biệt, bình thường thôi.",
        "Mua lần 2 rồi, vẫn dùng tốt, ưng ý lắm.",
        "Giao thiếu hàng, phải đòi mấy lần mới nhận được.",
    ]
    demo_labels = ["positive", "negative", "neutral", "positive", "negative"]

    logger.info("Running back-translation demo using '%s' backend...", backend)
    translator = get_translator(backend)

    print("\n" + "=" * 70)
    print("  BACK-TRANSLATION DEMO")
    print("=" * 70)

    for i, (text, label) in enumerate(zip(demo_texts, demo_labels), 1):
        aug = back_translate_text(text, translator)
        print(f"\n  [{i}] Label: {label}")
        print(f"      Original : {text}")
        print(f"      Augmented: {aug if aug else '[translation failed]'}")

    print("\n" + "=" * 70 + "\n")
    logger.info("Demo execution completed.")


def main() -> None:
    """CLI entrypoint for text data augmentation script."""
    parser = argparse.ArgumentParser(
        description="Text Data Augmentation via Back-translation for minority class balancing"
    )
    parser.add_argument("--demo", action="store_true", help="Run quick demo on sample texts")
    parser.add_argument("--data-path", default=None, help="Path to labeled input CSV")
    parser.add_argument("--text-col", default="text", help="Column name for review text")
    parser.add_argument("--label-col", default="sentiment", help="Column name for target label")
    parser.add_argument("--target-labels", nargs="+", default=None, help="Labels to augment")
    parser.add_argument("--multiply", type=int, default=2, help="Sample count multiplier")
    parser.add_argument("--output-path", default=None, help="Path to save output CSV")
    parser.add_argument("--backend", default="google_free", choices=["google_free", "deep_l"])
    parser.add_argument("--pivot-lang", default="en", help="Pivot language code")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API requests")
    args = parser.parse_args()

    if args.demo:
        run_demo(backend=args.backend)
        return

    if not args.data_path:
        parser.error("--data-path is required unless --demo is set.")

    logger.info("Loading dataset from '%s'", args.data_path)
    df = pd.read_csv(args.data_path)

    for col in [args.text_col, args.label_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    translator = get_translator(args.backend, delay=args.delay)

    df_aug = augment_minority_classes(
        df=df,
        text_col=args.text_col,
        label_col=args.label_col,
        target_labels=args.target_labels,
        multiply=args.multiply,
        translator=translator,
        pivot_lang=args.pivot_lang,
    )

    print_augmentation_report(df, df_aug, label_col=args.label_col)

    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_aug.to_csv(out, index=False, encoding="utf-8-sig")
        logger.info("Saved augmented dataset to '%s'", out)
    else:
        logger.info("No --output-path provided. Output not saved.")


if __name__ == "__main__":
    main()
