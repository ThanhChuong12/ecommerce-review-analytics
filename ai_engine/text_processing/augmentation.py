"""
augmentation.py
===============
Text Data Augmentation bang ky thuat Back-translation (Dich nguoc).

Muc dich: Tang cuong du lieu cho cac lop nhan Tieu cuc / Trung lap
trong tap huan luyen, giam mat can bang lop (class imbalance).

Chien luoc Back-translation:
  1. Dich van ban Tieng Viet -> Tieng Anh (vi -> en)
  2. Dich nguoc Tieng Anh -> Tieng Viet (en -> vi)
  3. Ket qua thu duoc cac mau moi co noi dung tuong tu nhung cau truc khac

Backend ho tro (co the tuning qua env var TRANSLATION_BACKEND):
  - "google_free"  : googletrans (khong can API key, co rate limit)
  - "deep_l"       : DeepL Free API (can DEEPL_API_KEY trong .env)

Usage:
    # Augment tu CSV
    py ai_engine/text_processing/augmentation.py \\
        --data-path data/processed/reviews_labeled.csv \\
        --label-col sentiment \\
        --target-labels "tieu cuc" "trung lap" \\
        --multiply 2 \\
        --output-path data/processed/reviews_augmented.csv

    # Nhanh test API
    py ai_engine/text_processing/augmentation.py --demo
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

# ── Force UTF-8 stdout ──────────────────────────────────────────────────────
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


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — TRANSLATION BACKENDS
# ════════════════════════════════════════════════════════════════════════════

class GoogleFreeTranslator:
    """Backend dich bong googletrans (free, rate-limited).

    Khong can API key nhung co the bi block neu request qua nhieu.
    Khuyen nghi dung cho prototype / so luong mau nho.
    """

    def __init__(self, delay: float = 1.0) -> None:
        """
        Args:
            delay: Thoi gian cho giua cac request (giay) de tranh rate-limit.
        """
        try:
            from googletrans import Translator  # type: ignore[import-untyped]
            self._translator = Translator()
        except ImportError as e:
            raise ImportError(
                "googletrans chua duoc cai dat. Chay: pip install googletrans==4.0.0rc1"
            ) from e
        self.delay = delay

    def translate(self, text: str, src: str, dest: str) -> str:
        """Dich van ban tu ngon ngu src sang dest.

        Args:
            text: Van ban can dich.
            src:  Ma ngon ngu nguon (vd: 'vi', 'en').
            dest: Ma ngon ngu dich (vd: 'en', 'vi').

        Returns:
            Van ban da dich, hoac chuoi rong neu co loi.
        """
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
    """Backend dich bang DeepL Free API.

    Can DEEPL_API_KEY trong .env hoac bien moi truong.
    Chinh xac hon googletrans, gioi han 500,000 ky tu/thang (free tier).
    """

    def __init__(self) -> None:
        try:
            import deepl  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "deepl chua duoc cai dat. Chay: pip install deepl"
            ) from e
        api_key = os.getenv("DEEPL_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPL_API_KEY chua duoc dat. "
                "Them vao file .env hoac bien moi truong."
            )
        self._client = deepl.Translator(api_key)

    def translate(self, text: str, src: str, dest: str) -> str:
        """Dich van ban qua DeepL API."""
        if not text or not text.strip():
            return ""
        # DeepL dung uppercase language code
        src_code = src.upper() if src.upper() != "VI" else None  # DeepL tu detect tieng Viet
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
    """Factory: tra ve translator object theo ten backend."""
    if backend == "google_free":
        return GoogleFreeTranslator(delay=delay)
    elif backend == "deep_l":
        return DeepLTranslator()
    else:
        raise ValueError(f"Unknown backend: '{backend}'. Choose 'google_free' or 'deep_l'.")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — BACK-TRANSLATION CORE
# ════════════════════════════════════════════════════════════════════════════

def back_translate_text(
    text: str,
    translator,
    pivot_lang: str = "en",
) -> str:
    """Thuc hien Back-translation cho 1 van ban.

    Pipeline: vi -> pivot_lang -> vi

    Args:
        text:        Van ban Tieng Viet goc.
        translator:  Object co method translate(text, src, dest) -> str.
        pivot_lang:  Ngon ngu trung gian (default: 'en').

    Returns:
        Van ban Tieng Viet moi (da augment), hoac chuoi rong neu dich that bai.
    """
    # Buoc 1: Tieng Viet -> Tieng Anh (hoac pivot lang)
    translated = translator.translate(text, src="vi", dest=pivot_lang)
    if not translated:
        return ""

    # Buoc 2: Tieng Anh -> Tieng Viet
    back_translated = translator.translate(translated, src=pivot_lang, dest="vi")
    return back_translated if back_translated else ""


def back_translate_batch(
    texts: List[str],
    translator,
    pivot_lang: str = "en",
    skip_short: int = 10,
) -> List[str]:
    """Thuc hien Back-translation cho 1 danh sach van ban.

    Args:
        texts:       Danh sach van ban Tieng Viet.
        translator:  Backend dich.
        pivot_lang:  Ngon ngu trung gian.
        skip_short:  Bo qua van ban ngan hon so ky tu nay (tranh loi dich).

    Returns:
        List[str]: Danh sach van ban da augment.
                   Cac mau dich that bai se giu nguyen van ban goc.
    """
    results = []
    total = len(texts)
    for i, text in enumerate(texts, 1):
        if i % 50 == 0 or i == 1:
            logger.info("Back-translating... %d/%d", i, total)

        if not text or len(text.strip()) < skip_short:
            results.append(text)  # giu nguyen neu qua ngan
            continue

        aug = back_translate_text(text, translator, pivot_lang)
        if aug and aug.strip() and aug.strip().lower() != text.strip().lower():
            results.append(aug)
        else:
            results.append(text)  # fallback: giu nguyen neu dich khong thay doi
    return results


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — AUGMENTATION PIPELINE
# ════════════════════════════════════════════════════════════════════════════

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
    """Augment cac lop thieu mau bang Back-translation.

    Chien luoc:
      - Voi moi lop trong target_labels, lay (multiply - 1) * n_existing
        mau ngau nhien, back-translate chung, roi ghep vao DataFrame goc.
      - Neu multiply = 2, tap du lieu cuoi cung co ~2x mau cho cac lop do.

    Args:
        df:            DataFrame goc co cot text va label.
        text_col:      Ten cot van ban.
        label_col:     Ten cot nhan.
        target_labels: Cac nhan muon augment (None = tu dong chon lop thieu mau nhat).
        multiply:      He so nhan so mau (default: 2).
        translator:    Backend dich (None = dung google_free).
        pivot_lang:    Ngon ngu trung gian cho back-translation.
        random_state:  Hat giong de tao lai.

    Returns:
        pd.DataFrame: DataFrame moi co them cac mau da augment.
                      Cot 'is_augmented' = True cho mau moi.
    """
    if translator is None:
        translator = get_translator("google_free")

    df = df.copy()
    df["is_augmented"] = False

    # Tu dong xac dinh target_labels neu khong truyen vao
    if target_labels is None:
        label_counts = df[label_col].value_counts()
        majority_count = label_counts.max()
        # Lay cac lop co so mau < 80% lop nhieu nhat
        target_labels = label_counts[label_counts < majority_count * 0.8].index.tolist()
        logger.info("Auto-selected minority labels: %s", target_labels)

    rng = np.random.default_rng(random_state)
    augmented_dfs = [df]

    for label in target_labels:
        subset = df[df[label_col] == label]
        n_existing = len(subset)
        n_needed = n_existing * (multiply - 1)

        if n_needed <= 0:
            logger.info("Label '%s': already has enough samples (%d), skipping.", label, n_existing)
            continue

        logger.info(
            "Augmenting label '%s': %d existing -> adding %d samples (x%d)...",
            label, n_existing, n_needed, multiply,
        )

        # Sample voi replacement neu can nhieu hon so mau hien co
        sample_indices = rng.choice(len(subset), size=n_needed, replace=(n_needed > n_existing))
        sample_texts = subset[text_col].iloc[sample_indices].tolist()

        # Back-translate
        aug_texts = back_translate_batch(sample_texts, translator, pivot_lang)

        # Tao DataFrame moi tu cac mau da augment
        aug_rows = subset.iloc[sample_indices].copy()
        aug_rows[text_col] = aug_texts
        aug_rows["is_augmented"] = True
        augmented_dfs.append(aug_rows)

        logger.info(
            "Label '%s': added %d augmented samples.",
            label, len(aug_rows),
        )

    result = pd.concat(augmented_dfs, ignore_index=True)
    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    logger.info(
        "Augmentation complete. Original: %d -> Augmented: %d samples.",
        len(df), len(result),
    )
    return result


def print_augmentation_report(df_original: pd.DataFrame, df_augmented: pd.DataFrame, label_col: str) -> None:
    """In bao cao so sanh phan phoi truoc/sau augmentation."""
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


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — DEMO MODE
# ════════════════════════════════════════════════════════════════════════════

def run_demo(backend: str = "google_free") -> None:
    """Demo nhanh back-translation tren 5 cau mau."""
    demo_texts = [
        "San pham rat dep, chat luong tot, giao hang nhanh.",
        "Hang bi loi, hop dap nat, rat that vong voi shop.",
        "Tam on, khong co gi dac biet, binh thuong thoi.",
        "Mua lan 2 roi, van dung tot, ung y lam.",
        "Giao thieu hang, phai doi may lan moi nhan duoc.",
    ]
    demo_labels = ["positive", "negative", "neutral", "positive", "negative"]

    logger.info("Running back-translation demo (%s backend)...", backend)
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
    logger.info("Demo complete.")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Text Data Augmentation via Back-translation for minority class balancing"
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run quick demo on hardcoded sample texts")
    parser.add_argument("--data-path", default=None,
                        help="Path to labeled CSV (requires --label-col)")
    parser.add_argument("--text-col", default="text",
                        help="Column name for review text (default: text)")
    parser.add_argument("--label-col", default="sentiment",
                        help="Column name for class label (default: sentiment)")
    parser.add_argument("--target-labels", nargs="+", default=None,
                        help="Labels to augment (default: auto-detect minority classes)")
    parser.add_argument("--multiply", type=int, default=2,
                        help="Target multiplier for minority class size (default: 2)")
    parser.add_argument("--output-path", default=None,
                        help="Path to save augmented CSV")
    parser.add_argument("--backend", default="google_free",
                        choices=["google_free", "deep_l"],
                        help="Translation backend (default: google_free)")
    parser.add_argument("--pivot-lang", default="en",
                        help="Pivot language for back-translation (default: en)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between API calls for google_free backend (default: 1.0)")
    args = parser.parse_args()

    if args.demo:
        run_demo(backend=args.backend)
        return

    if not args.data_path:
        parser.error("--data-path is required unless --demo is set.")

    logger.info("Loading data from: %s", args.data_path)
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
        logger.info("Augmented dataset saved -> %s", out)
    else:
        logger.info("No --output-path specified. Use --output-path to save results.")


if __name__ == "__main__":
    main()
