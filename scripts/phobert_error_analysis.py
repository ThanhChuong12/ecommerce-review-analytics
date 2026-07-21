"""
phobert_error_analysis.py
=========================
Error analysis for PhoBERT using Captum Integrated Gradients.

Unlike LIME, Integrated Gradients computes gradients backwards through
all Transformer layers -> attribution score reflects internal mechanism.

Usage:
  python scripts/phobert_error_analysis.py [--n-examples 3] [--sample-size 200]
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Force UTF-8 ───────────────────────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PHOBERT_DIR = ROOT / "artifacts/models/phobert"
TEST_CSV    = ROOT / "data/processed/processed_labeled_text_test.csv"
LABELS      = ["tích cực", "tiêu cực", "trung lập"]


# ══════════════════════════════════════════════════════════════════════════════
#  1. LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_phobert(model_dir: str):
    """Load PhoBERT model and tokenizer."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading PhoBERT from %s (device: %s) ...", model_dir, device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device).eval()

    logger.info("PhoBERT loaded. id2label: %s", model.config.id2label)
    return model, tokenizer, device


# ══════════════════════════════════════════════════════════════════════════════
#  2. PREDICT
# ══════════════════════════════════════════════════════════════════════════════

def predict_batch(texts: list[str], model, tokenizer, device, batch_size: int = 16) -> np.ndarray:
    """Predict probabilities in batches to avoid OOM."""
    import torch
    import torch.nn.functional as F

    all_proba = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        all_proba.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(all_proba)


# ══════════════════════════════════════════════════════════════════════════════
#  3. TÌM CÂU ĐoÁN SAI
# ══════════════════════════════════════════════════════════════════════════════

def find_wrong_predictions(
    model, tokenizer, device,
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    n: int = 2,
    sample_size: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample texts, predict, and return N misclassified examples.
    Prioritizes neutral misclassifications.
    """
    df_sample = df.sample(min(sample_size, len(df)), random_state=seed).copy()
    texts = df_sample[text_col].fillna("").astype(str).tolist()

    logger.info("Predicting %d texts ...", len(texts))
    proba = predict_batch(texts, model, tokenizer, device)
    df_sample["predicted"] = [LABELS[i] for i in proba.argmax(axis=1)]
    df_sample["wrong"]     = df_sample[label_col] != df_sample["predicted"]

    wrong_df = df_sample[df_sample["wrong"]].copy()
    logger.info("Misclassifications: %d / %d", len(wrong_df), len(df_sample))

    neutral_wrong     = wrong_df[wrong_df[label_col] == "trung lập"]
    non_neutral_wrong = wrong_df[wrong_df[label_col] != "trung lập"]

    picked = []
    if len(neutral_wrong) > 0:
        picked.append(neutral_wrong.sample(min(1, len(neutral_wrong)), random_state=seed))
    if len(picked) < n and len(non_neutral_wrong) > 0:
        rem = n - len(picked)
        picked.append(non_neutral_wrong.sample(min(rem, len(non_neutral_wrong)), random_state=seed))

    return pd.concat(picked).head(n) if picked else wrong_df.head(n)


# ══════════════════════════════════════════════════════════════════════════════
#  4. INTEGRATED GRADIENTS
# ══════════════════════════════════════════════════════════════════════════════

def explain_with_ig(
    text: str,
    target_class_idx: int,
    model,
    tokenizer,
    device,
    top_k: int = 8,
) -> list[tuple[str, float]]:
    """Compute Integrated Gradients for one text.
    Returns:
        List of (word, score) sorted by |score| descending.
    """
    import torch
    from captum.attr import LayerIntegratedGradients

    def forward_func(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    lig = LayerIntegratedGradients(
        forward_func, model.roberta.embeddings.word_embeddings
    )

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256
    ).to(device)
    input_ids      = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Baseline = toàn PAD token
    ref_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    ref_ids[0, 0]  = tokenizer.cls_token_id
    ref_ids[0, -1] = tokenizer.sep_token_id

    attributions, _ = lig.attribute(
        inputs=input_ids,
        baselines=ref_ids,
        additional_forward_args=(attention_mask,),
        target=target_class_idx,
        return_convergence_delta=True,
    )

    # Tổng qua embedding dim + normalize
    attr = attributions.sum(dim=-1).squeeze(0)
    norm = torch.norm(attr)
    if norm > 0:
        attr = attr / norm
    attr = attr.cpu().detach().numpy()

    tokens  = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    special = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}

    # Merge subwords (@@) into complete words
    word_scores: list[tuple[str, float]] = []
    cur_word, cur_score = "", 0.0
    for token, score in zip(tokens, attr):
        if token in special:
            continue
        if token.endswith("@@"):
            cur_word  += token[:-2]
            cur_score += float(score)
        else:
            cur_word  += token
            cur_score += float(score)
            word_scores.append((cur_word.replace("_", " "), cur_score))
            cur_word, cur_score = "", 0.0

    # Sort by |score| descending and take top_k
    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return word_scores[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
#  5. IN KẾT QUẢ + GIẢ THUYẾT
# ══════════════════════════════════════════════════════════════════════════════

def analyze_sample(
    text: str,
    true_label: str,
    pred_label: str,
    word_scores: list[tuple[str, float]],
    sample_idx: int,
) -> None:
    """Print analysis results and hypotheses."""
    SEP = "=" * 70
    print(f"\\n{SEP}")
    print(f"  [PhoBERT - IG] EXAMPLE #{sample_idx}")
    print(SEP)
    print(f"  TEXT     : {text[:200]}{'...' if len(text) > 200 else ''}")
    print(f"  TRUE     : {true_label}")
    print(f"  PREDICTED: {pred_label}  <- WRONG")

    print(f"\\n  [IG] Attribution scores for '{pred_label}':")
    print(f"  {'Word':<22} {'Score':>9}  Direction")
    print(f"  {'-'*22} {'-'*9}  {'-'*10}")
    for word, score in word_scores:
        direction = "-> supports" if score > 0 else "-> opposes"
        print(f"  {word:<22} {score:>+9.4f}  {direction}")

    # ── Hypotheses ────────────────────────────────────────────────────────────
    print(f"\\n  [HYPOTHESES]")
    pos_feats = [(w, s) for w, s in word_scores if s > 0]
    neg_feats = [(w, s) for w, s in word_scores if s < 0]

    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0\U000024C2-\U0001F251]+",
        flags=re.UNICODE,
    )
    has_emoji    = bool(emoji_pattern.search(text))
    neg_words    = ["không", "chưa", "chẳng", "đừng", "chả", "ko", "k ", "kh "]
    has_negation = any(w in text.lower() for w in neg_words)
    word_count   = len(text.split())

    hypotheses = []
    if has_emoji and true_label in ["tiêu cực", "trung lập"]:
        hypotheses.append(
            "⚠️  Contains emojis — model might confuse positive emojis "
            "in a sarcastic/complaint context."
        )
    if has_negation and pos_feats:
        hypotheses.append(
            f"⚠️  Contains negation near positive word "
            f"('{pos_feats[0][0]}' score={pos_feats[0][1]:+.3f}) — "
            "PhoBERT caught context but is still biased by the positive word."
        )
    if word_count < 8 and true_label == "trung lập":
        hypotheses.append(
            f"⚠️  Very short sentence ({word_count} words) — lacks context, "
            "so model defaults to majority class (positive)."
        )
    if neg_feats and true_label == "tích cực":
        neg_str = ", ".join([f"'{w}'" for w, _ in neg_feats[:2]])
        hypotheses.append(
            f"⚠️  Words {neg_str} push strongly against prediction — "
            "might be rare technical terms in training data."
        )
    if not hypotheses:
        hypotheses.append(
            "⚠️  No clear pattern detected — possibly complex structure "
            "or ambiguous context."
        )

    for h in hypotheses:
        print(f"  {h}")

    print(SEP + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PhoBERT Error Analysis using Captum Integrated Gradients"
    )
    parser.add_argument(
        "--phobert-dir", default=str(PHOBERT_DIR),
        help=f"Thư mục model PhoBERT (default: {PHOBERT_DIR})"
    )
    parser.add_argument(
        "--test-csv", default=str(TEST_CSV),
        help=f"CSV test (default: {TEST_CSV})"
    )
    parser.add_argument(
        "--text-col", default="text",
        help="Cột văn bản (default: text)"
    )
    parser.add_argument(
        "--label-col", default="sentiment_label",
        help="Cột nhãn thật (default: sentiment_label)"
    )
    parser.add_argument(
        "--n-examples", type=int, default=2,
        help="Số ví dụ sai cần phân tích (default: 2)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=100,
        help="Số câu lấy từ test set để tìm câu sai (default: 100)"
    )
    parser.add_argument(
        "--top-k", type=int, default=8,
        help="Số từ IG hiển thị (default: 8)"
    )
    args = parser.parse_args()

    # ── Kiểm tra captum ────────────────────────────────────────────────────────
    try:
        import captum  # noqa: F401
    except ImportError:
        logger.error("Captum chưa cài! Chạy: py -m pip install captum")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer, device = load_phobert(args.phobert_dir)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Reading test data from: %s", args.test_csv)
    df = pd.read_csv(args.test_csv)
    df[args.text_col] = df[args.text_col].fillna("").astype(str)

    # ── Find misclassifications ───────────────────────────────────────────────
    print("\\n" + "█" * 70)
    print("  ERROR ANALYSIS: PHOBERT (Integrated Gradients)")
    print("█" * 70)

    wrong_df = find_wrong_predictions(
        model, tokenizer, device, df,
        text_col=args.text_col,
        label_col=args.label_col,
        n=args.n_examples,
        sample_size=args.sample_size,
    )

    # ── Chạy IG + in kết quả ─────────────────────────────────────────────────
    for i, (_, row) in enumerate(wrong_df.iterrows(), start=1):
        pred_label = row["predicted"]
        pred_idx   = LABELS.index(pred_label)

        logger.info(
            "[IG] Computing attribution for sample #%d (target: '%s') ...",
            i, pred_label
        )
        word_scores = explain_with_ig(
            text=row[args.text_col],
            target_class_idx=pred_idx,
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_k=args.top_k,
        )

        analyze_sample(
            text=row[args.text_col],
            true_label=row[args.label_col],
            pred_label=pred_label,
            word_scores=word_scores,
            sample_idx=i,
        )

    print("✅ Done!")


if __name__ == "__main__":
    main()
