"""
lime_error_analysis.py
======================
Phân tích lỗi (Error Analysis) dùng LIME cho bài toán phân tích cảm xúc.

Yêu cầu bài báo cáo:
  - Lấy 1-2 ví dụ cụ thể mà Baseline / PhoBERT đoán sai.
  - Dùng LIME để đưa ra giả thuyết tại sao sai.

Cách chạy (từ thư mục gốc):
    # Phân tích cả 2 model:
    py scripts/lime_error_analysis.py

    # Chỉ Baseline:
    py scripts/lime_error_analysis.py --model baseline

    # Chỉ PhoBERT:
    py scripts/lime_error_analysis.py --model phobert

    # Xem N ví dụ sai (default: 2):
    py scripts/lime_error_analysis.py --n-examples 3

Output:
  - In giải thích ra console
  - Lưu HTML report vào artifacts/lime/ (mở bằng trình duyệt để xem highlight từ)
"""

from __future__ import annotations

import argparse
import io
import logging
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
BASELINE_MODEL   = ROOT / "artifacts/models/tuned/tuned_voting_ensemble.pkl"
PHOBERT_DIR      = ROOT / "artifacts/models/phobert"
TEST_CSV         = ROOT / "data/processed/processed_labeled_text_test.csv"
OUTPUT_DIR       = ROOT / "artifacts/lime"

LABEL_ORDER = ["tích cực", "tiêu cực", "trung lập"]  # thứ tự nhất quán với PhoBERT config


# ══════════════════════════════════════════════════════════════════════════════
#  1. WRAPPER BASELINE
# ══════════════════════════════════════════════════════════════════════════════

class BaselineWrapper:
    """Bọc TextEnsembleModel để LIME có thể gọi predict_proba(list[str])."""

    def __init__(self, model_path: str):
        import joblib
        logger.info("Đang load Baseline model từ %s ...", model_path)
        obj = joblib.load(model_path)

        # model pkl có thể là TextEnsembleModel hoặc Pipeline trực tiếp
        if hasattr(obj, "pipeline") and obj.pipeline is not None:
            # TextEnsembleModel wrapper
            self.pipeline = obj.pipeline
        else:
            # imblearn Pipeline trực tiếp
            self.pipeline = obj

        self.classes_ = self.pipeline.classes_
        logger.info("Baseline classes: %s", self.classes_)

    def predict_labels(self, texts):
        return self.pipeline.predict(pd.Series(texts))

    def predict_proba_lime(self, texts):
        """LIME gọi hàm này với list[str], trả về (n, n_classes)."""
        proba = self.pipeline.predict_proba(pd.Series(texts))
        # Sắp xếp lại theo LABEL_ORDER nếu cần
        label_to_col = {lbl: i for i, lbl in enumerate(self.classes_)}
        available = [lbl for lbl in LABEL_ORDER if lbl in label_to_col]
        reordered = np.column_stack([
            proba[:, label_to_col[lbl]]
            for lbl in available
        ])
        return reordered




# ══════════════════════════════════════════════════════════════════════════════
#  3. TÌM CÁC CÂU ĐoÁN SAI
# ══════════════════════════════════════════════════════════════════════════════

def find_wrong_predictions(
    model_wrapper,
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    n: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Lấy N câu đầu tiên mà model dự đoán sai.

    Ưu tiên lấy các loại lỗi đa dạng:
      - Câu ngắn (< 10 từ): dễ bị sai do thiếu ngữ cảnh
      - Câu có emoji/icon: mỉa mai, khó phân loại
      - Câu trung lập bị đoán nhầm: class khó nhất
    """
    logger.info("Đang dự đoán trên %d mẫu test ...", len(df))
    texts = df[text_col].fillna("").astype(str).tolist()
    labels_true = df[label_col].tolist()

    preds = model_wrapper.predict_labels(texts)

    df_eval = df[[text_col, label_col]].copy()
    df_eval["predicted"] = preds
    df_eval["wrong"] = df_eval[label_col] != df_eval["predicted"]

    wrong_df = df_eval[df_eval["wrong"]].copy()
    logger.info("Tổng câu đoán sai: %d / %d", len(wrong_df), len(df))

    # Ưu tiên lấy câu trung lập bị đoán sai (class khó nhất)
    neutral_wrong = wrong_df[wrong_df[label_col] == "trung lập"]
    non_neutral_wrong = wrong_df[wrong_df[label_col] != "trung lập"]

    picked = []
    if len(neutral_wrong) > 0:
        picked.append(neutral_wrong.sample(min(1, len(neutral_wrong)), random_state=seed))
    if len(picked) < n and len(non_neutral_wrong) > 0:
        rem = n - len(picked)
        picked.append(non_neutral_wrong.sample(min(rem, len(non_neutral_wrong)), random_state=seed))

    result = pd.concat(picked).head(n)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  4. CHẠY LIME VÀ IN KẾT QUẢ
# ══════════════════════════════════════════════════════════════════════════════

def run_lime_on_sample(
    model_wrapper,
    text: str,
    true_label: str,
    pred_label: str,
    model_name: str,
    class_names: list[str],
    sample_idx: int,
    num_features: int = 8,
    num_samples: int = 500,
) -> dict:
    """Chạy LIME cho 1 câu và in giải thích chi tiết.

    Returns:
        dict với top features và scores.
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        logger.error("LIME chưa cài! Chạy: pip install lime")
        sys.exit(1)

    explainer = LimeTextExplainer(class_names=class_names, random_state=42)

    label_to_idx = {lbl: i for i, lbl in enumerate(class_names)}
    target_idx = label_to_idx.get(pred_label, 0)  # Giải thích tại sao model đoán pred_label

    logger.info(
        "[LIME] Đang tạo explanation cho câu #%d (target class: '%s') ...",
        sample_idx, pred_label
    )
    exp = explainer.explain_instance(
        text,
        model_wrapper.predict_proba_lime,
        num_features=num_features,
        num_samples=num_samples,
        labels=[target_idx],
    )

    # ── In ra console ─────────────────────────────────────────────────────────
    SEP = "=" * 70
    print(f"\n{SEP}")
    print(f"  [{model_name}] VÍ DỤ #{sample_idx}")
    print(SEP)
    print(f"  VĂN BẢN  : {text[:200]}{'...' if len(text) > 200 else ''}")
    print(f"  NHÃN THẬT: {true_label}")
    print(f"  MÔ HÌNH ĐOÁN: {pred_label}  ← SAI")
    print(f"\n  [LIME] Các từ ảnh hưởng đến dự đoán '{pred_label}':")
    print(f"  {'Từ/cụm':<25} {'Ảnh hưởng':>12}  {'Hướng'}")
    print(f"  {'-'*25} {'-'*12}  {'-'*10}")

    features = exp.as_list(label=target_idx)
    for feat, weight in features:
        direction = "→ ủng hộ" if weight > 0 else "→ phản đối"
        print(f"  {feat:<25} {weight:>+12.4f}  {direction}")

    # ── Giả thuyết tự động ────────────────────────────────────────────────────
    print(f"\n  [GIẢ THUYẾT]")
    pos_feats = [(f, w) for f, w in features if w > 0]
    neg_feats = [(f, w) for f, w in features if w < 0]

    # Phát hiện emoji/icon trong văn bản gốc
    import re
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0\U000024C2-\U0001F251]+",
        flags=re.UNICODE,
    )
    has_emoji = bool(emoji_pattern.search(text))

    # Phát hiện từ phủ định gần từ tích cực
    negation_words = ["không", "chưa", "chẳng", "đừng", "chả", "ko", "k ", "kh "]
    has_negation = any(w in text.lower() for w in negation_words)

    # Từ hiếm: top feature ủng hộ nhưng nằm ở đuôi câu ngắn
    word_count = len(text.split())

    hypotheses = []

    if has_emoji and true_label in ["tiêu cực", "trung lập"]:
        hypotheses.append(
            "⚠️  Câu chứa emoji — model có thể bị nhầm emoji mang ý tích cực "
            "trong khi người dùng dùng theo kiểu mỉa mai hoặc ngữ cảnh phức tạp."
        )
    if has_negation and pos_feats:
        hypotheses.append(
            "⚠️  Câu có từ phủ định (không/chưa/ko) kết hợp với từ tích cực "
            f"('{pos_feats[0][0]}') — TF-IDF xử lý từng từ riêng lẻ nên "
            "bỏ qua cấu trúc phủ định → model bị nhiễu đặc trưng."
        )
    if word_count < 8 and true_label == "trung lập":
        hypotheses.append(
            f"⚠️  Câu rất ngắn ({word_count} từ) — ít đặc trưng → model thiên "
            "về class đa số (tích cực) thay vì nhận ra sắc thái trung lập."
        )
    if neg_feats and true_label == "tích cực":
        neg_words_str = ", ".join([f"'{f}'" for f, _ in neg_feats[:2]])
        hypotheses.append(
            f"⚠️  Từ {neg_words_str} kéo model về hướng tiêu cực — "
            "có thể là từ khóa hiếm gặp trong ngữ cảnh phê bình nhưng thực ra "
            "câu tổng thể vẫn tích cực."
        )
    if not hypotheses:
        hypotheses.append(
            "⚠️  Không phát hiện pattern rõ ràng — có thể do từ vựng mơ hồ "
            "hoặc câu có cấu trúc phức tạp (câu điều kiện, so sánh, ẩn dụ)."
        )

    for h in hypotheses:
        print(f"  {h}")

    print(SEP + "\n")

    return {
        "text": text,
        "true_label": true_label,
        "pred_label": pred_label,
        "features": features,
        "hypotheses": hypotheses,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LIME Error Analysis cho Baseline (TF-IDF + Voting Ensemble)"
    )
    parser.add_argument(
        "--n-examples", type=int, default=2,
        help="Số ví dụ sai cần phân tích (default: 2)"
    )
    parser.add_argument(
        "--baseline-path", default=str(BASELINE_MODEL),
        help=f"Path model Baseline pkl (default: {BASELINE_MODEL})"
    )
    parser.add_argument(
        "--test-csv", default=str(TEST_CSV),
        help=f"CSV test (default: {TEST_CSV})"
    )
    parser.add_argument(
        "--text-col", default="cleaned_text",
        help="Cột văn bản (default: cleaned_text)"
    )
    parser.add_argument(
        "--label-col", default="sentiment_label",
        help="Cột nhãn thật (default: sentiment_label)"
    )
    parser.add_argument(
        "--num-features", type=int, default=8,
        help="Số đặc trưng LIME hiển thị (default: 8)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=500,
        help="Số mẫu perturbation LIME (default: 500, tăng để chính xác hơn)"
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Đọc dữ liệu test từ: %s", args.test_csv)
    df = pd.read_csv(args.test_csv)
    df[args.text_col] = df[args.text_col].fillna("").astype(str)

    # ── Chạy Baseline ─────────────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  PHÂN TÍCH LỖI: BASELINE (TF-IDF + Voting Ensemble)")
    print("█" * 70)

    baseline = BaselineWrapper(args.baseline_path)
    wrong_df = find_wrong_predictions(
        baseline, df, args.text_col, args.label_col, n=args.n_examples
    )

    for i, (_, row) in enumerate(wrong_df.iterrows(), start=1):
        run_lime_on_sample(
            model_wrapper=baseline,
            text=row[args.text_col],
            true_label=row[args.label_col],
            pred_label=row["predicted"],
            model_name="Baseline",
            class_names=LABEL_ORDER,
            sample_idx=i,
            num_features=args.num_features,
            num_samples=args.num_samples,
        )

    print("\n✅ Xong!")


if __name__ == "__main__":
    main()
