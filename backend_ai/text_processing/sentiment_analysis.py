"""
sentiment_analysis.py
---------------------
Phân tích cảm xúc đánh giá (Positive / Negative / Neutral) sử dụng:
  - Logistic Regression + TF-IDF (baseline)
  - PhoBERT (BERT pre-trained cho tiếng Việt)

TODO:
  - [ ] Load dữ liệu từ data/processed/
  - [ ] Tiền xử lý văn bản (tokenize, remove stopwords)
  - [ ] Huấn luyện Logistic Regression + TF-IDF
  - [ ] Fine-tune hoặc load PhoBERT
  - [ ] So sánh Accuracy, F1-score giữa hai mô hình
  - [ ] Lưu kết quả vào notebooks/ để báo cáo
"""

def analyze_sentiment_tfidf(text: str) -> str:
    """Phân tích cảm xúc với Logistic Regression + TF-IDF."""
    raise NotImplementedError("TODO: Implement TF-IDF sentiment analysis")


def analyze_sentiment_phobert(text: str) -> str:
    """Phân tích cảm xúc với PhoBERT."""
    raise NotImplementedError("TODO: Implement PhoBERT sentiment analysis")
