"""
spam_filter.py
--------------
Lọc đánh giá seeding / spam sử dụng:
  - Isolation Forest (unsupervised anomaly detection)
  - SVM (Support Vector Machine)

TODO:
  - [ ] Load dataset và tiền xử lý văn bản
  - [ ] Huấn luyện / load mô hình Isolation Forest
  - [ ] Huấn luyện / load mô hình SVM
  - [ ] So sánh hiệu năng (Precision, Recall, F1-score)
  - [ ] Export kết quả sang data/processed/
"""

def filter_spam_isolation_forest(reviews: list) -> list:
    """Lọc spam bằng Isolation Forest."""
    raise NotImplementedError("TODO: Implement Isolation Forest filter")


def filter_spam_svm(reviews: list) -> list:
    """Lọc spam bằng SVM."""
    raise NotImplementedError("TODO: Implement SVM filter")
