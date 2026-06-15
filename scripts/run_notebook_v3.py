import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Đảm bảo thư mục lưu ảnh tồn tại
os.makedirs('../model_report/graphics', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'


# Đọc dữ liệu
text_df = pd.read_csv('../data/processed/spam_labeled_text.csv')
print(f"Tổng số mẫu văn bản: {len(text_df)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Biểu đồ Sentiment
sentiment_counts = text_df['sentiment_label'].value_counts()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis', hue=sentiment_counts.index, legend=False, ax=axes[0])
axes[0].set_title('Phân bố nhãn Cảm xúc (Sentiment)', fontsize=13)
for i, v in enumerate(sentiment_counts.values):
    axes[0].text(i, v + 100, str(v), ha='center', fontsize=11)

# Biểu đồ Spam
spam_counts = text_df['is_spam'].value_counts()
spam_counts.index = ['Non-spam (0)', 'Spam (1)']
sns.barplot(x=spam_counts.index, y=spam_counts.values, palette='Set2', hue=spam_counts.index, legend=False, ax=axes[1])
axes[1].set_title('Phân bố nhãn Spam', fontsize=13)
for i, v in enumerate(spam_counts.values):
    axes[1].text(i, v + 100, str(v), ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('../model_report/graphics/text_label_distributions.png', dpi=300)
plt.show()


text_df['word_count'] = text_df['cleaned_text'].astype(str).apply(lambda x: len(x.split()))

plt.figure(figsize=(9, 5))
sns.boxplot(
    data=text_df, 
    x='sentiment_label', 
    y='word_count', 
    hue='sentiment_label', 
    palette={"tích cực": "#2ECC71", "trung lập": "#F39C12", "tiêu cực": "#E74C3C"},
    order=["tích cực", "trung lập", "tiêu cực"],
    legend=False
)
plt.title("Phân bố số lượng từ (Word Count) theo nhãn Cảm xúc", fontsize=14)
plt.xlabel("Nhãn Cảm xúc", fontsize=12)
plt.ylabel("Số lượng từ", fontsize=12)
plt.tight_layout()
plt.savefig('../model_report/graphics/text_word_count_boxplot.png', dpi=300)
plt.show()

# In thống kê cơ bản
display(text_df.groupby('sentiment_label')['word_count'].describe().round(2))


def get_top_ngrams(corpus, ngram_range=(2, 2), top_k=15):
    vec = TfidfVectorizer(ngram_range=ngram_range, max_features=5000, use_idf=False)
    X = vec.fit_transform(corpus.fillna(""))
    freq = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    df_ngram = pd.DataFrame({"ngram": vocab, "count": freq})
    return df_ngram.nlargest(top_k, "count").reset_index(drop=True)

pos_bi = get_top_ngrams(text_df[text_df["sentiment_label"] == "tích cực"]["cleaned_text"])
neg_bi = get_top_ngrams(text_df[text_df["sentiment_label"] == "tiêu cực"]["cleaned_text"])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(x="count", y="ngram", data=pos_bi, ax=axes[0], color="#2ECC71", orient="h")
axes[0].set_title("Top 15 Bigrams - Tích cực", fontsize=13)
axes[0].set_xlabel("Tần suất")
axes[0].set_ylabel("")

sns.barplot(x="count", y="ngram", data=neg_bi, ax=axes[1], color="#E74C3C", orient="h")
axes[1].set_title("Top 15 Bigrams - Tiêu cực", fontsize=13)
axes[1].set_xlabel("Tần suất")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig('../model_report/graphics/text_top_bigrams.png', dpi=300)
plt.show()


image_dir = '../image_labeling/data/labeled'
classes = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
image_counts = {c: len(os.listdir(os.path.join(image_dir, c))) for c in classes}

image_df = pd.DataFrame(list(image_counts.items()), columns=['Class', 'Count']).sort_values(by='Count', ascending=False)
plt.figure(figsize=(9, 5))
sns.barplot(data=image_df, x='Class', y='Count', palette='magma', hue='Class', legend=False)
plt.title('Phân bố nhãn Hình ảnh', fontsize=14)
for i, v in enumerate(image_df['Count']):
    plt.text(i, v + 200, str(v), ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('../model_report/graphics/image_distribution.png', dpi=300)
plt.show()


import sys
sys.path.append('../')
from ai_engine.image_processing.augmentation.transforms import get_defect_transforms

# Chọn thử 1 ảnh bị lỗi (damaged) để minh hoạ
damaged_dir = os.path.join(image_dir, 'damaged')
sample_img_name = os.listdir(damaged_dir)[0]
sample_img_path = os.path.join(damaged_dir, sample_img_name)

# Đọc ảnh gốc bằng OpenCV
img_bgr = cv2.imread(sample_img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Áp dụng Augmentation (Dùng pipeline lấy từ mã nguồn của nhóm)
transform_pipeline = get_defect_transforms()
augmented = transform_pipeline(image=img_rgb)['image']
img_aug_np = augmented.permute(1, 2, 0).numpy()
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_aug_display = np.clip(std * img_aug_np + mean, 0, 1)

# Vẽ minh hoạ
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_rgb)
axes[0].set_title("Ảnh gốc (Original)")
axes[0].axis('off')

axes[1].imshow(img_aug_display)
axes[1].set_title("Sau khi Augmentation\n(Resize, Random Rotate, Normalize...)")
axes[1].axis('off')

plt.tight_layout()
plt.savefig('../model_report/graphics/augmentation_demo.png', dpi=300)
plt.show()


# Mô phỏng chia tập dữ liệu Text (Sentiment)
X = text_df.index.values
y = text_df['sentiment_label'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

split_df = pd.DataFrame({
    'Tập dữ liệu': ['Train (80%)', 'Validation (10%)', 'Test (10%)'],
    'Tổng số mẫu': [len(y_train), len(y_val), len(y_test)],
    'Tích cực': [np.sum(y_train == 'tích cực'), np.sum(y_val == 'tích cực'), np.sum(y_test == 'tích cực')],
    'Tiêu cực': [np.sum(y_train == 'tiêu cực'), np.sum(y_val == 'tiêu cực'), np.sum(y_test == 'tiêu cực')],
    'Trung lập': [np.sum(y_train == 'trung lập'), np.sum(y_val == 'trung lập'), np.sum(y_test == 'trung lập')]
})
print("Bảng phân chia tập dữ liệu Văn bản (Sentiment):")
display(split_df)

