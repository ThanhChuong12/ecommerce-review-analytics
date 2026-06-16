import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

# Cấu hình đồ thị
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (8, 6)

os.makedirs('model_report/graphics', exist_ok=True)

# 1. Text Data EDA
print("--- DỮ LIỆU VĂN BẢN (TEXT) ---")
text_df = pd.read_csv('data/processed/spam_labeled_text.csv')
print(f"Tổng số mẫu văn bản: {len(text_df)}\n")

sentiment_counts = text_df['sentiment_label'].value_counts()
print("Phân bố nhãn Cảm xúc:")
print(sentiment_counts)

plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis', hue=sentiment_counts.index, legend=False)
plt.title('Phân bố nhãn Cảm xúc (Sentiment)', fontsize=14)
plt.xlabel('Nhãn', fontsize=12)
plt.ylabel('Số lượng mẫu', fontsize=12)
for i, v in enumerate(sentiment_counts.values):
    plt.text(i, v + 100, str(v), ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('model_report/graphics/sentiment_distribution.png', dpi=300)
plt.close()

spam_counts = text_df['is_spam'].value_counts()
spam_counts.index = ['Non-spam (0)', 'Spam (1)']
print("\nPhân bố nhãn Spam:")
print(spam_counts)

plt.figure(figsize=(8, 6))
sns.barplot(x=spam_counts.index, y=spam_counts.values, palette='Set2', hue=spam_counts.index, legend=False)
plt.title('Phân bố nhãn Spam', fontsize=14)
plt.xlabel('Nhãn', fontsize=12)
plt.ylabel('Số lượng mẫu', fontsize=12)
for i, v in enumerate(spam_counts.values):
    plt.text(i, v + 100, str(v), ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('model_report/graphics/spam_distribution.png', dpi=300)
plt.close()

# 2. Image Data EDA
print("\n--- DỮ LIỆU HÌNH ẢNH (IMAGE) ---")
image_dir = 'image_labeling/data/labeled'
classes = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]

image_counts = {}
for c in classes:
    image_counts[c] = len(os.listdir(os.path.join(image_dir, c)))

image_df = pd.DataFrame(list(image_counts.items()), columns=['Class', 'Count'])
image_df = image_df.sort_values(by='Count', ascending=False)
print("Phân bố hình ảnh:")
print(image_df.to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(data=image_df, x='Class', y='Count', palette='magma', hue='Class', legend=False)
plt.title('Phân bố nhãn Hình ảnh', fontsize=14)
plt.xlabel('Lớp (Class)', fontsize=12)
plt.ylabel('Số lượng ảnh', fontsize=12)
for i, v in enumerate(image_df['Count']):
    plt.text(i, v + 200, str(v), ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('model_report/graphics/image_distribution.png', dpi=300)
plt.close()

# 3. Data Split
print("\n--- PHÂN CHIA TẬP DỮ LIỆU ---")
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
print(split_df.to_string(index=False))

print("\n--- HOÀN THÀNH ---")
