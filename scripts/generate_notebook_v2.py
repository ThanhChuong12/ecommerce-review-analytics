import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown: Tieu de
nb.cells.append(nbf.v4.new_markdown_cell("""# Chương 2: Tổng quan dữ liệu đầu vào
Notebook này cung cấp các phân tích EDA "vừa đủ" để trực quan hóa dữ liệu và tóm tắt phương pháp tiền xử lý dữ liệu cho cả Văn bản và Hình ảnh.
"""))

# Code: Setup
nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

# Đảm bảo thư mục lưu ảnh tồn tại
os.makedirs('../model_report/graphics', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (8, 6)
"""))

# Markdown: Text EDA
nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Dữ liệu Văn bản (Text)
### 1.1 Phân bố nhãn
Phân tích tỷ lệ các nhãn Cảm xúc và nhãn Spam trong tập dữ liệu.
"""))

# Code: Text EDA
nb.cells.append(nbf.v4.new_code_cell("""# Đọc dữ liệu
text_df = pd.read_csv('../data/processed/spam_labeled_text.csv')
print(f"Tổng số mẫu văn bản: {len(text_df)}")

# Vẽ biểu đồ phân bố Sentiment
sentiment_counts = text_df['sentiment_label'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis', hue=sentiment_counts.index, legend=False)
plt.title('Phân bố nhãn Cảm xúc (Sentiment)', fontsize=14)
for i, v in enumerate(sentiment_counts.values):
    plt.text(i, v + 100, str(v), ha='center', fontsize=11)
plt.savefig('../model_report/graphics/sentiment_distribution.png', dpi=300)
plt.show()

# Vẽ biểu đồ phân bố Spam
spam_counts = text_df['is_spam'].value_counts()
spam_counts.index = ['Non-spam (0)', 'Spam (1)']
plt.figure(figsize=(8, 6))
sns.barplot(x=spam_counts.index, y=spam_counts.values, palette='Set2', hue=spam_counts.index, legend=False)
plt.title('Phân bố nhãn Spam', fontsize=14)
for i, v in enumerate(spam_counts.values):
    plt.text(i, v + 100, str(v), ha='center', fontsize=11)
plt.savefig('../model_report/graphics/spam_distribution.png', dpi=300)
plt.show()
"""))

# Markdown: WordCloud
nb.cells.append(nbf.v4.new_markdown_cell("""### 1.2 Đám mây từ vựng (WordCloud)
Trực quan hóa các từ khóa xuất hiện nhiều nhất theo từng nhóm cảm xúc để hiểu rõ ngữ cảnh bình luận của khách hàng.
"""))

# Code: WordCloud
nb.cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
labels = ["tích cực", "trung lập", "tiêu cực"]
colormaps = ["Greens", "Oranges", "Reds"]

for ax, label, cmap in zip(axes, labels, colormaps):
    corpus = " ".join(text_df[text_df["sentiment_label"] == label]["cleaned_text"].astype(str))
    wc = WordCloud(width=500, height=300, background_color="white", colormap=cmap, max_words=100).generate(corpus)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(f"WordCloud - {label.capitalize()}", fontsize=14)
    ax.axis("off")

plt.tight_layout()
plt.savefig('../model_report/graphics/wordcloud_sentiment.png', dpi=300)
plt.show()
"""))

# Markdown: Image EDA
nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Dữ liệu Hình ảnh (Image)
### 2.1 Phân bố các lớp ảnh
Dữ liệu hình ảnh được chia thành 4 lớp: intact (nguyên vẹn), damaged (hư hỏng), wrong_item (giao sai hàng) và irrelevant (không liên quan).
"""))

# Code: Image Count
nb.cells.append(nbf.v4.new_code_cell("""image_dir = '../image_labeling/data/labeled'
classes = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
image_counts = {c: len(os.listdir(os.path.join(image_dir, c))) for c in classes}

image_df = pd.DataFrame(list(image_counts.items()), columns=['Class', 'Count']).sort_values(by='Count', ascending=False)
plt.figure(figsize=(9, 5))
sns.barplot(data=image_df, x='Class', y='Count', palette='magma', hue='Class', legend=False)
plt.title('Phân bố nhãn Hình ảnh', fontsize=14)
for i, v in enumerate(image_df['Count']):
    plt.text(i, v + 200, str(v), ha='center', fontsize=11)
plt.savefig('../model_report/graphics/image_distribution.png', dpi=300)
plt.show()
"""))

# Markdown: Image Preprocessing Explanation
nb.cells.append(nbf.v4.new_markdown_cell("""### 2.2 Chiến lược Tiền xử lý Ảnh (On-the-fly Transforms)

**Lưu ý quan trọng cho Báo cáo:** 
Đối với dữ liệu hình ảnh, chúng ta **KHÔNG** thực hiện tiền xử lý (Resize, Normalize) và Augmentation (Tăng cường dữ liệu như xoay, lật, chỉnh màu) rồi lưu ra thành các file ảnh tĩnh trên ổ cứng. Thay vào đó, toàn bộ quá trình này được cấu hình **trực tiếp trong lúc huấn luyện (On-the-fly)** thông qua Pytorch `DataLoader` và thư viện `Albumentations` (File: `ai_engine/image_processing/defect_dataloader.py`).

**Tại sao lại chọn phương pháp này?**
1. **Tránh bùng nổ dung lượng bộ nhớ:** Việc lưu hàng vạn bức ảnh đã qua xoay/lật/chỉnh sáng ra ổ đĩa sẽ tốn cực kỳ nhiều tài nguyên lưu trữ.
2. **Tính đa dạng và tối ưu chống Overfitting:** Khi thực hiện biến đổi *on-the-fly*, ở mỗi một Epoch (vòng lặp huấn luyện), mô hình sẽ nhìn thấy một phiên bản biến đổi ngẫu nhiên khác nhau của cùng một bức ảnh gốc. Điều này giúp mô hình học được rất nhiều biến thể, qua đó tổng quát hóa tốt hơn. Nếu lưu ảnh tĩnh offline, tính ngẫu nhiên này bị mất đi.
3. **Tiêu chuẩn ngành (Best Practice):** Biến đổi dữ liệu trực tiếp trong hàm `__getitem__` của DataLoader là chuẩn mực khi làm việc với PyTorch và Deep Learning hiện hành.

Dưới đây là một ví dụ minh họa trực quan sự thay đổi của ảnh trước và sau khi đi qua pipeline Augmentation:
"""))

# Code: Image Augmentation Demo
nb.cells.append(nbf.v4.new_code_cell("""import sys
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
# Chuyển tensor CHW về HWC để vẽ bằng matplotlib, giải chuẩn hoá để hiển thị
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
axes[1].set_title("Sau khi Augmentation & Normalize\\n(Resize 224x224, Random Rotate/Color...)")
axes[1].axis('off')

plt.tight_layout()
plt.savefig('../model_report/graphics/augmentation_demo.png', dpi=300)
plt.show()
"""))

# Markdown: Split
nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Phân chia tập dữ liệu (Train/Val/Test Split)
Vì dữ liệu bị mất cân bằng lớp (imbalanced), chúng ta sử dụng **Stratified Split** theo tỷ lệ 80% Train, 10% Val, 10% Test.
Điều này đảm bảo phân bố nhãn trong từng tập con giống hệt với tập tổng thể, giúp mô hình học và đánh giá khách quan hơn.
"""))

# Code: Split
nb.cells.append(nbf.v4.new_code_cell("""# Mô phỏng chia tập dữ liệu Text (Sentiment)
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
"""))


with open('notebooks/05_data_overview_for_report.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Đã ghi file 05_data_overview_for_report.ipynb thành công!")
