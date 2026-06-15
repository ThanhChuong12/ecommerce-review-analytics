import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Markdown: Introduction
nb.cells.append(nbf.v4.new_markdown_cell("""# Tổng quan dữ liệu đầu vào (Chapter 2)
Notebook này thực hiện Khai phá dữ liệu (EDA), phân chia tập dữ liệu và tóm tắt quá trình tiền xử lý cho cả dữ liệu Văn bản và Hình ảnh. Các biểu đồ và bảng số liệu sinh ra ở đây sẽ được sử dụng trực tiếp cho Chương 2 của báo cáo.
"""))

# Code: Setup
nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

# Cấu hình đồ thị
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (8, 6)

# Đảm bảo thư mục graphics tồn tại
os.makedirs('../model_report/graphics', exist_ok=True)
"""))

# Markdown: Text Data
nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Dữ liệu Văn bản (Text)
Đọc dữ liệu từ `spam_labeled_text.csv` và thống kê phân bố nhãn Cảm xúc (Sentiment) cùng với nhãn Rác (Spam).
"""))

# Code: Text Data EDA
nb.cells.append(nbf.v4.new_code_cell("""# Đọc dữ liệu
text_df = pd.read_csv('../data/processed/spam_labeled_text.csv')
print(f"Tổng số mẫu văn bản: {len(text_df)}")

# Phân bố nhãn Cảm xúc (Sentiment)
sentiment_counts = text_df['sentiment_label'].value_counts()
print("\\nPhân bố nhãn Cảm xúc:")
print(sentiment_counts)

# Vẽ biểu đồ phân bố Sentiment
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Phân bố nhãn Cảm xúc (Sentiment)', fontsize=14)
plt.xlabel('Nhãn', fontsize=12)
plt.ylabel('Số lượng mẫu', fontsize=12)
for i, v in enumerate(sentiment_counts.values):
    plt.text(i, v + 100, str(v), ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('../model_report/graphics/sentiment_distribution.png', dpi=300)
plt.show()

# Phân bố nhãn Spam
spam_counts = text_df['is_spam'].value_counts()
spam_counts.index = ['Non-spam (0)', 'Spam (1)']
print("\\nPhân bố nhãn Spam:")
print(spam_counts)

# Vẽ biểu đồ phân bố Spam
plt.figure(figsize=(8, 6))
sns.barplot(x=spam_counts.index, y=spam_counts.values, palette='Set2')
plt.title('Phân bố nhãn Spam', fontsize=14)
plt.xlabel('Nhãn', fontsize=12)
plt.ylabel('Số lượng mẫu', fontsize=12)
for i, v in enumerate(spam_counts.values):
    plt.text(i, v + 100, str(v), ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('../model_report/graphics/spam_distribution.png', dpi=300)
plt.show()
"""))

# Markdown: Image Data
nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Dữ liệu Hình ảnh (Image)
Phân tích phân bố các lớp ảnh trong thư mục `image_labeling/data/labeled`.
Lưu ý: Dữ liệu này được giữ ở định dạng gốc. Các phép biến đổi (Resize, Normalize) sẽ được thực hiện trực tiếp trong lúc huấn luyện (on-the-fly transforms) để tiết kiệm dung lượng lưu trữ và linh hoạt hơn.
"""))

# Code: Image Data EDA
nb.cells.append(nbf.v4.new_code_cell("""image_dir = '../image_labeling/data/labeled'
classes = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]

image_counts = {}
for c in classes:
    image_counts[c] = len(os.listdir(os.path.join(image_dir, c)))

image_df = pd.DataFrame(list(image_counts.items()), columns=['Class', 'Count'])
image_df = image_df.sort_values(by='Count', ascending=False)
print("Phân bố hình ảnh:")
print(image_df.to_string(index=False))

# Vẽ biểu đồ phân bố Image
plt.figure(figsize=(10, 6))
sns.barplot(data=image_df, x='Class', y='Count', palette='magma')
plt.title('Phân bố nhãn Hình ảnh', fontsize=14)
plt.xlabel('Lớp (Class)', fontsize=12)
plt.ylabel('Số lượng ảnh', fontsize=12)
for i, v in enumerate(image_df['Count']):
    plt.text(i, v + 200, str(v), ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('../model_report/graphics/image_distribution.png', dpi=300)
plt.show()
"""))

# Markdown: Data Split
nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Phân chia tập dữ liệu (Train/Validation/Test)
Để đánh giá mô hình khách quan và tinh chỉnh siêu tham số, dữ liệu được chia theo tỷ lệ **80% Train, 10% Validation, 10% Test**.
Đặc biệt, do cả dữ liệu Text và Image đều bị mất cân bằng lớp khá lớn (như Tích cực áp đảo Tiêu cực, Intact áp đảo Damaged), kỹ thuật **Stratified Split** được sử dụng. Kỹ thuật này đảm bảo tỷ lệ các nhãn trong từng tập con (Train, Val, Test) tương đương với tỷ lệ nhãn trong tập dữ liệu gốc.
Dưới đây là mô phỏng quá trình chia trên dữ liệu Text (Sentiment) để lấy bảng số liệu cho báo cáo.
"""))

# Code: Data Split Text
nb.cells.append(nbf.v4.new_code_cell("""# Mô phỏng chia tập dữ liệu Text (Sentiment)
X = text_df.index.values # Lấy index làm đại diện
y = text_df['sentiment_label'].values

# Chia lần 1: 80% Train, 20% Temp (Val + Test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Chia lần 2: 10% Val, 10% Test (từ 20% Temp)
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

# Markdown: Preprocessing Summary
nb.cells.append(nbf.v4.new_markdown_cell("""## 4. Tiền xử lý và Tăng cường dữ liệu
### 4.1. Dữ liệu Văn bản
Quá trình tiền xử lý văn bản bao gồm các bước:
1. Chuẩn hóa chuỗi (Lowercase, xóa khoảng trắng thừa).
2. Dịch Teencode và từ viết tắt sang tiếng Việt chuẩn.
3. Chuẩn hóa dấu câu (loại bỏ hoặc tách dấu câu).
4. Tokenization (Word Level hoặc Subword Level) để đưa vào mô hình học máy.
"""))

# Code: Preprocessing Example
nb.cells.append(nbf.v4.new_code_cell("""# Hiển thị ví dụ trước và sau khi làm sạch
sample = text_df[['text', 'cleaned_text', 'tokens_word']].dropna().head(3)
pd.set_option('display.max_colwidth', None)
print("Ví dụ Text trước và sau khi tiền xử lý:")
display(sample)
"""))

# Markdown: Image Preprocessing Summary
nb.cells.append(nbf.v4.new_markdown_cell("""### 4.2. Dữ liệu Hình ảnh
Thay vì tiền xử lý toàn bộ ảnh và lưu ra ổ cứng, quá trình này được tích hợp trực tiếp vào **PyTorch DataLoader** (on-the-fly transforms):
1. **Resize**: Đưa tất cả ảnh về kích thước cố định (ví dụ: `224x224`) phù hợp với kiến trúc mô hình (ResNet, CNN).
2. **Normalize**: Chuẩn hóa giá trị pixel ảnh (thường sử dụng mean và std của ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
3. **Data Augmentation (Tăng cường dữ liệu)**: Để xử lý mất cân bằng lớp (ví dụ lớp `damaged` hoặc `wrong_item` ít dữ liệu), các phép biến đổi được áp dụng ngẫu nhiên trong lúc train, bao gồm:
   - Random Horizontal/Vertical Flip (Lật ảnh)
   - Random Rotation (Xoay ảnh)
   - Color Jitter (Điều chỉnh độ sáng, độ tương phản, độ bão hòa)
   - *Thư viện thường dùng: `albumentations` hoặc `torchvision.transforms`.*
"""))

with open('e:/Nhập môn học máy/Project/ecommerce-review-analytics/notebooks/05_data_overview_for_report.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Đã tạo notebook notebooks/05_data_overview_for_report.ipynb thành công!")
