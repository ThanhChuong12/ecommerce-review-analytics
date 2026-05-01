import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import imagehash
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import albumentations as A
from scipy.stats import kstest, f_oneway
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading & Sampling
def load_image_paths_and_labels(labeled_dir):
    data = []
    if not os.path.exists(labeled_dir):
        return pd.DataFrame(columns=['filepath', 'label'])
        
    for label in os.listdir(labeled_dir):
        label_dir = os.path.join(labeled_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    data.append({
                        'filepath': os.path.join(label_dir, file),
                        'label': label
                    })
    return pd.DataFrame(data, columns=['filepath', 'label'])

def stratified_sample(df, sample_size, random_seed, stratified=True):
    if len(df) <= sample_size:
        return df
    if stratified:
        # Avoid ValueError if a group is smaller than sample_size / num_groups
        # Calculate exactly how many per group
        n_classes = df['label'].nunique()
        per_class = max(1, sample_size // n_classes)
        return df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), per_class), random_state=random_seed)
        )
    else:
        return df.sample(sample_size, random_state=random_seed)

def load_images(df, target_size=(128, 128), color_space=cv2.COLOR_BGR2RGB):
    images = []
    labels = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Images"):
        img = cv2.imread(row['filepath'])
        if img is not None:
            img = cv2.cvtColor(img, color_space)
            if target_size:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            images.append(img)
            labels.append(row['label'])
    return np.array(images), np.array(labels)

# 2. Basic EDA
def plot_pixel_distributions(images, labels, output_dir):
    plt.figure(figsize=(15, 5))
    colors = ['r', 'g', 'b']
    for i, color in enumerate(colors):
        plt.subplot(1, 3, i+1)
        channel_data = images[:, :, :, i].flatten()
        # Sample to avoid memory issues for plotting
        sample_size = min(len(channel_data), 100000)
        channel_data = np.random.choice(channel_data, sample_size, replace=False)
        sns.histplot(channel_data, color=color, kde=True, bins=50)
        plt.title(f'{color.upper()} Channel Pixel Distribution')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pixel_distribution.png'))
    plt.close()

def analyze_class_imbalance(df, imbalance_ratio, output_dir):
    class_counts = df['label'].value_counts()
    if len(class_counts) == 0:
        return class_counts, False
    min_class = class_counts.min()
    max_class = class_counts.max()
    is_imbalanced = (max_class / min_class) > imbalance_ratio if min_class > 0 else True
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.title('Class Distribution')
    plt.xlabel('Class Label')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    
    return class_counts, is_imbalanced

def detect_duplicates_phash(df, threshold, output_dir):
    hashes = {}
    duplicates = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing pHash"):
        try:
            img = Image.open(row['filepath'])
            h = imagehash.phash(img)
            hashes[row['filepath']] = h
        except Exception as e:
            continue
        
    paths = list(hashes.keys())
    for i in tqdm(range(len(paths)), desc="Finding Duplicates"):
        for j in range(i + 1, len(paths)):
            if hashes[paths[i]] - hashes[paths[j]] <= threshold:
                duplicates.append({
                    'img1': paths[i],
                    'img2': paths[j],
                    'distance': hashes[paths[i]] - hashes[paths[j]]
                })
                
    dup_df = pd.DataFrame(duplicates)
    if not dup_df.empty:
        dup_df.to_csv(os.path.join(output_dir, 'duplicate_report.csv'), index=False)
    
    total_pairs = (len(paths) * (len(paths)-1) / 2) if len(paths) > 1 else 1
    dup_rate = len(duplicates) / total_pairs
    return dup_rate, dup_df

def analyze_brightness_contrast(images, labels, output_dir):
    metrics = []
    for img, label in zip(images, labels):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        metrics.append({
            'label': label,
            'brightness (mean)': np.mean(gray),
            'contrast (std)': np.std(gray)
        })
    metrics_df = pd.DataFrame(metrics)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='label', y='brightness (mean)', data=metrics_df, palette="Set2")
    plt.title('Brightness by Class')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='label', y='contrast (std)', data=metrics_df, palette="Set2")
    plt.title('Contrast by Class')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'brightness_contrast_boxplot.png'))
    plt.close()
    return metrics_df.groupby('label').mean()

# 3. Preprocessing & Ablation
def evaluate_ablation(features, labels, model_type='knn', k=5, test_split=0.2, random_seed=42):
    if len(np.unique(labels)) < 2:
        return 0.0 # Cannot classify single class
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_split, random_state=random_seed, stratify=labels)
    
    if model_type == 'knn':
        clf = KNeighborsClassifier(n_neighbors=k)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=random_seed)
        
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

def analyze_resize_quality(original_images, labels, resize_targets, output_dir, random_seed=42, knn_k=5):
    results = []
    # Take a small subset for structural similarity analysis
    subset_size = min(100, len(original_images))
    indices = np.random.RandomState(random_seed).choice(len(original_images), subset_size, replace=False)
    sub_images = [original_images[i] for i in indices]
    
    avg_ssims = []
    avg_psnrs = []
    for size in resize_targets:
        ssims, psnrs = [], []
        for img in tqdm(sub_images, desc=f"Analyzing Resize {size}x{size}"):
            resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            resized_back = cv2.resize(resized, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            # Using win_size dynamically to avoid errors on small images
            win_size = min(7, img.shape[0], img.shape[1])
            if win_size % 2 == 0: win_size -= 1
            if win_size < 3: win_size = 3
            
            try:
                s = ssim(img, resized_back, channel_axis=-1, data_range=255, win_size=win_size)
                p = psnr(img, resized_back, data_range=255)
            except ValueError:
                s, p = 0.0, 0.0
            ssims.append(s)
            psnrs.append(p)
            
        avg_ssims.append(np.mean(ssims))
        avg_psnrs.append(np.mean(psnrs))
        
        # Ablation study
        features = []
        for img in original_images:
             res = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
             features.append(res.flatten())
        acc = evaluate_ablation(np.array(features), labels, k=knn_k)
        results.append({'size': size, 'ssim': np.mean(ssims), 'psnr': np.mean(psnrs), 'accuracy': acc})

    plt.figure(figsize=(8, 5))
    plt.plot(resize_targets, avg_ssims, marker='o', linewidth=2, color='b', label='SSIM')
    plt.title('SSIM vs Resize Dimension')
    plt.xlabel('Size (pixels)')
    plt.ylabel('SSIM Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'ssim_curve.png'))
    plt.close()
    
    return pd.DataFrame(results)

def analyze_color_spaces(images, labels, k_components=50, output_dir=None, knn_k=5):
    spaces = {
        'RGB': images,
        'Grayscale': np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images]),
        'HSV': np.array([cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in images]),
        'LAB': np.array([cv2.cvtColor(img, cv2.COLOR_RGB2LAB) for img in images])
    }
    
    results = []
    variances = {}
    for name, space_imgs in tqdm(spaces.items(), desc="Analyzing Color Spaces"):
        features = space_imgs.reshape((len(space_imgs), -1))
        
        n_comp = min(k_components, features.shape[0], features.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(features)
        variances[name] = np.sum(pca.explained_variance_ratio_)
        
        acc = evaluate_ablation(features, labels, k=knn_k)
        results.append({'color_space': name, 'explained_variance': variances[name], 'accuracy': acc})

    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(variances.keys()), y=list(variances.values()), palette="pastel")
    plt.title(f'Explained Variance by Color Space (PCA k={k_components})')
    plt.ylabel('Cumulative Explained Variance Ratio')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'color_space_pca.png'))
    plt.close()
    
    return pd.DataFrame(results)

def analyze_normalization(images, labels, knn_k=5):
    flat_images = images.reshape((len(images), -1)).astype(np.float32)
    results = []
    
    print("Analyzing Normalizations...")
    # 0. Original
    results.append({'method': 'Original', 'accuracy': evaluate_ablation(flat_images, labels, k=knn_k)})
    
    # 1. MinMax [0, 1]
    min_val, max_val = flat_images.min(), flat_images.max()
    norm_01 = (flat_images - min_val) / (max_val - min_val + 1e-8)
    results.append({'method': 'MinMax [0,1]', 'accuracy': evaluate_ablation(norm_01, labels, k=knn_k)})
    
    # 2. MinMax [-1, 1]
    norm_n11 = 2 * norm_01 - 1
    results.append({'method': 'MinMax [-1,1]', 'accuracy': evaluate_ablation(norm_n11, labels, k=knn_k)})
    
    # 3. Z-score (global)
    mean_val, std_val = flat_images.mean(), flat_images.std()
    norm_z_global = (flat_images - mean_val) / (std_val + 1e-8)
    results.append({'method': 'Z-score Global', 'accuracy': evaluate_ablation(norm_z_global, labels, k=knn_k)})
    
    # 4. Z-score (per-channel)
    norm_z_channel = np.zeros_like(images, dtype=np.float32)
    for i in range(3):
        c = images[:, :, :, i]
        norm_z_channel[:, :, :, i] = (c - c.mean()) / (c.std() + 1e-8)
    norm_z_channel_flat = norm_z_channel.reshape((len(images), -1))
    results.append({'method': 'Z-score Per-Channel', 'accuracy': evaluate_ablation(norm_z_channel_flat, labels, k=knn_k)})
    
    # KS-Test example between original and z-score global
    sample_size = min(1000, flat_images.size)
    sample_orig = np.random.choice(flat_images.flatten(), sample_size, replace=False)
    sample_norm = np.random.choice(norm_z_global.flatten(), sample_size, replace=False)
    statistic, p_value = kstest(sample_orig, sample_norm)
    
    return pd.DataFrame(results), p_value

def apply_augmentation_pipeline(images, labels, output_dir, knn_k=5):
    pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomResizedCrop(height=images.shape[1], width=images.shape[2], scale=(0.7, 1.0), p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
    ])
    
    aug_images = []
    for img in tqdm(images, desc="Augmenting Images"):
        aug_images.append(pipeline(image=img)['image'])
    aug_images = np.array(aug_images)
    
    # t-SNE Comparison
    sample_size = min(300, len(images))
    orig_features = images[:sample_size].reshape(sample_size, -1)
    aug_features = aug_images[:sample_size].reshape(sample_size, -1)
    
    combined_features = np.vstack([orig_features, aug_features])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(combined_features)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_results[:sample_size, 0], y=tsne_results[:sample_size, 1], label='Original', alpha=0.7, color='blue')
    sns.scatterplot(x=tsne_results[sample_size:, 0], y=tsne_results[sample_size:, 1], label='Augmented', alpha=0.7, color='red')
    plt.title('t-SNE: Original vs Augmented Features')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'augmentation_tsne.png'))
    plt.close()
    
    acc_orig = evaluate_ablation(images.reshape(len(images), -1), labels, k=knn_k)
    acc_aug = evaluate_ablation(aug_images.reshape(len(aug_images), -1), labels, k=knn_k)
    
    return acc_orig, acc_aug

def perform_pca_analysis(images, labels, output_dir):
    print("Performing PCA Analysis...")
    features = images.reshape((len(images), -1))
    n_comp = min(500, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_comp) 
    pca_features = pca.fit_transform(features)
    
    cum_variance = np.cumsum(pca.explained_variance_ratio_)
    n_90 = np.argmax(cum_variance >= 0.90) + 1 if np.any(cum_variance >= 0.90) else n_comp
    n_95 = np.argmax(cum_variance >= 0.95) + 1 if np.any(cum_variance >= 0.95) else n_comp
    n_99 = np.argmax(cum_variance >= 0.99) + 1 if np.any(cum_variance >= 0.99) else n_comp
    
    plt.figure(figsize=(8, 5))
    plt.plot(cum_variance, linewidth=2)
    plt.axhline(y=0.90, color='r', linestyle='--', label=f'90% ({n_90} components)')
    plt.axhline(y=0.95, color='g', linestyle='--', label=f'95% ({n_95} components)')
    plt.axhline(y=0.99, color='b', linestyle='--', label=f'99% ({n_99} components)')
    plt.title('PCA Scree Plot')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'pca_scree_plot.png'))
    plt.close()
    
    # 2D PCA Visualization
    plt.figure(figsize=(8, 6))
    df_pca = pd.DataFrame({'PC1': pca_features[:, 0], 'PC2': pca_features[:, 1], 'label': labels})
    sns.scatterplot(x='PC1', y='PC2', hue='label', data=df_pca, palette='tab10', alpha=0.7)
    plt.title('2D PCA Projection')
    plt.savefig(os.path.join(output_dir, 'pca_2d.png'))
    plt.close()
    
    # t-SNE Visualization on PCA features
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_features = tsne.fit_transform(pca_features[:, :min(50, n_comp)])
    df_tsne = pd.DataFrame({'Dim1': tsne_features[:, 0], 'Dim2': tsne_features[:, 1], 'label': labels})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Dim1', y='Dim2', hue='label', data=df_tsne, palette='tab10', alpha=0.7)
    plt.title('2D t-SNE Projection')
    plt.savefig(os.path.join(output_dir, 'tsne_2d.png'))
    plt.close()
    
    return {'n_90': n_90, 'n_95': n_95, 'n_99': n_99}

def analyze_edge_detection(images, labels, output_dir, knn_k=5):
    results = []
    edge_densities = {'Sobel': [], 'Prewitt': [], 'Canny': []}
    
    def calculate_density(edge_img):
        return np.sum(edge_img > 0) / (edge_img.size + 1e-8)
        
    for img in tqdm(images, desc="Detecting Edges"):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Sobel ksize=3
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.hypot(sobel_x, sobel_y)
        edge_densities['Sobel'].append(calculate_density(sobel))
        
        # Prewitt custom kernel
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float32)
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
        prewitt_x = cv2.filter2D(gray, cv2.CV_64F, kernelx)
        prewitt_y = cv2.filter2D(gray, cv2.CV_64F, kernely)
        prewitt = np.hypot(prewitt_x, prewitt_y)
        edge_densities['Prewitt'].append(calculate_density(prewitt))
        
        # Canny 100, 200
        canny = cv2.Canny(gray, 100, 200)
        edge_densities['Canny'].append(calculate_density(canny))
        
    df_density = pd.DataFrame({
        'Label': labels,
        'Sobel Density': edge_densities['Sobel'],
        'Prewitt Density': edge_densities['Prewitt'],
        'Canny Density': edge_densities['Canny']
    })
    
    # Melt for boxplot
    df_melt = pd.melt(df_density, id_vars=['Label'], value_vars=['Sobel Density', 'Prewitt Density', 'Canny Density'], var_name='Method', value_name='Density')
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Method', y='Density', hue='Label', data=df_melt)
    plt.title('Edge Density Distribution by Class')
    plt.savefig(os.path.join(output_dir, 'edge_density_boxplot.png'))
    plt.close()
    
    # ANOVA
    anova_results = {}
    for method in ['Sobel Density', 'Prewitt Density', 'Canny Density']:
        groups = [group['Density'].values for name, group in df_melt[df_melt['Method'] == method].groupby('Label')]
        if len(groups) > 1:
            f_stat, p_val = f_oneway(*groups)
            anova_results[method] = {'f_stat': f_stat, 'p_value': p_val}
            
    # Evaluate features using edge densities as 1D feature
    acc_sobel = evaluate_ablation(np.array(edge_densities['Sobel']).reshape(-1, 1), labels, k=knn_k)
    acc_prewitt = evaluate_ablation(np.array(edge_densities['Prewitt']).reshape(-1, 1), labels, k=knn_k)
    acc_canny = evaluate_ablation(np.array(edge_densities['Canny']).reshape(-1, 1), labels, k=knn_k)
    
    acc_results = {
        'Sobel': acc_sobel,
        'Prewitt': acc_prewitt,
        'Canny': acc_canny
    }
    
    return df_density.groupby('Label').mean(), anova_results, acc_results
