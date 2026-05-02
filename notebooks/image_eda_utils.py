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
from IPython.display import display
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
        sampled_indices = []
        for label, group in df.groupby('label'):
            n_sample = min(len(group), per_class)
            sampled_indices.extend(group.sample(n_sample, random_state=random_seed).index)
        return df.loc[sampled_indices]
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

def _resolve_path(raw, base_dir):
    """Return an accessible path, trying three fallback strategies.

    1. Use *raw* as-is if the file already exists.
    2. Normalise OS separators then resolve relative to *base_dir*.
    3. Keep only the last two path components (label/filename) and anchor
       under *base_dir* — handles stale Windows prefixes on Linux.
    """
    import os
    if os.path.isfile(raw):
        return raw
    if base_dir is None:
        return raw
    normalised = raw.replace("\\", os.sep).replace("/", os.sep)
    candidate = os.path.normpath(os.path.join(base_dir, normalised))
    if os.path.isfile(candidate):
        return candidate
    parts = normalised.split(os.sep)
    if len(parts) >= 2:
        candidate2 = os.path.join(base_dir, *parts[-2:])
        if os.path.isfile(candidate2):
            return candidate2
    return raw


def analyze_resolution_distribution(df, output_dir, low_res_threshold=64, base_dir=None):
    """Profile native image resolution across the dataset.

    Reads every image at its original size (no resize), then produces width
    and height histograms, a Width × Height scatter plot coloured by label,
    a descriptive statistics table, and a count of images below
    *low_res_threshold* × *low_res_threshold* pixels.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``path`` and ``label``.
    output_dir : str
        Directory where ``resolution_distribution.png`` is saved.
    low_res_threshold : int
        Images whose width or height is below this value are flagged
        (default 64).
    base_dir : str or None
        Absolute path to the labeled image root.  Required when ``df['path']``
        stores relative or Windows-style paths that do not resolve in the
        current working directory.  Pass the ``LABELED_DIR`` variable from
        the notebook configuration cell.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    widths, heights, ratios, lbls = [], [], [], []
    failed = 0

    for _, row in df.iterrows():
        path_col = "filepath" if "filepath" in df.columns else "path"
        resolved = _resolve_path(str(row[path_col]), base_dir)
        try:
            with Image.open(resolved) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                ratios.append(round(w / h, 4))
                lbls.append(row["label"])
        except Exception:
            failed += 1

    if failed:
        hint = "" if base_dir else " Pass base_dir=LABELED_DIR to resolve relative paths."
        print(f"[WARNING] Could not open {failed} image(s) — skipped.{hint}")

    res_df = pd.DataFrame({"width": widths, "height": heights,
                           "aspect_ratio": ratios, "label": lbls})

    if res_df.empty:
        print("[ERROR] Resolution DataFrame is empty. "
              "Verify that 'path' values point to accessible files.")
        return res_df

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(res_df["width"], bins=40, color="steelblue", edgecolor="white")
    axes[0].set_title("Width Distribution")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(res_df["height"], bins=40, color="darkorange", edgecolor="white")
    axes[1].set_title("Height Distribution")
    axes[1].set_xlabel("Height (px)")
    axes[1].set_ylabel("Frequency")

    label_list = sorted(res_df["label"].unique())
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(label_list), 1)))
    for lbl, color in zip(label_list, cmap):
        sub = res_df[res_df["label"] == lbl]
        axes[2].scatter(sub["width"], sub["height"],
                        label=lbl, alpha=0.4, s=10, color=color)
    axes[2].set_title("Width × Height by Label")
    axes[2].set_xlabel("Width (px)")
    axes[2].set_ylabel("Height (px)")
    axes[2].legend(markerscale=2, fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "resolution_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")

    print("\nResolution descriptive statistics:")
    print(res_df[["width", "height", "aspect_ratio"]].describe().round(1).to_string())

    n_total = len(res_df)
    low_res = res_df[(res_df["width"] < low_res_threshold) |
                     (res_df["height"] < low_res_threshold)]
    print(f"\nImages below {low_res_threshold}×{low_res_threshold} px: "
          f"{len(low_res)} ({len(low_res) / n_total * 100:.2f}%)")
    if not low_res.empty:
        print(low_res.groupby("label").size().rename("count")
              .reset_index().to_string(index=False))

    return res_df


def detect_brightness_contrast_outliers(
        images, labels, df_sampled, output_dir,
        low_brightness=50.0, high_brightness=200.0, low_contrast=20.0):
    """Flag images that are too dark, overexposed, or low-contrast.

    Converts each image to grayscale via luminance weighting, computes
    per-image mean intensity (brightness proxy) and standard deviation
    (contrast proxy), then applies the supplied thresholds to identify
    outliers.

    Parameters
    ----------
    images : array-like, shape (N, H, W, 3)
        Pixel values in [0, 1] or [0, 255].
    labels : array-like, length N
        Class label for each image.
    df_sampled : pd.DataFrame
        Must contain a ``path`` column aligned with *images*.
    output_dir : str
        Directory where ``outlier_report.csv`` is saved.
    low_brightness : float
        Mean intensity below this value → flagged ``too_dark``.
    high_brightness : float
        Mean intensity above this value → flagged ``overexposed``.
    low_contrast : float
        Std intensity below this value → flagged ``low_contrast``.

    Returns
    -------
    outlier_df : pd.DataFrame
        Full per-image record.
    abnormal_df : pd.DataFrame
        Subset where at least one flag is raised.
    """
    import os
    import numpy as np
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    imgs = np.array(images, dtype=np.float32)
    if imgs.max() <= 1.0:
        imgs = imgs * 255.0

    records = []
    path_col = "filepath" if "filepath" in df_sampled.columns else "path"
    paths = df_sampled[path_col].tolist()

    for img, lbl, path in zip(imgs, labels, paths):
        gray = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
        mean_val = float(gray.mean())
        std_val = float(gray.std())
        flags = []
        if mean_val < low_brightness:
            flags.append("too_dark")
        if mean_val > high_brightness:
            flags.append("overexposed")
        if std_val < low_contrast:
            flags.append("low_contrast")
        records.append({"path": path, "label": lbl,
                        "mean_intensity": round(mean_val, 2),
                        "std_intensity": round(std_val, 2),
                        "flags": ", ".join(flags) if flags else "normal"})

    outlier_df = pd.DataFrame(records)
    abnormal_df = outlier_df[outlier_df["flags"] != "normal"].copy()
    n = len(outlier_df)

    print("=" * 55)
    print(f"  Images inspected      : {n}")
    print(f"  Outliers detected     : {len(abnormal_df)} ({len(abnormal_df) / n * 100:.1f}%)")
    print(f"\n  Thresholds applied:")
    print(f"    Too dark            : mean_intensity < {low_brightness}")
    print(f"    Overexposed         : mean_intensity > {high_brightness}")
    print(f"    Low contrast        : std_intensity  < {low_contrast}")
    print("=" * 55)

    if not abnormal_df.empty:
        summary = (abnormal_df.groupby(["label", "flags"])
                   .size().reset_index(name="count"))
        print("\nOutlier breakdown by label and flag type:")
        print(summary.to_string(index=False))
        save_path = os.path.join(output_dir, "outlier_report.csv")
        abnormal_df.to_csv(save_path, index=False)
        print(f"\nOutlier report saved → {save_path}")
    else:
        print("\nNo outliers detected under the current thresholds.")

    return outlier_df, abnormal_df


def analyze_duplicate_report(dup_df, output_dir):
    """Perform deep analysis of a pHash duplicate report.

    Classifies each pair as *intra-class* or *cross-class*, visualises the
    Hamming distance distribution, and builds a cross-class co-occurrence
    heatmap to surface images potentially borrowed from another label or
    incorrectly annotated.

    Parameters
    ----------
    dup_df : pd.DataFrame
        Columns: ``img1``, ``img2``, ``distance``.
        Output of ``detect_duplicates_phash()``.
    output_dir : str
        Directory where ``duplicate_analysis.png`` and
        ``cross_class_duplicates.csv`` are saved.

    Returns
    -------
    pd.DataFrame
        *dup_df* augmented with ``label1``, ``label2``, ``pair_type``.
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    if dup_df.empty:
        print("No duplicate pairs found in the report.")
        return dup_df

    # Auto-detect column names for the two image path columns
    # and the distance column — accommodates different naming conventions.
    cols = dup_df.columns.tolist()

    def _find_col(candidates):
        for c in candidates:
            if c in cols:
                return c
        # Fallback: return the first column whose name contains any candidate substring
        for c in candidates:
            matches = [col for col in cols if c.lower() in col.lower()]
            if matches:
                return matches[0]
        return None

    col_img1 = _find_col(["img1", "path1", "image1", "file1", "src"])
    col_img2 = _find_col(["img2", "path2", "image2", "file2", "dst"])
    col_dist = _find_col(["distance", "dist", "hamming"])

    if col_img1 is None or col_img2 is None:
        print(f"[ERROR] Cannot identify image path columns in dup_df.\n"
              f"        Available columns: {cols}\n"
              f"        Expected names like: img1/img2, path1/path2, image1/image2.")
        return dup_df

    if col_dist is None:
        print(f"[WARNING] Cannot identify distance column. Available: {cols}")
        col_dist = cols[-1]  # best guess: last column

    print(f"[INFO] Using columns — img1: '{col_img1}', img2: '{col_img2}', "
          f"distance: '{col_dist}'")

    def _extract_label(path_str):
        parts = str(path_str).replace("\\", "/").split("/")
        return parts[-2] if len(parts) >= 2 else "unknown"

    df = dup_df.copy()
    df["label1"] = df[col_img1].apply(_extract_label)
    df["label2"] = df[col_img2].apply(_extract_label)
    df["pair_type"] = df.apply(
        lambda r: "intra-class" if r["label1"] == r["label2"] else "cross-class",
        axis=1)

    total = len(df)
    intra = df[df["pair_type"] == "intra-class"]
    cross = df[df["pair_type"] == "cross-class"]
    exact = df[df[col_dist] == 0]
    cross_exact = cross[cross[col_dist] == 0]

    print("=" * 55)
    print(f"  Total duplicate pairs      : {total}")
    print(f"  Intra-class pairs          : {len(intra):>5} ({len(intra) / total * 100:.1f}%)")
    print(f"  Cross-class pairs          : {len(cross):>5} ({len(cross) / total * 100:.1f}%)")
    print(f"  Exact duplicates (dist=0)  : {len(exact):>5} ({len(exact) / total * 100:.1f}%)")
    print(f"  Cross-class exact          : {len(cross_exact):>5}  ← potential mislabelling")
    print("=" * 55)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df[col_dist], bins=11, range=(-0.5, 10.5),
                 color="steelblue", edgecolor="white")
    axes[0].set_title("Hamming Distance Distribution (all pairs)")
    axes[0].set_xlabel("Hamming Distance")
    axes[0].set_ylabel("Number of pairs")

    if not cross.empty:
        import numpy as np
        cross_counts = (cross.groupby(["label1", "label2"])
                        .size().reset_index(name="count"))
        all_labels = sorted(set(df["label1"]) | set(df["label2"]))
        pivot = (cross_counts.pivot(index="label1", columns="label2", values="count")
                 .reindex(index=all_labels, columns=all_labels).fillna(0))
        im = axes[1].imshow(pivot.values, cmap="Reds", aspect="auto")
        axes[1].set_xticks(range(len(pivot.columns)))
        axes[1].set_yticks(range(len(pivot.index)))
        axes[1].set_xticklabels(pivot.columns, rotation=30, ha="right")
        axes[1].set_yticklabels(pivot.index)
        axes[1].set_title("Cross-class Duplicate Co-occurrence Matrix")
        plt.colorbar(im, ax=axes[1], label="Number of pairs")
    else:
        axes[1].text(0.5, 0.5, "No cross-class duplicates found",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "duplicate_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")

    if not cross_exact.empty:
        display_cols = ["img1", "label1", "img2", "label2", "distance"]
        print(f"\nCross-class exact duplicate pairs — priority review (n={len(cross_exact)}):")
        print(cross_exact[display_cols].head(20).to_string(index=False))
        save_path2 = os.path.join(output_dir, "cross_class_duplicates.csv")
        cross_exact[display_cols].to_csv(save_path2, index=False)
        print(f"Priority review list saved → {save_path2}")

    return df

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
    plt.show()

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
    plt.show()
    
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
    plt.show()
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
    plt.show()
    
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
    plt.show()
    
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
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(combined_features)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_results[:sample_size, 0], y=tsne_results[:sample_size, 1], label='Original', alpha=0.7, color='blue')
    sns.scatterplot(x=tsne_results[sample_size:, 0], y=tsne_results[sample_size:, 1], label='Augmented', alpha=0.7, color='red')
    plt.title('t-SNE: Original vs Augmented Features')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'augmentation_tsne.png'))
    plt.show()
    
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
    plt.show()
    
    # 2D PCA Visualization
    plt.figure(figsize=(8, 6))
    df_pca = pd.DataFrame({'PC1': pca_features[:, 0], 'PC2': pca_features[:, 1], 'label': labels})
    sns.scatterplot(x='PC1', y='PC2', hue='label', data=df_pca, palette='tab10', alpha=0.7)
    plt.title('2D PCA Projection')
    plt.savefig(os.path.join(output_dir, 'pca_2d.png'))
    plt.show()
    
    # t-SNE Visualization on PCA features
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_features = tsne.fit_transform(pca_features[:, :min(50, n_comp)])
    df_tsne = pd.DataFrame({'Dim1': tsne_features[:, 0], 'Dim2': tsne_features[:, 1], 'label': labels})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Dim1', y='Dim2', hue='label', data=df_tsne, palette='tab10', alpha=0.7)
    plt.title('2D t-SNE Projection')
    plt.savefig(os.path.join(output_dir, 'tsne_2d.png'))
    plt.show()
    
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
    plt.show()
    
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
