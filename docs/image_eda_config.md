# Image EDA Notebook — Configuration & Guide

> This document describes the configuration, data paths, and parameters used in `image_eda.ipynb`.
> Update the constants below before running if your directory layout differs.

---

## 1. Data Paths

| Variable | Default Value | Description |
|---|---|---|
| `PROJECT_ROOT` | `..` (relative to `notebooks/`) | Root of the `ecommerce-review-analytics` repo |
| `LABELED_DIR` | `{PROJECT_ROOT}/image_labeling/data/labeled` | Folder containing labeled subfolders |
| `RAW_MEDIA_DIR` | `{PROJECT_ROOT}/image_labeling/data/raw_media` | Downloaded raw images and videos |
| `FRAMES_DIR` | `{PROJECT_ROOT}/image_labeling/data/frames` | Extracted video frames |
| `MANIFEST_CSV` | `{PROJECT_ROOT}/image_labeling/data/manifests/images.csv` | Image manifest with metadata |
| `OUTPUT_DIR` | `{PROJECT_ROOT}/notebooks/outputs` | Directory for saved plots and reports |

### Label Folder Structure

```
image_labeling/data/labeled/
├── intact/        # Product appears correct and undamaged
├── damaged/       # Product shows visible damage
├── irrelevant/    # Image is unrelated to product condition
└── wrong_item/    # Wrong product was delivered
```

---

## 2. Sampling & Reproducibility

| Parameter | Value | Description |
|---|---|---|
| `SAMPLE_SIZE` | `800` | Number of images sampled for heavy analyses (pHash, PCA, t-SNE) |
| `RANDOM_SEED` | `42` | Global seed for reproducibility |
| `STRATIFIED` | `True` | Sampling is stratified by class label to preserve class proportions |

> **Note**: Some analyses (class imbalance counting, full pHash scan) use the **entire dataset**.
> The sample is used for computationally expensive tasks like PCA, t-SNE, and ablation studies.

---

## 3. Image Processing Parameters

### 3.1 Resize

| Parameter | Value | Description |
|---|---|---|
| `WORKING_SIZE` | `(128, 128)` | Default working resolution for most analyses |
| `RESIZE_TARGETS` | `[32, 64, 128]` | Sizes used in the resize quality analysis |
| `INTERPOLATION` | `cv2.INTER_AREA` | Interpolation method for downscaling |

### 3.2 Color Spaces

| Color Space | OpenCV Code | Channels | Notes |
|---|---|---|---|
| RGB | (default after `cvtColor(BGR→RGB)`) | 3 | Standard color space |
| Grayscale | `cv2.COLOR_RGB2GRAY` | 1 | Single channel intensity |
| HSV | `cv2.COLOR_RGB2HSV` | 3 | Hue-Saturation-Value |
| LAB | `cv2.COLOR_RGB2LAB` | 3 | Perceptually uniform |

### 3.3 Normalization Methods

| Method | Formula | Range |
|---|---|---|
| MinMax [0, 1] | `(x - min) / (max - min)` | [0, 1] |
| MinMax [-1, 1] | `2 * (x - min) / (max - min) - 1` | [-1, 1] |
| Z-score (global) | `(x - μ_global) / σ_global` | unbounded |
| Z-score (per-channel) | `(x_c - μ_c) / σ_c` for each channel c | unbounded |

### 3.4 Data Augmentation Pipeline

| Transform | Library | Parameters |
|---|---|---|
| Horizontal Flip | `albumentations` | p=0.5 |
| Vertical Flip | `albumentations` | p=0.5 |
| Rotation | `albumentations` | limit=30°, p=0.5 |
| Random Crop + Resize | `albumentations` | scale=(0.7, 1.0), p=0.5 |
| Gaussian Noise | `albumentations` | var_limit=(10, 50), p=0.5 |
| Brightness/Contrast | `albumentations` | brightness=0.2, contrast=0.2, p=0.5 |

### 3.5 Edge Detection

| Detector | Hyperparameter Set 1 | Hyperparameter Set 2 |
|---|---|---|
| Sobel | ksize=3 | ksize=5 |
| Prewitt | Custom 3×3 kernel | Custom 5×5 kernel |
| Canny | low=50, high=150 | low=100, high=200 |

---

## 4. Analysis Parameters

| Parameter | Value | Description |
|---|---|---|
| `PCA_COMPONENTS` | `50` | Number of PCA components for explained variance analysis |
| `TSNE_PERPLEXITY` | `30` | t-SNE perplexity |
| `TSNE_N_ITER` | `1000` | t-SNE iterations |
| `KNN_K` | `5` | k for k-NN classifier in ablation studies |
| `TEST_SPLIT` | `0.2` | Train/test split ratio for ablation |
| `PHASH_THRESHOLD` | `10` | Hamming distance threshold for near-duplicate detection |
| `CLASS_IMBALANCE_RATIO` | `3.0` | Threshold ratio for flagging class imbalance |
| `KS_ALPHA` | `0.05` | Significance level for Kolmogorov–Smirnov test |
| `ANOVA_ALPHA` | `0.05` | Significance level for one-way ANOVA |

---

## 5. Required Libraries

```
opencv-python>=4.8
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
scikit-image>=0.21
scipy>=1.11
imagehash>=4.3
Pillow>=10.0
albumentations>=1.3
tqdm>=4.65
```

Install all at once:

```bash
pip install opencv-python numpy pandas matplotlib seaborn scikit-learn scikit-image scipy imagehash Pillow albumentations tqdm
```

---

## 6. Output Artifacts

The notebook saves the following outputs to `notebooks/outputs/`:

| File | Description |
|---|---|
| `class_distribution.png` | Bar chart of class counts |
| `pixel_distribution.png` | Per-channel pixel histograms |
| `duplicate_report.csv` | Detected near-duplicate pairs |
| `brightness_contrast_boxplot.png` | Boxplot of brightness/contrast by class |
| `ssim_curve.png` | SSIM vs resize dimension |
| `color_space_pca.png` | Explained variance by color space |
| `augmentation_tsne.png` | t-SNE before/after augmentation |
| `pca_scree_plot.png` | Scree plot with variance thresholds |
| `pca_2d.png` / `tsne_2d.png` | 2D projections colored by class |
| `edge_density_boxplot.png` | Edge density distributions by class |
| `ablation_results.csv` | Consolidated ablation study results |

---

## 7. Notebook Conventions

| Rule | Details |
|---|---|
| **Headings** | All section titles in English |
| **Code output** | All prints, plot titles, axis labels in English |
| **Markdown commentary** | Explanations, observations, and conclusions written in Vietnamese |
| **Random seed** | Set once at the top; all stochastic operations use this seed |
