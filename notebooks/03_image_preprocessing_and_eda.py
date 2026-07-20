#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA) for Unstructured Image Data
# 
# This notebook performs a comprehensive descriptive statistical analysis (EDA) and an ablation study to evaluate the impact of preprocessing methods on unstructured image data, based on the configuration defined in `IMAGE_EDA_CONFIG.md`.

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import image_eda_utils as utils

pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## **1. Configuration & Initialization**
# Set up environment variables and directories for storing outputs (artifacts).

# In[2]:


PROJECT_ROOT = '..'
LABELED_DIR = os.path.join(PROJECT_ROOT, 'image_labeling', 'data', 'labeled')
OUTPUT_DIR = 'outputs'

os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_SIZE = 800
RANDOM_SEED = 42
WORKING_SIZE = (128, 128)
KNN_K = 5


# ## **2. Load Data**
# Scan directory to get list of image files. Perform stratified sampling for heavy processing, while counting and perceptual hash (pHash) scans run on the entire dataset.

# In[3]:


df_all = utils.load_image_paths_and_labels(LABELED_DIR)
print(f"Total images found: {len(df_all)}")

df_sampled = utils.stratified_sample(df_all, SAMPLE_SIZE, RANDOM_SEED, stratified=True)
print(f"Sampled images for heavy processing: {len(df_sampled)}")

if len(df_sampled) > 0:
    images, labels = utils.load_images(df_sampled, target_size=WORKING_SIZE)
else:
    images, labels = np.array([]), np.array([])
    print("No images to load. Ensure LABELED_DIR is correctly populated.")


# ## **Part 1: Descriptive Statistical Analysis (Basic EDA)**
# 
# ### **1.1 Pixel Value Distribution**
# Compute and visualize the pixel intensity value distribution across the entire dataset (histogram, KDE) for each color channel (Red, Green, Blue).

# In[4]:


if len(images) > 0:
    utils.plot_pixel_distributions(images, labels, OUTPUT_DIR)
    print(f"Saved pixel distribution plot to {OUTPUT_DIR}/pixel_distribution.png")


# In[6]:


res_df = utils.analyze_resolution_distribution(
    df_all, OUTPUT_DIR, base_dir=LABELED_DIR
)


# ### **1.2 Class Imbalance**
# Calculate the ratio of each class and check if any class exceeds a 3x ratio compared to the minority class (a sign of severe imbalance).

# In[7]:


class_counts, is_imbalanced = utils.analyze_class_imbalance(df_all, 3.0, OUTPUT_DIR)
print("Class Counts:")
print(class_counts)
print(f"\nIs dataset imbalanced (>3x): {is_imbalanced}")


# ### **1.3 Duplicate or nearly duplicate image detection (pHash)**
# Scan the entire dataset, compute the perceptual hash (pHash). Report duplication rate with a Hamming distance threshold of 10.

# In[8]:


dup_rate, dup_df = utils.detect_duplicates_phash(df_all, threshold=10, output_dir=OUTPUT_DIR)
print(f"Duplication Rate: {dup_rate:.2%}")
print(f"Found {len(dup_df)} duplicate pairs. Report saved to {OUTPUT_DIR}/duplicate_report.csv")
dup_df_analyzed = utils.analyze_duplicate_report(dup_df, OUTPUT_DIR)


# ### **1.4 Analysis of overall contrast and brightness**
# Convert images to Grayscale to compute Mean Intensity and Standard Deviation, representing brightness and contrast, presented via per-class boxplots.

# In[9]:


if len(images) > 0:
    brightness_contrast_stats = utils.analyze_brightness_contrast(images, labels, OUTPUT_DIR)
    print("Mean Intensity & Contrast per class:")
    display(brightness_contrast_stats)


# In[10]:


outlier_df, abnormal_df = utils.detect_brightness_contrast_outliers(
    images, labels, df_sampled, OUTPUT_DIR
)


# ## **Part 2: Assessing the Impact of Pre-treatment Techniques (Ablation Study)**
# Compare basic classification results (using k-NN) before and after applying each preprocessing technique.

# ### **2.1 Change the size and quality of the image**
# Resize to 32x32, 64x64, 128x128. For each size, compute Structural Similarity Index (SSIM) and PSNR compared to the original image to quantify information loss.

# In[11]:


if len(images) > 0:
    resize_results = utils.analyze_resize_quality(images, labels, [32, 64, 128], OUTPUT_DIR, RANDOM_SEED, KNN_K)
    display(resize_results)


# ### **2.2 Color Space Conversion**
# Convert to RGB, Grayscale, HSV, and LAB. Measure Explained Variance using PCA (k=50) and classification accuracy.

# In[12]:


if len(images) > 0:
    color_space_results = utils.analyze_color_spaces(images, labels, k_components=50, output_dir=OUTPUT_DIR, knn_k=KNN_K)
    display(color_space_results)


# ### **2.3 Normalization**
# Apply Min-Max [0,1], Min-Max [-1,1], Z-score overall, and channel-wise Z-score. Test distribution difference before and after normalization using Kolmogorov-Smirnov (KS test).

# In[13]:


if len(images) > 0:
    norm_results, ks_p_value = utils.analyze_normalization(images, labels, knn_k=KNN_K)
    print(f"KS Test (Original vs Z-score Global) p-value: {ks_p_value:.4e}")
    display(norm_results)


# ### **2.4 Data Augmentation**
# Use Albumentations to apply a pipeline of 5 transformations. The impact is visually evaluated via t-SNE (original vs augmented set).

# In[14]:


if len(images) > 0:
    acc_orig, acc_aug = utils.apply_augmentation_pipeline(images, labels, OUTPUT_DIR, knn_k=KNN_K)
    print(f"Accuracy on Original dataset: {acc_orig:.4f}")
    print(f"Accuracy on Augmented dataset: {acc_aug:.4f}")


# ### **2.5 PCA analysis on image feature space**
# Reduce feature space dimensionality using PCA. Determine components needed for 90%, 95%, and 99% explained variance thresholds (Scree plot). Visualize 2D PCA and t-SNE to assess class separation.

# In[15]:


if len(images) > 0:
    pca_components = utils.perform_pca_analysis(images, labels, OUTPUT_DIR)
    print("Components needed for explained variance thresholds:")
    print(pca_components)


# ### **2.6 Edge Detection and Local Feature Analysis**
# Use Sobel, Prewitt, and Canny filters to extract edges. Compute Edge Density and run a One-way ANOVA to check if this variable distinguishes the classes.

# In[16]:


if len(images) > 0:
    edge_density_means, anova_res, edge_acc = utils.analyze_edge_detection(images, labels, OUTPUT_DIR, knn_k=KNN_K)
    print("Edge Density Means by Class:")
    display(edge_density_means)

    print("\nANOVA Results (Testing difference between classes):")
    for method, res in anova_res.items():
        print(f"{method}: F-stat = {res['f_stat']:.4f}, p-value = {res['p_value']:.4e}")

    print("\nClassification Accuracy using ONLY Edge Densities (1D feature):")
    for method, acc in edge_acc.items():
        print(f"{method}: {acc:.4f}")


# ## **3. Summary and Discussion**
# 1. **Image Resizing**: (Based on the SSIM curve, noting the quality degradation and justifying a suitable size).
# 2. **Color Space**: (Confirm if converting to HSV or LAB improves class separability compared to RGB, using PCA and k-NN).
# 3. **Normalization**: (KS test helps confirm the statistical significance of shifting features to the same distribution).
# 4. **Augmentation**: (t-SNE shows the diversification of samples when noise and rotation are introduced, helping to avoid overfitting).
# 5. **PCA Components**: (Helps determine how many dimensions to retain to balance performance and memory for the subsequent machine learning model).
# 6. **Edge Density**: (ANOVA p-value < 0.05 indicates that edge detail and texture are useful features for defect detection).
