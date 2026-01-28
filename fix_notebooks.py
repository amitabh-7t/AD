#!/usr/bin/env python3
"""Fix all notebooks 03-13 with properly formatted code cells."""
import json

def nb(path, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(path, 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"✓ {path.split('/')[-1]}")

def m(text): return {"cell_type": "markdown", "metadata": {}, "source": text.split('\n') if isinstance(text, str) else text}
def c(code): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.split('\n') if isinstance(code, str) else code}

# 03: Preprocessing
nb("notebooks/03_preprocessing.ipynb", [
    m("# Component 3: Preprocessing & Data Pipelines\n\nDefine preprocessing variants, augmentation, and build tf.data pipelines"),
    c("""import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load manifests
train_df = pd.read_csv('../outputs/train_manifest.csv')
val_df = pd.read_csv('../outputs/val_manifest.csv')
test_df = pd.read_csv('../outputs/test_manifest.csv')

print(f'Train: {len(train_df)} images')
print(f'Val:   {len(val_df)} images')
print(f'Test:  {len(test_df)} images')
print(f'Classes: {len(train_df["class_label"].unique())}')"""),
    m("## 3.1 Preprocessing Functions"),
    c("""def preprocess_resize_rescale(filepath, label):
    \"\"\"Simple resize and rescale to [0,1].\"\"\"
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label

def preprocess_resnet(filepath, label):
    \"\"\"ResNet50-specific preprocessing.\"\"\"
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label

def preprocess_efficientnet(filepath, label):
    \"\"\"EfficientNet-specific preprocessing.\"\"\"
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, label

def preprocess_densenet(filepath, label):
    \"\"\"DenseNet-specific preprocessing.\"\"\"
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.densenet.preprocess_input(img)
    return img, label

print("✓ Preprocessing functions defined")"""),
    m("## 3.2 Data Augmentation"),
    c("""# Define augmentation pipeline
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2)
], name='augmentation')

print("✓ Augmentation pipeline created")
print("  - RandomFlip (horizontal)")
print("  - RandomRotation (±20%)")
print("  - RandomZoom (±20%)")
print("  - RandomContrast (±20%)")"""),
    m("## 3.3 Build tf.data Pipelines"),
    c("""def build_dataset(dataframe, preprocess_fn, augment=False, shuffle=True, batch_size=BATCH_SIZE):
    \"\"\"Build optimized tf.data pipeline.\"\"\"
    filepaths = dataframe['filepath'].values
    labels = dataframe['class_label'].values
    
    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    # Preprocessing
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Cache before augmentation
    ds = ds.cache()
    
    # Augmentation (only for training)
    if augment:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), 
                   num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle
    if shuffle:
        ds = ds.shuffle(buffer_size=1000, seed=SEED)
    
    # Batch and prefetch
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

print("✓ Dataset building function defined")"""),
    m("## 3.4 Create Datasets (Basic Preprocessing)"),
    c("""# Build datasets with simple resize+rescale
train_ds = build_dataset(train_df, preprocess_resize_rescale, augment=True, shuffle=True)
val_ds = build_dataset(val_df, preprocess_resize_rescale, augment=False, shuffle=False)
test_ds = build_dataset(test_df, preprocess_resize_rescale, augment=False, shuffle=False)

print("✓ Datasets created")
print(f"  Train: {len(train_df)} images, {len(train_ds)} batches")
print(f"  Val:   {len(val_df)} images, {len(val_ds)} batches")
print(f"  Test:  {len(test_df)} images, {len(test_ds)} batches")"""),
    m("## 3.5 Test Pipeline"),
    c("""# Test the pipeline
print("Testing pipeline with a sample batch...")
for images, labels in train_ds.take(1):
    print(f"\\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Image range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
    print(f"Labels: {labels.numpy()}")
    
    # Visualize a few samples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flatten()):
        if i < images.shape[0]:
            ax.imshow(images[i].numpy())
            ax.set_title(f'Label: {labels[i].numpy()}')
            ax.axis('off')
    plt.suptitle('Sample Batch from Training Pipeline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

print("\\n" + "="*60)
print("✅ PREPROCESSING COMPLETE")
print("="*60)
print("\\nDatasets are ready for model training!")
print("\\nAvailable preprocessing functions:")
print("  - preprocess_resize_rescale (for Baseline CNN)")
print("  - preprocess_resnet (for ResNet50)")
print("  - preprocess_efficientnet (for EfficientNetB0)")
print("  - preprocess_densenet (for DenseNet121)")"""),
])

# 04: Feature Visualization
nb("notebooks/04_feature_visualization.ipynb", [
    m("# Component 4: Feature Visualization\n\nExtract deep features and visualize using t-SNE, UMAP, and hierarchical clustering"),
    c("""import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from scipy.cluster.hierarchy import dendrogram, linkage
from tqdm.auto import tqdm
import os

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = '../outputs/features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("✓ Setup complete")"""),
    m("## 4.1 Load Data & Extract Features"),
    c("""# Load full dataset manifest
full_df = pd.read_csv('../outputs/dataset_manifest.csv')

# Sample up to 4000 images for efficiency
MAX_SAMPLES = min(4000, len(full_df))
sample_df = full_df.sample(n=MAX_SAMPLES, random_state=SEED)

print(f"Extracting features from {len(sample_df)} images...")
print(f"Class distribution in sample:")
print(sample_df['class_name'].value_counts())

# Load pretrained ResNet50 for feature extraction
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    input_shape=(224, 224, 3)
)

print(f"\\n✓ Loaded ResNet50 (feature dim: 2048)")"""),
    c("""# Extract features
features = []
labels = []
class_names = []

print("\\nExtracting features...")
for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    try:
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            row['filepath'], target_size=(224, 224)
        )
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = tf.keras.applications.resnet50.preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        
        # Extract features
        feat = base_model.predict(x, verbose=0)[0]
        
        features.append(feat)
        labels.append(row['class_label'])
        class_names.append(row['class_name'])
    except Exception as e:
        print(f"Error processing {row['filepath']}: {e}")

features = np.array(features)
labels = np.array(labels)

print(f"\\n✓ Extracted features: {features.shape}")

# Save features
np.save(f'{OUTPUT_DIR}/features_resnet50.npy', features)
np.save(f'{OUTPUT_DIR}/labels.npy', labels)
print(f"✓ Features saved to {OUTPUT_DIR}/")"""),
    m("## 4.2 t-SNE Visualization"),
    c("""# Apply t-SNE
print("\\nComputing t-SNE (this may take a few minutes)...")
tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
tsne_embeddings = tsne.fit_transform(features)

# Save embeddings
tsne_df = pd.DataFrame(tsne_embeddings, columns=['x', 'y'])
tsne_df['label'] = labels
tsne_df['class_name'] = class_names
tsne_df.to_csv(f'{OUTPUT_DIR}/tsne_embeddings.csv', index=False)

# Visualize
plt.figure(figsize=(12, 10))
unique_classes = sorted(set(class_names))
colors = sns.color_palette('husl', len(unique_classes))

for idx, class_name in enumerate(unique_classes):
    mask = np.array(class_names) == class_name
    plt.scatter(
        tsne_embeddings[mask, 0],
        tsne_embeddings[mask, 1],
        label=class_name,
        alpha=0.6,
        s=50,
        color=colors[idx]
    )

plt.title('t-SNE Visualization of MRI Features', fontsize=16, fontweight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tsne_plot.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ t-SNE visualization saved")"""),
    m("## 4.3 UMAP Visualization"),
    c("""# Apply UMAP
print("\\nComputing UMAP...")
reducer = umap.UMAP(random_state=SEED, n_neighbors=15, min_dist=0.1)
umap_embeddings = reducer.fit_transform(features)

# Save embeddings
umap_df = pd.DataFrame(umap_embeddings, columns=['x', 'y'])
umap_df['label'] = labels
umap_df['class_name'] = class_names
umap_df.to_csv(f'{OUTPUT_DIR}/umap_embeddings.csv', index=False)

# Visualize
plt.figure(figsize=(12, 10))

for idx, class_name in enumerate(unique_classes):
    mask = np.array(class_names) == class_name
    plt.scatter(
        umap_embeddings[mask, 0],
        umap_embeddings[mask, 1],
        label=class_name,
        alpha=0.6,
        s=50,
        color=colors[idx]
    )

plt.title('UMAP Visualization of MRI Features', fontsize=16, fontweight='bold')
plt.xlabel('UMAP Dimension 1', fontsize=12)
plt.ylabel('UMAP Dimension 2', fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/umap_plot.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ UMAP visualization saved")"""),
    m("## 4.4 Hierarchical Clustering Tree"),
    c("""# Sample subset for tree visualization (hierarchical clustering is O(n²))
TREE_SAMPLES = min(500, len(features))
sample_indices = np.random.choice(len(features), TREE_SAMPLES, replace=False)
sample_features = features[sample_indices]

print(f"\\nComputing hierarchical clustering on {TREE_SAMPLES} samples...")
Z = linkage(sample_features, method='ward')

# Visualize dendrogram
plt.figure(figsize=(15, 8))
dendrogram(Z, no_labels=True)
plt.title('Hierarchical Clustering of MRI Features', fontsize=16, fontweight='bold')
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tmap_plot.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Tree map saved")"""),
    m("## 4.5 Summary"),
    c("""print("\\n" + "="*60)
print("✅ FEATURE VISUALIZATION COMPLETE")
print("="*60)
print(f"\\nArtifacts saved to: {OUTPUT_DIR}/")
print(f"  - features_resnet50.npy ({features.shape})")
print(f"  - labels.npy ({labels.shape})")
print(f"  - tsne_embeddings.csv")
print(f"  - umap_embeddings.csv")
print(f"  - tsne_plot.png")
print(f"  - umap_plot.png")
print(f"  - tmap_plot.png")
print("\\nKey observations:")
print("  - Check if classes form distinct clusters")
print("  - Overlapping regions may indicate similar features")
print("  - t-SNE preserves local structure, UMAP preserves global")"""),
])

print("✅ Created notebooks 03 and 04")
