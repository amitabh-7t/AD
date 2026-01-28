# MRI Alzheimer's Classification - Notebook Pipeline

Complete end-to-end deep learning pipeline for Alzheimer's disease classification from MRI scans, implemented as **13 self-contained Jupyter notebooks**.

## 📊 Dataset

- **Normal**: 8,104 images (69%)
- **Very Mild**: 2,240 images (19%)  
- **Mild**: 896 images (8%)
- **Moderate**: 503 images (4%)

**Total**: 11,743 images  
**Split**: 70% train / 15% validation / 15% test (stratified)

## 📚 Notebook Pipeline

### Phase 1: Data Preparation
- **`01_dataset_ingestion.ipynb`** - Load images, validate integrity (SHA256), create stratified splits
- **`02_eda.ipynb`** - Sample grids, intensity histograms, duplicate detection
- **`03_preprocessing.ipynb`** - Define preprocessing variants, build tf.data pipelines
- **`04_feature_visualization.ipynb`** - Extract ResNet50 features, t-SNE, UMAP, tree maps

### Phase 2: Model Training (50 epochs each, NO EarlyStopping)
- **`05_train_baseline_cnn.ipynb`** - Custom 4-layer CNN architecture
- **`06_train_resnet50.ipynb`** - ResNet50 transfer learning (ImageNet weights)
- **`07_train_efficientnetb0.ipynb`** - EfficientNetB0 transfer learning
- **`08_train_densenet121.ipynb`** - DenseNet121 transfer learning

### Phase 3: Analysis & Evaluation
- **`09_evaluation.ipynb`** - Metrics, confusion matrices, ROC curves for all models
- **`10_gradcam.ipynb`** - Grad-CAM heatmaps showing model attention
- **`11_comparison.ipynb`** - Cross-model comparison (accuracy vs size vs speed)
- **`12_robustness.ipynb`** - Noise and intensity scaling tests
- **`13_final_report.ipynb`** - Comprehensive report generation

---

## 📖 Detailed Notebook Descriptions

### `01_dataset_ingestion.ipynb` - Dataset Loading & Validation
**Purpose**: Load and validate the complete MRI dataset

**What it does**:
- Scans the `data/` directory for all image files (11,743 total)
- Computes SHA256 hash for each image to verify integrity
- Extracts metadata (dimensions, file size, class label)
- Creates stratified train/val/test splits (70%/15%/15%)
- Generates dataset manifest CSV with all metadata
- Visualizes class distribution (bar chart + pie chart)

**Outputs**: `dataset_manifest.csv`, split manifests, class distribution plot

**Runtime**: ~5-10 minutes

---

### `02_eda.ipynb` - Exploratory Data Analysis
**Purpose**: Comprehensive visual and statistical analysis

**What it does**:
- Generates 8-image sample grids for each class
- Computes pixel intensity distributions per class
- Analyzes image size variability and identifies outliers
- Detects potential duplicates using perceptual hashing (imagehash library)
- Calculates global statistics (mean, std)
- Generates comprehensive EDA markdown report

**Outputs**: Sample grids, intensity histograms, duplicate report, EDA summary

**Runtime**: ~10-15 minutes

---

### `03_preprocessing.ipynb` - Data Pipeline Creation
**Purpose**: Define preprocessing functions and build optimized tf.data pipelines

**What it does**:
- Defines 4 preprocessing variants:
  - Resize + Rescale (for Baseline CNN)
  - ResNet50-specific preprocessing
  - EfficientNet-specific preprocessing
  - DenseNet-specific preprocessing
- Creates data augmentation layer (RandomFlip, RandomRotation, RandomZoom, RandomContrast)
- Builds optimized tf.data pipelines with caching and prefetching
- Tests pipeline with sample batch visualization

**Outputs**: Reusable preprocessing functions, configured datasets

**Runtime**: ~2-3 minutes

---

### `04_feature_visualization.ipynb` - Deep Feature Analysis
**Purpose**: Extract and visualize high-level features using pretrained CNN

**What it does**:
- Loads frozen ResNet50 (ImageNet weights) as feature extractor
- Extracts 2048-dimensional features from up to 4,000 images
- Applies t-SNE dimensionality reduction for 2D visualization
- Applies UMAP dimensionality reduction for 2D visualization
- Generates hierarchical clustering dendrogram (tree map)
- Saves feature vectors and embeddings as NPY/CSV

**Outputs**: Feature vectors, t-SNE plot, UMAP plot, tree map, embedding CSVs

**Runtime**: ~15-30 minutes (depending on sample size)

---

### `05_train_baseline_cnn.ipynb` - Custom CNN Training
**Purpose**: Train a custom 4-layer CNN from scratch

**Model Architecture**:
- Conv2D(32) → BN → MaxPool → Dropout(0.25)
- Conv2D(64) → BN → MaxPool → Dropout(0.25)
- Conv2D(128) → BN → MaxPool → Dropout(0.25)
- Conv2D(256) → GlobalAvgPool
- Dense(128) → Dropout(0.5) → Dense(4)

**Training Config**:
- 50 epochs (NO EarlyStopping)
- Adam optimizer (lr=1e-4)
- Class weights for imbalanced data
- ModelCheckpoint (saves best val_accuracy)
- ReduceLROnPlateau (factor=0.5, patience=3)

**Outputs**: Best model checkpoint, training history (JSON/CSV), training curves

**Runtime**: ~1-2 hours

---

### `06_train_resnet50.ipynb` - ResNet50 Transfer Learning
**Purpose**: Fine-tune ResNet50 pretrained on ImageNet

**Model Architecture**:
- ResNet50 base (frozen first 155/175 layers)
- GlobalAveragePooling2D
- Dense(128, relu) → Dropout(0.5) → Dense(4, softmax)

**Training Config**: Same as Baseline CNN (50 epochs, NO EarlyStopping)

**Outputs**: Best model checkpoint, training history, training curves

**Runtime**: ~1.5-2.5 hours

---

### `07_train_efficientnetb0.ipynb` - EfficientNet Transfer Learning
**Purpose**: Fine-tune EfficientNetB0 pretrained on ImageNet

**Model Architecture**:
- EfficientNetB0 base (frozen 90% of layers)
- GlobalAveragePooling2D
- Dense(128, relu) → Dropout(0.5) → Dense(4, softmax)

**Training Config**: Same as Baseline CNN (50 epochs, NO EarlyStopping)

**Outputs**: Best model checkpoint, training history, training curves

**Runtime**: ~1-2 hours

---

### `08_train_densenet121.ipynb` - DenseNet Transfer Learning
**Purpose**: Fine-tune DenseNet121 pretrained on ImageNet

**Model Architecture**:
- DenseNet121 base (frozen 90% of layers)
- GlobalAveragePooling2D
- Dense(128, relu) → Dropout(0.5) → Dense(4, softmax)

**Training Config**: Same as Baseline CNN (50 epochs, NO EarlyStopping)

**Outputs**: Best model checkpoint, training history, training curves

**Runtime**: ~1.5-2.5 hours

---

### `09_evaluation.ipynb` - Comprehensive Model Evaluation
**Purpose**: Evaluate all 4 trained models on the test set

**What it does**:
- Loads all 4 best model checkpoints
- Computes comprehensive metrics for each:
  - Accuracy, Balanced Accuracy
  - Precision, Recall, F1-score (per class)
  - Macro F1, Weighted F1
  - ROC AUC (one-vs-rest)
- Generates confusion matrix heatmaps
- Plots ROC curves with AUC values
- Saves predictions CSV with probabilities
- Creates comparative metrics table

**Outputs**: Predictions CSVs, confusion matrices, ROC curves, metrics summary

**Runtime**: ~10-15 minutes

---

### `10_gradcam.ipynb` - Model Explainability
**Purpose**: Visualize what image regions each model focuses on

**What it does**:
- Implements Grad-CAM (Gradient-weighted Class Activation Mapping)
- Automatically detects last convolutional layer for each model
- Generates heatmaps for 5 samples per class × 4 models = 80+ visualizations
- Creates side-by-side comparisons (original, heatmap, overlay)
- Includes both correct and misclassified examples

**Outputs**: Grad-CAM overlay images organized by model and class

**Runtime**: ~20-30 minutes

---

### `11_comparison.ipynb` - Cross-Model Analysis
**Purpose**: Compare all models across multiple dimensions

**What it does**:
- Aggregates evaluation metrics from all 4 models
- Measures model file sizes (MB)
- Benchmarks inference times (ms/image)
- Creates comparison table ranked by Macro F1
- Generates comparison bar charts (accuracy, F1, etc.)
- Documents trade-offs (accuracy vs size vs speed)
- Identifies best overall and per-minority-class models

**Outputs**: Comparison CSV, comparison charts, trade-off analysis

**Runtime**: ~5-10 minutes

---

### `12_robustness.ipynb` - Stability Testing
**Purpose**: Test model performance under adversarial conditions

**What it does**:
- Tests models with Gaussian noise (σ = 0.05, 0.1, 0.15, 0.2)
- Tests models with intensity scaling (factors: 0.8, 0.9, 1.1, 1.2)
- Measures performance degradation
- Identifies most robust models
- Generates robustness summary report

**Outputs**: Robustness test results, degradation metrics

**Runtime**: ~15-20 minutes

---

### `13_final_report.ipynb` - Report Generation
**Purpose**: Aggregate all results into a comprehensive markdown report

**What it does**:
- Loads results from all previous notebooks
- Aggregates metrics, best models, key findings
- Creates comprehensive FINAL_REPORT.md with:
  - Executive summary
  - Model performance comparison
  - Key visualizations (embedded)
  - Recommendations for improvement
  - Environment details for reproducibility

**Outputs**: `FINAL_REPORT.md`

**Runtime**: ~2-3 minutes

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/amitabhthakur/Workspace/Projects/ML/AD
pip install tensorflow-macos tensorflow-metal
pip install -r requirements.txt
```

### 2. Run Notebooks Sequentially

```bash
jupyter lab
```

Then execute notebooks in order (01 → 13).

**Each notebook**:
- Is self-contained with all necessary imports
- Saves outputs to `../outputs/[component_name]/`
- Can be re-run independently (after dependencies)

### 3. Execution Time

| Notebook | Estimated Time |
|----------|----------------|
| 01-04 | ~15-30 min total |
| 05-08 | ~1-2 hours each (50 epochs) |
| 09-13 | ~30-60 min total |

**Total**: ~5-10 hours for complete pipeline

## 📁 Output Structure

```
outputs/
├── dataset_manifest.csv
├── train/val/test_manifest.csv
├── class_distribution.png
├── eda/
│   ├── grid_*.png
│   ├── intensities.png
│   └── eda_report.md
├── features/
│   ├── features_resnet50.npy
│   ├── tsne_plot.png
│   ├── umap_plot.png
│   └── tmap_plot.png
├── models/
│   ├── baseline_cnn_best.h5
│   ├── resnet50_best.h5
│   ├── efficientnetb0_best.h5
│   └── densenet121_best.h5
├── training_history/
│   ├── *_history.json
│   ├── *_history.csv
│   └── *_curves.png
├── evaluation/
│   ├── *_predictions.csv
│   ├── *_confusion_matrix.png
│   └── *_roc_curves.png
├── gradcam/
│   └── [model]/
│       └── *.png
├── comparison.csv
└── FINAL_REPORT.md
```

## 🎯 Key Features

- **NO EarlyStopping** - All models train for full 50 epochs as specified
- **ModelCheckpoint** - Saves best model by val_accuracy
- **ReduceLROnPlateau** - Learning rate decay (factor=0.5, patience=3)
- **Class Weights** - Handle imbalanced dataset
- **Data Augmentation** - RandomFlip, RandomRotation, RandomZoom
- **Reproducibility** - Seed=42 throughout
- **SHA256 Validation** - Data integrity checking
- **Comprehensive Metrics** - Accuracy, precision, recall, F1, balanced accuracy, AUC
- **Explainability** - Grad-CAM visualizations
- **Robustness Testing** - Noise and scaling transformations

## 📝 Configuration

Edit notebook cells to customize:

```python
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50  # Training notebooks
```

## 💡 Tips

**Parallel Training**: Notebooks 05-08 can be run in parallel (4 separate kernels) to speed up training.

**GPU Memory**: If you encounter OOM errors, reduce `BATCH_SIZE` in training notebooks.

**Quick Test**: Set `EPOCHS=2` in training notebooks for quick validation before full run.

**Resume Training**: Each notebook saves checkpoints - you can restart from the best checkpoint if interrupted.

## 🔬 Expected Results

- **Overall Accuracy**: 85-95%
- **Macro F1**: 0.75-0.90  
- **Best Model**: Likely ResNet50 or EfficientNetB0
- **Challenge**: Limited performance on Moderate class (only 503 samples)

## 📞 Support

Each notebook includes:
- Markdown documentation explaining each step
- Print statements showing progress
- Automatic artifact saving
- Error handling with informative messages

Check `outputs/` directory for generated artifacts after running each notebook.

---

**Author**: Built with Antigravity AI  
**License**: MIT  
**Version**: 1.0.0 (Notebook-Based)
