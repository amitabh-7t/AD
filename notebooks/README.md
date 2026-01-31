# Notebook Guide - MRI Alzheimer's Classification Pipeline

This directory contains 13 Jupyter notebooks that form a complete end-to-end machine learning pipeline for Alzheimer's disease classification from MRI scans.

## 🚀 Quick Start

Execute notebooks sequentially in order (01 → 02 → ... → 13):

```bash
jupyter lab
```

**Total Runtime**: ~5-10 hours for complete pipeline (or ~2.5-3.5 hours if running training notebooks 05-08 in parallel)

---

## 📋 Notebook Execution Order

### Phase 1: Data Preparation (01-04) - ~30-60 minutes

#### 01. Dataset Ingestion
**File**: `01_dataset_ingestion.ipynb`  
**Runtime**: ~5-10 minutes

Loads 11,743 MRI images, validates integrity using SHA256 hashing, and creates stratified train/val/test splits.

**Key Outputs**:
- `outputs/dataset_manifest.csv` - Complete dataset metadata
- `outputs/train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv` - Split files
- `outputs/class_distribution.png` - Visual class distribution

**What to check**: Verify class distribution is stratified across splits

---

#### 02. Exploratory Data Analysis (EDA)
**File**: `02_eda.ipynb`  
**Runtime**: ~10-15 minutes

Comprehensive visual and statistical analysis of the dataset.

**Key Outputs**:
- `outputs/eda/sample_grid_*.png` - 8-image grids per class
- `outputs/eda/intensity_histograms.png` - Pixel intensity distributions
- `outputs/eda/duplicates_report.csv` - Potential duplicate images
- `outputs/eda/eda_report.md` - Summary report

**What to check**: Review intensity distributions and duplicate count

---

#### 03. Preprocessing
**File**: `03_preprocessing.ipynb`  
**Runtime**: ~2-3 minutes

Defines preprocessing functions and builds optimized tf.data pipelines.

**Key Functions Created**:
- `preprocess_resize_rescale()` - For Baseline CNN
- `preprocess_resnet()` - For ResNet50
- `preprocess_efficientnet()` - For EfficientNetB0
- `preprocess_densenet()` - For DenseNet121
- `build_dataset()` - Optimized pipeline with caching/prefetching

**What to check**: Verify sample batch visualization looks correct

---

#### 04. Feature Visualization  
**File**: `04_feature_visualization.ipynb`  
**Runtime**: ~15-30 minutes

Extracts deep features using frozen ResNet50 and visualizes using dimensionality reduction.

**Key Outputs**:
- `outputs/features/features_resnet50.npy` - 2048-d feature vectors
- `outputs/features/tsne_plot.png` - t-SNE 2D visualization
- `outputs/features/umap_plot.png` - UMAP 2D visualization
- `outputs/features/tmap_plot.png` - Hierarchical clustering tree

**What to check**: Look for class separation in t-SNE/UMAP plots

---

### Phase 2: Model Training (05-08) - ~4-8 hours total

**Important Notes**:
- Each notebook trains for exactly **50 epochs** (NO EarlyStopping)
- Uses **ModelCheckpoint** to save best model by val_accuracy
- Uses **ReduceLROnPlateau** for dynamic learning rate adjustment
- Applies **class weights** to handle imbalanced dataset
- Can run in **parallel** to reduce total time to ~1.5-2.5 hours

#### 05. Train Baseline CNN
**File**: `05_train_baseline_cnn.ipynb`  
**Runtime**: ~1-2 hours

Custom 4-layer CNN architecture trained from scratch.

**Architecture**:
```
Conv2D(32) → BN → MaxPool → Dropout(0.25)
Conv2D(64) → BN → MaxPool → Dropout(0.25)
Conv2D(128) → BN → MaxPool → Dropout(0.25)
Conv2D(256) → GlobalAvgPool
Dense(128) → Dropout(0.5) → Dense(4)
```

**Outputs**:
- `outputs/models/baseline_cnn_best.h5` - Best model checkpoint
- `outputs/training_history/baseline_cnn_history.json` - Training metrics
- `outputs/training_history/baseline_cnn_curves.png` - Training curves

**What to check**: Training/validation curves should show learning without overfitting

---

#### 06. Train ResNet50
**File**: `06_train_resnet50.ipynb`  
**Runtime**: ~1.5-2.5 hours

Transfer learning with ResNet50 (ImageNet weights), fine-tuning last 20 layers.

**Outputs**: Same structure as Baseline CNN
- `outputs/models/resnet50_best.h5`
- `outputs/training_history/resnet50_*.{json,csv,png}`

**What to check**: Should achieve higher accuracy than Baseline CNN

---

#### 06b. Train ResNet50 Attention
**File**: `06b_train_resnet50_attention.ipynb`
**Runtime**: ~2-3 hours

ResNet50 with added Spatial Attention modules to focus on salient features.

**Outputs**:
- `outputs/models/resnet50_attention_best.h5`
- `outputs/training_history/resnet50_attention_*.{json,csv,png}`

**What to check**: Attention maps should highlight relevant brain regions

---

#### 07. Train EfficientNetB0
**File**: `07_train_efficientnetb0.ipynb`  
**Runtime**: ~1-2 hours

Transfer learning with EfficientNetB0, known for efficiency and accuracy balance.

**Outputs**: Same structure as Baseline CNN
- `outputs/models/efficientnetb0_best.h5`
- `outputs/training_history/efficientnetb0_*.{json,csv,png}`

**What to check**: Typically achieves good accuracy with smallest model size

---



### Phase 3: Analysis & Reporting (09-13) - ~45-75 minutes

#### 09. Evaluation
**File**: `09_evaluation.ipynb`  
**Runtime**: ~10-15 minutes

Comprehensive evaluation of all 4 models on the test set.

**Metrics Computed**:
- Accuracy, Balanced Accuracy
- Precision, Recall, F1-score (per class + macro/weighted)
- ROC AUC (one-vs-rest multilabel)

**Key Outputs**:
- `outputs/evaluation/*_predictions.csv` - Predictions with probabilities
- `outputs/evaluation/*_confusion_matrix.png` - Confusion matrices
- `outputs/evaluation/*_roc_curves.png` - ROC curves
- `outputs/evaluation/all_models_metrics.csv` - Comparative metrics

**What to check**: Review confusion matrices to identify class-specific performance

---

#### 10. Grad-CAM Explainability
**File**: `10_gradcam.ipynb`  
**Runtime**: ~20-30 minutes

Generates Grad-CAM heatmaps showing which image regions each model focuses on.

**Key Outputs**:
- `outputs/gradcam/baseline_cnn/*.png` - 20+ heatmap overlays
- `outputs/gradcam/resnet50/*.png` - 20+ heatmap overlays
- `outputs/gradcam/efficientnetb0/*.png` - 20+ heatmap overlays
- `outputs/gradcam/densenet121/*.png` - 20+ heatmap overlays

**What to check**: Verify models focus on relevant brain regions, not background

---

#### 11. Model Comparison
**File**: `11_comparison.ipynb`  
**Runtime**: ~5-10 minutes

Cross-model comparison across accuracy, model size, and inference speed.

**Key Outputs**:
- `outputs/comparison/model_comparison.png` - Comparison charts
- Rankings by Macro F1, accuracy, size, speed

**What to check**: Identify best model for your deployment constraints

---

#### 12. Robustness Testing
**File**: `12_robustness.ipynb`  
**Runtime**: ~15-20 minutes

Tests model stability under adversarial conditions (noise, intensity variations).

**Tests**:
- Gaussian noise (σ = 0.05, 0.1, 0.15, 0.2)
- Intensity scaling (factors: 0.8, 0.9, 1.1, 1.2)

**Key Outputs**:
- Performance degradation metrics
- Robustness rankings

**What to check**: Identify which models are most robust to image variations

---

#### 13. Final Report
**File**: `13_final_report.ipynb`  
**Runtime**: ~2-3 minutes

Aggregates all results into a comprehensive markdown report.

**Key Output**:
- `outputs/FINAL_REPORT.md` - Complete results summary with embedded visualizations

**What to check**: Review for key findings and recommendations

---

## 📊 Expected Results

After running all notebooks, you should see:

**Best Model**: Likely ResNet50 or EfficientNetB0  
**Overall Accuracy**: 85-95%  
**Macro F1**: 0.75-0.90  
**Challenge**: Limited performance on Moderate class (only 503 samples)

## 🔧 Customization

To modify training:
- **Quick test**: Set `EPOCHS=5` in training notebooks (05-08)
- **Batch size**: Adjust `BATCH_SIZE` if GPU memory limited
- **Learning rate**: Modify `Adam(learning_rate=...)` in compile sections

## 📝 Notes

- All notebooks save outputs to `../outputs/[component_name]/`
- Notebooks can be re-run independently (after dependencies)
- Models are large (~50-130 MB each) - excluded from git via `.gitignore`
- Full pipeline generates ~15-20 GB of artifacts

## 🆘 Troubleshooting

**OOM Error**: Reduce `BATCH_SIZE` in training notebooks  
**Slow Training**: Enable GPU acceleration (should use Metal on M3)  
**Missing Dependencies**: Run `pip install -r ../requirements.txt`  
**Corrupted Images**: Check `outputs/eda/eda_report.md` for flagged files

---

**Happy Training! 🚀**

For questions or issues, check the main `README.md` in the project root.
