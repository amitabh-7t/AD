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
