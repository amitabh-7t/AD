# Exploratory Data Analysis Report

## Dataset Overview
- **Total images**: 11743
- **Classes**: 4
- **Unique dimensions**: 4
- **Potential duplicates**: 778

## Class Distribution
- **1 Very Mild**: 2240 images (19.1%)
- **2 Mild**: 896 images (7.6%)
- **3 Moderate**: 503 images (4.3%)
- **Normal**: 8104 images (69.0%)

## Key Findings

### Imbalance
- Significant class imbalance detected
- Normal class: ~69%, Moderate class: ~4%
- **Recommendation**: Use class weights or stratified sampling

### Consistency
- Most common size: (256, 256)
- Images with different sizes: 4
- **Recommendation**: Resize all images to consistent dimensions

### Data Quality
- Potential duplicates found: 778
- **Recommendation**: Review and remove duplicates if necessary
