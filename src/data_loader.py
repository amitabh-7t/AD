import tensorflow as tf
import os
import pathlib
import numpy as np

# Configuration
BATCH_SIZE = 32
IMG_HEIGHT = 224  # Standard size for many CNNs
IMG_WIDTH = 224
DATA_DIR = '../data' # Relative to src/ directory or notebook
SEED = 42

def load_data(data_dir=DATA_DIR, batch_size=BATCH_SIZE, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Loads data from directory, splits into train/val/test, and returns tf.data.Datasets.
    We'll use an 80/10/10 split or 70/15/15.
    """
    data_dir = pathlib.Path(data_dir)
    print(f"Loading data from: {data_dir.resolve()}")
    
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    # Use validation_split for Train/Val
    # We will grab a larger chunk for "training" first (e.g. 80%)
    # and then split the remaining "validation" set into val and test manually if needed, 
    # or just use a simple 80/20 train/val split for now as per minimal requirements,
    # but for research grade, we want a hold-out test set.
    
    # Strategy: 
    # 1. Load full dataset flattened or by list_files to create custom splits (best for control).
    # 2. OR use image_dataset_from_directory twice with seed. 
    
    # Let's use image_dataset_from_directory for specific splits
    # Training (70%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3, # Reserve 30% for Val+Test
        subset="training",
        seed=SEED,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical' # For multi-class
    )

    # Validation + Test (30%)
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="validation",
        seed=SEED,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    # Split val_test_ds into 50% Val and 50% Test (so 15% / 15% of total)
    val_batches = tf.data.experimental.cardinality(val_test_ds)
    test_ds = val_test_ds.take(val_batches // 2)
    val_ds = val_test_ds.skip(val_batches // 2)

    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")
    
    # Optimization: cache and prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names

if __name__ == "__main__":
    # Test the loader
    train, val, test, classes = load_data(data_dir='/Users/amitabhthakur/Workspace/Projects/ML/AD/data')
    print(f"Training batches: {tf.data.experimental.cardinality(train)}")
    print(f"Validation batches: {tf.data.experimental.cardinality(val)}")
    print(f"Test batches: {tf.data.experimental.cardinality(test)}")
