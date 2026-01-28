#!/usr/bin/env python3
import json
import os

def create_nb(path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"✓ {os.path.basename(path)}")

def mc(t, s):
    return {
        "cell_type": "markdown" if t=="m" else "code",
        "metadata": {},
        "source": s if isinstance(s, list) else [s],
        **({

"execution_count": None, "outputs": []} if t=="c" else {})
    }

# Training notebooks 05-08
models_config = [
    ("05", "baseline_cnn", "Baseline CNN", ""),
    ("06", "resnet50", "ResNet50", "resnet50"),
    ("07", "efficientnetb0", "EfficientNetB0", "efficientnet"),
    ("08", "densenet121", "DenseNet121", "densenet")
]

for num, model_id, model_name, app in models_config:
    cells = [
        mc("m", f"# Component {num}: Train {model_name}\\n\\n50 epochs, ModelCheckpoint + ReduceLROnPlateau, NO EarlyStopping"),
        mc("c", "import tensorflow as tf\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport os\nimport json\nfrom sklearn.utils.class_weight import compute_class_weight\n\nSEED=42\ntf.random.set_seed(SEED)\nnp.random.seed(SEED)\nOUTPUT_DIR='../outputs'\nos.makedirs(f'{OUTPUT_DIR}/models',exist_ok=True)\nos.makedirs(f'{OUTPUT_DIR}/training_history',exist_ok=True)\nprint('✓ Setup complete')"),
        mc("m", "## Load Data & Build Datasets"),
    ]
    
    if model_id == "baseline_cnn":
        cells.append(mc("c", "train_df=pd.read_csv('../outputs/train_manifest.csv')\nval_df=pd.read_csv('../outputs/val_manifest.csv')\nIMG_SIZE=(224,224)\nBATCH_SIZE=32\nEPOCHS=50\nNUM_CLASSES=len(train_df['class_label'].unique())\n\ndef prep(p,l):\n    img=tf.io.read_file(p)\n    img=tf.image.decode_jpeg(img,3)\n    img=tf.image.resize(img,IMG_SIZE)\n    return img/255.0,l\n\naug=tf.keras.Sequential([tf.keras.layers.RandomFlip(),tf.keras.layers.RandomRotation(0.2)])\n\ndef build_ds(df,augment=False,shuffle=True):\n    ds=tf.data.Dataset.from_tensor_slices((df['filepath'].values,df['class_label'].values))\n    ds=ds.map(prep,tf.data.AUTOTUNE).cache()\n    if augment:\n        ds=ds.map(lambda x,y:(aug(x),y))\n    if shuffle:\n        ds=ds.shuffle(1000,SEED)\n    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)\n\ntrain_ds=build_ds(train_df,True,True)\nval_ds=build_ds(val_df,False,False)\nprint(f'Train:{len(train_df)} Val:{len(val_df)} Classes:{NUM_CLASSES}')"))
        cells.extend([
            mc("m", f"## Build {model_name}"),
            mc("c", "model=tf.keras.Sequential([\n    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(*IMG_SIZE,3)),\n    tf.keras.layers.BatchNormalization(),\n    tf.keras.layers.MaxPooling2D(),\n    tf.keras.layers.Dropout(0.25),\n    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),\n    tf.keras.layers.BatchNormalization(),\n    tf.keras.layers.MaxPooling2D(),\n    tf.keras.layers.Dropout(0.25),\n    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),\n    tf.keras.layers.BatchNormalization(),\n    tf.keras.layers.MaxPooling2D(),\n    tf.keras.layers.Dropout(0.25),\n    tf.keras.layers.Conv2D(256,(3,3),activation='relu'),\n    tf.keras.layers.GlobalAveragePooling2D(),\n    tf.keras.layers.Dense(128,activation='relu'),\n    tf.keras.layers.Dropout(0.5),\n    tf.keras.layers.Dense(NUM_CLASSES,activation='softmax')\n])\nmodel.summary()")
        ])
    else:
        cells.append(mc("c", f"train_df=pd.read_csv('../outputs/train_manifest.csv')\nval_df=pd.read_csv('../outputs/val_manifest.csv')\nIMG_SIZE=(224,224)\nBATCH_SIZE=32\nEPOCHS=50\nNUM_CLASSES=len(train_df['class_label'].unique())\n\ndef prep(p,l):\n    img=tf.io.read_file(p)\n    img=tf.image.decode_jpeg(img,3)\n    img=tf.image.resize(img,IMG_SIZE)\n    img=tf.keras.applications.{app}.preprocess_input(img)\n    return img,l\n\naug=tf.keras.Sequential([tf.keras.layers.RandomFlip(),tf.keras.layers.RandomRotation(0.2)])\n\ndef build_ds(df,augment=False,shuffle=True):\n    ds=tf.data.Dataset.from_tensor_slices((df['filepath'].values,df['class_label'].values))\n    ds=ds.map(prep,tf.data.AUTOTUNE).cache()\n    if augment:\n        ds=ds.map(lambda x,y:(aug(x),y))\n    if shuffle:\n        ds=ds.shuffle(1000,SEED)\n    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)\n\ntrain_ds=build_ds(train_df,True,True)\nval_ds=build_ds(val_df,False,False)\nprint(f'Train:{{len(train_df)}} Val:{{len(val_df)}} Classes:{{NUM_CLASSES}}')"))
        cells.extend([
            mc("m", f"## Build {model_name}"),
            mc("c", f"base=tf.keras.applications.{model_name}(include_top=False,pooling='avg',weights='imagenet',input_shape=(*IMG_SIZE,3))\nfor layer in base.layers[:-20]:\n    layer.trainable=False\nmodel=tf.keras.Sequential([\n    base,\n    tf.keras.layers.Dense(128,activation='relu'),\n    tf.keras.layers.Dropout(0.5),\n    tf.keras.layers.Dense(NUM_CLASSES,activation='softmax')\n])\nmodel.summary()")
        ])
    
    cells.extend([
        mc("m", "## Compile & Class Weights"),
        mc("c", "model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])\nclass_weights=compute_class_weight('balanced',classes=np.unique(train_df['class_label']),y=train_df['class_label'])\nclass_weight_dict={i:w for i,w in enumerate(class_weights)}\nprint('Class weights:',class_weight_dict)"),
        mc("m", "## Callbacks (NO EarlyStopping)"),
        mc("c", f"model_path=f'{{OUTPUT_DIR}}/models/{model_id}_best.h5'\ncheckpoint=tf.keras.callbacks.ModelCheckpoint(model_path,monitor='val_accuracy',save_best_only=True,verbose=1)\nreduce_lr=tf.keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-7,verbose=1)\ncallbacks=[checkpoint,reduce_lr]"),
        mc("m", "## Train"),
        mc("c", "history=model.fit(train_ds,validation_data=val_ds,epochs=EPOCHS,callbacks=callbacks,class_weight=class_weight_dict)\nprint('✓ Training complete')"),
        mc("m", "## Save History & Plots"),
        mc("c", f"hist_json=f'{{OUTPUT_DIR}}/training_history/{model_id}_history.json'\nhist_csv=f'{{OUTPUT_DIR}}/training_history/{model_id}_history.csv'\nwith open(hist_json,'w') as f:\n    json.dump(history.history,f)\npd.DataFrame(history.history).to_csv(hist_csv,index=False)\n\nfig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))\nax1.plot(history.history['accuracy'],label='Train')\nax1.plot(history.history['val_accuracy'],label='Val')\nax1.set_title('Accuracy',fontweight='bold')\nax1.legend()\nax1.grid(alpha=0.3)\nax2.plot(history.history['loss'],label='Train')\nax2.plot(history.history['val_loss'],label='Val')\nax2.set_title('Loss',fontweight='bold')\nax2.legend()\nax2.grid(alpha=0.3)\nplt.suptitle('{model_name} Training Progress',fontsize=14,fontweight='bold')\nplt.tight_layout()\nplt.savefig(f'{{OUTPUT_DIR}}/training_history/{model_id}_curves.png',dpi=200)\nplt.show()\nprint(f'✓ Best model saved to: {{model_path}}')")
    ])
    
    create_nb(f"notebooks/{num}_train_{model_id}.ipynb", cells)

print("\n✅ Created training notebooks 05-08")
