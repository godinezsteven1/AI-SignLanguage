# evaluate_model.py

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE

# Import your data preparation and generator
from combine_training import prepare_combined_dataset, CombinedDataGenerator

def main():
    # 1) Prepare dataset & validation generator
    combined_data, _ = prepare_combined_dataset()
    class_names = [cls for _, cls in combined_data]
    batch_size = 64
    val_gen = CombinedDataGenerator(combined_data, batch_size, is_training=False)
    n_batches = len(val_gen)

    # 2) Load & compile saved model
    model = load_model('../models/full_model_combined_classes.h5')
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 3) Evaluate overall accuracy & timing
    start = time.time()
    val_loss, val_acc = model.evaluate(val_gen, verbose=1)
    elapsed = time.time() - start
    print(f"\nValidation accuracy: {val_acc*100:.2f}%  (took {elapsed:.1f}s)")

    # 4) Gather all true & predicted labels
    y_true, y_pred = [], []
    for i in range(n_batches):
        X, y = val_gen[i]
        preds = model.predict(X, verbose=0).argmax(axis=1)
        trues = y.argmax(axis=1)
        y_pred.extend(preds.tolist())
        y_true.extend(trues.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 5) Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12,10))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90)
    plt.yticks(ticks, class_names)
    plt.tight_layout()
    plt.show()

    # 6) Classification report as a table figure
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).T

    plt.figure(figsize=(10, report_df.shape[0] * 0.5))
    plt.axis('off')
    tbl = plt.table(
        cellText=np.round(report_df.values, 2),
        rowLabels=report_df.index,
        colLabels=report_df.columns,
        cellLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.title("Classification Report")
    plt.show()

    # 7) Compute per‑language accuracy
    lang_acc = {'asl': [0,0], 'dgs': [0,0], 'lse': [0,0]}  # [correct, total]
    for t, p in zip(y_true, y_pred):
        langs = class_names[t].split('_')[1].split('-')
        for l in langs:
            lang_acc[l][1] += 1
            if t == p:
                lang_acc[l][0] += 1
    print("\nPer‑language accuracy:")
    for l, (corr, total) in lang_acc.items():
        acc = (corr / total * 100) if total else 0
        print(f"  {l.upper():3}: {acc:.2f}% ({corr}/{total})")

    # 8) t‑SNE of penultimate features
    # Build feature‑extractor submodel
    try:
        input_tensor = model.inputs[0]
    except Exception:
        input_tensor = model.input

    try:
        pool_layer = model.get_layer("global_average_pooling2d")
    except ValueError:
        from tensorflow.keras.layers import GlobalAveragePooling2D
        pool_layer = next(layer for layer in model.layers if isinstance(layer, GlobalAveragePooling2D))

    feat_model = tf.keras.Model(inputs=input_tensor, outputs=pool_layer.output)

    feats, labels = [], []
    for i in range(n_batches):
        X, y = val_gen[i]
        feats.append(feat_model.predict(X, verbose=0))
        labels.append(y.argmax(axis=1))
    feats = np.vstack(feats)
    labels = np.concatenate(labels)

    tsne = TSNE(n_components=2, init='random', random_state=0)
    proj = tsne.fit_transform(feats)

    plt.figure(figsize=(8,8))
    for idx, cls in enumerate(class_names):
        ix = labels == idx
        plt.scatter(proj[ix,0], proj[ix,1], s=5, label=cls)
    plt.legend(markerscale=3, bbox_to_anchor=(1.05,1), loc='upper left')
    plt.title("t-SNE of CNN Features")
    plt.show()

if __name__ == "__main__":
    main()
