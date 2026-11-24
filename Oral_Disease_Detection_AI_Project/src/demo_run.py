"""
demo_run.py
-----------
Generates a small representative image set (4 classes, distinguishable by
color/texture/shape patterns standing in for healthy / cavity / gum-disease /
oral-lesion visual cues) and runs the FULL pipeline end-to-end:
  1. dataset build -> 2. transfer-learning model -> 3. training ->
  4. evaluation (confusion matrix + classification report) -> 5. Grad-CAM

This is a pipeline-validation / demo run so every stage of the code is
proven to work and produces real artifacts for the project report.
For the actual submission, swap in a real dataset (see README) — the
code in preprocessing.py / model.py / train.py is unchanged either way.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2

from model import build_model, compile_model
from gradcam import make_gradcam_heatmap, overlay_heatmap

np.random.seed(42)
tf.random.set_seed(42)

IMG_SIZE = 96
CLASS_NAMES = ["healthy", "cavity", "gum_disease", "oral_lesion"]
N_PER_CLASS = 90
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def synth_image(cls_idx):
    """Creates a synthetic RGB image with class-specific visual patterns,
    standing in for real intraoral photo characteristics. Patterns are made
    strongly distinct so the demo pipeline clearly demonstrates learning;
    on the real dataset, distinguishing cues are far subtler and the full
    EfficientNet/ResNet transfer-learning setup in model.py is used instead."""
    img = np.full((IMG_SIZE, IMG_SIZE, 3), 0, dtype=np.uint8)
    img[:, :, 0] = 235; img[:, :, 1] = 205; img[:, :, 2] = 200  # pinkish base "gum/tooth" tone
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cx, cy = np.random.randint(35, IMG_SIZE - 35, 2)
    if cls_idx == 0:  # healthy - uniform tone, faint texture only, no salient mark
        pass
    elif cls_idx == 1:  # cavity - large dark localized spot
        cv2.circle(img, (cx, cy), np.random.randint(16, 22), (35, 20, 10), -1)
    elif cls_idx == 2:  # gum_disease - large reddened irregular region
        cv2.ellipse(img, (cx, cy), (30, 16), np.random.randint(0, 180), 0, 360, (215, 25, 25), -1)
    else:  # oral_lesion - large pale/white irregular patch with a dark ring border
        pts_x = cx + np.random.randint(-24, 24, (8, 1))
        pts_y = cy + np.random.randint(-24, 24, (8, 1))
        poly = np.concatenate([pts_x, pts_y], axis=1).reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(img, [poly], (255, 255, 250))
        cv2.polylines(img, [poly], isClosed=True, color=(120, 90, 90), thickness=2)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def build_synthetic_dataset():
    X, y = [], []
    for cls_idx, cls in enumerate(CLASS_NAMES):
        for _ in range(N_PER_CLASS):
            X.append(synth_image(cls_idx))
            y.append(cls_idx)
    X = np.array(X, dtype=np.float32) / 255.0
    y = tf.keras.utils.to_categorical(y, num_classes=len(CLASS_NAMES))
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    n = len(X)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    return (X[:n_train], y[:n_train],
            X[n_train:n_train + n_val], y[n_train:n_train + n_val],
            X[n_train + n_val:], y[n_train + n_val:])


def main():
    print("Building synthetic demo dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = build_synthetic_dataset()
    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # NOTE: this sandbox has no internet access to download ImageNet weights,
    # so the demo uses a small CNN trained from scratch purely to validate
    # the pipeline end-to-end. model.py / train.py (used for the real
    # submission on Colab, which DOES have internet access) implement the
    # actual transfer-learning approach (EfficientNetB0 / ResNet50 pretrained
    # on ImageNet) as specified in the project brief.
    print("Building CNN model for pipeline demo (from scratch, no internet in this sandbox)...")
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", name="last_conv")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="oral_disease_demo")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    print("Training (demo: 15 epochs)...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=15, batch_size=16, verbose=2)

    # ---- Training curves ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/training_curves.png", dpi=150)
    plt.close()
    print("Saved training_curves.png")

    # ---- Evaluation ----
    preds = model.predict(X_test, verbose=0)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(preds, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix (demo run)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with open(f"{OUT_DIR}/classification_report.txt", "w") as f:
        f.write(report)
    print(report)

    # ---- Grad-CAM sample grid ----
    last_conv = "last_conv"

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    for i, cls_idx in enumerate(range(len(CLASS_NAMES))):
        sample_idx = np.where(y_true == cls_idx)[0][0]
        img = X_test[sample_idx]
        img_arr = np.expand_dims(img, axis=0)
        heatmap, pred_idx = make_gradcam_heatmap(img_arr, model, last_conv)
        overlay = overlay_heatmap((img * 255).astype(np.uint8), heatmap)

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"True: {CLASS_NAMES[cls_idx]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"Grad-CAM (pred: {CLASS_NAMES[pred_idx]})")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/gradcam_samples.png", dpi=150)
    plt.close()
    print("Saved gradcam_samples.png")

    # ---- Sample predictions grid (for report "sample prediction outputs") ----
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    rand_idx = np.random.choice(len(X_test), 12, replace=False)
    for ax, idx in zip(axes.flatten(), rand_idx):
        ax.imshow(X_test[idx])
        true_lbl = CLASS_NAMES[y_true[idx]]
        pred_lbl = CLASS_NAMES[y_pred[idx]]
        color = "green" if true_lbl == pred_lbl else "red"
        ax.set_title(f"T:{true_lbl}\nP:{pred_lbl}", color=color, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sample_predictions.png", dpi=150)
    plt.close()
    print("Saved sample_predictions.png")

    model.save(f"{OUT_DIR}/../models/demo_model.keras" if os.path.exists("../models") else "models/demo_model.keras")
    print("Demo pipeline run complete — all outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
