"""
train.py
--------
End-to-end training entry point.

Usage:
    python src/train.py --data_dir data --epochs 15 --backbone efficientnet

Produces:
    models/best_model.keras
    outputs/training_curves.png
    outputs/confusion_matrix.png
    outputs/classification_report.txt
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from preprocessing import build_datasets
from model import build_model, compile_model, unfreeze_top_layers


def plot_training_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_and_report(model, test_ds, class_names, out_dir):
    y_true, y_pred = [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
    plt.close()

    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(f"{out_dir}/classification_report.txt", "w") as f:
        f.write(report)
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine_tune_epochs", type=int, default=5)
    parser.add_argument("--backbone", default="efficientnet", choices=["efficientnet", "resnet"])
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = build_datasets(args.data_dir)
    print("Classes:", class_names)

    model, base = build_model(len(class_names), backbone=args.backbone)
    model = compile_model(model)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f"{args.model_dir}/best_model.keras", save_best_only=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Fine-tuning stage
    model = unfreeze_top_layers(model, base)
    history_ft = model.fit(
        train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=callbacks
    )

    plot_training_curves(history, f"{args.out_dir}/training_curves.png")
    evaluate_and_report(model, test_ds, class_names, args.out_dir)

    model.save(f"{args.model_dir}/final_model.keras")
    print("Training complete. Model + evaluation artifacts saved to", args.out_dir, args.model_dir)


if __name__ == "__main__":
    main()
