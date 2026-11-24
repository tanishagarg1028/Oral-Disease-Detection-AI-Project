"""
model.py
--------
Builds a transfer-learning classifier for oral disease detection.
Backbone options: EfficientNetB0 (default) or ResNet50, both pretrained on
ImageNet. The base is frozen initially; call `unfreeze_top_layers` for
fine-tuning once the classification head has converged.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(num_classes, img_size=(224, 224), backbone="efficientnet"):
    inputs = tf.keras.Input(shape=(*img_size, 3))

    if backbone == "efficientnet":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
    elif backbone == "resnet":
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
    else:
        raise ValueError("backbone must be 'efficientnet' or 'resnet'")

    base.trainable = False  # freeze pretrained weights initially

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name=f"oral_disease_{backbone}")
    return model, base


def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def unfreeze_top_layers(model, base, num_layers=20, lr=1e-5):
    """Fine-tuning step: unfreeze the last `num_layers` of the backbone."""
    base.trainable = True
    for layer in base.layers[:-num_layers]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model
