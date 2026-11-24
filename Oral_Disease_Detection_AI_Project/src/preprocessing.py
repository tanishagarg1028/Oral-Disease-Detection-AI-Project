"""
preprocessing.py
-----------------
Loads oral disease image data from a folder structure and prepares it for
training. Expected folder layout (standard Keras "flow_from_directory" style):

    data/
        train/
            healthy/
            cavity/
            gum_disease/
            oral_lesion/
        val/
            healthy/
            cavity/
            gum_disease/
            oral_lesion/
        test/
            healthy/
            cavity/
            gum_disease/
            oral_lesion/

This matches the class structure used by the public Kaggle datasets:
  - "Oral Diseases (panoramic x-ray)" by salmansajid05
  - "Oral Cancer (Lips and Tongue) Images" by shivam17299

If you are using one of those datasets, first split it into train/val/test
sub-folders (see scripts/split_dataset.py) before pointing this module at it.
"""

import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_datasets(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=42):
    """
    Builds train/val/test tf.data pipelines with resizing, normalization,
    and augmentation (train only) applied.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        seed=seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        seed=seed,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        seed=seed,
    )

    class_names = train_ds.class_names

    # Data augmentation — applied only to the training set
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    normalize = tf.keras.layers.Rescaling(1. / 255)

    def prep_train(x, y):
        x = augment(x, training=True)
        x = normalize(x)
        return x, y

    def prep_eval(x, y):
        x = normalize(x)
        return x, y

    train_ds = train_ds.map(prep_train, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(prep_eval, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(prep_eval, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
