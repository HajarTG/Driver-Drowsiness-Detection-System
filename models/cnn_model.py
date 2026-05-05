"""
CNN from Scratch model for Driver Drowsiness Detection.

Architecture designed to classify eye/face images as Awake or Drowsy.
Approximate parameter count: ~9.8M
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_cnn_model(input_shape=(64, 64, 3), num_classes=1):
    """
    Build a CNN model from scratch for drowsiness detection.

    The architecture consists of 5 convolutional blocks followed by
    fully connected layers. Batch Normalization and Dropout are applied
    throughout to reduce overfitting.

    Args:
        input_shape (tuple): Shape of input images (H, W, C).
                             Default is (64, 64, 3).
        num_classes (int): Number of output classes.
                           1 for binary (sigmoid), >1 for softmax.

    Returns:
        tf.keras.Model: Compiled CNN model (~9.8M parameters).

    Example:
        >>> model = build_cnn_model(input_shape=(64, 64, 3))
        >>> model.summary()
    """
    inputs = layers.Input(shape=input_shape, name="input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(0.25, name="drop3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="relu4_1")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4_2")(x)
    x = layers.BatchNormalization(name="bn4_2")(x)
    x = layers.Activation("relu", name="relu4_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)
    x = layers.Dropout(0.25, name="drop4")(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), padding="same", name="conv5_1")(x)
    x = layers.BatchNormalization(name="bn5_1")(x)
    x = layers.Activation("relu", name="relu5_1")(x)
    x = layers.Conv2D(512, (3, 3), padding="same", name="conv5_2")(x)
    x = layers.BatchNormalization(name="bn5_2")(x)
    x = layers.Activation("relu", name="relu5_2")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="drop5")(x)

    # Fully Connected Layers
    x = layers.Dense(
        4096,
        kernel_regularizer=regularizers.l2(1e-4),
        name="fc1",
    )(x)
    x = layers.BatchNormalization(name="bn_fc1")(x)
    x = layers.Activation("relu", name="relu_fc1")(x)
    x = layers.Dropout(0.5, name="drop_fc1")(x)

    x = layers.Dense(
        768,
        kernel_regularizer=regularizers.l2(1e-4),
        name="fc2",
    )(x)
    x = layers.BatchNormalization(name="bn_fc2")(x)
    x = layers.Activation("relu", name="relu_fc2")(x)
    x = layers.Dropout(0.5, name="drop_fc2")(x)

    # Output layer
    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs, outputs, name="CNN_Drowsiness_Detector")

    # Compile
    if num_classes == 1:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    return model


def get_cnn_callbacks(checkpoint_path="saved_models/cnn_best.weights.h5"):
    """
    Return a standard list of Keras callbacks for CNN training.

    Args:
        checkpoint_path (str): Path to save the best model weights.

    Returns:
        list: List of tf.keras.callbacks.Callback objects.
    """
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="logs/cnn",
            histogram_freq=1,
        ),
    ]
