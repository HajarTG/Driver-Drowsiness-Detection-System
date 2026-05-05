"""
MobileNetV2 Transfer Learning model for Driver Drowsiness Detection.

Uses pretrained MobileNetV2 weights (ImageNet) as a feature extractor,
with a custom classification head for drowsiness detection.
Approximate parameter count: ~2.4M (trainable after fine-tuning)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2


def build_mobilenet_model(input_shape=(96, 96, 3), num_classes=1, fine_tune_at=100):
    """
    Build a MobileNetV2 transfer learning model for drowsiness detection.

    The base MobileNetV2 is first frozen (feature extraction phase), then
    layers above `fine_tune_at` are unfrozen for fine-tuning. Call
    ``unfreeze_for_fine_tuning()`` after the initial training phase.

    Args:
        input_shape (tuple): Shape of input images (H, W, C).
                             Minimum size supported by MobileNetV2 is 32x32x3.
                             Default is (96, 96, 3).
        num_classes (int): Number of output classes.
                           1 for binary (sigmoid), >1 for softmax.
        fine_tune_at (int): Layer index from which to unfreeze during
                            fine-tuning. Default is 100.

    Returns:
        tuple[tf.keras.Model, tf.keras.Model]:
            (full_model, base_model) where base_model is the MobileNetV2
            backbone, useful for controlling fine-tuning.

    Example:
        >>> model, base = build_mobilenet_model()
        >>> model.summary()
        >>> # After initial training, unfreeze for fine-tuning:
        >>> unfreeze_for_fine_tuning(base, model, fine_tune_at=100)
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze base during feature extraction

    inputs = layers.Input(shape=input_shape, name="input")

    # Preprocessing: scale pixel values to [-1, 1] as expected by MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Base model (feature extraction)
    x = base_model(x, training=False)

    # Custom classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(
        256,
        kernel_regularizer=regularizers.l2(1e-4),
        name="fc1",
    )(x)
    x = layers.BatchNormalization(name="bn_fc1")(x)
    x = layers.Activation("relu", name="relu_fc1")(x)
    x = layers.Dropout(0.5, name="drop_fc1")(x)

    x = layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(1e-4),
        name="fc2",
    )(x)
    x = layers.BatchNormalization(name="bn_fc2")(x)
    x = layers.Activation("relu", name="relu_fc2")(x)
    x = layers.Dropout(0.3, name="drop_fc2")(x)

    # Output layer
    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs, outputs, name="MobileNetV2_Drowsiness_Detector")

    _compile_model(model, num_classes, learning_rate=1e-3)

    return model, base_model


def unfreeze_for_fine_tuning(base_model, model, fine_tune_at=100, learning_rate=1e-5):
    """
    Unfreeze MobileNetV2 layers above `fine_tune_at` for fine-tuning.

    Call this after the initial feature-extraction training phase has
    converged. Using a lower learning rate avoids destroying the pretrained
    weights.

    Args:
        base_model (tf.keras.Model): The MobileNetV2 backbone.
        model (tf.keras.Model): The full compiled model.
        fine_tune_at (int): Index of the first layer to unfreeze.
        learning_rate (float): Lower learning rate for fine-tuning.
    """
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    _compile_model(model, num_classes=1, learning_rate=learning_rate)
    print(
        f"Fine-tuning enabled: {sum(1 for l in base_model.layers if l.trainable)} "
        f"trainable layers in backbone (from layer {fine_tune_at} onward)."
    )


def _compile_model(model, num_classes, learning_rate):
    """Compile the model with appropriate loss and metrics."""
    if num_classes == 1:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )


def get_mobilenet_callbacks(
    checkpoint_path="saved_models/mobilenet_best.weights.h5",
):
    """
    Return a standard list of Keras callbacks for MobileNetV2 training.

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
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="logs/mobilenet",
            histogram_freq=1,
        ),
    ]
