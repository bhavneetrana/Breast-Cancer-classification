import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Reshape,
    LSTM,
    Bidirectional,
    Input,
    BatchNormalization,
    Layer,
    SpatialDropout2D
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras import regularizers
from tensorflow.keras import mixed_precision

import tensorflow.keras.backend as K

from sklearn.utils import class_weight

# ======================================================
# MIXED PRECISION TRAINING
# ======================================================
mixed_precision.set_global_policy("mixed_float16")

# ======================================================
# CONFIG
# ======================================================
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 50

# ======================================================
# ATTENTION LAYER
# ======================================================
class Attention(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)

    def call(self, x):

        e = K.tanh(K.dot(x, self.W) + self.b)

        a = K.softmax(e, axis=1)

        return K.sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()

# ======================================================
# FGSM ADVERSARIAL ATTACK
# ======================================================
def create_adversarial_examples(
    model,
    images,
    labels,
    epsilon=0.01
):

    images = tf.cast(images, tf.float32)

    with tf.GradientTape() as tape:

        tape.watch(images)

        predictions = model(images, training=False)

        loss = tf.keras.losses.binary_crossentropy(
            labels,
            predictions
        )

    gradients = tape.gradient(
        loss,
        images
    )

    signed_gradients = tf.sign(
        gradients
    )

    adversarial_images = (
        images + epsilon * signed_gradients
    )

    adversarial_images = tf.clip_by_value(
        adversarial_images,
        0.0,
        1.0
    )

    return adversarial_images

# ======================================================
# DATA AUGMENTATION
# ======================================================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

val_gen = ImageDataGenerator(
    rescale=1./255
)

train_data = train_gen.flow_from_directory(
    "dataset/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    "dataset/val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ======================================================
# CLASS WEIGHTS
# ======================================================
cw = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

cw_dict = dict(
    enumerate(cw)
)
# ======================================================
# MODEL ARCHITECTURE
# ======================================================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = base_model(
    inputs,
    training=False
)

x = SpatialDropout2D(0.3)(x)

x = BatchNormalization()(x)

# MobileNetV2 output:
# 3 x 3 x 1280

x = Reshape(
    (-1, 1280)
)(x)

x = Bidirectional(
    LSTM(
        128,
        return_sequences=True,
        kernel_regularizer=regularizers.l2(0.001)
    )
)(x)

x = Attention()(x)

x = Dense(
    128,
    activation="relu",
    kernel_regularizer=regularizers.l2(0.001)
)(x)

x = Dropout(0.5)(x)

outputs = Dense(
    1,
    activation="sigmoid",
    dtype="float32"
)(x)

model = Model(
    inputs,
    outputs
)

model.summary()

# ======================================================
# LEARNING RATE SCHEDULE
# ======================================================
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000
)

# ======================================================
# COMPILE MODEL
# ======================================================
model.compile(
    optimizer=Adam(
        learning_rate=lr_schedule
    ),
    loss=tf.keras.losses.BinaryCrossentropy(
        label_smoothing=0.1
    ),
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

# ======================================================
# CALLBACKS
# ======================================================
callbacks = [

    EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    ),

    ModelCheckpoint(
        "best_model.h5",
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    )
]

# ======================================================
# TRAIN MODEL
# ======================================================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=cw_dict
)

# ======================================================
# ADVERSARIAL ROBUSTNESS TEST
# ======================================================
print("\n")
print("=" * 50)
print("FGSM ADVERSARIAL ROBUSTNESS TEST")
print("=" * 50)

x_batch, y_batch = next(val_data)

y_batch = y_batch.reshape(-1, 1)

adv_images = create_adversarial_examples(
    model,
    x_batch,
    y_batch,
    epsilon=0.01
)

clean_loss, clean_acc, clean_auc = model.evaluate(
    x_batch,
    y_batch,
    verbose=0
)

adv_loss, adv_acc, adv_auc = model.evaluate(
    adv_images,
    y_batch,
    verbose=0
)

print(f"Clean Accuracy       : {clean_acc*100:.2f}%")
print(f"Adversarial Accuracy : {adv_acc*100:.2f}%")
print(f"Clean AUC            : {clean_auc:.4f}")
print(f"Adversarial AUC      : {adv_auc:.4f}")
print(
    f"Robustness Drop      : "
    f"{(clean_acc-adv_acc)*100:.2f}%"
)

print("=" * 50)

# ======================================================
# SAVE TRAINING CURVES
# ======================================================
plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig(
    "accuracy_curve.png"
)

plt.close()

# ======================================================
# SAVE AUC CURVE
# ======================================================
plt.figure(figsize=(8,5))

plt.plot(
    history.history["auc"],
    label="Train AUC"
)

plt.plot(
    history.history["val_auc"],
    label="Validation AUC"
)

plt.title("Training AUC")

plt.xlabel("Epoch")

plt.ylabel("AUC")

plt.legend()

plt.savefig(
    "auc_curve.png"
)

plt.close()

# ======================================================
# SAVE FINAL MODEL
# ======================================================
model.save(
    "cnn_bilstm_attention_model.h5"
)

print(
    "\n✅ High-Accuracy Model saved successfully."
)

print(
    "📈 Accuracy curve saved as accuracy_curve.png"
)

print(
    "📈 AUC curve saved as auc_curve.png"
)

print(
    "🔒 FGSM Adversarial Robustness Testing completed."
)
