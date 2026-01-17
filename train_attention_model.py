import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Reshape, LSTM, Bidirectional,
    Input, BatchNormalization, Layer, SpatialDropout2D
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from sklearn.utils import class_weight

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 50 

# ===============================
# ATTENTION LAYER (SERIALIZATION SAFE)
# ===============================
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()

# ===============================
# DATA LOADING & AUGMENTATION
# ===============================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_gen = ImageDataGenerator(rescale=1./255)

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

# ===============================
# MODEL ARCHITECTURE
# ===============================
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False 

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)

# Spatial Dropout prevents overfitting in CNN feature maps
x = SpatialDropout2D(0.3)(x)
x = BatchNormalization()(x)

# Convert to sequence for BiLSTM
x = Reshape((-1, x.shape[-1]))(x)

# BiLSTM with L2 Regularization
x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))(x)
x = Attention()(x)

x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)

# Cosine Decay Learning Rate for smoother convergence
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000
)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name='auc')]
)

# ===============================
# CALLBACKS & TRAINING
# ===============================
cw = class_weight.compute_class_weight('balanced', classes=np.unique(train_data.classes), y=train_data.classes)
cw_dict = dict(enumerate(cw))

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=cw_dict
)

model.save("cnn_bilstm_attention_model.h5")
print("✅ High-Accuracy Model saved as cnn_bilstm_attention_model.h5")
