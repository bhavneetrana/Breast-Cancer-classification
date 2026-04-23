"""
model_utils.py — Optimized model loading, prediction, and GradCAM visualization.
"""

import os
import urllib.request
import numpy as np
from PIL import Image
import streamlit as st

try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
MODEL_PATH = "cnn_bilstm_attention_model.h5"
IMG_SIZE = (96, 96)

if TF_AVAILABLE:
    @tf.keras.utils.register_keras_serializable(package="Custom")
    class Attention(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.last_attention = None

        def build(self, input_shape):
            self.W = self.add_weight(
                name="att_weight",
                shape=(input_shape[-1], 1),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.b = self.add_weight(
                name="att_bias",
                shape=(input_shape[1], 1),
                initializer="zeros",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, x):
            e = K.tanh(K.dot(x, self.W) + self.b)
            a = K.softmax(e, axis=1)
            self.last_attention = a
            return K.sum(x * a, axis=1)

        def get_config(self):
            return super().get_config()


@st.cache_resource(show_spinner=False)
def load_model():
    if not TF_AVAILABLE:
        return None
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model weights (first run only)…"):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"Attention": Attention},
        compile=False,
    )
    return model


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict_with_uncertainty(model, image: Image.Image, n_passes: int = 20):
    img_batch = preprocess(image)
    scores = []
    for _ in range(n_passes):
        score = float(model(img_batch, training=True)[0][0])
        scores.append(score)
    scores = np.array(scores)
    return float(scores.mean()), float(scores.std()), scores


def predict_image(model, image: Image.Image):
    img_batch = preprocess(image)
    return float(model.predict(img_batch, verbose=0)[0][0])


def make_gradcam_heatmap(model, image: Image.Image, last_conv_layer_name=None):
    if not TF_AVAILABLE:
        return None
    import cv2

    img_batch = preprocess(image)

    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            for layer in model.layers:
                if hasattr(layer, "layers"):
                    for sub in reversed(layer.layers):
                        if isinstance(sub, tf.keras.layers.Conv2D):
                            last_conv_layer_name = sub.name
                            break
                if last_conv_layer_name:
                    break

    if last_conv_layer_name is None:
        return None

    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )
    except Exception:
        return None

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    orig = np.array(image.convert("RGB").resize(IMG_SIZE))
    heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(orig, 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(superimposed)


RISK_LEVELS = [
    (0.25, "🟢 Low Risk",      "#22c55e", "Findings suggest benign tissue. Recommend routine follow-up in 12 months."),
    (0.50, "🟡 Borderline",    "#eab308", "Borderline findings. Recommend repeat imaging in 3–6 months."),
    (0.75, "🟠 Moderate Risk", "#f97316", "Moderate probability of malignancy. Specialist consultation advised within 2 weeks."),
    (1.01, "🔴 High Risk",     "#ef4444", "High probability of malignancy. Immediate oncologist referral strongly advised."),
]

def interpret(score: float):
    for threshold, label, color, rec in RISK_LEVELS:
        if score < threshold:
            return label, color, rec
    return RISK_LEVELS[-1][1], RISK_LEVELS[-1][2], RISK_LEVELS[-1][3]
