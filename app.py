import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import urllib.request
from PIL import Image

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="OncoVision – Histopathology Patch Analysis",
    page_icon="🔬",
    layout="wide"
)

# ======================================================
# CUSTOM ATTENTION LAYER
# ======================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class Attention(tf.keras.layers.Layer):
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
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        return tf.keras.backend.sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()

# ======================================================
# SAFE GRAD-CAM (FAIL-GRACEFULLY)
# ======================================================
def find_last_conv_layer_recursive(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        if isinstance(layer, tf.keras.Model):
            try:
                return find_last_conv_layer_recursive(layer)
            except ValueError:
                pass
    raise ValueError("No Conv2D layer found.")

def generate_gradcam(img_batch, model, original_img):
    """
    Safe Grad-CAM for CNN + BiLSTM + Attention.
    Returns None if Grad-CAM cannot be computed.
    """
    try:
        conv_layer = find_last_conv_layer_recursive(model)

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.outputs[0]]
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_batch)
            loss = preds[:, 0]

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]

        heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + 1e-8

        heatmap = cv2.resize(heatmap.numpy(), original_img.size)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        return cv2.addWeighted(
            np.array(original_img), 0.6, heatmap, 0.4, 0
        )

    except Exception:
        return None

# ======================================================
# MODEL LOADING
# ======================================================
MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
MODEL_PATH = "cnn_bilstm_attention_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"Attention": Attention},
        compile=False
    )

# ======================================================
# UI HEADER
# ======================================================
st.title("🔬 OncoVision – Histopathology Patch Analyzer")
st.markdown("""
This system analyzes **individual 96×96 histopathology patches** using a  
**CNN + BiLSTM + Attention** model trained on the **PCam dataset**.

⚠️ **Educational & research use only – NOT a medical diagnosis**
""")
st.divider()

# ======================================================
# MAIN UI
# ======================================================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📸 Upload Patch Image")
    file = st.file_uploader(
        "Upload histopathology patch (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Patch", use_container_width=True)

with col2:
    st.subheader("🧠 Model Analysis")

    if file and st.button("🚀 Run Analysis", use_container_width=True):
        model = load_model()

        # --- PREPROCESSING (PCam-consistent) ---
        img_resized = image.resize((96, 96), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized).astype("float32") / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # --- INFERENCE ---
        score = float(model.predict(img_batch, verbose=0)[0][0])

        st.markdown("### 📊 Model Output")
        st.progress(score)

        # --- UNCERTAINTY-AWARE INTERPRETATION ---
        if score >= 0.85:
            st.error("🟥 Strong tumor-like patterns detected (high confidence)")
            confidence_note = "High confidence"
        elif score <= 0.15:
            st.success("🟩 No strong tumor-like patterns detected (high confidence)")
            confidence_note = "High confidence"
        else:
            st.warning("🟨 Uncertain prediction – expert review recommended")
            confidence_note = "Low confidence"

        st.caption(f"Raw model score: **{score:.3f}** | {confidence_note}")

        st.divider()

        # --- GRAD-CAM ---
        with st.expander("🔍 Visual Explanation (Grad-CAM)"):
            cam = generate_gradcam(img_batch, model, image)
            if cam is not None:
                st.image(
                    cam,
                    caption="Highlighted regions influencing the model decision",
                    use_container_width=True
                )
            else:
                st.info(
                    "Grad-CAM visualization is not available for this architecture/input."
                )

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("ℹ️ System Information")
st.sidebar.info("""
**Model**
- CNN + BiLSTM + Attention
- Trained on PatchCamelyon (PCam)

**Key Notes**
- Patch-level analysis only
- Output is NOT a diagnosis
- Designed for education & research
""")













