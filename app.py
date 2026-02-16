import streamlit as st
import tensorflow as tf
import numpy as np
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
# CUSTOM ATTENTION LAYER (STORES WEIGHTS SAFELY)
# ======================================================
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
        self.last_attention = a  # store attention weights
        return tf.keras.backend.sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()

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
# HELPER: FIND ATTENTION LAYER (EVEN IF NESTED)
# ======================================================
def find_attention_layer(model):
    for layer in model.layers:
        if isinstance(layer, Attention):
            return layer
        if isinstance(layer, tf.keras.Model):
            try:
                return find_attention_layer(layer)
            except:
                pass
    return None

# ======================================================
# UI HEADER
# ======================================================
st.title("🔬 OncoVision – Histopathology Patch Analyzer")
st.markdown("""
Patch-level breast tissue analysis using a  
**CNN + BiLSTM + Attention** architecture.

📌 Visual explanation is provided through **Attention Weights**
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
        st.image(image, use_container_width=True)

with col2:
    st.subheader("🧠 Model Analysis")

    if file and st.button("🚀 Run Analysis", use_container_width=True):

        model = load_model()

        # Preprocessing (PCam-consistent)
        img_resized = image.resize((96, 96), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized).astype("float32") / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Inference
        score = float(model.predict(img_batch, verbose=0)[0][0])

        st.markdown("### 📊 Model Output")
        st.progress(score)

        if score >= 0.85:
            st.error("🟥 Strong tumor-like patterns detected")
            confidence = "High confidence"
        elif score <= 0.15:
            st.success("🟩 No strong tumor-like patterns detected")
            confidence = "High confidence"
        else:
            st.warning("🟨 Uncertain prediction – expert review recommended")
            confidence = "Low confidence"

        st.caption(f"Raw model score: **{score:.3f}** | {confidence}")

        st.divider()

        # ======================================================
        # ATTENTION VISUALIZATION
        # ======================================================
        with st.expander("🔍 Visual Explanation (Attention Weights)"):

            att_layer = find_attention_layer(model)

            if att_layer and att_layer.last_attention is not None:
                attention_values = att_layer.last_attention.numpy().squeeze()

                # Normalize for better visualization
                attention_values = attention_values / (
                    np.max(attention_values) + 1e-8
                )

                st.line_chart(attention_values)
                st.caption(
                    "Higher peaks indicate sequence regions the model focused on most."
                )
            else:
                st.info("Attention weights not available for this prediction.")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("ℹ️ System Information")
st.sidebar.info("""
**Model Architecture**
- CNN
- Bidirectional LSTM
- Attention Mechanism

**Dataset**
- PatchCamelyon (PCam)

⚠️ Educational & research use only.
""")
















