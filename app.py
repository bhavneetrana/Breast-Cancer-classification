import streamlit as st
import tensorflow as tf
import numpy as np
import os
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="OncoVision – Histopathology Patch Analysis",
    page_icon="🔬",
    layout="wide"
)

# ======================================================
# CUSTOM ATTENTION LAYER (WITH WEIGHT ACCESS)
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
        self.last_attention = a  # <-- STORE ATTENTION WEIGHTS
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
# ATTENTION VISUALIZATION
# ======================================================
def plot_attention_weights(att_weights):
    """
    Plots attention distribution over sequence steps
    """
    att = att_weights.squeeze()

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(att, color="#ff9800", linewidth=2)
    ax.set_title("Attention Weight Distribution")
    ax.set_xlabel("Sequence Step")
    ax.set_ylabel("Attention Weight")
    ax.grid(alpha=0.3)

    return fig

# ======================================================
# UI HEADER
# ======================================================
st.title("🔬 OncoVision – Histopathology Patch Analyzer")
st.markdown("""
This system performs **patch-level analysis** using a  
**CNN + BiLSTM + Attention** architecture.

📌 Visual explanation is provided via **Attention Weights**  
(not Grad-CAM, which is unsuitable for this architecture).
""")
st.divider()

# ======================================================
# UI MAIN
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

        # --- PREPROCESSING (PCam-consistent) ---
        img_resized = image.resize((96, 96), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized).astype("float32") / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # --- INFERENCE ---
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

        # --- ATTENTION VISUALIZATION ---
        with st.expander("🔍 Visual Explanation (Attention Weights)"):
            # Find Attention layer
            att_layer = None
            for layer in model.layers:
                if isinstance(layer, Attention):
                    att_layer = layer
                    break

            if att_layer and att_layer.last_attention is not None:
                fig = plot_attention_weights(att_layer.last_attention.numpy())
                st.pyplot(fig)
                st.caption(
                    "Peaks indicate sequence regions the model focused on most."
                )
            else:
                st.info("Attention weights not available for this input.")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("ℹ️ System Information")
st.sidebar.info("""
**Model**
- CNN + BiLSTM + Attention
- Trained on PCam (PatchCamelyon)

**Explainability**
- Attention visualization (primary)
- Grad-CAM intentionally disabled

⚠️ Educational & research use only
""")














