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
    page_title="Breast Cancer AI Diagnostic Suite",
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
        self.last_attention = a
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
# FIND ATTENTION LAYER
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
# HEADER
# ======================================================
st.title("🔬 Breast Cancer AI Diagnostic Suite")
st.markdown("""
AI-powered histopathology patch analysis using  
**CNN + BiLSTM + Attention Architecture**
""")
st.divider()

# ======================================================
# CLINICAL 3-COLUMN DASHBOARD
# ======================================================
left, center, right = st.columns([1, 2, 1], gap="large")

# ---------------- LEFT PANEL ----------------
with left:
    st.markdown("## 👤 Patient Profile")

    st.image(
        "https://cdn-icons-png.flaticon.com/512/847/847969.png",
        width=110
    )

    st.write("**Name:** Mary Johnson")
    st.write("**Age:** 58")
    st.write("**Patient ID:** BC-1042")

    st.markdown("---")
    st.markdown("### 📂 Upload Histopathology Patch")

    file = st.file_uploader(
        "Upload Image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

# ---------------- CENTER PANEL ----------------
with center:
    st.markdown("## 🖼 Diagnostic Viewer")

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True)
    else:
        st.info("Upload a histopathology patch to begin analysis.")

# ---------------- RIGHT PANEL ----------------
with right:
    st.markdown("## 🧠 AI Analysis Panel")

    if file:
        if st.button("🚀 Run Diagnostic", use_container_width=True):

            model = load_model()

            # Preprocessing
            img_resized = image.resize((96, 96), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized).astype("float32") / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            # Inference
            score = float(model.predict(img_batch, verbose=0)[0][0])

            st.markdown("### 📊 Malignancy Probability")
            st.metric("Cancer Risk", f"{score*100:.1f}%")
            st.progress(score)

            # BI-RADS Logic
            if score >= 0.75:
                st.error("🟥 BI-RADS 4 – Suspicious Abnormality")
                confidence = "High confidence"
            elif score <= 0.25:
                st.success("🟩 BI-RADS 2 – Likely Benign")
                confidence = "High confidence"
            else:
                st.warning("🟨 BI-RADS 3 – Probably Benign")
                confidence = "Low confidence"

            st.caption(f"Raw model score: {score:.3f} | {confidence}")

            st.divider()

            # Simulated tissue metrics
            st.markdown("### 📈 Tissue Characteristics")
            st.progress(min(score + 0.2, 1.0))
            st.caption("Tissue Density")

            st.progress(score)
            st.caption("Mass Margin Irregularity")

            st.divider()

            # Action buttons
            colA, colB = st.columns(2)

            with colA:
                st.success("Approve")

            with colB:
                st.error("Request Biopsy")

            st.divider()

            # ======================================================
            # ATTENTION VISUALIZATION
            # ======================================================
            with st.expander("🔍 Visual Explanation (Attention Weights)"):

                att_layer = find_attention_layer(model)

                if att_layer and att_layer.last_attention is not None:
                    attention_values = att_layer.last_attention.numpy().squeeze()
                    attention_values = attention_values / (
                        np.max(attention_values) + 1e-8
                    )

                    st.line_chart(attention_values)
                    st.caption(
                        "Peaks indicate sequence regions the model focused on most."
                    )
                else:
                    st.info("Attention weights not available.")

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


















