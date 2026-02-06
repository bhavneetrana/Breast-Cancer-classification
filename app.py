import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
import os
import urllib.request
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

#helper function

def is_valid_histopathology_image(img: Image.Image) -> bool:
    """
    Basic heuristic check to reject non-histopathology images.
    This is NOT a medical decision, only input suitability validation.
    """
    img = img.resize((96, 96))
    arr = np.array(img)

    # 1. Variance check (reject very flat / uniform images)
    variance = np.var(arr)
    if variance < 200:   # empirically safe threshold
        return False

    # 2. Color distribution check (natural photos often dominate one channel)
    channel_means = np.mean(arr, axis=(0, 1))
    max_channel_ratio = np.max(channel_means) / (np.mean(channel_means) + 1e-6)
    if max_channel_ratio > 1.8:
        return False

    # 3. Extremely dark or bright images
    brightness = np.mean(arr)
    if brightness < 30 or brightness > 225:
        return False

    return True


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Breast Cancer Classification Using Deep Learning",
    page_icon="🔬",
    layout="wide"
)

# ======================================================
# HERO
# ======================================================
st.markdown("""
<div style="background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
padding:40px;border-radius:20px;">
<h1 style="color:white;">🔬 Breast Cancer Classification</h1>
<p style="color:#e0e0e0;">
Educational AI demo for histopathology image analysis.
</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# MODEL SETUP
# ======================================================
MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
MODEL_PATH = "cnn_bilstm_attention_model.h5"

if not os.path.exists(MODEL_PATH):
    try:
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception:
        st.error("Model download failed.")
        st.stop()

class Attention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="glorot_uniform")
        self.b = self.add_weight(shape=(input_shape[1], 1), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"Attention": Attention},
        compile=False
    )

# ======================================================
# SESSION STATE
# ======================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================================================
# UI
# ======================================================
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Histopathology Image", type=["jpg","png","jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

with col2:
   if uploaded_file and st.button("🚀 Analyze Image"):

    # ---------------- IMAGE VALIDATION ----------------
    if not is_valid_histopathology_image(image):
        st.error("🚫 This image is not appropriate for histopathology-based breast cancer analysis.")
        st.info(
            "Please upload a valid microscopic histopathology image. "
            "Natural images, selfies, objects, or unrelated scans are not supported."
        )
        st.stop()

        model = load_model()
        img = image.resize((96,96))
        arr = np.expand_dims(np.array(img)/255.0, axis=0)

        prob = float(model.predict(arr, verbose=0)[0][0])
        risk = prob * 100
        label = "Malignant" if risk > 50 else "Benign"

        st.markdown("### 🔍 Prediction Confidence")
        st.progress(int(risk))
        st.metric("Cancer Risk", f"{risk:.1f}%")

        if label == "Malignant":
            st.error("🔴 Malignant Tumor Detected")
            st.warning("Consult an oncologist immediately.")
        else:
            st.success("🟢 Benign / Non-Cancerous")
            st.info("Maintain healthy lifestyle & regular screening.")

        # Save history
        st.session_state.history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_%": round(risk,2),
            "label": label
        })

        # PDF
        if st.button("📄 Download Report"):
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=A4)
            c.drawString(2*cm, 28*cm, "Breast Cancer AI Report")
            c.drawString(2*cm, 26*cm, f"Result: {label}")
            c.drawString(2*cm, 25*cm, f"Confidence: {risk:.1f}%")
            c.drawString(2*cm, 23*cm, "Educational use only. Not a diagnosis.")
            c.save()
            buffer.seek(0)
            st.download_button(
                "Download PDF",
                buffer,
                "breast_cancer_report.pdf",
                "application/pdf"
            )

# ======================================================
# HISTORY
# ======================================================
st.markdown("## 📚 Prediction History")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False).encode(),
        "history.csv",
        "text/csv"
    )

    if st.button("🧹 Clear History"):
        st.session_state.history.clear()
        st.experimental_rerun()
else:
    st.caption("No predictions yet.")

st.sidebar.warning("⚠️ Educational demo only. Not medical advice.")

