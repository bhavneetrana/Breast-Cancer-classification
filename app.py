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
import cv2
import matplotlib.pyplot as plt

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Breast Cancer Classification Using Deep Learning",
    page_icon="🔬",
    layout="wide"
)

# ======================================================
# HELPER: IMAGE VALIDATION (INPUT SUITABILITY)
# ======================================================
def is_valid_histopathology_image(img: Image.Image) -> bool:
    img = img.resize((96, 96))
    arr = np.array(img)

    # Reject flat / blank images
    if np.var(arr) < 80:
        return False

    # Reject extremely dark or bright images
    brightness = np.mean(arr)
    if brightness < 30 or brightness > 225:
        return False

    return True

# ======================================================
# GRAD-CAM HELPERS
# ======================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, original_img.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.array(original_img)
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
    return overlay

# ======================================================
# HERO
# ======================================================
st.markdown("""
<div style="background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
padding:40px;border-radius:20px;margin-bottom:30px;">
<h1 style="color:white;">🔬 Breast Cancer Classification</h1>
<p style="color:#e0e0e0;font-size:16px;">
Educational AI demo for histopathology image analysis
</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# MODEL SETUP
# ======================================================
MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
MODEL_PATH = "cnn_bilstm_attention_model.h5"
LAST_CONV_LAYER = "conv2d_3"   # change only if your model uses a different name

if not os.path.exists(MODEL_PATH):
    try:
        with st.spinner("🔽 Downloading model (first-time setup)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception:
        st.error("❌ Model download failed.")
        st.stop()

# ======================================================
# CUSTOM ATTENTION LAYER
# ======================================================
class Attention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

# ======================================================
# LOAD MODEL (CACHED)
# ======================================================
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
# UI LAYOUT
# ======================================================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📸 Upload Histopathology Image")
    uploaded_file = st.file_uploader(
        "Upload JPG / PNG / JPEG",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.markdown("### 📊 Prediction Result")

    if uploaded_file and st.button("🚀 Analyze Image", use_container_width=True):

        # ---------- INPUT VALIDATION ----------
        if not is_valid_histopathology_image(image):
            st.error("🚫 This image is not appropriate for histopathology-based analysis.")
            st.info(
                "Please upload a valid microscopic histopathology image. "
                "Natural images, objects, or screenshots are not supported."
            )
            st.stop()

        # ---------- MODEL PREDICTION ----------
        model = load_model()
        img = image.resize((96, 96))
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)

        prob = float(model.predict(arr, verbose=0)[0][0])
        risk = prob * 100

        # ---------- OOD / UNKNOWN IMAGE REJECTION ----------
        if 35 <= risk <= 65 or risk < 5 or risk > 95:
            st.error("🚫 The uploaded image does not resemble histopathology tissue.")
            st.info(
                "The AI model is uncertain or overconfident, which typically occurs "
                "with non-medical images."
            )
            st.stop()

        label = "Malignant" if risk > 50 else "Benign"

        # ---------- CONFIDENCE BAR ----------
        st.markdown("#### 🔍 Prediction Confidence")
        st.progress(int(risk))
        st.metric("Cancer Risk", f"{risk:.1f}%")

        if label == "Malignant":
            st.error("🔴 Malignant Tumor Detected")
            st.warning("Consult a qualified oncologist immediately.")
        else:
            st.success("🟢 Benign / Non-Cancerous")
            st.info("Maintain a healthy lifestyle and regular screening.")

        # ---------- GRAD-CAM ----------
        st.markdown("### 🔥 Grad-CAM Heatmap (Model Attention)")
        heatmap = make_gradcam_heatmap(arr, model, LAST_CONV_LAYER)
        gradcam_img = overlay_gradcam(img, heatmap)
        st.image(
            gradcam_img,
            caption="Highlighted regions influencing the prediction",
            use_container_width=True
        )

        # ---------- SAVE HISTORY ----------
        st.session_state.history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_%": round(risk, 2),
            "label": label
        })

        # ---------- PDF REPORT ----------
        if st.button("📄 Download Report"):
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=A4)

            c.setFont("Helvetica-Bold", 16)
            c.drawString(2*cm, 28*cm, "Breast Cancer AI Diagnostic Report")

            c.setFont("Helvetica", 12)
            c.drawString(2*cm, 26*cm, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            c.drawString(2*cm, 25*cm, f"Result: {label}")
            c.drawString(2*cm, 24*cm, f"Risk Confidence: {risk:.1f}%")

            c.setFont("Helvetica-Oblique", 10)
            c.drawString(
                2*cm, 22*cm,
                "Disclaimer: Educational use only. Not a medical diagnosis."
            )

            c.save()
            buffer.seek(0)

            st.download_button(
                "⬇️ Download PDF Report",
                buffer,
                "breast_cancer_ai_report.pdf",
                "application/pdf"
            )

# ======================================================
# HISTORY SECTION
# ======================================================
st.markdown("## 📚 Prediction History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "⬇️ Download History (CSV)",
            df.to_csv(index=False).encode(),
            "prediction_history.csv",
            "text/csv"
        )
    with col_b:
        if st.button("🧹 Clear History"):
            st.session_state.history.clear()
            st.experimental_rerun()
else:
    st.caption("No predictions made yet.")

# ======================================================
# SIDEBAR DISCLAIMER
# ======================================================
st.sidebar.warning("⚠️ Educational demo only. Not a medical diagnosis tool.")


