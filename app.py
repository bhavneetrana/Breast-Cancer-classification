import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Conv2D
import os
import urllib.request
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import cv2

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Breast Cancer Classification Using Deep Learning",
    page_icon="🔬",
    layout="wide"
)

# ======================================================
# IMAGE VALIDATION (INPUT SUITABILITY)
# ======================================================
def is_valid_histopathology_image(img: Image.Image) -> bool:
    img = img.resize((96, 96))
    arr = np.array(img)

    # Reject flat images
    if np.var(arr) < 80:
        return False

    # Reject extremely dark / bright
    brightness = np.mean(arr)
    if brightness < 30 or brightness > 225:
        return False

    return True

# ======================================================
# GRAD-CAM HELPERS (ROBUST)
# ======================================================
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found for Grad-CAM.")

def make_gradcam_heatmap(img_array, model):
    last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def overlay_gradcam(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, image.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    image = np.array(image)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

# ======================================================
# HERO
# ======================================================
st.markdown("""
<div style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
padding:40px;border-radius:20px;margin-bottom:30px;">
<h1 style="color:white;">🔬 Breast Cancer Classification</h1>
<p style="color:#e0e0e0;">
Educational AI demo using histopathology images
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
col1, col2 = st.columns(2, gap="large")

with col1:
    uploaded_file = st.file_uploader(
        "Upload Histopathology Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

with col2:
    if uploaded_file and st.button("🚀 Analyze Image", use_container_width=True):

        # ---------- INPUT VALIDATION ----------
        if not is_valid_histopathology_image(image):
            st.error("🚫 Invalid image for histopathology analysis.")
            st.info("Upload a microscopic histopathology image only.")
            st.stop()

        # ---------- MODEL PREDICTION ----------
        model = load_model()
        img = image.resize((96, 96))
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)

        prob = float(model.predict(arr, verbose=0)[0][0])
        risk = prob * 100

        # ---------- OOD REJECTION ----------
        if 35 <= risk <= 65 or risk < 5 or risk > 95:
            st.error("🚫 Image does not resemble histopathology tissue.")
            st.stop()

        label = "Malignant" if risk > 50 else "Benign"

        st.markdown("### 🔍 Prediction Confidence")
        st.progress(int(risk))
        st.metric("Cancer Risk", f"{risk:.1f}%")

        if label == "Malignant":
            st.error("🔴 Malignant Tumor Detected")
        else:
            st.success("🟢 Benign / Non-Cancerous")

        # ---------- GRAD-CAM ----------
        st.markdown("### 🔥 Grad-CAM Heatmap")
        heatmap = make_gradcam_heatmap(arr, model)
        gradcam_img = overlay_gradcam(img, heatmap)
        st.image(
            gradcam_img,
            caption="Regions influencing the prediction",
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
            c.drawString(2*cm, 26*cm, f"Result: {label}")
            c.drawString(2*cm, 25*cm, f"Confidence: {risk:.1f}%")
            c.drawString(
                2*cm, 23*cm,
                "Disclaimer: Educational use only. Not a medical diagnosis."
            )

            c.save()
            buffer.seek(0)

            st.download_button(
                "⬇️ Download PDF",
                buffer,
                "breast_cancer_ai_report.pdf",
                "application/pdf"
            )

# ======================================================
# HISTORY
# ======================================================
st.markdown("## 📚 Prediction History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    if st.button("🧹 Clear History"):
        st.session_state.history.clear()
        st.experimental_rerun()
else:
    st.caption("No predictions yet.")

# ======================================================
# DISCLAIMER
# ======================================================
st.sidebar.warning("⚠️ Educational demo only. Not a medical diagnosis tool.")




