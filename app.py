import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
import os
import urllib.request

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Breast Cancer Classification Using Deep Learning",
    page_icon="🔬",
    layout="wide"
)

# ======================================================
# SEO / PROJECT DESCRIPTION (VERY IMPORTANT)
# ======================================================
# ======================================================
# FOOTER / ABOUT SECTION
# ======================================================
st.markdown("---")

st.markdown("""
<div style="
    background-color:#0e1117;
    padding:30px;
    border-radius:15px;
    border:1px solid #262730;
">
    <h2 style="color:#4CAF50;">📌 About This Project</h2>
    <p style="font-size:16px;">
        This AI-based Breast Cancer Classification system uses a deep learning
        architecture combining <b>CNN, BiLSTM, and Attention mechanism</b>
        to analyze histopathology images and predict whether a tumor is
        <b>Benign or Malignant</b>.
    </p>

    <h3 style="color:#03A9F4;">🧠 Model Details</h3>
    <ul style="font-size:15px;">
        <li><b>Architecture:</b> CNN + BiLSTM + Attention</li>
        <li><b>Base Model:</b> MobileNetV2 (Feature Extraction)</li>
        <li><b>Input Image Size:</b> 96 × 96 × 3</li>
        <li><b>Output:</b> Binary Classification (Benign / Malignant)</li>
        <li><b>Framework:</b> TensorFlow & Keras</li>
    </ul>

    <h3 style="color:#FF9800;">👨‍💻 About the Developer</h3>
    <ul style="font-size:15px;">
        <li><b>Name:</b> Bhavneet Rana</li>
        <li><b>Role:</b> Student | AI & Machine Learning Enthusiast</li>
        <li><b>Skills:</b> Python, Deep Learning, TensorFlow, Computer Vision</li>
        <li><b>Project Type:</b> Academic & Research Project</li>
    </ul>

    <p style="font-size:14px; color:#9e9e9e; margin-top:20px;">
        ⚠️ <b>Disclaimer:</b> This application is for educational and research
        purposes only and should not be used as a substitute for professional
        medical diagnosis.
    </p>
</div>
""", unsafe_allow_html=True)


# ======================================================
# DOWNLOAD MODEL FROM GITHUB RELEASE (ONE TIME)
# ======================================================
MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
MODEL_PATH = "cnn_bilstm_attention_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Downloading AI model (first-time setup)..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ======================================================
# CUSTOM ATTENTION LAYER
# ======================================================
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

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
# LOAD TRAINED MODEL
# ======================================================
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"Attention": Attention},
        compile=False
    )

# ======================================================
# APP UI
# ======================================================
st.title("🧠 AI Breast Cancer Diagnostic System")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Upload Histopathology Image")
    uploaded_file = st.file_uploader(
        "Upload JPG or PNG image (96×96)",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Medical Image", use_container_width=True)

with col2:
    st.subheader("📊 Prediction Result")

    if uploaded_file:
        img_resized = image.resize((96, 96))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("🚀 Analyze Image"):
            model = load_trained_model()
            prediction = model.predict(img_array, verbose=0)[0][0]
            risk_pct = float(prediction * 100)

            color = "green" if risk_pct < 30 else "orange" if risk_pct < 70 else "red"

            st.markdown(f"""
            <div style="background-color:#f0f2f6;border-radius:10px;padding:20px;text-align:center;">
                <h3 style="color:{color};">Cancer Risk: {risk_pct:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            if risk_pct > 50:
                st.error("### Malignant Tumor Detected")
            else:
                st.success("### Benign / Non-Cancerous")

    else:
        st.info("Upload a medical image to begin analysis.")

# ======================================================
# DISCLAIMER
# ======================================================
st.sidebar.warning("⚠️ This application is for educational and research purposes only.")








