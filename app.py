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
# HERO / PROJECT INTRO SECTION
# ======================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 40px;
    border-radius: 18px;
    margin-bottom: 30px;
">
    <h1 style="color:white;">🔬 Breast Cancer Classification Using Deep Learning</h1>
    <p style="color:#dcdcdc; font-size:17px;">
        This AI-powered web application detects <b>Breast Cancer</b> from 
        <b>histopathology images</b> using a deep learning model based on 
        <b>CNN, BiLSTM, and Attention mechanism</b>.
    </p>
    <p style="color:#b0bec5;">
        The system predicts whether the tumor is <b>Benign</b> or <b>Malignant</b>,
        supporting early diagnosis and medical research.
    </p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# DOWNLOAD MODEL FROM GITHUB RELEASE
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
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight("att_weight", (input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight("att_bias", (input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

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
# MAIN APPLICATION UI
# ======================================================
st.markdown("## 🧠 AI Breast Cancer Diagnostic System")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📸 Upload Histopathology Image")
    uploaded_file = st.file_uploader(
        "Upload JPG / PNG image (96×96)",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Medical Image", use_container_width=True)

with col2:
    st.markdown("### 📊 Prediction Result")

    if uploaded_file:
        img = image.resize((96, 96))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        if st.button("🚀 Analyze Image"):
            model = load_trained_model()
            pred = model.predict(img_array, verbose=0)[0][0]
            risk = float(pred * 100)

            color = "#4CAF50" if risk < 30 else "#FF9800" if risk < 70 else "#F44336"

            st.markdown(f"""
            <div style="
                background-color:#111827;
                padding:25px;
                border-radius:15px;
                text-align:center;
                border:1px solid #1f2937;
            ">
                <h2 style="color:{color};">Cancer Risk: {risk:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)

            if risk > 50:
                st.error("### 🔴 Malignant Tumor Detected")
            else:
                st.success("### 🟢 Benign / Non-Cancerous")
    else:
        st.info("Upload a medical image to begin analysis.")

# ======================================================
# FOOTER / ABOUT SECTION (BOTTOM)
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
    <p>
        This project demonstrates the application of <b>Deep Learning</b> in 
        <b>Medical Image Analysis</b> for Breast Cancer classification using
        a hybrid CNN–BiLSTM–Attention architecture.
    </p>

    <h3 style="color:#03A9F4;">🧠 Model Details</h3>
    <ul>
        <li><b>Architecture:</b> CNN + BiLSTM + Attention</li>
        <li><b>Base Model:</b> MobileNetV2</li>
        <li><b>Input Size:</b> 96 × 96 × 3</li>
        <li><b>Output:</b> Benign / Malignant</li>
        <li><b>Framework:</b> TensorFlow & Keras</li>
    </ul>

    <h3 style="color:#FF9800;">👨‍💻 Developer</h3>
    <ul>
        <li><b>Name:</b> Bhavneet Rana</li>
        <li><b>Role:</b> Student | AI & ML Enthusiast</li>
        <li><b>Skills:</b> Python, Deep Learning, TensorFlow, Computer Vision</li>
        <li><b>Project Type:</b> Academic / Research</li>
    </ul>

    <p style="font-size:14px; color:#9e9e9e;">
        ⚠️ <b>Disclaimer:</b> This application is for educational and research purposes only.
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.warning("⚠️ Educational use only. Not a medical diagnosis tool.")









