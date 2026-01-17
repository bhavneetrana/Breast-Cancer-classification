import streamlit as st
import tensorflow as tf
import numpy as np
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
# HERO / INTRO SECTION
# ======================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 45px;
    border-radius: 20px;
    margin-bottom: 35px;
">
    <h1 style="color:white;">🔬 Breast Cancer Classification Using Deep Learning</h1>
    <p style="color:#e0e0e0; font-size:17px; line-height:1.6;">
        This AI-powered web application detects <b>Breast Cancer</b> from 
        <b>histopathology images</b> using a deep learning model based on
        <b>CNN, BiLSTM, and Attention mechanism</b>.
    </p>
    <p style="color:#cfd8dc;">
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
# CUSTOM ATTENTION LAYER (KERAS 3 SAFE)
# ======================================================
class Attention(Layer):
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
# MAIN APPLICATION UI
# ======================================================
st.markdown("## 🧠 AI Breast Cancer Diagnostic System")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📸 Upload Histopathology Image")
    uploaded_file = st.file_uploader(
        "Upload JPG or PNG image (96×96)",
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

        if st.button("🚀 Analyze Image", use_container_width=True):
            model = load_trained_model()
            prediction = model.predict(img_array, verbose=0)[0][0]
            risk_pct = float(prediction * 100)

            color = "#4CAF50" if risk_pct < 30 else "#FF9800" if risk_pct < 70 else "#F44336"

            st.markdown(f"""
            <div style="
                background-color:#111827;
                padding:30px;
                border-radius:16px;
                text-align:center;
                border:1px solid #1f2937;
            ">
                <h2 style="color:{color}; margin-bottom:10px;">
                    Cancer Risk: {risk_pct:.1f}%
                </h2>
            </div>
            """, unsafe_allow_html=True)

            if risk_pct > 50:
                st.error("### 🔴 Malignant Tumor Detected")
            else:
                st.success("### 🟢 Benign / Non-Cancerous")
    else:
        st.info("Upload a medical image to begin analysis.")

# ======================================================
# FOOTER / ABOUT SECTION
# ======================================================
st.markdown("---")

st.markdown("""
<div style="
    background: linear-gradient(180deg, #0e1117, #111827);
    padding:40px;
    border-radius:22px;
    border:1px solid #262730;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.35);
">

    <!-- ABOUT PROJECT -->
    <h2 style="color:#4CAF50; margin-bottom:15px;">📌 About This Project</h2>
    <p style="font-size:16px; line-height:1.7; color:#d1d5db;">
        This project focuses on applying <b>Artificial Intelligence and Deep Learning</b>
        techniques to the field of <b>medical image analysis</b>.
        The system analyzes histopathology images to assist in the detection of
        <b>Breast Cancer</b> by classifying samples as <b>Benign</b> or <b>Malignant</b>.
    </p>

    <p style="font-size:16px; line-height:1.7; color:#d1d5db;">
        The application is built as an interactive web interface using
        <b>Streamlit</b> and a trained deep learning model, making it suitable for
        academic demonstrations, research exploration, and portfolio presentation.
    </p>

    <div style="height:1px; background:#262730; margin:30px 0;"></div>

    <!-- DEVELOPER DETAILS -->
    <h2 style="color:#FF9800; margin-bottom:15px;">👨‍💻 Developer Details</h2>
    <ul style="font-size:15px; line-height:1.9; color:#e5e7eb; list-style:none; padding-left:0;">
        <li>👤 <b>Name:</b> Bhavneet Rana</li>
        <li>🎓 <b>Role:</b> Student | AI & Machine Learning Enthusiast</li>
        <li>💡 <b>Interests:</b> Deep Learning, Computer Vision, Healthcare AI</li>
        <li>🛠️ <b>Skills:</b> Python, TensorFlow, Keras, Streamlit</li>
        <li>📘 <b>Project Type:</b> Academic / Research</li>
    </ul>

    <p style="font-size:14px; color:#9e9e9e; margin-top:28px;">
        ⚠️ <b>Disclaimer:</b> This application is intended for educational and research
        purposes only and should not be used as a substitute for professional
        medical diagnosis.
    </p>

</div>
""", unsafe_allow_html=True)

st.sidebar.warning("⚠️ Educational use only. Not a medical diagnosis tool.")




