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
st.markdown("""
# 🔬 Breast Cancer Classification Using Deep Learning

This web application performs **breast cancer detection** using a **deep learning model**
based on **CNN, BiLSTM, and Attention mechanism**.

The system analyzes **histopathology images** and predicts whether the tumor is
**Benign or Malignant**, helping in early cancer diagnosis.

### 🔑 Keywords
Breast Cancer Detection, Deep Learning, CNN, BiLSTM, Attention Model,  
Medical Image Classification, TensorFlow, Streamlit, AI in Healthcare

**Author:** Bhavneet Rana  
**Tech Stack:** Python, TensorFlow, Deep Learning, Streamlit
""")

st.divider()

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







