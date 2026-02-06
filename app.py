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
# PAGE CONFIG & STYLING
# ======================================================
st.set_page_config(
    page_title="OncoVision AI | Breast Cancer Analysis",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for a Professional Look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .malignant-bg { background: linear-gradient(135deg, #e53935, #b71c1c); }
    .benign-bg { background: linear-gradient(135deg, #43a047, #1b5e20); }
    .header-style {
        color: #1e3a8a;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# ======================================================
# MODEL & ATTENTION LAYER
# ======================================================
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(shape=(input_shape[1], 1), initializer="zeros", trainable=True)
        super().build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

@st.cache_resource
def load_trained_model():
    MODEL_PATH = "cnn_bilstm_attention_model.h5"
    MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={"Attention": Attention}, compile=False)

# ======================================================
# IMAGE PROCESSING
# ======================================================
def make_gradcam_heatmap(img_array, model):
    last_conv_layer_name = next(l.name for l in reversed(model.layers) if isinstance(l, Conv2D))
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

# ======================================================
# MAIN UI
# ======================================================

# Sidebar - Settings & Branding
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2862/2862356.png", width=100)
    st.title("OncoVision AI")
    st.markdown("---")
    st.info("🔬 **System Status:** Ready\n\nAI Diagnostic Support Tool v1.2")
    st.warning("⚠️ For Educational Use Only.")

# Hero Section
st.markdown('<h1 class="header-style">Breast Cancer Histopathology Dashboard</h1>', unsafe_allow_html=True)
st.write("Analyze microscopic tissue slides with Deep Learning and Grad-CAM interpretability.")

# Layout: Tabs for cleaner navigation
tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📜 History", "📖 Documentation"])

with tab1:
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("Slide Upload")
        uploaded_file = st.file_uploader("Drop biopsy image here", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original Slide Image", use_container_width=True)
    
    with col_output:
        st.subheader("Diagnostic Engine")
        if uploaded_file:
            if st.button("🚀 Analyze Tissue", use_container_width=True):
                model = load_trained_model()
                img_prep = image.resize((96, 96))
                arr = np.expand_dims(np.array(img_prep) / 255.0, axis=0)
                
                # Inference
                prob = float(model.predict(arr, verbose=0)[0][0])
                risk = prob * 100
                label = "Malignant" if risk > 50 else "Benign"
                
                # Result Card
                card_class = "malignant-bg" if label == "Malignant" else "benign-bg"
                st.markdown(f"""
                    <div class="prediction-card {card_class}">
                        <h2 style="color:white; margin:0;">{label.upper()}</h2>
                        <p style="color:white; opacity:0.8; margin:0;">Confidence Score: {risk:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Visual Analytics
                
                st.write("### AI Focus Areas (Grad-CAM)")
                heatmap = make_gradcam_heatmap(arr, model)
                overlay = overlay_gradcam(image, heatmap)
                st.image(overlay, caption="Heatmap highlighting high-risk cellular patterns", use_container_width=True)
                
                # Record History
                if "history" not in st.session_state: st.session_state.history = []
                st.session_state.history.append({"Date": datetime.now().strftime("%H:%M"), "Result": label, "Risk": f"{risk:.1f}%"})

                # Report Generation
                buffer = BytesIO()
                pdf = canvas.Canvas(buffer, pagesize=A4)
                pdf.drawString(100, 800, "ONCOVISION AI DIAGNOSTIC REPORT")
                pdf.drawString(100, 780, f"Result: {label} ({risk:.1f}%)")
                pdf.save()
                st.download_button("📥 Download Clinical Report", buffer.getvalue(), "report.pdf", "application/pdf", use_container_width=True)
        else:
            st.info("Upload an image in the left panel to begin analysis.")

with tab2:
    st.subheader("Previous Analyses")
    if "history" in st.session_state and st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        if st.button("Clear Logs"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("No session data found.")

with tab3:
    st.subheader("How it works")
    
    st.markdown("""
    This AI system utilizes a **CNN-BiLSTM-Attention** hybrid architecture:
    1. **Convolutional Layers (CNN):** Extract spatial features (cell shapes, density).
    2. **Bi-LSTM:** Analyzes the spatial sequence of features.
    3. **Attention Mechanism:** Focuses on specific "hotspots" in the tissue slide.
    4. **Grad-CAM:** Produces the heatmap to explain the AI's decision process.
    """)
