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
import cv2

# ======================================================
# 1. CUSTOM ATTENTION LAYER (SERIALIZABLE)
# ======================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

    def get_config(self):
        return super(Attention, self).get_config()

# ======================================================
# 2. GRAD-CAM UTILITY (KERAS 3 COMPATIBLE)
# ======================================================


def get_gradcam_overlay(img_array, model, original_image):
    target_layer = None
    
    # Iterate backwards to find the last layer with a 4D output tensor
    for layer in reversed(model.layers):
        try:
            # Check the rank of the output tensor directly for Keras 3 compatibility
            if len(layer.output.shape) == 4:
                target_layer = layer
                break
        except (AttributeError, ValueError):
            continue
    
    if target_layer is None:
        return None

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[target_layer.output, model.output]
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
    heatmap = heatmap.numpy()

    heatmap_resized = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    img_np = np.array(original_image)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
    return overlay

# ======================================================
# 3. STREAMLIT UI & DASHBOARD
# ======================================================
st.set_page_config(page_title="OncoVision AI", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .result-card { padding: 25px; border-radius: 15px; text-align: center; color: white; margin-bottom: 20px; font-weight: bold; }
    .malignant-bg { background: linear-gradient(135deg, #e53935, #b71c1c); }
    .benign-bg { background: linear-gradient(135deg, #43a047, #1b5e20); }
    </style>
""", unsafe_allow_html=True)

MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
MODEL_PATH = "cnn_bilstm_attention_model.h5"

@st.cache_resource
def load_app_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading clinical model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={"Attention": Attention}, compile=False)

st.title("🔬 OncoVision: Advanced Breast Cancer AI")
st.write("Utilizing CNN, Bi-LSTM, and Global Attention for histopathology classification.")

tab1, tab2 = st.tabs(["Analysis", "Session History"])

if 'history' not in st.session_state:
    st.session_state.history = []

with tab1:
    col_u, col_r = st.columns([1, 1], gap="large")
    
    with col_u:
        st.subheader("Image Input")
        uploaded_file = st.file_uploader("Upload slide patch", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Biopsy Tissue Sample", use_container_width=True)

    with col_r:
        st.subheader("AI Diagnostics")
        if uploaded_file and st.button("🚀 Analyze Tissue", use_container_width=True):
            model = load_app_model()
            
            # Prepare image
            img_resized = image.resize((96, 96))
            arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
            
            # Prediction
            prediction = model.predict(arr, verbose=0)[0][0]
            risk = float(prediction * 100)
            label = "Malignant" if risk > 50 else "Benign"
            
            # Result Display
            bg = "malignant-bg" if label == "Malignant" else "benign-bg"
            st.markdown(f'<div class="result-card {bg}"><h2>{label.upper()}</h2><p>Malignancy Probability: {risk:.2f}%</p></div>', unsafe_allow_html=True)
            
            # Grad-CAM Display
            
            st.write("### Interpretability Map")
            overlay = get_gradcam_overlay(arr, model, image)
            if overlay is not None:
                st.image(overlay, caption="Grad-CAM: Regions driving the AI's decision", use_container_width=True)
            else:
                st.info("Feature map not available for this image scale.")
            
            # Log History
            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Diagnosis": label,
                "Risk": f"{risk:.1f}%"
            })

            # PDF Report
            buf = BytesIO()
            can = canvas.Canvas(buf, pagesize=A4)
            can.setFont("Helvetica-Bold", 16)
            can.drawString(100, 800, "ONCOVISION AI DIAGNOSTIC REPORT")
            can.setFont("Helvetica", 12)
            can.drawString(100, 770, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
            can.drawString(100, 750, f"Prediction: {label}")
            can.drawString(100, 730, f"Confidence: {risk:.2f}%")
            can.save()
            st.download_button("⬇️ Download PDF Report", buf.getvalue(), "report.pdf", "application/pdf")

with tab2:
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("No session records found.")

st.sidebar.markdown("---")
st.sidebar.warning("Educational demo only. Not a medical substitute.")



