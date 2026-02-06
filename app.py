import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization
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
# 2. ROBUST GRAD-CAM FOR HYBRID ARCHITECTURES
# ======================================================
def get_gradcam_overlay(img_array, model, original_image):
    """
    Finds the last 4D convolutional layer and computes the Grad-CAM heatmap.
    """
    # Identify the last layer with spatial dimensions (4D: Batch, H, W, C)
    target_layer = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            target_layer = layer
            break
    
    if target_layer is None:
        return None

    # Create a sub-model mapping input to the target layer and final prediction
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Target the prediction score (Sigmoid output)
        loss = predictions[:, 0]

    # Calculate gradients of the loss w.r.t the feature map
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the spatial feature map by the gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU and Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Overlay on original image
    heatmap_resized = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    img_np = np.array(original_image)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
    return overlay

# ======================================================
# 3. STREAMLIT UI & LOGIC
# ======================================================
st.set_page_config(page_title="OncoVision AI", page_icon="🔬", layout="wide")

# Styling
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .result-card { padding: 20px; border-radius: 15px; text-align: center; color: white; margin-bottom: 20px; font-weight: bold; }
    .malignant-bg { background: linear-gradient(135deg, #e53935, #b71c1c); }
    .benign-bg { background: linear-gradient(135deg, #43a047, #1b5e20); }
    </style>
""", unsafe_allow_html=True)

MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
MODEL_PATH = "cnn_bilstm_attention_model.h5"

@st.cache_resource
def load_app_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={"Attention": Attention}, compile=False)

# Main Interface
st.title("🔬 OncoVision: Breast Cancer Diagnostic AI")
st.write("Hybrid CNN-BiLSTM-Attention Model for Histopathology Analysis")

tab_analysis, tab_history = st.tabs(["🔍 Analysis Workspace", "📜 History Log"])

if 'history' not in st.session_state:
    st.session_state.history = []

with tab_analysis:
    col_upload, col_result = st.columns([1, 1], gap="large")
    
    with col_upload:
        st.subheader("Upload Tissue Patch")
        uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Biopsy Slide Patch", use_container_width=True)

    with col_result:
        st.subheader("AI Diagnostics")
        if uploaded_file and st.button("🚀 Analyze Sample", use_container_width=True):
            model = load_app_model()
            
            # Preprocessing
            img_resized = image.resize((96, 96))
            arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
            
            # Inference
            prediction = model.predict(arr, verbose=0)[0][0]
            risk = float(prediction * 100)
            label = "Malignant" if risk > 50 else "Benign"
            
            # UI Feedback
            bg_class = "malignant-bg" if label == "Malignant" else "benign-bg"
            st.markdown(f'<div class="result-card {bg_class}"><h2>{label.upper()}</h2><p>Risk Score: {risk:.2f}%</p></div>', unsafe_allow_html=True)
            
            # Grad-CAM Visualization
            
            st.write("### AI Focus Map (Grad-CAM)")
            overlay = get_gradcam_overlay(arr, model, image)
            
            if overlay is not None:
                st.image(overlay, caption="Heatmap highlighting high-risk cellular clusters", use_container_width=True)
            else:
                st.warning("Spatial layers not detected for Grad-CAM generation.")
            
            # History Tracking
            st.session_state.history.append({
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Diagnosis": label,
                "Confidence": f"{risk:.1f}%"
            })

            # Report Download
            buffer = BytesIO()
            p = canvas.Canvas(buffer, pagesize=A4)
            p.drawString(100, 800, f"OncoVision AI Report - {label}")
            p.drawString(100, 780, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
            p.drawString(100, 760, f"Malignancy Risk: {risk:.2f}%")
            p.save()
            st.download_button("📥 Download Report", buffer.getvalue(), "diagnostic_report.pdf", "application/pdf")

with tab_history:
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        if st.button("Clear Session History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No analyses performed in this session.")


