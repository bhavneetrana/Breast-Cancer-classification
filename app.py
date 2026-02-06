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
# 1. CUSTOM ATTENTION LAYER (Fixed for Loading)
# ======================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape[-1] is the feature dimension
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = K.sum(x * a, axis=1)
        return output

    def get_config(self):
        return super(Attention, self).get_config()

# ======================================================
# 2. PAGE CONFIG & UI STYLING
# ======================================================
st.set_page_config(page_title="OncoVision AI", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .status-card { padding: 20px; border-radius: 10px; text-align: center; color: white; margin-bottom: 20px; }
    .malignant { background: linear-gradient(135deg, #ff4b2b, #ff416c); }
    .benign { background: linear-gradient(135deg, #56ab2f, #a8e063); }
    </style>
""", unsafe_allow_html=True)

# ======================================================
# 3. CORE LOGIC (MODEL & GRAD-CAM)
# ======================================================
MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
MODEL_PATH = "cnn_bilstm_attention_model.h5"

@st.cache_resource
def load_model_instance():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={"Attention": Attention}, 
        compile=False
    )

def get_gradcam_overlay(img_array, model, original_image):
    # Find last conv layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            last_conv_layer_name = layer.name
            break
    
    # Create sub-model for Grad-CAM
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize and Overlay
    heatmap_resized = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    img_np = np.array(original_image)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
    return overlay

# ======================================================
# 4. DASHBOARD UI
# ======================================================
st.title("🔬 OncoVision: Breast Cancer Diagnostic AI")
st.write("Professional-grade histopathology analysis using Deep Learning.")

tab1, tab2 = st.tabs(["Analysis Workspace", "Analysis History"])

if 'history' not in st.session_state:
    st.session_state.history = []

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Upload Slide")
        uploaded_file = st.file_uploader("Upload a histopathology patch (JPG/PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Tissue Sample", use_container_width=True)

    with col2:
        st.subheader("AI Prediction")
        if uploaded_file and st.button("Analyze Sample"):
            model = load_model_instance()
            
            # Prep Image
            img_resized = image.resize((96, 96))
            img_arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
            
            # Predict
            prediction = model.predict(img_arr, verbose=0)[0][0]
            risk_score = float(prediction * 100)
            label = "Malignant" if risk_score > 50 else "Benign"
            
            # UI Feedback
            style_class = "malignant" if label == "Malignant" else "benign"
            st.markdown(f"""
                <div class="status-card {style_class}">
                    <h2 style="color:white;">{label.upper()}</h2>
                    <p style="font-size:1.2em; color:white;">Confidence: {risk_score:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Grad-CAM
            st.write("### Interpretability (Grad-CAM)")
            
            overlay_img = get_gradcam_overlay(img_arr, model, image)
            st.image(overlay_img, caption="Red areas indicate high-risk features identified by AI", use_container_width=True)
            
            # Add to history
            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Diagnosis": label,
                "Risk %": round(risk_score, 2)
            })

with tab2:
    st.subheader("Session Log")
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
    else:
        st.write("No analyses performed in this session.")

st.sidebar.markdown("---")
st.sidebar.caption("Educational Tool - Not for Clinical Diagnosis.")
