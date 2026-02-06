import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Conv2D, InputLayer
import os
import urllib.request
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import cv2

# ======================================================
# 1. CUSTOM ATTENTION LAYER
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
# 2. UPDATED GRAD-CAM FUNCTION (The Fix)
# ======================================================
def get_gradcam_overlay(img_array, model, original_image):
    # Search for the last layer that has 4D output (Height, Width, Channels)
    # This is more reliable than searching by class type alone
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer_name = layer.name
            break
    
    if not last_conv_layer_name:
        # Fallback: if no 4D layer found, we can't do Grad-CAM
        return np.array(original_image)

    # Build the Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # We target the specific prediction score
        class_channel = preds[:, 0]

    # Calculate gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the gradient importance
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Overlay logic
    heatmap_img = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
    heatmap_img = np.uint8(255 * heatmap_img)
    heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
    
    img_np = np.array(original_image)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
    return overlay

# ======================================================
# 3. APP LOGIC & UI
# ======================================================
st.set_page_config(page_title="OncoVision", layout="wide")

# UI Styling
st.markdown("""
    <style>
    .report-card { background-color: white; padding: 20px; border-radius: 15px; border-left: 10px solid #007bff; }
    .malignant-text { color: #d9534f; font-weight: bold; }
    .benign-text { color: #5cb85c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
MODEL_PATH = "cnn_bilstm_attention_model.h5"

@st.cache_resource
def load_app_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={"Attention": Attention}, compile=False)

st.title("🔬 Breast Cancer AI Analysis")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload biopsy image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Sample", use_container_width=True)

with col2:
    if uploaded_file and st.button("🚀 Run Analysis"):
        model = load_app_model()
        
        # Image processing
        img_prep = image.resize((96, 96))
        arr = np.expand_dims(np.array(img_prep) / 255.0, axis=0)
        
        # Prediction
        preds = model.predict(arr, verbose=0)
        risk = float(preds[0][0] * 100)
        label = "Malignant" if risk > 50 else "Benign"
        
        # Results Display
        st.markdown(f"""
            <div class="report-card">
                <h3>Result: <span class="{label.lower()}-text">{label}</span></h3>
                <p>Confidence Level: <b>{risk:.2f}%</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        # Grad-CAM Visualization
        
        st.subheader("Visual Explanation (Grad-CAM)")
        try:
            overlay = get_gradcam_overlay(arr, model, image)
            st.image(overlay, caption="Heatmap highlighting high-risk regions", use_container_width=True)
        except Exception as e:
            st.warning("Visual heatmap generation skipped for this specific model architecture.")

        # History
        if 'log' not in st.session_state: st.session_state.log = []
        st.session_state.log.append({"Time": datetime.now().strftime("%H:%M"), "Result": label, "Risk": f"{risk:.1f}%"})

if 'log' in st.session_state and st.session_state.log:
    with st.expander("View Analysis History"):
        st.table(pd.DataFrame(st.session_state.log))

