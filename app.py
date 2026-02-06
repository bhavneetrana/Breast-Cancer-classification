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
# 2. IMAGE VALIDATION (The "Wrong Image" Fix)
# ======================================================
def is_valid_medical_slide(image):
    """
    Checks if the image is likely a histopathology slide by analyzing 
    color saturation and texture typical of H&E staining.
    """
    img_np = np.array(image.convert("RGB"))
    
    # 1. Check for Colorfulness (Slides are usually Pink/Purple/Blue)
    # Convert to HSV and check saturation
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    avg_saturation = np.mean(hsv[:,:,1])
    
    # 2. Check for Texture (Laplacian Variance)
    # Real slides have complex cellular structures. Flat images/noise don't.
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Heuristic thresholds (Adjustable)
    if avg_saturation < 20: # Too gray/monochrome
        return False, "Image lacks the color profile of a stained slide."
    if variance < 100: # Too blurry or flat
        return False, "Image lacks the cellular texture required for analysis."
        
    return True, "Valid"

# ======================================================
# 3. GRAD-CAM (KERAS 3 COMPATIBLE)
# ======================================================

def get_gradcam_overlay(img_array, model, original_image):
    target_layer = None
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                target_layer = layer
                break
        except: continue
    
    if target_layer is None: return None

    grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    
    heatmap_resized = cv2.resize(heatmap.numpy(), (original_image.size[0], original_image.size[1]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    return cv2.addWeighted(np.array(original_image), 0.6, heatmap_color, 0.4, 0)

# ======================================================
# 4. APP UI
# ======================================================
st.set_page_config(page_title="OncoVision Pro", layout="wide")

st.markdown("""
    <style>
    .report-card { padding: 20px; border-radius: 12px; margin-bottom: 20px; color: white; text-align: center; }
    .malignant { background: linear-gradient(135deg, #ed213a, #93291e); }
    .benign { background: linear-gradient(135deg, #11998e, #38ef7d); }
    </style>
""", unsafe_allow_html=True)

MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
MODEL_PATH = "cnn_bilstm_attention_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={"Attention": Attention}, compile=False)

st.title("🔬 OncoVision: Medical Image Integrity Analysis")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Data Input")
    file = st.file_uploader("Upload histopathology patch", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, use_container_width=True)

with col2:
    st.subheader("2. AI Diagnosis")
    if file and st.button("🚀 Execute Analysis", use_container_width=True):
        # STEP 1: VALIDATE IMAGE TYPE
        is_valid, reason = is_valid_medical_slide(img)
        
        if not is_valid:
            st.error(f"🛑 **Invalid Image Detected!**")
            st.warning(f"Reason: {reason}")
            st.info("Please upload a microscopic biopsy slide (H&E stained). General photos cannot be processed.")
        else:
            # STEP 2: PROCEED TO PREDICTION
            model = load_model()
            arr = np.expand_dims(np.array(img.resize((96, 96))) / 255.0, axis=0)
            
            prob = model.predict(arr, verbose=0)[0][0]
            risk = prob * 100
            label = "Malignant" if risk > 50 else "Benign"
            
            # Display Result
            cls = "malignant" if label == "Malignant" else "benign"
            st.markdown(f'<div class="report-card {cls}"><h2>{label.upper()}</h2><h3>Risk: {risk:.2f}%</h3></div>', unsafe_allow_html=True)
            
            # Grad-CAM
            overlay = get_gradcam_overlay(arr, model, img)
            if overlay is not None:
                st.write("### AI Feature Mapping")
                st.image(overlay, use_container_width=True)




