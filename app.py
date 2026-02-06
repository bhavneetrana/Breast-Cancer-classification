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
# 2. ADVANCED IMAGE VALIDATOR (The "Wrong Image" Fix)
# ======================================================
def validate_input_image(img):
    """
    Analyzes if the image is actually a histopathology slide.
    """
    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # A. Color Check: Most H&E slides are dominated by Pink/Purple/Blue
    # We check the Hue and Saturation ranges
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    avg_sat = np.mean(hsv[:,:,1])
    
    # B. Texture Check: Real tissue has high entropy (complexity)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # C. Sharpness/Edges: Reject pure solid colors or noisy garbage
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

    # Thresholds for 'Real Histopathology Slides'
    if avg_sat < 15:
        return False, "Low color saturation (Image is too gray/monochrome)."
    if laplacian_var < 80:
        return False, "Low texture detail (Image is too flat or blurry)."
    if edge_density < 0.01:
        return False, "Insufficient structural detail (Doesn't look like tissue cells)."
        
    return True, "Success"

# ======================================================
# 3. GRAD-CAM (COMPATIBILITY FIX)
# ======================================================

def get_gradcam_overlay(img_array, model, original_image):
    target_layer = None
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                target_layer = layer
                break
        except: continue
    
    if not target_layer: return None

    grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)).numpy()

    heatmap_resized = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(np.array(original_image), 0.6, heatmap_color, 0.4, 0)

# ======================================================
# 4. DASHBOARD UI
# ======================================================
st.set_page_config(page_title="OncoVision Diagnostic", layout="wide")

st.markdown("""
<style>
    .metric-container { background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .alert-box { padding: 20px; border-radius: 10px; color: white; text-align: center; font-weight: bold; }
    .malig { background: linear-gradient(to right, #cb2d3e, #ef473a); }
    .benig { background: linear-gradient(to right, #11998e, #38ef7d); }
</style>
""", unsafe_allow_html=True)

MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
MODEL_PATH = "cnn_bilstm_attention_model.h5"

@st.cache_resource
def load_app_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={"Attention": Attention}, compile=False)

st.title("🔬 Clinical Histopathology Analysis")
st.write("Ensuring image integrity before AI classification.")

col_a, col_b = st.columns(2, gap="large")

with col_a:
    uploaded_file = st.file_uploader("Upload Biopsy Slide Patch", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        raw_img = Image.open(uploaded_file).convert("RGB")
        st.image(raw_img, use_container_width=True, caption="Uploaded Image")

with col_b:
    if uploaded_file and st.button("🚀 Run AI Analysis", use_container_width=True):
        # --- VALIDATION STAGE ---
        valid, message = validate_input_image(raw_img)
        
        if not valid:
            st.error("🛑 **Analysis Aborted: Invalid Image Type**")
            st.warning(f"Technical Reason: {message}")
            st.info("The AI only accepts high-resolution H&E stained biopsy slides. Please do not upload photos of faces, objects, or non-medical images.")
        else:
            # --- PREDICTION STAGE ---
            with st.spinner("Processing medical data..."):
                model = load_app_model()
                img_prep = raw_img.resize((96, 96))
                arr = np.expand_dims(np.array(img_prep) / 255.0, axis=0)
                
                prediction = model.predict(arr, verbose=0)[0][0]
                risk = prediction * 100
                label = "Malignant" if risk > 50 else "Benign"
            
            # UI Output
            css_class = "malig" if label == "Malignant" else "benig"
            st.markdown(f'<div class="alert-box {css_class}"><h2>{label.upper()}</h2><h3>Malignancy Risk: {risk:.2f}%</h3></div>', unsafe_allow_html=True)
            
            # Explainability
            
            st.subheader("Structural Focus (Grad-CAM)")
            overlay = get_gradcam_overlay(arr, model, raw_img)
            if overlay is not None:
                st.image(overlay, use_container_width=True, caption="Heatmap highlighting diagnostic areas")





