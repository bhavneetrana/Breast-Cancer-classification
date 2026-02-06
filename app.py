import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import urllib.request
from PIL import Image
from datetime import datetime

# ======================================================
# 1. MODEL UTILITIES & ATTENTION
# ======================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        return tf.keras.backend.sum(x * a, axis=1)
    def get_config(self):
        return super(Attention, self).get_config()

# ======================================================
# 2. THE GATEKEEPER: MEDICAL IMAGE VALIDATION
# ======================================================
def is_valid_biopsy(image):
    """Checks if the image matches the profile of a stained medical slide."""
    img_np = np.array(image.convert("RGB"))
    
    # A. Texture Analysis (Tissue is complex, objects/noise are not)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # B. Color Profile (H&E Stains are specific to Pink/Purple)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    avg_sat = np.mean(hsv[:,:,1])
    
    # Logic: Real histopathology has high texture and specific saturation
    if laplacian_var < 100: # Rejects flat colors, gradients, or blurry photos
        return False, "Insufficient cellular texture detected."
    if avg_sat < 25: # Rejects gray/dull non-medical photos
        return False, "Color profile does not match H&E staining."
        
    return True, "Success"

# ======================================================
# 3. INTERPRETABILITY (GRAD-CAM)
# ======================================================

def get_gradcam(img_array, model, original_image):
    target_layer = next((l for l in reversed(model.layers) if len(l.output.shape) == 4), None)
    if not target_layer: return None

    grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(img_array)
        loss = preds[:, 0]
    
    grads = tape.gradient(loss, conv_outs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outs[0]), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    
    heatmap = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(np.array(original_image), 0.6, heatmap, 0.4, 0)

# ======================================================
# 4. MODERN UI CONFIGURATION
# ======================================================
st.set_page_config(page_title="OncoVision AI", layout="wide", page_icon="🔬")

# Professional CSS Injection
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e0e0e0; }
    .main { background-color: #ffffff; }
    .diag-card { padding: 2rem; border-radius: 1rem; color: white; margin-bottom: 2rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
    .malig { background: linear-gradient(135deg, #e53935, #b71c1c); border-left: 8px solid #7f0000; }
    .benig { background: linear-gradient(135deg, #43a047, #1b5e20); border-left: 8px solid #003300; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Model Loading
MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
MODEL_PATH = "cnn_bilstm_attention_model.h5"

@st.cache_resource
def load_app_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={"Attention": Attention}, compile=False)

# ======================================================
# 5. MAIN EXECUTION FLOW
# ======================================================
st.sidebar.title("🧬 Control Panel")
st.sidebar.info("Upload a microscopic slide patch (96x96 px recommended) for AI analysis.")

uploaded_file = st.sidebar.file_uploader("Choose Tissue Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Input Visualization")
        st.image(img, use_container_width=True, caption="Microscopic Sample")
        
    with col2:
        st.subheader("Diagnostic Engine")
        if st.button("🚀 Analyze Biopsy"):
            # Stage 1: Security Check
            is_valid, reason = is_valid_biopsy(img)
            
            if not is_valid:
                st.error("🛑 **Access Denied: Non-Medical Data**")
                st.warning(f"Reason: {reason}")
                st.info("The AI system is calibrated only for stained histopathology. Standard photos cannot be analyzed.")
            else:
                # Stage 2: Inference
                model = load_app_model()
                prep = np.expand_dims(np.array(img.resize((96, 96))) / 255.0, axis=0)
                
                score = model.predict(prep, verbose=0)[0][0]
                label = "Malignant" if score > 0.5 else "Benign"
                color_class = "malig" if label == "Malignant" else "benig"
                
                # Professional Result Card
                st.markdown(f"""
                    <div class="diag-card {color_class}">
                        <h1 style='margin:0;'>{label.upper()}</h1>
                        <p style='font-size:1.2rem; opacity:0.9;'>Malignancy Confidence: {score*100:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Explainability
                
                st.write("### AI Interpretability (Grad-CAM)")
                overlay = get_gradcam(prep, model, img)
                if overlay is not None:
                    st.image(overlay, use_container_width=True, caption="Heatmap: Regions influencing the diagnosis")

else:
    st.title("🔬 OncoVision Clinical Dashboard")
    st.write("Welcome. Please upload a biopsy patch in the sidebar to begin automated screening.")
    st.image("https://img.freepik.com/free-vector/medical-science-banner-with-dna-structure_1017-23190.jpg", use_container_width=True)






