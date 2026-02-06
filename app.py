import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import urllib.request
from PIL import Image

# ======================================================
# 1. CORE AI ENGINE
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

def get_gradcam(img_array, model, original_image):
    target_layer = next((l for l in reversed(model.layers) if len(l.output.shape) == 4), None)
    if not target_layer: return None
    grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(img_array)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_outs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)).numpy()
    heatmap = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(np.array(original_image), 0.6, heatmap, 0.4, 0)

# ======================================================
# 2. UI CONFIGURATION
# ======================================================
st.set_page_config(page_title="OncoVision AI", layout="wide")

MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
MODEL_PATH = "cnn_bilstm_attention_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={"Attention": Attention}, compile=False)

# Header
st.title("🔬 Clinical Breast Cancer Classifier")
st.caption("Hybrid CNN-BiLSTM-Attention Architecture for Histopathology Analysis")

# Main Layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Data Upload")
    uploaded_file = st.file_uploader("Upload a biopsy slide patch (JPG/PNG)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Input Sample", use_container_width=True)

with col_right:
    st.subheader("Diagnostic Output")
    if uploaded_file:
        if st.button("🚀 Run Analysis", use_container_width=True):
            with st.spinner("Processing neural layers..."):
                model = load_model()
                prep = np.expand_dims(np.array(img.resize((96, 96))) / 255.0, axis=0)
                
                # Inference
                score = float(model.predict(prep, verbose=0)[0][0])
                label = "Malignant" if score > 0.5 else "Benign"
                
                # 1. Interactive Metrics
                m1, m2 = st.columns(2)
                m1.metric("Classification", label, delta="Warning" if label == "Malignant" else "Normal", delta_color="inverse")
                m2.metric("Probability", f"{score*100:.1f}%")

                # 2. Visual Probability Bar
                st.write("**Confidence Level Indicator:**")
                st.progress(score)
                st.caption(f"0% (Benign) {' ' * 40} 50% (Threshold) {' ' * 40} 100% (Malignant)")

                # 3. Smart Validation Warning (Non-blocking)
                hsv_sat = np.mean(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)[:,:,1])
                if hsv_sat < 20:
                    st.warning("⚠️ Image metadata suggests this may not be a standard H&E stained slide.")

                # 4. Technical Expanders
                with st.expander("🔍 View AI Interpretability (Grad-CAM)"):
                    overlay = get_gradcam(prep, model, img)
                    if overlay is not None:
                        st.image(overlay, use_container_width=True, caption="Heatmap highlighting high-risk cellular regions")
                        
                    
                with st.expander("🛠 Architecture Details"):
                    st.write("Current analysis uses a CNN backbone for spatial features, Bi-LSTM for sequence context, and a Global Attention mechanism to weigh diagnostic pixels.")
    else:
        st.info("Awaiting input file to begin diagnostic sequence.")

st.divider()
st.caption("Confidentiality Notice: This tool is for research support only. Final diagnosis must be confirmed by a board-certified pathologist.")









