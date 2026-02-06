import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import urllib.request
from PIL import Image

# ======================================================
# 1. CORE ENGINE (CNN + Bi-LSTM + Attention)
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
    target_layer = None
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            target_layer = layer
            break
    if not target_layer: return None

    grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(img_array)
        loss = preds[:, 0]
    
    grads = tape.gradient(loss, conv_outs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8))
    
    heatmap_np = cv2.resize(heatmap.numpy(), (original_image.size[0], original_image.size[1]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_np), cv2.COLORMAP_JET)
    return cv2.addWeighted(np.array(original_image), 0.6, heatmap_color, 0.4, 0)

# ======================================================
# 2. SIMPLE UI LAYOUT
# ======================================================
st.set_page_config(page_title="OncoVision AI", layout="centered")

MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
MODEL_PATH = "cnn_bilstm_attention_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={"Attention": Attention}, compile=False)

st.title("Breast Cancer Classification System")
st.text("Clinical Decision Support Tool - Version 1.0")
st.divider()

# Input Section
uploaded_file = st.file_uploader("Upload Histopathology Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    # Simple side-by-side display
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Patient Sample", use_container_width=True)
    
    with col2:
        if st.button("Start AI Prediction"):
            model = load_model()
            
            # Basic Image Prep
            prep = np.expand_dims(np.array(img.resize((96, 96))) / 255.0, axis=0)
            
            # Predict
            score = model.predict(prep, verbose=0)[0][0]
            label = "MALIGNANT" if score > 0.5 else "BENIGN"
            
            # Standard Medical Output
            st.subheader(f"Result: {label}")
            st.write(f"Confidence Score: {score*100:.2f}%")
            
            # Soft Validation Warning
            hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
            if np.mean(hsv[:,:,1]) < 25:
                st.warning("Notice: Image color profile deviates from standard H&E staining.")

            st.divider()
            
            # Grad-CAM Display
            overlay = get_gradcam(prep, model, img)
            if overlay is not None:
                st.image(overlay, caption="Grad-CAM Visualization", use_container_width=True)
                

else:
    st.write("Please upload a tissue patch to begin analysis.")

st.sidebar.markdown("### System Info")
st.sidebar.text("Architecture: CNN-BiLSTM")
st.sidebar.text("Layer: Global Attention")







