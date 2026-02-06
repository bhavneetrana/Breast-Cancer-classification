import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import urllib.request
from PIL import Image

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="OncoVision – Histopathology Patch Analysis",
    page_icon="🔬",
    layout="wide"
)

# ======================================================
# CUSTOM ATTENTION LAYER (REQUIRED FOR MODEL LOAD)
# ======================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        return tf.keras.backend.sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()

# ======================================================
# GRAD-CAM (ROBUST FOR NESTED CNNs)
# ======================================================
def find_last_conv_layer_recursive(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        if isinstance(layer, tf.keras.Model):
            try:
                return find_last_conv_layer_recursive(layer)
            except ValueError:
                pass
    raise ValueError("No Conv2D layer found.")

def generate_gradcam(img_batch, model, original_img):
    try:
        conv_layer = find_last_conv_layer_recursive(model)
    except ValueError:
        return None

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap.numpy(), original_img.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(np.array(original_img), 0.6, heatmap, 0.4, 0)

# ======================================================
# MODEL LOADING
# ======================================================
MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases/download/v1.0/cnn_bilstm_attention_model.h5"
MODEL_PATH = "cnn_bilstm_attention_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"Attention": Attention},
        compile=False
    )

# ======================================================
# UI – HEADER
# ======================================================
st.title("🔬 OncoVision – Histopathology Patch Analyzer")
st.markdown("""
This system analyzes **individual 96×96 histopathology patches** and detects  
**tumor-like patterns** learned from the **PCam dataset**.

⚠️ **Not a clinical diagnosis tool**
""")
st.divider()

# ======================================================
# UI – MAIN
# ======================================================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📸 Upload Patch Image")
    file = st.file_uploader(
        "Upload histopathology patch (96×96 recommended)",
        type=["jpg", "jpeg", "png"]
    )

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Patch", use_container_width=True)

with col2:
    st.subheader("🧠 Model Analysis")

    if file and st.button("🚀 Run Analysis", use_container_width=True):
        model = load_model()

        # --- PREPROCESSING (MATCHES PCam) ---
        img_resized = image.resize((96, 96), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized).astype("float32") / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # --- INFERENCE ---
        score = float(model.predict(img_batch, verbose=0)[0][0])

        st.markdown("### 📊 Prediction Confidence")
        st.progress(score)

        # --- UNCERTAINTY-AWARE INTERPRETATION ---
        if score >= 0.85:
            verdict = "🟥 Strong tumor-like patterns detected"
            reliability = "High confidence"
            color = "error"
        elif score <= 0.15:
            verdict = "🟩 No strong tumor-like patterns detected"
            reliability = "High confidence"
            color = "success"
        else:
            verdict = "🟨 Uncertain prediction"
            reliability = "Low confidence – expert review recommended"
            color = "warning"

        getattr(st, color)(f"**Result:** {verdict}")
        st.caption(f"Model output score: **{score:.3f}** | {reliability}")

        st.divider()

        # --- GRAD-CAM ---
        with st.expander("🔍 Visual Explanation (Grad-CAM)"):
            cam = generate_gradcam(img_batch, model, image)
            if cam is not None:
                st.image(
                    cam,
                    caption="Highlighted regions influencing the model",
                    use_container_width=True
                )
            else:
                st.info("Grad-CAM not available for this input.")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("ℹ️ System Information")
st.sidebar.info("""
**Model**
- CNN + BiLSTM + Attention
- Trained on PatchCamelyon (PCam)

**Important Notes**
- Patch-level analysis only
- Output is NOT a diagnosis
- Use for education & research
""")











