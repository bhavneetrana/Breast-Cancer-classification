import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

# ======================================================
# ATTENTION LAYER (REQUIRED FOR LOADING)
# ======================================================
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)
    def get_config(self):
        return super().get_config()

# ======================================================
# UI SETUP & MODEL LOAD
# ======================================================
st.set_page_config(page_title="AI Breast Cancer Diagnostic", page_icon="🔬", layout="wide")

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(
        "cnn_bilstm_attention_model.h5",
        custom_objects={"Attention": Attention}
    )

# Custom CSS for better styling
st.markdown("""
    <style>
    .stProgress > div > div > div > div { background-color: #ff4b4b; }
    .report-card { background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e6e9ef; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 Breast Cancer AI Diagnostic System")
st.divider()

# ======================================================
# MAIN CONTENT
# ======================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Upload Histopathology Slide")
    uploaded_file = st.file_uploader("Select JPG or PNG (96x96 trained size)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Slide", use_container_width=True)

with col2:
    st.subheader("📊 Diagnostic Analysis")
    if uploaded_file:
        # Preprocessing
        img_resized = image.resize((96, 96))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("🚀 Run AI Analysis"):
            model = load_trained_model()
            prediction = model.predict(img_array)[0][0]
            risk_pct = float(prediction * 100)
            
            # Risk Gauge UI
            color = "green" if risk_pct < 30 else "orange" if risk_pct < 70 else "red"
            st.markdown(f"""
                <div style="background-color: #f0f2f6; border-radius: 10px; padding: 20px; text-align: center;">
                    <h3 style="color: {color}; margin-bottom: 5px;">MALIGNANCY RISK: {risk_pct:.1f}%</h3>
                    <div style="background-color: #ddd; border-radius: 20px; height: 30px; width: 100%;">
                        <div style="background-color: {color}; width: {risk_pct}%; height: 30px; border-radius: 20px;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Results Summary
            st.markdown("---")
            if risk_pct > 50:
                st.error("### Result: MALIGNANT DETECTED")
                st.write("The model identified patterns consistent with Invasive Ductal Carcinoma.")
            else:
                st.success("### Result: BENIGN / HEALTHY")
                st.write("No significant malignant patterns were identified in this sample.")

            # Save to history
            if "history" not in st.session_state: st.session_state.history = []
            st.session_state.history.append({"File": uploaded_file.name, "Risk": f"{risk_pct:.1f}%", "Status": "Malignant" if risk_pct > 50 else "Benign"})
    else:
        st.info("Waiting for a tissue slide upload...")

# ======================================================
# SIDEBAR HISTORY
# ======================================================
with st.sidebar:
    st.header("🕒 Prediction History")
    if "history" in st.session_state and st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report", data=csv, file_name="biopsy_results.csv", mime="text/csv")
    else:
        st.write("No scans analyzed yet.")

st.sidebar.warning("Disclaimer: This tool is for personal use")