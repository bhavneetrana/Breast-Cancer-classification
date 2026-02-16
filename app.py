import streamlit as st
import tensorflow as tf
import numpy as np
import os
import urllib.request
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Breast Cancer AI Diagnostic Suite",
    page_icon="🔬",
    layout="wide"
)

# ======================================================
# CUSTOM ATTENTION LAYER
# ======================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_attention = None

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
        self.last_attention = a
        return tf.keras.backend.sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()

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
# HELPER FUNCTIONS
# ======================================================
def find_attention_layer(model):
    for layer in model.layers:
        if isinstance(layer, Attention):
            return layer
    return None


def generate_health_advice(score):
    if score < 0.25:
        return """🟢 Low Risk Guidance:
• Maintain balanced diet (leafy greens, berries, whole grains)
• Regular screening every 6–12 months
• Exercise 30 minutes daily
"""
    elif score < 0.75:
        return """🟡 Moderate Risk Guidance:
• Schedule specialist consultation
• Reduce processed foods
• Increase antioxidant-rich diet
• Manage stress levels
"""
    else:
        return """🔴 High Risk Guidance:
• Immediate oncologist consultation recommended
• Diagnostic biopsy advised
• Maintain high-protein nutrient-rich diet
• Seek emotional & psychological support
"""


def generate_pdf_report(patient, score, advice):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Breast Cancer AI Diagnostic Report", styles['Heading1']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Patient Name: {patient['name']}", styles['Normal']))
    elements.append(Paragraph(f"Age: {patient['age']}", styles['Normal']))
    elements.append(Paragraph(f"Gender: {patient['gender']}", styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Malignancy Probability: {score*100:.2f}%", styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Health Recommendations:", styles['Heading2']))
    elements.append(Paragraph(advice.replace("\n", "<br/>"), styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        "⚠️ This AI-generated report is for educational purposes only. "
        "Consult a certified medical professional for diagnosis.",
        styles['Normal']
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ======================================================
# HEADER
# ======================================================
st.title("🔬 Breast Cancer AI Diagnostic Suite")
st.divider()

# ======================================================
# PATIENT PROFILE FORM
# ======================================================
st.markdown("## 📝 Create / Update Patient Profile")

with st.form("patient_form"):
    name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=1, max_value=120)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    symptoms = st.text_area("Symptoms / Notes")

    submitted = st.form_submit_button("Save Profile")

if submitted:
    st.session_state.patient = {
        "name": name,
        "age": age,
        "gender": gender,
        "symptoms": symptoms
    }
    st.success("Patient profile saved successfully.")

st.divider()

# ======================================================
# CLINICAL DASHBOARD
# ======================================================
left, center, right = st.columns([1, 2, 1], gap="large")

# LEFT PANEL
with left:
    st.markdown("## 👤 Patient Details")
    if "patient" in st.session_state:
        p = st.session_state.patient
        st.write(f"**Name:** {p['name']}")
        st.write(f"**Age:** {p['age']}")
        st.write(f"**Gender:** {p['gender']}")
        st.write(f"**Symptoms:** {p['symptoms']}")
    else:
        st.info("No patient profile created yet.")

    st.markdown("---")
    file = st.file_uploader("Upload Histopathology Patch", type=["jpg", "jpeg", "png"])

# CENTER PANEL
with center:
    st.markdown("## 🖼 Diagnostic Viewer")
    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True)
    else:
        st.info("Upload image to begin.")

# RIGHT PANEL
with right:
    st.markdown("## 🧠 AI Analysis Panel")

    if file and "patient" in st.session_state:
        if st.button("🚀 Run Diagnostic", use_container_width=True):

            model = load_model()

            img_resized = image.resize((96, 96), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized).astype("float32") / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            score = float(model.predict(img_batch, verbose=0)[0][0])

            st.metric("Cancer Risk", f"{score*100:.1f}%")
            st.progress(score)

            advice = generate_health_advice(score)

            st.markdown("### 🥗 Health Recommendations")
            st.write(advice)

            pdf = generate_pdf_report(st.session_state.patient, score, advice)

            st.download_button(
                "📄 Download Diagnostic Report",
                pdf,
                file_name="AI_Diagnostic_Report.pdf",
                mime="application/pdf"
            )

            st.divider()

            with st.expander("🔍 Visual Explanation (Attention Weights)"):
                att_layer = find_attention_layer(model)

                if att_layer and att_layer.last_attention is not None:
                    attention_values = att_layer.last_attention.numpy().squeeze()
                    attention_values = attention_values / (
                        np.max(attention_values) + 1e-8
                    )
                    st.line_chart(attention_values)
                else:
                    st.info("Attention weights not available.")

    elif file and "patient" not in st.session_state:
        st.warning("Please create patient profile first.")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("ℹ️ System Information")
st.sidebar.info("""
AI Model: CNN + BiLSTM + Attention  
Dataset: PatchCamelyon (PCam)

⚠️ Educational use only.
""")




















