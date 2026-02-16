import streamlit as st
from PIL import Image
from model_utils import load_model, predict_image

st.title("🧠 Diagnostic Dashboard")

if "patient" not in st.session_state:
    st.warning("Please create patient profile first.")
    st.stop()

file = st.file_uploader("Upload Histopathology Patch", type=["jpg","png","jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("Run Diagnostic"):

        model = load_model()
        score = predict_image(model, image)

        st.session_state.last_score = score

        st.metric("Cancer Risk", f"{score*100:.2f}%")
        st.progress(score)
