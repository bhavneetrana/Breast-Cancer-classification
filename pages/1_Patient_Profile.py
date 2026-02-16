import streamlit as st

st.title("📝 Patient Profile")

with st.form("patient_form"):
    name = st.text_input("Patient Name")
    age = st.number_input("Age", 1, 120)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    symptoms = st.text_area("Symptoms")

    submit = st.form_submit_button("Save")

if submit:
    st.session_state.patient = {
        "name": name,
        "age": age,
        "gender": gender,
        "symptoms": symptoms
    }
    st.success("Patient profile saved.")

if "patient" in st.session_state:
    st.write("### Current Patient")
    st.write(st.session_state.patient)
