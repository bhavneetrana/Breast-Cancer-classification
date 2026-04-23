import streamlit as st
from datetime import date

st.set_page_config(page_title="Patient Profile", page_icon="📝", layout="wide")

st.markdown("""
<style>
.section-header {
    font-size: 1.05rem; font-weight: 600; color: #E91E8C;
    border-bottom: 1px solid #30363D;
    padding-bottom: .4rem; margin: 1.2rem 0 .8rem;
}
.profile-card {
    background: #161B22; border: 1px solid #30363D;
    border-radius: 12px; padding: 1.4rem;
}
</style>
""", unsafe_allow_html=True)

st.title("📝 Patient Profile")
st.caption("All fields marked ✱ are required before proceeding to diagnosis.")

if "patient" not in st.session_state:
    st.session_state.patient = {}

p = st.session_state.patient

with st.form("patient_form", clear_on_submit=False):
    st.markdown('<div class="section-header">👤 Demographics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        name = st.text_input("Patient Name ✱", value=p.get("name", ""), placeholder="Full name")
    with col2:
        age = st.number_input("Age ✱", min_value=1, max_value=120, value=int(p.get("age", 45)))
    with col3:
        gender = st.selectbox("Gender ✱", ["Female", "Male", "Other"],
                              index=["Female", "Male", "Other"].index(p.get("gender", "Female")))

    col4, col5 = st.columns(2)
    with col4:
        patient_id = st.text_input("Patient ID / MRN", value=p.get("patient_id", ""), placeholder="Optional")
    with col5:
        visit_date = st.date_input("Visit Date", value=date.today())

    st.markdown('<div class="section-header">🏥 Clinical History</div>', unsafe_allow_html=True)
    col6, col7 = st.columns(2)
    with col6:
        family_history = st.selectbox(
            "Family History of Breast Cancer",
            ["Unknown", "None", "First-degree relative", "Second-degree relative"],
            index=["Unknown", "None", "First-degree relative", "Second-degree relative"]
                  .index(p.get("family_history", "Unknown")),
        )
        prior_biopsies = st.selectbox("Prior Biopsies", ["None", "1", "2", "3+"],
                                      index=["None", "1", "2", "3+"].index(p.get("prior_biopsies", "None")))
    with col7:
        menopausal = st.selectbox(
            "Menopausal Status",
            ["Unknown", "Pre-menopausal", "Peri-menopausal", "Post-menopausal"],
            index=["Unknown", "Pre-menopausal", "Peri-menopausal", "Post-menopausal"]
                  .index(p.get("menopausal", "Unknown")),
        )
        hormone_therapy = st.selectbox(
            "Hormone Therapy / Contraceptives",
            ["None / Unknown", "Current user", "Past user"],
            index=["None / Unknown", "Current user", "Past user"]
                  .index(p.get("hormone_therapy", "None / Unknown")),
        )

    st.markdown('<div class="section-header">🩺 Presenting Symptoms</div>', unsafe_allow_html=True)
    symptom_options = ["Palpable lump", "Nipple discharge", "Skin changes",
                       "Pain / tenderness", "Nipple retraction", "Axillary lymphadenopathy",
                       "Asymmetry", "None"]
    symptoms_selected = st.multiselect("Select all that apply", symptom_options,
                                       default=p.get("symptoms_list", ["None"]))
    notes = st.text_area("Additional Clinical Notes", value=p.get("notes", ""),
                         placeholder="Relevant history, medications, allergies…", height=100)

    submitted = st.form_submit_button("💾 Save Patient Profile", use_container_width=True)

if submitted:
    if not name.strip():
        st.error("Patient name is required.")
    else:
        st.session_state.patient = {
            "name": name.strip(), "age": age, "gender": gender,
            "patient_id": patient_id.strip() or "N/A",
            "visit_date": str(visit_date),
            "family_history": family_history, "prior_biopsies": prior_biopsies,
            "menopausal": menopausal, "hormone_therapy": hormone_therapy,
            "symptoms_list": symptoms_selected, "notes": notes.strip(),
        }
        st.success(f"✅ Profile saved for **{name.strip()}**")

if st.session_state.patient:
    p = st.session_state.patient
    st.markdown("---")
    st.subheader("📋 Current Patient on Record")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Name", p["name"])
        st.metric("Age", p["age"])
        st.metric("Gender", p["gender"])
    with c2:
        st.metric("Patient ID", p.get("patient_id", "N/A"))
        st.metric("Visit Date", p.get("visit_date", "—"))
        st.metric("Menopausal Status", p.get("menopausal", "—"))
    with c3:
        st.metric("Family History", p.get("family_history", "—"))
        st.metric("Prior Biopsies", p.get("prior_biopsies", "—"))
        st.metric("Hormone Therapy", p.get("hormone_therapy", "—"))
    st.markdown(f"**Symptoms:** {', '.join(p.get('symptoms_list', ['—']))}")
    if p.get("notes"):
        st.markdown(f"**Notes:** {p['notes']}")
    if st.button("🗑️ Clear Profile"):
        st.session_state.patient = {}
        st.rerun()
else:
    st.info("No patient profile saved yet. Fill the form above to get started.")
