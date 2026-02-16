import streamlit as st
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO

st.title("📄 Reports & Recommendations")

if "last_score" not in st.session_state:
    st.warning("Run diagnostic first.")
    st.stop()

score = st.session_state.last_score
patient = st.session_state.patient

def generate_advice(score):
    if score < 0.25:
        return "Low Risk: Maintain healthy diet and regular screening."
    elif score < 0.75:
        return "Moderate Risk: Schedule specialist consultation."
    else:
        return "High Risk: Immediate oncologist consultation advised."

advice = generate_advice(score)

st.write("### Health Recommendations")
st.write(advice)

def generate_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AI Diagnostic Report", styles['Heading1']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Patient: {patient['name']}", styles['Normal']))
    elements.append(Paragraph(f"Risk Score: {score*100:.2f}%", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(advice, styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

pdf = generate_pdf()

st.download_button(
    "Download Report",
    pdf,
    file_name="Diagnostic_Report.pdf",
    mime="application/pdf"
)
