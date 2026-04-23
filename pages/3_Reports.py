import streamlit as st
from io import BytesIO
from datetime import date

st.set_page_config(page_title="Reports", page_icon="📄", layout="wide")

st.markdown("""
<style>
.rec-card {
    background: #161B22; border-left: 4px solid;
    border-radius: 8px; padding: 1rem 1.4rem; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

if "last_result" not in st.session_state:
    st.warning("⚠️ No diagnostic result found. Run analysis first.")
    st.page_link("pages/2_Diagnostic_Dashboard.py", label="→ Go to Diagnostic Dashboard")
    st.stop()

patient = st.session_state.get("patient", {})
result  = st.session_state.last_result
history = st.session_state.get("scan_history", [])
score   = result["score"]
label   = result["label"]
color   = result["color"]
ci_low  = result["ci_low"]
ci_high = result["ci_high"]
std_dev = result["std"]

RECOMMENDATIONS = {
    "🟢 Low Risk": {
        "border": "#22c55e",
        "items": [
            "Continue routine annual mammography screening.",
            "Maintain a healthy body weight and regular physical activity.",
            "Limit alcohol consumption and avoid smoking.",
            "Follow up in 12 months unless new symptoms develop.",
        ],
    },
    "🟡 Borderline": {
        "border": "#eab308",
        "items": [
            "Repeat imaging (ultrasound or MRI) in 3–6 months.",
            "Document findings and compare with prior studies.",
            "Discuss BRCA genetic testing if family history is present.",
            "Consider referral to a breast clinic for closer monitoring.",
        ],
    },
    "🟠 Moderate Risk": {
        "border": "#f97316",
        "items": [
            "Specialist consultation within 2 weeks is strongly recommended.",
            "Core needle biopsy should be considered for tissue diagnosis.",
            "MRI breast imaging may provide additional characterisation.",
            "Ensure complete clinical, radiological and pathological correlation.",
        ],
    },
    "🔴 High Risk": {
        "border": "#ef4444",
        "items": [
            "Immediate referral to oncology is advised.",
            "Urgent biopsy and histological grading required.",
            "Initiate tumour staging work-up (CT chest/abdomen/pelvis).",
            "Multidisciplinary team (MDT) discussion should be scheduled.",
        ],
    },
}

st.title("📄 Reports & Recommendations")
st.caption(f"Patient: **{patient.get('name','—')}** · Latest scan: **{result['filename']}** · {result['timestamp']}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Risk Score",         f"{score*100:.1f}%")
m2.metric("Uncertainty (σ)",    f"{std_dev*100:.1f}%")
m3.metric("CI Low (5th pct)",   f"{ci_low*100:.0f}%")
m4.metric("CI High (95th pct)", f"{ci_high*100:.0f}%")

st.markdown("---")
st.subheader("🩺 Clinical Recommendations")

rec_data = RECOMMENDATIONS.get(label, RECOMMENDATIONS["🟡 Borderline"])
st.markdown(
    f'<div class="rec-card" style="border-color:{rec_data["border"]};">'
    f'<strong style="color:{rec_data["border"]};">{label}</strong><br><br>'
    + "".join(f"• {item}<br>" for item in rec_data["items"])
    + "</div>",
    unsafe_allow_html=True,
)

with st.expander("⚠️ Patient Risk Factor Summary"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Family History:** {patient.get('family_history','—')}")
        st.markdown(f"**Prior Biopsies:** {patient.get('prior_biopsies','—')}")
        st.markdown(f"**Menopausal Status:** {patient.get('menopausal','—')}")
    with col2:
        st.markdown(f"**Hormone Therapy:** {patient.get('hormone_therapy','—')}")
        st.markdown(f"**Symptoms:** {', '.join(patient.get('symptoms_list', ['—']))}")
        st.markdown(f"**Age:** {patient.get('age','—')}")
    if patient.get("notes"):
        st.markdown(f"**Notes:** {patient['notes']}")

st.markdown("---")
st.subheader("📁 Session Scan History")

if history:
    import pandas as pd
    df = pd.DataFrame([{
        "Time": r["timestamp"], "File": r["filename"],
        "Score (%)": f"{r['score']*100:.1f}", "Std (%)": f"{r['std']*100:.1f}",
        "CI Low": f"{r['ci_low']*100:.0f}%", "CI High": f"{r['ci_high']*100:.0f}%",
        "Result": r["label"],
    } for r in history])
    st.dataframe(df, use_container_width=True, hide_index=True)
    if st.button("🗑️ Clear History"):
        st.session_state.scan_history = []
        del st.session_state["last_result"]
        st.rerun()
else:
    st.info("No scans recorded in this session.")

st.markdown("---")
st.subheader("📥 Download Report")


def generate_pdf(patient, result, history, rec_data):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    title_style   = ParagraphStyle("Title2",   parent=styles["Heading1"], fontSize=18, spaceAfter=6)
    heading_style = ParagraphStyle("Heading2", parent=styles["Heading2"], fontSize=13, spaceAfter=4,
                                   textColor=colors.HexColor("#c0392b"))
    body_style    = ParagraphStyle("Body2",    parent=styles["Normal"],   fontSize=10, spaceAfter=3, leading=14)
    small_style   = ParagraphStyle("Small",    parent=styles["Normal"],   fontSize=8,  textColor=colors.grey)

    elems = []
    elems.append(Paragraph("AI Diagnostic Report", title_style))
    elems.append(Paragraph("Breast Cancer Histopathology Analysis — For Research Use Only", small_style))
    elems.append(Spacer(1, 6))
    elems.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c0392b")))
    elems.append(Spacer(1, 10))

    elems.append(Paragraph("Patient Information", heading_style))
    info_data = [
        ["Name", patient.get("name","—"), "Patient ID", patient.get("patient_id","—")],
        ["Age", str(patient.get("age","—")), "Gender", patient.get("gender","—")],
        ["Visit Date", patient.get("visit_date", str(date.today())), "Menopausal", patient.get("menopausal","—")],
        ["Family History", patient.get("family_history","—"), "Prior Biopsies", patient.get("prior_biopsies","—")],
    ]
    t = Table(info_data, colWidths=[35*mm, 55*mm, 35*mm, 50*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(0,-1), colors.HexColor("#f5f5f5")),
        ("BACKGROUND", (2,0),(2,-1), colors.HexColor("#f5f5f5")),
        ("FONTSIZE",   (0,0),(-1,-1), 9),
        ("GRID",       (0,0),(-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING",    (0,0),(-1,-1), 4),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 10))

    elems.append(Paragraph("Diagnostic Result", heading_style))
    res_data = [
        ["Risk Score", f"{result['score']*100:.1f}%", "Result Label", result["label"]],
        ["Uncertainty σ", f"{result['std']*100:.1f}%", "90% CI",
         f"{result['ci_low']*100:.0f}% – {result['ci_high']*100:.0f}%"],
        ["Scan File", result["filename"], "Timestamp", result["timestamp"]],
    ]
    t2 = Table(res_data, colWidths=[35*mm, 55*mm, 35*mm, 50*mm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(0,-1), colors.HexColor("#f5f5f5")),
        ("BACKGROUND", (2,0),(2,-1), colors.HexColor("#f5f5f5")),
        ("FONTSIZE",   (0,0),(-1,-1), 9),
        ("GRID",       (0,0),(-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING",    (0,0),(-1,-1), 4),
    ]))
    elems.append(t2)
    elems.append(Spacer(1, 10))

    elems.append(Paragraph("Clinical Recommendations", heading_style))
    for item in rec_data["items"]:
        elems.append(Paragraph(f"• {item}", body_style))
    elems.append(Spacer(1, 10))

    symptoms = ", ".join(patient.get("symptoms_list", ["—"]))
    elems.append(Paragraph("Presenting Symptoms", heading_style))
    elems.append(Paragraph(symptoms, body_style))
    if patient.get("notes"):
        elems.append(Paragraph(f"Notes: {patient['notes']}", body_style))
    elems.append(Spacer(1, 10))

    if len(history) > 1:
        elems.append(Paragraph("Session Scan History", heading_style))
        hist_header = [["Time","File","Score","σ","CI Low","CI High","Result"]]
        hist_rows = [
            [r["timestamp"], r["filename"][:20], f"{r['score']*100:.1f}%",
             f"{r['std']*100:.1f}%", f"{r['ci_low']*100:.0f}%",
             f"{r['ci_high']*100:.0f}%", r["label"]]
            for r in history
        ]
        ht = Table(hist_header + hist_rows,
                   colWidths=[18*mm, 45*mm, 16*mm, 12*mm, 16*mm, 16*mm, 30*mm])
        ht.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,0), colors.HexColor("#c0392b")),
            ("TEXTCOLOR",      (0,0),(-1,0), colors.white),
            ("FONTSIZE",       (0,0),(-1,-1), 8),
            ("GRID",           (0,0),(-1,-1), 0.4, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, colors.HexColor("#fafafa")]),
            ("PADDING",        (0,0),(-1,-1), 3),
        ]))
        elems.append(ht)
        elems.append(Spacer(1, 10))

    elems.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        f"Generated by Breast Cancer AI Suite on {date.today()}. "
        "FOR RESEARCH AND EDUCATIONAL USE ONLY. "
        "This report does not constitute a clinical diagnosis.",
        small_style,
    ))

    doc.build(elems)
    buffer.seek(0)
    return buffer


pdf_buffer = generate_pdf(patient, result, history, rec_data)
st.download_button(
    label="📥 Download Full PDF Report",
    data=pdf_buffer,
    file_name=f"AI_Diagnostic_{patient.get('name','Patient').replace(' ','_')}_{date.today()}.pdf",
    mime="application/pdf",
    use_container_width=True,
    type="primary",
)
