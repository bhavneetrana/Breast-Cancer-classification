import streamlit as st

st.set_page_config(
    page_title="Breast Cancer AI Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color .2s;
}
.card:hover { border-color: #E91E8C; }
.hero {
    background: linear-gradient(135deg, #1a0533 0%, #0D1117 60%, #0a1628 100%);
    border: 1px solid #E91E8C44;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 { font-size: 2.4rem; margin-bottom: .4rem; }
.hero p  { color: #8B949E; font-size: 1.05rem; }
.stat-pill {
    display: inline-block;
    background: #21262D;
    border: 1px solid #30363D;
    border-radius: 999px;
    padding: .35rem 1rem;
    font-size: .85rem;
    margin: .2rem;
    color: #E6EDF3;
}
.warning-box {
    background: #1f1a0e;
    border-left: 4px solid #f97316;
    border-radius: 6px;
    padding: .9rem 1.2rem;
    margin-top: 1.5rem;
    font-size: .9rem;
    color: #fbbf24;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🔬 Breast Cancer AI Diagnostic Suite</h1>
  <p>CNN · BiLSTM · Attention · MobileNetV2 — histopathology patch analysis</p>
  <div style="margin-top:1rem;">
    <span class="stat-pill">⚡ MC-Dropout Uncertainty</span>
    <span class="stat-pill">🗺️ Grad-CAM Heatmaps</span>
    <span class="stat-pill">📊 Confidence Intervals</span>
    <span class="stat-pill">📄 PDF Report Export</span>
    <span class="stat-pill">📁 Session History</span>
  </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="card">
      <h3>📝 Step 1 — Patient Profile</h3>
      <p style="color:#8B949E;font-size:.92rem;">
        Enter patient demographics, clinical history, and symptoms.
        Data persists across all pages in this session.
      </p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="card">
      <h3>🧠 Step 2 — Diagnostic Dashboard</h3>
      <p style="color:#8B949E;font-size:.92rem;">
        Upload a histopathology patch. The AI analyses it with
        Monte-Carlo Dropout for uncertainty estimation and renders
        a Grad-CAM attention heatmap.
      </p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="card">
      <h3>📄 Step 3 — Reports</h3>
      <p style="color:#8B949E;font-size:.92rem;">
        Review clinical recommendations, view scan history for the
        session, and download a formatted PDF diagnostic report.
      </p>
    </div>
    """, unsafe_allow_html=True)

with st.expander("ℹ️ About the Model"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Architecture**
        - MobileNetV2 backbone (ImageNet weights, frozen)
        - SpatialDropout2D + BatchNorm
        - Bidirectional LSTM (128 units, L2 reg)
        - Custom Attention layer
        - Dense → Dropout → Sigmoid
        """)
    with c2:
        st.markdown("""
        **Training details**
        - Dataset: BreaKHis / PCam histopathology patches (96×96)
        - Augmentation: rotation, flip, zoom, shift
        - Optimizer: Adam + Cosine Decay LR
        - Monitoring: val-AUC with early stopping
        - Class-balanced sample weights
        """)

st.markdown("""
<div class="warning-box">
  ⚠️ <strong>Educational & Research Use Only.</strong>
  This tool is not a certified medical device and must not be used
  for clinical diagnosis or treatment decisions. Always consult a
  qualified healthcare professional.
</div>
""", unsafe_allow_html=True)

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []
if "patient" not in st.session_state:
    st.session_state.patient = {}





















