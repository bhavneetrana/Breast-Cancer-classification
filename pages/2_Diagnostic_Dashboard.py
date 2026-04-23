import streamlit as st
from PIL import Image
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Diagnostic Dashboard", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.risk-badge {
    display: inline-block; padding: .45rem 1.2rem;
    border-radius: 999px; font-weight: 700; font-size: 1.1rem; margin: .5rem 0;
}
.metric-card {
    background: #161B22; border: 1px solid #30363D;
    border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
}
.metric-label { color: #8B949E; font-size:.82rem; margin-bottom:.3rem; }
.metric-value { font-size: 1.6rem; font-weight: 700; }
.ci-bar-container {
    background: #21262D; border-radius: 6px;
    height: 18px; position: relative; overflow: hidden; margin: .3rem 0;
}
.ci-fill { height: 100%; border-radius: 6px; transition: width .4s ease; }
</style>
""", unsafe_allow_html=True)

if "patient" not in st.session_state or not st.session_state.patient:
    st.warning("⚠️ Please complete the **Patient Profile** first.")
    st.page_link("pages/1_Patient_Profile.py", label="→ Go to Patient Profile")
    st.stop()

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

try:
    from model_utils import load_model, predict_with_uncertainty, make_gradcam_heatmap, interpret
    MODEL_AVAILABLE = True
except Exception as e:
    MODEL_AVAILABLE = False
    st.error(f"Model utilities could not be loaded: {e}")

p = st.session_state.patient
st.title("🧠 Diagnostic Dashboard")
st.caption(f"Patient: **{p['name']}** · Age {p['age']} · {p['gender']} · Visit {p.get('visit_date','—')}")
st.markdown("---")

col_up, col_opts = st.columns([2, 1])
with col_up:
    uploaded = st.file_uploader(
        "Upload Histopathology Patch (96×96 or larger)",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="BreaKHis / PCam-style H&E stained patch images work best.",
    )
with col_opts:
    n_passes = st.slider("MC-Dropout passes", min_value=5, max_value=50, value=20, step=5)
    show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)
    confidence_threshold = st.slider("Flag if CI upper-bound exceeds (%)", 40, 90, 75)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    img_col, result_col = st.columns([1, 1])

    with img_col:
        st.subheader("🖼️ Input Image")
        st.image(image, use_container_width=True, caption=uploaded.name)

    with result_col:
        st.subheader("🔬 Analysis")
        run_btn = st.button("▶️ Run Diagnostic", use_container_width=True, type="primary")

        if run_btn:
            if not MODEL_AVAILABLE:
                st.error("Model is not available.")
                st.stop()

            with st.spinner("Loading model…"):
                model = load_model()

            if model is None:
                st.error("TensorFlow / Keras model could not be loaded.")
                st.stop()

            with st.spinner(f"Running {n_passes} stochastic forward passes…"):
                mean_score, std_score, raw_scores = predict_with_uncertainty(
                    model, image, n_passes=n_passes)

            ci_low  = float(np.percentile(raw_scores, 5))
            ci_high = float(np.percentile(raw_scores, 95))
            label, color, recommendation = interpret(mean_score)

            record = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "filename":  uploaded.name,
                "score":     mean_score,
                "std":       std_score,
                "ci_low":    ci_low,
                "ci_high":   ci_high,
                "label":     label,
                "color":     color,
            }
            st.session_state.last_result = record
            st.session_state.last_image  = image.copy()
            st.session_state.scan_history.append(record)
            # store raw_scores for histogram
            st.session_state.last_raw_scores = raw_scores

            st.markdown(
                f'<div class="risk-badge" style="background:{color}22;'
                f'border:2px solid {color};color:{color};">{label}</div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Cancer Risk Score</div>'
                            f'<div class="metric-value" style="color:{color};">{mean_score*100:.1f}%</div></div>',
                            unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Uncertainty (σ)</div>'
                            f'<div class="metric-value">{std_score*100:.1f}%</div></div>',
                            unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">90% CI</div>'
                            f'<div class="metric-value" style="font-size:1.1rem;">'
                            f'{ci_low*100:.0f}% – {ci_high*100:.0f}%</div></div>',
                            unsafe_allow_html=True)

            st.markdown("**Risk probability bar:**")
            st.markdown(
                f'<div class="ci-bar-container"><div class="ci-fill" '
                f'style="width:{mean_score*100:.1f}%;background:{color};"></div></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Mean: {mean_score*100:.1f}% | 95th-pct upper: {ci_high*100:.1f}% | {n_passes} MC passes")

            if ci_high * 100 >= confidence_threshold:
                st.warning(f"⚠️ Upper CI ({ci_high*100:.0f}%) exceeds {confidence_threshold}% threshold.")

            st.info(f"**Clinical Guidance:** {recommendation}")

    if show_gradcam and "last_result" in st.session_state and MODEL_AVAILABLE:
        st.markdown("---")
        st.subheader("🗺️ Grad-CAM Attention Heatmap")
        st.caption("Warmer colours (red/yellow) indicate higher model attention.")
        with st.spinner("Generating Grad-CAM heatmap…"):
            try:
                model = load_model()
                heatmap = make_gradcam_heatmap(model, image)
            except Exception as e:
                heatmap = None
                st.warning(f"Grad-CAM could not be generated: {e}")
        if heatmap is not None:
            gc1, gc2 = st.columns(2)
            with gc1:
                st.image(image, caption="Original patch", use_container_width=True)
            with gc2:
                st.image(heatmap, caption="Grad-CAM overlay", use_container_width=True)
        else:
            st.info("Grad-CAM unavailable for this model architecture.")

    if "last_raw_scores" in st.session_state:
        with st.expander("📈 MC-Dropout Score Distribution"):
            import matplotlib.pyplot as plt
            raw = st.session_state.last_raw_scores
            res = st.session_state.last_result
            fig, ax = plt.subplots(figsize=(7, 3), facecolor="#0D1117")
            ax.set_facecolor("#161B22")
            ax.hist(raw * 100, bins=15, color=res["color"], alpha=0.8, edgecolor="#30363D")
            ax.axvline(res["score"]*100, color="white", linestyle="--", linewidth=1.5,
                       label=f"Mean {res['score']*100:.1f}%")
            ax.axvline(res["ci_low"]*100, color="#8B949E", linestyle=":", linewidth=1,
                       label=f"5th pct {res['ci_low']*100:.1f}%")
            ax.axvline(res["ci_high"]*100, color="#8B949E", linestyle=":", linewidth=1,
                       label=f"95th pct {res['ci_high']*100:.1f}%")
            ax.set_xlabel("Risk Score (%)", color="#E6EDF3")
            ax.set_ylabel("Count", color="#E6EDF3")
            ax.tick_params(colors="#8B949E")
            ax.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="#E6EDF3", fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363D")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
else:
    st.info("☝️ Upload a histopathology patch image to begin analysis.")

with st.sidebar:
    st.subheader("📁 Session Scan History")
    history = st.session_state.get("scan_history", [])
    if history:
        for rec in reversed(history[-5:]):
            st.markdown(f"**{rec['timestamp']}** — {rec['filename'][:18]}…  \n{rec['label']} · {rec['score']*100:.1f}%")
            st.markdown("---")
    else:
        st.caption("No scans yet.")
