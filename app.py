diff --git a/app.py b/app.py
index 2c8e9a1..9fd31c2 100644
--- a/app.py
+++ b/app.py
@@ -1,6 +1,7 @@
 import streamlit as st
 import tensorflow as tf
 import numpy as np
+import pandas as pd
 from PIL import Image
 import tensorflow.keras.backend as K
 from tensorflow.keras.layers import Layer
@@ -8,6 +9,11 @@ import os
 import urllib.request
+from datetime import datetime
+from io import BytesIO
+from reportlab.lib.pagesizes import A4
+from reportlab.pdfgen import canvas
+from reportlab.lib.units import cm

 # ======================================================
 # PAGE CONFIG
@@ -47,12 +53,22 @@ MODEL_URL = "https://github.com/bhavneetrana/Breast-Cancer-classification/releases
 MODEL_PATH = "cnn_bilstm_attention_model.h5"

 if not os.path.exists(MODEL_PATH):
-    with st.spinner("🔽 Downloading AI model (first-time setup)..."):
-        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
+    try:
+        with st.spinner("🔽 Downloading AI model (first-time setup)..."):
+            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
+    except Exception as e:
+        st.error("❌ Model download failed. Please check your connection or repository.")
+        st.stop()

 # ======================================================
 # CUSTOM ATTENTION LAYER (KERAS 3 SAFE)
@@ -96,6 +112,12 @@ def load_trained_model():
     return tf.keras.models.load_model(
         MODEL_PATH,
         custom_objects={"Attention": Attention},
         compile=False
     )

+# ======================================================
+# SESSION STATE INITIALIZATION
+# ======================================================
+if "history" not in st.session_state:
+    st.session_state.history = []

 # ======================================================
 # MAIN APPLICATION UI
@@ -138,6 +160,10 @@ with col2:
     if uploaded_file:
         img = image.resize((96, 96))
         img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
+        label = None
+        risk_pct = None
+        prediction = None

         if st.button("🚀 Analyze Image", use_container_width=True):
             model = load_trained_model()
@@ -145,6 +171,8 @@ with col2:
             prediction = model.predict(img_array, verbose=0)[0][0]
             risk_pct = float(prediction * 100)
+            label = "Malignant" if risk_pct > 50 else "Benign"
+            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

             color = "#4CAF50" if risk_pct < 30 else "#FF9800" if risk_pct < 70 else "#F44336"

@@ -158,11 +186,27 @@ with col2:
                 <h2 style="color:{color}; margin-bottom:10px;">
                     Cancer Risk: {risk_pct:.1f}%
                 </h2>
             </div>
             """, unsafe_allow_html=True)

-            if risk_pct > 50:
-                st.error("### 🔴 Malignant Tumor Detected")
-            else:
-                st.success("### 🟢 Benign / Non-Cancerous")
+            # ---------------- CONFIDENCE BAR ----------------
+            st.markdown("#### 🔍 Prediction Confidence")
+            st.progress(int(risk_pct))
+            st.caption(f"Model confidence: **{risk_pct:.1f}%**")
+
+            if label == "Malignant":
+                st.error("### 🔴 Malignant Tumor Detected")
+            else:
+                st.success("### 🟢 Benign / Non-Cancerous")
+
+            # ---------------- SAVE HISTORY ----------------
+            st.session_state.history.append({
+                "timestamp": timestamp,
+                "risk_%": round(risk_pct, 2),
+                "label": label
+            })

+            # ---------------- HEALTH TIPS ----------------
+            st.markdown("### 🩺 Health Guidance (Informational)")
+            if label == "Benign":
+                st.info("""
+                - Maintain a healthy lifestyle (balanced diet, regular exercise)
+                - Follow routine screening schedules
+                - Stay aware of any unusual changes
+                """)
+            else:
+                st.warning("""
+                - Consult a qualified oncologist immediately
+                - Early professional evaluation is critical
+                - Seek emotional and family support
+                """)

+            # ---------------- PDF REPORT ----------------
+            if st.button("📄 Download Medical Report"):
+                buffer = BytesIO()
+                c = canvas.Canvas(buffer, pagesize=A4)
+                c.setFont("Helvetica-Bold", 16)
+                c.drawString(2*cm, 28*cm, "Breast Cancer AI Diagnostic Report")
+
+                c.setFont("Helvetica", 11)
+                c.drawString(2*cm, 26.5*cm, f"Date & Time: {timestamp}")
+                c.drawString(2*cm, 25.5*cm, f"Prediction: {label}")
+                c.drawString(2*cm, 24.5*cm, f"Risk Confidence: {risk_pct:.1f}%")
+
+                c.setFont("Helvetica-Oblique", 10)
+                c.drawString(2*cm, 22.5*cm, "Disclaimer:")
+                c.drawString(2*cm, 21.8*cm,
+                             "This report is generated by an AI system for educational purposes only.")
+                c.drawString(2*cm, 21.1*cm,
+                             "It is NOT a substitute for professional medical diagnosis.")
+
+                c.showPage()
+                c.save()
+
+                buffer.seek(0)
+                st.download_button(
+                    label="⬇️ Download PDF Report",
+                    data=buffer,
+                    file_name="breast_cancer_ai_report.pdf",
+                    mime="application/pdf"
+                )

     else:
         st.info("Upload a medical image to begin analysis.")

+# ======================================================
+# PREDICTION HISTORY
+# ======================================================
+st.markdown("## 📚 Prediction History")
+
+if st.session_state.history:
+    df = pd.DataFrame(st.session_state.history)
+    st.dataframe(df, use_container_width=True)
+
+    col_a, col_b = st.columns(2)
+    with col_a:
+        st.download_button(
+            "⬇️ Download History (CSV)",
+            df.to_csv(index=False).encode("utf-8"),
+            "prediction_history.csv",
+            "text/csv"
+        )
+    with col_b:
+        if st.button("🧹 Clear History"):
+            st.session_state.history.clear()
+            st.experimental_rerun()
+else:
+    st.caption("No predictions made yet.")






