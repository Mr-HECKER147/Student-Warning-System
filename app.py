#!/usr/bin/env python3
import streamlit as st
from model import predict_student

st.set_page_config(page_title="Academic Risk EWS", layout="wide")

st.markdown("""
<style>
:root {
  --ink: #0f172a;
  --muted: #475569;
  --panel: #f8fafc;
  --panel-2: #eef2f7;
  --accent: #0ea5a6;
  --accent-2: #2563eb;
  --danger: #dc2626;
  --ok: #16a34a;
}

.app-hero {
  padding: 18px 20px;
  border-radius: 16px;
  background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 40%, #f8fafc 100%);
  border: 1px solid #e5e7eb;
}

.app-hero h1 {
  margin: 0 0 6px 0;
  font-size: 28px;
  color: var(--ink);
}

.app-hero p {
  margin: 0;
  color: var(--muted);
}

.summary-card {
  border-radius: 16px;
  padding: 16px 18px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  border: 1px solid #e5e7eb;
  box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 12px;
}

.summary-tile {
  padding: 12px;
  border-radius: 12px;
  background: var(--panel);
  border: 1px solid #e5e7eb;
}

.summary-tile h4 {
  margin: 0 0 6px 0;
  font-size: 12px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.summary-tile p {
  margin: 0;
  font-size: 16px;
  color: var(--ink);
  font-weight: 600;
}

.section-card {
  padding: 16px;
  border-radius: 14px;
  border: 1px solid #e5e7eb;
  background: #ffffff;
}

.insights-card {
  padding: 16px;
  border-radius: 14px;
  border: 1px dashed #cbd5f5;
  background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
}

.chip {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  margin-right: 6px;
  background: var(--panel-2);
  border: 1px solid #e5e7eb;
  color: var(--muted);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-hero">
  <h1>Academic Risk Early Warning System</h1>
  <p>VAP Capstone | 95% Accuracy | Live Demo</p>
</div>
""", unsafe_allow_html=True)

st.divider()

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("### Enter Student Data")
    st.markdown("Use the sliders below. When ready, click Predict Risk to get the result.")

    with st.form("student_inputs", clear_on_submit=False):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            attendance = st.slider("Attendance (%)", 40.0, 100.0, 75.0, help="Overall attendance percentage.")
            test1 = st.slider("Internal Test 1 (/25)", 0.0, 25.0, 18.0)
            assignments = st.slider("Assignments (/20)", 0.0, 20.0, 14.0)
        with c2:
            test2 = st.slider("Internal Test 2 (/25)", 0.0, 25.0, 20.0)
            participation = st.slider("Participation (1-5)", 1, 5, 3)
            gpa = st.slider("Previous GPA", 4.0, 10.0, 7.5)
        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    summary_reason = "Awaiting prediction"
    summary_risk = "Not computed"
    summary_prob = "—"

    if submitted:
        result = predict_student(test1, test2, attendance, assignments, participation, gpa)

        summary_risk = result["risk"]
        summary_prob = result["probability"]
        summary_reason = ", ".join(result["reasons"])

        st.markdown("### Result")
        if "SAFE" in result["risk"]:
            st.success(result["risk"])
        else:
            st.error(result["risk"])

        st.info(f"Risk Probability: {result['probability']}")
        st.info(f"To improve: {', '.join(result['reasons'])}")

    st.markdown("### Summary")
    st.markdown(
        f"""
<div class="summary-card">
  <div class="chip">Prediction Engine v1</div>
  <div class="chip">Feature Signals</div>
  <div class="chip">Risk Scoring</div>
  <div class="summary-grid">
    <div class="summary-tile">
      <h4>Risk Status</h4>
      <p>{summary_risk}</p>
    </div>
    <div class="summary-tile">
      <h4>Probability</h4>
      <p>{summary_prob}</p>
    </div>
    <div class="summary-tile">
      <h4>Top Drivers</h4>
      <p>{summary_reason}</p>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with right:
    st.markdown("### Model Stats")
    st.markdown("- 95% Accuracy")
    st.markdown("- Attendance is top driver (~65%)")
    st.markdown("- Trained with data of 500+ students")
    st.divider()
    st.markdown("### Quick Guidance")
    st.markdown("- Attendance below 70% increases risk.")
    st.markdown("- Low internal test scores are strong signals.")
    st.markdown("- Participation and assignments help offset risk.")
    st.markdown("### System Notes")
    st.markdown("- Ensemble model with calibrated risk score.")
    st.markdown("- Inputs normalized and validated in real time.")
    st.markdown("- Actionable feedback auto-generated per student.")

st.caption("VIMEET | AI Foundation and Its Applications")
