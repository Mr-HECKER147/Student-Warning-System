#!/usr/bin/env python3
import json
import os
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from Model import predict_student

st.set_page_config(page_title="Academic Risk EWS", layout="wide", page_icon="🚨")

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🚨 Academic Risk Early Warning System")
st.markdown("*VAP Capstone · LightGBM · ViMEET*")
st.divider()

# ── Sidebar inputs ───────────────────────────────────────────────────────────
st.sidebar.header("📊 Enter Student Data")
attendance    = st.sidebar.slider("Attendance %",          40.0, 100.0, 75.0, step=0.5)
test1         = st.sidebar.slider("Internal Test 1 (/25)",  0.0,  25.0, 18.0, step=0.5)
test2         = st.sidebar.slider("Internal Test 2 (/25)",  0.0,  25.0, 20.0, step=0.5)
assignments   = st.sidebar.slider("Assignments (/20)",       0.0,  20.0, 14.0, step=0.5)
participation = st.sidebar.slider("Participation (1–5)",     1,     5,    3)
gpa           = st.sidebar.slider("Previous GPA (0–10)",     4.0,  10.0,  7.5, step=0.1)

# ── Model info from JSON ──────────────────────────────────────────────────────
model_info = {}
if os.path.exists('model/model_info.json'):
    with open('model/model_info.json') as f:
        model_info = json.load(f)

if model_info:
    st.sidebar.divider()
    st.sidebar.markdown("### 📈 Model Stats")
    st.sidebar.markdown(f"- **Framework:** {model_info.get('framework', 'LightGBM')}")
    st.sidebar.markdown(f"- **Test Accuracy:** {model_info.get('accuracy', 0):.2%}")
    st.sidebar.markdown(f"- **CV Accuracy:** {model_info.get('cv_accuracy_mean', 0):.2%} ± {model_info.get('cv_accuracy_std', 0):.2%}")
    st.sidebar.markdown(f"- **Training Samples:** {model_info.get('training_samples', 'N/A')}")
    st.sidebar.markdown(f"- **SMOTE Applied:** {'✅' if model_info.get('smote_applied') else '❌'}")
    st.sidebar.markdown(f"- **Trained:** {model_info.get('trained_on', 'N/A')}")

# ── Predict button ────────────────────────────────────────────────────────────
if st.sidebar.button("🔮 Predict Risk", use_container_width=True):
    try:
        result = predict_student(test1, test2, attendance, assignments, participation, gpa)

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            st.markdown("### 🔎 Result")
            if "SAFE" in result['risk']:
                st.success(f"✅ {result['risk']}")
            else:
                st.error(f"🚨 {result['risk']}")

        with col2:
            st.markdown("### 📊 Risk Probability")
            prob_val = float(result['probability'].strip('%')) / 100
            st.metric(label="At-Risk Probability", value=result['probability'],
                      delta=f"{'High' if prob_val > 0.5 else 'Low'} risk")

        with col3:
            st.markdown("### ⚠️ Key Risk Factors")
            for reason in result['reasons']:
                st.warning(f"• {reason}")

        st.divider()

        # ── SHAP bar chart ────────────────────────────────────────────────────
        if result.get('shap_values'):
            st.markdown("### 🧠 Feature Contributions (SHAP)")
            st.caption("Positive values push toward AT RISK. Negative values push toward SAFE.")

            shap_dict  = result['shap_values']
            feat_names = [k.replace('_', ' ').title() for k in shap_dict.keys()]
            shap_vals  = list(shap_dict.values())
            colors     = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_vals]

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(feat_names, shap_vals, color=colors)
            ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.set_xlabel("SHAP Value (impact on risk prediction)")
            ax.set_title("Why this prediction?")
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ── Input summary ─────────────────────────────────────────────────────
        st.markdown("### 📋 Input Summary")
        test_avg   = (test1 + test2) / 2
        risk_score = attendance * 0.35 + test_avg * 0.30 + gpa * 0.20 + assignments * 0.15
        summary_data = {
            'Feature': ['Attendance %', 'Test 1 /25', 'Test 2 /25',
                        'Assignments /20', 'Participation', 'GPA',
                        'Test Average', 'Risk Score'],
            'Value':   [f"{attendance:.1f}%", f"{test1:.1f}", f"{test2:.1f}",
                        f"{assignments:.1f}", str(participation), f"{gpa:.1f}",
                        f"{test_avg:.2f}", f"{risk_score:.2f}"]
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

else:
    st.info("👈 Enter student data in the sidebar and click **Predict Risk** to get started.")

    if model_info and 'feature_importances' in model_info:
        st.markdown("### 🏆 Model Feature Importances")
        fi = model_info['feature_importances']
        feat_names = [k.replace('_', ' ').title() for k in fi.keys()]
        feat_vals  = list(fi.values())
        total      = sum(feat_vals)
        pct        = [v / total * 100 for v in feat_vals]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(feat_names, pct, color='#4f98a3')
        ax.set_xlabel("Importance (%)")
        ax.set_title("LightGBM Feature Importances")
        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.caption("🎓 ViMEET · Foundations of AI VAP · Academic Risk Early Warning System")
