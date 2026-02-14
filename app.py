#!/usr/bin/env python3
import streamlit as st
from model import predict_student

st.set_page_config(page_title="Academic Risk EWS", layout="wide")
st.title("🚨 Academic Risk Early Warning System")
st.markdown("*VAP Capstone | 95% Accuracy | Live Demo*")

st.sidebar.header("📊 Enter Student Data")
attendance = st.sidebar.slider("Attendance %", 40.0, 100.0, 75.0)
test1 = st.sidebar.slider("Internal Test 1 (/25)", 0.0, 25.0, 18.0)
test2 = st.sidebar.slider("Internal Test 2 (/25)", 0.0, 25.0, 20.0)
assignments = st.sidebar.slider("Assignments (/20)", 0.0, 20.0, 14.0)
participation = st.sidebar.slider("Participation (1-5)", 1, 5, 3)
gpa = st.sidebar.slider("Previous GPA", 4.0, 10.0, 7.5)

if st.sidebar.button("🔮 Predict Risk", use_container_width=True):
    result = predict_student(attendance, test1, test2, assignments, participation, gpa)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 📈 Result")
        if "SAFE" in result['risk']:
            st.success(result['risk'])
        else:
            st.error(result['risk'])
    
    with col2:
        st.metric("Risk Probability", result['probability'])
        st.info(f"**Reasons:** {', '.join(result['reasons'])}")

st.sidebar.markdown("""
### 📈 Model Stats
- **95% Accuracy**
- **Attendance = #1** (65%)
- **Trained with data of 500+ students**
""")

st.caption("🎓 VIMEET| AI Foundation And Its Applications")
