#!/usr/bin/env python3
import streamlit as st

from model import predict_student


st.set_page_config(page_title="Student Risk Checker", layout="wide")


def parse_probability(probability_text: str) -> float:
    return float(probability_text.replace("%", "")) if probability_text else 0.0


def get_status_message(risk_label: str, probability: float) -> tuple[str, str]:
    if "SAFE" in risk_label:
        return "On Track", "The student currently looks stable based on the entered details."
    if probability >= 75:
        return "Needs Attention", "The student may need support soon because several warning signs are present."
    return "Needs Attention", "The student shows some warning signs and should be monitored."


def build_action_steps(attendance, test1, test2, assignments, participation, gpa):
    actions = []

    if attendance < 75:
        actions.append("Improve attendance and try to stay above 75%.")
    if min(test1, test2) < 15:
        actions.append("Give extra focus to internal test preparation.")
    if assignments < 12:
        actions.append("Complete assignments on time and improve assignment quality.")
    if participation <= 2:
        actions.append("Encourage more class participation and engagement.")
    if gpa < 6.5:
        actions.append("Provide extra academic support for the coming weeks.")

    if not actions:
        actions.append("Keep following the current study plan and review progress regularly.")

    return actions[:3]


def clean_reasons(reasons):
    replacements = {
        "Internal Test1": "Internal Test 1",
        "Internal Test2": "Internal Test 2",
        "Prev Sem Gpa": "previous GPA",
        "Attendance Pct": "attendance",
        "Assignments Avg": "assignment score",
        "Participation": "class participation",
        "Risk Score": "overall performance",
        "Test Avg": "average test score",
    }

    cleaned = []
    for reason in reasons:
        text = reason
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = text.replace(" is a key risk factor", "")
        cleaned.append(text.strip())

    if cleaned == ["All indicators are within acceptable range"]:
        return ["No major warning signs were found."]

    return cleaned


st.markdown(
    """
<style>
:root {
    color-scheme: light;
}
body, .stApp, .block-container, .main, .css-1l02zno {
    background: #eef4fb !important;
    color: #0f2130 !important;
    font-family: Inter, sans-serif !important;
}

/* Page container spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1440px;
}

/* Metric cards */
[data-testid="stMetric"], .css-1v3fvcr, .css-1d391kg, .css-13sdm1f {
    background: #ffffff !important;
    border: 1px solid #d7dae0 !important;
    border-radius: 18px !important;
    padding: 18px 20px !important;
    box-shadow: 0 16px 35px rgba(15, 23, 42, 0.08) !important;
}

/* Buttons */
.stButton>button {
    background-color: #0f4c81 !important;
    color: #000000 !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 0.95rem 1.2rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
}
.stButton>button:hover {
    background-color: #133f67 !important;
}

/* Headings and text */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #0f2130 !important;
}

/* Alerts and info boxes */
div[data-testid="stInfo"], div[data-testid="stWarning"], div[data-testid="stSuccess"], div[data-testid="stError"] {
    background-color: rgba(255,255,255,0.94) !important;
    border: 1px solid rgba(15, 23, 42, 0.12) !important;
    color: #0f2130 !important;
}
div[data-testid="stInfo"] p, div[data-testid="stWarning"] p, div[data-testid="stSuccess"] p, div[data-testid="stError"] p {
    color: #0f2130 !important;
}

/* Slider labels and values */
.css-1v0mbdj input, .css-10trblm, .css-190qc4u {
    color: #0f2130 !important;
}

/* Keep text visible around cards */
section > div > div > div {
    opacity: 1 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Student Risk Checker")
st.write("Enter the student's current performance details to get a simple and readable summary.")

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Student Details")
    st.caption("Adjust the values below and click `Check Result`.")

    with st.form("student_inputs", clear_on_submit=False):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            attendance = st.slider("Attendance (%)", 40.0, 100.0, 75.0, format="%.0f")
            test1 = st.slider("Internal Test 1", 0.0, 25.0, 18.0)
            assignments = st.slider("Assignments", 0.0, 20.0, 14.0)
        with c2:
            test2 = st.slider("Internal Test 2", 0.0, 25.0, 20.0)
            participation = st.slider("Class Participation", 1, 5, 3)
            gpa = st.slider("Previous GPA", 4.0, 10.0, 7.5)

        submitted = st.form_submit_button("Check Result", use_container_width=True)

    if submitted:
        result = predict_student(test1, test2, attendance, assignments, participation, gpa)
        probability = parse_probability(result["probability"])
        short_reasons = clean_reasons(result["reasons"])
        status_title, status_text = get_status_message(result["risk"], probability)
        action_steps = build_action_steps(attendance, test1, test2, assignments, participation, gpa)

        st.divider()
        st.subheader("Result Summary")

        m1, m2 = st.columns(2, gap="large")
        with m1:
            st.metric("Current Status", status_title)
        with m2:
            st.metric("Risk Chance", result["probability"])

        if status_title == "On Track":
            st.success(status_text)
        else:
            st.warning(status_text)

        st.write("**What this means**")
        st.write(status_text)

        st.write("**Main things affecting the result**")
        for reason in short_reasons:
            st.write(f"- {reason}")

        st.write("**Suggested next steps**")
        for step in action_steps:
            st.write(f"- {step}")
    else:
        st.info("Use the sliders above and then click **Check Result** to generate the risk summary.")

with right:
    st.subheader("How To Read This")
    st.success("On Track: the student currently looks stable.")
    st.warning("Needs Attention: the student may need support.")
    st.write("A higher Risk Chance means there are more warning signs.")

    st.subheader("What Helps Most")
    st.markdown("- Better attendance\n- Stronger internal test scores\n- Consistent assignment completion\n- Better class participation")

    st.subheader("Tip")
    st.info("Use this result as a quick support guide, not as a final judgment.")

st.caption("Simple academic risk summary for easier student review.")
