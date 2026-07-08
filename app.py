"""
app.py
-------
Streamlit frontend for the Student Performance Prediction Model.
Run with:  streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — cards & accents only (base theme comes from
# .streamlit/config.toml, so all default widget text stays readable)
# ============================================================
st.markdown("""
<style>
    /* Headings */
    .main-title {
        font-size: 44px;
        font-weight: 800;
        text-align: center;
        color: #ffd200 !important;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #e6e6e6 !important;
        font-size: 17px;
        margin-top: 4px;
        margin-bottom: 25px;
    }

    /* Glass cards */
    .card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 25px;
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        margin-bottom: 20px;
        color: #ffffff !important;
    }
    .card * { color: #ffffff !important; }

    /* Result cards */
    .result-pass, .result-fail {
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        font-size: 30px;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .result-pass { background: linear-gradient(135deg, #11998e, #38ef7d); color: #08251e !important; }
    .result-fail { background: linear-gradient(135deg, #cb2d3e, #ef473a); color: #fff8f7 !important; }
    .result-pass span, .result-fail span { font-size: 18px; }
    .result-pass * , .result-fail * { color: inherit !important; }

    /* Suggestion boxes */
    .suggestion-box {
        background: rgba(255,255,255,0.10);
        border-left: 4px solid #ffd200;
        padding: 12px 18px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: #ffffff !important;
        font-size: 15px;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #f7971e, #ffd200);
        color: #16222a !important;
        font-weight: 700;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(255,210,0,0.4);
    }
    div.stButton > button p { color: #16222a !important; font-weight: 700; }

    /* Tabs label contrast */
    button[data-baseweb="tab"] p { color: #f5f5f5 !important; font-weight: 600; }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("student_prediction_model.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# ============================================================
# ENCODING MAPS (must match notebook's LabelEncoder logic)
# ============================================================
GENDER_MAP = {"Female": 0, "Male": 1}
YES_NO_MAP = {"No": 0, "Yes": 1}
PARENTAL_MAP = {"Bachelors": 0, "High School": 1, "Masters": 2, "PhD": 3}

FEATURE_ORDER = [
    "Gender",
    "Study_Hours_per_Week",
    "Attendance_Rate",
    "Past_Exam_Scores",
    "Parental_Education_Level",
    "Internet_Access_at_Home",
    "Extracurricular_Activities",
]

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered Pass/Fail prediction using a tuned Random Forest model</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR — INPUT FORM
# ============================================================
with st.sidebar:
    st.markdown("## 📝 Student Details")
    st.write("Fill in the details below:")

    gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
    study_hours = st.slider("Study Hours per Week", 0, 60, 20)
    attendance = st.slider("Attendance Rate (%)", 0, 100, 80)
    past_scores = st.slider("Past Exam Scores", 0, 100, 70)
    parental_edu = st.selectbox("Parental Education Level", list(PARENTAL_MAP.keys()))
    internet = st.selectbox("Internet Access at Home", list(YES_NO_MAP.keys()))
    extracurricular = st.selectbox("Extracurricular Activities", list(YES_NO_MAP.keys()))

    predict_btn = st.button("🔮 Predict Result")

# ============================================================
# MAIN CONTENT
# ============================================================
tab1, tab2 = st.tabs(["📊 Prediction", "ℹ️ About Model"])

with tab1:
    if not model_loaded:
        st.error(
            "⚠️ Model file 'student_prediction_model.pkl' not found. "
            "Please place it in the same directory as app.py. Details: " + load_error
        )
    else:
        if predict_btn:
            input_dict = {
                "Gender": GENDER_MAP[gender],
                "Study_Hours_per_Week": study_hours,
                "Attendance_Rate": attendance,
                "Past_Exam_Scores": past_scores,
                "Parental_Education_Level": PARENTAL_MAP[parental_edu],
                "Internet_Access_at_Home": YES_NO_MAP[internet],
                "Extracurricular_Activities": YES_NO_MAP[extracurricular],
            }
            input_df = pd.DataFrame([input_dict])[FEATURE_ORDER]

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            confidence = np.max(probability) * 100

            col1, col2 = st.columns([1.1, 1])

            # ---------------- RESULT CARD ----------------
            with col1:
                if prediction == 1:
                    st.markdown(
                        f'<div class="result-pass">✅ PASS<br><span style="font-size:18px">Model Confidence: {confidence:.2f}%</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="result-fail">❌ FAIL<br><span style="font-size:18px">Model Confidence: {confidence:.2f}%</span></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Star rating logic (same as notebook)
                score = 0
                score += study_hours >= 15
                score += attendance >= 80
                score += past_scores >= 70
                score += YES_NO_MAP[internet] == 1
                score += YES_NO_MAP[extracurricular] == 1

                if score == 5:
                    stars, level = "⭐⭐⭐⭐⭐", "Excellent"
                elif score >= 3:
                    stars, level = "⭐⭐⭐", "Good"
                else:
                    stars, level = "⭐", "Needs Improvement"

                st.markdown(
                    f'<div class="card" style="text-align:center;">'
                    f'<span style="font-size:28px">{stars}</span><br>'
                    f'<span style="color:#ffd200; font-weight:700;">Performance Level: {level}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # ---------------- CONFIDENCE GAUGE ----------------
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    number={'suffix': "%", 'font': {'color': 'white', 'size': 40}},
                    title={'text': "Model Confidence", 'font': {'color': 'white', 'size': 18}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': 'white'},
                        'bar': {'color': "#ffd200"},
                        'bgcolor': "rgba(255,255,255,0.05)",
                        'steps': [
                            {'range': [0, 50], 'color': "#cb2d3e"},
                            {'range': [50, 80], 'color': "#f7971e"},
                            {'range': [80, 100], 'color': "#38ef7d"},
                        ],
                    }
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white"},
                    height=320,
                    margin=dict(t=50, b=10, l=20, r=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            # ---------------- PERSONALIZED SUGGESTIONS ----------------
            st.markdown("### 💡 Personalized Suggestions")
            if prediction == 1:
                st.markdown(
                    '<div class="suggestion-box">🎉 Great job! Keep maintaining your study habits, '
                    'attendance and engagement — you are on track to pass.</div>',
                    unsafe_allow_html=True,
                )
            else:
                suggestions = []
                if study_hours < 10:
                    suggestions.append("📚 Increase study hours to at least 10–15 hours per week.")
                if attendance < 75:
                    suggestions.append("🏫 Improve attendance. Try to maintain above 75%.")
                if past_scores < 50:
                    suggestions.append("📝 Practice previous exam papers and revise weak subjects.")
                if YES_NO_MAP[internet] == 0:
                    suggestions.append("🌐 Try to use online learning resources.")
                if YES_NO_MAP[extracurricular] == 0:
                    suggestions.append("🎭 Participate in extracurricular activities to build confidence.")
                if PARENTAL_MAP[parental_edu] == 1:  # High School
                    suggestions.append("🧑‍🏫 Seek extra guidance from teachers or mentors whenever possible.")

                if not suggestions:
                    suggestions.append("Keep working hard — small consistent improvements will help!")

                for s in suggestions:
                    st.markdown(f'<div class="suggestion-box">{s}</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Fill in the details in the sidebar and click **Predict Result**.")

with tab2:
    st.markdown("""
    <div class="card">
    <h4>📌 About this Model</h4>
    <p>This app uses a <b>Random Forest Classifier</b>, tuned with GridSearchCV,
    trained on a student performance dataset. It predicts whether a student will
    <b>Pass</b> or <b>Fail</b> based on:</p>
    <ul>
        <li>Gender</li>
        <li>Study Hours per Week</li>
        <li>Attendance Rate</li>
        <li>Past Exam Scores</li>
        <li>Parental Education Level</li>
        <li>Internet Access at Home</li>
        <li>Extracurricular Activities</li>
    </ul>
    <p>Model file: <code>student_prediction_model.pkl</code> (produced by <code>train_model.py</code>,
    the same pipeline as the original notebook).</p>
    </div>
    """, unsafe_allow_html=True)
