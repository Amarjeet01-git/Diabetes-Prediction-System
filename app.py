import streamlit as st
import numpy as np
import pickle

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f1117 100%);
        min-height: 100vh;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Hero banner */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero-icon {
        font-size: 3.5rem;
        line-height: 1;
        margin-bottom: 0.6rem;
        animation: pulse 2.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50%       { transform: scale(1.08); }
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.4rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 400;
        margin: 0;
    }

    /* Card wrapper */
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.4rem;
        backdrop-filter: blur(10px);
    }
    .card-title {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #60a5fa;
        margin-bottom: 1.2rem;
    }

    /* Streamlit number inputs */
    div[data-testid="stNumberInput"] label {
        color: #cbd5e1 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stNumberInput"] input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 2px rgba(96,165,250,0.2) !important;
    }

    /* Predict button */
    div[data-testid="stButton"] button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 0 !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        cursor: pointer !important;
        transition: opacity 0.2s !important;
        margin-top: 0.4rem;
    }
    div[data-testid="stButton"] button:hover {
        opacity: 0.88 !important;
    }

    /* Result boxes */
    .result-positive {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
        border: 1px solid rgba(239,68,68,0.35);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        animation: fadeIn 0.5s ease;
    }
    .result-negative {
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(22,163,74,0.08));
        border: 1px solid rgba(34,197,94,0.35);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        animation: fadeIn 0.5s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0);    }
    }
    .result-icon  { font-size: 3rem; margin-bottom: 0.5rem; }
    .result-title { font-size: 1.5rem; font-weight: 700; margin: 0 0 0.3rem; }
    .result-sub   { font-size: 0.9rem; color: #94a3b8; margin: 0; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }

    /* Probability bar */
    .prob-label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }
    .prob-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 99px;
        height: 10px;
        overflow: hidden;
        margin-bottom: 0.25rem;
    }
    .prob-bar-fill {
        height: 10px;
        border-radius: 99px;
        transition: width 0.6s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("diabetes_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️  Model file not found. Run `python train_model.py` first.")
    st.stop()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">🩺</div>
        <h1>Diabetes Predictor</h1>
        <p>Enter patient diagnostics below and get an instant ML-powered prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Input Form ─────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">🔬 Patient Diagnostics</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose     = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=110)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=72)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin     = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=80)
    bmi         = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf         = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age         = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict Button ─────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict Diabetes")

# ── Prediction Output ──────────────────────────────────────────────────────────
if predict_clicked:
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, dpf, age]])

    prediction   = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]  # [prob_no_diabetes, prob_diabetes]

    prob_diabetic     = probabilities[1] * 100
    prob_not_diabetic = probabilities[0] * 100

    st.markdown("<hr>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-positive">
                <div class="result-icon">⚠️</div>
                <div class="result-title" style="color:#f87171;">The person is diabetic</div>
                <p class="result-sub">The model detected patterns associated with diabetes. Please consult a healthcare professional.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-negative">
                <div class="result-icon">✅</div>
                <div class="result-title" style="color:#4ade80;">The person is not diabetic</div>
                <p class="result-sub">The model found no strong indicators of diabetes. Maintain a healthy lifestyle!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Probability bars
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">📊 Prediction Confidence</div>', unsafe_allow_html=True)

    color_d  = "#f87171"
    color_nd = "#4ade80"

    st.markdown(
        f"""
        <p class="prob-label">Diabetic probability</p>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{prob_diabetic:.1f}%; background:{color_d};"></div>
        </div>
        <p class="prob-label" style="margin-bottom:1rem;">{prob_diabetic:.1f}%</p>

        <p class="prob-label">Non-diabetic probability</p>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{prob_not_diabetic:.1f}%; background:{color_nd};"></div>
        </div>
        <p class="prob-label">{prob_not_diabetic:.1f}%</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:#475569; font-size:0.78rem; margin:0;">
        Built with Streamlit · Logistic Regression · scikit-learn
        &nbsp;|&nbsp; <em>For educational purposes only — not medical advice.</em>
    </p>
    """,
    unsafe_allow_html=True,
)