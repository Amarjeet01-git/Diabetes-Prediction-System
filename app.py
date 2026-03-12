import streamlit as st
import numpy as np
import pickle
import os
from datetime import datetime

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f1117 100%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }

.hero { text-align:center; padding:2rem 1rem 1.2rem; }
.hero-icon { font-size:3rem; animation:pulse 2.5s ease-in-out infinite; display:block; }
@keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.08)} }
.hero h1 {
    font-size:2.2rem; font-weight:700;
    background:linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0.3rem 0;
}
.hero p { color:#94a3b8; font-size:0.95rem; margin:0; }

.card {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:18px; padding:1.6rem 1.8rem; margin-bottom:1.2rem;
}
.card-title {
    font-size:0.7rem; font-weight:600; letter-spacing:0.12em;
    text-transform:uppercase; color:#60a5fa; margin-bottom:1rem;
}

div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label {
    color:#cbd5e1 !important; font-size:0.88rem !important; font-weight:500 !important;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background:rgba(255,255,255,0.06) !important;
    border:1px solid rgba(255,255,255,0.12) !important;
    border-radius:10px !important; color:#f1f5f9 !important; font-size:0.95rem !important;
}

div[data-testid="stButton"] > button {
    width:100%;
    background:linear-gradient(135deg,#3b82f6,#8b5cf6) !important;
    color:white !important; border:none !important;
    border-radius:12px !important; padding:0.72rem 0 !important;
    font-size:1rem !important; font-weight:600 !important;
    transition:opacity 0.2s !important;
}
div[data-testid="stButton"] > button:hover { opacity:0.85 !important; }

.result-positive {
    background:linear-gradient(135deg,rgba(239,68,68,0.15),rgba(220,38,38,0.08));
    border:1px solid rgba(239,68,68,0.35);
    border-radius:16px; padding:1.6rem; text-align:center; animation:fadeIn 0.5s ease;
}
.result-negative {
    background:linear-gradient(135deg,rgba(34,197,94,0.15),rgba(22,163,74,0.08));
    border:1px solid rgba(34,197,94,0.35);
    border-radius:16px; padding:1.6rem; text-align:center; animation:fadeIn 0.5s ease;
}
@keyframes fadeIn { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
.result-icon  { font-size:2.6rem; margin-bottom:0.4rem; }
.result-title { font-size:1.35rem; font-weight:700; margin:0 0 0.3rem; }
.result-sub   { font-size:0.85rem; color:#94a3b8; margin:0; }

.prob-label   { font-size:0.8rem; color:#94a3b8; margin-bottom:0.35rem; }
.prob-bar-bg  { background:rgba(255,255,255,0.08); border-radius:99px; height:9px; overflow:hidden; margin-bottom:0.2rem; }
.prob-bar-fill{ height:9px; border-radius:99px; }

.hist-card {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:14px; padding:0.9rem 1.1rem; margin-bottom:0.8rem;
}
.hist-card-d { border-left:4px solid #f87171; }
.hist-card-n { border-left:4px solid #4ade80; }
.hist-name   { font-weight:600; color:#f1f5f9; font-size:0.95rem; }
.hist-time   { font-size:0.72rem; color:#475569; margin-top:2px; }
.badge-d { background:rgba(239,68,68,0.2); color:#f87171; padding:2px 10px; border-radius:99px; font-size:0.72rem; font-weight:600; }
.badge-n { background:rgba(34,197,94,0.2);  color:#4ade80; padding:2px 10px; border-radius:99px; font-size:0.72rem; font-weight:600; }

.det-grid {
    display:grid; grid-template-columns:repeat(4,1fr);
    gap:0.4rem 1rem; margin-top:0.8rem; padding-top:0.8rem;
    border-top:1px solid rgba(255,255,255,0.07);
}
.det-label { font-size:0.68rem; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; }
.det-value { font-size:0.88rem; color:#e2e8f0; font-weight:600; font-family:'DM Mono',monospace; }

section[data-testid="stSidebar"] {
    background:rgba(15,17,23,0.95) !important;
    border-right:1px solid rgba(255,255,255,0.07) !important;
}
hr { border-color:rgba(255,255,255,0.07) !important; margin:1.2rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Model load failed: {e}")
    st.info("Please run `python train_model.py` locally and push `diabetes_model.pkl` to your repo.")
    st.stop()

# ── Session State ─────────────────────────────────────────────────────────────
# NOTE: History stored in session_state only (memory) — works on Streamlit Cloud
if "history"         not in st.session_state:
    st.session_state.history        = []
if "last_result"     not in st.session_state:
    st.session_state.last_result    = None
if "expanded_cards"  not in st.session_state:
    st.session_state.expanded_cards = {}

# ── SIDEBAR — History ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.6rem'>
        <span style='font-size:1.3rem'>📋</span>
        <span style='font-size:1.1rem;font-weight:700;color:#f1f5f9;margin-left:8px'>Patient History</span>
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.history

    if history:
        st.markdown(
            f"<p style='color:#64748b;font-size:0.8rem;margin-bottom:0.8rem'>"
            f"{len(history)} record(s) this session</p>",
            unsafe_allow_html=True,
        )

        for display_idx, record in enumerate(reversed(history)):
            real_idx  = len(history) - 1 - display_idx
            is_d      = record["result"] == 1
            badge     = '<span class="badge-d">Diabetic</span>' if is_d else '<span class="badge-n">Not Diabetic</span>'
            c_class   = "hist-card-d" if is_d else "hist-card-n"
            key_exp   = f"exp_{real_idx}"

            st.markdown(f"""
            <div class="hist-card {c_class}">
                <div style='display:flex;justify-content:space-between;align-items:center'>
                    <div>
                        <div class="hist-name">👤 {record['name']}</div>
                        <div class="hist-time">🕐 {record['time']}</div>
                    </div>
                    <div>{badge}</div>
                </div>
                <div style='margin-top:0.6rem'>
                    <div class="prob-label">Diabetic {record['prob_d']}%</div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{record['prob_d']}%;background:#f87171"></div>
                    </div>
                    <div class="prob-label" style="margin-top:0.4rem">Not Diabetic {record['prob_nd']}%</div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{record['prob_nd']}%;background:#4ade80"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            is_expanded  = st.session_state.expanded_cards.get(key_exp, False)
            toggle_label = "🙈 Hide Details" if is_expanded else "👁️ Show Details"

            if st.button(toggle_label, key=f"toggle_{real_idx}"):
                st.session_state.expanded_cards[key_exp] = not is_expanded
                st.rerun()

            if is_expanded:
                st.markdown(f"""
                <div class="hist-card {c_class}" style="margin-top:-0.5rem;padding-top:0.6rem">
                    <div class="det-grid">
                        <div><div class="det-label">Pregnancies</div><div class="det-value">{record['pregnancies']}</div></div>
                        <div><div class="det-label">Glucose</div><div class="det-value">{record['glucose']}</div></div>
                        <div><div class="det-label">Blood Pressure</div><div class="det-value">{record['bp']}</div></div>
                        <div><div class="det-label">Skin Thickness</div><div class="det-value">{record['skin']}</div></div>
                        <div><div class="det-label">Insulin</div><div class="det-value">{record['insulin']}</div></div>
                        <div><div class="det-label">BMI</div><div class="det-value">{record['bmi']}</div></div>
                        <div><div class="det-label">DPF</div><div class="det-value">{record['dpf']}</div></div>
                        <div><div class="det-label">Age</div><div class="det-value">{record['age']}</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All History"):
            st.session_state.history        = []
            st.session_state.expanded_cards = {}
            st.session_state.last_result    = None
            st.rerun()
    else:
        st.markdown("""
        <div style='text-align:center;padding:2rem 0;color:#475569'>
            <div style='font-size:2rem'>🗂️</div>
            <p style='font-size:0.85rem;margin-top:0.5rem'>No records yet.<br>Run a prediction to start.</p>
        </div>
        """, unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-icon">🩺</span>
    <h1>Diabetes Predictor</h1>
    <p>Enter patient details and click Predict to get an instant ML-powered result.</p>
</div>
""", unsafe_allow_html=True)

# Patient Name
st.markdown('<div class="card"><div class="card-title">👤 Patient Information</div>', unsafe_allow_html=True)
patient_name = st.text_input("Patient Name", placeholder="e.g. AmarJeet thakur")
st.markdown('</div>', unsafe_allow_html=True)

# Diagnostics
st.markdown('<div class="card"><div class="card-title">🔬 Medical Diagnostics</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    pregnancies    = st.number_input("Pregnancies",          min_value=0,   max_value=20,   value=1,    step=1)
    glucose        = st.number_input("Glucose (mg/dL)",      min_value=0,   max_value=300,  value=110)
with c2:
    blood_pressure = st.number_input("Blood Pressure (mmHg)",min_value=0,   max_value=200,  value=72)
    skin_thickness = st.number_input("Skin Thickness (mm)",  min_value=0,   max_value=100,  value=20)
with c3:
    insulin        = st.number_input("Insulin (μU/mL)",      min_value=0,   max_value=900,  value=80)
    bmi            = st.number_input("BMI",                  min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
with c4:
    dpf            = st.number_input("Diabetes Pedigree Fn", min_value=0.0, max_value=3.0,  value=0.5,  format="%.3f")
    age            = st.number_input("Age (years)",          min_value=1,   max_value=120,  value=30,   step=1)
st.markdown('</div>', unsafe_allow_html=True)

# Predict Button
predict_clicked = st.button("🔍 Predict Diabetes")

if predict_clicked:
    name     = patient_name.strip() or "Unknown Patient"
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, dpf, age]])
    prediction = model.predict(features)[0]
    probs      = model.predict_proba(features)[0]
    prob_d     = round(float(probs[1]) * 100, 2)
    prob_nd    = round(float(probs[0]) * 100, 2)

    record = {
        "name": name, "result": int(prediction),
        "time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "glucose": glucose, "bmi": bmi, "age": age,
        "pregnancies": pregnancies, "bp": blood_pressure,
        "skin": skin_thickness, "insulin": insulin,
        "dpf": dpf, "prob_d": prob_d, "prob_nd": prob_nd,
    }
    st.session_state.last_result = record
    st.session_state.history.append(record)
    st.rerun()

# ── Result shown below form ───────────────────────────────────────────────────
if st.session_state.last_result:
    r    = st.session_state.last_result
    is_d = r["result"] == 1
    st.markdown("<hr>", unsafe_allow_html=True)

    if is_d:
        st.markdown(f"""
        <div class="result-positive">
            <div class="result-icon">⚠️</div>
            <div class="result-title" style="color:#f87171">The person is diabetic</div>
            <p class="result-sub">Patient: <strong style="color:#f1f5f9">{r['name']}</strong>
            — Please consult a healthcare professional immediately.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <div class="result-icon">✅</div>
            <div class="result-title" style="color:#4ade80">The person is not diabetic</div>
            <p class="result-sub">Patient: <strong style="color:#f1f5f9">{r['name']}</strong>
            — No strong indicators found. Keep maintaining a healthy lifestyle!</p>
        </div>""", unsafe_allow_html=True)

    # Confidence bars
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">📊 Prediction Confidence</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <p class="prob-label">Diabetic probability</p>
    <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{r['prob_d']}%;background:#f87171"></div></div>
    <p class="prob-label" style="margin-bottom:1rem">{r['prob_d']}%</p>
    <p class="prob-label">Non-diabetic probability</p>
    <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{r['prob_nd']}%;background:#4ade80"></div></div>
    <p class="prob-label">{r['prob_nd']}%</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input summary
    st.markdown('<div class="card"><div class="card-title">🧾 Input Summary</div>', unsafe_allow_html=True)
    cols = st.columns(8)
    fields = [
        ("Pregnancies", r["pregnancies"]), ("Glucose",  r["glucose"]),
        ("BP",          r["bp"]),          ("Skin",     r["skin"]),
        ("Insulin",     r["insulin"]),     ("BMI",      r["bmi"]),
        ("DPF",         r["dpf"]),         ("Age",      r["age"]),
    ]
    for col, (label, val) in zip(cols, fields):
        with col:
            st.markdown(f"""
            <div style='text-align:center'>
                <div style='font-size:0.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em'>{label}</div>
                <div style='font-size:1.05rem;font-weight:700;color:#e2e8f0;font-family:"DM Mono",monospace'>{val}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<p style="text-align:center;color:#475569;font-size:0.78rem;margin:0">
    Built with Streamlit · Logistic Regression · scikit-learn
    &nbsp;|&nbsp; <em>For educational purposes only — not medical advice.</em>
</p>
""", unsafe_allow_html=True)