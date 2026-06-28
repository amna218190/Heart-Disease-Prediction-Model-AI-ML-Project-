import streamlit as st
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* Root variables */
:root {
    --red:      #E63946;
    --red-soft: #FF6B6B;
    --dark:     #0D1117;
    --card:     #161B22;
    --border:   #21262D;
    --text:     #E6EDF3;
    --muted:    #8B949E;
    --green:    #3FB950;
    --yellow:   #D29922;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--dark) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}

* {
    box-sizing: border-box;
}

[data-testid="stAppViewContainer"] {
    padding-top: 0.5rem;
}

[data-testid="stSidebar"] {
    background-color: var(--card) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebarContent"] {
    padding-top: 1rem;
}

/* Hide default streamlit elements */
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }

/* Headings */
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
}

/* Cards */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}

.card-red {
    background: linear-gradient(135deg, #1a0a0a 0%, #2d0f0f 100%);
    border: 1px solid #5a1515;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0D1117 0%, #1a0a0a 50%, #0d1117 100%);
    border: 1px solid #5a1515;
    border-radius: 16px;
    padding: 40px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '🫀';
    position: absolute;
    right: 32px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 80px;
    opacity: 0.12;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #fff;
    margin: 0 0 8px 0;
    line-height: 1.1;
}

.hero-title span {
    color: var(--red);
}

.hero-subtitle {
    font-size: 1rem;
    color: var(--muted);
    margin: 0;
    font-weight: 400;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
}

.metric-box {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--red);
    display: block;
}

.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
    display: block;
}

/* Section labels */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--red);
    margin-bottom: 8px;
}

/* Result boxes */
.result-safe {
    background: linear-gradient(135deg, #051a0a 0%, #0a2d15 100%);
    border: 2px solid var(--green);
    border-radius: 14px;
    padding: 32px;
    text-align: center;
}

.result-danger {
    background: linear-gradient(135deg, #1a0505 0%, #2d0a0a 100%);
    border: 2px solid var(--red);
    border-radius: 14px;
    padding: 32px;
    text-align: center;
}

.result-emoji {
    font-size: 3.5rem;
    display: block;
    margin-bottom: 12px;
}

.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 8px 0;
}

.result-sub {
    font-size: 0.95rem;
    color: var(--muted);
    margin: 0;
}

/* Risk meter */
.risk-bar-bg {
    background: var(--border);
    border-radius: 999px;
    height: 12px;
    margin: 10px 0;
    overflow: hidden;
}

.risk-bar-fill-low    { background: var(--green);  height: 100%; border-radius: 999px; transition: width 0.5s; }
.risk-bar-fill-medium { background: var(--yellow); height: 100%; border-radius: 999px; transition: width 0.5s; }
.risk-bar-fill-high   { background: var(--red);    height: 100%; border-radius: 999px; transition: width 0.5s; }

/* Risk factor pills */
.pill-danger  { display:inline-block; background:#3d0c0c; color:#ff6b6b; border:1px solid #7a1515; border-radius:20px; padding:4px 12px; font-size:0.8rem; margin:3px; }
.pill-safe    { display:inline-block; background:#0c2d15; color:#3fb950; border:1px solid #155e27; border-radius:20px; padding:4px 12px; font-size:0.8rem; margin:3px; }
.pill-neutral { display:inline-block; background:#1c1c2e; color:#8b949e; border:1px solid #30304a; border-radius:20px; padding:4px 12px; font-size:0.8rem; margin:3px; }

/* Sidebar inputs */
.sidebar-section {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--red);
    padding: 12px 0 6px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 10px;
}

/* Inputs */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label {
    color: var(--text) !important;
    font-size: 0.85rem !important;
}

div[data-baseweb="select"] > div {
    background-color: #1c2128 !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Predict button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--red) 0%, #c0392b 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    padding: 14px 32px;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
    letter-spacing: 0.02em;
}

div[data-testid="stButton"] > button:hover {
    opacity: 0.88;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border);
}

[data-testid="stTabs"] button {
    color: var(--muted) !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--text) !important;
    border-bottom: 2px solid var(--red) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Info box */
.info-box {
    background: #0d1f2d;
    border-left: 3px solid #1f6feb;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #8b949e;
    margin: 12px 0;
}

.warn-box {
    background: #1c1400;
    border-left: 3px solid var(--yellow);
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #8b949e;
    margin: 12px 0;
}

/* Factor grid */
.factor-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 12px;
}

.factor-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85rem;
}

.factor-name { color: var(--muted); }
.factor-val-yes { color: var(--red); font-weight: 600; }
.factor-val-no  { color: var(--green); font-weight: 600; }

/* Responsive helpers */
.table-wrap {
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}

.table-wrap table {
    min-width: 100%;
}

@media (max-width: 1024px) {
    .hero {
        padding: 32px 28px;
    }

    .hero-title {
        font-size: 2rem;
    }

    .metric-row {
        flex-wrap: wrap;
    }

    .metric-box {
        min-width: calc(50% - 6px);
    }
}

@media (max-width: 768px) {
    html, body, [data-testid="stAppViewContainer"] {
        font-size: 15px;
    }

    [data-testid="stAppViewContainer"] {
        padding-left: 0.2rem;
        padding-right: 0.2rem;
    }

    [data-testid="stSidebar"] {
        border-right: none;
        border-bottom: 1px solid var(--border);
    }

    [data-testid="stSidebarContent"] {
        padding: 0.6rem 0.8rem 1rem 0.8rem;
    }

    .hero {
        padding: 24px 20px;
        border-radius: 14px;
        margin-bottom: 18px;
    }

    .hero::before {
        right: 16px;
        font-size: 52px;
        opacity: 0.08;
    }

    .hero-title {
        font-size: 1.65rem;
    }

    .hero-subtitle {
        font-size: 0.92rem;
        max-width: 26rem;
    }

    .card, .card-red, .result-safe, .result-danger {
        padding: 18px;
    }

    .metric-row {
        gap: 8px;
    }

    .metric-box {
        min-width: 100%;
        padding: 14px;
    }

    .result-emoji {
        font-size: 2.8rem;
    }

    .result-title {
        font-size: 1.35rem;
    }

    .factor-grid {
        grid-template-columns: 1fr;
    }

    [data-testid="stTabs"] button {
        font-size: 0.82rem;
        padding: 0.45rem 0.25rem;
    }

    div[data-testid="stButton"] > button {
        padding: 12px 18px;
        font-size: 0.95rem;
    }

    .pill-danger,
    .pill-safe,
    .pill-neutral {
        font-size: 0.75rem;
        padding: 4px 10px;
        margin: 2px;
    }
}

@media (max-width: 480px) {
    .hero {
        padding: 20px 16px;
    }

    .hero-title {
        font-size: 1.45rem;
    }

    .hero-subtitle {
        font-size: 0.86rem;
    }

    .card, .card-red, .result-safe, .result-danger {
        padding: 16px;
        border-radius: 12px;
    }

    .section-label, .sidebar-section {
        letter-spacing: 0.08em;
    }

    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label,
    [data-testid="stNumberInput"] label {
        font-size: 0.8rem !important;
    }

    [data-testid="stTabs"] button {
        font-size: 0.74rem;
        padding: 0.4rem 0.15rem;
    }
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONFIGURATION - Pkl Files Directory
# ─────────────────────────────────────────────
# Get current app directory automatically
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = r"heart_model.pkl"
SCALER_FILE = r"scaler.pkl"

# ─────────────────────────────────────────────
# LOAD MODEL & SCALER
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_path  = os.path.join(APP_DIR, MODEL_FILE)
    scaler_path = os.path.join(APP_DIR, SCALER_FILE)
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

model, scaler = load_artifacts()


# ─────────────────────────────────────────────
# PREDICTION FUNCTION (matches notebook exactly)
# ─────────────────────────────────────────────
def predict_patient(model, scaler, p):
    bmi = p['BMI']

    def bmi_cat(b):
        if b < 18.5: return 0
        elif b < 25: return 1
        elif b < 30: return 2
        else:        return 3

    bmi_c      = bmi_cat(bmi)
    risk_score = (
        p['HighBP'] + p['HighChol'] + p['Smoker'] + p['Stroke'] + p['Diabetes'] +
        (1 if p['PhysActivity'] == 0 else 0) +
        (1 if p['GenHlth'] >= 4 else 0)
    )
    age_bp       = p['Age'] * p['HighBP']
    age_diabetes = p['Age'] * p['Diabetes']

    numerical_vals   = [[bmi, p['MentHlth'], p['PhysHlth'], risk_score, age_bp, age_diabetes]]
    categorical_vals = [[
        p['HighBP'], p['HighChol'], p['CholCheck'], p['Smoker'],
        p['Stroke'], p['Diabetes'], p['PhysActivity'], p['Fruits'],
        p['Veggies'], p['HvyAlcoholConsump'], p['AnyHealthcare'],
        p['NoDocbcCost'], p['GenHlth'], p['DiffWalk'], p['Sex'],
        p['Age'], p['Education'], p['Income'], bmi_c
    ]]

    num_scaled  = scaler.transform(numerical_vals)
    final_input = np.hstack((num_scaled, categorical_vals))
    prediction  = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0] if hasattr(model, 'predict_proba') else None

    return int(prediction), probability, risk_score, bmi_c


# ─────────────────────────────────────────────
# SIDEBAR — ALL INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="padding: 16px 0 8px 0;">', unsafe_allow_html=True)
    st.markdown("## 🫀 Patient Input")
    st.markdown('<p style="color:#8b949e; font-size:0.82rem; margin-top:-8px;">Fill in the patient\'s health information</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── VITALS
    st.markdown('<div class="sidebar-section">📊 Vitals</div>', unsafe_allow_html=True)
    bmi      = st.number_input("BMI", min_value=10.0, max_value=80.0, value=26.5, step=0.1,
                                help="Body Mass Index (weight/height²)")
    ment_hlth = st.slider("Poor Mental Health Days (last 30)", 0, 30, 3,
                           help="Days of poor mental health in the past month")
    phys_hlth = st.slider("Poor Physical Health Days (last 30)", 0, 30, 5,
                           help="Days of poor physical health in the past month")

    # ── CARDIOVASCULAR RISK
    st.markdown('<div class="sidebar-section">❤️ Cardiovascular</div>', unsafe_allow_html=True)

    high_bp   = st.selectbox("High Blood Pressure",    ["No", "Yes"])
    high_chol = st.selectbox("High Cholesterol",       ["No", "Yes"])
    chol_chk  = st.selectbox("Cholesterol Check (5yr)", ["No", "Yes"], index=1)
    stroke    = st.selectbox("History of Stroke",      ["No", "Yes"])

    # ── LIFESTYLE
    st.markdown('<div class="sidebar-section">🏃 Lifestyle</div>', unsafe_allow_html=True)

    smoker       = st.selectbox("Smoker (100+ cigarettes lifetime)", ["No", "Yes"])
    hvy_alcohol  = st.selectbox("Heavy Alcohol Consumption",         ["No", "Yes"])
    phys_act     = st.selectbox("Physically Active (last 30 days)",  ["No", "Yes"], index=1)
    fruits       = st.selectbox("Eats Fruit Daily",                  ["No", "Yes"], index=1)
    veggies      = st.selectbox("Eats Vegetables Daily",             ["No", "Yes"], index=1)

    # ── MEDICAL HISTORY
    st.markdown('<div class="sidebar-section">🏥 Medical History</div>', unsafe_allow_html=True)

    diabetes    = st.selectbox("Diabetes",               ["No", "Yes"])
    diff_walk   = st.selectbox("Difficulty Walking",     ["No", "Yes"])
    any_hlthcr  = st.selectbox("Has Healthcare Coverage",["No", "Yes"], index=1)
    no_doc_cost = st.selectbox("Couldn't See Doctor (Cost)", ["No", "Yes"])

    gen_hlth_map = {
        "Excellent (1)": 1, "Very Good (2)": 2,
        "Good (3)": 3, "Fair (4)": 4, "Poor (5)": 5
    }
    gen_hlth_label = st.selectbox("General Health", list(gen_hlth_map.keys()), index=2)
    gen_hlth = gen_hlth_map[gen_hlth_label]

    # ── DEMOGRAPHICS
    st.markdown('<div class="sidebar-section">👤 Demographics</div>', unsafe_allow_html=True)

    sex_map  = {"Female": 0, "Male": 1}
    sex_lbl  = st.selectbox("Sex", list(sex_map.keys()))
    sex      = sex_map[sex_lbl]

    age_map = {
        "18–24 (1)":1, "25–29 (2)":2, "30–34 (3)":3,
        "35–39 (4)":4, "40–44 (5)":5, "45–49 (6)":6,
        "50–54 (7)":7, "55–59 (8)":8, "60–64 (9)":9,
        "65–69 (10)":10,"70–74 (11)":11,"75–79 (12)":12,"80+ (13)":13
    }
    age_lbl = st.selectbox("Age Group", list(age_map.keys()), index=6)
    age     = age_map[age_lbl]

    edu_map = {
        "Never attended (1)":1, "Elementary (2)":2,
        "Some High School (3)":3, "High School Graduate (4)":4,
        "Some College (5)":5, "College Graduate (6)":6
    }
    edu_lbl = st.selectbox("Education Level", list(edu_map.keys()), index=4)
    education = edu_map[edu_lbl]

    inc_map = {
        "< $10K (1)":1, "$10K–$15K (2)":2, "$15K–$20K (3)":3,
        "$20K–$25K (4)":4, "$25K–$35K (5)":5, "$35K–$50K (6)":6,
        "$50K–$75K (7)":7, "> $75K (8)":8
    }
    inc_lbl = st.selectbox("Annual Income", list(inc_map.keys()), index=5)
    income  = inc_map[inc_lbl]

    st.markdown("---")

    # Model selector
    model_choice = st.selectbox(
        "Model",
        ["Best (Auto)", "Neural Network", "KNN", "Naive Bayes"],
        help="Select which trained model to use for prediction"
    )

    predict_btn = st.button("🔍 Run Prediction", use_container_width=True)


# ─────────────────────────────────────────────
# HELPER — convert Yes/No to 0/1
# ─────────────────────────────────────────────
def yn(s): return 1 if s == "Yes" else 0

patient = {
    'HighBP': yn(high_bp), 'HighChol': yn(high_chol),
    'CholCheck': yn(chol_chk), 'BMI': bmi,
    'Smoker': yn(smoker), 'Stroke': yn(stroke),
    'Diabetes': yn(diabetes), 'PhysActivity': yn(phys_act),
    'Fruits': yn(fruits), 'Veggies': yn(veggies),
    'HvyAlcoholConsump': yn(hvy_alcohol), 'AnyHealthcare': yn(any_hlthcr),
    'NoDocbcCost': yn(no_doc_cost), 'GenHlth': gen_hlth,
    'MentHlth': ment_hlth, 'PhysHlth': phys_hlth,
    'DiffWalk': yn(diff_walk), 'Sex': sex,
    'Age': age, 'Education': education, 'Income': income,
}


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <p class="hero-title">Heart Disease<br><span>Risk Predictor</span></p>
    <p class="hero-subtitle">BRFSS 2015 · CDC Dataset · 253,680 patient records · Neural Network, KNN & Naive Bayes</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍  Prediction", "📊  Model Info", "ℹ️  Feature Guide"])

# ──────────────────────────────
# TAB 1 — PREDICTION
# ──────────────────────────────
with tab1:

    if not predict_btn:
        # Default state — show instructions
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="card">
                <div class="section-label">Step 1</div>
                <h4 style="margin:4px 0 8px 0; font-family:'Space Grotesk',sans-serif;">Fill the sidebar</h4>
                <p style="color:#8b949e; font-size:0.85rem; margin:0;">Enter the patient's health indicators, lifestyle factors, and demographics on the left panel.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="card">
                <div class="section-label">Step 2</div>
                <h4 style="margin:4px 0 8px 0; font-family:'Space Grotesk',sans-serif;">Choose a model</h4>
                <p style="color:#8b949e; font-size:0.85rem; margin:0;">Select Neural Network, KNN, or Naive Bayes. "Best (Auto)" picks the highest-accuracy model automatically.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="card">
                <div class="section-label">Step 3</div>
                <h4 style="margin:4px 0 8px 0; font-family:'Space Grotesk',sans-serif;">Run Prediction</h4>
                <p style="color:#8b949e; font-size:0.85rem; margin:0;">Click the button to get an instant prediction with confidence score and risk breakdown.</p>
            </div>
            """, unsafe_allow_html=True)

        if model is None:
            st.markdown("""
            <div class="warn-box">
                ⚠️ <strong>No model files found.</strong> Make sure <code>heart_model.pkl</code> and <code>scaler.pkl</code>
                are in the same folder as <code>app.py</code>. Download them from your Colab notebook (Section 9).
            </div>
            """, unsafe_allow_html=True)
        else:
            model_type = type(model).__name__
            st.markdown(f"""
            <div class="info-box">
                ✅ Model loaded: <strong>{model_type}</strong> · expects <strong>{model.n_features_in_} features</strong>
            </div>
            """, unsafe_allow_html=True)

    else:
        # ── RUN PREDICTION
        if model is None:
            st.error("❌ Model not loaded. Place `heart_model.pkl` and `scaler.pkl` in the app folder.")
        else:
            try:
                pred, proba, risk_score, bmi_cat_val = predict_patient(model, scaler, patient)
                risk_pct = int(proba[1] * 100) if proba is not None else (100 if pred == 1 else 10)
                conf_pct = int(max(proba) * 100) if proba is not None else 0

                col_res, col_detail = st.columns([1, 1], gap="large")

                with col_res:
                    # Result box
                    if pred == 0:
                        st.markdown(f"""
                        <div class="result-safe">
                            <span class="result-emoji">✅</span>
                            <h2 class="result-title" style="color:#3fb950;">Low Risk</h2>
                            <p class="result-sub">No significant heart disease risk detected based on the provided indicators.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-danger">
                            <span class="result-emoji">⚠️</span>
                            <h2 class="result-title" style="color:#E63946;">Elevated Risk</h2>
                            <p class="result-sub">Heart disease risk indicators detected. Clinical evaluation is strongly recommended.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Confidence + Risk probability
                    st.markdown(f"""
                    <div class="card">
                        <div class="section-label">Risk Probability</div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-size:0.85rem; color:#8b949e;">Heart Disease Risk</span>
                            <span style="font-family:'Space Grotesk',sans-serif; font-weight:700; color:{'#E63946' if risk_pct > 50 else '#3fb950'}; font-size:1.2rem;">{risk_pct}%</span>
                        </div>
                        <div class="risk-bar-bg">
                            <div class="{'risk-bar-fill-high' if risk_pct > 60 else 'risk-bar-fill-medium' if risk_pct > 35 else 'risk-bar-fill-low'}" style="width:{risk_pct}%;"></div>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:16px; padding-top:14px; border-top:1px solid #21262d;">
                            <div style="text-align:center;">
                                <span style="font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:700; color:#E6EDF3;">{conf_pct}%</span>
                                <span style="display:block; font-size:0.72rem; color:#8b949e; margin-top:2px;">Model Confidence</span>
                            </div>
                            <div style="text-align:center;">
                                <span style="font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:700; color:#E6EDF3;">{risk_score}/7</span>
                                <span style="display:block; font-size:0.72rem; color:#8b949e; margin-top:2px;">Risk Score</span>
                            </div>
                            <div style="text-align:center;">
                                <span style="font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:700; color:#E6EDF3;">{'Obese' if bmi_cat_val==3 else 'Overweight' if bmi_cat_val==2 else 'Normal' if bmi_cat_val==1 else 'Under'}</span>
                                <span style="display:block; font-size:0.72rem; color:#8b949e; margin-top:2px;">BMI Category</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_detail:
                    # Risk factor breakdown
                    st.markdown('<div class="section-label">Risk Factor Breakdown</div>', unsafe_allow_html=True)

                    risk_factors = [
                        ("High Blood Pressure", patient['HighBP']),
                        ("High Cholesterol",    patient['HighChol']),
                        ("Diabetes",            patient['Diabetes']),
                        ("Smoker",              patient['Smoker']),
                        ("History of Stroke",   patient['Stroke']),
                        ("Physically Inactive", 1 - patient['PhysActivity']),
                        ("Heavy Alcohol Use",   patient['HvyAlcoholConsump']),
                        ("Difficulty Walking",  patient['DiffWalk']),
                    ]

                    pills_html = ""
                    for name, val in risk_factors:
                        if val == 1:
                            pills_html += f'<span class="pill-danger">⚠ {name}</span>'
                        else:
                            pills_html += f'<span class="pill-safe">✓ {name}</span>'

                    st.markdown(f'<div style="margin-bottom:16px;">{pills_html}</div>', unsafe_allow_html=True)

                    # Protective factors
                    st.markdown('<div class="section-label" style="margin-top:12px;">Protective Factors</div>', unsafe_allow_html=True)
                    prot_html = ""
                    if patient['PhysActivity']: prot_html += '<span class="pill-safe">✓ Physically Active</span>'
                    if patient['Fruits']:       prot_html += '<span class="pill-safe">✓ Eats Fruit Daily</span>'
                    if patient['Veggies']:      prot_html += '<span class="pill-safe">✓ Eats Vegetables</span>'
                    if patient['AnyHealthcare']:prot_html += '<span class="pill-safe">✓ Has Healthcare</span>'
                    if not patient['HvyAlcoholConsump']: prot_html += '<span class="pill-safe">✓ Moderate Alcohol</span>'
                    if not prot_html:
                        prot_html = '<span class="pill-neutral">No protective factors noted</span>'
                    st.markdown(f'<div style="margin-bottom:16px;">{prot_html}</div>', unsafe_allow_html=True)

                    # Patient summary card
                    bmi_labels = ["Underweight", "Normal Weight", "Overweight", "Obese"]
                    gen_hlth_labels = {1:"Excellent", 2:"Very Good", 3:"Good", 4:"Fair", 5:"Poor"}
                    st.markdown(f"""
                    <div class="card" style="margin-top:8px;">
                        <div class="section-label">Patient Summary</div>
                        <div class="table-wrap">
                            <table style="width:100%; font-size:0.83rem; border-collapse:collapse;">
                                <tr>
                                    <td style="color:#8b949e; padding:4px 0;">BMI</td>
                                    <td style="text-align:right; color:#E6EDF3; font-weight:500;">{bmi:.1f} — {bmi_labels[bmi_cat_val]}</td>
                                </tr>
                                <tr>
                                    <td style="color:#8b949e; padding:4px 0;">Age Group</td>
                                    <td style="text-align:right; color:#E6EDF3; font-weight:500;">{age_lbl}</td>
                                </tr>
                                <tr>
                                    <td style="color:#8b949e; padding:4px 0;">General Health</td>
                                    <td style="text-align:right; color:#E6EDF3; font-weight:500;">{gen_hlth_labels[gen_hlth]}</td>
                                </tr>
                                <tr>
                                    <td style="color:#8b949e; padding:4px 0;">Mental Health Days</td>
                                    <td style="text-align:right; color:#E6EDF3; font-weight:500;">{ment_hlth} / 30</td>
                                </tr>
                                <tr>
                                    <td style="color:#8b949e; padding:4px 0;">Physical Health Days</td>
                                    <td style="text-align:right; color:#E6EDF3; font-weight:500;">{phys_hlth} / 30</td>
                                </tr>
                                <tr>
                                    <td style="color:#8b949e; padding:4px 0;">Risk Score</td>
                                    <td style="text-align:right; color:{'#E63946' if risk_score >= 4 else '#D29922' if risk_score >= 2 else '#3fb950'}; font-weight:600;">{risk_score} / 7</td>
                                </tr>
                            </table>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Disclaimer
                st.markdown("""
                <div class="warn-box" style="margin-top:20px;">
                    ⚕️ <strong>Medical Disclaimer:</strong> This tool is for academic purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Make sure the model and scaler were trained with the same features defined in the notebook.")


# ──────────────────────────────
# TAB 2 — MODEL INFO
# ──────────────────────────────
with tab2:
    st.markdown("### Model & Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="section-label">Dataset</div>
            <h4 style="margin:4px 0 16px 0; font-family:'Space Grotesk',sans-serif;">BRFSS 2015 — CDC</h4>
            <div class="table-wrap">
                <table style="width:100%; font-size:0.85rem; border-collapse:collapse;">
                    <tr><td style="color:#8b949e; padding:5px 0;">Total Records</td><td style="text-align:right; font-weight:500;">253,680</td></tr>
                    <tr><td style="color:#8b949e; padding:5px 0;">After Cleaning</td><td style="text-align:right; font-weight:500;">~240,000+</td></tr>
                    <tr><td style="color:#8b949e; padding:5px 0;">Total Features</td><td style="text-align:right; font-weight:500;">25 (21 raw + 4 engineered)</td></tr>
                    <tr><td style="color:#8b949e; padding:5px 0;">Target</td><td style="text-align:right; font-weight:500;">HeartDiseaseorAttack</td></tr>
                    <tr><td style="color:#8b949e; padding:5px 0;">Train Split</td><td style="text-align:right; font-weight:500;">70%</td></tr>
                    <tr><td style="color:#8b949e; padding:5px 0;">Val / Test Split</td><td style="text-align:right; font-weight:500;">15% / 15%</td></tr>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="section-label">Engineered Features</div>
            <div class="table-wrap">
                <table style="width:100%; font-size:0.84rem; border-collapse:collapse;">
                    <tr>
                        <td style="color:#8b949e; padding:5px 0;">BMI_category</td>
                        <td style="text-align:right; font-weight:500; color:#E6EDF3;">WHO BMI class (0–3)</td>
                    </tr>
                    <tr>
                        <td style="color:#8b949e; padding:5px 0;">risk_score</td>
                        <td style="text-align:right; font-weight:500; color:#E6EDF3;">Composite score (0–7)</td>
                    </tr>
                    <tr>
                        <td style="color:#8b949e; padding:5px 0;">age_bp</td>
                        <td style="text-align:right; font-weight:500; color:#E6EDF3;">Age × HighBP interaction</td>
                    </tr>
                    <tr>
                        <td style="color:#8b949e; padding:5px 0;">age_diabetes</td>
                        <td style="text-align:right; font-weight:500; color:#E6EDF3;">Age × Diabetes interaction</td>
                    </tr>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="section-label">Models Trained</div>
            <div class="table-wrap">
                <table style="width:100%; font-size:0.85rem; border-collapse:collapse;">
                    <tr>
                        <td style="color:#8b949e; padding:6px 0; border-bottom:1px solid #21262d;">KNN</td>
                        <td style="text-align:right; color:#E6EDF3; border-bottom:1px solid #21262d;">Best K selected via validation accuracy</td>
                    </tr>
                    <tr>
                        <td style="color:#8b949e; padding:6px 0; border-bottom:1px solid #21262d;">Naive Bayes</td>
                        <td style="text-align:right; color:#E6EDF3; border-bottom:1px solid #21262d;">Gaussian NB, probabilistic</td>
                    </tr>
                    <tr>
                        <td style="color:#8b949e; padding:6px 0;">Neural Network</td>
                        <td style="text-align:right; color:#E6EDF3;">MLP (256→128→64→32), Adam</td>
                    </tr>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="section-label">Neural Network Architecture</div>
            <div class="table-wrap">
                <table style="width:100%; font-size:0.84rem; border-collapse:collapse;">
                    <tr><td style="color:#8b949e; padding:4px 0;">Hidden Layers</td><td style="text-align:right; font-weight:500;">(256, 128, 64, 32)</td></tr>
                    <tr><td style="color:#8b949e; padding:4px 0;">Activation</td><td style="text-align:right; font-weight:500;">ReLU</td></tr>
                    <tr><td style="color:#8b949e; padding:4px 0;">Optimizer</td><td style="text-align:right; font-weight:500;">Adam</td></tr>
                    <tr><td style="color:#8b949e; padding:4px 0;">Regularization</td><td style="text-align:right; font-weight:500;">L2 (α=0.0001)</td></tr>
                    <tr><td style="color:#8b949e; padding:4px 0;">Batch Size</td><td style="text-align:right; font-weight:500;">512</td></tr>
                    <tr><td style="color:#8b949e; padding:4px 0;">Learning Rate</td><td style="text-align:right; font-weight:500;">Adaptive (init 0.001)</td></tr>
                    <tr><td style="color:#8b949e; padding:4px 0;">Early Stopping</td><td style="text-align:right; font-weight:500;">Yes (20 no-change epochs)</td></tr>
                    <tr><td style="color:#8b949e; padding:4px 0;">Max Iterations</td><td style="text-align:right; font-weight:500;">500</td></tr>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Preprocessing steps
    st.markdown("""
    <div class="card" style="margin-top:8px;">
        <div class="section-label">Preprocessing Pipeline</div>
        <div style="display:flex; gap:0; margin-top:12px; flex-wrap:wrap;">
            <div style="flex:1; min-width:150px; text-align:center; padding:12px;">
                <div style="font-size:1.4rem;">📥</div>
                <div style="font-size:0.8rem; color:#E6EDF3; font-weight:600; margin-top:6px;">Load CSV</div>
                <div style="font-size:0.72rem; color:#8b949e;">BRFSS 2015</div>
            </div>
            <div style="flex:0; align-self:center; color:#8b949e;">→</div>
            <div style="flex:1; min-width:150px; text-align:center; padding:12px;">
                <div style="font-size:1.4rem;">🧹</div>
                <div style="font-size:0.8rem; color:#E6EDF3; font-weight:600; margin-top:6px;">Clean</div>
                <div style="font-size:0.72rem; color:#8b949e;">Dedup + BMI filter</div>
            </div>
            <div style="flex:0; align-self:center; color:#8b949e;">→</div>
            <div style="flex:1; min-width:150px; text-align:center; padding:12px;">
                <div style="font-size:1.4rem;">⚙️</div>
                <div style="font-size:0.8rem; color:#E6EDF3; font-weight:600; margin-top:6px;">Feature Eng.</div>
                <div style="font-size:0.72rem; color:#8b949e;">4 new features</div>
            </div>
            <div style="flex:0; align-self:center; color:#8b949e;">→</div>
            <div style="flex:1; min-width:150px; text-align:center; padding:12px;">
                <div style="font-size:1.4rem;">✂️</div>
                <div style="font-size:0.8rem; color:#E6EDF3; font-weight:600; margin-top:6px;">Split</div>
                <div style="font-size:0.72rem; color:#8b949e;">70 / 15 / 15 %</div>
            </div>
            <div style="flex:0; align-self:center; color:#8b949e;">→</div>
            <div style="flex:1; min-width:150px; text-align:center; padding:12px;">
                <div style="font-size:1.4rem;">⚖️</div>
                <div style="font-size:0.8rem; color:#E6EDF3; font-weight:600; margin-top:6px;">Scale</div>
                <div style="font-size:0.72rem; color:#8b949e;">StandardScaler (num only)</div>
            </div>
            <div style="flex:0; align-self:center; color:#8b949e;">→</div>
            <div style="flex:1; min-width:150px; text-align:center; padding:12px;">
                <div style="font-size:1.4rem;">🤖</div>
                <div style="font-size:0.8rem; color:#E6EDF3; font-weight:600; margin-top:6px;">Train</div>
                <div style="font-size:0.72rem; color:#8b949e;">KNN, NB, MLP</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────
# TAB 3 — FEATURE GUIDE
# ──────────────────────────────
with tab3:
    st.markdown("### Feature Reference Guide")
    st.markdown('<p style="color:#8b949e; font-size:0.9rem;">Complete description of every input feature used in the model.</p>', unsafe_allow_html=True)

    features_info = [
        ("HighBP",            "Binary",  "0 = No, 1 = Yes",                    "Has the patient been told they have high blood pressure?"),
        ("HighChol",          "Binary",  "0 = No, 1 = Yes",                    "Has the patient been told they have high cholesterol?"),
        ("CholCheck",         "Binary",  "0 = No, 1 = Yes",                    "Had a cholesterol check in the past 5 years?"),
        ("BMI",               "Numeric", "10.0 – 80.0",                         "Body Mass Index calculated from height and weight."),
        ("Smoker",            "Binary",  "0 = No, 1 = Yes",                    "Has the patient smoked at least 100 cigarettes in their lifetime?"),
        ("Stroke",            "Binary",  "0 = No, 1 = Yes",                    "Has the patient ever been told they had a stroke?"),
        ("Diabetes",          "Binary",  "0 = No, 1 = Yes",                    "Has the patient been told they have diabetes?"),
        ("PhysActivity",      "Binary",  "0 = No, 1 = Yes",                    "Physical activity in the past 30 days (not counting work)?"),
        ("Fruits",            "Binary",  "0 = No, 1 = Yes",                    "Does the patient consume fruit one or more times per day?"),
        ("Veggies",           "Binary",  "0 = No, 1 = Yes",                    "Does the patient consume vegetables one or more times per day?"),
        ("HvyAlcoholConsump", "Binary",  "0 = No, 1 = Yes",                    "Heavy drinker? (Men: >14/week, Women: >7/week)"),
        ("AnyHealthcare",     "Binary",  "0 = No, 1 = Yes",                    "Does the patient have any kind of health care coverage?"),
        ("NoDocbcCost",       "Binary",  "0 = No, 1 = Yes",                    "Couldn't see a doctor in past 12 months due to cost?"),
        ("GenHlth",           "Ordinal", "1=Excellent → 5=Poor",               "How would the patient rate their general health?"),
        ("MentHlth",          "Numeric", "0 – 30 days",                        "Days of poor mental health in the past 30 days."),
        ("PhysHlth",          "Numeric", "0 – 30 days",                        "Days of poor physical health in the past 30 days."),
        ("DiffWalk",          "Binary",  "0 = No, 1 = Yes",                    "Does the patient have serious difficulty walking or climbing stairs?"),
        ("Sex",               "Binary",  "0 = Female, 1 = Male",               "Biological sex of the patient."),
        ("Age",               "Ordinal", "1=18-24 → 13=80+",                   "Age category in 5-year bands. 13 categories total."),
        ("Education",         "Ordinal", "1=Never → 6=College Graduate",       "Highest education level completed."),
        ("Income",            "Ordinal", "1=<$10K → 8=>$75K",                  "Annual household income category."),
    ]

    for feat, ftype, values, desc in features_info:
        type_color = "#1f6feb" if ftype == "Binary" else "#388bfd" if ftype == "Ordinal" else "#bc8cff"
        st.markdown(f"""
        <div style="display:flex; gap:14px; padding:10px 0; border-bottom:1px solid #21262d; align-items:flex-start;">
            <div style="min-width:160px;">
                <span style="font-family:'Space Grotesk',sans-serif; font-weight:600; color:#E6EDF3; font-size:0.88rem;">{feat}</span>
                <span style="display:inline-block; margin-left:8px; background:{type_color}22; color:{type_color}; font-size:0.68rem; padding:2px 7px; border-radius:10px; border:1px solid {type_color}44;">{ftype}</span>
            </div>
            <div style="min-width:160px; font-size:0.8rem; color:#8b949e;">{values}</div>
            <div style="flex:1; font-size:0.83rem; color:#c9d1d9;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:20px;">
        📌 <strong>Source:</strong> Behavioral Risk Factor Surveillance System (BRFSS) 2015, Centers for Disease Control and Prevention (CDC).
        Dataset available at <a href="https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset" target="_blank" style="color:#58a6ff;">Kaggle</a>.
    </div>
    """, unsafe_allow_html=True)
