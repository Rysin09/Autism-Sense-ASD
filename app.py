import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutismSense · ASD Screening Tool",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = Path(__file__).parent / "models"
    with open(base / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(base / "label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, label_encoders = load_artifacts()

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --primary: #6C63FF;
    --secondary: #48CFAD;
    --danger: #FC5C7D;
    --bg-dark: #0D0F1A;
    --bg-card: #151828;
    --bg-card2: #1C2035;
    --text-primary: #E8EAF6;
    --text-muted: #7B82A8;
    --border: rgba(108, 99, 255, 0.2);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-dark);
    color: var(--text-primary);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}

/* Headers */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* Metric cards */
.metric-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-card .label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary);
}

/* Result box */
.result-positive {
    background: linear-gradient(135deg, rgba(252,92,125,0.15), rgba(252,92,125,0.05));
    border: 1px solid rgba(252,92,125,0.5);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, rgba(72,207,173,0.15), rgba(72,207,173,0.05));
    border: 1px solid rgba(72,207,173,0.5);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
}
.result-subtitle {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-top: 8px;
}

/* Info box */
.info-box {
    background: var(--bg-card2);
    border-left: 3px solid var(--primary);
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: var(--text-muted);
}

/* Score badge */
.score-badge {
    display: inline-block;
    background: rgba(108,99,255,0.2);
    border: 1px solid rgba(108,99,255,0.4);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: var(--primary);
    font-weight: 500;
}

/* Streamlit overrides */
.stSelectbox label, .stSlider label, .stRadio label {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6C63FF, #48CFAD) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    font-size: 1rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

div[data-testid="stMetric"] {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 24px 0;">
        <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800; color:#6C63FF;">🧩 AutismSense</div>
        <div style="font-size:0.75rem; color:#7B82A8; margin-top:4px;">ASD Early Screening Tool</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🔍 Screening Tool", "📊 About the Model", "ℹ️ How It Works"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; color:#7B82A8; line-height:1.6;">
        <b style="color:#E8EAF6;">Disclaimer</b><br>
        This tool is for educational and screening purposes only. It is <b>not</b> a clinical diagnosis. 
        Please consult a licensed healthcare professional for any medical concerns.
    </div>
    """, unsafe_allow_html=True)

# ─── AQ-10 Questions ─────────────────────────────────────────────────────────
AQ_QUESTIONS = [
    "I often notice small sounds when others do not",
    "I usually concentrate more on the whole picture, rather than the small details",
    "I find it easy to do more than one thing at once",
    "If there is an interruption, I can switch back to what I was doing very quickly",
    "I find it easy to 'read between the lines' when someone is talking to me",
    "I know how to tell if someone listening to me is getting bored",
    "When I'm reading a story, I find it difficult to work out the characters' intentions",
    "I like to collect information about categories of things",
    "I find it easy to work out what someone is thinking or feeling just by looking at their face",
    "I find it difficult to work out people's intentions",
]

SCORE_MAP = {"Definitely Agree": 1, "Slightly Agree": 1, "Slightly Disagree": 0, "Definitely Disagree": 0}
# Q2,3,4,5,6,9 are reverse-scored in standard AQ-10
REVERSE_Q = [1, 2, 3, 4, 5, 8]  # 0-indexed

# ─── PAGE: Screening Tool ─────────────────────────────────────────────────────
if page == "🔍 Screening Tool":
    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif; font-weight:800; font-size:2rem; margin-bottom:4px;">
        ASD Screening Assessment
    </h1>
    <p style="color:#7B82A8; margin-bottom:28px; font-size:0.95rem;">
        Answer the AQ-10 questionnaire and provide demographic details for a machine-learning-assisted screening result.
    </p>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown("#### 🧬 Demographic Information")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", min_value=1, max_value=100, value=25, step=1)
            gender = st.selectbox("Gender", ["m", "f"], format_func=lambda x: "Male" if x == "m" else "Female")
            jaundice = st.selectbox("Born with jaundice?", ["no", "yes"])
            used_app_before = st.selectbox("Used ASD screening app before?", ["no", "yes"])
        with c2:
            ethnicity = st.selectbox("Ethnicity", list(label_encoders["ethnicity"].classes_))
            country = st.selectbox("Country of Residence", list(label_encoders["contry_of_res"].classes_))
            austim = st.selectbox("Family member with ASD?", ["no", "yes"])
            relation = st.selectbox("Relation to subject", ["Self", "Others"])

        st.markdown("<div style='margin:20px 0 8px 0;'></div>", unsafe_allow_html=True)
        st.markdown("#### 📝 AQ-10 Questionnaire")
        st.markdown('<div class="info-box">For each statement, indicate how strongly it applies to you. Score range: 0–10.</div>', unsafe_allow_html=True)

        scores = []
        for i, q in enumerate(AQ_QUESTIONS):
            st.markdown(f"<div style='font-size:0.88rem; color:#7B82A8; margin-top:14px; margin-bottom:4px;'>Q{i+1}</div>", unsafe_allow_html=True)
            ans = st.radio(q, list(SCORE_MAP.keys()), horizontal=True, key=f"q{i}", label_visibility="visible")
            raw = SCORE_MAP[ans]
            score = (1 - raw) if i in REVERSE_Q else raw
            scores.append(score)

        result_sum = sum(scores)
        st.markdown(f"<div style='margin-top:16px;'>AQ-10 Score: <span class='score-badge'>{result_sum} / 10</span></div>", unsafe_allow_html=True)

        predict_btn = st.button("Run Screening →")

    with col_result:
        st.markdown("#### 🎯 Prediction Result")

        if predict_btn:
            # Encode categoricals
            gender_enc = label_encoders["gender"].transform([gender])[0]
            ethnicity_enc = label_encoders["ethnicity"].transform([ethnicity])[0]
            jaundice_enc = label_encoders["jaundice"].transform([jaundice])[0]
            austim_enc = label_encoders["austim"].transform([austim])[0]
            country_enc = label_encoders["contry_of_res"].transform([country])[0]
            used_app_enc = label_encoders["used_app_before"].transform([used_app_before])[0]
            relation_enc = label_encoders["relation"].transform([relation])[0]

            # Build feature array — same order as training
            features = np.array([[
                scores[0], scores[1], scores[2], scores[3], scores[4],
                scores[5], scores[6], scores[7], scores[8], scores[9],
                age, gender_enc, ethnicity_enc, jaundice_enc,
                austim_enc, country_enc, used_app_enc, result_sum, relation_enc
            ]])

            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            confidence = proba[prediction] * 100

            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <div style="font-size:2.5rem; margin-bottom:8px;">⚠️</div>
                    <div class="result-title" style="color:#FC5C7D;">ASD Traits Indicated</div>
                    <div class="result-subtitle">The model detected patterns associated with ASD.</div>
                    <div style="margin-top:16px; font-size:0.85rem; color:#FC5C7D;">
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.warning("**Please consult a licensed psychologist or psychiatrist** for a formal evaluation. This is not a diagnosis.")
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <div style="font-size:2.5rem; margin-bottom:8px;">✅</div>
                    <div class="result-title" style="color:#48CFAD;">No ASD Traits Detected</div>
                    <div class="result-subtitle">The model did not detect strong ASD-associated patterns.</div>
                    <div style="margin-top:16px; font-size:0.85rem; color:#48CFAD;">
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Probability gauge
            st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[1] * 100,
                title={"text": "ASD Probability (%)", "font": {"size": 14, "color": "#7B82A8"}},
                number={"suffix": "%", "font": {"size": 28, "color": "#E8EAF6"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#7B82A8"},
                    "bar": {"color": "#6C63FF"},
                    "bgcolor": "#151828",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(72,207,173,0.15)"},
                        {"range": [40, 65], "color": "rgba(255,196,0,0.15)"},
                        {"range": [65, 100], "color": "rgba(252,92,125,0.15)"},
                    ],
                    "threshold": {"line": {"color": "#FC5C7D", "width": 2}, "value": 50},
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(t=40, b=10, l=20, r=20),
                font=dict(color="#E8EAF6"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # AQ breakdown bar
            st.markdown("**AQ-10 Score Breakdown**")
            fig_bar = go.Figure(go.Bar(
                x=[f"Q{i+1}" for i in range(10)],
                y=scores,
                marker_color=["#6C63FF" if s == 1 else "#1C2035" for s in scores],
                marker_line_color="#6C63FF",
                marker_line_width=1,
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=180, margin=dict(t=10, b=10, l=0, r=0),
                yaxis=dict(range=[0, 1.2], showgrid=False, tickvals=[0, 1], ticktext=["0", "1"], tickfont=dict(color="#7B82A8")),
                xaxis=dict(tickfont=dict(color="#7B82A8")),
                font=dict(color="#E8EAF6"),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.markdown("""
            <div style="background:#151828; border:1px dashed rgba(108,99,255,0.3); border-radius:12px; 
                        padding:40px 24px; text-align:center; color:#7B82A8;">
                <div style="font-size:2.5rem; margin-bottom:12px;">🔬</div>
                <div style="font-size:0.9rem;">Complete the form and click <b style="color:#6C63FF;">Run Screening</b> to see your result here.</div>
            </div>
            """, unsafe_allow_html=True)

# ─── PAGE: About the Model ────────────────────────────────────────────────────
elif page == "📊 About the Model":
    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif; font-weight:800; font-size:2rem; margin-bottom:4px;">
        Model Performance & Insights
    </h1>
    <p style="color:#7B82A8; margin-bottom:28px;">
        Transparency about how the underlying ML model was built and evaluated.
    </p>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Test Accuracy", "82%", "#6C63FF"),
        ("ASD Recall", "64%", "#FC5C7D"),
        ("ASD Precision", "59%", "#48CFAD"),
        ("ASD F1-Score", "0.61", "#FFD700"),
    ]
    for col, (label, val, color) in zip([c1, c2, c3, c4], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value" style="color:{color};">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:28px 0 0 0;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 🏆 Model Comparison (5-Fold CV Accuracy)")
        models_data = {"Decision Tree": 0.796, "Random Forest": 0.841, "XGBoost": 0.823}
        fig_models = go.Figure(go.Bar(
            x=list(models_data.keys()),
            y=list(models_data.values()),
            marker_color=["#7B82A8", "#6C63FF", "#48CFAD"],
            text=[f"{v:.1%}" for v in models_data.values()],
            textposition="outside",
            textfont=dict(color="#E8EAF6"),
        ))
        fig_models.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(t=20, b=20, l=0, r=0),
            yaxis=dict(range=[0.7, 0.9], tickformat=".0%", gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#7B82A8")),
            xaxis=dict(tickfont=dict(color="#E8EAF6")),
            font=dict(color="#E8EAF6"),
            showlegend=False,
        )
        st.plotly_chart(fig_models, use_container_width=True)

    with col_right:
        st.markdown("#### 🔬 Pipeline Overview")
        steps = [
            ("📥 Data Ingestion", "Autism Screening dataset with AQ-10 scores + demographics"),
            ("🧹 Preprocessing", "Label encoding, missing value imputation, country standardization"),
            ("⚖️ SMOTE", "Synthetic oversampling to handle class imbalance"),
            ("🔧 Tuning", "RandomizedSearchCV · 5-fold CV · 20 iterations"),
            ("🌲 Best Model", "Random Forest (bootstrap=False, max_depth=20, n_estimators=50)"),
        ]
        for icon_title, desc in steps:
            st.markdown(f"""
            <div class="info-box">
                <b style="color:#E8EAF6;">{icon_title}</b><br>{desc}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### 📌 Key Findings")
    col_a, col_b = st.columns(2)
    findings = [
        ("Class Imbalance Handled", "SMOTE was applied to the training set to balance ASD vs Non-ASD classes before model fitting.", "#6C63FF"),
        ("Feature Importance", "AQ-10 questionnaire scores (A1–A10) and the aggregate `result` score were the strongest predictors.", "#48CFAD"),
        ("Precision-Recall Trade-off", "Model prioritizes recall (catching real ASD cases) over precision, appropriate for a screening context.", "#FC5C7D"),
        ("Generalization", "82% test accuracy is consistent with CV accuracy, indicating no significant overfitting.", "#FFD700"),
    ]
    for i, (title, desc, color) in enumerate(findings):
        col = col_a if i % 2 == 0 else col_b
        col.markdown(f"""
        <div style="background:#151828; border:1px solid rgba(255,255,255,0.07); border-top:3px solid {color}; 
                    border-radius:10px; padding:18px; margin-bottom:14px;">
            <div style="font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:600; color:#E8EAF6; margin-bottom:6px;">{title}</div>
            <div style="font-size:0.83rem; color:#7B82A8; line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ─── PAGE: How It Works ───────────────────────────────────────────────────────
elif page == "ℹ️ How It Works":
    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif; font-weight:800; font-size:2rem; margin-bottom:4px;">
        How It Works
    </h1>
    <p style="color:#7B82A8; margin-bottom:28px;">Understanding ASD screening and this tool's methodology.</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧩 What is ASD?")
        st.markdown("""
        <div style="color:#7B82A8; font-size:0.9rem; line-height:1.75;">
        Autism Spectrum Disorder (ASD) is a neurodevelopmental condition characterized by differences in social communication, 
        repetitive behaviors, and sensory processing. It exists on a spectrum — no two individuals experience it the same way.
        <br><br>
        Early identification significantly improves outcomes. Screening tools like the <b style="color:#E8EAF6;">AQ-10</b> provide 
        a fast, evidence-based way to flag individuals who may benefit from a formal clinical assessment.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin:20px 0 0 0;'></div>", unsafe_allow_html=True)
        st.markdown("#### 📋 AQ-10 Scoring Logic")
        st.markdown("""
        <div style="color:#7B82A8; font-size:0.9rem; line-height:1.75;">
        The AQ-10 (Autism Quotient – 10 item) is a validated screening instrument. Each item scores 0 or 1:
        <ul style="margin-top:8px; padding-left:20px;">
            <li>For <b style="color:#E8EAF6;">odd-numbered traits</b>: "Definitely/Slightly Agree" → 1 point</li>
            <li>For <b style="color:#E8EAF6;">even-numbered traits</b>: "Definitely/Slightly Disagree" → 1 point (reverse scored)</li>
        </ul>
        A score ≥ 6 is typically considered a positive screening result and warrants further clinical evaluation.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### ⚙️ How the Prediction Works")
        steps_hw = [
            ("1", "You fill the AQ-10 questionnaire — 10 behavioral questions scored 0 or 1.", "#6C63FF"),
            ("2", "Demographic features (age, gender, ethnicity, etc.) are collected and encoded.", "#48CFAD"),
            ("3", "The aggregate AQ-10 score and all 18 features are passed to the Random Forest model.", "#FFD700"),
            ("4", "The model outputs a probability for ASD (class 1) and No ASD (class 0).", "#FC5C7D"),
            ("5", "If predicted class = 1, a recommendation to seek clinical evaluation is shown.", "#6C63FF"),
        ]
        for num, text, color in steps_hw:
            st.markdown(f"""
            <div style="display:flex; gap:14px; margin-bottom:14px; align-items:flex-start;">
                <div style="min-width:32px; height:32px; border-radius:50%; background:{color}22; 
                            border:1px solid {color}55; display:flex; align-items:center; justify-content:center;
                            font-family:'Syne',sans-serif; font-weight:700; color:{color}; font-size:0.85rem;">{num}</div>
                <div style="color:#7B82A8; font-size:0.88rem; line-height:1.6; padding-top:5px;">{text}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin:8px 0 0 0;'></div>", unsafe_allow_html=True)
        st.markdown("#### ⚠️ Limitations")
        st.markdown("""
        <div style="background:rgba(252,92,125,0.08); border:1px solid rgba(252,92,125,0.3); border-radius:10px; 
                    padding:16px; font-size:0.86rem; color:#7B82A8; line-height:1.7;">
            <ul style="padding-left:18px; margin:0;">
                <li>Trained on a limited public dataset — may not generalize across all populations.</li>
                <li>ASD recall is 64% — the model misses ~36% of true ASD cases.</li>
                <li>Self-reported answers introduce subjective bias.</li>
                <li>This tool <b style="color:#FC5C7D;">cannot replace</b> clinical diagnosis by a qualified professional.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#7B82A8; font-size:0.8rem; padding:8px 0;">
        Built with Random Forest · scikit-learn · Streamlit · Plotly &nbsp;|&nbsp; 
        <b style="color:#6C63FF;">AutismSense</b> is for educational use only
    </div>
    """, unsafe_allow_html=True)
