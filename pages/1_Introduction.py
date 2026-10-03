"""
pages/1_Introduction.py — Project overview page.
"""
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Introduction | Airline Complaint AI", page_icon="🏠", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #0ea5e9 100%);
        border-radius: 16px; padding: 48px 40px; color: white; margin-bottom: 32px;
        box-shadow: 0 20px 60px rgba(37,99,235,0.3);
    }
    .hero-banner h1 { font-size: 2.6rem; font-weight: 800; margin: 0 0 12px 0; }
    .hero-banner p  { font-size: 1.1rem; opacity: 0.9; line-height: 1.6; margin: 0; }
    .kpi-card {
        background: white; border-radius: 12px; padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 4px solid #2563eb; text-align: center;
    }
    .kpi-card .num  { font-size: 2rem; font-weight: 800; color: #1e3a8a; }
    .kpi-card .lbl  { font-size: 0.85rem; color: #64748b; margin-top: 4px; }
    .arch-box {
        background: #f8fafc; border-radius: 12px; padding: 20px;
        border: 1px solid #e2e8f0; font-size: 0.92rem;
    }
    .use-case-card {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border-radius: 10px; padding: 18px; margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>✈️ AI-Based Airline Complaint Severity Detection<br>&amp; Service Risk Classification</h1>
        <p>
            An end-to-end NLP system that analyses airline customer reviews, detects complaint
            severity (Low → Critical), and flags service-risk patterns — powered by Traditional ML,
            Deep Learning, and Transformer models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPI Cards ────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    ("23,171", "Airline Reviews"),
    ("4", "Severity Classes"),
    ("2", "Risk Classes"),
    ("8+", "Models Trained"),
    ("3", "Model Families"),
]
for col, (num, lbl) in zip([k1, k2, k3, k4, k5], kpis):
    col.markdown(
        f'<div class="kpi-card"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────────────────────
left, right = st.columns([1.4, 1], gap="large")

with left:
    st.subheader("📌 Problem Statement")
    st.markdown(
        """
        Airlines receive thousands of customer reviews daily across booking, in-flight, and
        ground services. Manual review triage is slow and inconsistent.

        This system **automatically**:
        - Classifies complaint severity into **Low / Medium / High / Critical**
        - Flags whether a review signals a **service risk** requiring intervention
        - Explains predictions with **highlighted complaint phrases**
        - Supports **bulk CSV prediction** for operations teams
        """
    )

    st.subheader("🎯 Objectives")
    objectives = [
        "Automate severity labelling of airline reviews",
        "Detect service-risk patterns before they escalate",
        "Compare ML, DL, and Transformer model families",
        "Provide explainable predictions (LIME / keyword highlights)",
        "Support real-time single-review and batch CSV prediction",
    ]
    for obj in objectives:
        st.markdown(f"✅ {obj}")

with right:
    st.subheader("💼 Business Use Cases")
    use_cases = [
        ("🚨 Critical Alert Routing", "Automatically escalate Critical reviews to supervisors"),
        ("📊 Airline Benchmarking", "Compare complaint rates across airlines and time"),
        ("🛡️ Risk Early Warning", "Flag service-risk patterns before customer churn"),
        ("📈 Quality Analytics", "Track severity trends by route, seat type, traveller type"),
        ("🤖 Review Triage", "Prioritise agent response queues by severity"),
    ]
    for title, desc in use_cases:
        st.markdown(
            f'<div class="use-case-card"><strong>{title}</strong><br>'
            f'<span style="color:#475569;font-size:0.9rem">{desc}</span></div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Dataset Summary ───────────────────────────────────────────────────────────
st.subheader("📂 Dataset Summary")

processed_path = ROOT / "data" / "processed" / "airline_processed.parquet"
raw_path = ROOT / "data" / "raw" / "Airline_review.csv"

# GitHub raw URL for cloud deployment (data not in repo)
GITHUB_RAW_URL = "https://github.com/Raghulsri9786506332/airline-complaint-ai/raw/main/data/raw/Airline_review.csv"

@st.cache_data(show_spinner="Loading dataset …")
def load_dataset():
    if processed_path.exists():
        return pd.read_parquet(processed_path)
    if raw_path.exists():
        return pd.read_csv(raw_path, encoding="utf-8", on_bad_lines="skip")
    # Cloud fallback: download from GitHub
    try:
        return pd.read_csv(GITHUB_RAW_URL, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        return None

df = load_dataset()

if df is not None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Reviews", f"{len(df):,}")
    with c2:
        st.metric("Airlines", df["Airline Name"].nunique() if "Airline Name" in df.columns else "—")
    with c3:
        st.metric("Columns", df.shape[1])

    st.markdown("**Dataset columns:**")
    cols_df = pd.DataFrame({"Column": df.columns.tolist(),
                             "Dtype": [str(d) for d in df.dtypes]})
    st.dataframe(cols_df, height=300, width="stretch")
else:
    st.info("Dataset not yet loaded. Run `train_pipeline.py` first.")

st.markdown("---")

# ── Architecture overview ─────────────────────────────────────────────────────
st.subheader("🏗️ System Architecture")
st.markdown(
    """
    <div class="arch-box">

    ```
    Raw Airline Reviews (CSV)
            │
            ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Phase 1: Dataset Inspection  →  Phase 2: Data Cleaning │
    │  Phase 3: Text Preprocessing  →  Phase 4: Label Gen     │
    │  Phase 5: EDA + Sentiment      →  Phase 6: Feature Eng  │
    └─────────────────────────────────────────────────────────┘
            │
            ▼
    ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐
    │ Traditional  │  │  Custom Deep     │  │  Pretrained           │
    │ ML Models    │  │  Learning        │  │  Transformers         │
    │              │  │                  │  │                       │
    │ Logistic Reg │  │  CNN             │  │  distilbert-base-     │
    │ Linear SVM   │  │  LSTM            │  │    uncased (primary)  │
    │ Naive Bayes  │  │  BiLSTM          │  │  bert-base-uncased    │
    │ Random Forest│  │                  │  │                       │
    │ XGBoost      │  └──────────────────┘  └──────────────────────┘
    └──────────────┘
            │
            ▼
    ┌─────────────────────────────────────────────┐
    │  Model Evaluation & Comparison Table        │
    │  Severity: Acc · Macro-F1 · Critical-Recall │
    │  Risk    : ROC-AUC · PR-AUC · FN Rate       │
    └─────────────────────────────────────────────┘
            │
            ▼
    ┌──────────────────────────────────────────────┐
    │  Streamlit App (3 Pages)                     │
    │  ├─ Introduction   ├─ EDA Dashboard          │
    │  └─ Prediction (Single Review + CSV Bulk)    │
    └──────────────────────────────────────────────┘
    ```

    </div>
    """,
    unsafe_allow_html=True,
)
