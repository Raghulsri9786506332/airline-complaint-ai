"""
app.py — Streamlit multi-page application entry point.
Airline Complaint Severity & Service Risk AI System.
"""
import sys
from pathlib import Path

import streamlit as st

# ── Ensure project root is on path ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Complaint AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar branding ─────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="text-align:center; padding:12px 0 4px 0;">
        <span style="font-size:2.8rem;">✈️</span>
        <h2 style="margin:0; color:#1e3a8a; font-weight:800; letter-spacing:-0.5px;">
            AirComplaint AI
        </h2>
        <p style="color:#64748b; font-size:0.85rem; margin-top:4px;">
            Severity & Risk Classification
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Navigation**
    - 🏠 Introduction
    - 📊 EDA Dashboard
    - 🔍 Prediction
    """
)
st.sidebar.markdown("---")
st.sidebar.caption("Powered by ML · DL · Transformers")

# ── Landing content (the home page used to be blank apart from the sidebar) ──
st.title("✈️ Airline Complaint Severity & Service Risk AI")
st.markdown(
    """
    Use the pages in the sidebar:

    - **Introduction** — project overview and pipeline
    - **EDA Dashboard** — explore severity, ratings and trends
    - **Prediction** — classify a single review or upload a CSV

    > Models must be trained first: `python train_pipeline.py --skip-dl --skip-transformers`
    """
)
