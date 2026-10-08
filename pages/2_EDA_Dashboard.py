"""
pages/2_EDA_Dashboard.py — Exploratory Data Analysis interactive dashboard.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="EDA Dashboard | Airline Complaint AI",
                   page_icon="📊", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.dash-header {
    background: linear-gradient(135deg, #1e40af, #0ea5e9);
    color: white; border-radius: 14px; padding: 28px 36px; margin-bottom: 24px;
}
.dash-header h1 { margin: 0; font-size: 2rem; font-weight: 800; }
.section-card {
    background: #f8fafc; border-radius: 12px; padding: 16px;
    border: 1px solid #e2e8f0; margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dash-header">
  <h1>📊 EDA Dashboard</h1>
  <p style="margin:4px 0 0 0; opacity:0.9">Explore complaint severity trends, sentiment, and service aspects</p>
</div>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
GITHUB_RAW_URL = "https://github.com/Raghulsri9786506332/airline-complaint-ai/raw/main/data/raw/Airline_review.csv"

@st.cache_data(show_spinner="Loading dataset …")
def load_data() -> pd.DataFrame | None:
    processed = ROOT / "data" / "processed" / "airline_processed.parquet"
    raw = ROOT / "data" / "raw" / "Airline_review.csv"

    if processed.exists():
        df = pd.read_parquet(processed)
    elif raw.exists():
        df = pd.read_csv(raw, encoding="utf-8", on_bad_lines="skip")
        df["Overall_Rating"] = pd.to_numeric(df["Overall_Rating"], errors="coerce")
    else:
        # Cloud fallback: download from GitHub
        try:
            df = pd.read_csv(GITHUB_RAW_URL, encoding="utf-8", on_bad_lines="skip")
            df["Overall_Rating"] = pd.to_numeric(df["Overall_Rating"], errors="coerce")
        except Exception:
            return None

    # Add derived columns if missing
    if "severity_label" not in df.columns:
        from src.data.label_generator import generate_labels
        from src.data.cleaner import clean_dataset
        df = clean_dataset(df)
        df = generate_labels(df)
    return df


df = load_data()

if df is None:
    st.error("Dataset not found. Please place `Airline_review.csv` in `data/raw/`.")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("🔎 Filters")
SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]
COLORS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#f97316", "Critical": "#ef4444"}
sev_filter = st.sidebar.multiselect(
    "Severity", SEVERITY_ORDER, default=SEVERITY_ORDER,
)
if "Airline Name" in df.columns:
    top_airlines = df["Airline Name"].value_counts().nlargest(15).index.tolist()
    airline_filter = st.sidebar.multiselect("Airlines (top 15)", top_airlines, default=top_airlines[:5])
else:
    airline_filter = []

df_f = df[df["severity_label"].isin(sev_filter)] if "severity_label" in df.columns else df
if airline_filter and "Airline Name" in df.columns:
    df_f = df_f[df_f["Airline Name"].isin(airline_filter)]

st.caption(f"Showing **{len(df_f):,}** of **{len(df):,}** reviews after filters.")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# Row 1 — Severity & Rating distributions
# ═══════════════════════════════════════════════════════════════════════════════
r1c1, r1c2 = st.columns(2)

with r1c1:
    st.subheader("Severity Distribution")
    if "severity_label" in df_f.columns:
        sev_counts = df_f["severity_label"].value_counts().reindex(SEVERITY_ORDER).fillna(0)
        fig = px.bar(
            x=sev_counts.index.tolist(), y=sev_counts.values.tolist(),
            color=sev_counts.index.tolist(), color_discrete_map=COLORS,
            labels={"x": "Severity", "y": "Count"},
            title="",
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white", height=320)
        st.plotly_chart(fig, width="stretch")

with r1c2:
    st.subheader("Overall Rating Distribution")
    if "Overall_Rating" in df_f.columns:
        rating_counts = df_f["Overall_Rating"].dropna().astype(int).value_counts().sort_index()
        fig2 = px.bar(
            x=rating_counts.index.tolist(), y=rating_counts.values.tolist(),
            labels={"x": "Rating", "y": "Count"},
            color_discrete_sequence=["#2563eb"],
        )
        fig2.update_layout(plot_bgcolor="white", height=320)
        st.plotly_chart(fig2, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# Row 2 — Airline severity heatmap + Risk distribution
# ═══════════════════════════════════════════════════════════════════════════════
r2c1, r2c2 = st.columns([1.6, 1])

with r2c1:
    st.subheader("Severity by Airline")
    if "Airline Name" in df_f.columns and "severity_label" in df_f.columns:
        top_n = 10
        top_airlines_data = df_f["Airline Name"].value_counts().nlargest(top_n).index
        heat_df = (
            df_f[df_f["Airline Name"].isin(top_airlines_data)]
            .groupby(["Airline Name", "severity_label"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=SEVERITY_ORDER, fill_value=0)
        )
        fig3 = px.imshow(
            heat_df, text_auto=True, aspect="auto",
            color_continuous_scale="RdYlGn_r",
            labels={"x": "Severity", "y": "Airline"},
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, width="stretch")

with r2c2:
    st.subheader("Service Risk Distribution")
    if "risk_flag" in df_f.columns:
        risk_map = {0: "No Risk", 1: "Risk"}
        rc = df_f["risk_flag"].map(risk_map).value_counts()
        fig4 = px.pie(
            names=rc.index, values=rc.values,
            color=rc.index,
            color_discrete_map={"Risk": "#ef4444", "No Risk": "#22c55e"},
            hole=0.55,
        )
        fig4.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom"))
        st.plotly_chart(fig4, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# Row 3 — Sentiment & Aspect analysis
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📐 Aspect Rating Distribution by Severity")

ASPECT_COLS = [c for c in [
    "Seat Comfort", "Cabin Staff Service", "Food & Beverages",
    "Ground Service", "Inflight Entertainment", "Wifi & Connectivity", "Value For Money",
] if c in df_f.columns]

if ASPECT_COLS and "severity_label" in df_f.columns:
    aspect_means = df_f.groupby("severity_label")[ASPECT_COLS].mean().reindex(SEVERITY_ORDER)
    fig5 = px.bar(
        aspect_means.T, barmode="group",
        color_discrete_sequence=["#22c55e", "#f59e0b", "#f97316", "#ef4444"],
        labels={"value": "Mean Rating", "variable": "Severity", "index": "Aspect"},
    )
    fig5.update_layout(plot_bgcolor="white", height=380)
    st.plotly_chart(fig5, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# Row 4 — Temporal trends
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📅 Temporal Complaint Trends")

date_col = None
for c in ["Review Date", "Date Flown"]:
    if c in df_f.columns:
        try:
            # Dates look like "11th November 2019" / "November 2019": strip ordinal suffixes,
            # otherwise pandas cannot parse them and the trend chart never shows.
            cleaned_dates = (
                df_f[c].astype(str)
                .str.replace(r"(?<=\d)(st|nd|rd|th)\b", "", regex=True)
                .str.strip()
            )
            parsed = pd.to_datetime(cleaned_dates, errors="coerce", format="mixed")
            if parsed.notna().sum() > 100:
                date_col = c
                df_f = df_f.copy()
                df_f["_parsed_date"] = parsed
                break
        except Exception:
            pass

if date_col and "severity_label" in df_f.columns:
    df_temp = df_f.dropna(subset=["_parsed_date"]).copy()
    df_temp["YearMonth"] = df_temp["_parsed_date"].dt.to_period("M").astype(str)
    trend = (
        df_temp.groupby(["YearMonth", "severity_label"])
        .size()
        .reset_index(name="Count")
        .sort_values("YearMonth")
    )
    fig6 = px.line(
        trend, x="YearMonth", y="Count", color="severity_label",
        color_discrete_map=COLORS,
        labels={"YearMonth": "Month", "Count": "Reviews"},
        markers=False,
    )
    fig6.update_layout(plot_bgcolor="white", height=350,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig6, width="stretch")
else:
    st.info("Date fields could not be parsed. Run `train_pipeline.py` to generate processed data.")

# ═══════════════════════════════════════════════════════════════════════════════
# Row 5 — Recommended vs Severity
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
r5c1, r5c2 = st.columns(2)

with r5c1:
    st.subheader("Recommended vs Severity")
    if "Recommended" in df_f.columns and "severity_label" in df_f.columns:
        rec_sev = (
            df_f.groupby(["Recommended", "severity_label"])
            .size()
            .reset_index(name="Count")
        )
        fig7 = px.bar(
            rec_sev, x="severity_label", y="Count", color="Recommended",
            color_discrete_map={"yes": "#22c55e", "no": "#ef4444"},
            barmode="group",
            category_orders={"severity_label": SEVERITY_ORDER},
            labels={"severity_label": "Severity"},
        )
        fig7.update_layout(plot_bgcolor="white", height=340)
        st.plotly_chart(fig7, width="stretch")

with r5c2:
    st.subheader("Review Length by Severity")
    if "severity_label" in df_f.columns and "Review" in df_f.columns:
        df_rl = df_f.copy()
        df_rl["review_length"] = df_rl["Review"].fillna("").str.len()
        fig8 = px.box(
            df_rl, x="severity_label", y="review_length",
            color="severity_label", color_discrete_map=COLORS,
            category_orders={"severity_label": SEVERITY_ORDER},
            labels={"severity_label": "Severity", "review_length": "Character Count"},
        )
        fig8.update_layout(showlegend=False, plot_bgcolor="white", height=340)
        st.plotly_chart(fig8, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# Model comparison table (if exists)
# ═══════════════════════════════════════════════════════════════════════════════
comp_path = ROOT / "data" / "processed" / "model_comparison_table.csv"
if comp_path.exists():
    st.markdown("---")
    st.subheader("🏆 Model Comparison Table")
    comp = pd.read_csv(comp_path)
    st.dataframe(comp, width="stretch")
