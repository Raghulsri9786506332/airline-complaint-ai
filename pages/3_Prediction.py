"""
pages/3_Prediction.py — Severity & Risk prediction page.

Supports:
  - Single review prediction (with explanation)
  - Bulk CSV prediction with download
"""
import sys
import io
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Prediction | Airline Complaint AI",
    page_icon="🔍", layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.pred-header {
    background: linear-gradient(135deg, #1e3a8a, #7c3aed);
    color: white; border-radius: 14px; padding: 28px 36px; margin-bottom: 24px;
}
.severity-chip {
    display: inline-block; padding: 6px 18px; border-radius: 999px;
    font-weight: 700; font-size: 1rem;
}
.sev-Low      { background:#dcfce7; color:#166534; }
.sev-Medium   { background:#fef9c3; color:#854d0e; }
.sev-High     { background:#ffedd5; color:#9a3412; }
.sev-Critical { background:#fee2e2; color:#991b1b; }
.risk-yes { background:#fee2e2; color:#991b1b; padding:6px 18px;
             border-radius:999px; font-weight:700; }
.risk-no  { background:#dcfce7; color:#166534; padding:6px 18px;
             border-radius:999px; font-weight:700; }
.result-card {
    background: white; border-radius: 14px;
    padding: 24px; box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
}
.kw-pill {
    display:inline-block; background:#fef3c7; color:#92400e;
    border:1px solid #fde68a; border-radius:6px;
    padding:2px 10px; font-size:0.82rem; margin:2px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pred-header">
  <h1>🔍 Severity &amp; Risk Prediction</h1>
  <p style="opacity:0.9; margin:4px 0 0 0">
    Enter a review or upload a CSV to classify complaint severity and service risk.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models …")
def load_models():
    """Load the best available severity and risk models + vectorizer."""
    try:
        tfidf_path = ROOT / "artifacts" / "vectorizers" / "tfidf_vectorizer.pkl"
        sev_model_path = ROOT / "models" / "ml" / "severity_Logistic_Regression_tfidf.pkl"
        risk_model_path = ROOT / "models" / "ml" / "risk_Logistic_Regression_tfidf_struct.pkl"
        struct_scaler_path = ROOT / "artifacts" / "vectorizers" / "struct_scaler.pkl"

        models = {}
        if tfidf_path.exists():
            models["tfidf"] = joblib.load(tfidf_path)
        if sev_model_path.exists():
            models["severity_clf"] = joblib.load(sev_model_path)
        if risk_model_path.exists():
            models["risk_clf"] = joblib.load(risk_model_path)
        if struct_scaler_path.exists():
            models["struct_scaler"] = joblib.load(struct_scaler_path)

        return models
    except Exception as e:
        st.warning(f"Could not load trained models: {e}. Using rule-based fallback.")
        return {}


models = load_models()
models_available = "tfidf" in models and "severity_clf" in models


# ── Prediction helpers ────────────────────────────────────────────────────────

_RATING_COLS = [
    "Seat Comfort", "Cabin Staff Service", "Food & Beverages", "Ground Service",
    "Inflight Entertainment", "Wifi & Connectivity", "Value For Money",
]


def _bulk_ratings(frame: pd.DataFrame):
    """Per-row rating dicts (NaN -> None) and a boolean mask of rows with any rating."""
    ratings_list, has_any = [], []
    for _, row in frame.iterrows():
        d = {}
        for c in _RATING_COLS:
            if c in frame.columns:
                v = pd.to_numeric(row.get(c), errors="coerce")
                d[c] = None if pd.isna(v) else float(v)
        ratings_list.append(d)
        has_any.append(any(v is not None for v in d.values()))
    return ratings_list, np.array(has_any, dtype=bool)


def _risk_features(X_tfidf, titles, reviews, ratings_list):
    """
    Build the exact feature matrix the risk model was trained on:
    [TF-IDF | scaled structured features (7 ratings + review_length + keyword_count)].
    Falls back to TF-IDF only if the scaler artifact is missing AND the model expects
    that many features.
    """
    import scipy.sparse as sp
    from src.features.feature_engineer import build_inference_struct

    scaler = models.get("struct_scaler")
    if scaler is None:
        return X_tfidf
    rows = [
        build_inference_struct(t, r, rt, scaler)
        for t, r, rt in zip(titles, reviews, ratings_list)
    ]
    return sp.hstack([X_tfidf, sp.csr_matrix(np.vstack(rows))]).tocsr()


def predict_single(
    title: str,
    review: str,
    rating: Optional[float] = None,
    seat_comfort: Optional[float] = None,
    cabin_staff: Optional[float] = None,
    food_bev: Optional[float] = None,
    ground_svc: Optional[float] = None,
    inflight_ent: Optional[float] = None,
    wifi: Optional[float] = None,
    value_money: Optional[float] = None,
) -> dict:
    """Run severity + risk prediction on a single review."""
    from src.preprocessing.text_preprocessor import build_full_text
    from src.explainability.explainer import explain_prediction

    SEVERITY_LABELS = ["Low", "Medium", "High", "Critical"]

    text_ml = build_full_text(title, review, pipeline="ml")

    # ── ML path ──────────────────────────────────────────────────────────────
    if models_available:
        vec = models["tfidf"]
        sev_clf = models["severity_clf"]
        risk_clf = models.get("risk_clf")

        X = vec.transform([text_ml])
        ratings = {
            "Seat Comfort": seat_comfort, "Cabin Staff Service": cabin_staff,
            "Food & Beverages": food_bev, "Ground Service": ground_svc,
            "Inflight Entertainment": inflight_ent, "Wifi & Connectivity": wifi,
            "Value For Money": value_money,
        }

        # Severity uses the text-only model (it behaves far more stably on short / unseen
        # reviews than the text+ratings model); ratings are used for the risk model.
        sev_proba = sev_clf.predict_proba(X)[0]
        severity_source = "review text"
        sev_id = int(np.argmax(sev_proba))
        sev_label = SEVERITY_LABELS[sev_id]
        sev_conf = float(sev_proba[sev_id])

        # Risk — use structured features if scaler available
        risk_label, risk_prob = 0, 0.0
        if risk_clf is not None:
            X_comb = _risk_features(X, [title], [review], [ratings])
            risk_label = int(risk_clf.predict(X_comb)[0])
            risk_prob = float(risk_clf.predict_proba(X_comb)[0][1])

    else:
        # ── Rule-based fallback ───────────────────────────────────────────
        from src.data.label_generator import _assign_severity, _assign_risk

        row = pd.Series({
            "Overall_Rating": rating if rating is not None else np.nan,
            # Unknown at inference time: assume the reviewer would not recommend
            # (the rule-based labels are only meaningful for complaints)
            "Recommended": "no",
            "_text": f"{title or ''} {review or ''}".lower(),
            "Seat Comfort": seat_comfort,
            "Cabin Staff Service": cabin_staff,
            "Food & Beverages": food_bev,
            "Ground Service": ground_svc,
            "Inflight Entertainment": inflight_ent,
            "Wifi & Connectivity": wifi,
            "Value For Money": value_money,
        })
        sev_label = _assign_severity(row)
        sev_id = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}[sev_label]
        sev_conf = 0.75  # placeholder
        severity_source = "rule-based (models not found)"
        risk_label = _assign_risk(row)
        risk_prob = 1.0 if risk_label else 0.0

    explanation = explain_prediction(
        review, sev_label, risk_label, sev_conf, risk_prob
    )
    return {
        "severity_label": sev_label,
        "severity_id": sev_id,
        "severity_confidence": sev_conf,
        "severity_source": severity_source,
        "risk_flag": risk_label,
        "risk_probability": risk_prob,
        "explanation": explanation,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tab layout
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["✏️ Single Review Prediction", "📋 Bulk CSV Prediction"])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Single Review
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Single Review")

    col_a, col_b = st.columns([1.4, 1], gap="large")

    with col_a:
        title_input = st.text_input("Review Title", placeholder="e.g., Worst flight ever")
        review_input = st.text_area(
            "Review Text",
            height=200,
            placeholder="Paste the full airline review here …",
        )
        st.caption("**Optional: Add structured ratings (1–5 scale)**")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            seat_c = st.slider("Seat Comfort", 0.0, 5.0, 0.0, 0.5)
        with r2:
            cabin_s = st.slider("Cabin Staff", 0.0, 5.0, 0.0, 0.5)
        with r3:
            food_b = st.slider("Food & Bev", 0.0, 5.0, 0.0, 0.5)
        with r4:
            ground = st.slider("Ground Svc", 0.0, 5.0, 0.0, 0.5)

        r5, r6, r7, _ = st.columns(4)
        with r5:
            inflight = st.slider("Inflight Ent", 0.0, 5.0, 0.0, 0.5)
        with r6:
            wifi = st.slider("WiFi", 0.0, 5.0, 0.0, 0.5)
        with r7:
            value = st.slider("Value Money", 0.0, 5.0, 0.0, 0.5)

        predict_btn = st.button("🔍 Predict", type="primary", width="stretch")

    with col_b:
        if predict_btn:
            if not review_input.strip():
                st.error("Please enter a review text.")
            else:
                with st.spinner("Analysing review …"):
                    try:
                        result = predict_single(
                            title_input, review_input,
                            seat_comfort=seat_c or None,
                            cabin_staff=cabin_s or None,
                            food_bev=food_b or None,
                            ground_svc=ground or None,
                            inflight_ent=inflight or None,
                            wifi=wifi or None,
                            value_money=value or None,
                        )

                        sev = result["severity_label"]
                        risk = result["risk_flag"]
                        exp = result["explanation"]

                        # Severity
                        st.markdown("**Complaint Severity**")
                        st.markdown(
                            f'<span class="severity-chip sev-{sev}">{sev}</span>&nbsp;'
                            f'<span style="color:#64748b;font-size:0.9rem">'
                            f'Confidence: {result["severity_confidence"]*100:.1f}%</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption(f'Severity based on: {result["severity_source"]}')

                        st.markdown("<br>**Service Risk**", unsafe_allow_html=True)
                        risk_cls = "risk-yes" if risk else "risk-no"
                        risk_lbl = "⚠️ Risk" if risk else "✅ No Risk"
                        st.markdown(
                            f'<span class="{risk_cls}">{risk_lbl}</span>&nbsp;'
                            f'<span style="color:#64748b;font-size:0.9rem">'
                            f'Probability: {result["risk_probability"]*100:.1f}%</span>',
                            unsafe_allow_html=True,
                        )

                        # Complaint phrases
                        if exp.get("complaint_phrases_found"):
                            st.markdown("<br>**🔑 Complaint Phrases Detected**", unsafe_allow_html=True)
                            pills = " ".join(
                                f'<span class="kw-pill">{kw}</span>'
                                for kw in exp["complaint_phrases_found"]
                            )
                            st.markdown(pills, unsafe_allow_html=True)

                        # Signals
                        if exp.get("severity_related_signals"):
                            st.markdown(
                                f'*Severity signals:* {", ".join(exp["severity_related_signals"])}',
                            )
                        if exp.get("risk_related_signals"):
                            st.markdown(
                                f'*Risk signals:* {", ".join(exp["risk_related_signals"])}',
                            )

                        st.caption(exp.get("disclaimer", ""))

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
        else:
            st.info("Enter a review title and text, then click **Predict**.")
            # Show sample review
            with st.expander("💡 Try a sample review"):
                st.write(
                    "**Title:** Cancelled flight, no refund, rude staff\n\n"
                    "**Review:** The flight was cancelled without notice. Staff were rude and "
                    "unhelpful. They refused to issue a refund and lost our baggage. "
                    "Never flying with this airline again."
                )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Bulk CSV Prediction
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Bulk CSV Prediction")
    st.markdown(
        """
        Upload a CSV file containing airline reviews. Required columns:
        - **Review_Title** — Review title (or empty string)
        - **Review** — Review text

        Optional columns (improves risk prediction):
        `Seat Comfort`, `Cabin Staff Service`, `Food & Beverages`,
        `Ground Service`, `Inflight Entertainment`, `Wifi & Connectivity`, `Value For Money`
        """
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="bulk_csv")

    if uploaded is not None:
        try:
            bulk_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(bulk_df):,} rows.")

            # Validate required columns
            required = {"Review"}  # Review_Title is optional (defaults to "")
            missing_cols = required - set(bulk_df.columns)
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.stop()

            if "Review_Title" not in bulk_df.columns:
                bulk_df["Review_Title"] = ""

            st.dataframe(bulk_df.head(5), width="stretch")

            if st.button("▶️ Run Bulk Prediction", type="primary"):
                progress = st.progress(0, text="Predicting …")
                results_rows = []
                total = len(bulk_df)

                from src.preprocessing.text_preprocessor import build_full_text

                if models_available:
                    vec = models["tfidf"]
                    sev_clf = models["severity_clf"]
                    risk_clf = models.get("risk_clf")

                    SEVERITY_LABELS = ["Low", "Medium", "High", "Critical"]
                    batch_texts = [
                        build_full_text(row.get("Review_Title", ""), row.get("Review", ""), "ml")
                        for _, row in bulk_df.iterrows()
                    ]
                    X_batch = vec.transform(batch_texts)
                    sev_probas = sev_clf.predict_proba(X_batch)
                    sev_ids = np.argmax(sev_probas, axis=1)
                    sev_labels_arr = [SEVERITY_LABELS[s] for s in sev_ids]
                    sev_confs = sev_probas.max(axis=1)

                    if risk_clf is not None:
                        # The risk model was trained on TF-IDF + structured features,
                        # so the same structure must be rebuilt here (TF-IDF alone
                        # has the wrong number of columns and would raise).
                        ratings_list, _ = _bulk_ratings(bulk_df)
                        titles = bulk_df["Review_Title"].fillna("").astype(str).tolist()
                        reviews = bulk_df["Review"].fillna("").astype(str).tolist()
                        X_risk = _risk_features(X_batch, titles, reviews, ratings_list)
                        risk_preds = risk_clf.predict(X_risk)
                        risk_probs = risk_clf.predict_proba(X_risk)[:, 1]
                    else:
                        risk_preds = np.zeros(total, dtype=int)
                        risk_probs = np.zeros(total)

                    for i in range(total):
                        results_rows.append({
                            "severity_prediction": sev_labels_arr[i],
                            "severity_confidence": round(float(sev_confs[i]), 4),
                            "risk_flag": int(risk_preds[i]),
                            "risk_probability": round(float(risk_probs[i]), 4),
                        })
                        progress.progress((i + 1) / total, text=f"Predicting … {i+1}/{total}")

                else:
                    # Rule-based fallback
                    from src.data.label_generator import generate_labels

                    # NOTE: do not use clean_dataset() here — it drops duplicate rows, which
                    # would misalign the predictions with the uploaded rows.
                    temp = bulk_df.copy().reset_index(drop=True)
                    temp = generate_labels(temp)

                    for i, (_, row) in enumerate(temp.iterrows()):
                        results_rows.append({
                            "severity_prediction": row.get("severity_label", "Medium"),
                            "severity_confidence": 0.75,
                            "risk_flag": int(row.get("risk_flag", 0)),
                            "risk_probability": float(row.get("risk_flag", 0)),
                        })
                        progress.progress((i + 1) / total)

                progress.empty()

                out_df = pd.concat(
                    [bulk_df.reset_index(drop=True),
                     pd.DataFrame(results_rows)],
                    axis=1,
                )

                st.success(f"✅ Predictions complete for {total:,} reviews.")
                st.dataframe(out_df, width="stretch")

                # Download button
                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Predictions CSV",
                    data=csv_bytes,
                    file_name="airline_predictions.csv",
                    mime="text/csv",
                    width="stretch",
                )

        except Exception as e:
            st.error(f"Error processing CSV: {e}")
