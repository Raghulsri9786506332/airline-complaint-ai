"""
explainer.py — Explainability for severity and risk predictions.

Provides:
  - TF-IDF / LinearModel keyword importance (for ML models)
  - LIME text explanations (for any model with predict_proba)
  - Keyword highlighting (always available)
  - SHAP values (optional, for tree models)
"""
from __future__ import annotations

import re
from typing import Any, Optional
import numpy as np
from src.utils.config import CRITICAL_KEYWORDS, RISK_KEYWORDS


# ── 1. Keyword highlighting ───────────────────────────────────────────────────

def highlight_complaint_phrases(text: str) -> dict:
    """
    Find complaint-related phrases in the review text.
    Returns a dict with highlighted spans and a cleaned list.
    """
    ALL_KW = set(CRITICAL_KEYWORDS + RISK_KEYWORDS)
    lowered = (text or "").lower()
    found = [kw for kw in ALL_KW if kw in lowered]
    # Drop phrases fully contained in a longer matched phrase
    # (e.g. "cancel" when "cancelled" was also found, "delay" when "delayed" was found)
    found = [kw for kw in found if not any(kw != o and kw in o for o in found)]
    return {"complaint_phrases": sorted(set(found)), "text": text}


# ── 2. Linear-model top features ─────────────────────────────────────────────

def get_linear_top_features(
    clf,  # LogisticRegression or LinearSVC (calibrated)
    vectorizer,
    class_idx: int,
    top_n: int = 15,
) -> list[tuple[str, float]]:
    """
    Return top-N features by absolute coefficient for a given class.
    Works for LR and LinearSVC.
    """
    try:
        # Unwrap CalibratedClassifierCV: the *fitted* estimators live in
        # calibrated_classifiers_ (clf.estimator is the unfitted template)
        if hasattr(clf, "calibrated_classifiers_"):
            coefs = [cc.estimator.coef_ for cc in clf.calibrated_classifiers_]
            coef = np.mean(coefs, axis=0)
        else:
            coef = clf.coef_
        if coef.ndim == 1:
            coef_row = coef
        else:
            coef_row = coef[class_idx]
        feature_names = vectorizer.get_feature_names_out()
        top_idx = np.argsort(np.abs(coef_row))[::-1][:top_n]
        return [(feature_names[i], float(coef_row[i])) for i in top_idx]
    except Exception as e:
        return [("error", str(e))]


# ── 3. LIME explanation ───────────────────────────────────────────────────────

def get_lime_explanation(
    text: str,
    predict_fn,
    class_names: list[str],
    num_features: int = 10,
    num_samples: int = 500,
) -> Optional[dict]:
    """
    Generate a LIME text explanation.
    predict_fn must accept a list of strings and return probability array.
    Returns dict with feature weights, or None if LIME not installed.
    """
    try:
        from lime.lime_text import LimeTextExplainer
        explainer = LimeTextExplainer(class_names=class_names, random_state=42)
        exp = explainer.explain_instance(
            text, predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=1,
        )
        label = exp.available_labels()[0]
        return {
            "label": class_names[label],
            "weights": exp.as_list(label=label),
        }
    except ImportError:
        return None


# ── 4. SHAP explanation (tree models) ────────────────────────────────────────

def get_shap_values(clf, X_sample, max_display: int = 15) -> Optional[dict]:
    """
    Compute SHAP values for a tree-based model.
    Returns dict with values and feature names, or None if SHAP not installed.
    """
    try:
        import shap
        inner = clf
        if hasattr(clf, "calibrated_classifiers_"):
            inner = clf.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(inner)
        shap_vals = explainer.shap_values(X_sample)
        return {"shap_values": shap_vals}
    except Exception:  # ImportError or any SHAP incompatibility -> graceful None
        return None


# ── 5. Severity/Risk prediction explanation ───────────────────────────────────

def explain_prediction(
    text: str,
    severity_label: str,
    risk_flag: int,
    severity_conf: float,
    risk_prob: float,
) -> dict:
    """
    Build a human-readable explanation dict for a single prediction.
    Always available (no external library dependency).
    """
    phrases = highlight_complaint_phrases(text)["complaint_phrases"]
    severity_signals = [p for p in phrases if p in CRITICAL_KEYWORDS]
    risk_signals = [p for p in phrases if p in RISK_KEYWORDS]

    explanation = {
        "severity_label": severity_label,
        "severity_confidence": f"{severity_conf * 100:.1f}%",
        "risk_flag": "Risk" if risk_flag == 1 else "No Risk",
        "risk_probability": f"{risk_prob * 100:.1f}%",
        "complaint_phrases_found": phrases,
        "severity_related_signals": severity_signals,
        "risk_related_signals": risk_signals,
        "disclaimer": (
            "This explanation highlights words/phrases associated with the prediction. "
            "It does not imply causal influence."
        ),
    }
    return explanation
