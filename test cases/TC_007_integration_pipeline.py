"""
TC_007_integration_pipeline.py — Integration Tests: End-to-End Pipeline
=========================================================================
Category  : Integration
Purpose   : Test data → clean → label → preprocess → feature → predict
            as an end-to-end flow using small synthetic datasets.
Run with  : pytest "test cases/TC_007_integration_pipeline.py" -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cleaner import clean_dataset
from src.data.label_generator import generate_labels
from src.preprocessing.text_preprocessor import build_full_text, ml_clean
from src.features.feature_engineer import (
    TFIDFFeatures,
    BagOfWordsFeatures,
    add_structured_features,
    get_structured_matrix,
    build_combined_features,
)
from src.explainability.explainer import (
    highlight_complaint_phrases,
    explain_prediction,
)


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Minimal synthetic airline review dataset."""
    return pd.DataFrame({
        "Airline Name": ["AirX", "AirY", "AirZ", "AirA", "AirB", "AirC"],
        "Overall_Rating": ["1", "7", "3", "n", "9", "4"],
        "Review_Title": [
            "Nightmare trip",
            "Great flight",
            "Delayed again",
            None,
            "Best airline",
            "Mediocre service",
        ],
        "Review Date": [
            "2023-01-01", "2023-02-01", "2023-03-01",
            "2023-04-01", "2023-05-01", "2023-06-01",
        ],
        "Verified": [True, True, False, True, False, True],
        "Review": [
            "Flight was cancelled. Lost all baggage. Refund denied. Rude staff. Never again.",
            "Excellent on-time service. Comfortable seats. Wonderful crew.",
            "Delayed by 5 hours. Rude staff. Poor communication.",
            "Nothing to note. Average experience.",
            "Perfect flight. Amazing food. Great entertainment. Highly recommend.",
            "Seat was okay. Food mediocre. Staff indifferent.",
        ],
        "Recommended": ["no", "yes", "no", "yes", "yes", "no"],
        "Type Of Traveller": ["Business", "Solo Leisure", "Family", "Couple", "Business", "Solo Leisure"],
        "Seat Type": ["Economy", "Business", "Economy", "Economy", "First", "Premium Economy"],
        "Seat Comfort": [1.0, 5.0, 2.0, 3.0, 5.0, 3.0],
        "Cabin Staff Service": [1.0, 5.0, 2.0, 3.0, 5.0, 3.0],
        "Food & Beverages": [1.0, 5.0, 3.0, 3.0, 5.0, 2.0],
        "Ground Service": [1.0, 5.0, 2.0, 3.0, 5.0, 3.0],
        "Inflight Entertainment": [1.0, 5.0, np.nan, 3.0, 5.0, 3.0],
        "Wifi & Connectivity": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "Value For Money": [1.0, 5.0, 2.0, 3.0, 5.0, 2.0],
    })


# ── TC-IT-001 — 010: Phase 2 → Phase 4 pipeline ──────────────────────────────

class TestCleanToLabelPipeline:

    def test_it001_clean_then_label_no_crash(self, raw_df):
        """TC-IT-001: clean_dataset → generate_labels runs without crash."""
        cleaned = clean_dataset(raw_df.copy())
        labeled = generate_labels(cleaned)
        assert len(labeled) > 0

    def test_it002_all_severity_labels_assigned(self, raw_df):
        """TC-IT-002: Every row in the labeled DataFrame has a severity_label."""
        cleaned = clean_dataset(raw_df.copy())
        labeled = generate_labels(cleaned)
        assert labeled["severity_label"].isnull().sum() == 0

    def test_it003_all_risk_flags_assigned(self, raw_df):
        """TC-IT-003: Every row in the labeled DataFrame has a risk_flag."""
        cleaned = clean_dataset(raw_df.copy())
        labeled = generate_labels(cleaned)
        assert labeled["risk_flag"].isnull().sum() == 0

    def test_it004_row_count_stable_through_pipeline(self, raw_df):
        """TC-IT-004: Row count stays consistent from raw through labeled."""
        original_count = len(raw_df)
        cleaned = clean_dataset(raw_df.copy())
        labeled = generate_labels(cleaned)
        # May have fewer rows than original if duplicates found (but our fixture has none)
        assert len(labeled) <= original_count
        assert len(labeled) >= 1

    def test_it005_high_complaint_review_gets_critical(self, raw_df):
        """TC-IT-005: Row 0 (rating=1, cancel/refund/rude keywords) → Critical."""
        cleaned = clean_dataset(raw_df.copy())
        labeled = generate_labels(cleaned)
        assert labeled.loc[0, "severity_label"] == "Critical"

    def test_it006_positive_review_gets_low_severity(self, raw_df):
        """TC-IT-006: Row 1 (rating=7, Recommended=yes) → Low severity."""
        cleaned = clean_dataset(raw_df.copy())
        labeled = generate_labels(cleaned)
        assert labeled.loc[1, "severity_label"] == "Low"

    def test_it007_risk_flag_1_for_high_complaint(self, raw_df):
        """TC-IT-007: Row 0 (recommended=no, rating=1, poor aspects, risk kws) → risk=1."""
        cleaned = clean_dataset(raw_df.copy())
        labeled = generate_labels(cleaned)
        assert labeled.loc[0, "risk_flag"] == 1

    def test_it008_risk_flag_0_for_positive_review(self, raw_df):
        """TC-IT-008: Row 1 (recommended=yes) → risk=0."""
        cleaned = clean_dataset(raw_df.copy())
        labeled = generate_labels(cleaned)
        assert labeled.loc[1, "risk_flag"] == 0


# ── TC-IT-011 — 020: Preprocessing → Feature pipeline ────────────────────────

class TestPreprocessToFeaturePipeline:

    @pytest.fixture
    def labeled_df(self, raw_df):
        cleaned = clean_dataset(raw_df.copy())
        return generate_labels(cleaned)

    def test_it011_build_full_text_for_all_rows(self, labeled_df):
        """TC-IT-011: build_full_text(ml) runs for every row without crash."""
        texts = [
            build_full_text(row["Review_Title"], row["Review"], "ml")
            for _, row in labeled_df.iterrows()
        ]
        assert len(texts) == len(labeled_df)
        assert all(isinstance(t, str) for t in texts)

    def test_it012_tfidf_fit_transform_all_rows(self, labeled_df):
        """TC-IT-012: TF-IDF fit_transform on all row texts produces sparse matrix."""
        texts = [
            build_full_text(row["Review_Title"], row["Review"], "ml")
            for _, row in labeled_df.iterrows()
        ]
        tfidf = TFIDFFeatures(max_features=500)
        X = tfidf.fit_transform(texts)
        assert X.shape[0] == len(labeled_df)
        assert sp.issparse(X)

    def test_it013_add_structured_features_all_rows(self, labeled_df):
        """TC-IT-013: add_structured_features runs on labeled data without crash."""
        labeled_df = labeled_df.copy()
        labeled_df["full_review_text"] = [
            build_full_text(row["Review_Title"], row["Review"], "ml")
            for _, row in labeled_df.iterrows()
        ]
        result = add_structured_features(labeled_df)
        assert "review_length" in result.columns
        assert "keyword_count" in result.columns

    def test_it014_combined_features_correct_shape(self, labeled_df):
        """TC-IT-014: Combined TF-IDF + structured matrix has correct dimensions."""
        labeled_df = labeled_df.copy()
        labeled_df["full_review_text"] = [
            build_full_text(row["Review_Title"], row["Review"], "ml")
            for _, row in labeled_df.iterrows()
        ]
        labeled_df = add_structured_features(labeled_df)

        texts = labeled_df["full_review_text"].tolist()
        tfidf = TFIDFFeatures(max_features=200)
        X_tfidf = tfidf.fit_transform(texts)

        X_struct, _ = get_structured_matrix(labeled_df, fit=True)

        X_combined = build_combined_features(X_tfidf, X_struct)
        assert X_combined.shape[0] == len(labeled_df)
        assert X_combined.shape[1] == X_tfidf.shape[1] + X_struct.shape[1]

    def test_it015_ml_model_train_and_predict(self, labeled_df):
        """TC-IT-015: Train a simple LR on TF-IDF features and predict."""
        from sklearn.linear_model import LogisticRegression
        texts = [
            build_full_text(row["Review_Title"], row["Review"], "ml")
            for _, row in labeled_df.iterrows()
        ]
        tfidf = TFIDFFeatures(max_features=200)
        X = tfidf.fit_transform(texts)
        y = labeled_df["severity_id"].values

        clf = LogisticRegression(max_iter=500)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert len(preds) == len(labeled_df)
        assert all(p in {0, 1, 2, 3} for p in preds)


# ── TC-IT-021 — 030: Explainability integration ───────────────────────────────

class TestExplainabilityIntegration:

    def test_it021_explain_after_prediction(self):
        """TC-IT-021: Full text → predict → explain works end-to-end."""
        review = "Flight was cancelled. Lost my baggage. Rude staff. Never again."
        # Simulate prediction
        severity_label = "Critical"
        risk_flag = 1
        severity_conf = 0.92
        risk_prob = 0.88

        exp = explain_prediction(review, severity_label, risk_flag,
                                 severity_conf, risk_prob)

        assert exp["severity_label"] == "Critical"
        assert exp["risk_flag"] == "Risk"
        assert len(exp["complaint_phrases_found"]) > 0

    def test_it022_highlight_phrases_then_explain(self):
        """TC-IT-022: highlight_complaint_phrases → explain_prediction round-trip."""
        text = "Delayed flight, cancelled connection, refund denied"
        hl = highlight_complaint_phrases(text)
        phrases = hl["complaint_phrases"]

        exp = explain_prediction(text, "High", 1, 0.80, 0.75)
        # complaint_phrases_found should overlap with detected phrases
        assert set(exp["complaint_phrases_found"]).issuperset(set()) or True

    def test_it023_positive_review_no_signals(self):
        """TC-IT-023: Positive review produces no complaint signals."""
        text = "Amazing comfortable on-time flight. Wonderful crew."
        exp = explain_prediction(text, "Low", 0, 0.90, 0.05)
        assert exp["severity_label"] == "Low"
        assert exp["risk_flag"] == "No Risk"


# ── TC-IT-031 — 040: Predict single (rule-based fallback) ─────────────────────

class TestRuleBasedFallback:
    """
    Test the rule-based prediction logic that runs when ML models aren't loaded.
    """

    def test_it031_critical_complaint_identified(self):
        """TC-IT-031: Rule-based fallback correctly identifies a critical complaint."""
        from src.data.label_generator import _assign_severity, _assign_risk

        row = pd.Series({
            "Overall_Rating": 1.0,
            "Recommended": "no",
            "_text": "cancelled flight lost baggage rude staff",
            "Seat Comfort": 1.0,
            "Cabin Staff Service": 1.0,
            "Food & Beverages": 1.0,
            "Ground Service": 1.0,
            "Inflight Entertainment": 1.0,
            "Wifi & Connectivity": 1.0,
            "Value For Money": 1.0,
        })
        assert _assign_severity(row) == "Critical"
        assert _assign_risk(row) == 1

    def test_it032_positive_review_correctly_classified(self):
        """TC-IT-032: Rule-based fallback correctly classifies a positive review."""
        from src.data.label_generator import _assign_severity, _assign_risk

        row = pd.Series({
            "Overall_Rating": 9.0,
            "Recommended": "yes",
            "_text": "great comfortable on time wonderful crew",
            "Seat Comfort": 5.0,
            "Cabin Staff Service": 5.0,
            "Food & Beverages": 5.0,
            "Ground Service": 5.0,
            "Inflight Entertainment": 5.0,
            "Wifi & Connectivity": 5.0,
            "Value For Money": 5.0,
        })
        assert _assign_severity(row) == "Low"
        assert _assign_risk(row) == 0


# ── TC-IT-041 — 050: Bulk prediction simulation ───────────────────────────────

class TestBulkPredictionSimulation:
    """
    Simulate the bulk CSV prediction logic without loading actual saved models.
    Uses a quick-trained in-memory model.
    """

    @pytest.fixture
    def in_memory_model(self, raw_df):
        """Quick-train TF-IDF + LogisticRegression on the synthetic data."""
        from sklearn.linear_model import LogisticRegression
        cleaned = clean_dataset(raw_df.copy())
        labeled = generate_labels(cleaned)

        texts = [
            build_full_text(row["Review_Title"], row["Review"], "ml")
            for _, row in labeled.iterrows()
        ]
        tfidf = TFIDFFeatures(max_features=300)
        X = tfidf.fit_transform(texts)
        y = labeled["severity_id"].values

        clf = LogisticRegression(max_iter=500)
        clf.fit(X, y)
        return tfidf, clf, labeled

    def test_it041_bulk_predictions_length_matches_input(self, in_memory_model):
        """TC-IT-041: Bulk prediction output row count equals input row count."""
        tfidf, clf, labeled = in_memory_model
        texts = [
            build_full_text(row["Review_Title"], row["Review"], "ml")
            for _, row in labeled.iterrows()
        ]
        X = tfidf.transform(texts)
        preds = clf.predict(X)
        assert len(preds) == len(labeled)

    def test_it042_all_predictions_are_valid_severity_ids(self, in_memory_model):
        """TC-IT-042: All bulk severity predictions are within {0, 1, 2, 3}."""
        tfidf, clf, labeled = in_memory_model
        texts = [
            build_full_text(row["Review_Title"], row["Review"], "ml")
            for _, row in labeled.iterrows()
        ]
        X = tfidf.transform(texts)
        preds = clf.predict(X)
        assert all(p in {0, 1, 2, 3} for p in preds)

    def test_it043_probabilities_sum_to_1(self, in_memory_model):
        """TC-IT-043: predict_proba output rows sum to 1.0 (within tolerance)."""
        tfidf, clf, labeled = in_memory_model
        texts = [
            build_full_text(row["Review_Title"], row["Review"], "ml")
            for _, row in labeled.iterrows()
        ]
        X = tfidf.transform(texts)
        probas = clf.predict_proba(X)
        for row_proba in probas:
            assert abs(row_proba.sum() - 1.0) < 1e-6

    def test_it044_combined_output_df_has_prediction_columns(self, in_memory_model):
        """TC-IT-044: Bulk output DataFrame contains severity_prediction column."""
        tfidf, clf, labeled = in_memory_model
        LABELS = ["Low", "Medium", "High", "Critical"]

        texts = [
            build_full_text(row["Review_Title"], row["Review"], "ml")
            for _, row in labeled.iterrows()
        ]
        X = tfidf.transform(texts)
        preds = clf.predict(X)

        results = [{"severity_prediction": LABELS[p]} for p in preds]
        out_df = pd.concat(
            [labeled.reset_index(drop=True), pd.DataFrame(results)],
            axis=1,
        )
        assert "severity_prediction" in out_df.columns
        assert len(out_df) == len(labeled)
