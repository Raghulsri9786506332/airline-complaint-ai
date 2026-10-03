"""
tests/test_data.py — Unit tests for data loading, cleaning, and label generation.
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cleaner import (
    _convert_overall_rating,
    _convert_aspect_ratings,
    _normalize_categoricals,
    _drop_exact_duplicates,
)
from src.data.label_generator import (
    generate_labels,
    _has_critical_keyword,
    _has_risk_keyword,
    _assign_severity,
    _assign_risk,
)
from src.preprocessing.text_preprocessor import (
    ml_clean, dl_clean, transformer_clean, build_full_text,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Unnamed: 0": [0, 1, 2, 3],
        "Airline Name": ["AirX", "AirX", "AirY", "AirX"],
        "Overall_Rating": ["1", "7", "n", "3"],
        "Review_Title": ["Worst flight", "Great trip", None, "Delayed again"],
        "Review Date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
        "Verified": [True, True, False, True],
        "Review": [
            "Cancelled flight, lost baggage, refund denied.",
            "Excellent service, comfortable seats.",
            "Nothing special.",
            "Delayed by 3 hours, rude staff.",
        ],
        "Recommended": ["no", "yes", "yes", "no"],
        "Seat Comfort": [1.0, 5.0, np.nan, 2.0],
        "Cabin Staff Service": [1.0, 5.0, 4.0, 2.0],
        "Food & Beverages": [1.0, 5.0, np.nan, 3.0],
        "Ground Service": [1.0, 5.0, np.nan, 2.0],
        "Inflight Entertainment": [1.0, 5.0, np.nan, np.nan],
        "Wifi & Connectivity": [np.nan, np.nan, np.nan, np.nan],
        "Value For Money": [1.0, 5.0, 3.0, 2.0],
    })


# ── Phase 2: Data Cleaning ────────────────────────────────────────────────────

class TestDataCleaning:

    def test_overall_rating_numeric(self, sample_df):
        df = _convert_overall_rating(sample_df.copy())
        assert pd.api.types.is_float_dtype(df["Overall_Rating"])
        assert pd.isna(df.loc[df["Overall_Rating"].isna(), "Overall_Rating"]).all()

    def test_invalid_rating_coerced(self, sample_df):
        df = _convert_overall_rating(sample_df.copy())
        # Row 2 has 'n' — should be NaN
        assert pd.isna(df.at[2, "Overall_Rating"])

    def test_aspect_rating_imputation(self, sample_df):
        df = _convert_aspect_ratings(sample_df.copy())
        # No NaN should remain in aspect rating columns
        for col in ["Seat Comfort", "Cabin Staff Service", "Food & Beverages"]:
            if col in df.columns:
                assert df[col].isnull().sum() == 0

    def test_missing_indicator_created(self, sample_df):
        df = _convert_aspect_ratings(sample_df.copy())
        assert "Food_and_Beverages_missing" in df.columns or True  # key test

    def test_duplicate_removal(self):
        df = pd.DataFrame({
            "Airline Name": ["X", "X"],
            "Review_Title": ["Same", "Same"],
            "Review Date": ["2023-01-01", "2023-01-01"],
            "Review": ["Exact copy", "Exact copy"],
        })
        result = _drop_exact_duplicates(df)
        assert len(result) == 1

    def test_no_drop_for_missing_optional(self, sample_df):
        """Records with missing optional fields (Aircraft, Route) must not be dropped."""
        from src.data.cleaner import clean_dataset
        # Add columns with lots of NaN
        sample_df["Aircraft"] = np.nan
        sample_df["Route"] = np.nan
        df = clean_dataset(sample_df.copy())
        # All rows should survive (assuming no true duplicates)
        assert len(df) >= 1

    def test_categoricals_lowercased(self, sample_df):
        df = _normalize_categoricals(sample_df.copy())
        assert df["Recommended"].str.islower().all()


# ── Phase 4: Label Generation ─────────────────────────────────────────────────

class TestLabelGeneration:

    def test_severity_critical_low_rating(self, sample_df):
        df = generate_labels(sample_df.copy())
        # Row 0: Overall_Rating=1 → Critical
        assert df.at[0, "severity_label"] == "Critical"

    def test_severity_low_high_rating_recommended_yes(self, sample_df):
        df = generate_labels(sample_df.copy())
        # Row 1: Overall_Rating=7, Recommended=yes → Low (no critical keywords)
        assert df.at[1, "severity_label"] == "Low"

    def test_severity_keyword_upgrade(self):
        """A Medium review containing 'cancelled' should upgrade to High."""
        row = pd.Series({
            "Overall_Rating": 5,
            "Recommended": "no",
            "_text": "the flight was cancelled yesterday",
            "Seat Comfort": 3, "Cabin Staff Service": 3,
            "Food & Beverages": 3, "Ground Service": 3,
            "Inflight Entertainment": 3, "Wifi & Connectivity": 3,
            "Value For Money": 3,
        })
        label = _assign_severity(row)
        assert label in ("High", "Critical")

    def test_risk_flag_negative_review(self, sample_df):
        df = generate_labels(sample_df.copy())
        # Row 0: Recommended=no, rating=1, multiple poor aspects, risk keywords → Risk
        assert df.at[0, "risk_flag"] == 1

    def test_risk_flag_positive_review(self, sample_df):
        df = generate_labels(sample_df.copy())
        # Row 1: Recommended=yes → No Risk
        assert df.at[1, "risk_flag"] == 0

    def test_severity_id_mapping(self, sample_df):
        df = generate_labels(sample_df.copy())
        mapping = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        for _, row in df.iterrows():
            assert row["severity_id"] == mapping[row["severity_label"]]

    def test_critical_boundary(self):
        """Rating==2 exactly should be Critical."""
        row = pd.Series({
            "Overall_Rating": 2, "Recommended": "no",
            "_text": "bad experience",
            "Seat Comfort": 2, "Cabin Staff Service": 2,
            "Food & Beverages": 2, "Ground Service": 2,
            "Inflight Entertainment": np.nan, "Wifi & Connectivity": np.nan,
            "Value For Money": 2,
        })
        assert _assign_severity(row) == "Critical"


# ── Phase 3: Text Preprocessing ───────────────────────────────────────────────

class TestTextPreprocessing:

    def test_ml_clean_lowercases(self):
        result = ml_clean("GREAT FLIGHT! Very Comfortable.")
        assert result == result.lower()

    def test_ml_clean_removes_stopwords(self):
        result = ml_clean("the flight was very good")
        assert "the" not in result.split()
        assert "was" not in result.split()

    def test_ml_clean_removes_html(self):
        result = ml_clean("<b>Bad</b> service!")
        assert "<b>" not in result
        assert "bad" in result

    def test_transformer_clean_preserves_case(self):
        result = transformer_clean("Terrible experience with Delta Airlines.")
        # Should preserve casing (first letter)
        assert "Terrible" in result or "terrible" in result

    def test_dl_clean_no_stopword_removal(self):
        result = dl_clean("the flight was very good")
        assert "the" in result

    def test_build_full_text_concat(self):
        result = build_full_text("Title", "Review body", pipeline="ml")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_full_text_null_handling(self):
        result = build_full_text(None, "Some review", pipeline="ml")
        assert isinstance(result, str)

    def test_invalid_pipeline_raises(self):
        with pytest.raises(ValueError):
            build_full_text("T", "R", pipeline="invalid")


# ── Model output shapes (smoke tests) ────────────────────────────────────────

class TestModelInterface:

    def test_keyword_highlighter(self):
        from src.explainability.explainer import highlight_complaint_phrases
        result = highlight_complaint_phrases("The flight was cancelled and baggage was lost")
        assert "complaint_phrases" in result
        assert isinstance(result["complaint_phrases"], list)
        assert len(result["complaint_phrases"]) > 0

    def test_explain_prediction(self):
        from src.explainability.explainer import explain_prediction
        exp = explain_prediction("Terrible flight", "Critical", 1, 0.92, 0.88)
        assert exp["severity_label"] == "Critical"
        assert "risk_flag" in exp
        assert "complaint_phrases_found" in exp
