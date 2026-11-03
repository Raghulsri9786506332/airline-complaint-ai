"""
TC_003_unit_label_generation.py — Unit Tests: Label Generation (Phase 4)
=========================================================================
Category  : Unit
Module    : src/data/label_generator.py
Purpose   : Validate every severity and risk labelling rule, boundary conditions,
            keyword upgrades, and edge cases.
Run with  : pytest "test cases/TC_003_unit_label_generation.py" -v

NOTE: generate_labels() expects Overall_Rating to already be numeric (float).
      The fixture pre-converts this, as clean_dataset() would do in production.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.label_generator import (
    generate_labels,
    _has_critical_keyword,
    _has_risk_keyword,
    _assign_severity,
    _assign_risk,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_row(
    rating: float | None = 5.0,
    recommended: str = "yes",
    text: str = "nothing special",
    seat=3.0, cabin=3.0, food=3.0, ground=3.0,
    inflight=3.0, wifi=3.0, value=3.0,
) -> pd.Series:
    """Build a minimal pd.Series for _assign_severity / _assign_risk."""
    return pd.Series({
        "Overall_Rating": rating,
        "Recommended": recommended,
        "_text": text.lower(),
        "Seat Comfort": seat,
        "Cabin Staff Service": cabin,
        "Food & Beverages": food,
        "Ground Service": ground,
        "Inflight Entertainment": inflight,
        "Wifi & Connectivity": wifi,
        "Value For Money": value,
    })


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """Pre-cleaned DataFrame with numeric Overall_Rating (as cleaner would produce)."""
    return pd.DataFrame({
        "Airline Name": ["AirX", "AirX", "AirY", "AirX"],
        "Overall_Rating": [1.0, 7.0, np.nan, 3.0],
        "Review_Title": ["Worst flight", "Great trip", "", "Delayed again"],
        "Review Date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
        "Verified": ["yes", "yes", "no", "yes"],
        "Review": [
            "Cancelled flight, lost baggage, refund denied.",
            "Excellent service, comfortable seats.",
            "Nothing special.",
            "Delayed by 3 hours, rude staff.",
        ],
        "Recommended": ["no", "yes", "yes", "no"],
        "Seat Comfort": [1.0, 5.0, 3.0, 2.0],
        "Cabin Staff Service": [1.0, 5.0, 4.0, 2.0],
        "Food & Beverages": [1.0, 5.0, 3.0, 3.0],
        "Ground Service": [1.0, 5.0, 3.0, 2.0],
        "Inflight Entertainment": [1.0, 5.0, 3.0, 3.0],
        "Wifi & Connectivity": [3.0, 3.0, 3.0, 3.0],
        "Value For Money": [1.0, 5.0, 3.0, 2.0],
    })


# ── TC-LG-001 — 015: Severity rules ─────────────────────────────────────────

class TestSeverityRules:

    def test_lg001_rating_1_is_critical(self):
        """TC-LG-001: Rating = 1 → Critical (hard lower boundary)."""
        assert _assign_severity(make_row(rating=1.0)) == "Critical"

    def test_lg002_rating_2_is_critical(self):
        """TC-LG-002: Rating = 2 → Critical (inclusive upper boundary for critical rule)."""
        assert _assign_severity(make_row(rating=2.0)) == "Critical"

    def test_lg003_rating_3_is_high(self):
        """TC-LG-003: Rating = 3 → High (no keyword, few poor aspects)."""
        row = make_row(rating=3.0, recommended="yes",
                       seat=4.0, cabin=4.0, food=4.0, ground=4.0,
                       inflight=4.0, wifi=4.0, value=4.0)
        assert _assign_severity(row) == "High"

    def test_lg004_rating_4_is_high(self):
        """TC-LG-004: Rating = 4 → High (boundary)."""
        row = make_row(rating=4.0, recommended="yes",
                       seat=4.0, cabin=4.0, food=4.0, ground=4.0,
                       inflight=4.0, wifi=4.0, value=4.0)
        assert _assign_severity(row) == "High"

    def test_lg005_rating_5_is_medium(self):
        """TC-LG-005: Rating = 5 → Medium (no keywords)."""
        row = make_row(rating=5.0, recommended="yes", text="average flight")
        assert _assign_severity(row) == "Medium"

    def test_lg006_rating_6_is_medium(self):
        """TC-LG-006: Rating = 6 → Medium (upper boundary for medium rule)."""
        row = make_row(rating=6.0, recommended="yes", text="okay flight")
        assert _assign_severity(row) == "Medium"

    def test_lg007_rating_7_recommended_yes_is_low(self):
        """TC-LG-007: Rating = 7, Recommended='yes' → Low."""
        row = make_row(rating=7.0, recommended="yes", text="good flight")
        assert _assign_severity(row) == "Low"

    def test_lg008_rating_10_recommended_yes_is_low(self):
        """TC-LG-008: Rating = 10, Recommended='yes' → Low (max rating)."""
        row = make_row(rating=10.0, recommended="yes", text="amazing flight")
        assert _assign_severity(row) == "Low"

    def test_lg009_keyword_upgrades_low_to_high(self):
        """TC-LG-009: Low base severity + critical keyword → High."""
        row = make_row(rating=8.0, recommended="yes",
                       text="the flight was cancelled yesterday")
        result = _assign_severity(row)
        assert result in ("High", "Critical")

    def test_lg010_keyword_upgrades_medium_to_high(self):
        """TC-LG-010: Medium base severity + critical keyword → High."""
        row = make_row(rating=5.0, recommended="no",
                       text="the flight was cancelled")
        result = _assign_severity(row)
        assert result in ("High", "Critical")

    def test_lg011_keyword_and_no_recommendation_upgrades_high_to_critical(self):
        """TC-LG-011: High + critical keyword + Recommended='no' → Critical."""
        row = make_row(rating=3.0, recommended="no",
                       text="rude staff and terrible experience",
                       seat=4.0, cabin=4.0, food=4.0, ground=4.0,
                       inflight=4.0, wifi=4.0, value=4.0)
        result = _assign_severity(row)
        assert result == "Critical"

    def test_lg012_nan_rating_no_keyword_is_medium(self):
        """TC-LG-012: NaN Overall_Rating + no critical keyword → Medium."""
        row = make_row(rating=np.nan, text="the flight was okay")
        result = _assign_severity(row)
        assert result == "Medium"

    def test_lg013_nan_rating_with_keyword_is_high(self):
        """TC-LG-013: NaN Overall_Rating + critical keyword → High."""
        row = make_row(rating=np.nan, text="the flight was cancelled")
        result = _assign_severity(row)
        assert result == "High"

    def test_lg014_poor_aspects_critical_from_high_base(self):
        """TC-LG-014: Rating in [3,4] + >=3 poor aspects → Critical."""
        row = make_row(
            rating=3.0, recommended="no",
            seat=1.0, cabin=1.0, food=1.0,
            ground=4.0, inflight=4.0, wifi=4.0, value=4.0
        )
        result = _assign_severity(row)
        assert result == "Critical"

    def test_lg015_severity_label_type_is_string(self):
        """TC-LG-015: _assign_severity always returns a Python string."""
        for rating in [1.0, 3.0, 5.0, 7.0]:
            result = _assign_severity(make_row(rating=rating))
            assert isinstance(result, str)


# ── TC-LG-016 — 025: Risk rules ──────────────────────────────────────────────

class TestRiskRules:

    def test_lg016_no_risk_when_recommended_yes(self):
        """TC-LG-016: Recommended='yes' → risk_flag = 0 regardless of other fields."""
        row = make_row(rating=1.0, recommended="yes",
                       seat=1.0, cabin=1.0, text="cancelled worst")
        assert _assign_risk(row) == 0

    def test_lg017_no_risk_when_rating_above_4(self):
        """TC-LG-017: Recommended='no' but rating=5 → risk_flag = 0."""
        row = make_row(rating=5.0, recommended="no",
                       seat=1.0, cabin=1.0, text="delayed")
        assert _assign_risk(row) == 0

    def test_lg018_risk_when_recommended_no_low_rating_poor_aspects(self):
        """TC-LG-018: Recommended='no' + rating<=4 + >=2 poor aspects → risk_flag=1."""
        row = make_row(rating=2.0, recommended="no",
                       seat=1.0, cabin=1.0,
                       food=4.0, ground=4.0, inflight=4.0, wifi=4.0, value=4.0)
        assert _assign_risk(row) == 1

    def test_lg019_risk_when_recommended_no_low_rating_risk_keyword(self):
        """TC-LG-019: Recommended='no' + rating<=4 + risk keyword → risk_flag=1."""
        row = make_row(rating=3.0, recommended="no", text="flight delayed by hours",
                       seat=4.0, cabin=4.0, food=4.0, ground=4.0,
                       inflight=4.0, wifi=4.0, value=4.0)
        assert _assign_risk(row) == 1

    def test_lg020_no_risk_one_poor_aspect_no_keyword(self):
        """TC-LG-020: Recommended='no' + rating<=4 + only 1 poor aspect + no keyword → 0."""
        row = make_row(rating=3.0, recommended="no",
                       seat=1.0, cabin=4.0, food=4.0, ground=4.0,
                       inflight=4.0, wifi=4.0, value=4.0,
                       text="average flight")
        assert _assign_risk(row) == 0

    def test_lg021_risk_flag_is_int(self):
        """TC-LG-021: _assign_risk always returns 0 or 1 (integer)."""
        for rating, rec in [(1.0, "no"), (7.0, "yes"), (3.0, "no")]:
            result = _assign_risk(make_row(rating=rating, recommended=rec))
            assert result in (0, 1)

    def test_lg022_rating_exactly_4_qualifies_for_risk(self):
        """TC-LG-022: Rating = 4 exactly still qualifies as low enough for risk."""
        row = make_row(rating=4.0, recommended="no",
                       seat=1.0, cabin=1.0, food=4.0, ground=4.0,
                       inflight=4.0, wifi=4.0, value=4.0)
        assert _assign_risk(row) == 1


# ── TC-LG-026 — 035: Keyword detection ───────────────────────────────────────

class TestKeywordDetection:

    def test_lg026_has_critical_keyword_found(self):
        """TC-LG-026: _has_critical_keyword returns True for 'cancelled'."""
        assert _has_critical_keyword("the flight was cancelled")

    def test_lg027_has_critical_keyword_not_found(self):
        """TC-LG-027: _has_critical_keyword returns False for benign text."""
        assert not _has_critical_keyword("great comfortable flight")

    def test_lg028_has_risk_keyword_found(self):
        """TC-LG-028: _has_risk_keyword returns True for 'delayed'."""
        assert _has_risk_keyword("the flight was delayed by 3 hours")

    def test_lg029_has_risk_keyword_not_found(self):
        """TC-LG-029: _has_risk_keyword returns False for benign text."""
        assert not _has_risk_keyword("on time and comfortable")

    def test_lg030_keyword_detection_case_insensitive(self):
        """TC-LG-030: Keyword matching works on pre-lowercased text (default)."""
        # generate_labels lowercases before scanning
        assert _has_critical_keyword("the flight was cancelled by staff")

    def test_lg031_multi_keywords_in_same_text(self):
        """TC-LG-031: Text with multiple keywords correctly detected."""
        text = "cancelled, lost baggage, and rude staff"
        assert _has_critical_keyword(text)
        assert _has_risk_keyword(text)


# ── TC-LG-036 — 045: generate_labels (integration) ───────────────────────────

class TestGenerateLabelsIntegration:

    def test_lg036_returns_severity_label_column(self, clean_df):
        """TC-LG-036: generate_labels adds 'severity_label' column."""
        result = generate_labels(clean_df.copy())
        assert "severity_label" in result.columns

    def test_lg037_returns_severity_id_column(self, clean_df):
        """TC-LG-037: generate_labels adds 'severity_id' column."""
        result = generate_labels(clean_df.copy())
        assert "severity_id" in result.columns

    def test_lg038_returns_risk_flag_column(self, clean_df):
        """TC-LG-038: generate_labels adds 'risk_flag' column."""
        result = generate_labels(clean_df.copy())
        assert "risk_flag" in result.columns

    def test_lg039_severity_id_mapping_consistent(self, clean_df):
        """TC-LG-039: Every severity_id matches the severity_label mapping."""
        result = generate_labels(clean_df.copy())
        mapping = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        for _, row in result.iterrows():
            assert row["severity_id"] == mapping[row["severity_label"]]

    def test_lg040_risk_flag_only_0_or_1(self, clean_df):
        """TC-LG-040: risk_flag column contains only 0 or 1."""
        result = generate_labels(clean_df.copy())
        assert set(result["risk_flag"].unique()).issubset({0, 1})

    def test_lg041_no_temp_text_column_in_output(self, clean_df):
        """TC-LG-041: The internal '_text' helper column is not in the output."""
        result = generate_labels(clean_df.copy())
        assert "_text" not in result.columns

    def test_lg042_row_count_unchanged(self, clean_df):
        """TC-LG-042: generate_labels does not add or remove rows."""
        original_len = len(clean_df)
        result = generate_labels(clean_df.copy())
        assert len(result) == original_len

    def test_lg043_rating_1_labelled_critical(self, clean_df):
        """TC-LG-043: Row with Overall_Rating=1.0 gets severity_label='Critical'."""
        result = generate_labels(clean_df.copy())
        assert result.loc[0, "severity_label"] == "Critical"

    def test_lg044_rating_7_recommended_yes_labelled_low(self, clean_df):
        """TC-LG-044: Row with Overall_Rating=7.0 and Recommended='yes' → Low."""
        result = generate_labels(clean_df.copy())
        assert result.loc[1, "severity_label"] == "Low"

    def test_lg045_original_df_not_mutated(self, clean_df):
        """TC-LG-045: generate_labels does not mutate the input DataFrame."""
        original_cols = list(clean_df.columns)
        _ = generate_labels(clean_df.copy())
        assert list(clean_df.columns) == original_cols
