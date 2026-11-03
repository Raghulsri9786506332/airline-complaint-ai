"""
TC_009_boundary_edge_cases.py — Boundary & Edge Case Tests
============================================================
Category  : Boundary / Edge Cases
Purpose   : Test system behaviour at rating boundaries, extreme inputs,
            empty DataFrames, single-row DataFrames, and unusual data formats.
Run with  : pytest "test cases/TC_009_boundary_edge_cases.py" -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.label_generator import _assign_severity, _assign_risk


# ── Shared helpers ────────────────────────────────────────────────────────────

def row(rating=5.0, rec="yes", text="average experience",
        seat=3.0, cabin=3.0, food=3.0, ground=3.0,
        inflight=3.0, wifi=3.0, value=3.0) -> pd.Series:
    return pd.Series({
        "Overall_Rating": rating,
        "Recommended": rec,
        "_text": text.lower(),
        "Seat Comfort": seat, "Cabin Staff Service": cabin,
        "Food & Beverages": food, "Ground Service": ground,
        "Inflight Entertainment": inflight,
        "Wifi & Connectivity": wifi, "Value For Money": value,
    })


# ── TC-BC-001 — 015: Rating boundary conditions ───────────────────────────────

class TestRatingBoundaries:

    def test_bc001_rating_0_treated_as_critical(self):
        """TC-BC-001: Rating=0 (below scale minimum) → Critical."""
        assert _assign_severity(row(rating=0.0)) == "Critical"

    def test_bc002_rating_1_critical(self):
        """TC-BC-002: Rating=1 → Critical (lower inclusive boundary)."""
        assert _assign_severity(row(rating=1.0)) == "Critical"

    def test_bc003_rating_2_critical(self):
        """TC-BC-003: Rating=2 → Critical (upper boundary of critical zone)."""
        assert _assign_severity(row(rating=2.0)) == "Critical"

    def test_bc004_rating_2_5_is_high(self):
        """TC-BC-004: Rating=2.5 (between zones) → High."""
        result = _assign_severity(row(rating=2.5,
            seat=4.0, cabin=4.0, food=4.0, ground=4.0,
            inflight=4.0, wifi=4.0, value=4.0))
        assert result == "High"

    def test_bc005_rating_3_is_high(self):
        """TC-BC-005: Rating=3 → High (no keywords, good aspects)."""
        result = _assign_severity(row(rating=3.0,
            seat=4.0, cabin=4.0, food=4.0, ground=4.0,
            inflight=4.0, wifi=4.0, value=4.0))
        assert result == "High"

    def test_bc006_rating_4_is_high(self):
        """TC-BC-006: Rating=4 → High (boundary)."""
        result = _assign_severity(row(rating=4.0,
            seat=4.0, cabin=4.0, food=4.0, ground=4.0,
            inflight=4.0, wifi=4.0, value=4.0))
        assert result == "High"

    def test_bc007_rating_4_5_is_medium(self):
        """TC-BC-007: Rating=4.5 → Medium (falls into [5,6] zone after ceiling)."""
        result = _assign_severity(row(rating=4.5))
        # 4.5 is > 4, so it enters the Medium branch (rating <= 6)
        assert result == "Medium"

    def test_bc008_rating_5_is_medium(self):
        """TC-BC-008: Rating=5 → Medium."""
        assert _assign_severity(row(rating=5.0, text="average flight")) == "Medium"

    def test_bc009_rating_6_is_medium(self):
        """TC-BC-009: Rating=6 → Medium (upper boundary)."""
        assert _assign_severity(row(rating=6.0, text="okay flight")) == "Medium"

    def test_bc010_rating_6_5_is_low(self):
        """TC-BC-010: Rating=6.5 → Low (falls into >=7 zone? Actually 6.5 < 7)."""
        # 6.5: the rule is elif rating <= 6 → Medium; else → Low
        # 6.5 > 6, so it goes to else → Low
        result = _assign_severity(row(rating=6.5, rec="yes", text="decent flight"))
        assert result == "Low"

    def test_bc011_rating_7_recommended_yes_is_low(self):
        """TC-BC-011: Rating=7 + Recommended=yes → Low."""
        assert _assign_severity(row(rating=7.0, rec="yes")) == "Low"

    def test_bc012_rating_10_is_low(self):
        """TC-BC-012: Rating=10 (max) + Recommended=yes → Low."""
        assert _assign_severity(row(rating=10.0, rec="yes")) == "Low"

    def test_bc013_rating_above_10_handled(self):
        """TC-BC-013: Rating > 10 (outside scale) doesn't crash."""
        result = _assign_severity(row(rating=15.0, rec="yes"))
        assert result in {"Low", "Medium", "High", "Critical"}

    def test_bc014_negative_rating_handled(self):
        """TC-BC-014: Negative rating doesn't crash."""
        result = _assign_severity(row(rating=-1.0))
        assert result in {"Low", "Medium", "High", "Critical"}

    def test_bc015_nan_rating_falls_back_correctly(self):
        """TC-BC-015: NaN Overall_Rating defaults to Medium (no keyword)."""
        result = _assign_severity(row(rating=np.nan, text="average experience"))
        assert result == "Medium"


# ── TC-BC-016 — 025: Aspect rating edge cases ─────────────────────────────────

class TestAspectRatingBoundaries:

    def test_bc016_exactly_2_poor_aspects_triggers_risk(self):
        """TC-BC-016: Exactly 2 poor aspects (<=2) + recommended=no + rating<=4 → risk=1."""
        r = row(rating=3.0, rec="no",
                seat=2.0, cabin=2.0,
                food=4.0, ground=4.0, inflight=4.0, wifi=4.0, value=4.0)
        assert _assign_risk(r) == 1

    def test_bc017_exactly_1_poor_aspect_no_keyword_no_risk(self):
        """TC-BC-017: Only 1 poor aspect + no risk keyword + recommended=no + rating=3 → risk=0."""
        r = row(rating=3.0, rec="no", text="ordinary flight",
                seat=2.0, cabin=4.0, food=4.0,
                ground=4.0, inflight=4.0, wifi=4.0, value=4.0)
        assert _assign_risk(r) == 0

    def test_bc018_exactly_3_poor_aspects_triggers_critical(self):
        """TC-BC-018: Rating=3 + exactly 3 poor aspects + recommended=no → Critical."""
        r = row(rating=3.0, rec="no",
                seat=1.0, cabin=1.0, food=1.0,
                ground=4.0, inflight=4.0, wifi=4.0, value=4.0)
        assert _assign_severity(r) == "Critical"

    def test_bc019_all_aspects_at_boundary_value_2(self):
        """TC-BC-019: All aspects at exactly 2.0 → all count as poor."""
        r = row(rating=3.0, rec="no",
                seat=2.0, cabin=2.0, food=2.0,
                ground=2.0, inflight=2.0, wifi=2.0, value=2.0)
        result = _assign_severity(r)
        assert result == "Critical"  # >=3 poor aspects → Critical

    def test_bc020_all_aspects_at_boundary_value_3(self):
        """TC-BC-020: All aspects at exactly 3.0 → none count as poor (<=2 rule)."""
        r = row(rating=3.0, rec="no",
                seat=3.0, cabin=3.0, food=3.0,
                ground=3.0, inflight=3.0, wifi=3.0, value=3.0,
                text="delayed")
        result = _assign_severity(r)
        # No poor aspects (3 > 2), keyword 'delayed' is a risk kw but not critical
        # Rating=3, recommended=no, has_critical_kw=False → High base
        # No critical keyword upgrade → stays High
        assert result == "High"


# ── TC-BC-021 — 030: DataFrame edge cases ────────────────────────────────────

class TestDataFrameEdgeCases:

    def test_bc021_single_row_dataframe_clean(self):
        """TC-BC-021: clean_dataset handles a single-row DataFrame."""
        from src.data.cleaner import clean_dataset
        df = pd.DataFrame({
            "Airline Name": ["AirX"],
            "Overall_Rating": ["5"],
            "Review_Title": ["OK"],
            "Review Date": ["2023-01-01"],
            "Verified": [True],
            "Review": ["Decent flight"],
            "Recommended": ["Yes"],
            "Seat Comfort": [3.0],
            "Cabin Staff Service": [3.0],
            "Food & Beverages": [3.0],
            "Ground Service": [3.0],
            "Inflight Entertainment": [3.0],
            "Wifi & Connectivity": [3.0],
            "Value For Money": [3.0],
        })
        result = clean_dataset(df)
        assert len(result) == 1

    def test_bc022_single_row_dataframe_label(self):
        """TC-BC-022: generate_labels handles a single-row DataFrame."""
        from src.data.label_generator import generate_labels
        df = pd.DataFrame({
            "Airline Name": ["AirX"],
            "Overall_Rating": [5.0],
            "Review_Title": ["OK"],
            "Review": ["Decent flight nothing special"],
            "Recommended": ["yes"],
            "Seat Comfort": [3.0],
            "Cabin Staff Service": [3.0],
            "Food & Beverages": [3.0],
            "Ground Service": [3.0],
            "Inflight Entertainment": [3.0],
            "Wifi & Connectivity": [3.0],
            "Value For Money": [3.0],
        })
        result = generate_labels(df)
        assert len(result) == 1
        assert "severity_label" in result.columns

    def test_bc023_all_same_reviews_deduplication(self):
        """TC-BC-023: DataFrame with 5 identical rows deduplicates to 1."""
        from src.data.cleaner import _drop_exact_duplicates
        df = pd.DataFrame({
            "Airline Name": ["X"] * 5,
            "Review_Title": ["Same"] * 5,
            "Review Date": ["2023-01-01"] * 5,
            "Review": ["Exact copy"] * 5,
        })
        result = _drop_exact_duplicates(df)
        assert len(result) == 1

    def test_bc024_all_nan_recommended_column(self):
        """TC-BC-024: All-NaN 'Recommended' column doesn't crash label generation."""
        from src.data.label_generator import generate_labels
        df = pd.DataFrame({
            "Overall_Rating": [5.0, 3.0],
            "Review_Title": ["OK", "Bad"],
            "Review": ["OK flight", "Delayed flight"],
            "Recommended": [np.nan, np.nan],
            "Seat Comfort": [3.0, 2.0],
            "Cabin Staff Service": [3.0, 2.0],
            "Food & Beverages": [3.0, 2.0],
            "Ground Service": [3.0, 2.0],
            "Inflight Entertainment": [3.0, 2.0],
            "Wifi & Connectivity": [3.0, 2.0],
            "Value For Money": [3.0, 2.0],
        })
        try:
            result = generate_labels(df)
            assert "severity_label" in result.columns
        except Exception as e:
            pytest.fail(f"All-NaN Recommended crashed: {e}")

    def test_bc025_very_long_review_text(self):
        """TC-BC-025: Very long review text (>10000 chars) processed without crash."""
        from src.preprocessing.text_preprocessor import build_full_text
        long_review = "The flight was delayed and terrible. " * 300  # ~11k chars
        result = build_full_text("Long review", long_review, "ml")
        assert isinstance(result, str)


# ── TC-BC-031 — 040: Text edge cases ─────────────────────────────────────────

class TestTextEdgeCases:

    def test_bc031_only_html_tags_in_review(self):
        """TC-BC-031: Review consisting only of HTML tags returns empty or near-empty."""
        from src.preprocessing.text_preprocessor import ml_clean
        result = ml_clean("<br/><p></p><div></div>")
        assert isinstance(result, str)

    def test_bc032_only_urls_in_review(self):
        """TC-BC-032: Review with only URLs returns empty or near-empty string."""
        from src.preprocessing.text_preprocessor import ml_clean
        result = ml_clean("https://airline.com http://reviews.example.com")
        assert isinstance(result, str)

    def test_bc033_only_numbers_in_review(self):
        """TC-BC-033: Review with only digits handled without crash."""
        from src.preprocessing.text_preprocessor import ml_clean
        result = ml_clean("1234 5678 9999")
        assert isinstance(result, str)

    def test_bc034_unicode_characters_handled(self):
        """TC-BC-034: Unicode (accents, non-Latin) in review doesn't crash."""
        from src.preprocessing.text_preprocessor import ml_clean
        result = ml_clean("Très bien! Schöne Fahrt. 旅行 とても良い")
        assert isinstance(result, str)

    def test_bc035_single_character_review(self):
        """TC-BC-035: Single character review handled gracefully."""
        from src.preprocessing.text_preprocessor import build_full_text
        result = build_full_text("A", "B", "ml")
        assert isinstance(result, str)

    def test_bc036_whitespace_only_review(self):
        """TC-BC-036: Whitespace-only review returns empty string."""
        from src.preprocessing.text_preprocessor import ml_clean
        result = ml_clean("     \n\t   ")
        assert result.strip() == ""

    def test_bc037_review_with_mixed_encodings(self):
        """TC-BC-037: Review with mixed UTF-8 encoding issues handled."""
        from src.preprocessing.text_preprocessor import transformer_clean
        try:
            text = "Flight was délai\u00e9d (\u00e0 l'h\u00f4tel)"
            result = transformer_clean(text)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"Unicode handling failed: {e}")

    def test_bc038_escape_sequences_in_review(self):
        """TC-BC-038: Escape characters (\\n, \\t) in review handled."""
        from src.preprocessing.text_preprocessor import dl_clean
        result = dl_clean("Flight was\\nterrible\\tand rude")
        assert isinstance(result, str)
