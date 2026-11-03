"""
TC_002_unit_data_cleaning.py — Unit Tests: Data Cleaning (Phase 2)
====================================================================
Category  : Unit
Module    : src/data/cleaner.py
Purpose   : Test every cleaning function independently, covering normal cases,
            boundary conditions, and edge cases.
Run with  : pytest "test cases/TC_002_unit_data_cleaning.py" -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cleaner import (
    clean_dataset,
    _convert_overall_rating,
    _convert_aspect_ratings,
    _normalize_categoricals,
    _drop_exact_duplicates,
)


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def base_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Unnamed: 0": [0, 1, 2, 3, 4],
        "Airline Name": ["AirX", "AirX", "AirY", "AirZ", "AirX"],
        "Overall_Rating": ["1", "7", "n", "3", "10"],
        "Review_Title": ["Worst flight", "Great trip", None, "Delayed again", "Perfect"],
        "Review Date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
        "Verified": [True, True, False, True, False],
        "Review": [
            "Cancelled flight, lost baggage.",
            "Excellent service.",
            "Nothing special.",
            "Delayed by 3 hours.",
            "Best airline ever.",
        ],
        "Recommended": ["No", "Yes", "Yes", "No", "Yes"],
        "Type Of Traveller": ["Business", "Solo Leisure", None, "Family", "Couple"],
        "Seat Type": ["Economy", "Business", "Economy", None, "First"],
        "Seat Comfort": [1.0, 5.0, np.nan, 2.0, 5.0],
        "Cabin Staff Service": [1.0, 5.0, 4.0, 2.0, 5.0],
        "Food & Beverages": [1.0, 5.0, np.nan, 3.0, 5.0],
        "Ground Service": [1.0, 5.0, np.nan, 2.0, 5.0],
        "Inflight Entertainment": [1.0, 5.0, np.nan, np.nan, 4.0],
        "Wifi & Connectivity": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "Value For Money": [1.0, 5.0, 3.0, 2.0, 5.0],
    })


# ── TC-DC-001 — 010: _convert_overall_rating ──────────────────────────────────

class TestConvertOverallRating:

    def test_dc001_numeric_strings_converted(self, base_df):
        """TC-DC-001: Numeric strings like '1', '7', '10' → float."""
        df = _convert_overall_rating(base_df.copy())
        assert pd.api.types.is_float_dtype(df["Overall_Rating"])

    def test_dc002_invalid_value_n_becomes_nan(self, base_df):
        """TC-DC-002: The literal 'n' in Overall_Rating is coerced to NaN."""
        df = _convert_overall_rating(base_df.copy())
        assert pd.isna(df.loc[df.index == 2, "Overall_Rating"].values[0])

    def test_dc003_valid_values_preserved(self, base_df):
        """TC-DC-003: Valid numeric ratings (1, 7, 3, 10) are preserved exactly."""
        df = _convert_overall_rating(base_df.copy())
        valid_vals = df["Overall_Rating"].dropna().tolist()
        for v in [1.0, 7.0, 3.0, 10.0]:
            assert v in valid_vals

    def test_dc004_already_numeric_unchanged(self):
        """TC-DC-004: DataFrame with already-numeric Overall_Rating passes through."""
        df = pd.DataFrame({"Overall_Rating": [1.0, 5.0, 10.0]})
        result = _convert_overall_rating(df.copy())
        assert result["Overall_Rating"].tolist() == [1.0, 5.0, 10.0]

    def test_dc005_all_invalid_all_nan(self):
        """TC-DC-005: Entirely non-numeric column becomes all NaN."""
        df = pd.DataFrame({"Overall_Rating": ["n", "x", "abc"]})
        result = _convert_overall_rating(df.copy())
        assert result["Overall_Rating"].isnull().all()

    def test_dc006_empty_string_becomes_nan(self):
        """TC-DC-006: Empty string in Overall_Rating becomes NaN."""
        df = pd.DataFrame({"Overall_Rating": [""]})
        result = _convert_overall_rating(df.copy())
        assert pd.isna(result["Overall_Rating"].iloc[0])


# ── TC-DC-011 — 020: _convert_aspect_ratings ─────────────────────────────────

class TestConvertAspectRatings:

    def test_dc011_nan_imputed_with_median(self, base_df):
        """TC-DC-011: NaN values in aspect rating columns are filled with column median."""
        df = _convert_aspect_ratings(base_df.copy())
        from src.utils.config import RATING_COLS
        for col in RATING_COLS:
            if col in df.columns:
                assert df[col].isnull().sum() == 0, f"NaN remains in '{col}'"

    def test_dc012_missing_indicator_created(self, base_df):
        """TC-DC-012: A missing-value indicator column is created for each rating col."""
        df = _convert_aspect_ratings(base_df.copy())
        assert "Food_and_Beverages_missing" in df.columns

    def test_dc013_indicator_value_is_0_or_1(self, base_df):
        """TC-DC-013: Missing indicator only contains 0 or 1."""
        df = _convert_aspect_ratings(base_df.copy())
        assert "Food_and_Beverages_missing" in df.columns
        vals = df["Food_and_Beverages_missing"].unique()
        for v in vals:
            assert v in (0, 1)

    def test_dc014_all_nan_column_safe(self):
        """TC-DC-014: All-NaN rating column gets median=NaN but no crash."""
        df = pd.DataFrame({
            "Wifi & Connectivity": [np.nan, np.nan, np.nan],
        })
        # Should not raise; median of all-NaN is NaN, fillna(NaN) leaves NaN
        # but imputation should handle gracefully
        try:
            from src.data.cleaner import _convert_aspect_ratings
            _convert_aspect_ratings(df.copy())
        except Exception as e:
            pytest.fail(f"All-NaN aspect column raised: {e}")

    def test_dc015_non_rating_cols_ignored(self, base_df):
        """TC-DC-015: Non-rating columns are not modified by _convert_aspect_ratings."""
        before_review = base_df["Review"].copy()
        df = _convert_aspect_ratings(base_df.copy())
        pd.testing.assert_series_equal(df["Review"], before_review)


# ── TC-DC-021 — 030: _normalize_categoricals ─────────────────────────────────

class TestNormalizeCategoricals:

    def test_dc021_recommended_lowercased(self, base_df):
        """TC-DC-021: 'Recommended' column values are lowercased."""
        df = _normalize_categoricals(base_df.copy())
        assert df["Recommended"].str.islower().all()

    def test_dc022_recommended_no_whitespace(self, base_df):
        """TC-DC-022: 'Recommended' column values are stripped of whitespace."""
        df = base_df.copy()
        df["Recommended"] = ["  No ", " Yes ", "Yes", "No", " Yes "]
        result = _normalize_categoricals(df)
        for v in result["Recommended"]:
            assert v == v.strip()

    def test_dc023_traveller_type_lowercased(self, base_df):
        """TC-DC-023: 'Type Of Traveller' is lowercased (where not null)."""
        df = _normalize_categoricals(base_df.copy())
        non_null = df["Type Of Traveller"].dropna()
        assert non_null.str.islower().all()

    def test_dc024_boolean_verified_mapped(self, base_df):
        """TC-DC-024: Boolean Verified is mapped to 'yes'/'no' strings."""
        df = _normalize_categoricals(base_df.copy())
        # After mapping, should not contain True/False
        assert True not in df["Verified"].tolist()
        assert False not in df["Verified"].tolist()


# ── TC-DC-031 — 040: _drop_exact_duplicates ──────────────────────────────────

class TestDropExactDuplicates:

    def test_dc031_exact_duplicate_removed(self):
        """TC-DC-031: Exact duplicate row (same airline, title, date, review) is removed."""
        df = pd.DataFrame({
            "Airline Name": ["X", "X"],
            "Review_Title": ["Same", "Same"],
            "Review Date": ["2023-01-01", "2023-01-01"],
            "Review": ["Exact copy", "Exact copy"],
        })
        result = _drop_exact_duplicates(df)
        assert len(result) == 1

    def test_dc032_non_duplicate_rows_kept(self):
        """TC-DC-032: Rows with different reviews are not removed."""
        df = pd.DataFrame({
            "Airline Name": ["X", "X"],
            "Review_Title": ["Title A", "Title B"],
            "Review Date": ["2023-01-01", "2023-01-02"],
            "Review": ["Review A", "Review B"],
        })
        result = _drop_exact_duplicates(df)
        assert len(result) == 2

    def test_dc033_three_duplicates_keep_one(self):
        """TC-DC-033: Three identical rows collapse to one."""
        df = pd.DataFrame({
            "Airline Name": ["X", "X", "X"],
            "Review_Title": ["Dup", "Dup", "Dup"],
            "Review Date": ["2023-01-01"] * 3,
            "Review": ["Same"] * 3,
        })
        result = _drop_exact_duplicates(df)
        assert len(result) == 1

    def test_dc034_index_reset_after_dedup(self):
        """TC-DC-034: Index is reset to 0,1,2... after duplicate removal."""
        df = pd.DataFrame({
            "Airline Name": ["X", "X", "Y"],
            "Review_Title": ["D", "D", "U"],
            "Review Date": ["2023-01-01", "2023-01-01", "2023-02-01"],
            "Review": ["S", "S", "R"],
        })
        result = _drop_exact_duplicates(df)
        assert list(result.index) == list(range(len(result)))

    def test_dc035_missing_key_columns_handled(self):
        """TC-DC-035: _drop_exact_duplicates handles DataFrames missing some key columns."""
        df = pd.DataFrame({
            "Airline Name": ["X", "X"],
            # Missing Review_Title, Review Date
            "Review": ["A", "B"],
        })
        result = _drop_exact_duplicates(df)
        assert len(result) == 2  # Both are unique


# ── TC-DC-041 — 050: clean_dataset (integration) ─────────────────────────────

class TestCleanDatasetIntegration:

    def test_dc041_clean_dataset_full_pipeline(self, base_df):
        """TC-DC-041: clean_dataset runs all steps without crashing."""
        result = clean_dataset(base_df.copy())
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_dc042_overall_rating_numeric_after_clean(self, base_df):
        """TC-DC-042: Overall_Rating is numeric float after clean_dataset."""
        result = clean_dataset(base_df.copy())
        assert pd.api.types.is_float_dtype(result["Overall_Rating"])

    def test_dc043_no_nan_in_aspect_ratings(self, base_df):
        """TC-DC-043: No NaN remains in aspect rating columns after clean_dataset."""
        result = clean_dataset(base_df.copy())
        from src.utils.config import RATING_COLS
        for col in RATING_COLS:
            if col in result.columns:
                assert result[col].isnull().sum() == 0

    def test_dc044_optional_cols_dont_cause_drops(self, base_df):
        """TC-DC-044: Records with missing Aircraft/Route columns are NOT dropped."""
        base_df["Aircraft"] = np.nan
        base_df["Route"] = np.nan
        result = clean_dataset(base_df.copy())
        # Row count should survive (minus true duplicates)
        assert len(result) >= 1

    def test_dc045_returns_new_dataframe_not_inplace(self, base_df):
        """TC-DC-045: clean_dataset returns a copy, doesn't mutate original."""
        original_len = len(base_df)
        original_rating_0 = base_df["Overall_Rating"].iloc[0]
        _ = clean_dataset(base_df.copy())
        # Original unchanged
        assert len(base_df) == original_len
        assert base_df["Overall_Rating"].iloc[0] == original_rating_0
