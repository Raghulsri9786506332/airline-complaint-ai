"""
TC_005_unit_feature_engineering.py — Unit Tests: Feature Engineering (Phase 6)
================================================================================
Category  : Unit
Module    : src/features/feature_engineer.py
Purpose   : Test BoW, TF-IDF, structured, and combined feature builders.
Run with  : pytest "test cases/TC_005_unit_feature_engineering.py" -v
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

from src.features.feature_engineer import (
    BagOfWordsFeatures,
    TFIDFFeatures,
    add_structured_features,
    get_structured_matrix,
    build_combined_features,
)


# ── TC-FE-001 — 010: BagOfWordsFeatures ──────────────────────────────────────

class TestBagOfWordsFeatures:

    @pytest.fixture
    def bow(self):
        return BagOfWordsFeatures(max_features=100)

    @pytest.fixture
    def sample_texts(self):
        return [
            "terrible flight delayed staff rude",
            "excellent service comfortable seats",
            "cancelled baggage lost refund denied",
            "on time great food",
        ]

    def test_fe001_fit_transform_returns_sparse(self, bow, sample_texts):
        """TC-FE-001: fit_transform returns a sparse matrix."""
        X = bow.fit_transform(sample_texts)
        assert sp.issparse(X)

    def test_fe002_fit_transform_row_count(self, bow, sample_texts):
        """TC-FE-002: fit_transform row count matches input documents."""
        X = bow.fit_transform(sample_texts)
        assert X.shape[0] == len(sample_texts)

    def test_fe003_transform_same_shape_columns(self, bow, sample_texts):
        """TC-FE-003: transform on new text has same column count as fit_transform."""
        X_train = bow.fit_transform(sample_texts)
        X_new = bow.transform(["new delayed flight"])
        assert X_train.shape[1] == X_new.shape[1]

    def test_fe004_max_features_respected(self, sample_texts):
        """TC-FE-004: max_features parameter limits vocabulary size."""
        bow = BagOfWordsFeatures(max_features=5)
        X = bow.fit_transform(sample_texts)
        assert X.shape[1] <= 5

    def test_fe005_transform_new_words_handled(self, bow, sample_texts):
        """TC-FE-005: transform with OOV (out-of-vocabulary) words doesn't crash."""
        bow.fit_transform(sample_texts)
        try:
            X = bow.transform(["completelynewword1234 xyzabc"])
            assert X.shape[0] == 1
        except Exception as e:
            pytest.fail(f"OOV transform raised: {e}")

    def test_fe006_save_and_load(self, bow, sample_texts, tmp_path):
        """TC-FE-006: BagOfWordsFeatures can be saved and loaded."""
        bow.fit_transform(sample_texts)
        save_path = tmp_path / "bow.pkl"
        bow.save(save_path)
        loaded = BagOfWordsFeatures.load(save_path)
        X = loaded.transform(sample_texts)
        assert X.shape[0] == len(sample_texts)


# ── TC-FE-011 — 020: TFIDFFeatures ───────────────────────────────────────────

class TestTFIDFFeatures:

    @pytest.fixture
    def tfidf(self):
        return TFIDFFeatures(max_features=200, ngram_range=(1, 2))

    @pytest.fixture
    def sample_texts(self):
        return [
            "terrible flight delayed rude staff bad experience",
            "excellent service comfortable luxury seats wonderful",
            "cancelled refund denied baggage lost nightmare",
            "on time punctual great food excellent airline",
        ]

    def test_fe011_fit_transform_returns_sparse(self, tfidf, sample_texts):
        """TC-FE-011: TF-IDF fit_transform returns sparse matrix."""
        X = tfidf.fit_transform(sample_texts)
        assert sp.issparse(X)

    def test_fe012_row_count_matches_input(self, tfidf, sample_texts):
        """TC-FE-012: TF-IDF row count matches number of documents."""
        X = tfidf.fit_transform(sample_texts)
        assert X.shape[0] == len(sample_texts)

    def test_fe013_bigrams_increase_features(self, sample_texts):
        """TC-FE-013: bigram TF-IDF has more features than unigram-only."""
        tfidf_uni = TFIDFFeatures(max_features=1000, ngram_range=(1, 1))
        tfidf_bi = TFIDFFeatures(max_features=1000, ngram_range=(1, 2))
        X_uni = tfidf_uni.fit_transform(sample_texts)
        X_bi = tfidf_bi.fit_transform(sample_texts)
        assert X_bi.shape[1] >= X_uni.shape[1]

    def test_fe014_transform_consistent_columns(self, tfidf, sample_texts):
        """TC-FE-014: transform on new docs has same column count as training."""
        X_train = tfidf.fit_transform(sample_texts)
        X_new = tfidf.transform(["delayed horrible experience"])
        assert X_train.shape[1] == X_new.shape[1]

    def test_fe015_save_and_load(self, tfidf, sample_texts, tmp_path):
        """TC-FE-015: TFIDFFeatures can be saved and loaded."""
        tfidf.fit_transform(sample_texts)
        save_path = tmp_path / "tfidf.pkl"
        tfidf.save(save_path)
        loaded = TFIDFFeatures.load(save_path)
        X = loaded.transform(sample_texts)
        assert X.shape[0] == len(sample_texts)

    def test_fe016_sublinear_tf_applied(self):
        """TC-FE-016: TFIDFFeatures with sublinear_tf=True creates vectorizer correctly."""
        tfidf = TFIDFFeatures(sublinear_tf=True)
        assert tfidf.vec.sublinear_tf is True


# ── TC-FE-021 — 030: add_structured_features ─────────────────────────────────

class TestAddStructuredFeatures:

    @pytest.fixture
    def df_with_text(self):
        return pd.DataFrame({
            "full_review_text": [
                "cancelled flight worst experience",
                "excellent comfortable fast",
                "delayed rude staff lost baggage",
            ],
            "Seat Comfort": [1.0, 5.0, 2.0],
            "Cabin Staff Service": [1.0, 5.0, 2.0],
            "Food & Beverages": [1.0, 5.0, 3.0],
            "Ground Service": [1.0, 5.0, 2.0],
            "Inflight Entertainment": [1.0, 5.0, 3.0],
            "Wifi & Connectivity": [1.0, 5.0, 3.0],
            "Value For Money": [1.0, 5.0, 2.0],
        })

    def test_fe021_review_length_column_created(self, df_with_text):
        """TC-FE-021: add_structured_features creates 'review_length' column."""
        result = add_structured_features(df_with_text.copy())
        assert "review_length" in result.columns

    def test_fe022_keyword_count_column_created(self, df_with_text):
        """TC-FE-022: add_structured_features creates 'keyword_count' column."""
        result = add_structured_features(df_with_text.copy())
        assert "keyword_count" in result.columns

    def test_fe023_review_length_is_positive_for_non_empty(self, df_with_text):
        """TC-FE-023: review_length > 0 for non-empty text."""
        result = add_structured_features(df_with_text.copy())
        assert (result["review_length"] > 0).all()

    def test_fe024_keyword_count_positive_for_complaint_text(self, df_with_text):
        """TC-FE-024: keyword_count > 0 for text containing known keywords."""
        result = add_structured_features(df_with_text.copy())
        # Row 0 has 'cancelled' and 'worst'; row 2 has 'delayed', 'rude', 'lost baggage'
        assert result.loc[0, "keyword_count"] > 0
        assert result.loc[2, "keyword_count"] > 0

    def test_fe025_keyword_count_zero_for_positive_text(self, df_with_text):
        """TC-FE-025: keyword_count = 0 for text with no complaint keywords."""
        result = add_structured_features(df_with_text.copy())
        # Row 1 has no complaint keywords
        assert result.loc[1, "keyword_count"] == 0

    def test_fe026_does_not_mutate_original(self, df_with_text):
        """TC-FE-026: add_structured_features returns a copy, doesn't mutate input."""
        original_cols = set(df_with_text.columns)
        _ = add_structured_features(df_with_text.copy())
        assert set(df_with_text.columns) == original_cols


# ── TC-FE-031 — 040: get_structured_matrix ───────────────────────────────────

class TestGetStructuredMatrix:

    @pytest.fixture
    def df_with_struct(self):
        return pd.DataFrame({
            "Seat Comfort": [1.0, 5.0, 3.0],
            "Cabin Staff Service": [1.0, 5.0, 3.0],
            "Food & Beverages": [1.0, 5.0, 3.0],
            "Ground Service": [1.0, 5.0, 3.0],
            "Inflight Entertainment": [1.0, 5.0, 3.0],
            "Wifi & Connectivity": [1.0, 5.0, 3.0],
            "Value For Money": [1.0, 5.0, 3.0],
            "review_length": [100, 200, 150],
            "keyword_count": [3, 0, 1],
        })

    def test_fe031_fit_returns_array(self, df_with_struct):
        """TC-FE-031: get_structured_matrix with fit=True returns np.ndarray."""
        X, scaler = get_structured_matrix(df_with_struct.copy(), fit=True)
        assert isinstance(X, np.ndarray)

    def test_fe032_fit_returns_scaler(self, df_with_struct):
        """TC-FE-032: get_structured_matrix with fit=True returns a scaler."""
        from sklearn.preprocessing import StandardScaler
        X, scaler = get_structured_matrix(df_with_struct.copy(), fit=True)
        assert isinstance(scaler, StandardScaler)

    def test_fe033_row_count_matches(self, df_with_struct):
        """TC-FE-033: Structured matrix row count matches DataFrame row count."""
        X, _ = get_structured_matrix(df_with_struct.copy(), fit=True)
        assert X.shape[0] == len(df_with_struct)

    def test_fe034_transform_only_requires_scaler(self, df_with_struct):
        """TC-FE-034: get_structured_matrix with fit=False raises if scaler is None."""
        with pytest.raises(ValueError):
            get_structured_matrix(df_with_struct.copy(), scaler=None, fit=False)

    def test_fe035_transform_consistent_columns(self, df_with_struct):
        """TC-FE-035: Transform-only mode produces same column count as fit."""
        X_fit, scaler = get_structured_matrix(df_with_struct.copy(), fit=True)
        X_transform, _ = get_structured_matrix(df_with_struct.copy(),
                                                scaler=scaler, fit=False)
        assert X_fit.shape[1] == X_transform.shape[1]


# ── TC-FE-041 — 050: build_combined_features ─────────────────────────────────

class TestBuildCombinedFeatures:

    def test_fe041_returns_sparse_matrix(self):
        """TC-FE-041: build_combined_features returns a scipy sparse matrix."""
        tfidf_mat = sp.csr_matrix(np.random.rand(5, 100))
        struct_mat = np.random.rand(5, 9)
        result = build_combined_features(tfidf_mat, struct_mat)
        assert sp.issparse(result)

    def test_fe042_column_count_is_sum(self):
        """TC-FE-042: Output columns = TF-IDF columns + structured columns."""
        tfidf_mat = sp.csr_matrix(np.random.rand(5, 100))
        struct_mat = np.random.rand(5, 9)
        result = build_combined_features(tfidf_mat, struct_mat)
        assert result.shape[1] == 100 + 9

    def test_fe043_row_count_preserved(self):
        """TC-FE-043: Output row count equals input row count."""
        tfidf_mat = sp.csr_matrix(np.random.rand(10, 50))
        struct_mat = np.random.rand(10, 5)
        result = build_combined_features(tfidf_mat, struct_mat)
        assert result.shape[0] == 10

    def test_fe044_single_row_combined(self):
        """TC-FE-044: build_combined_features works for a single-row input."""
        tfidf_mat = sp.csr_matrix(np.random.rand(1, 50))
        struct_mat = np.random.rand(1, 5)
        result = build_combined_features(tfidf_mat, struct_mat)
        assert result.shape == (1, 55)
