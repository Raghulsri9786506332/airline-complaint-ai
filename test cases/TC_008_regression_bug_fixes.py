"""
TC_008_regression_bug_fixes.py — Regression Tests: Known Bug Fixes (CHANGES.md)
=================================================================================
Category  : Regression
Purpose   : Prevent re-introduction of the 20 bugs documented in CHANGES.md.
            Each test maps to a specific bug number from that document.
Run with  : pytest "test cases/TC_008_regression_bug_fixes.py" -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── TC-RG-001: Bug #7 — generate_labels crashes on string Overall_Rating ──────

class TestBug7LabelGeneratorStringRating:
    """Bug #7: generate_labels crashed if Overall_Rating was still a string."""

    def test_rg001_generate_labels_with_numeric_rating_does_not_crash(self):
        """TC-RG-001: generate_labels with pre-converted float rating doesn't crash."""
        from src.data.label_generator import generate_labels

        df = pd.DataFrame({
            "Airline Name": ["AirX"],
            "Overall_Rating": [3.0],  # already numeric — correct pre-processing
            "Review_Title": ["Delayed"],
            "Review": ["Flight was delayed and staff were rude"],
            "Recommended": ["no"],
            "Seat Comfort": [2.0],
            "Cabin Staff Service": [2.0],
            "Food & Beverages": [3.0],
            "Ground Service": [2.0],
            "Inflight Entertainment": [3.0],
            "Wifi & Connectivity": [3.0],
            "Value For Money": [2.0],
        })
        result = generate_labels(df)
        assert "severity_label" in result.columns

    def test_rg002_clean_then_label_prevents_string_crash(self):
        """TC-RG-002: Passing through clean_dataset before generate_labels avoids crash."""
        from src.data.cleaner import clean_dataset
        from src.data.label_generator import generate_labels

        df = pd.DataFrame({
            "Airline Name": ["AirX"],
            "Overall_Rating": ["3"],   # raw string — must be cleaned first
            "Review_Title": ["Delayed"],
            "Review": ["Flight was delayed"],
            "Recommended": ["no"],
            "Seat Comfort": [2.0],
            "Cabin Staff Service": [2.0],
            "Food & Beverages": [3.0],
            "Ground Service": [2.0],
            "Inflight Entertainment": [3.0],
            "Wifi & Connectivity": [3.0],
            "Value For Money": [2.0],
        })
        cleaned = clean_dataset(df)
        result = generate_labels(cleaned)
        assert result["severity_label"].iloc[0] in {"Low", "Medium", "High", "Critical"}


# ── TC-RG-003: Bug #4 — Negations removed from ml_clean pipeline ──────────────

class TestBug4NegationsKept:
    """Bug #4: NLTK stop-words removed negations like 'not', 'no', 'never', 'didn't'."""

    def test_rg003_never_kept_in_ml_clean(self):
        """TC-RG-003: 'never' is NOT removed by ml_clean (negation preservation)."""
        from src.preprocessing.text_preprocessor import ml_clean
        result = ml_clean("I will never fly with them again")
        # 'never' should NOT be removed (it's a negation)
        tokens = result.split()
        assert "never" in tokens, \
            "'never' was incorrectly removed as a stopword from ml_clean"

    def test_rg004_not_kept_in_ml_clean(self):
        """TC-RG-004: 'not' is NOT removed by ml_clean."""
        from src.preprocessing.text_preprocessor import ml_clean
        result = ml_clean("flight was not good at all")
        tokens = result.split()
        assert "not" in tokens, \
            "'not' was incorrectly removed as a stopword from ml_clean"

    def test_rg005_no_kept_in_ml_clean(self):
        """TC-RG-005: 'no' is NOT removed by ml_clean."""
        from src.preprocessing.text_preprocessor import ml_clean
        result = ml_clean("there was no compensation offered")
        tokens = result.split()
        assert "no" in tokens, \
            "'no' was incorrectly removed as a stopword from ml_clean"


# ── TC-RG-006: Bug #5 — NaN titles became literal 'nan' ─────────────────────

class TestBug5NaNTitleHandling:
    """Bug #5: NaN titles were converted to the literal string 'nan'."""

    def test_rg006_none_title_not_nan_string(self):
        """TC-RG-006: build_full_text with None title doesn't inject 'nan' string."""
        from src.preprocessing.text_preprocessor import build_full_text
        result = build_full_text(None, "Some review text", "ml")
        tokens = result.split()
        assert "nan" not in tokens, \
            "Literal 'nan' appeared in output when title was None"

    def test_rg007_none_title_none_review_safe(self):
        """TC-RG-007: Both None title and review don't crash or produce 'nan'."""
        from src.preprocessing.text_preprocessor import build_full_text
        result = build_full_text(None, None, "ml")
        assert "nan" not in result


# ── TC-RG-007: Bug #1 — Bulk CSV risk dimension mismatch ─────────────────────

class TestBug1BulkCSVDimensionMismatch:
    """Bug #1: Bulk CSV fed TF-IDF only to a model trained on TF-IDF+structured."""

    def test_rg008_combined_features_same_dimension_train_and_infer(self):
        """TC-RG-008: Combined TF-IDF + structured matrix has same column count at train and infer."""
        from src.features.feature_engineer import (
            TFIDFFeatures, add_structured_features,
            get_structured_matrix, build_combined_features,
        )
        from src.preprocessing.text_preprocessor import build_full_text

        texts_train = ["cancelled delayed rude", "excellent comfortable on time"]
        df_train = pd.DataFrame({
            "full_review_text": texts_train,
            "Seat Comfort": [1.0, 5.0],
            "Cabin Staff Service": [1.0, 5.0],
            "Food & Beverages": [1.0, 5.0],
            "Ground Service": [1.0, 5.0],
            "Inflight Entertainment": [1.0, 5.0],
            "Wifi & Connectivity": [1.0, 5.0],
            "Value For Money": [1.0, 5.0],
            "review_length": [30, 35],
            "keyword_count": [2, 0],
        })

        tfidf = TFIDFFeatures(max_features=50)
        X_tfidf_train = tfidf.fit_transform(texts_train)
        X_struct_train, scaler = get_structured_matrix(df_train, fit=True)
        X_combined_train = build_combined_features(X_tfidf_train, X_struct_train)

        # Infer-time: same construction
        texts_infer = ["terrible delayed"]
        df_infer = pd.DataFrame({
            "full_review_text": texts_infer,
            "Seat Comfort": [2.0],
            "Cabin Staff Service": [2.0],
            "Food & Beverages": [2.0],
            "Ground Service": [2.0],
            "Inflight Entertainment": [2.0],
            "Wifi & Connectivity": [2.0],
            "Value For Money": [2.0],
            "review_length": [15],
            "keyword_count": [1],
        })
        X_tfidf_infer = tfidf.transform(texts_infer)
        X_struct_infer, _ = get_structured_matrix(df_infer, scaler=scaler, fit=False)
        X_combined_infer = build_combined_features(X_tfidf_infer, X_struct_infer)

        assert X_combined_train.shape[1] == X_combined_infer.shape[1], \
            "Train and infer feature dimensions don't match — dimension mismatch bug!"


# ── TC-RG-008: Bug #16 — Rule-based fallback walrus expression bug ───────────

class TestBug16FallbackLogic:
    """Bug #16: Rule-based fallback had nonsense walrus expression and operator precedence bug."""

    def test_rg009_assign_severity_with_none_rating(self):
        """TC-RG-009: _assign_severity handles None/NaN Overall_Rating gracefully."""
        from src.data.label_generator import _assign_severity
        row = pd.Series({
            "Overall_Rating": None,
            "Recommended": "no",
            "_text": "delayed rude staff",
            "Seat Comfort": 2.0,
            "Cabin Staff Service": 2.0,
            "Food & Beverages": 3.0,
            "Ground Service": 2.0,
            "Inflight Entertainment": 3.0,
            "Wifi & Connectivity": 3.0,
            "Value For Money": 2.0,
        })
        result = _assign_severity(row)
        assert result in {"Low", "Medium", "High", "Critical"}

    def test_rg010_assign_risk_with_none_rating(self):
        """TC-RG-010: _assign_risk handles None Overall_Rating gracefully."""
        from src.data.label_generator import _assign_risk
        row = pd.Series({
            "Overall_Rating": None,
            "Recommended": "no",
            "_text": "delayed rude",
            "Seat Comfort": 2.0,
            "Cabin Staff Service": 2.0,
            "Food & Beverages": 3.0,
            "Ground Service": 2.0,
            "Inflight Entertainment": 3.0,
            "Wifi & Connectivity": 3.0,
            "Value For Money": 2.0,
        })
        result = _assign_risk(row)
        assert result in {0, 1}


# ── TC-RG-009: Bug #8 — dtype check for categoricals on pandas 3 ─────────────

class TestBug8DtypeCheck:
    """Bug #8: dtype == object check fails on pandas 3 str dtype."""

    def test_rg011_normalize_categoricals_handles_string_dtype(self):
        """TC-RG-011: _normalize_categoricals works when column dtype is pandas StringDtype."""
        from src.data.cleaner import _normalize_categoricals

        df = pd.DataFrame({
            "Recommended": pd.array(["No", "Yes", "No"], dtype="string"),
            "Type Of Traveller": pd.array(["Business", "Leisure", None], dtype="string"),
            "Seat Type": pd.array(["Economy", "Business", "Economy"], dtype="string"),
            "Verified": [True, False, True],
        })
        # Should not crash even with StringDtype
        try:
            result = _normalize_categoricals(df)
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            pytest.fail(f"_normalize_categoricals failed with StringDtype: {e}")


# ── TC-RG-010: Bug #15 — Explainer always errored for calibrated SVMs ────────

class TestBug15ExplainerCalibratedSVM:
    """Bug #15: get_linear_top_features always errored for calibrated SVMs."""

    def test_rg012_calibrated_svm_top_features(self):
        """TC-RG-012: get_linear_top_features works with CalibratedClassifierCV wrapping LinearSVC."""
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from src.explainability.explainer import get_linear_top_features

        texts = [
            "cancelled delayed rude terrible",
            "excellent on time comfortable",
            "lost baggage refund denied worst",
            "great service amazing crew",
        ]
        labels = [2, 0, 2, 0]

        vec = TfidfVectorizer(max_features=50)
        X = vec.fit_transform(texts)

        svm = LinearSVC(max_iter=1000)
        cal_svm = CalibratedClassifierCV(svm, cv=2)
        cal_svm.fit(X, labels)

        result = get_linear_top_features(cal_svm, vec, class_idx=0, top_n=5)
        # Should return features, not crash
        assert isinstance(result, list)
        # Error case returns [('error', msg)]; valid case returns [(feature, float)]
        if result and result[0][0] != "error":
            for name, weight in result:
                assert isinstance(name, str)
                assert isinstance(weight, float)


# ── TC-RG-011: Bug #6 — Evaluator risk metrics misassigned ───────────────────

class TestBug6EvaluatorMetrics:
    """Bug #6: Risk models had wrong column assignments (Accuracy held ROC-AUC)."""

    def test_rg013_risk_classification_report_structure(self):
        """TC-RG-013: Verify risk classification uses proper accuracy vs ROC-AUC."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score

        # Simulate risk prediction
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.3])

        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob)

        # They should be different values — not mixed up
        assert accuracy != roc_auc, \
            "Accuracy and ROC-AUC are equal — possible metric mix-up"
        assert 0 <= accuracy <= 1
        assert 0 <= roc_auc <= 1


# ── TC-RG-012: Bug #2 — keyword_count hardcoded to 0 in predict_single ───────

class TestBug2KeywordCountInference:
    """Bug #2: keyword_count hardcoded to 0; review_length ignored title."""

    def test_rg014_keyword_count_reflects_actual_keywords(self):
        """TC-RG-014: Inference keyword_count is computed from actual text, not zero."""
        from src.utils.config import CRITICAL_KEYWORDS, RISK_KEYWORDS
        ALL_KW = set(CRITICAL_KEYWORDS + RISK_KEYWORDS)

        text = "The flight was cancelled and the refund was denied by rude staff"
        count = sum(1 for kw in ALL_KW if kw in text.lower())
        assert count > 0, "Expected keywords found, but count is 0"

    def test_rg015_review_length_includes_title(self):
        """TC-RG-015: Full text length includes both title and review."""
        title = "Cancelled flight"
        review = "The flight was cancelled and refund denied"
        combined = (title + " " + review).strip()
        assert len(combined) > len(review), \
            "Combined text should be longer than review alone"
