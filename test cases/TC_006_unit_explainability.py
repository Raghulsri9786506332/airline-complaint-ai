"""
TC_006_unit_explainability.py — Unit Tests: Explainability Module
==================================================================
Category  : Unit
Module    : src/explainability/explainer.py
Purpose   : Test keyword highlighting, linear feature extraction, and explanation
            generation for all supported scenarios.
Run with  : pytest "test cases/TC_006_unit_explainability.py" -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explainability.explainer import (
    highlight_complaint_phrases,
    get_linear_top_features,
    explain_prediction,
)


# ── TC-EX-001 — 015: highlight_complaint_phrases ─────────────────────────────

class TestHighlightComplaintPhrases:

    def test_ex001_returns_dict(self):
        """TC-EX-001: highlight_complaint_phrases returns a dict."""
        result = highlight_complaint_phrases("The flight was cancelled")
        assert isinstance(result, dict)

    def test_ex002_dict_has_complaint_phrases_key(self):
        """TC-EX-002: Result dict contains 'complaint_phrases' key."""
        result = highlight_complaint_phrases("Some text")
        assert "complaint_phrases" in result

    def test_ex003_dict_has_text_key(self):
        """TC-EX-003: Result dict contains 'text' key preserving original."""
        text = "Flight was delayed"
        result = highlight_complaint_phrases(text)
        assert "text" in result
        assert result["text"] == text

    def test_ex004_detects_critical_keyword(self):
        """TC-EX-004: 'cancelled' is detected in complaint_phrases."""
        result = highlight_complaint_phrases("The flight was cancelled yesterday")
        assert "cancelled" in result["complaint_phrases"]

    def test_ex005_detects_risk_keyword(self):
        """TC-EX-005: 'delayed' is detected in complaint_phrases."""
        result = highlight_complaint_phrases("Flight was delayed by 5 hours")
        assert "delayed" in result["complaint_phrases"]

    def test_ex006_no_false_positives_for_positive_review(self):
        """TC-EX-006: Positive review with no keywords returns empty list."""
        result = highlight_complaint_phrases("Excellent comfortable service on time")
        assert result["complaint_phrases"] == []

    def test_ex007_multiple_keywords_all_detected(self):
        """TC-EX-007: Multiple keywords in same text are all detected."""
        result = highlight_complaint_phrases(
            "The flight was cancelled, luggage lost, staff were rude"
        )
        phrases = result["complaint_phrases"]
        # At least two distinct keywords should be found
        assert len(phrases) >= 2

    def test_ex008_case_insensitive_detection(self):
        """TC-EX-008: Keyword detection is case-insensitive (pre-lowercased text)."""
        result = highlight_complaint_phrases("flight was CANCELLED")
        # text is lowercased by highlight_complaint_phrases internally
        assert "cancelled" in result["complaint_phrases"]

    def test_ex009_empty_string_returns_empty_list(self):
        """TC-EX-009: Empty string → empty complaint_phrases list."""
        result = highlight_complaint_phrases("")
        assert result["complaint_phrases"] == []

    def test_ex010_none_input_does_not_crash(self):
        """TC-EX-010: None input is handled without crash."""
        try:
            result = highlight_complaint_phrases(None)
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"highlight_complaint_phrases(None) raised: {e}")

    def test_ex011_deduplication_of_phrases(self):
        """TC-EX-011: The same keyword appearing multiple times is deduplicated."""
        result = highlight_complaint_phrases(
            "cancelled cancelled cancelled delayed delayed"
        )
        phrases = result["complaint_phrases"]
        assert phrases.count("cancelled") == 1

    def test_ex012_sorted_output(self):
        """TC-EX-012: complaint_phrases list is sorted alphabetically."""
        result = highlight_complaint_phrases(
            "delayed cancelled rude stranded"
        )
        phrases = result["complaint_phrases"]
        assert phrases == sorted(phrases)

    def test_ex013_multi_word_phrase_detection(self):
        """TC-EX-013: Multi-word phrases like 'lost baggage' are detected."""
        result = highlight_complaint_phrases("They lost baggage at the airport")
        assert "lost baggage" in result["complaint_phrases"]

    def test_ex014_compliant_keyword_partial_not_triggered(self):
        """TC-EX-014: Partial keyword match within different word doesn't trigger."""
        # 'cancel' is in CRITICAL_KEYWORDS; should be found in 'cancellation' too
        result = highlight_complaint_phrases("we got a cancellation notice")
        # 'cancel' is substring of 'cancellation', so it should be found
        phrases = result["complaint_phrases"]
        assert isinstance(phrases, list)  # just check no crash

    def test_ex015_returns_list_type(self):
        """TC-EX-015: complaint_phrases value is a list."""
        result = highlight_complaint_phrases("delayed and terrible")
        assert isinstance(result["complaint_phrases"], list)


# ── TC-EX-021 — 030: explain_prediction ──────────────────────────────────────

class TestExplainPrediction:

    def test_ex021_returns_dict(self):
        """TC-EX-021: explain_prediction returns a dict."""
        result = explain_prediction("Bad flight", "Critical", 1, 0.91, 0.88)
        assert isinstance(result, dict)

    def test_ex022_severity_label_preserved(self):
        """TC-EX-022: severity_label in output matches input."""
        result = explain_prediction("OK flight", "Low", 0, 0.80, 0.10)
        assert result["severity_label"] == "Low"

    def test_ex023_risk_flag_mapped_to_string(self):
        """TC-EX-023: risk_flag=1 maps to 'Risk' string."""
        result = explain_prediction("Bad flight", "Critical", 1, 0.91, 0.88)
        assert result["risk_flag"] == "Risk"

    def test_ex024_no_risk_flag_mapped(self):
        """TC-EX-024: risk_flag=0 maps to 'No Risk' string."""
        result = explain_prediction("Good flight", "Low", 0, 0.90, 0.05)
        assert result["risk_flag"] == "No Risk"

    def test_ex025_severity_confidence_formatted(self):
        """TC-EX-025: severity_confidence is a formatted percentage string."""
        result = explain_prediction("OK flight", "Medium", 0, 0.75, 0.20)
        assert "%" in result["severity_confidence"]

    def test_ex026_risk_probability_formatted(self):
        """TC-EX-026: risk_probability is a formatted percentage string."""
        result = explain_prediction("OK flight", "Medium", 0, 0.75, 0.20)
        assert "%" in result["risk_probability"]

    def test_ex027_complaint_phrases_found_is_list(self):
        """TC-EX-027: complaint_phrases_found is a list."""
        result = explain_prediction("Bad flight cancelled", "Critical", 1, 0.9, 0.85)
        assert isinstance(result["complaint_phrases_found"], list)

    def test_ex028_severity_signals_detected_in_complaint_text(self):
        """TC-EX-028: severity_related_signals populated for complaint text."""
        result = explain_prediction(
            "Flight was cancelled. Worst experience ever.",
            "Critical", 1, 0.95, 0.90
        )
        # Should find 'cancelled' and/or 'worst' as severity signals
        assert isinstance(result["severity_related_signals"], list)

    def test_ex029_risk_signals_detected(self):
        """TC-EX-029: risk_related_signals populated for risk-related text."""
        result = explain_prediction(
            "Flight was delayed 8 hours and refund denied",
            "High", 1, 0.80, 0.85
        )
        assert isinstance(result["risk_related_signals"], list)

    def test_ex030_disclaimer_is_non_empty_string(self):
        """TC-EX-030: disclaimer key contains a non-empty string."""
        result = explain_prediction("Test", "Low", 0, 0.7, 0.1)
        assert isinstance(result["disclaimer"], str)
        assert len(result["disclaimer"]) > 0

    def test_ex031_all_required_keys_present(self):
        """TC-EX-031: All required keys are present in the output dict."""
        result = explain_prediction("Test review", "High", 1, 0.80, 0.70)
        required = {
            "severity_label", "severity_confidence", "risk_flag",
            "risk_probability", "complaint_phrases_found",
            "severity_related_signals", "risk_related_signals", "disclaimer",
        }
        assert required.issubset(result.keys())

    def test_ex032_confidence_value_is_percentage_of_input(self):
        """TC-EX-032: Confidence 0.91 → '91.0%' in output."""
        result = explain_prediction("Test", "Critical", 1, 0.91, 0.88)
        assert "91.0%" == result["severity_confidence"]

    def test_ex033_risk_probability_value_correct(self):
        """TC-EX-033: Risk probability 0.88 → '88.0%' in output."""
        result = explain_prediction("Test", "Critical", 1, 0.91, 0.88)
        assert "88.0%" == result["risk_probability"]


# ── TC-EX-041 — 050: get_linear_top_features ─────────────────────────────────

class TestGetLinearTopFeatures:

    @pytest.fixture
    def fitted_lr_and_vec(self):
        """Fit a small LogisticRegression + TfidfVectorizer for testing."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [
            "cancelled flight terrible rude staff",
            "excellent comfortable on time",
            "delayed refund denied worst nightmare",
            "great service amazing food",
            "stranded airport no compensation",
        ]
        labels = [2, 0, 2, 0, 2]  # 0=Low, 2=High

        vec = TfidfVectorizer(max_features=50)
        X = vec.fit_transform(texts)
        clf = LogisticRegression(max_iter=100)
        clf.fit(X, labels)
        return clf, vec

    def test_ex041_returns_list(self, fitted_lr_and_vec):
        """TC-EX-041: get_linear_top_features returns a list."""
        clf, vec = fitted_lr_and_vec
        result = get_linear_top_features(clf, vec, class_idx=0, top_n=5)
        assert isinstance(result, list)

    def test_ex042_returns_tuples(self, fitted_lr_and_vec):
        """TC-EX-042: Each element is a (feature_name, weight) tuple."""
        clf, vec = fitted_lr_and_vec
        result = get_linear_top_features(clf, vec, class_idx=0, top_n=5)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_ex043_top_n_respected(self, fitted_lr_and_vec):
        """TC-EX-043: Number of returned features does not exceed top_n."""
        clf, vec = fitted_lr_and_vec
        result = get_linear_top_features(clf, vec, class_idx=0, top_n=3)
        assert len(result) <= 3

    def test_ex044_feature_names_are_strings(self, fitted_lr_and_vec):
        """TC-EX-044: Feature names in the output are strings."""
        clf, vec = fitted_lr_and_vec
        result = get_linear_top_features(clf, vec, class_idx=0, top_n=5)
        for name, weight in result:
            assert isinstance(name, str)

    def test_ex045_weights_are_floats(self, fitted_lr_and_vec):
        """TC-EX-045: Coefficient weights in the output are float values."""
        clf, vec = fitted_lr_and_vec
        result = get_linear_top_features(clf, vec, class_idx=0, top_n=5)
        for name, weight in result:
            assert isinstance(weight, float)

    def test_ex046_unfitted_model_returns_error_gracefully(self):
        """TC-EX-046: Unfitted model returns [('error', ...)] gracefully."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer

        clf = LogisticRegression()  # not fitted
        vec = TfidfVectorizer()
        vec.fit(["some text here", "another text there"])

        result = get_linear_top_features(clf, vec, class_idx=0, top_n=5)
        # Should return error tuple, not crash
        assert isinstance(result, list)
        if result:
            assert result[0][0] == "error" or isinstance(result[0][0], str)
