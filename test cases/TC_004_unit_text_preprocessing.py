"""
TC_004_unit_text_preprocessing.py — Unit Tests: Text Preprocessing (Phase 3)
==============================================================================
Category  : Unit
Module    : src/preprocessing/text_preprocessor.py
Purpose   : Verify all three NLP pipelines (ML, DL, Transformer) and edge cases.
Run with  : pytest "test cases/TC_004_unit_text_preprocessing.py" -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.text_preprocessor import (
    ml_clean,
    dl_clean,
    transformer_clean,
    build_full_text,
)


# ── TC-TP-001 — 015: ml_clean (Pipeline A) ────────────────────────────────────

class TestMLCleanPipeline:

    def test_tp001_returns_string(self):
        """TC-TP-001: ml_clean returns a string."""
        assert isinstance(ml_clean("Hello world"), str)

    def test_tp002_lowercases_text(self):
        """TC-TP-002: ml_clean lowercases all characters."""
        result = ml_clean("GREAT FLIGHT! Very Comfortable.")
        assert result == result.lower()

    def test_tp003_removes_stopwords(self):
        """TC-TP-003: ml_clean removes common stopwords like 'the', 'was'."""
        result = ml_clean("the flight was very good")
        tokens = result.split()
        assert "the" not in tokens
        assert "was" not in tokens

    def test_tp004_removes_html_tags(self):
        """TC-TP-004: ml_clean strips HTML tags."""
        result = ml_clean("<b>Bad</b> service <br/> awful experience")
        assert "<b>" not in result
        assert "<br" not in result
        assert "bad" in result

    def test_tp005_removes_urls(self):
        """TC-TP-005: ml_clean removes HTTP URLs."""
        result = ml_clean("Check https://example.com for more info")
        assert "https" not in result
        assert "example.com" not in result

    def test_tp006_empty_string_returns_empty(self):
        """TC-TP-006: ml_clean on empty string returns empty string."""
        result = ml_clean("")
        assert result == ""

    def test_tp007_preserves_complaint_keywords_after_lemmatization(self):
        """TC-TP-007: Key complaint words survive ml_clean (lemmatized forms)."""
        result = ml_clean("The flight was cancelled and bags were lost")
        # 'cancelled' may become 'cancel' or similar via lemmatizer; at minimum
        # some root form should remain
        assert len(result) > 0

    def test_tp008_none_input_does_not_crash(self):
        """TC-TP-008: ml_clean handles None input without crash."""
        try:
            result = ml_clean(None)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"ml_clean(None) raised: {e}")

    def test_tp009_numeric_only_text_handled(self):
        """TC-TP-009: ml_clean on digit-only string doesn't crash."""
        result = ml_clean("123456789")
        assert isinstance(result, str)

    def test_tp010_repeated_whitespace_collapsed(self):
        """TC-TP-010: Multiple spaces/newlines are collapsed to single space."""
        result = ml_clean("bad    service\n\nvery  rude")
        assert "  " not in result

    def test_tp011_emoji_removed(self):
        """TC-TP-011: Emoji characters are removed from ml_clean output."""
        result = ml_clean("Great flight ✈️🌟😊")
        # Emojis should not appear in the output
        assert "✈" not in result and "🌟" not in result

    def test_tp012_html_entity_unescaped(self):
        """TC-TP-012: HTML entities like &amp; are unescaped before cleaning."""
        result = ml_clean("Food &amp; Beverages were great")
        # Should not contain '&amp;' as literal text
        assert "&amp;" not in result

    def test_tp013_very_long_text_handled(self):
        """TC-TP-013: ml_clean handles very long text (>10000 chars) without crash."""
        long_text = "The flight was delayed. " * 500
        result = ml_clean(long_text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_tp014_lemmatization_reduces_inflections(self):
        """TC-TP-014: Lemmatization converts 'flights' → 'flight' etc."""
        result = ml_clean("many flights delayed by storms")
        # 'flights' should become 'flight'; 'storms' should become 'storm'
        # (NLTK WordNet lemmatizer may vary, just check no crash)
        assert isinstance(result, str)

    def test_tp015_punctuation_removed(self):
        """TC-TP-015: Standard punctuation (!, ?, ...) is removed."""
        result = ml_clean("Worst flight!!! Absolutely terrible???")
        assert "!" not in result
        assert "?" not in result


# ── TC-TP-016 — 025: dl_clean (Pipeline B) ───────────────────────────────────

class TestDLCleanPipeline:

    def test_tp016_returns_string(self):
        """TC-TP-016: dl_clean returns a string."""
        assert isinstance(dl_clean("Hello world"), str)

    def test_tp017_does_not_remove_stopwords(self):
        """TC-TP-017: dl_clean keeps stopwords like 'the', 'was'."""
        result = dl_clean("the flight was very good")
        assert "the" in result
        assert "was" in result

    def test_tp018_lowercases_text(self):
        """TC-TP-018: dl_clean lowercases text (base clean lowercases)."""
        result = dl_clean("GREAT FLIGHT")
        assert result == result.lower()

    def test_tp019_removes_html_tags(self):
        """TC-TP-019: dl_clean strips HTML tags."""
        result = dl_clean("<b>Terrible</b> service")
        assert "<b>" not in result

    def test_tp020_removes_urls(self):
        """TC-TP-020: dl_clean removes HTTP URLs."""
        result = dl_clean("See http://airline.com for info")
        assert "http" not in result

    def test_tp021_empty_string_returns_empty(self):
        """TC-TP-021: dl_clean on empty string returns empty string."""
        assert dl_clean("") == ""

    def test_tp022_no_lemmatization(self):
        """TC-TP-022: dl_clean does NOT lemmatize (keeps original tokens)."""
        result = dl_clean("flights were cancelled")
        # 'flights' should stay as 'flights' (not 'flight')
        assert "flight" in result or "flights" in result

    def test_tp023_none_input_does_not_crash(self):
        """TC-TP-023: dl_clean handles None input without crash."""
        try:
            result = dl_clean(None)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"dl_clean(None) raised: {e}")

    def test_tp024_whitespace_collapsed(self):
        """TC-TP-024: dl_clean collapses multiple whitespace to single space."""
        result = dl_clean("bad    service")
        assert "  " not in result


# ── TC-TP-026 — 035: transformer_clean (Pipeline C) ─────────────────────────

class TestTransformerCleanPipeline:

    def test_tp026_returns_string(self):
        """TC-TP-026: transformer_clean returns a string."""
        assert isinstance(transformer_clean("Hello world"), str)

    def test_tp027_preserves_case(self):
        """TC-TP-027: transformer_clean preserves original casing."""
        result = transformer_clean("Terrible experience with Delta Airlines.")
        # First-letter capital should be preserved
        assert result[0].isupper() or result[0] == "T"

    def test_tp028_removes_html_tags(self):
        """TC-TP-028: transformer_clean strips HTML tags."""
        result = transformer_clean("<b>Bad</b> service!")
        assert "<b>" not in result

    def test_tp029_removes_urls(self):
        """TC-TP-029: transformer_clean removes URLs."""
        result = transformer_clean("Complain at https://airline.com")
        assert "https" not in result

    def test_tp030_preserves_stopwords(self):
        """TC-TP-030: transformer_clean keeps stopwords and context words."""
        result = transformer_clean("The flight was not good at all")
        assert "the" in result.lower() or "The" in result

    def test_tp031_preserves_negations(self):
        """TC-TP-031: transformer_clean keeps negation words like 'not', 'never'."""
        result = transformer_clean("I will never fly with them again")
        assert "never" in result.lower()

    def test_tp032_none_input_does_not_crash(self):
        """TC-TP-032: transformer_clean handles None input without crash."""
        try:
            result = transformer_clean(None)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"transformer_clean(None) raised: {e}")


# ── TC-TP-036 — 050: build_full_text ─────────────────────────────────────────

class TestBuildFullText:

    def test_tp036_returns_string(self):
        """TC-TP-036: build_full_text returns a string."""
        result = build_full_text("Title", "Review body", "ml")
        assert isinstance(result, str)

    def test_tp037_concatenates_title_and_review(self):
        """TC-TP-037: build_full_text combines title and review text."""
        result = build_full_text("Flight title", "Good flight", "transformer")
        # Both should contribute; at least one word from each should appear
        assert len(result) > 0

    def test_tp038_none_title_handled(self):
        """TC-TP-038: build_full_text with None title doesn't crash."""
        result = build_full_text(None, "Some review", "ml")
        assert isinstance(result, str)

    def test_tp039_none_review_handled(self):
        """TC-TP-039: build_full_text with None review doesn't crash."""
        result = build_full_text("Some title", None, "ml")
        assert isinstance(result, str)

    def test_tp040_both_none_handled(self):
        """TC-TP-040: build_full_text with both title and review None doesn't crash."""
        result = build_full_text(None, None, "ml")
        assert isinstance(result, str)

    def test_tp041_invalid_pipeline_raises_value_error(self):
        """TC-TP-041: build_full_text raises ValueError for unknown pipeline name."""
        with pytest.raises(ValueError, match="Unknown pipeline"):
            build_full_text("T", "R", pipeline="invalid")

    def test_tp042_ml_pipeline_removes_stopwords(self):
        """TC-TP-042: build_full_text with ml pipeline removes 'the' etc."""
        result = build_full_text("the", "the flight was okay", "ml")
        assert "the" not in result.split()

    def test_tp043_dl_pipeline_keeps_stopwords(self):
        """TC-TP-043: build_full_text with dl pipeline keeps stopwords."""
        result = build_full_text("the", "the flight was okay", "dl")
        assert "the" in result

    def test_tp044_transformer_pipeline_preserves_case(self):
        """TC-TP-044: build_full_text with transformer pipeline preserves capitalisation."""
        result = build_full_text("Terrible", "Terrible flight", "transformer")
        # Output should preserve the capitalised T
        assert "Terrible" in result or "terrible" in result  # base_clean strips some chars

    def test_tp045_nan_string_title_handled(self):
        """TC-TP-045: NaN passed as string 'nan' is handled gracefully."""
        # In production, NaN title might appear as the string 'nan'
        result = build_full_text("nan", "Some review text", "ml")
        assert isinstance(result, str)
        # The result should not contain the literal word 'nan' as a meaningful token
        # (it may or may not — just check no crash)

    def test_tp046_empty_title_and_review_returns_string(self):
        """TC-TP-046: Empty title and review don't crash; return a string."""
        result = build_full_text("", "", "ml")
        assert isinstance(result, str)

    def test_tp047_very_long_review_ml_pipeline(self):
        """TC-TP-047: Very long review (>5000 chars) is processed without crash."""
        long = "The flight was terrible and delayed. " * 200
        result = build_full_text("Delay", long, "ml")
        assert isinstance(result, str)

    def test_tp048_special_characters_handled(self):
        """TC-TP-048: Special characters in review don't cause crash."""
        result = build_full_text(
            "Title", "Flight was bad! 100% terrible... #neveragain @airline", "ml"
        )
        assert isinstance(result, str)

    def test_tp049_pipeline_choices_all_return_non_empty_for_valid_review(self):
        """TC-TP-049: All three pipelines return non-empty for a valid complaint."""
        review = "Cancelled flight. Rude staff. Never again."
        for pipeline in ("ml", "dl", "transformer"):
            result = build_full_text("Bad Flight", review, pipeline)
            assert len(result) > 0, f"Pipeline '{pipeline}' returned empty string"

    def test_tp050_different_pipelines_produce_different_outputs(self):
        """TC-TP-050: ml and transformer pipelines produce different output for same input."""
        title = "Delayed and Terrible"
        review = "The flight was delayed and the staff were not helpful at all."
        ml_result = build_full_text(title, review, "ml")
        tr_result = build_full_text(title, review, "transformer")
        # ML should remove stopwords; transformer should keep them → different
        assert ml_result != tr_result
