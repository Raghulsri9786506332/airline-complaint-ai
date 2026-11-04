"""
TC_001_smoke_tests.py — Smoke Tests
====================================
Category  : Smoke
Purpose   : Verify the most critical imports, modules, and entry points load
            without errors. If ANY of these fail the application is broken.
Run with  : pytest "test cases/TC_001_smoke_tests.py" -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── TC-SM-001: Core module imports ───────────────────────────────────────────

class TestSmokeImports:
    """Verify all critical src modules can be imported."""

    def test_sm001_config_imports(self):
        """TC-SM-001: src.utils.config imports without error."""
        from src.utils.config import (
            CRITICAL_KEYWORDS, RISK_KEYWORDS, RATING_COLS,
            SEVERITY_LABELS, SEVERITY_ID_MAP, RISK_LABELS,
        )
        assert isinstance(CRITICAL_KEYWORDS, list) and len(CRITICAL_KEYWORDS) > 0
        assert isinstance(RISK_KEYWORDS, list) and len(RISK_KEYWORDS) > 0
        assert isinstance(RATING_COLS, list) and len(RATING_COLS) == 7

    def test_sm002_logger_imports(self):
        """TC-SM-002: src.utils.logger imports and creates a logger."""
        from src.utils.logger import get_logger
        from src.utils.config import LOGS_DIR
        log = get_logger("smoke_test", LOGS_DIR)
        assert log is not None

    def test_sm003_data_loader_imports(self):
        """TC-SM-003: src.data.data_loader imports without error."""
        from src.data.data_loader import load_raw_dataset
        assert callable(load_raw_dataset)

    def test_sm004_cleaner_imports(self):
        """TC-SM-004: src.data.cleaner imports all public functions."""
        from src.data.cleaner import (
            clean_dataset,
            _convert_overall_rating,
            _convert_aspect_ratings,
            _normalize_categoricals,
            _drop_exact_duplicates,
        )
        assert callable(clean_dataset)

    def test_sm005_label_generator_imports(self):
        """TC-SM-005: src.data.label_generator imports all public functions."""
        from src.data.label_generator import (
            generate_labels,
            _has_critical_keyword,
            _has_risk_keyword,
            _assign_severity,
            _assign_risk,
        )
        assert callable(generate_labels)

    def test_sm006_text_preprocessor_imports(self):
        """TC-SM-006: src.preprocessing.text_preprocessor imports."""
        from src.preprocessing.text_preprocessor import (
            ml_clean, dl_clean, transformer_clean, build_full_text,
        )
        assert callable(ml_clean)
        assert callable(dl_clean)
        assert callable(transformer_clean)
        assert callable(build_full_text)

    def test_sm007_feature_engineer_imports(self):
        """TC-SM-007: src.features.feature_engineer imports."""
        from src.features.feature_engineer import (
            BagOfWordsFeatures, TFIDFFeatures,
            add_structured_features, get_structured_matrix,
            build_combined_features,
        )
        assert BagOfWordsFeatures is not None

    def test_sm008_explainer_imports(self):
        """TC-SM-008: src.explainability.explainer imports."""
        from src.explainability.explainer import (
            highlight_complaint_phrases,
            get_linear_top_features,
            explain_prediction,
        )
        assert callable(highlight_complaint_phrases)
        assert callable(explain_prediction)


# ── TC-SM-009: Config path definitions ───────────────────────────────────────

class TestSmokeConfigPaths:
    """Verify all paths in config are properly typed."""

    def test_sm009_paths_are_path_objects(self):
        """TC-SM-009: All directory paths are pathlib.Path objects."""
        from src.utils.config import (
            BASE_DIR, DATA_RAW, DATA_PROCESSED,
            MODELS_ML, MODELS_DL, MODELS_TRANSFORMER,
            ARTIFACTS_VEC, ARTIFACTS_ENC, LOGS_DIR,
        )
        from pathlib import Path
        for p in [BASE_DIR, DATA_RAW, DATA_PROCESSED,
                  MODELS_ML, MODELS_DL, MODELS_TRANSFORMER,
                  ARTIFACTS_VEC, ARTIFACTS_ENC, LOGS_DIR]:
            assert isinstance(p, Path), f"{p} is not a Path"

    def test_sm010_base_dir_is_project_root(self):
        """TC-SM-010: BASE_DIR resolves to the project root containing app.py."""
        from src.utils.config import BASE_DIR
        assert (BASE_DIR / "app.py").exists(), \
            f"app.py not found in BASE_DIR={BASE_DIR}"


# ── TC-SM-011: Keyword lists are non-empty and have no duplicates ─────────────

class TestSmokeKeywordLists:
    """Verify keyword configuration."""

    def test_sm011_critical_keywords_no_duplicates(self):
        """TC-SM-011: CRITICAL_KEYWORDS has no duplicate entries."""
        from src.utils.config import CRITICAL_KEYWORDS
        assert len(CRITICAL_KEYWORDS) == len(set(CRITICAL_KEYWORDS))

    def test_sm012_risk_keywords_no_duplicates(self):
        """TC-SM-012: RISK_KEYWORDS has no duplicate entries."""
        from src.utils.config import RISK_KEYWORDS
        assert len(RISK_KEYWORDS) == len(set(RISK_KEYWORDS))

    def test_sm013_severity_labels_correct_order(self):
        """TC-SM-013: SEVERITY_LABELS are in ascending severity order."""
        from src.utils.config import SEVERITY_LABELS
        assert SEVERITY_LABELS == ["Low", "Medium", "High", "Critical"]

    def test_sm014_severity_id_map_bidirectional(self):
        """TC-SM-014: SEVERITY_ID_MAP and SEVERITY_ID_TO_LABEL are inverses."""
        from src.utils.config import SEVERITY_ID_MAP, SEVERITY_ID_TO_LABEL
        for label, idx in SEVERITY_ID_MAP.items():
            assert SEVERITY_ID_TO_LABEL[idx] == label


# ── TC-SM-015: Basic end-to-end pipeline smoke ────────────────────────────────

class TestSmokeEndToEnd:
    """Minimal end-to-end: build full text → ML clean → keyword scan."""

    def test_sm015_full_text_to_ml_clean(self):
        """TC-SM-015: build_full_text with ml pipeline returns a non-empty string."""
        from src.preprocessing.text_preprocessor import build_full_text
        result = build_full_text("Terrible flight", "Lost all my baggage.", "ml")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sm016_keyword_detection_in_clean_text(self):
        """TC-SM-016: highlight_complaint_phrases finds keywords in raw text."""
        from src.explainability.explainer import highlight_complaint_phrases
        result = highlight_complaint_phrases("The flight was cancelled and baggage was lost")
        assert len(result["complaint_phrases"]) > 0

    def test_sm017_explain_prediction_returns_dict(self):
        """TC-SM-017: explain_prediction returns a valid dict structure."""
        from src.explainability.explainer import explain_prediction
        out = explain_prediction("Terrible service", "Critical", 1, 0.91, 0.85)
        required_keys = {
            "severity_label", "severity_confidence", "risk_flag",
            "risk_probability", "complaint_phrases_found", "disclaimer",
        }
        assert required_keys.issubset(out.keys())
