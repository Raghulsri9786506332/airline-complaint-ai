"""
tests/test_fixes.py — Regression tests for bugs fixed during code review.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cleaner import clean_dataset
from src.data.label_generator import generate_labels
from src.evaluation.evaluator import build_comparison_table
from src.explainability.explainer import highlight_complaint_phrases
from src.features.feature_engineer import (
    STRUCTURED_COLS, build_inference_struct, count_keywords,
)
from src.preprocessing.text_preprocessor import build_full_text, ml_clean
from src.utils.config import RATING_COLS


# ── Text preprocessing ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_negations_are_kept(self):
        out = ml_clean("The staff did not help and the flight was never on time")
        assert "not" in out.split()
        assert "never" in out.split()

    def test_contractions_become_negations(self):
        out = ml_clean("They didn't refund me and I can't complain more")
        assert out.split().count("not") >= 2

    def test_nan_title_does_not_become_word_nan(self):
        assert "nan" not in build_full_text(float("nan"), "Terrible service", "ml").split()
        assert build_full_text(None, None, "ml") == ""


# ── Labels ───────────────────────────────────────────────────────────────────

class TestLabels:
    def _df(self):
        return pd.DataFrame({
            "Overall_Rating": ["1", "9", "n"],          # strings, one invalid
            "Review_Title": ["Awful", None, "ok"],
            "Review": ["Cancelled, lost baggage", "Lovely flight", "meh"],
            "Recommended": ["no", "yes", "yes"],
        })

    def test_generate_labels_accepts_string_ratings(self):
        out = generate_labels(self._df())
        assert out.loc[0, "severity_label"] == "Critical"
        assert out.loc[1, "severity_label"] == "Low"
        assert out.loc[2, "severity_label"] in {"Medium", "High"}  # NaN rating

    def test_generate_labels_without_rating_column(self):
        df = self._df().drop(columns=["Overall_Rating"])
        out = generate_labels(df)
        assert len(out) == 3 and "severity_label" in out


# ── Cleaning ─────────────────────────────────────────────────────────────────

def test_clean_dataset_lowercases_categoricals_and_keeps_text():
    df = pd.DataFrame({
        "Airline Name": ["A", "B"],
        "Overall_Rating": ["3", "x"],
        "Review_Title": [None, "t"],
        "Review Date": ["1st May 2020", "2nd May 2020"],
        "Review": ["r1", "r2"],
        "Recommended": [" YES ", "No"],
        **{c: [1.0, np.nan] for c in RATING_COLS},
    })
    out = clean_dataset(df)
    assert out["Recommended"].tolist() == ["yes", "no"]
    assert out["Review_Title"].tolist() == ["", "t"]
    assert out[RATING_COLS].isna().sum().sum() == 0


# ── Features ─────────────────────────────────────────────────────────────────

class _FakeScaler:
    """Mimics a fitted StandardScaler (mean=3, scale=1) for all structured cols."""
    mean_ = np.full(len(STRUCTURED_COLS), 3.0)

    def transform(self, X):
        return (X - self.mean_) / 1.0


class TestInferenceFeatures:
    def test_shape_matches_training_columns(self):
        X = build_inference_struct("t", "some review", {}, _FakeScaler())
        assert X.shape == (1, len(STRUCTURED_COLS))

    def test_missing_ratings_are_neutral_not_zero(self):
        X = build_inference_struct("t", "r", {"Seat Comfort": None, "Value For Money": 0}, _FakeScaler())
        i = STRUCTURED_COLS.index("Seat Comfort")
        j = STRUCTURED_COLS.index("Value For Money")
        assert X[0, i] == 0.0 and X[0, j] == 0.0  # mean-imputed -> 0 after scaling

    def test_keyword_count_is_computed(self):
        X = build_inference_struct("", "cancelled and delayed, rude staff", {}, _FakeScaler())
        k = STRUCTURED_COLS.index("keyword_count")
        assert X[0, k] + 3.0 == count_keywords("cancelled and delayed, rude staff") >= 3

    def test_saved_scaler_is_compatible(self):
        path = ROOT / "artifacts" / "vectorizers" / "struct_scaler.pkl"
        if not path.exists():
            pytest.skip("trained scaler not present")
        import joblib
        scaler = joblib.load(path)
        X = build_inference_struct("t", "r", {"Seat Comfort": 4}, scaler)
        assert X.shape[1] == scaler.n_features_in_


# ── Explainer ────────────────────────────────────────────────────────────────

def test_highlight_drops_overlapping_substrings():
    found = highlight_complaint_phrases("Flight cancelled and delayed")["complaint_phrases"]
    assert "cancelled" in found and "cancel" not in found
    assert "delayed" in found and "delay" not in found


# ── Evaluator ────────────────────────────────────────────────────────────────

def test_comparison_table_does_not_mix_roc_auc_into_accuracy():
    table = build_comparison_table([
        {"model": "m", "task": "risk", "feature": "f", "roc_auc": 0.97,
         "f1": 0.9, "recall": 0.9, "accuracy": 0.95, "macro_f1": 0.91},
        {"model": "n", "task": "risk", "feature": "f", "roc_auc": 0.80, "f1": 0.5, "recall": 0.4},
    ])
    assert table.loc[0, "Accuracy"] == 0.95
    assert np.isnan(table.loc[1, "Accuracy"])  # unknown -> NaN, NOT the ROC-AUC value


# ── Deep learning (only when PyTorch is installed) ───────────────────────────

class TestDeepLearning:
    def test_lstm_and_cnn_handle_short_and_padded_batches(self):
        pytest.importorskip("torch")
        from src.models.dl_trainer import (
            ReviewDataset, SimpleTokenizer, TextCNN, TextLSTM, _collate_fn,
        )
        tok = SimpleTokenizer()
        tok.build_vocab(["bad flight", "great seat comfort and food"])
        texts = ["bad", "great seat comfort and food bad flight", ""]
        ds = ReviewDataset([tok.encode(t) for t in texts], [0, 1, 2])
        batch = _collate_fn([ds[i] for i in range(3)])
        V = len(tok.word2idx)
        for model in (TextCNN(V, 8, 4), TextLSTM(V, 8, 4), TextLSTM(V, 8, 4, bidirectional=True)):
            out = model(batch["input_ids"])
            assert out.shape == (3, 4)
