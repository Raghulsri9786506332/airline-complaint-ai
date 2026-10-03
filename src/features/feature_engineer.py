"""
feature_engineer.py — Phase 6: Feature engineering for ML models.

Three feature representations:
  1. Bag-of-Words (CountVectorizer)
  2. TF-IDF (TfidfVectorizer, unigrams + bigrams)
  3. TF-IDF + structured aspect ratings

Always fit on training data only; transform train + test separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.utils.config import (
    BOW_MAX_FEATURES, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, RATING_COLS,
    ARTIFACTS_VEC,
)
from src.utils.config import LOGS_DIR
from src.utils.logger import get_logger

log = get_logger("feature_engineer", LOGS_DIR)

STRUCTURED_COLS = RATING_COLS + ["review_length", "keyword_count"]


# ── Structured feature helpers ───────────────────────────────────────────────

def add_structured_features(df: pd.DataFrame, raw_text_col: str = "full_review_text") -> pd.DataFrame:
    """Add review_length and keyword_count columns to df in-place."""
    from src.utils.config import CRITICAL_KEYWORDS, RISK_KEYWORDS
    ALL_KW = set(CRITICAL_KEYWORDS + RISK_KEYWORDS)

    df = df.copy()
    df["review_length"] = df[raw_text_col].fillna("").str.len()
    df["keyword_count"] = df[raw_text_col].fillna("").str.lower().apply(
        lambda t: sum(1 for kw in ALL_KW if kw in t)
    )
    return df


def count_keywords(text: str) -> int:
    """Number of distinct complaint/risk keywords in text (same rule as training)."""
    from src.utils.config import CRITICAL_KEYWORDS, RISK_KEYWORDS
    t = (text or "").lower()
    return sum(1 for kw in set(CRITICAL_KEYWORDS + RISK_KEYWORDS) if kw in t)


def build_inference_struct(
    title: str,
    review: str,
    ratings: dict | None,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Build ONE scaled structured-feature row for inference, in exactly the same
    column order / definition as training (STRUCTURED_COLS).

    * review_length / keyword_count are computed on "title + review" like training.
    * Aspect ratings that the user did not provide (None / 0) are filled with the
      training mean (scaler.mean_), i.e. they become a neutral 0 after scaling —
      instead of a fake rating of 0 which never occurs in the real data (1-5).
    """
    ratings = ratings or {}
    full_text = f"{title or ''} {review or ''}"
    row = []
    for i, col in enumerate(STRUCTURED_COLS):
        if col in RATING_COLS:
            v = ratings.get(col)
            row.append(float(v) if v not in (None, 0, 0.0) else float(scaler.mean_[i]))
        elif col == "review_length":
            row.append(float(len(full_text)))
        elif col == "keyword_count":
            row.append(float(count_keywords(full_text)))
    X = np.asarray([row], dtype=np.float32)
    return scaler.transform(X)


def get_structured_matrix(df: pd.DataFrame, scaler: StandardScaler | None = None,
                           fit: bool = False) -> tuple[np.ndarray, StandardScaler]:
    """
    Extract structured feature matrix.
    Returns (X_struct, scaler).  Fit scaler if fit=True, else transform only.
    """
    cols = [c for c in STRUCTURED_COLS if c in df.columns]
    X = df[cols].fillna(0).values.astype(np.float32)
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("Provide a fitted scaler when fit=False.")
        X = scaler.transform(X)
    return X, scaler


# ── Vectoriser wrappers ──────────────────────────────────────────────────────

class BagOfWordsFeatures:
    """Wrapper around CountVectorizer with save/load support."""

    def __init__(self, max_features: int = BOW_MAX_FEATURES):
        self.vec = CountVectorizer(max_features=max_features, ngram_range=(1, 1))

    def fit_transform(self, texts: list[str]) -> sp.csr_matrix:
        return self.vec.fit_transform(texts)

    def transform(self, texts: list[str]) -> sp.csr_matrix:
        return self.vec.transform(texts)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vec, path)
        log.info(f"BoW vectorizer saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BagOfWordsFeatures":
        obj = cls.__new__(cls)
        obj.vec = joblib.load(path)
        return obj


class TFIDFFeatures:
    """Wrapper around TfidfVectorizer with save/load support."""

    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
        sublinear_tf: bool = True,
    ):
        self.vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
        )

    def fit_transform(self, texts: list[str]) -> sp.csr_matrix:
        return self.vec.fit_transform(texts)

    def transform(self, texts: list[str]) -> sp.csr_matrix:
        return self.vec.transform(texts)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vec, path)
        log.info(f"TF-IDF vectorizer saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TFIDFFeatures":
        obj = cls.__new__(cls)
        obj.vec = joblib.load(path)
        return obj


# ── Combined: TF-IDF + structured ───────────────────────────────────────────

def build_combined_features(
    tfidf_matrix: sp.csr_matrix,
    struct_matrix: np.ndarray,
) -> sp.csr_matrix:
    """Horizontally stack TF-IDF sparse matrix and structured dense matrix."""
    struct_sparse = sp.csr_matrix(struct_matrix)
    return sp.hstack([tfidf_matrix, struct_sparse])
