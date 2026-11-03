"""
text_preprocessor.py — Phase 3: Text cleaning with three separate pipelines.

Pipeline A (ml_clean)         : for TF-IDF / BoW models
Pipeline B (dl_clean)         : for embedding-based DL models
Pipeline C (transformer_clean): for Hugging Face Transformer tokenizers
"""
from __future__ import annotations

import re
import html
import unicodedata
from functools import lru_cache
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (only on first run)
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# ── Shared low-level cleaner ────────────────────────────────────────────────

def _base_clean(text: str) -> str:
    """Encoding fix → HTML unescape → URL/HTML removal → whitespace."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    # Fix encoding artefacts
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    # HTML unescape
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove emojis / non-ASCII symbols
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "So"  # Symbol, Other
    )
    # Collapse excess punctuation
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Pipeline A — ML (TF-IDF / BoW) ─────────────────────────────────────────

def ml_clean(text: str, lemmatize: bool = True) -> str:
    """
    Full NLP preprocessing for traditional ML models.
    Steps: base_clean → lowercase → tokenize → stop-word removal → lemmatize.
    """
    text = _base_clean(text).lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in _STOP_WORDS]
    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ── Pipeline B — DL (Embedding-based) ───────────────────────────────────────

def dl_clean(text: str) -> str:
    """
    Moderate cleaning for DL embedding models.
    Keeps original casing and does NOT remove stop-words (embedding context matters).
    """
    text = _base_clean(text).lower()
    # Remove non-alphanumeric except apostrophes
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Pipeline C — Transformer ─────────────────────────────────────────────────

def transformer_clean(text: str) -> str:
    """
    Minimal cleaning for Transformer tokenizers.
    Preserves case and stopwords; just fixes encoding, HTML, URLs and excess whitespace.
    """
    return _base_clean(text)


# ── Combined review builder ──────────────────────────────────────────────────

def build_full_text(
    title: Optional[str],
    review: Optional[str],
    pipeline: str = "ml",
) -> str:
    """
    Concatenate title + review and apply the requested pipeline.

    Args:
        title    : Review_Title field value.
        review   : Review field value.
        pipeline : 'ml' | 'dl' | 'transformer'
    """
    combined = (str(title or "") + " " + str(review or "")).strip()
    if pipeline == "ml":
        return ml_clean(combined)
    elif pipeline == "dl":
        return dl_clean(combined)
    elif pipeline == "transformer":
        return transformer_clean(combined)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline!r}. Choose 'ml', 'dl', or 'transformer'.")


def preprocess_dataframe(df, pipeline: str = "ml", title_col: str = "Review_Title",
                          review_col: str = "Review") -> "pd.Series":
    """Apply text preprocessing to a DataFrame and return the cleaned text Series."""
    import pandas as pd
    return df.apply(
        lambda row: build_full_text(row[title_col], row[review_col], pipeline),
        axis=1,
    )
