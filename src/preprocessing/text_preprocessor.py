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

# Download required NLTK data only if it is missing (avoids a network call on every import)
_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}
for _pkg, _path in _NLTK_RESOURCES.items():
    try:
        nltk.data.find(_path)
    except LookupError:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass

# Negations carry the sentiment of a complaint ("not clean", "never again", "no refund"),
# so they must NOT be removed together with ordinary stop-words.
_NEGATIONS = {
    "no", "nor", "not", "never", "cannot", "against",
    "don", "didn", "doesn", "isn", "wasn", "weren", "aren", "wouldn", "couldn",
    "shouldn", "hasn", "haven", "hadn", "won",
}


def _expand_contractions(text: str) -> str:
    """Turn "didn't" -> "did not" so the negation survives tokenisation."""
    text = text.replace("\u2019", "'")
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "cannot", text)
    text = re.sub(r"n't\b", " not", text)
    return text
_STOP_WORDS = set(stopwords.words("english")) - _NEGATIONS
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
    text = _expand_contractions(_base_clean(text).lower())
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in _STOP_WORDS]
    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ── Pipeline B — DL (Embedding-based) ───────────────────────────────────────

def dl_clean(text: str) -> str:
    """
    Moderate cleaning for DL embedding models.
    Lower-cases the text but does NOT remove stop-words (embedding context matters).
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
    def _s(v) -> str:
        # NaN / None -> "" (otherwise float NaN would become the literal word "nan")
        if v is None or (isinstance(v, float) and v != v):
            return ""
        return str(v)

    combined = (_s(title) + " " + _s(review)).strip()
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
    return pd.Series(
        [build_full_text(t, r, pipeline) for t, r in zip(df[title_col], df[review_col])],
        index=df.index,
    )
