"""
cleaner.py — Phase 2: Structured data cleaning.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from src.utils.config import RATING_COLS
from src.utils.config import LOGS_DIR
from src.utils.logger import get_logger

log = get_logger("cleaner", LOGS_DIR)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all Phase-2 cleaning steps in order.
    Returns a cleaned copy; never silently drops records.
    """
    df = df.copy()
    log.info("Starting Phase-2 data cleaning …")

    df = _convert_overall_rating(df)
    df = _convert_aspect_ratings(df)
    df = _normalize_categoricals(df)
    df = _drop_exact_duplicates(df)
    df = _fill_text_nulls(df)

    log.info(f"Cleaning complete. Final shape: {df.shape}")
    return df


# ── Step helpers ────────────────────────────────────────────────────────────

def _convert_overall_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Overall_Rating to numeric; mark invalid as NaN."""
    original_nulls = df["Overall_Rating"].isnull().sum()
    df["Overall_Rating"] = pd.to_numeric(df["Overall_Rating"], errors="coerce")
    new_nulls = df["Overall_Rating"].isnull().sum()
    log.info(
        f"Overall_Rating: {new_nulls - original_nulls} values coerced to NaN "
        f"(total NaN after: {new_nulls})"
    )
    return df


def _convert_aspect_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure aspect rating columns are numeric, then median-impute missing values."""
    for col in RATING_COLS:
        if col not in df.columns:
            log.warning(f"Column '{col}' not found — skipping.")
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Create missing-value indicator before imputation
        indicator = f"{col.replace(' ', '_').replace('&', 'and')}_missing"
        df[indicator] = df[col].isnull().astype(int)
        median_val = df[col].median()
        if pd.isna(median_val):  # column entirely missing
            median_val = 0.0
        df[col] = df[col].fillna(median_val)
        log.debug(f"  {col}: imputed {df[indicator].sum()} NaNs with median={median_val:.2f}")
    return df


def _normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip categorical fields."""
    for col in ["Type Of Traveller", "Seat Type", "Recommended", "Verified"]:
        if col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype("string").str.strip().str.lower().astype(object)
    # Standardise boolean-like Verified to True/False string
    if "Verified" in df.columns and pd.api.types.is_bool_dtype(df["Verified"]):
        df["Verified"] = df["Verified"].map({True: "yes", False: "no"})
    return df


def _drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates based on key text fields."""
    key_cols = ["Airline Name", "Review_Title", "Review Date", "Review"]
    key_cols = [c for c in key_cols if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    dropped = before - len(df)
    log.info(f"Duplicate removal: dropped {dropped} rows.")
    return df


def _fill_text_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Fill null Review_Title with empty string; never drop a row for missing title."""
    for col in ("Review_Title", "Review"):
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna("")
    return df
