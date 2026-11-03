"""
data_loader.py — Phase 1: Dataset inspection and loading utilities.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import LOGS_DIR

log = get_logger("data_loader", LOGS_DIR)


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load the airline review CSV, report basic stats, and return the DataFrame."""
    path = Path(path)
    log.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    log.info(f"Dataset loaded — shape: {df.shape}")
    _inspect(df)
    return df


def _inspect(df: pd.DataFrame) -> None:
    """Log all Phase-1 inspection metrics."""
    log.info("=" * 60)
    log.info("PHASE 1 — DATASET INSPECTION")
    log.info("=" * 60)

    log.info(f"Shape          : {df.shape}")
    log.info(f"Columns        : {list(df.columns)}")
    log.info("\nData types:\n" + df.dtypes.to_string())
    log.info("\nMissing values:\n" + df.isnull().sum().to_string())
    log.info(f"Duplicate rows : {df.duplicated().sum()}")

    # Overall_Rating unique values
    log.info(f"Overall_Rating unique values: {df['Overall_Rating'].unique().tolist()}")

    # Recommended distribution
    if "Recommended" in df.columns:
        log.info(f"Recommended values:\n{df['Recommended'].value_counts().to_string()}")

    # Categorical fields
    for col in ["Type Of Traveller", "Seat Type", "Aircraft"]:
        if col in df.columns:
            log.info(f"{col} unique count: {df[col].nunique()}")

    # Rating ranges
    for col in ["Seat Comfort", "Cabin Staff Service", "Food & Beverages",
                "Ground Service", "Inflight Entertainment",
                "Wifi & Connectivity", "Value For Money"]:
        if col in df.columns:
            log.info(
                f"{col}: min={df[col].min()}, max={df[col].max()}, "
                f"nulls={df[col].isnull().sum()}"
            )

    # Review text nulls
    log.info(f"Review nulls   : {df['Review'].isnull().sum()}")
    log.info(f"Review_Title nulls: {df['Review_Title'].isnull().sum()}")
    log.info("=" * 60)


def get_inspection_report(df: pd.DataFrame) -> dict:
    """Return a structured inspection report dict (used by the Streamlit EDA page)."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "overall_rating_unique": df["Overall_Rating"].unique().tolist(),
        "recommended_counts": df["Recommended"].value_counts().to_dict()
        if "Recommended" in df.columns else {},
    }
