"""
label_generator.py — Phase 4: Severity & Risk label construction.

IMPORTANT: These are derived (rule-based) labels constructed from rating and
text signals. They have NOT been manually verified by human annotators.
Use them as a proxy ground truth for model training and evaluation only.

Severity rules (in priority order):
─────────────────────────────────────
Critical : Overall_Rating <= 2
           OR (Overall_Rating in [3, 4] AND (>= 3 aspect ratings <= 2
               OR (Recommended == 'no' AND critical keyword present)))
High     : Overall_Rating in [3, 4] (otherwise)
Medium   : Overall_Rating in [5, 6]
Low      : Overall_Rating >= 7
Missing rating: High if critical keyword present else Medium.

Keyword upgrades (critical keyword present):
  Low / Medium -> High;  High + Recommended == 'no' -> Critical.

Risk rules:
─────────────────────────────────────
risk_flag = 1 if:
  - Recommended == 'no'  AND
  - Overall_Rating <= 4  AND
  - (>= 2 poor aspect ratings OR risk keyword in review)
risk_flag = 0 otherwise.
"""
from __future__ import annotations

import re
import pandas as pd
import numpy as np
from src.utils.config import CRITICAL_KEYWORDS, RISK_KEYWORDS, RATING_COLS
from src.utils.config import LOGS_DIR
from src.utils.logger import get_logger

log = get_logger("label_generator", LOGS_DIR)


# ── Public API ──────────────────────────────────────────────────────────────

def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'severity_label', 'severity_id', and 'risk_flag' columns."""
    df = df.copy()
    log.info("Generating severity and risk labels …")

    # Robust to un-cleaned input (e.g. Overall_Rating still a string like "7" or "n")
    if "Overall_Rating" not in df.columns:
        df["Overall_Rating"] = np.nan
    df["Overall_Rating"] = pd.to_numeric(df["Overall_Rating"], errors="coerce")
    for col in RATING_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("Review_Title", "Review"):
        if col not in df.columns:
            df[col] = ""

    # Build combined text for keyword scanning
    df["_text"] = (
        df["Review_Title"].fillna("") + " " + df["Review"].fillna("")
    ).str.lower()

    df["severity_label"] = df.apply(_assign_severity, axis=1)
    df["severity_id"] = df["severity_label"].map(
        {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    )
    df["risk_flag"] = df.apply(_assign_risk, axis=1)

    # Drop helper column
    df.drop(columns=["_text"], inplace=True)

    log.info(f"Severity distribution:\n{df['severity_label'].value_counts().to_string()}")
    log.info(f"Risk flag distribution:\n{df['risk_flag'].value_counts().to_string()}")
    return df


# ── Internal helpers ────────────────────────────────────────────────────────

def _has_critical_keyword(text: str) -> bool:
    return any(kw in text for kw in CRITICAL_KEYWORDS)


def _has_risk_keyword(text: str) -> bool:
    return any(kw in text for kw in RISK_KEYWORDS)


def _count_poor_aspects(row: pd.Series) -> int:
    """Count aspect rating columns that are <= 2 (very poor)."""
    count = 0
    for col in RATING_COLS:
        if col in row.index and not pd.isna(row[col]) and row[col] <= 2:
            count += 1
    return count


def _assign_severity(row: pd.Series) -> str:
    rating = row["Overall_Rating"]
    recommended = str(row.get("Recommended", "")).strip().lower()
    text = row.get("_text", "")
    has_critical_kw = _has_critical_keyword(text)
    poor_aspects = _count_poor_aspects(row)

    # ── Base rules ──────────────────────────────────────────────────────
    if pd.isna(rating):
        severity = "High" if has_critical_kw else "Medium"
    elif rating <= 2:
        severity = "Critical"
    elif rating <= 4:
        if poor_aspects >= 3 or (recommended == "no" and has_critical_kw):
            severity = "Critical"
        else:
            severity = "High"
    elif rating <= 6:
        severity = "Medium"
    else:  # >= 7
        severity = "Low"

    # ── Keyword upgrades ─────────────────────────────────────────────────
    if has_critical_kw:
        # Upgrade Low/Medium to High; upgrade High to Critical only when
        # combined with negative recommendation
        if severity == "Low":
            severity = "High"
        elif severity == "Medium":
            severity = "High"
        elif severity == "High" and recommended == "no":
            severity = "Critical"

    return severity


def _assign_risk(row: pd.Series) -> int:
    """
    Risk = 1 if: recommended == 'no' AND overall_rating <= 4
                 AND (>= 2 poor aspects OR risk keyword in text)
    """
    recommended = str(row.get("Recommended", "")).strip().lower()
    rating = row.get("Overall_Rating", np.nan)
    text = row.get("_text", "")
    poor_aspects = _count_poor_aspects(row)

    if recommended == "no" and (not pd.isna(rating) and rating <= 4):
        if poor_aspects >= 2 or _has_risk_keyword(text):
            return 1
    return 0
