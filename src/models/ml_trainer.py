"""
ml_trainer.py — Phase 7: Traditional ML model training & evaluation.

Models trained:
  Severity : Logistic Regression, Linear SVM, Multinomial Naive Bayes
  Risk     : Logistic Regression, Random Forest, XGBoost (or LightGBM)

Feature sets:
  - BoW
  - TF-IDF
  - TF-IDF + structured features
"""
from __future__ import annotations

import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    try:
        from lightgbm import LGBMClassifier as _LGBM
        _HAS_LGBM = True
    except ImportError:
        _HAS_LGBM = False

from src.utils.config import RANDOM_SEED, MODELS_ML, TEST_SIZE
from src.utils.config import LOGS_DIR
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
log = get_logger("ml_trainer", LOGS_DIR)

# ── Model Factories ──────────────────────────────────────────────────────────

def _severity_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED,
            class_weight="balanced", C=1.0, solver="saga", n_jobs=-1,
        ),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=RANDOM_SEED,
                      class_weight="balanced"), cv=3,
        ),
        "Multinomial NB": MultinomialNB(alpha=0.1),
    }


def _risk_models() -> dict[str, Any]:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED,
            class_weight="balanced", C=1.0, solver="saga", n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_SEED,
            class_weight="balanced", n_jobs=-1,
        ),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, random_state=RANDOM_SEED,
            eval_metric="logloss", use_label_encoder=False,
            scale_pos_weight=1, n_jobs=-1,
        )
    elif _HAS_LGBM:
        models["LightGBM"] = _LGBM(
            n_estimators=300, random_state=RANDOM_SEED,
            class_weight="balanced", n_jobs=-1,
        )
    return models


# ── Training helpers ─────────────────────────────────────────────────────────

def train_severity_models(
    X_train, X_test, y_train, y_test,
    feature_name: str = "tfidf",
) -> list[dict]:
    """Train all severity classifiers; return list of result dicts."""
    results = []
    models = _severity_models()

    for name, clf in models.items():
        log.info(f"Training severity | {name} | {feature_name} …")
        try:
            t0 = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - t0

            t1 = time.time()
            y_pred = clf.predict(X_test)
            inf_time = time.time() - t1

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            macro_f1 = report["macro avg"]["f1-score"]
            macro_rec = report["macro avg"]["recall"]
            macro_prec = report["macro avg"]["precision"]

            # Per-class recall
            class_recalls = {
                str(k): v["recall"]
                for k, v in report.items()
                if k not in ("accuracy", "macro avg", "weighted avg")
            }

            result = {
                "model": name,
                "feature": feature_name,
                "task": "severity",
                "accuracy": acc,
                "macro_f1": macro_f1,
                "macro_recall": macro_rec,
                "macro_precision": macro_prec,
                "class_recalls": class_recalls,
                "confusion_matrix": cm.tolist(),
                "train_time_s": train_time,
                "inf_time_s": inf_time,
                "clf_object": clf,
            }
            results.append(result)
            log.info(
                f"  -> Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}, "
                f"Train={train_time:.1f}s"
            )

            # Save model
            _save_model(clf, MODELS_ML / f"severity_{name.replace(' ', '_')}_{feature_name}.pkl")

        except Exception as e:
            log.error(f"  [X] {name} failed: {e}")

    return results


def train_risk_models(
    X_train, X_test, y_train, y_test,
    feature_name: str = "tfidf_struct",
) -> list[dict]:
    """Train all risk classifiers; return list of result dicts."""
    results = []
    models = _risk_models()

    for name, clf in models.items():
        log.info(f"Training risk | {name} | {feature_name} …")
        try:
            t0 = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - t0

            t1 = time.time()
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            inf_time = time.time() - t1

            roc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            fn_count = int(cm[1, 0]) if cm.shape == (2, 2) else -1

            result = {
                "model": name,
                "feature": feature_name,
                "task": "risk",
                "roc_auc": roc,
                "pr_auc": pr_auc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "false_negatives": fn_count,
                "confusion_matrix": cm.tolist(),
                "train_time_s": train_time,
                "inf_time_s": inf_time,
                "clf_object": clf,
            }
            results.append(result)
            log.info(
                f"  -> ROC-AUC={roc:.4f}, Recall={rec:.4f}, FN={fn_count}, "
                f"Train={train_time:.1f}s"
            )

            _save_model(clf, MODELS_ML / f"risk_{name.replace(' ', '_')}_{feature_name}.pkl")

        except Exception as e:
            log.error(f"  [X] {name} failed: {e}")

    return results


def _save_model(clf, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)
    log.debug(f"Model saved: {path}")
