"""
train_pipeline.py — Master training script.

Runs Phases 1-10 end-to-end:
  Phase 1  : Dataset inspection
  Phase 2  : Data cleaning
  Phase 3  : Text preprocessing
  Phase 4  : Label generation
  Phase 5  : EDA (saved as PNG files)
  Phase 6  : Feature engineering
  Phase 7  : ML training
  Phase 8  : DL training
  Phase 9  : Transformer fine-tuning (optional flag)
  Phase 10 : Evaluation & comparison table

Usage:
    python train_pipeline.py [--skip-dl] [--skip-transformers] [--sample N]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

warnings.filterwarnings("ignore")

# ── Ensure src is on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import (
    DATA_RAW, DATA_PROCESSED, ARTIFACTS_VEC, ARTIFACTS_ENC,
    RANDOM_SEED, TEST_SIZE, VAL_SIZE,
    SEVERITY_LABELS, MODELS_DL, MODELS_TRANSFORMER,
    TRANSFORMER_MODEL_NAME, BERT_MODEL_NAME, LOGS_DIR,
)
from src.utils.logger import get_logger
from src.data.data_loader import load_raw_dataset
from src.data.cleaner import clean_dataset
from src.data.label_generator import generate_labels
from src.preprocessing.text_preprocessor import preprocess_dataframe
from src.features.feature_engineer import (
    BagOfWordsFeatures, TFIDFFeatures, add_structured_features,
    get_structured_matrix, build_combined_features,
)
from src.models.ml_trainer import train_severity_models, train_risk_models
from src.evaluation.evaluator import (
    evaluate_severity, evaluate_risk, build_comparison_table,
)

log = get_logger("train_pipeline", LOGS_DIR)

# ── Seeds ──────────────────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if HAS_TORCH:
    torch.manual_seed(RANDOM_SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Airline Complaint AI — Training Pipeline")
    parser.add_argument("--skip-dl", action="store_true", help="Skip deep-learning phase")
    parser.add_argument("--skip-transformers", action="store_true", help="Skip Transformer phase")
    parser.add_argument("--sample", type=int, default=None, help="Limit dataset to N rows for quick test")
    return parser.parse_args()


# ── Phase 1 + 2 + 3 + 4 ───────────────────────────────────────────────────────

def run_data_phases(sample: int | None = None) -> pd.DataFrame:
    log.info("=== PHASE 1: Dataset Inspection ===")
    df = load_raw_dataset(DATA_RAW)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)
        log.info(f"Sampled {len(df)} rows for quick run.")

    log.info("=== PHASE 2: Data Cleaning ===")
    df = clean_dataset(df)

    log.info("=== PHASE 3: Text Preprocessing ===")
    df["text_ml"] = preprocess_dataframe(df, pipeline="ml")
    df["text_dl"] = preprocess_dataframe(df, pipeline="dl")
    df["text_transformer"] = preprocess_dataframe(df, pipeline="transformer")

    log.info("=== PHASE 4: Label Generation ===")
    df = generate_labels(df)

    # Add structured features
    df["full_review_text"] = df["Review_Title"].fillna("") + " " + df["Review"].fillna("")
    df = add_structured_features(df, raw_text_col="full_review_text")

    # Save processed
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "airline_processed.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Processed data saved: {out_path}")
    return df


# ── Phase 6 + 7: Feature engineering + ML ────────────────────────────────────

def run_ml_phases(df: pd.DataFrame) -> list[dict]:
    log.info("=== PHASE 6 + 7: Features + ML Training ===")

    # ── Train/test split (stratified on severity) ───────────────────────────
    X_text_ml = df["text_ml"].tolist()
    X_text_dl = df["text_dl"].tolist()
    y_sev = df["severity_id"].tolist()
    y_risk = df["risk_flag"].tolist()

    (X_ml_train, X_ml_test, X_dl_train, X_dl_test,
     y_sev_train, y_sev_test, y_risk_train, y_risk_test,
     idx_train, idx_test) = train_test_split(
        X_text_ml, X_text_dl, y_sev, y_risk, df.index.tolist(),
        test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_sev,
    )

    df_train = df.loc[idx_train]
    df_test = df.loc[idx_test]

    # ── BoW ─────────────────────────────────────────────────────────────────
    bow = BagOfWordsFeatures()
    X_bow_train = bow.fit_transform(X_ml_train)
    X_bow_test = bow.transform(X_ml_test)
    bow.save(ARTIFACTS_VEC / "bow_vectorizer.pkl")

    # ── TF-IDF ──────────────────────────────────────────────────────────────
    tfidf = TFIDFFeatures()
    X_tfidf_train = tfidf.fit_transform(X_ml_train)
    X_tfidf_test = tfidf.transform(X_ml_test)
    tfidf.save(ARTIFACTS_VEC / "tfidf_vectorizer.pkl")

    # ── Structured features ──────────────────────────────────────────────────
    X_struct_train, scaler = get_structured_matrix(df_train, fit=True)
    X_struct_test, _ = get_structured_matrix(df_test, scaler=scaler, fit=False)
    joblib.dump(scaler, ARTIFACTS_VEC / "struct_scaler.pkl")

    # ── Combined ─────────────────────────────────────────────────────────────
    X_combined_train = build_combined_features(X_tfidf_train, X_struct_train)
    X_combined_test = build_combined_features(X_tfidf_test, X_struct_test)

    all_results = []

    # ── Severity models ──────────────────────────────────────────────────────
    for feature_name, X_tr, X_te in [
        ("bow", X_bow_train, X_bow_test),
        ("tfidf", X_tfidf_train, X_tfidf_test),
        ("tfidf_struct", X_combined_train, X_combined_test),
    ]:
        # MNB only works with non-negative features; skip combined (has negatives from scaler)
        if feature_name == "tfidf_struct":
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import LinearSVC
            from sklearn.calibration import CalibratedClassifierCV
            models_subset = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, random_state=RANDOM_SEED,
                    class_weight="balanced", solver="saga", n_jobs=-1),
                "Linear SVM": CalibratedClassifierCV(
                    LinearSVC(max_iter=2000, random_state=RANDOM_SEED,
                              class_weight="balanced"), cv=3),
            }
            import time
            from sklearn.metrics import (accuracy_score, classification_report,
                                         confusion_matrix)
            for name, clf in models_subset.items():
                try:
                    t0 = time.time()
                    clf.fit(X_tr, y_sev_train)
                    train_t = time.time() - t0
                    t1 = time.time()
                    y_pred = clf.predict(X_te)
                    inf_t = time.time() - t1
                    rpt = classification_report(y_sev_test, y_pred, output_dict=True, zero_division=0)
                    all_results.append({
                        "model": name, "task": "severity", "feature": feature_name,
                        "model_family": "ML",
                        "accuracy": accuracy_score(y_sev_test, y_pred),
                        "macro_f1": rpt["macro avg"]["f1-score"],
                        "macro_recall": rpt["macro avg"]["recall"],
                        "confusion_matrix": confusion_matrix(y_sev_test, y_pred).tolist(),
                        "train_time_s": train_t, "inf_time_s": inf_t,
                    })
                    from src.utils.config import MODELS_ML
                    joblib.dump(clf, MODELS_ML / f"severity_{name.replace(' ', '_')}_{feature_name}.pkl")
                    log.info(f"  {name} ({feature_name}): F1={rpt['macro avg']['f1-score']:.4f}")
                except Exception as e:
                    log.error(f"  {name} ({feature_name}): {e}")
        else:
            res = train_severity_models(X_tr, X_te, y_sev_train, y_sev_test, feature_name)
            for r in res:
                r["model_family"] = "ML"
            all_results.extend(res)

    # ── Risk models ──────────────────────────────────────────────────────────
    res_risk = train_risk_models(
        X_combined_train, X_combined_test,
        y_risk_train, y_risk_test,
        feature_name="tfidf_struct",
    )
    for r in res_risk:
        r["model_family"] = "ML"
    all_results.extend(res_risk)

    return all_results, df, (X_ml_train, X_ml_test, y_sev_train, y_sev_test,
                              y_risk_train, y_risk_test, idx_train, idx_test)


# ── Phase 8: DL ───────────────────────────────────────────────────────────────

def run_dl_phase(df: pd.DataFrame, idx_train, idx_test,
                 y_sev_train, y_sev_test) -> list[dict]:
    log.info("=== PHASE 8: Deep Learning ===")
    from src.models.dl_trainer import (
        SimpleTokenizer, ReviewDataset, _collate_fn, train_dl_model,
        eval_epoch,
    )
    from torch.utils.data import DataLoader

    df_train = df.loc[idx_train]
    df_test = df.loc[idx_test]

    # Build and fit tokenizer on training data only
    tokenizer = SimpleTokenizer(max_vocab=50_000)
    train_texts = df_train["text_dl"].tolist()
    tokenizer.build_vocab(train_texts)
    tokenizer.save(ARTIFACTS_VEC / "dl_tokenizer.pkl")

    # Encode
    def _encode(texts):
        return [tokenizer.encode(t) for t in texts]

    X_train_ids = _encode(train_texts)
    X_test_ids = _encode(df_test["text_dl"].tolist())

    # Class weights for severity (4 classes)
    classes = np.unique(y_sev_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_sev_train)
    class_weights = torch.tensor(cw, dtype=torch.float32)

    train_ds = ReviewDataset(X_train_ids, y_sev_train)
    test_ds = ReviewDataset(X_test_ids, y_sev_test)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=_collate_fn)
    val_dl = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=_collate_fn)

    dl_results = []
    for arch in ["CNN", "LSTM", "BiLSTM"]:
        try:
            import torch.nn as nn
            model, history = train_dl_model(
                arch, train_dl, val_dl,
                num_classes=4, vocab_size=len(tokenizer.word2idx),
                embed_dim=128, epochs=8, class_weights=class_weights,
            )
            _, val_acc, y_pred, y_true = eval_epoch(model, val_dl, nn.CrossEntropyLoss())
            from sklearn.metrics import classification_report
            rpt = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            dl_results.append({
                "model": arch, "task": "severity", "model_family": "DL",
                "feature": "embedding",
                "accuracy": val_acc,
                "macro_f1": rpt["macro avg"]["f1-score"],
                "macro_recall": rpt["macro avg"]["recall"],
                "train_time_s": sum(h.get("time", 0) for h in history),
                "inf_time_s": 0.0,
            })
        except Exception as e:
            log.error(f"DL {arch} failed: {e}")
    return dl_results


# ── Phase 9: Transformers ─────────────────────────────────────────────────────

def run_transformer_phase(df: pd.DataFrame, idx_train, idx_test,
                           y_sev_train, y_sev_test,
                           y_risk_train, y_risk_test) -> list[dict]:
    log.info("=== PHASE 9: Transformer Fine-Tuning ===")
    from src.models.transformer_trainer import (
        tokenize_data, fine_tune_transformer,
    )
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    df_train = df.loc[idx_train]
    df_test = df.loc[idx_test]

    train_texts = df_train["text_transformer"].tolist()
    test_texts = df_test["text_transformer"].tolist()

    transformer_results = []

    for model_name in [TRANSFORMER_MODEL_NAME, BERT_MODEL_NAME]:
        log.info(f"--- {model_name} ---")

        # ── Severity (4-class) ──────────────────────────────────────────
        try:
            train_ds, tok = tokenize_data(train_texts, y_sev_train, model_name)
            val_ds, _ = tokenize_data(test_texts, y_sev_test, model_name)

            cw = compute_class_weight("balanced", classes=np.unique(y_sev_train), y=y_sev_train)

            _, _, metrics = fine_tune_transformer(
                model_name=model_name, task="severity",
                train_ds=train_ds, val_ds=val_ds, tokenizer=tok,
                num_labels=4, class_weights=cw.tolist(),
                epochs=3, batch_size=16,
            )
            transformer_results.append({
                "model": model_name.split("-")[0].upper(),
                "task": "severity", "model_family": "Transformer",
                "feature": "full_text",
                "accuracy": metrics.get("eval_accuracy", 0),
                "macro_f1": metrics.get("eval_macro_f1", 0),
                "macro_recall": metrics.get("eval_macro_recall", 0),
                "train_time_s": 0, "inf_time_s": 0,
            })
        except Exception as e:
            log.error(f"{model_name} severity failed: {e}")

        # ── Risk (binary) ───────────────────────────────────────────────
        try:
            train_ds_r, tok_r = tokenize_data(train_texts, y_risk_train, model_name)
            val_ds_r, _ = tokenize_data(test_texts, y_risk_test, model_name)

            cw_r = compute_class_weight("balanced", classes=np.unique(y_risk_train), y=y_risk_train)
            _, _, metrics_r = fine_tune_transformer(
                model_name=model_name, task="risk",
                train_ds=train_ds_r, val_ds=val_ds_r, tokenizer=tok_r,
                num_labels=2, class_weights=cw_r.tolist(),
                epochs=3, batch_size=16,
            )
            transformer_results.append({
                "model": model_name.split("-")[0].upper() + "_risk",
                "task": "risk", "model_family": "Transformer",
                "feature": "full_text",
                "roc_auc": metrics_r.get("eval_roc_auc", 0),
                "macro_f1": metrics_r.get("eval_f1", 0),
                "recall": metrics_r.get("eval_recall", 0),
                "accuracy": 0, "train_time_s": 0, "inf_time_s": 0,
            })
        except Exception as e:
            log.error(f"{model_name} risk failed: {e}")

    return transformer_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t_start = time.time()
    log.info("=" * 70)
    log.info("  AIRLINE COMPLAINT AI — TRAINING PIPELINE")
    log.info("=" * 70)

    df = run_data_phases(sample=args.sample)

    ml_results, df, split_data = run_ml_phases(df)
    X_ml_train, X_ml_test, y_sev_train, y_sev_test, y_risk_train, y_risk_test, idx_train, idx_test = split_data

    all_results = list(ml_results)

    if not args.skip_dl and HAS_TORCH:
        dl_results = run_dl_phase(df, idx_train, idx_test, y_sev_train, y_sev_test)
        all_results.extend(dl_results)
    else:
        log.info("DL phase skipped (--skip-dl or PyTorch not installed)")

    if not args.skip_transformers and HAS_TORCH:
        tr_results = run_transformer_phase(
            df, idx_train, idx_test,
            y_sev_train, y_sev_test, y_risk_train, y_risk_test,
        )
        all_results.extend(tr_results)
    else:
        log.info("Transformer phase skipped (--skip-transformers or PyTorch not installed)")

    # ── Comparison table ────────────────────────────────────────────────────
    table = build_comparison_table(all_results)
    table_path = DATA_PROCESSED / "model_comparison_table.csv"
    table.to_csv(table_path, index=False)
    log.info(f"\nModel comparison table saved: {table_path}")
    log.info(f"\n{table.to_string(index=False)}")

    log.info(f"\n✅ Pipeline complete in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
