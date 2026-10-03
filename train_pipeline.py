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
    if sample is not None and sample < 200:
        raise SystemExit("--sample must be >= 200 (stratified splitting needs enough rows per class).")
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

def run_ml_phases(df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame, dict]:
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

    # Validation split carved out of the TRAIN part only (used by DL / Transformer early
    # stopping and model selection so the test set is never used for tuning).
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    idx_fit, idx_val = train_test_split(
        idx_train, test_size=val_frac, random_state=RANDOM_SEED, stratify=y_sev_train,
    )

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
        # (MultinomialNB is skipped automatically for "tfidf_struct": scaled features are negative)
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

    splits = {"fit": idx_fit, "val": idx_val, "train": idx_train, "test": idx_test}
    return all_results, df, splits


# ── Phase 8: DL ───────────────────────────────────────────────────────────────

def run_dl_phase(df: pd.DataFrame, splits: dict) -> list[dict]:
    """Train CNN / LSTM / BiLSTM. Early stopping uses the VAL split; metrics use TEST."""
    log.info("=== PHASE 8: Deep Learning ===")
    import torch.nn as nn
    from sklearn.metrics import classification_report
    from torch.utils.data import DataLoader
    from src.models.dl_trainer import (
        SimpleTokenizer, ReviewDataset, _collate_fn, train_dl_model, eval_epoch,
    )

    df_fit, df_val, df_test = (df.loc[splits[k]] for k in ("fit", "val", "test"))

    # Build and fit tokenizer on training data only
    tokenizer = SimpleTokenizer(max_vocab=50_000)
    tokenizer.build_vocab(df_fit["text_dl"].tolist())
    tokenizer.save(ARTIFACTS_VEC / "dl_tokenizer.pkl")

    def _make_loader(frame: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
        ids = [tokenizer.encode(t) for t in frame["text_dl"].tolist()]
        ds = ReviewDataset(ids, frame["severity_id"].tolist())
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn)

    train_dl = _make_loader(df_fit, 64, True)
    val_dl = _make_loader(df_val, 128, False)
    test_dl = _make_loader(df_test, 128, False)

    y_fit = df_fit["severity_id"].tolist()
    cw = compute_class_weight("balanced", classes=np.unique(y_fit), y=y_fit)
    class_weights = torch.tensor(cw, dtype=torch.float32)

    dl_results = []
    for arch in ["CNN", "LSTM", "BiLSTM"]:
        try:
            model, history = train_dl_model(
                arch, train_dl, val_dl,
                num_classes=4, vocab_size=len(tokenizer.word2idx),
                embed_dim=128, epochs=8, class_weights=class_weights,
            )
            t0 = time.time()
            _, test_acc, y_pred, y_true = eval_epoch(model, test_dl, nn.CrossEntropyLoss())
            inf_time = time.time() - t0
            rpt = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            dl_results.append({
                "model": arch, "task": "severity", "model_family": "DL",
                "feature": "embedding",
                "accuracy": test_acc,
                "macro_f1": rpt["macro avg"]["f1-score"],
                "macro_recall": rpt["macro avg"]["recall"],
                "train_time_s": sum(h.get("time", 0) for h in history),
                "inf_time_s": inf_time,
            })
        except Exception as e:
            log.error(f"DL {arch} failed: {e}")
    return dl_results


# ── Phase 9: Transformers ─────────────────────────────────────────────────────

def run_transformer_phase(df: pd.DataFrame, splits: dict) -> list[dict]:
    """Fine-tune transformers. Early stopping uses VAL; reported metrics use TEST."""
    log.info("=== PHASE 9: Transformer Fine-Tuning ===")
    from src.models.transformer_trainer import tokenize_data, fine_tune_transformer

    df_fit, df_val, df_test = (df.loc[splits[k]] for k in ("fit", "val", "test"))
    fit_texts = df_fit["text_transformer"].tolist()
    val_texts = df_val["text_transformer"].tolist()
    test_texts = df_test["text_transformer"].tolist()

    transformer_results = []

    for model_name in [TRANSFORMER_MODEL_NAME, BERT_MODEL_NAME]:
        log.info(f"--- {model_name} ---")
        short = model_name.split("-")[0].upper()

        for task, col, n_labels in [("severity", "severity_id", 4), ("risk", "risk_flag", 2)]:
            try:
                y_fit, y_val, y_test = (d[col].tolist() for d in (df_fit, df_val, df_test))
                train_ds, tok = tokenize_data(fit_texts, y_fit, model_name)
                val_ds, _ = tokenize_data(val_texts, y_val, model_name)
                test_ds, _ = tokenize_data(test_texts, y_test, model_name)
                cw = compute_class_weight("balanced", classes=np.unique(y_fit), y=y_fit)

                _, _, m = fine_tune_transformer(
                    model_name=model_name, task=task,
                    train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, tokenizer=tok,
                    num_labels=n_labels, class_weights=cw.tolist(),
                    epochs=3, batch_size=16,
                )
                if task == "severity":
                    transformer_results.append({
                        "model": short, "task": "severity", "model_family": "Transformer",
                        "feature": "full_text",
                        "accuracy": m.get("eval_accuracy"),
                        "macro_f1": m.get("eval_macro_f1"),
                        "macro_recall": m.get("eval_macro_recall"),
                        "train_time_s": m.get("train_time_s", 0), "inf_time_s": 0,
                    })
                else:
                    transformer_results.append({
                        "model": f"{short}_risk", "task": "risk", "model_family": "Transformer",
                        "feature": "full_text",
                        "roc_auc": m.get("eval_roc_auc"),
                        "f1": m.get("eval_f1"),
                        "recall": m.get("eval_recall"),
                        "train_time_s": m.get("train_time_s", 0), "inf_time_s": 0,
                    })
            except Exception as e:
                log.error(f"{model_name} {task} failed: {e}")

    return transformer_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t_start = time.time()
    log.info("=" * 70)
    log.info("  AIRLINE COMPLAINT AI — TRAINING PIPELINE")
    log.info("=" * 70)

    df = run_data_phases(sample=args.sample)

    ml_results, df, splits = run_ml_phases(df)

    all_results = list(ml_results)

    if not args.skip_dl and HAS_TORCH:
        dl_results = run_dl_phase(df, splits)
        all_results.extend(dl_results)
    else:
        log.info("DL phase skipped (--skip-dl or PyTorch not installed)")

    if not args.skip_transformers and HAS_TORCH:
        tr_results = run_transformer_phase(df, splits)
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
