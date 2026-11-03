# Code review — changes & fixes

36 tests pass and 1 is skipped when PyTorch is not installed (`python -m pytest tests`); every Streamlit page was run headless with no
exceptions, and the single + bulk prediction paths were exercised end-to-end.
Models/artifacts were **re-trained on the full dataset (23,051 rows)** with the fixed code.

## Bugs fixed

| # | Where | Problem | Fix |
|---|-------|---------|-----|
| 1 | `pages/3_Prediction.py` (bulk CSV) | Risk model was trained on TF-IDF **+ 9 structured features** but bulk mode fed it TF-IDF only → dimension-mismatch crash | Rebuild the same combined features in bulk mode |
| 2 | `pages/3_Prediction.py` (single) | `keyword_count` hard-coded to `0`; `review_length` ignored the title; unset ratings sent as `0` (never occurs in data, range is 1-5) | New `build_inference_struct()` in `feature_engineer.py` — same definition as training; missing ratings → training mean |
| 3 | `ml_trainer.py` | Logistic Regression used `saga`; it hit `max_iter=1000` **without converging** (warning was hidden by a global `filterwarnings("ignore")`) → TF-IDF severity macro-F1 0.42 | `lbfgs` solver: macro-F1 **0.42 → 0.55**, ~15× faster. Warnings are now logged, not hidden |
| 4 | `text_preprocessor.py` | NLTK stop-words removed **negations** (`not`, `no`, `never`, `didn't`…) — "not good" became "good", a serious flaw for complaint text | Negations kept; contractions expanded (`didn't` → `did not`) |
| 5 | `text_preprocessor.py` | `NaN` titles became the literal word `"nan"` in the text | NaN/None-safe `build_full_text` |
| 6 | `evaluator.py` | For risk models the **"Accuracy" column actually contained ROC-AUC**, "Macro F1" contained binary F1, and n/a metrics showed `0.0` | Risk results now record real accuracy & macro-F1; n/a metrics are `NaN` |
| 7 | `label_generator.py` | Crashed if `Overall_Rating` was still a string (5 failing tests); docstring didn't match the real rules | Coerces ratings/aspects to numeric; docstring corrected |
| 8 | `cleaner.py` | `dtype == object` check fails on pandas 3 `str` dtype → categoricals never normalised; all-NaN rating column stayed NaN | Version-safe dtype checks; fallback fill |
| 9 | `dl_trainer.py` | `ReduceLROnPlateau(verbose=True)` raises `TypeError` on recent PyTorch | Removed |
| 10 | `dl_trainer.py` | Every review padded to 256 tokens and LSTM read the *last* state after dozens of `<PAD>` steps; CNN crashed on batches shorter than the kernel size | Dynamic padding, `pack_padded_sequence`, min length 5 |
| 11 | `transformer_trainer.py` | `Trainer(tokenizer=…)` / `eval_strategy` break across transformers versions | Version-aware (`processing_class` / `evaluation_strategy`) |
| 12 | `train_pipeline.py` | **Test set used for early stopping / best-model selection** (DL & Transformers) → optimistic scores; `VAL_SIZE` was unused | Proper train / val / test split; metrics reported on test only |
| 13 | `train_pipeline.py` | ~40 lines of duplicated training code; wrong return annotation; transformer train-time always `0` | Reuses `train_severity_models` (skips MultinomialNB on negative features); real times |
| 14 | `ml_trainer.py` | XGBoost: deprecated `use_label_encoder`, `scale_pos_weight=1` ignored class imbalance | Removed; weight computed from `y_train` |
| 15 | `explainer.py` | `get_linear_top_features` always errored for calibrated SVMs (read unfitted estimator); duplicate phrases (`cancel` + `cancelled`) | Reads fitted estimators; overlapping phrases de-duplicated |
| 16 | `pages/3_Prediction.py` | Rule-based fallback: nonsense walrus expression, operator-precedence bug in text building; bulk fallback dropped duplicates → rows misaligned | Fixed |
| 17 | `pages/2_EDA_Dashboard.py` | Dates like `11th November 2019` never parsed → trend chart never shown; `COLORS` could be undefined | Ordinal suffixes stripped; `COLORS` defined up-front |
| 18 | all pages | `use_container_width` removed from Streamlit | `width="stretch"` (requires `streamlit>=1.50`) |
| 19 | prediction page default model | Default severity model was the worst one (it labelled a clearly positive review *Critical*) | Now the fixed LR-TF-IDF model |
| 20 | `app.py` | Home page was blank | Added landing text |

New tests: `tests/test_fixes.py` (13 regression tests).

## Results after retraining (test split, full data)

| Task | Model | Accuracy | Macro-F1 |
|------|-------|---------:|---------:|
| Severity (text only) | Logistic Regression + TF-IDF | 0.759 | 0.547 |
| Severity (text + ratings) | Logistic Regression | 0.769 | 0.617 |
| Risk | XGBoost | 0.932 | 0.929 (ROC-AUC 0.983) |

## Things to be aware of (not changed — design decisions for you)

1. **Label leakage in the Risk task.** `risk_flag` is *defined* from `Recommended`, `Overall_Rating`
   and the aspect ratings, and the risk model is given the aspect ratings (+ keyword count, which
   is also in the label rule). The 0.93 / 0.98 scores therefore partly measure how well the model
   re-learns your rules, not real-world risk detection. Consider a text-only risk model, or
   human-labelled data, before reporting these numbers.
2. **Severity labels are rule-derived** (mostly from `Overall_Rating`), so scores are against proxy
   labels, as your own docstring says.
3. **Minority classes are weak** (macro-F1 ≈ 0.55 vs accuracy ≈ 0.76) — Medium/High are hard to
   separate. Try threshold tuning, merging classes, or the Transformer phase.
4. **The DL / Transformer code could not be run here** (no PyTorch in my environment). The fixes
   above are written against the PyTorch / transformers APIs but please run
   `python train_pipeline.py --sample 2000` once on your machine to confirm.
