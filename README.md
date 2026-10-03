# ✈️ AI-Based Airline Complaint Severity Detection & Service Risk Classification System

> End-to-end NLP system for classifying airline review complaint severity (Low / Medium / High / Critical) and binary service-risk detection (Risk / No Risk) — powered by Traditional ML, Custom Deep Learning, and Pretrained Transformers.

---

## 📋 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Objectives](#objectives)
3. [Business Use Cases](#business-use-cases)
4. [Dataset Description](#dataset-description)
5. [Data Preprocessing](#data-preprocessing)
6. [Label-Generation Rules](#label-generation-rules)
7. [EDA](#eda)
8. [Feature Engineering](#feature-engineering)
9. [ML Models](#ml-models)
10. [Deep Learning Models](#deep-learning-models)
11. [Transformer Models](#transformer-models)
12. [Evaluation Metrics](#evaluation-metrics)
13. [Model Comparison](#model-comparison)
14. [Explainability](#explainability)
15. [Streamlit Application](#streamlit-application)
16. [Project Structure](#project-structure)
17. [Installation](#installation)
18. [Usage](#usage)
19. [Training Instructions](#training-instructions)
20. [Model Artifacts](#model-artifacts)
21. [AWS Deployment](#aws-deployment)
22. [Hugging Face Deployment](#hugging-face-deployment)
23. [Testing](#testing)
24. [Limitations](#limitations)
25. [Future Improvements](#future-improvements)

---

## Problem Statement

Airlines receive thousands of customer reviews daily. Manual triage is slow, inconsistent, and cannot scale. This system automates:

- **Severity classification**: Identifying how serious a complaint is (Low → Critical)
- **Service risk detection**: Flagging reviews that signal potential churn or operational failure

---

## Objectives

- Automate severity labelling of airline customer reviews
- Detect service-risk patterns requiring operational intervention
- Compare ML, DL, and Transformer model families fairly
- Provide explainable predictions for non-technical stakeholders
- Support real-time single-review and batch CSV prediction

---

## Business Use Cases

| Use Case | Description |
|----------|-------------|
| 🚨 Critical Alert Routing | Auto-escalate Critical reviews to supervisors |
| 📊 Airline Benchmarking | Compare complaint rates across airlines and periods |
| 🛡️ Risk Early Warning | Flag service-risk patterns before customer churn |
| 📈 Quality Analytics | Track severity trends by route, seat class, traveller type |
| 🤖 Review Triage | Prioritise agent response queues by severity |

---

## Dataset Description

**Source:** `Airline_review.csv`
**Size:** 23,171 rows × 20 columns

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `Unnamed: 0` | int | Exported row index (NOT used as feature) |
| `Airline Name` | str | Airline identifier |
| `Overall_Rating` | obj→float | 1–10 rating (raw has invalid 'n' values) |
| `Review_Title` | str | Short review heading |
| `Review Date` | str | Date of review submission |
| `Verified` | bool | Whether reviewer is verified |
| `Review` | str | Full review text |
| `Aircraft` | str | Aircraft type (16,042 missing) |
| `Type Of Traveller` | str | Solo Leisure, Business, Family, Couple |
| `Seat Type` | str | Economy, Business, Premium Economy, First |
| `Route` | str | Flight route (3,828 missing) |
| `Date Flown` | str | Date of actual flight |
| `Seat Comfort` | float | 1–5 rating |
| `Cabin Staff Service` | float | 1–5 rating |
| `Food & Beverages` | float | 1–5 rating (8,671 missing) |
| `Ground Service` | float | 1–5 rating |
| `Inflight Entertainment` | float | 1–5 rating (12,342 missing) |
| `Wifi & Connectivity` | float | 1–5 rating (17,251 missing — most sparse) |
| `Value For Money` | float | 1–5 rating |
| `Recommended` | str | yes / no |

---

## Data Preprocessing

### Phase 2 — Structured Cleaning
- `Overall_Rating` coerced to numeric; 'n' → NaN
- Aspect rating columns median-imputed per column; missing-value indicator created
- `Recommended`, `Type Of Traveller`, `Seat Type` lowercased and stripped
- Duplicates removed using `[Airline Name, Review_Title, Review Date, Review]`
- Records with missing optional fields (`Aircraft`, `Route`) are **NOT** dropped

### Phase 3 — Text Pipelines (Three Separate Pipelines)

| Pipeline | Used For | Stopwords Removed | Lemmatised |
|----------|----------|-------------------|------------|
| `ml_clean` | TF-IDF / BoW models | ✅ Yes | ✅ Yes |
| `dl_clean` | Embedding DL models | ❌ No | ❌ No |
| `transformer_clean` | Hugging Face Tokenizers | ❌ No | ❌ No |

All pipelines apply: encoding fix, HTML removal, URL removal, emoji handling, punctuation normalisation, whitespace normalisation.

---

## Label-Generation Rules

> ⚠️ **Important:** These are **derived (rule-based) labels**, not manually annotated ground truth. They are constructed from rating signals and keyword patterns.

### Severity Classification

```
Critical : Overall_Rating <= 2  OR  (Recommended = no AND critical keyword present)
High     : Overall_Rating in [3,4]  OR  >= 3 aspect ratings <= 2
Medium   : Overall_Rating in [5,6]
Low      : Overall_Rating >= 7  AND  Recommended = yes

Keyword upgrade (applies after base rule):
  Low/Medium + critical keyword → High
  High + critical keyword + Recommended=no → Critical
```

Critical keywords: `cancelled, lost baggage, refund denied, unsafe, rude staff, never again, stranded, nightmare, …`

### Risk Classification

```
risk_flag = 1 if:
  Recommended = no  AND
  Overall_Rating <= 4  AND
  (>= 2 aspect ratings <= 2  OR  risk keyword in text)

risk_flag = 0 otherwise
```

Risk keywords: `delay, cancellation, overbooking, refund, lost baggage, rude, unsafe, stranded, …`

---

## EDA

EDA covers:
- Severity distribution and airline-level heatmaps
- Overall rating distributions
- Aspect rating means by severity
- Review length distributions by severity
- Recommended vs Severity cross-tabulation
- Monthly/yearly complaint trends
- Word clouds (Low-severity vs Critical-severity)
- TF-IDF top keywords per severity class

---

## Feature Engineering

| Feature Set | Method | Notes |
|-------------|--------|-------|
| Bag-of-Words | `CountVectorizer` (unigrams) | 20,000 features |
| TF-IDF | `TfidfVectorizer` (unigrams+bigrams) | 30,000 features, sublinear_tf |
| TF-IDF + Structured | TF-IDF ⊕ aspect ratings + review_length + keyword_count | Scaler fit on train only |

**Data leakage prevention:** All vectorisers, imputers, and scalers are fit **only on training data**.

---

## ML Models

### Severity

| Model | Feature Sets |
|-------|-------------|
| Logistic Regression | BoW, TF-IDF, TF-IDF+Struct |
| Linear SVM (calibrated) | BoW, TF-IDF, TF-IDF+Struct |
| Multinomial Naive Bayes | BoW, TF-IDF |

### Risk

| Model | Feature Sets |
|-------|-------------|
| Logistic Regression | TF-IDF+Struct |
| Random Forest | TF-IDF+Struct |
| XGBoost | TF-IDF+Struct |

---

## Deep Learning Models

Architecture:
```
Input Text → SimpleTokenizer → Embedding(128d) → CNN/LSTM/BiLSTM → Dropout(0.4) → Dense → Output
```

| Model | Architecture |
|-------|-------------|
| TextCNN | Multi-kernel Conv1D (3,4,5), MaxPool |
| TextLSTM | 2-layer LSTM (hidden=256) |
| TextBiLSTM | 2-layer Bidirectional LSTM |

Training: class weights, early stopping (patience=3), Adam optimizer, ReduceLROnPlateau scheduler.

---

## Transformer Models

| Model | Task | Notes |
|-------|------|-------|
| `distilbert-base-uncased` | Severity (4-class) | **Mandatory** |
| `distilbert-base-uncased` | Risk (binary) | |
| `bert-base-uncased` | Severity (4-class) | Comparison |
| `bert-base-uncased` | Risk (binary) | Comparison |

Fine-tuning: Hugging Face Trainer API, dynamic padding, early stopping (patience=2), class weights, fp16 when GPU available.

---

## Evaluation Metrics

### Severity
- Accuracy
- Macro Precision / Recall / F1
- Confusion Matrix
- **Critical-class recall** (key business metric)

### Risk
- ROC-AUC
- PR-AUC
- Precision / Recall / F1
- Confusion Matrix
- **False Negative count and rate** (missed risky complaints = business loss)

---

## Model Comparison

A cross-model comparison table (`data/processed/model_comparison_table.csv`) is generated containing:

```
Model | Task | Model Family | Feature/Input | Accuracy | Macro F1 | Recall | ROC-AUC | Train Time | Inf Time
```

Final model selection considers: predictive performance, interpretability, deployment feasibility, inference cost, and business requirements.

---

## Explainability

| Level | Method | Models |
|-------|--------|--------|
| Keyword highlighting | Always available | All |
| Linear coefficients | TF-IDF feature importance | LR, SVM |
| LIME | Text perturbation explanations | Any predict_proba model |
| SHAP | Tree SHAP values | Random Forest, XGBoost |

Explanations identify: important words, complaint phrases, severity signals, risk signals.

> **Disclaimer:** Explanations highlight correlating features; they do not prove causal influence.

---

## Streamlit Application

### Page 1 — Introduction
- Project overview, problem statement, dataset summary, KPI metrics, architecture diagram

### Page 2 — EDA Dashboard
- Severity distribution, airline heatmap, aspect ratings by severity, temporal trends, recommendation vs severity, model comparison table

### Page 3 — Severity & Risk Prediction
- **Single Review:** Input title + text + optional ratings → Severity label + confidence + Risk flag + probability + complaint phrase highlights
- **Bulk CSV:** Upload CSV → Download predictions CSV

---

## Project Structure

```
airline-complaint-ai/
├── app.py                          # Streamlit entry point
├── train_pipeline.py               # Master training script
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/                        # Original Airline_review.csv
│   └── processed/                  # Cleaned parquet + model comparison CSV
│
├── notebooks/                      # Jupyter notebooks for each phase
│
├── src/
│   ├── data/
│   │   ├── data_loader.py          # Phase 1: Inspection + loading
│   │   ├── cleaner.py              # Phase 2: Structured cleaning
│   │   └── label_generator.py     # Phase 4: Severity + risk labels
│   ├── preprocessing/
│   │   └── text_preprocessor.py   # Phase 3: Three NLP pipelines
│   ├── features/
│   │   └── feature_engineer.py    # Phase 6: BoW, TF-IDF, combined
│   ├── models/
│   │   ├── ml_trainer.py          # Phase 7: Traditional ML
│   │   ├── dl_trainer.py          # Phase 8: CNN, LSTM, BiLSTM
│   │   └── transformer_trainer.py # Phase 9: DistilBERT, BERT
│   ├── evaluation/
│   │   └── evaluator.py           # Phase 10: Metrics + comparison table
│   ├── explainability/
│   │   └── explainer.py           # LIME, SHAP, keyword highlights
│   └── utils/
│       ├── config.py              # Global constants and paths
│       └── logger.py             # Centralised logging
│
├── pages/
│   ├── 1_Introduction.py
│   ├── 2_EDA_Dashboard.py
│   └── 3_Prediction.py
│
├── models/
│   ├── ml/                         # Saved .pkl models
│   ├── dl/                         # Saved .pt PyTorch models
│   └── transformers/               # Fine-tuned Transformer checkpoints
│
├── artifacts/
│   ├── vectorizers/                # TF-IDF, BoW, scaler
│   └── encoders/                   # Label encoders
│
├── tests/
│   └── test_data.py               # Unit tests
│
└── deployment/
    ├── aws/README_aws.md
    └── huggingface/README_hf.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd airline-complaint-ai

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 5. Place the dataset
# Copy Airline_review.csv to: data/raw/Airline_review.csv

# 6. Create your .env file (for deployment)
cp .env.example .env
# Edit .env with your actual values
```

---

## Usage

### Run the Streamlit Application

```bash
streamlit run app.py
```

Navigate to: http://localhost:8501

The app works in **demo mode** even without trained models — it uses rule-based fallback for predictions.

---

## Training Instructions

### Quick test run (ML only, 2000 rows sample)

```bash
python train_pipeline.py --skip-dl --skip-transformers --sample 2000
```

### Full ML + DL (no Transformers)

```bash
python train_pipeline.py --skip-transformers
```

### Full pipeline (requires GPU for Transformers)

```bash
python train_pipeline.py
```

Training artifacts are saved to:
- `models/ml/` — Scikit-learn pickled models
- `models/dl/` — PyTorch `.pt` model weights
- `models/transformers/` — Hugging Face model checkpoints
- `artifacts/vectorizers/` — TF-IDF, BoW vectorisers and scaler
- `data/processed/model_comparison_table.csv` — Results table

---

## Model Artifacts

After training, the following files should exist:

```
artifacts/vectorizers/tfidf_vectorizer.pkl
artifacts/vectorizers/bow_vectorizer.pkl
artifacts/vectorizers/struct_scaler.pkl
artifacts/vectorizers/dl_tokenizer.pkl
models/ml/severity_Logistic_Regression_tfidf.pkl
models/ml/severity_Linear_SVM_tfidf.pkl
models/ml/risk_Logistic_Regression_tfidf_struct.pkl
models/ml/risk_Random_Forest_tfidf_struct.pkl
models/dl/CNN_best.pt
models/dl/LSTM_best.pt
models/dl/BiLSTM_best.pt
models/transformers/severity_distilbert-base-uncased/best_model/
models/transformers/risk_distilbert-base-uncased/best_model/
```

---

## AWS Deployment

See [deployment/aws/README_aws.md](deployment/aws/README_aws.md) for full instructions.

**Never commit AWS credentials.** Use IAM roles or AWS Secrets Manager.

---

## Hugging Face Deployment

See [deployment/huggingface/README_hf.md](deployment/huggingface/README_hf.md) for full instructions.

Load models from the HF Hub to avoid large files in the Space repo.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run only data tests
pytest tests/test_data.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

Tests cover:
- Column validation and data type conversion
- Missing-value handling and imputation
- Duplicate removal
- Severity rule correctness (boundary conditions)
- Risk flag rule correctness
- Text preprocessing (lowercasing, stopwords, HTML removal)
- Keyword detection
- Explainability output structure

---

## Limitations

1. **Derived labels:** Severity and risk labels are rule-based, not human-annotated. They may not perfectly reflect true complaint severity in all cases.
2. **Target leakage risk:** `Overall_Rating` and `Recommended` are used both in label generation and as potential model features. Models trained with these features may learn circular patterns. Document carefully.
3. **High missing rates:** `Wifi & Connectivity` (74.5% missing), `Inflight Entertainment` (53.3% missing) — median imputation is an approximation.
4. **Class imbalance:** The severity distribution is uneven; Critical class recall is harder to optimise.
5. **Transformer resource requirements:** DistilBERT and BERT fine-tuning require significant GPU memory and time.
6. **No ground truth:** Without manually verified labels, evaluation metrics reflect self-consistency, not true annotation accuracy.

---

## Future Improvements

- [ ] Human annotation of a gold-standard benchmark subset
- [ ] Aspect-level sentiment analysis (ABSA) per service category
- [ ] Online learning / incremental retraining on new reviews
- [ ] Confidence calibration for all model families
- [ ] Multi-lingual support for non-English reviews
- [ ] Named entity recognition for airline/route/aircraft extraction
- [ ] Integration with airline CRM / ticketing systems
- [ ] Active learning interface for human-in-the-loop label correction
- [ ] Time-series anomaly detection for sudden complaint spikes
- [ ] Severity trend forecasting per airline

---

## Reproducibility

```
Python    : 3.10+
Random Seed: 42 (set for Python, NumPy, scikit-learn, PyTorch)
Dataset   : Airline_review.csv (23,171 rows, 20 columns)
```

All preprocessing artifacts are saved with their corresponding models to ensure consistent train/inference behaviour.
