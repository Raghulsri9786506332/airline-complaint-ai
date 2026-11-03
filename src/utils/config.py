"""
config.py — Global project configuration.
All constants, paths, and hyperparameters live here.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_RAW = BASE_DIR / "data" / "raw" / "Airline_review.csv"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_ML = BASE_DIR / "models" / "ml"
MODELS_DL = BASE_DIR / "models" / "dl"
MODELS_TRANSFORMER = BASE_DIR / "models" / "transformers"
ARTIFACTS_VEC = BASE_DIR / "artifacts" / "vectorizers"
ARTIFACTS_ENC = BASE_DIR / "artifacts" / "encoders"
LOGS_DIR = BASE_DIR / "logs"

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Label Mappings ─────────────────────────────────────────────────────────
SEVERITY_LABELS = ["Low", "Medium", "High", "Critical"]
SEVERITY_ID_MAP = {label: idx for idx, label in enumerate(SEVERITY_LABELS)}
SEVERITY_ID_TO_LABEL = {v: k for k, v in SEVERITY_ID_MAP.items()}

RISK_LABELS = {0: "No Risk", 1: "Risk"}

# ── Rating Columns ─────────────────────────────────────────────────────────
RATING_COLS = [
    "Seat Comfort",
    "Cabin Staff Service",
    "Food & Beverages",
    "Ground Service",
    "Inflight Entertainment",
    "Wifi & Connectivity",
    "Value For Money",
]

# ── Complaint Keywords ─────────────────────────────────────────────────────
CRITICAL_KEYWORDS = [
    "cancelled", "cancel", "lost baggage", "refund denied", "unsafe",
    "rude staff", "never again", "terrible", "worst", "awful",
    "disgusting", "horrible", "scam", "theft", "stranded", "nightmare",
]

RISK_KEYWORDS = [
    "delay", "delayed", "cancellation", "cancelled", "overbooking",
    "overbooked", "refund", "lost baggage", "missing luggage",
    "rude", "unsafe", "stranded", "no show", "compensation denied",
]

# ── Model / Training ───────────────────────────────────────────────────────
TEST_SIZE = 0.20
VAL_SIZE = 0.10
MAX_LEN_DL = 256          # Max token length for DL models
MAX_LEN_TRANSFORMER = 256  # Max token length for Transformers
BATCH_SIZE = 32
DL_EPOCHS = 10
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
BERT_MODEL_NAME = "bert-base-uncased"

# ── TF-IDF ─────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 30_000
TFIDF_NGRAM_RANGE = (1, 2)
BOW_MAX_FEATURES = 20_000
