"""
transformer_trainer.py — Phase 9: Hugging Face Transformer fine-tuning.

Primary model  : distilbert-base-uncased (mandatory)
Comparison     : bert-base-uncased

Tasks:
  - severity (4-class)
  - risk (binary)
"""
from __future__ import annotations

import os
import random
import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from datasets import Dataset as HFDataset
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
)

from src.utils.config import (
    RANDOM_SEED, MODELS_TRANSFORMER, MAX_LEN_TRANSFORMER,
    TRANSFORMER_MODEL_NAME, BERT_MODEL_NAME, LOGS_DIR,
)
from src.utils.logger import get_logger

log = get_logger("transformer_trainer", LOGS_DIR)

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Tokenisation ──────────────────────────────────────────────────────────────

def tokenize_data(
    texts: list[str],
    labels: list[int],
    model_name: str,
    max_len: int = MAX_LEN_TRANSFORMER,
) -> HFDataset:
    """Tokenise texts and return a Hugging Face Dataset."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenise(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding=False,  # dynamic padding done by DataCollator
            max_length=max_len,
        )
        return enc

    hf_ds = HFDataset.from_dict({"text": texts, "label": labels})
    hf_ds = hf_ds.map(_tokenise, batched=True, remove_columns=["text"])
    return hf_ds, tokenizer


# ── Metrics ───────────────────────────────────────────────────────────────────

def _severity_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


def _risk_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    return {
        "roc_auc": roc_auc_score(labels, probs),
        "f1": f1_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
    }


# ── Fine-tuning ────────────────────────────────────────────────────────────────

def fine_tune_transformer(
    model_name: str,
    task: str,
    train_ds: HFDataset,
    val_ds: HFDataset,
    tokenizer,
    num_labels: int,
    class_weights: Optional[list[float]] = None,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
) -> tuple:
    """
    Fine-tune a Transformer model for classification.

    Args:
        model_name  : HuggingFace model identifier
        task        : 'severity' or 'risk'
        train_ds    : Training HFDataset
        val_ds      : Validation HFDataset
        tokenizer   : Matching tokenizer
        num_labels  : Number of output classes
        class_weights: Optional per-class weight list
        epochs      : Number of training epochs
        batch_size  : Batch size
        lr          : Learning rate

    Returns:
        (trainer, model, metrics_dict)
    """
    log.info(f"Fine-tuning {model_name} for task={task}, num_labels={num_labels}")

    output_dir = MODELS_TRANSFORMER / f"{task}_{model_name.replace('/', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True,
    ).to(DEVICE)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    compute_metrics = _severity_metrics if task == "severity" else _risk_metrics

    # Custom Trainer for class weighting
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if class_weights is not None:
                weights = torch.tensor(class_weights, dtype=torch.float).to(logits.device)
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1" if task == "severity" else "f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=RANDOM_SEED,
        logging_dir=str(LOGS_DIR),
        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    log.info(f"{model_name} training complete in {elapsed:.1f}s")

    # Evaluate
    metrics = trainer.evaluate()
    log.info(f"Eval metrics: {metrics}")

    # Save
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))
    log.info(f"Model saved: {output_dir / 'best_model'}")

    return trainer, model, metrics


# ── Inference helper ──────────────────────────────────────────────────────────

def load_transformer_for_inference(model_dir: str | Path, num_labels: int):
    """Load a fine-tuned transformer model and tokenizer for inference."""
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), num_labels=num_labels
    )
    model.eval()
    model.to(DEVICE)
    return model, tokenizer


def predict_transformer(
    texts: list[str],
    model,
    tokenizer,
    max_len: int = MAX_LEN_TRANSFORMER,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a list of texts.
    Returns (predicted_class_ids, probabilities).
    """
    inputs = tokenizer(
        texts, truncation=True, padding=True, max_length=max_len,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = np.argmax(probs, axis=1)
    return preds, probs
