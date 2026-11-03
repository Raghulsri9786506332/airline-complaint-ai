"""
dl_trainer.py — Phase 8: Custom Deep Learning models (CNN, LSTM, BiLSTM).

Architecture:
  Input → Embedding → CNN/LSTM/BiLSTM → Dropout → Dense → Output

Optional fusion:
  Text representation + structured features → Concatenation → Dense → Output
"""
from __future__ import annotations

import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from typing import Optional
import joblib

from src.utils.config import (
    RANDOM_SEED, MODELS_DL, MAX_LEN_DL, BATCH_SIZE, DL_EPOCHS, LOGS_DIR,
)
from src.utils.logger import get_logger

log = get_logger("dl_trainer", LOGS_DIR)

# ── Seeds ────────────────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────

class ReviewDataset(Dataset):
    """PyTorch Dataset for tokenised airline reviews."""

    def __init__(
        self,
        token_ids: list[list[int]],
        labels: list[int],
        struct_features: Optional[np.ndarray] = None,
    ):
        self.token_ids = token_ids
        self.labels = labels
        self.struct_features = struct_features

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids": torch.tensor(self.token_ids[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.struct_features is not None:
            item["struct"] = torch.tensor(
                self.struct_features[idx], dtype=torch.float32
            )
        return item


def _collate_fn(batch: list[dict]) -> dict:
    """Pad sequences to max length in the batch."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.stack([item["label"] for item in batch])
    for i, item in enumerate(batch):
        seq = item["input_ids"]
        padded[i, : len(seq)] = seq
    out = {"input_ids": padded, "labels": labels}
    if "struct" in batch[0]:
        out["struct"] = torch.stack([item["struct"] for item in batch])
    return out


# ── Simple Tokenizer ─────────────────────────────────────────────────────────

class SimpleTokenizer:
    """Character/word-level vocabulary-based tokenizer."""

    def __init__(self, max_vocab: int = 50_000):
        self.max_vocab = max_vocab
        self.word2idx: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: dict[int, str] = {0: "<PAD>", 1: "<UNK>"}

    def build_vocab(self, texts: list[str]) -> None:
        from collections import Counter
        counter: Counter = Counter()
        for text in texts:
            counter.update(text.split())
        for word, _ in counter.most_common(self.max_vocab - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        log.info(f"Vocabulary built: {len(self.word2idx)} tokens")

    def encode(self, text: str, max_len: int = MAX_LEN_DL) -> list[int]:
        tokens = [
            self.word2idx.get(t, 1) for t in text.split()[:max_len]
        ]
        # Pad
        tokens += [0] * (max_len - len(tokens))
        return tokens

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "SimpleTokenizer":
        return joblib.load(path)


# ── Model architectures ───────────────────────────────────────────────────────

class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int,
                 n_filters: int = 128, kernel_sizes=(3, 4, 5),
                 struct_dim: int = 0, dropout: float = 0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        fc_in = n_filters * len(kernel_sizes) + struct_dim
        self.fc = nn.Linear(fc_in, num_classes)

    def forward(self, x: torch.Tensor, struct: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(x).permute(0, 2, 1)  # (B, E, L)
        pooled = [torch.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        out = torch.cat(pooled, dim=1)
        if struct is not None:
            out = torch.cat([out, struct], dim=1)
        return self.fc(self.dropout(out))


class TextLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int,
                 hidden_dim: int = 256, n_layers: int = 2,
                 bidirectional: bool = False, struct_dim: int = 0,
                 dropout: float = 0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        directions = 2 if bidirectional else 1
        fc_in = hidden_dim * directions + struct_dim
        self.fc = nn.Linear(fc_in, num_classes)

    def forward(self, x: torch.Tensor, struct: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))
        _, (h, _) = self.lstm(emb)
        # Concat last layer forward + backward
        if self.lstm.bidirectional:
            out = torch.cat([h[-2], h[-1]], dim=1)
        else:
            out = h[-1]
        if struct is not None:
            out = torch.cat([out, struct], dim=1)
        return self.fc(self.dropout(out))


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        struct = batch.get("struct")
        if struct is not None:
            struct = struct.to(DEVICE)
        optimizer.zero_grad()
        logits = model(ids, struct)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            struct = batch.get("struct")
            if struct is not None:
                struct = struct.to(DEVICE)
            logits = model(ids, struct)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels


def train_dl_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    vocab_size: int,
    embed_dim: int = 128,
    epochs: int = DL_EPOCHS,
    class_weights: Optional[torch.Tensor] = None,
    struct_dim: int = 0,
    patience: int = 3,
) -> tuple[nn.Module, list[dict]]:
    """Train a DL model; return (model, history)."""
    log.info(f"Training DL model: {model_name}")

    if model_name == "CNN":
        model = TextCNN(vocab_size, embed_dim, num_classes, struct_dim=struct_dim)
    elif model_name == "LSTM":
        model = TextLSTM(vocab_size, embed_dim, num_classes,
                         bidirectional=False, struct_dim=struct_dim)
    elif model_name == "BiLSTM":
        model = TextLSTM(vocab_size, embed_dim, num_classes,
                         bidirectional=True, struct_dim=struct_dim)
    else:
        raise ValueError(f"Unknown DL model: {model_name}")

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE) if class_weights is not None else None)
    optimizer = Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=True)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_acc": val_acc})
        log.info(
            f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"time={time.time()-t0:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stopping at epoch {epoch}.")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Save
    save_path = MODELS_DL / f"{model_name}_best.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    log.info(f"DL model saved: {save_path}")

    return model, history
