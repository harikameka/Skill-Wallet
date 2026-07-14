"""
BiLSTM-based emotion classifier (PyTorch).

A lightweight, fast, fully-custom deep learning model that is trained from
scratch on the project's labeled dataset. Serves as the "quick" model in the
dual-model comparison view (BiLSTM vs BERT).
"""
import json
import re
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.split()


class Vocabulary:
    def __init__(self):
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN}

    def build(self, texts: List[str], min_freq: int = 1):
        freq = {}
        for t in texts:
            for tok in tokenize(t):
                freq[tok] = freq.get(tok, 0) + 1
        for tok, count in freq.items():
            if count >= min_freq and tok not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[tok] = idx
                self.idx2word[idx] = tok

    def encode(self, text: str, max_len: int) -> List[int]:
        tokens = tokenize(text)[:max_len]
        ids = [self.word2idx.get(tok, self.word2idx[UNK_TOKEN]) for tok in tokens]
        if len(ids) < max_len:
            ids += [self.word2idx[PAD_TOKEN]] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            vocab.word2idx = json.load(f)
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        return vocab


class BiLSTMEmotionClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int,
                 embedding_dim: int = 128, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)                      # [B, T, E]
        lstm_out, _ = self.lstm(embedded)                  # [B, T, 2H]

        # simple additive attention pooling over time steps
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)  # [B, T]
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)   # [B, 2H]

        out = self.dropout(context)
        logits = self.fc(out)
        return logits


class BiLSTMPredictor:
    """Convenience wrapper for loading and running inference."""

    def __init__(self, model_path: Path, vocab_path: Path, emotions: List[str],
                 config: dict, device: str = "cpu"):
        self.emotions = emotions
        self.device = torch.device(device)
        self.vocab = Vocabulary.load(vocab_path)
        self.max_len = config["max_len"]

        self.model = BiLSTMEmotionClassifier(
            vocab_size=len(self.vocab),
            num_classes=len(emotions),
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, float]:
        ids = self.vocab.encode(text, self.max_len)
        tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        return {emotion: float(p) for emotion, p in zip(self.emotions, probs)}

    @classmethod
    def is_available(cls, model_path: Path, vocab_path: Path) -> bool:
        return model_path.exists() and vocab_path.exists()
