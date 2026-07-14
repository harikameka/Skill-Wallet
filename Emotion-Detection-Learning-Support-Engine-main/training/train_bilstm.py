"""
Train the BiLSTM emotion classifier from scratch on data/training_data.csv.

Usage:
    python training/train_bilstm.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from config import (
    TRAINING_DATA_CSV, BILSTM_WEIGHTS, BILSTM_VOCAB, BILSTM_DIR,
    EMOTIONS, BILSTM_CONFIG,
)
from models.bilstm_model import BiLSTMEmotionClassifier, Vocabulary


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocabulary, max_len: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.vocab.encode(self.texts[idx], self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def main():
    print("Loading dataset...")
    df = pd.read_csv(TRAINING_DATA_CSV)
    label2idx = {e: i for i, e in enumerate(EMOTIONS)}
    df["label_id"] = df["emotion"].map(label2idx)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(), df["label_id"].tolist(),
        test_size=0.15, random_state=42, stratify=df["label_id"],
    )

    print("Building vocabulary...")
    vocab = Vocabulary()
    vocab.build(train_texts, min_freq=BILSTM_CONFIG["min_freq"])
    print(f"Vocab size: {len(vocab)}")

    max_len = BILSTM_CONFIG["max_len"]
    train_ds = EmotionDataset(train_texts, train_labels, vocab, max_len)
    val_ds = EmotionDataset(val_texts, val_labels, vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=BILSTM_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BILSTM_CONFIG["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMEmotionClassifier(
        vocab_size=len(vocab),
        num_classes=len(EMOTIONS),
        embedding_dim=BILSTM_CONFIG["embedding_dim"],
        hidden_dim=BILSTM_CONFIG["hidden_dim"],
        num_layers=BILSTM_CONFIG["num_layers"],
        dropout=BILSTM_CONFIG["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=BILSTM_CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    BILSTM_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, BILSTM_CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_ds)

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total if total else 0.0

        print(f"Epoch {epoch:02d}/{BILSTM_CONFIG['epochs']} "
              f"- train_loss: {train_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BILSTM_WEIGHTS)
            vocab.save(BILSTM_VOCAB)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {BILSTM_WEIGHTS}")
    print(f"Vocab saved to: {BILSTM_VOCAB}")


if __name__ == "__main__":
    main()
