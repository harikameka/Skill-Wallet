"""
Fine-tune bert-base-uncased for 5-way emotion classification on
data/training_data.csv.

Usage:
    python training/train_bert.py

Note: requires an internet connection the first time (to download the base
bert-base-uncased weights from HuggingFace) and a GPU is recommended but not
required for this small dataset.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments,
)

from config import TRAINING_DATA_CSV, BERT_MODEL_DIR, EMOTIONS, BERT_CONFIG


class TorchTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_len,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def main():
    print("Loading dataset...")
    df = pd.read_csv(TRAINING_DATA_CSV)
    label2idx = {e: i for i, e in enumerate(EMOTIONS)}
    df["label_id"] = df["emotion"].map(label2idx)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(), df["label_id"].tolist(),
        test_size=0.15, random_state=42, stratify=df["label_id"],
    )

    print(f"Loading base model: {BERT_CONFIG['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_CONFIG["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_CONFIG["base_model"], num_labels=len(EMOTIONS),
    )

    max_len = BERT_CONFIG["max_len"]
    train_ds = TorchTextDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds = TorchTextDataset(val_texts, val_labels, tokenizer, max_len)

    BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(BERT_MODEL_DIR / "checkpoints"),
        num_train_epochs=BERT_CONFIG["epochs"],
        per_device_train_batch_size=BERT_CONFIG["batch_size"],
        per_device_eval_batch_size=BERT_CONFIG["batch_size"],
        learning_rate=BERT_CONFIG["lr"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning...")
    trainer.train()

    metrics = trainer.evaluate()
    print(f"Final validation accuracy: {metrics['eval_accuracy']:.4f}")

    print(f"Saving fine-tuned model to {BERT_MODEL_DIR} ...")
    model.save_pretrained(BERT_MODEL_DIR)
    tokenizer.save_pretrained(BERT_MODEL_DIR)

    # persist label mapping alongside the model
    with open(BERT_MODEL_DIR / "labels.txt", "w") as f:
        f.write("\n".join(EMOTIONS))

    print("Done.")


if __name__ == "__main__":
    main()
