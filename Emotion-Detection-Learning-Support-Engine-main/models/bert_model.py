"""
BERT-based emotion classifier (HuggingFace Transformers).

Fine-tunes bert-base-uncased on the project's labeled dataset for 5-way
emotion classification. This is the "deep" / high-accuracy model in the
dual-model comparison view.
"""
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BertPredictor:
    """Convenience wrapper for loading a fine-tuned BERT model for inference."""

    def __init__(self, model_dir: Path, emotions: List[str],
                 max_len: int = 48, device: str = "cpu"):
        self.emotions = emotions
        self.max_len = max_len
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, float]:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding="max_length", max_length=self.max_len,
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        return {emotion: float(p) for emotion, p in zip(self.emotions, probs)}

    @classmethod
    def is_available(cls, model_dir: Path) -> bool:
        return (model_dir / "config.json").exists()
