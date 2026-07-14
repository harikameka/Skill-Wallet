"""
Central configuration for the AI-Driven Emotion Detection & Personalized
Learning Support Platform.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SAVED_DIR = MODELS_DIR / "saved"
BILSTM_DIR = SAVED_DIR / "bilstm"
BERT_DIR = SAVED_DIR / "bert"
ASSETS_DIR = BASE_DIR / "assets"

TRAINING_DATA_CSV = DATA_DIR / "training_data.csv"
INTERACTION_LOG_CSV = DATA_DIR / "interaction_logs.csv"

BILSTM_WEIGHTS = BILSTM_DIR / "bilstm_model.pt"
BILSTM_VOCAB = BILSTM_DIR / "vocab.json"
BERT_MODEL_DIR = BERT_DIR  # transformers save_pretrained target

# ---------------------------------------------------------------------------
# Emotion labels
# ---------------------------------------------------------------------------
EMOTIONS = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

EMOTION_META = {
    "Bored":      {"emoji": "😴", "color": "#8A8FA3"},
    "Confident":  {"emoji": "💪", "color": "#2ECC71"},
    "Confused":   {"emoji": "😕", "color": "#F39C12"},
    "Curious":    {"emoji": "🤔", "color": "#3498DB"},
    "Frustrated": {"emoji": "😤", "color": "#E74C3C"},
}

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
BILSTM_CONFIG = {
    "embedding_dim": 128,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "max_len": 40,
    "min_freq": 1,
    "batch_size": 16,
    "epochs": 18,
    "lr": 1e-3,
}

BERT_CONFIG = {
    "base_model": "bert-base-uncased",
    "max_len": 48,
    "batch_size": 8,
    "epochs": 4,
    "lr": 2e-5,
}

# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = "gemini-1.5-flash"

# Mixed-emotion detection: if the runner-up emotion's probability is within
# this margin of the top prediction, we report it as a secondary emotion.
MIXED_EMOTION_MARGIN = 0.15
