"""CSV-based interaction logging for analytics and continuous learning."""
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config import INTERACTION_LOG_CSV

_FIELDS = [
    "timestamp", "text", "model_used",
    "primary_emotion", "primary_confidence",
    "secondary_emotion", "secondary_confidence",
    "gemini_used",
]


def _ensure_log_exists():
    if not INTERACTION_LOG_CSV.exists():
        INTERACTION_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(INTERACTION_LOG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()


def log_interaction(
    text: str,
    model_used: str,
    primary_emotion: str,
    primary_confidence: float,
    secondary_emotion: Optional[str] = None,
    secondary_confidence: Optional[float] = None,
    gemini_used: bool = False,
):
    _ensure_log_exists()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "text": text,
        "model_used": model_used,
        "primary_emotion": primary_emotion,
        "primary_confidence": round(primary_confidence, 4),
        "secondary_emotion": secondary_emotion or "",
        "secondary_confidence": round(secondary_confidence, 4) if secondary_confidence else "",
        "gemini_used": gemini_used,
    }
    with open(INTERACTION_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writerow(row)


def load_logs() -> pd.DataFrame:
    _ensure_log_exists()
    if INTERACTION_LOG_CSV.stat().st_size == 0:
        return pd.DataFrame(columns=_FIELDS)
    df = pd.read_csv(INTERACTION_LOG_CSV)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
