"""Lightweight text cleaning utilities shared across models."""
import re


def clean_text(text: str) -> str:
    """Basic normalization: strip whitespace, collapse spaces."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_valid_input(text: str, min_chars: int = 3) -> bool:
    return bool(text) and len(text.strip()) >= min_chars
