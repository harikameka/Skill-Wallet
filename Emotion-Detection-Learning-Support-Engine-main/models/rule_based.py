"""
Rule-based keyword emotion enhancer.

This module does NOT replace the deep learning models — it nudges / sanity
checks their output using hand-crafted lexical cues. This is a common
pattern in production emotion-AI systems: use a fast, transparent
rule layer to catch obvious cases and correct low-confidence model calls.
"""
import re
from typing import Dict

KEYWORDS = {
    "Bored": [
        "bored", "boring", "dull", "monotonous", "tedious", "same old",
        "zoning out", "uninterested", "nothing new", "repetitive",
        "can't focus", "losing interest", "meh",
    ],
    "Confident": [
        "confident", "i understand", "i got this", "makes sense now",
        "i can do this", "prepared", "nailed", "aced", "strong", "ready",
        "i know this", "clicked", "capable",
    ],
    "Confused": [
        "confused", "lost", "don't understand", "don't get it", "unclear",
        "no idea", "stuck", "doesn't make sense", "mixing up",
        "not sure how", "what does this mean", "can't follow",
    ],
    "Curious": [
        "curious", "wonder", "interested to know", "what if", "explore",
        "intrigued", "how does", "why does", "fascinating", "want to learn",
        "tell me more", "dig into",
    ],
    "Frustrated": [
        "frustrated", "annoyed", "angry", "give up", "hate this",
        "sick of", "infuriating", "stressed", "keep failing", "driving me crazy",
        "nothing works", "so hard", "exhausted", "irritated",
    ],
}

# Precompile regex patterns for speed
_COMPILED = {
    emotion: [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
              if " " not in kw else re.compile(re.escape(kw), re.IGNORECASE)
              for kw in kws]
    for emotion, kws in KEYWORDS.items()
}


def keyword_scores(text: str) -> Dict[str, float]:
    """
    Return a normalized 0-1 score per emotion based on keyword hits.
    A text with no keyword matches returns all zeros.
    """
    scores = {e: 0 for e in KEYWORDS}
    for emotion, patterns in _COMPILED.items():
        for pat in patterns:
            if pat.search(text):
                scores[emotion] += 1

    total = sum(scores.values())
    if total == 0:
        return {e: 0.0 for e in KEYWORDS}
    return {e: v / total for e, v in scores.items()}


def enhance_predictions(model_probs: Dict[str, float], text: str,
                         keyword_weight: float = 0.25) -> Dict[str, float]:
    """
    Blend model probabilities with rule-based keyword scores.

    keyword_weight: how much influence (0-1) the keyword layer has.
    """
    kw = keyword_scores(text)
    if sum(kw.values()) == 0:
        return model_probs  # no lexical signal, trust the model fully

    blended = {}
    for emotion in model_probs:
        blended[emotion] = (
            (1 - keyword_weight) * model_probs[emotion]
            + keyword_weight * kw.get(emotion, 0.0)
        )

    # renormalize
    total = sum(blended.values()) or 1.0
    return {e: v / total for e, v in blended.items()}
