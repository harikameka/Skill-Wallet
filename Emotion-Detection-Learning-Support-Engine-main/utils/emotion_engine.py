"""
Emotion Engine — orchestrates the full prediction pipeline:

  raw text -> preprocessing -> (BiLSTM and/or BERT) -> rule-based keyword
  enhancement -> mixed-emotion detection -> result dict

This is the single entry point the Streamlit app calls.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import (
    EMOTIONS, MIXED_EMOTION_MARGIN, BILSTM_WEIGHTS, BILSTM_VOCAB,
    BILSTM_CONFIG, BERT_MODEL_DIR, BERT_CONFIG,
)
from models.rule_based import enhance_predictions
from models.bilstm_model import BiLSTMPredictor
from utils.preprocessing import clean_text


@dataclass
class EmotionResult:
    model_used: str
    probabilities: Dict[str, float]
    primary_emotion: str
    primary_confidence: float
    secondary_emotion: Optional[str] = None
    secondary_confidence: Optional[float] = None
    is_mixed: bool = False


class EmotionEngine:
    """Lazily loads models on first use and caches them in memory."""

    def __init__(self):
        self._bilstm: Optional[BiLSTMPredictor] = None
        self._bert = None
        self._bilstm_load_error: Optional[str] = None
        self._bert_load_error: Optional[str] = None

    # -- availability checks ------------------------------------------------
    def bilstm_available(self) -> bool:
        return BiLSTMPredictor.is_available(BILSTM_WEIGHTS, BILSTM_VOCAB)

    def bert_available(self) -> bool:
        from models.bert_model import BertPredictor
        return BertPredictor.is_available(BERT_MODEL_DIR)

    # -- lazy loaders ---------------------------------------------------------
    def _get_bilstm(self) -> Optional[BiLSTMPredictor]:
        if self._bilstm is not None:
            return self._bilstm
        if not self.bilstm_available():
            return None
        try:
            self._bilstm = BiLSTMPredictor(
                BILSTM_WEIGHTS, BILSTM_VOCAB, EMOTIONS, BILSTM_CONFIG,
            )
        except Exception as exc:  # noqa: BLE001
            self._bilstm_load_error = str(exc)
            return None
        return self._bilstm

    def _get_bert(self):
        if self._bert is not None:
            return self._bert
        if not self.bert_available():
            return None
        try:
            from models.bert_model import BertPredictor
            self._bert = BertPredictor(
                BERT_MODEL_DIR, EMOTIONS, max_len=BERT_CONFIG["max_len"],
            )
        except Exception as exc:  # noqa: BLE001
            self._bert_load_error = str(exc)
            return None
        return self._bert

    # -- core prediction ------------------------------------------------------
    def _raw_predict(self, text: str, model_name: str) -> Optional[Dict[str, float]]:
        text = clean_text(text)
        if model_name == "BiLSTM":
            predictor = self._get_bilstm()
        elif model_name == "BERT":
            predictor = self._get_bert()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if predictor is None:
            return None
        return predictor.predict(text)

    def predict(self, text: str, model_name: str = "BiLSTM",
                use_rule_enhancement: bool = True) -> Optional[EmotionResult]:
        """
        Run full pipeline for a single model. Returns None if the requested
        model isn't trained/available yet.
        """
        probs = self._raw_predict(text, model_name)
        if probs is None:
            return None

        if use_rule_enhancement:
            probs = enhance_predictions(probs, text)

        return self._build_result(model_name, probs)

    def predict_both(self, text: str, use_rule_enhancement: bool = True) -> Dict[str, Optional[EmotionResult]]:
        """Run both models for side-by-side comparison."""
        return {
            "BiLSTM": self.predict(text, "BiLSTM", use_rule_enhancement),
            "BERT": self.predict(text, "BERT", use_rule_enhancement),
        }

    @staticmethod
    def _build_result(model_name: str, probs: Dict[str, float]) -> EmotionResult:
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        primary_emotion, primary_conf = ranked[0]

        secondary_emotion, secondary_conf, is_mixed = None, None, False
        if len(ranked) > 1:
            runner_up_emotion, runner_up_conf = ranked[1]
            if (primary_conf - runner_up_conf) <= MIXED_EMOTION_MARGIN:
                secondary_emotion, secondary_conf = runner_up_emotion, runner_up_conf
                is_mixed = True

        return EmotionResult(
            model_used=model_name,
            probabilities=probs,
            primary_emotion=primary_emotion,
            primary_confidence=primary_conf,
            secondary_emotion=secondary_emotion,
            secondary_confidence=secondary_conf,
            is_mixed=is_mixed,
        )


# Module-level singleton so Streamlit's rerun cycle doesn't reload models
# every interaction (paired with st.cache_resource in app.py).
def get_engine() -> EmotionEngine:
    return EmotionEngine()
