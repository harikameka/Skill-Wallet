"""Model Comparison — run the same text through BiLSTM and BERT side by side."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import EMOTIONS, EMOTION_META
from utils.emotion_engine import EmotionEngine
from utils.preprocessing import is_valid_input

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")


def load_css():
    try:
        with open(Path(__file__).resolve().parent.parent / "assets" / "style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


load_css()


@st.cache_resource(show_spinner=False)
def get_engine() -> EmotionEngine:
    return EmotionEngine()


engine = get_engine()

st.markdown(
    """
    <div class="hero-card">
        <h1>⚖️ Model Comparison</h1>
        <p>Run the same student message through BiLSTM and BERT to compare
        predictions, confidence, and agreement.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

bilstm_ok = engine.bilstm_available()
bert_ok = engine.bert_available()

if not (bilstm_ok and bert_ok):
    st.warning(
        "Both models need to be trained for a full comparison.\n\n"
        f"- BiLSTM: {'✅ ready' if bilstm_ok else '⚠️ run `python training/train_bilstm.py`'}\n"
        f"- BERT: {'✅ ready' if bert_ok else '⚠️ run `python training/train_bert.py`'}"
    )

text = st.text_area(
    "Enter a student message to compare",
    placeholder="e.g. I don't understand recursion at all, I've tried three times.",
    height=120,
)
run = st.button("⚖️ Compare Models", use_container_width=True)

if run:
    if not is_valid_input(text):
        st.warning("Please enter at least a few words.")
    else:
        with st.spinner("Running both models..."):
            results = engine.predict_both(text)

        cols = st.columns(2)
        for col, model_name in zip(cols, ["BiLSTM", "BERT"]):
            result = results[model_name]
            with col:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="section-title">🧠 {model_name}</div>', unsafe_allow_html=True)
                if result is None:
                    st.error(f"{model_name} model is not trained yet.")
                else:
                    meta = EMOTION_META[result.primary_emotion]
                    st.markdown(
                        f'<span class="emotion-badge" style="background:{meta["color"]}">'
                        f'{meta["emoji"]} {result.primary_emotion} · {result.primary_confidence:.0%}</span>',
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    for emotion, prob in sorted(result.probabilities.items(), key=lambda kv: kv[1], reverse=True):
                        st.progress(prob, text=f"{EMOTION_META[emotion]['emoji']} {emotion} — {prob:.1%}")
                st.markdown("</div>", unsafe_allow_html=True)

        if results["BiLSTM"] and results["BERT"]:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Side-by-Side Probability Chart</div>', unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="BiLSTM",
                x=EMOTIONS,
                y=[results["BiLSTM"].probabilities[e] for e in EMOTIONS],
                marker_color="#6C5CE7",
            ))
            fig.add_trace(go.Bar(
                name="BERT",
                x=EMOTIONS,
                y=[results["BERT"].probabilities[e] for e in EMOTIONS],
                marker_color="#00CEC9",
            ))
            fig.update_layout(
                barmode="group", yaxis_tickformat=".0%",
                plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Model",
            )
            st.plotly_chart(fig, use_container_width=True)

            agree = results["BiLSTM"].primary_emotion == results["BERT"].primary_emotion
            if agree:
                st.success(f"✅ Both models agree: **{results['BiLSTM'].primary_emotion}**")
            else:
                st.warning(
                    f"⚠️ Models disagree — BiLSTM says **{results['BiLSTM'].primary_emotion}**, "
                    f"BERT says **{results['BERT'].primary_emotion}**. BERT's deeper contextual "
                    f"understanding is generally more reliable for ambiguous phrasing."
                )
            st.markdown("</div>", unsafe_allow_html=True)
