"""Analytics Dashboard — visualize emotion trends from logged interactions."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from config import EMOTION_META
from utils.logger import load_logs

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")


def load_css():
    try:
        with open(Path(__file__).resolve().parent.parent / "assets" / "style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


load_css()

st.markdown(
    """
    <div class="hero-card">
        <h1>📊 Analytics Dashboard</h1>
        <p>Trends across every logged interaction — emotion distribution, confidence,
        and activity over time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = load_logs()

if df.empty:
    st.info("No interactions logged yet. Head back to the main page and analyze a few messages first!")
    st.stop()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔎 Filters")
    models = sorted(df["model_used"].unique().tolist())
    selected_models = st.multiselect("Model", models, default=models)
    emotions = sorted(df["primary_emotion"].unique().tolist())
    selected_emotions = st.multiselect("Emotion", emotions, default=emotions)

filtered = df[
    df["model_used"].isin(selected_models) & df["primary_emotion"].isin(selected_emotions)
]

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Interactions", len(filtered))
k2.metric("Avg. Confidence", f"{filtered['primary_confidence'].mean():.1%}" if len(filtered) else "—")
most_common = filtered["primary_emotion"].mode()[0] if len(filtered) else "—"
k3.metric("Most Common Emotion", f"{EMOTION_META.get(most_common, {}).get('emoji','')} {most_common}")
mixed_rate = (filtered["secondary_emotion"].astype(str) != "").mean() if len(filtered) else 0
k4.metric("Mixed-Emotion Rate", f"{mixed_rate:.1%}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
color_map = {k: v["color"] for k, v in EMOTION_META.items()}

c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="section-title">🎭 Emotion Distribution</div>', unsafe_allow_html=True)
    dist = filtered["primary_emotion"].value_counts().reset_index()
    dist.columns = ["emotion", "count"]
    fig = px.bar(
        dist, x="emotion", y="count", color="emotion",
        color_discrete_map=color_map, text="count",
    )
    fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown('<div class="section-title">🥧 Share of Emotions</div>', unsafe_allow_html=True)
    fig2 = px.pie(
        dist, names="emotion", values="count",
        color="emotion", color_discrete_map=color_map, hole=0.45,
    )
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown('<div class="section-title">📈 Emotion Trend Over Time</div>', unsafe_allow_html=True)
if len(filtered) > 1:
    trend = filtered.copy()
    trend["date"] = trend["timestamp"].dt.date
    trend_counts = trend.groupby(["date", "primary_emotion"]).size().reset_index(name="count")
    fig3 = px.line(
        trend_counts, x="date", y="count", color="primary_emotion",
        color_discrete_map=color_map, markers=True,
    )
    fig3.update_layout(plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.caption("Log more interactions across different times to see trend lines.")

st.markdown('<div class="section-title">🧠 Avg. Confidence by Model</div>', unsafe_allow_html=True)
conf_by_model = filtered.groupby("model_used")["primary_confidence"].mean().reset_index()
fig4 = px.bar(
    conf_by_model, x="model_used", y="primary_confidence", color="model_used",
    text_auto=".1%",
)
fig4.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", yaxis_tickformat=".0%")
st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------------------
# Raw log table
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">🗂️ Interaction Log</div>', unsafe_allow_html=True)
st.dataframe(
    filtered.sort_values("timestamp", ascending=False),
    use_container_width=True,
    hide_index=True,
)

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered log as CSV", csv_bytes, "emotion_logs.csv", "text/csv")
