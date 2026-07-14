"""
AI-Driven Emotion Detection & Personalized Learning Support Platform
Main Streamlit application.

Run with:
    streamlit run app.py
"""
import streamlit as st

from config import EMOTIONS, EMOTION_META, GEMINI_API_KEY
from utils.emotion_engine import EmotionEngine, EmotionResult
from utils.preprocessing import is_valid_input
from utils.gemini_helper import generate_supportive_response
from utils.logger import log_interaction

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


load_css()


@st.cache_resource(show_spinner=False)
def get_engine() -> EmotionEngine:
    return EmotionEngine()


engine = get_engine()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 AI Learning Assistant")
    st.caption("Emotion-aware academic support")

    st.markdown("### ⚙️ Settings")

    model_choice = st.radio(
        "Emotion classifier model",
        options=["BiLSTM", "BERT", "Compare Both"],
        index=0,
        help="BiLSTM = fast, lightweight. BERT = deeper contextual understanding.",
    )

    use_rules = st.toggle("Apply rule-based keyword enhancement", value=True)
    use_gemini = st.toggle("Generate AI guidance with Gemini", value=True)

    api_key_input = st.text_input(
        "Gemini API Key (optional)",
        value=GEMINI_API_KEY,
        type="password",
        help="Leave blank to use built-in fallback responses instead of live Gemini calls.",
    )

    st.markdown("---")
    st.markdown("### 📦 Model status")
    bilstm_ok = engine.bilstm_available()
    bert_ok = engine.bert_available()
    st.markdown(f"{'✅' if bilstm_ok else '⚠️'} BiLSTM {'ready' if bilstm_ok else 'not trained yet'}")
    st.markdown(f"{'✅' if bert_ok else '⚠️'} BERT {'ready' if bert_ok else 'not trained yet'}")
    if not bilstm_ok:
        st.caption("Run `python training/train_bilstm.py` to train it.")
    if not bert_ok:
        st.caption("Run `python training/train_bert.py` to train it.")

    st.markdown("---")
    st.markdown("### 🧭 Navigate")
    st.page_link("app.py", label="🏠 Home", icon=None)
    st.page_link("pages/1_Analytics_Dashboard.py", label="Analytics Dashboard")
    st.page_link("pages/2_Model_Comparison.py", label="Model Comparison")


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <h1>🎓 AI-Driven Emotion Detection & Learning Support</h1>
        <p>Describe your study challenge in plain language — get instant emotion insight,
        mixed-emotion breakdowns, and personalized, empathetic guidance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1.15, 1])

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">✍️ Describe your challenge</div>', unsafe_allow_html=True)

    example_prompts = [
        "I'm completely lost on recursion, nothing makes sense.",
        "I just aced my practice quiz on gradient descent!",
        "I keep wondering how neural networks actually learn.",
        "This chapter on thermodynamics is so repetitive, I can't focus.",
        "I've tried debugging this for hours and I'm about to give up.",
    ]
    picked_example = st.selectbox(
        "Or try an example:", ["— choose an example —"] + example_prompts,
    )

    default_text = "" if picked_example == "— choose an example —" else picked_example
    user_text = st.text_area(
        "Your message",
        value=default_text,
        placeholder="e.g. I'm lost on recursion and don't know where to start...",
        height=140,
        label_visibility="collapsed",
    )

    analyze_clicked = st.button("🔍 Analyze Emotion & Get Guidance", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💡 How it works</div>', unsafe_allow_html=True)
    st.markdown(
        """
1. **Type** your study challenge in plain language
2. Our **BiLSTM / BERT** models classify your emotional state
3. A **rule-based layer** sharpens the prediction with keyword cues
4. **Mixed emotions** (e.g. Curious + Confused) are surfaced when close
5. **Gemini AI** generates a tailored tip, next step, and encouragement
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎭 Emotions we detect</div>', unsafe_allow_html=True)
    badge_html = "".join(
        f'<span class="emotion-badge" style="background:{EMOTION_META[e]["color"]}; margin:4px;">'
        f'{EMOTION_META[e]["emoji"]} {e}</span>'
        for e in EMOTIONS
    )
    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_confidence_bars(probs: dict):
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    for emotion, prob in ranked:
        color = EMOTION_META[emotion]["color"]
        pct = prob * 100
        st.markdown(
            f"""
            <div class="conf-row">
                <div class="conf-label">{EMOTION_META[emotion]['emoji']} {emotion}</div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{pct:.1f}%; background:{color};"></div>
                </div>
                <div class="conf-pct">{pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_result_card(result: EmotionResult, container=None):
    target = container if container else st.container()
    meta = EMOTION_META[result.primary_emotion]

    badges = (
        f'<span class="emotion-badge" style="background:{meta["color"]}">'
        f'{meta["emoji"]} {result.primary_emotion} · {result.primary_confidence:.0%}</span>'
    )
    if result.is_mixed and result.secondary_emotion:
        sec_meta = EMOTION_META[result.secondary_emotion]
        badges += (
            f'<span class="secondary-badge" style="background:{sec_meta["color"]}">'
            f'+ {sec_meta["emoji"]} {result.secondary_emotion} · {result.secondary_confidence:.0%}</span>'
        )

    target.markdown(
        f'<div class="glass-card"><div class="section-title">🧠 {result.model_used} Prediction</div>{badges}<br><br>',
        unsafe_allow_html=True,
    )
    with target:
        render_confidence_bars(result.probabilities)
        if result.is_mixed:
            st.info(
                f"Mixed emotion detected: primarily **{result.primary_emotion}**, "
                f"with notable **{result.secondary_emotion}** signals."
            )
    target.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Analysis + results
# ---------------------------------------------------------------------------
if analyze_clicked:
    if not is_valid_input(user_text):
        st.warning("Please enter at least a few words describing your challenge.")
    else:
        with st.spinner("Analyzing emotional tone..."):
            if model_choice == "Compare Both":
                results = engine.predict_both(user_text, use_rule_enhancement=use_rules)
                available_results = {k: v for k, v in results.items() if v is not None}
            else:
                single = engine.predict(user_text, model_choice, use_rule_enhancement=use_rules)
                available_results = {model_choice: single} if single else {}

        if not available_results:
            st.error(
                "No trained model is available yet. Please train BiLSTM and/or BERT "
                "using the scripts in the `training/` folder, then reload the app."
            )
        else:
            st.markdown("## 📈 Results")

            if len(available_results) == 2:
                c1, c2 = st.columns(2)
                render_result_card(available_results["BiLSTM"], c1)
                render_result_card(available_results["BERT"], c2)
                primary_result = available_results["BiLSTM"]  # used for Gemini guidance
            else:
                primary_result = list(available_results.values())[0]
                render_result_card(primary_result)

            # ---------------- Gemini guidance ----------------
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🤖 Personalized Guidance</div>', unsafe_allow_html=True)

            if use_gemini:
                with st.spinner("Generating tailored guidance..."):
                    guidance = generate_supportive_response(
                        text=user_text,
                        primary_emotion=primary_result.primary_emotion,
                        primary_conf=primary_result.primary_confidence,
                        api_key=api_key_input,
                        secondary_emotion=primary_result.secondary_emotion,
                        secondary_conf=primary_result.secondary_confidence,
                    )
                st.markdown(guidance)
            else:
                st.caption("Toggle 'Generate AI guidance with Gemini' in the sidebar to see tailored tips here.")
            st.markdown("</div>", unsafe_allow_html=True)

            # ---------------- logging ----------------
            for model_name, result in available_results.items():
                log_interaction(
                    text=user_text,
                    model_used=model_name,
                    primary_emotion=result.primary_emotion,
                    primary_confidence=result.primary_confidence,
                    secondary_emotion=result.secondary_emotion,
                    secondary_confidence=result.secondary_confidence,
                    gemini_used=use_gemini,
                )

            st.session_state.history.insert(0, {
                "text": user_text,
                "emotion": primary_result.primary_emotion,
                "confidence": primary_result.primary_confidence,
                "guidance": guidance if use_gemini else None,
            })
            st.toast("Interaction logged for analytics ✅")

# ---------------------------------------------------------------------------
# Recent activity (this session)
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.markdown("## 🕘 This Session")
    for i, item in enumerate(st.session_state.history[:5]):
        meta = EMOTION_META[item["emotion"]]
        st.markdown(
            f"""
            <div class="glass-card" style="padding:0.9rem 1.3rem;">
                <span class="emotion-badge" style="background:{meta['color']}; font-size:0.85rem; padding:0.35rem 0.8rem;">
                    {meta['emoji']} {item['emotion']} · {item['confidence']:.0%}
                </span>
                <span style="margin-left:10px; color:#555;">{item['text'][:110]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if item.get("guidance"):
            with st.expander("💡 View personalized guidance", expanded=(i == 0)):
                st.markdown(item["guidance"])

st.markdown(
    '<div class="footer-note">Built for academic support teams & self-paced learners · '
    'BiLSTM + BERT + Rule-Based + Gemini AI</div>',
    unsafe_allow_html=True,
)
