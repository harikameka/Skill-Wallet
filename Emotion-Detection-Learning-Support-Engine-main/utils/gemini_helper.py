"""
Gemini AI integration.

Generates personalized, emotion-aware learning guidance (tips, next steps,
encouragement) based on the detected emotion(s) and the student's raw text.

Requires: pip install google-generativeai
Set your API key via the GEMINI_API_KEY environment variable, or paste it
directly into the sidebar of the Streamlit app at runtime.
"""
from typing import Dict, List, Optional

from config import GEMINI_MODEL_NAME

_PROMPT_TEMPLATE = """You are an empathetic, encouraging academic support assistant
embedded in a learning platform. A student described their study challenge below.

Student's message: "{text}"

Detected primary emotion: {primary_emotion} (confidence: {primary_conf:.0%})
{secondary_line}

Write a short, warm, and genuinely helpful response with exactly three parts,
using these exact markdown headers:

### Understanding
One or two sentences showing you understand how the student feels, in a
natural, non-clinical tone. Do not just repeat the emotion label.

### Tips
2-3 concise, concrete, actionable study tips relevant to what they described.
Use a bullet list.

### Next Step
One single, specific, small next action they can take right now.

Keep the whole response under 150 words. Be encouraging but not saccharine.
Do not mention that you are an AI model or reference "detected emotions" explicitly.
"""


def _build_prompt(text: str, primary_emotion: str, primary_conf: float,
                   secondary_emotion: Optional[str] = None,
                   secondary_conf: Optional[float] = None) -> str:
    secondary_line = ""
    if secondary_emotion:
        secondary_line = f"Secondary emotion also present: {secondary_emotion} (confidence: {secondary_conf:.0%})"
    return _PROMPT_TEMPLATE.format(
        text=text,
        primary_emotion=primary_emotion,
        primary_conf=primary_conf,
        secondary_line=secondary_line,
    )


def generate_supportive_response(
    text: str,
    primary_emotion: str,
    primary_conf: float,
    api_key: str,
    secondary_emotion: Optional[str] = None,
    secondary_conf: Optional[float] = None,
) -> str:
    """
    Calls the Gemini API to generate a tailored supportive response.
    Falls back to a template-based response if no API key is configured
    or the call fails for any reason (network, quota, etc.).
    """
    if not api_key:
        return _fallback_response(primary_emotion, secondary_emotion)

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)

        prompt = _build_prompt(
            text, primary_emotion, primary_conf, secondary_emotion, secondary_conf,
        )
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as exc:  # noqa: BLE001 - surface a friendly fallback
        fallback = _fallback_response(primary_emotion, secondary_emotion)
        return (
            f"{fallback}\n\n"
            f"_(Note: Gemini API call failed — showing a rule-based fallback response. "
            f"Details: {exc})_"
        )


# ---------------------------------------------------------------------------
# Fallback templates (used when no Gemini API key is provided)
# ---------------------------------------------------------------------------
_FALLBACKS: Dict[str, str] = {
    "Bored": (
        "### Understanding\nIt sounds like the material has stopped holding your attention.\n\n"
        "### Tips\n- Switch up the format: try a video, a short quiz, or teach the concept to someone else\n"
        "- Break study time into 20-minute focused sprints with short breaks\n"
        "- Connect the topic to a real-world example you actually care about\n\n"
        "### Next Step\nPick one small, unfamiliar sub-topic within this subject and spend 10 minutes exploring just that."
    ),
    "Confident": (
        "### Understanding\nYou're in a great headspace right now — that confidence is worth building on.\n\n"
        "### Tips\n- Test yourself with harder, mixed-topic practice problems\n"
        "- Try explaining the concept out loud without notes\n"
        "- Help a classmate who's stuck — teaching deepens mastery\n\n"
        "### Next Step\nAttempt one challenging problem that's a level above what you've already mastered."
    ),
    "Confused": (
        "### Understanding\nFeeling lost here is completely normal — this is a genuinely tricky concept.\n\n"
        "### Tips\n- Break the problem into the smallest possible steps and check where it breaks down\n"
        "- Find a different explanation (video, diagram, or classmate's phrasing)\n"
        "- Write down exactly what you don't understand — that's often the fastest way to clarity\n\n"
        "### Next Step\nRe-read just the first step you got stuck on, and try explaining it in your own words."
    ),
    "Curious": (
        "### Understanding\nThat curiosity is a great sign — it means you're genuinely engaging with the material.\n\n"
        "### Tips\n- Follow the thread: look up one related concept you're wondering about\n"
        "- Try a small experiment or variation to see what changes\n"
        "- Keep a running list of interesting questions to revisit later\n\n"
        "### Next Step\nSpend 10 minutes exploring the specific question that sparked your curiosity."
    ),
    "Frustrated": (
        "### Understanding\nThat frustration makes sense — you've clearly put in real effort here.\n\n"
        "### Tips\n- Step away for 5-10 minutes; frustration often blocks clear thinking\n"
        "- Isolate the exact point where things stopped working\n"
        "- Ask for help on just that one piece, not the whole problem\n\n"
        "### Next Step\nTake a short break, then come back and re-attempt only the smallest failing piece."
    ),
}


def _fallback_response(primary_emotion: str, secondary_emotion: Optional[str] = None) -> str:
    base = _FALLBACKS.get(primary_emotion, _FALLBACKS["Confused"])
    if secondary_emotion and secondary_emotion in _FALLBACKS:
        base += f"\n\n_You may also be feeling a bit **{secondary_emotion}** — that's worth noticing too._"
    return base
