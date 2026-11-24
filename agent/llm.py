"""יצירת מודל השפה. ספק המודל נקבע במשתנה הסביבה MODEL."""
import os

from langchain.chat_models import init_chat_model


def get_llm():
    # דוגמאות: "openai:gpt-4o-mini" או "google_genai:gemini-3.5-flash"
    model_name = os.getenv("MODEL", "openai:gpt-4o-mini")
    kwargs = {"temperature": 0, "timeout": 120, "max_retries": 2}

    if model_name.startswith("google_genai:"):
        # ב-Gemini 3 ברירת המחדל היא "חשיבה" ברמה גבוהה, וזה מאט מאוד.
        # למשימות פשוטות כמו שלנו מספיקה רמה נמוכה.
        kwargs["thinking_level"] = os.getenv("THINKING_LEVEL", "low")

    return init_chat_model(model_name, **kwargs)