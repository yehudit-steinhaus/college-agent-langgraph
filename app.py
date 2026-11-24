"""צ'אט בטרמינל עם עוזר המזכירות, כולל אישור אנושי לבקשות.
הרצה:  python app.py"""
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agent.graph import build_graph  # noqa: E402

REQUEST_TYPES = {
    "extension": "הארכת הגשה",
    "exam_appeal": "ערעור על ציון",
    "certificate": "אישור לימודים",
    "absence": "רישום היעדרות",
    "other": "אחר",
}


def ask_human(payload: dict) -> dict:
    """מציג את הבקשה לאדם ומחזיר את ההחלטה שלו (זה ה-resume value)."""
    req = payload["request"]
    print("\n========== נדרש אישור ==========")
    print(payload["question"])
    print(f"  סוג:   {REQUEST_TYPES.get(req.get('request_type'), req.get('request_type'))}")
    print(f"  קורס:  {req.get('course') or '-'}")
    print(f"  תיאור: {req.get('summary')}")
    choice = input("לאשר? [a=אישור / r=דחייה / e=עריכת התיאור ואישור]: ").strip().lower()

    if choice == "e":
        new_summary = input("תיאור מתוקן: ").strip()
        return {"approved": True, "edits": {"summary": new_summary}}
    if choice == "r":
        note = input("סיבת הדחייה (אפשר להשאיר ריק): ").strip()
        return {"approved": False, "note": note}
    return {"approved": True}


def main():
    graph = build_graph(checkpointer=InMemorySaver())
    version = os.getenv("PROMPT_VERSION", "v2")
    config = {"configurable": {"thread_id": str(uuid.uuid4()), "prompt_version": version}}

    print(f"עוזר המזכירות (פרומפט {version}). להפסקה: exit")
    while True:
        question = input("\nאת/ה: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        result = graph.invoke({"messages": [HumanMessage(question)]}, config)

        # כל עוד הגרף עצר ב-interrupt – שואלים את האדם וממשיכים
        while "__interrupt__" in result:
            decision = ask_human(result["__interrupt__"][0].value)
            result = graph.invoke(Command(resume=decision), config)

        print(f"\nעוזר: {result.get('final_answer', '')}")


if __name__ == "__main__":
    main()
