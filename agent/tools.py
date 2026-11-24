"""הכלים שה-Agent יכול להפעיל (ReAct) ופעולת הגשת הבקשה."""
import json
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

FAQ = json.loads((DATA_DIR / "faq.json").read_text(encoding="utf-8"))
COURSES = json.loads((DATA_DIR / "courses.json").read_text(encoding="utf-8"))

# אותיות שימוש נפוצות בתחילת מילה בעברית (ו, ה, ב, ל, מ, ש, כ)
_PREFIXES = "והבלמשכ"


def _word_variants(word: str) -> set[str]:
    variants = {word}
    if len(word) > 3 and word[0] in _PREFIXES:
        variants.add(word[1:])
    return variants


@tool
def search_faq(query: str) -> str:
    """חיפוש בתקנון ובשאלות הנפוצות של המכללה: נוכחות, הארכות, ערעורים,
    מועדי ב', אישורים, שעות פעילות, ציון עובר, פרויקט גמר, התאמות בבחינות."""
    words = [w.strip("?.,!\"'") for w in query.split()]
    words = [w for w in words if len(w) > 1]

    scored = []
    for item in FAQ:
        text = " ".join([item["question"], item["answer"], *item.get("keywords", [])])
        score = sum(1 for w in words if any(v in text for v in _word_variants(w)))
        if score:
            scored.append((score, item))

    if not scored:
        return "לא נמצא מידע רלוונטי בתקנון."

    scored.sort(key=lambda pair: -pair[0])
    return "\n\n".join(
        f"[{item['id']}] {item['question']}\n{item['answer']}" for _, item in scored[:3]
    )


@tool
def get_course_info(course_name: str) -> str:
    """פרטי קורס: שם המרצה, יום ושעות השיעור, חדר ומועד הבחינה."""
    name = course_name.strip().lower()
    if not name:
        return "לא צוין שם קורס."
    for course in COURSES:
        candidates = [course["name"].lower(), *[a.lower() for a in course["aliases"]]]
        if any(c in name or name in c for c in candidates):
            return (
                f"קורס: {course['name']}\n"
                f"מרצה: {course['lecturer']}\n"
                f"מועד השיעור: {course['day']} {course['hours']}\n"
                f"מיקום: {course['room']}\n"
                f"מועד בחינה: {course['exam_date']}"
            )
    available = ", ".join(c["name"] for c in COURSES)
    return f"לא נמצא קורס בשם '{course_name}'. הקורסים הקיימים: {available}"


TOOLS = [search_faq, get_course_info]


def submit_request_to_office(request: dict) -> str:
    """'שליחת' הבקשה למזכירות – כאן נשמרת לקובץ. זו הפעולה הרגישה שדורשת אישור אדם."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    request_id = uuid.uuid4().hex[:6].upper()
    record = {"id": request_id, "submitted_at": datetime.now().isoformat(), **request}
    with open(OUTPUT_DIR / "requests.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return request_id
