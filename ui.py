"""ממשק צ'אט בדפדפן (Streamlit) לעוזר המזכירות, כולל אישור אנושי לבקשות.
הרצה:  streamlit run ui.py"""
import uuid

import streamlit as st
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
INTENT_NAMES = {"info": "שאלת מידע", "request": "בקשה", "other": "מחוץ לתחום"}

st.set_page_config(page_title="עוזר המזכירות")
st.markdown(
    "<style>.stApp, .stChatMessage, .stMarkdown, p, label, input, textarea "
    "{direction: rtl; text-align: right;}</style>",
    unsafe_allow_html=True,
)


def new_conversation():
    st.session_state.graph = build_graph(checkpointer=InMemorySaver())
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat = []        # רשימת הודעות: {"role", "text", "meta"}
    st.session_state.pending = None   # בקשה שמחכה לאישור (interrupt)


if "graph" not in st.session_state:
    new_conversation()

# ---------- סרגל צד ----------
with st.sidebar:
    st.header("הגדרות")
    version = st.radio("גרסת פרומפט", ["v2", "v1"], help="v1 = לפני השיפור, v2 = אחרי השיפור")
    if st.button("שיחה חדשה"):
        new_conversation()
        st.rerun()
    st.caption("כל ריצה נשמרת ב-LangSmith כ-trace.")

config = {"configurable": {"thread_id": st.session_state.thread_id, "prompt_version": version}}
graph = st.session_state.graph


def handle_result(result: dict):
    """אם הגרף עצר ב-interrupt – שומרים את הבקשה לאישור; אחרת מציגים את התשובה."""
    if "__interrupt__" in result:
        st.session_state.pending = result["__interrupt__"][0].value
        return
    st.session_state.pending = None
    meta = INTENT_NAMES.get(result.get("intent"), "")
    if result.get("review_count"):
        meta += f" · סבבי בדיקה: {result['review_count']}"
    st.session_state.chat.append({"role": "assistant", "text": result.get("final_answer", ""), "meta": meta})


# ---------- היסטוריית השיחה ----------
st.title("עוזר המזכירות")
st.caption("מכללת הדר להנדסת תוכנה – שאלות על נהלים וקורסים, והגשת בקשות למזכירות")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])
        if msg.get("meta"):
            st.caption(msg["meta"])

# ---------- אישור אנושי (Human-in-the-loop) ----------
pending = st.session_state.pending
if pending:
    req = pending["request"]
    with st.container(border=True):
        st.subheader("נדרש אישור")
        st.write(pending["question"])
        st.write(f"**סוג:** {REQUEST_TYPES.get(req.get('request_type'), req.get('request_type'))}")
        st.write(f"**קורס:** {req.get('course') or '-'}")
        summary = st.text_input("תיאור הבקשה (אפשר לתקן לפני אישור)", value=req.get("summary", ""))
        note = st.text_input("הערה (במקרה של דחייה)")

        col1, col2 = st.columns(2)
        decision = None
        if col1.button("אישור והגשה", type="primary", use_container_width=True):
            decision = {"approved": True}
            if summary != req.get("summary"):
                decision["edits"] = {"summary": summary}
        if col2.button("דחייה", use_container_width=True):
            decision = {"approved": False, "note": note}

        if decision is not None:
            with st.spinner("ממשיך..."):
                handle_result(graph.invoke(Command(resume=decision), config))
            st.rerun()

# ---------- קלט ----------
question = st.chat_input("כתבו שאלה או בקשה...", disabled=pending is not None)
if question:
    st.session_state.chat.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.write(question)
    with st.spinner("חושב..."):
        handle_result(graph.invoke({"messages": [HumanMessage(question)]}, config))
    st.rerun()
