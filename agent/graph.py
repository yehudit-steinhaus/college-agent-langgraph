"""הגרף של עוזר המזכירות.

מבנה:
    START -> classify (Router)
        info    -> agent <-> tools (לולאת ReAct) -> review -> (לולאה חזרה ל-agent | END)
        request -> prepare_request -> human_approval (interrupt) -> submit_request | reject_request -> END
        other   -> fallback -> END
"""
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from agent.llm import get_llm
from agent.prompts import PROMPTS
from agent.state import AgentState
from agent.tools import TOOLS, submit_request_to_office

MAX_REVIEWS = 2  # מספר מקסימלי של סבבי בדיקה, כדי שהלולאה לא תהיה אינסופית


# ---------- עזרים ----------
def _prompts(config: RunnableConfig) -> dict:
    version = (config or {}).get("configurable", {}).get("prompt_version", "v2")
    return PROMPTS[version]


def _text(message) -> str:
    """תוכן הודעה כטקסט (חלק מהספקים מחזירים רשימת חלקים)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def _last_human_index(messages) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return 0


# ---------- סכמות לפלט מובנה ----------
class Route(BaseModel):
    intent: Literal["info", "request", "other"]
    reason: str = Field(description="נימוק קצר לסיווג")


class Review(BaseModel):
    ok: bool = Field(description="האם התשובה מבוססת על הכלים ועונה על השאלה")
    feedback: str = Field(description="מה לתקן; ריק אם ok=true")


class RequestDetails(BaseModel):
    request_type: Literal["extension", "exam_appeal", "certificate", "absence", "other"]
    course: str = Field(default="", description="שם הקורס אם הוזכר")
    summary: str = Field(description="תיאור קצר של הבקשה")


# ---------- Nodes ----------
def classify(state: AgentState, config: RunnableConfig) -> dict:
    """Router: מסווג את ההודעה האחרונה."""
    question = _text(state["messages"][-1])
    llm = get_llm().with_structured_output(Route)
    route = llm.invoke([SystemMessage(_prompts(config)["router"]), HumanMessage(question)])
    # איפוס שדות מהסבב הקודם בשיחה
    return {
        "intent": route.intent,
        "review_count": 0,
        "review_feedback": "",
        "review_passed": False,
        "final_answer": "",
        "request": {},
        "request_status": "",
    }


def agent(state: AgentState, config: RunnableConfig) -> dict:
    """ReAct: המודל מחליט אם להפעיל כלי או לענות."""
    messages = [SystemMessage(_prompts(config)["agent"]), *state["messages"]]
    if state.get("review_feedback"):
        # ההערות נשלחות למודל אבל לא נשמרות בהיסטוריית השיחה
        messages.append(HumanMessage(
            "[בקרת איכות] הטיוטה הקודמת נפסלה. תקן לפי ההערות:\n" + state["review_feedback"]
        ))
    response = get_llm().bind_tools(TOOLS).invoke(messages)
    return {"messages": [response]}


def review(state: AgentState, config: RunnableConfig) -> dict:
    """בודק שהתשובה מבוססת על תוצאות הכלים. אם לא – מחזיר ל-agent עם הערות."""
    messages = state["messages"]
    start = _last_human_index(messages)
    question = _text(messages[start])
    answer = _text(messages[-1])
    tool_outputs = [_text(m) for m in messages[start:] if isinstance(m, ToolMessage)]
    evidence = "\n---\n".join(tool_outputs) if tool_outputs else "(לא הופעלו כלים)"

    llm = get_llm().with_structured_output(Review)
    result = llm.invoke([
        SystemMessage(_prompts(config)["review"]),
        HumanMessage(f"שאלה:\n{question}\n\nמידע שהחזירו הכלים:\n{evidence}\n\nתשובה:\n{answer}"),
    ])

    count = state.get("review_count", 0) + 1
    if result.ok or count >= MAX_REVIEWS:
        return {
            "review_count": count,
            "review_feedback": "",
            "review_passed": result.ok,
            "final_answer": answer,
        }
    return {"review_count": count, "review_feedback": result.feedback or "התשובה לא מבוססת על הכלים."}


def prepare_request(state: AgentState, config: RunnableConfig) -> dict:
    """מחלץ את פרטי הבקשה לפני שמבקשים אישור."""
    llm = get_llm().with_structured_output(RequestDetails)
    details = llm.invoke([
        SystemMessage(_prompts(config)["extract"]),
        HumanMessage(_text(state["messages"][-1])),
    ])
    return {"request": details.model_dump()}


def human_approval(state: AgentState) -> Command[Literal["submit_request", "reject_request"]]:
    """Human-in-the-loop: הגרף נעצר כאן עד שאדם מאשר, דוחה או מתקן את הבקשה.
    שימי לב: אחרי resume ה-Node רץ מההתחלה, לכן אין כאן פעולות לפני ה-interrupt."""
    decision = interrupt({
        "question": "לאשר את הגשת הבקשה הבאה למזכירות?",
        "request": state["request"],
    })
    if isinstance(decision, bool):
        decision = {"approved": decision}

    request = {**state["request"], **(decision.get("edits") or {})}
    if decision.get("note"):
        request["reviewer_note"] = decision["note"]

    target = "submit_request" if decision.get("approved") else "reject_request"
    return Command(goto=target, update={"request": request})


def submit_request(state: AgentState) -> dict:
    request_id = submit_request_to_office(state["request"])
    answer = f"הבקשה הוגשה למזכירות. מספר הבקשה: {request_id}. עדכון יישלח אלייך בדוא\"ל."
    return {"messages": [AIMessage(answer)], "final_answer": answer, "request_status": "submitted"}


def reject_request(state: AgentState) -> dict:
    note = state["request"].get("reviewer_note")
    answer = "הבקשה לא הוגשה."
    if note:
        answer += f" הערת הבודק: {note}"
    return {"messages": [AIMessage(answer)], "final_answer": answer, "request_status": "rejected"}


def fallback(state: AgentState) -> dict:
    answer = (
        "אני עוזר המזכירות של המכללה ויכול לעזור בשאלות על נהלים, קורסים ומועדים, "
        "או להגיש בקשה למזכירות (הארכה, ערעור, אישור לימודים, היעדרות)."
    )
    return {"messages": [AIMessage(answer)], "final_answer": answer}


# ---------- Conditional edges ----------
def route_by_intent(state: AgentState) -> str:
    return state["intent"]


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "review"


def route_after_review(state: AgentState) -> str:
    return "agent" if state.get("review_feedback") else END


# ---------- בניית הגרף ----------
def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("review", review)
    builder.add_node("prepare_request", prepare_request)
    builder.add_node("human_approval", human_approval)
    builder.add_node("submit_request", submit_request)
    builder.add_node("reject_request", reject_request)
    builder.add_node("fallback", fallback)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify", route_by_intent,
        {"info": "agent", "request": "prepare_request", "other": "fallback"},
    )

    # לולאת ReAct
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "review": "review"})
    builder.add_edge("tools", "agent")

    # לולאת בקרת איכות
    builder.add_conditional_edges("review", route_after_review, {"agent": "agent", END: END})

    # ענף בקשות עם אישור אנושי (human_approval מחזיר Command ולכן אין לו edges)
    builder.add_edge("prepare_request", "human_approval")
    builder.add_edge("submit_request", END)
    builder.add_edge("reject_request", END)
    builder.add_edge("fallback", END)

    return builder.compile(checkpointer=checkpointer)


# בשביל LangGraph Studio (הוא מוסיף checkpointer בעצמו)
graph = build_graph()
