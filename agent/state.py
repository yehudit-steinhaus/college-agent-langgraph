"""ה-State שעובר בין ה-Nodes בגרף."""
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # היסטוריית השיחה – add_messages מוסיף הודעות במקום לדרוס
    messages: Annotated[list[AnyMessage], add_messages]

    # Router
    intent: str            # info / request / other

    # לולאת בקרת האיכות
    review_count: int      # כמה פעמים הבודק כבר רץ
    review_feedback: str   # הערות לתיקון (ריק = אין מה לתקן)
    review_passed: bool    # האם התשובה האחרונה עברה בדיקה

    # ענף הבקשות + Human-in-the-loop
    request: dict          # פרטי הבקשה שחולצו מההודעה
    request_status: str    # submitted / rejected

    final_answer: str
