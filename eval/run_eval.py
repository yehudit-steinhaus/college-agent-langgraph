"""מריץ ניסוי LangSmith על גרסת פרומפט אחת.
הרצה (מתיקיית הפרויקט):
    python -m eval.run_eval --version v1    # לפני השיפור
    python -m eval.run_eval --version v2    # אחרי השיפור
"""
import argparse
import uuid

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402
from langsmith import Client  # noqa: E402

from agent.graph import build_graph  # noqa: E402
from eval.dataset import DATASET_NAME  # noqa: E402
from eval.evaluators import ALL_EVALUATORS  # noqa: E402


def make_target(version: str):
    graph = build_graph(checkpointer=InMemorySaver())

    def target(inputs: dict) -> dict:
        config = {"configurable": {"thread_id": str(uuid.uuid4()), "prompt_version": version}}
        result = graph.invoke({"messages": [HumanMessage(inputs["question"])]}, config)

        # בהערכה אוטומטית אין אדם – מאשרים אוטומטית, אבל רושמים שהעצירה קרתה
        hitl_triggered = False
        while "__interrupt__" in result:
            hitl_triggered = True
            result = graph.invoke(Command(resume={"approved": True}), config)

        return {
            "intent": result.get("intent"),
            "answer": result.get("final_answer", ""),
            "hitl_triggered": hitl_triggered,
            "request_type": (result.get("request") or {}).get("request_type"),
            "review_count": result.get("review_count", 0),
            "review_passed": result.get("review_passed"),
        }

    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2"], required=True)
    args = parser.parse_args()

    client = Client()
    results = client.evaluate(
        make_target(args.version),
        data=DATASET_NAME,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=f"college-assistant-{args.version}",
        metadata={"prompt_version": args.version},
        max_concurrency=2,
    )
    print(f"הניסוי הסתיים: {results.experiment_name}")
    print("את ההשוואה בין הניסויים רואים ב-LangSmith: Datasets -> college-assistant-eval -> Experiments")


if __name__ == "__main__":
    main()
