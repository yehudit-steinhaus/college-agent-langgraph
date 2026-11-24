"""מריץ את כל שלב ההערכה בפקודה אחת:
1. יוצר את הדאטהסט ב-LangSmith (אם עוד לא קיים)
2. מריץ ניסוי עם v1 (לפני השיפור) ועם v2 (אחרי השיפור)
3. מדפיס טבלת השוואה וממלא אותה אוטומטית ב-README.md

הרצה (מתיקיית הפרויקט):  python -m eval.run_all
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client  # noqa: E402

from eval import dataset  # noqa: E402
from eval.evaluators import ALL_EVALUATORS  # noqa: E402
from eval.run_eval import make_target  # noqa: E402

METRICS = ["correct_route", "facts_included", "hitl_when_needed", "correct_request_type"]
README = Path(__file__).resolve().parent.parent / "README.md"


def run_experiment(client: Client, version: str):
    print(f"\n=== מריץ ניסוי {version} (זה לוקח כמה דקות) ===")
    results = client.evaluate(
        make_target(version),
        data=dataset.DATASET_NAME,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=f"college-assistant-{version}",
        metadata={"prompt_version": version},
        max_concurrency=2,
    )

    scores = {m: [] for m in METRICS}
    for row in results:
        for res in row["evaluation_results"]["results"]:
            if res.key in scores and res.score is not None:
                scores[res.key].append(float(res.score))

    averages = {m: (sum(v) / len(v) if v else None) for m, v in scores.items()}
    print(f"הניסוי {results.experiment_name} הסתיים.")
    return results.experiment_name, averages


def fmt(value):
    return "-" if value is None else f"{value:.2f}"


def update_readme(names: dict, avgs: dict):
    lines = [
        "| מעריך | v1 | v2 |",
        "|---|---|---|",
        *[f"| {m} | {fmt(avgs['v1'][m])} | {fmt(avgs['v2'][m])} |" for m in METRICS],
        "",
        f"ניסויים ב-LangSmith: `{names['v1']}` מול `{names['v2']}` (ציון ממוצע בין 0 ל-1).",
    ]
    table = "\n".join(lines)

    text = README.read_text(encoding="utf-8")
    start = text.index("| מעריך | v1 | v2 |")
    # מחליפים רק את הטבלה ואת שורת שמות הניסויים, ומשאירים את הניתוח שמתחתיה
    marker = "ניסויים ב-LangSmith:"
    if marker in text[start:]:
        end = text.index("\n", text.index(marker, start)) + 1
    else:
        end = text.index("![השוואת ניסויים]")
    text = text[:start] + table + "\n" + text[end:]
    README.write_text(text, encoding="utf-8")
    print("\nטבלת התוצאות עודכנה ב-README.md")


def main():
    dataset.main()
    client = Client()
    names, avgs = {}, {}
    for version in ["v1", "v2"]:
        names[version], avgs[version] = run_experiment(client, version)

    print("\n========== השוואה ==========")
    print(f"{'מעריך':<24}{'v1':>8}{'v2':>8}")
    for m in METRICS:
        print(f"{m:<24}{fmt(avgs['v1'][m]):>8}{fmt(avgs['v2'][m]):>8}")

    update_readme(names, avgs)


if __name__ == "__main__":
    main()
