"""יוצר את הדאטהסט ב-LangSmith.
הרצה (מתיקיית הפרויקט):  python -m eval.dataset"""
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATASET_NAME = "college-assistant-eval"

# outputs = התשובה המצופה:
#   intent       – לאן הנתב צריך לשלוח
#   must_include – מחרוזות שחייבות להופיע בתשובה (רק לשאלות מידע)
#   request_type – סוג הבקשה (רק לבקשות)
EXAMPLES = [
    # --- שאלות מידע ---
    {"inputs": {"question": "מה אחוז הנוכחות הנדרש בקורסים?"},
     "outputs": {"intent": "info", "must_include": ["80"]}},
    {"inputs": {"question": "מתי המזכירות פתוחה?"},
     "outputs": {"intent": "info", "must_include": ["9:00", "15:00"]}},
    {"inputs": {"question": "מה ציון עובר בפרויקט גמר?"},
     "outputs": {"intent": "info", "must_include": ["65"]}},
    {"inputs": {"question": "תוך כמה זמן אפשר לערער על ציון?"},
     "outputs": {"intent": "info", "must_include": ["14"]}},
    {"inputs": {"question": "כמה תוספת זמן מקבלים בבחינה?"},
     "outputs": {"intent": "info", "must_include": ["25"]}},
    {"inputs": {"question": "מי מלמד את פיתוח Web?"},
     "outputs": {"intent": "info", "must_include": ["לוי"]}},
    {"inputs": {"question": "באיזה חדר מתקיים הקורס בפייתון?"},
     "outputs": {"intent": "info", "must_include": ["מעבדה 3"]}},
    # מקרי קצה: שואלים "איך" – זה מידע, לא בקשה
    {"inputs": {"question": "איך מבקשים הארכה להגשת מטלה?"},
     "outputs": {"intent": "info", "must_include": ["7", "48"]}},
    {"inputs": {"question": "מה הנוהל אם אני צריכה אישור לימודים?"},
     "outputs": {"intent": "info", "must_include": ["3"]}},
    # מקרה קצה: מידע שלא קיים בתקנון – בודק שהמודל לא ממציא
    {"inputs": {"question": "כמה עולה חניה במכללה?"},
     "outputs": {"intent": "info", "must_include": ["לא מצאתי"]}},

    # --- בקשות (צריכות לעבור דרך אישור אנושי) ---
    {"inputs": {"question": "אני רוצה לבקש הארכה של 3 ימים לתרגיל בפייתון"},
     "outputs": {"intent": "request", "request_type": "extension"}},
    {"inputs": {"question": "תגישו בבקשה ערעור על הציון שלי בבסיסי נתונים, שאלה 3 נבדקה לא נכון"},
     "outputs": {"intent": "request", "request_type": "exam_appeal"}},
    {"inputs": {"question": "תוציאו לי בבקשה אישור לימודים לעבודה"},
     "outputs": {"intent": "request", "request_type": "certificate"}},
    {"inputs": {"question": "לא אוכל להגיע לשיעור DevOps ביום חמישי, תרשמו לי היעדרות"},
     "outputs": {"intent": "request", "request_type": "absence"}},

    # --- מחוץ לתחום ---
    {"inputs": {"question": "מה מזג האוויר מחר בירושלים?"},
     "outputs": {"intent": "other"}},
    {"inputs": {"question": "תכתוב לי שיר על חתולים"},
     "outputs": {"intent": "other"}},
]


def main():
    client = Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"הדאטהסט '{DATASET_NAME}' כבר קיים – לא נוצר מחדש.")
        return
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="שאלות ובקשות לעוזר המזכירות של המכללה",
    )
    client.create_examples(dataset_id=dataset.id, examples=EXAMPLES)
    print(f"נוצר דאטהסט '{DATASET_NAME}' עם {len(EXAMPLES)} דוגמאות.")


if __name__ == "__main__":
    main()
