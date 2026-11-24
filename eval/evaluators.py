"""מעריכים אוטומטיים. כל אחד מקבל את הפלט של הגרף ואת התשובה המצופה ומחזיר ציון."""


def correct_route(outputs: dict, reference_outputs: dict) -> dict:
    """האם הנתב שלח לענף הנכון."""
    score = int(outputs.get("intent") == reference_outputs.get("intent"))
    return {"key": "correct_route", "score": score}


def facts_included(outputs: dict, reference_outputs: dict) -> dict:
    """איזה חלק מהעובדות הנדרשות מופיע בתשובה (רק לשאלות מידע)."""
    required = reference_outputs.get("must_include") or []
    if not required:
        return {"key": "facts_included", "score": None, "comment": "לא רלוונטי"}
    answer = outputs.get("answer") or ""
    found = [item for item in required if item in answer]
    missing = [item for item in required if item not in answer]
    return {
        "key": "facts_included",
        "score": len(found) / len(required),
        "comment": f"חסר: {missing}" if missing else "הכול נמצא",
    }


def hitl_when_needed(outputs: dict, reference_outputs: dict) -> dict:
    """אישור אנושי הופעל אם ורק אם זו בקשה."""
    should_interrupt = reference_outputs.get("intent") == "request"
    score = int(bool(outputs.get("hitl_triggered")) == should_interrupt)
    return {"key": "hitl_when_needed", "score": score}


def correct_request_type(outputs: dict, reference_outputs: dict) -> dict:
    """סוג הבקשה שחולץ נכון (רק לבקשות)."""
    expected = reference_outputs.get("request_type")
    if not expected:
        return {"key": "correct_request_type", "score": None, "comment": "לא רלוונטי"}
    return {"key": "correct_request_type", "score": int(outputs.get("request_type") == expected)}


ALL_EVALUATORS = [correct_route, facts_included, hitl_when_needed, correct_request_type]
