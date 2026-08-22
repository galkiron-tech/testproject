"""
MedExplain AI
==============
מערכת AI להסבר והנגשת תוצאות בדיקות דם למטופלים

University Proof of Concept for an Artificial Intelligence in Medicine course.
See README.md for the full technical write-up (this file intentionally keeps
the patient-facing product free of developer terminology).

Run with:
    streamlit run app.py

Repository layout (flat, no folders):
    app.py              -- application logic and UI (this file)
    lab_tests.json       -- structured laboratory knowledge base
    scenarios.json        -- synthetic patient demonstration scenarios
    test_classifier.py   -- unit tests for the classification engine
    requirements.txt
    README.md

SAFETY NOTE: No language model decides whether a lab value is normal,
borderline or abnormal. That decision is made entirely by explicit,
deterministic numeric comparisons in Section 6, reading thresholds from
lab_tests.json. The same input always produces the same classification.

Sections:
    1.  Imports
    2.  Global configuration
    3.  Data loading (lab_tests.json, scenarios.json)
    4.  Data models / dataclasses
    5.  Input validation
    6.  Rule-based classification engine
    7.  Trend analysis
    8.  Combination / context logic
    9.  Explanation engine
    10. Physician-question engine
    11. Summary engine
    12. Feedback / evaluation logic
    13. UI / CSS helpers
    14. Page rendering functions
    15. Main navigation / routing
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import streamlit as st


# =============================================================================
# 2. GLOBAL CONFIGURATION
# =============================================================================

PAGES = [
    "עמוד הבית",
    "תוצאות והסברים",
    "איך זה עובד",
    "למה זה חשוב?",
    "למה לא ChatGPT?",
    "בטיחות ופרטיות",
    "משוב",
]

TEST_ORDER = ["wbc", "hemoglobin", "ferritin", "hba1c", "ldl", "hdl", "triglycerides", "crp"]

# Tests that expose an optional "previous value" field for trend analysis.
TREND_TEST_KEYS = ["hba1c", "ldl", "hemoglobin", "ferritin", "crp"]

STATUS_LABELS = {"normal": "תקין", "borderline": "גבולי", "abnormal": "חריג"}
STATUS_ICONS = {"normal": "🟢", "borderline": "🟡", "abnormal": "🔴"}
STATUS_PRIORITY = {"abnormal": 0, "borderline": 1, "normal": 2}  # abnormal first

TREND_TEXT = {"up": "עלה", "down": "ירד", "unchanged": "ללא שינוי"}
TREND_TEXT_FULL = {
    "up": "עלה לעומת הבדיקה הקודמת",
    "down": "ירד לעומת הבדיקה הקודמת",
    "unchanged": "ללא שינוי לעומת הבדיקה הקודמת",
}

REFERENCE_RANGE_DISCLAIMER = (
    "הטווחים המספריים במערכת זו הם ערכי ייחוס לדוגמה למטרות אב-טיפוס לימודי בלבד. "
    "טווחי הייחוס בפועל משתנים בין מעבדות שונות ותלויים בשיטת המדידה, ואינם תחליף "
    "לפרשנות של רופא/ה."
)

# Success metrics a real deployment could track -- shown briefly on the
# feedback page and detailed in README.md.
SUCCESS_METRICS = [
    ("בהירות ההסבר", "עד כמה קל היה להבין את משמעות התוצאה מהניסוח שהוצג."),
    ("הבנת המטופל", "האם המטופל/ת יצא/ה עם תמונה נכונה וברורה יותר של הממצא."),
    ("מוכנות לשיחה עם הרופא", "האם המטופל/ת מרגיש/ה שיש לו/ה שאלות ממוקדות לקראת הפגישה."),
    ("שימושיות", "כמה פשוט וברור היה להשתמש בכלי ולמצוא את המידע הרלוונטי."),
    ("הפחתת אי-ודאות", "האם רמת החשש או הבלבול הראשוניים פחתו לאחר קריאת ההסבר."),
    ("זיהוי שאלות מתאימות", "האם השאלות שהוצגו אכן רלוונטיות למצב הספציפי של המטופל/ת."),
]


# =============================================================================
# 3. DATA LOADING (lab_tests.json, scenarios.json)
# =============================================================================
# Structured medical knowledge (LAB_CONFIG) and synthetic demonstration
# scenarios (SCENARIOS) are kept as separate JSON files, not hard-coded in
# this module, so medical content can be reviewed and edited independently
# of application logic. Both files must sit alongside app.py -- paths are
# resolved relative to this file's own location so loading works regardless
# of the current working directory the app or tests are launched from.
#
# lab_tests.json direction / threshold conventions:
#   direction: "both" | "high_only" | "low_only"
#   thresholds: numeric boundaries. Sex-specific tests use "male"/"female"
#     sub-dicts with the same boundary keys:
#       abnormal_low_max   -> below this = abnormal-low
#       normal_min         -> below this (and >= abnormal_low_max) = borderline-low
#       normal_max         -> above this (and < abnormal_high_min) = borderline-high
#       abnormal_high_min  -> at/above this = abnormal-high

_APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_json(filename: str):
    """Load a required JSON data file from the same directory as app.py.

    Raises a clear, descriptive error if the file is missing or malformed --
    both lab_tests.json and scenarios.json are required for the application
    to function, so failing loudly here is preferable to a confusing error
    deep inside the UI later.
    """
    path = os.path.join(_APP_DIR, filename)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Required data file '{filename}' was not found at {path}. "
            "Make sure it is uploaded in the same directory as app.py."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Data file '{filename}' is not valid JSON: {exc}") from exc


LAB_CONFIG: dict = _load_json("lab_tests.json")
SCENARIOS: list = _load_json("scenarios.json")


# =============================================================================
# 4. DATA MODELS / DATACLASSES
# =============================================================================

@dataclass
class LabResult:
    """A single raw laboratory measurement entered by a patient or scenario."""
    test_key: str
    value: float
    previous_value: Optional[float] = None


@dataclass
class ClassifiedLabResult:
    """Output of the deterministic classification engine for one test."""
    test_key: str
    name_he: str
    abbreviation: str
    value: float
    unit: str
    status: str       # "normal" | "borderline" | "abnormal"
    direction: str     # "low" | "high" | "normal"
    reference_text: str
    previous_value: Optional[float] = None
    trend: Optional[str] = None


@dataclass
class PatientSummary:
    total_tests: int
    normal_count: int
    borderline_count: int
    abnormal_count: int
    key_findings: list = field(default_factory=list)
    headline: str = ""


# =============================================================================
# 5. INPUT VALIDATION
# =============================================================================

class ValidationError(Exception):
    """Raised when a raw patient-entered lab value cannot be safely used."""


def validate_value(test_key: str, raw_value, lab_config: dict = LAB_CONFIG) -> float:
    """Validate a raw (possibly string) input value for a given test.

    Raises ValidationError with a calm Hebrew message: unknown test, missing
    value, non-numeric input, or an impossible (negative / absurd) value.
    """
    if test_key not in lab_config:
        raise ValidationError(f"בדיקה לא מוכרת במערכת: {test_key}")
    if raw_value is None or raw_value == "":
        raise ValidationError("לא הוזן ערך לבדיקה זו.")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("הערך שהוזן אינו מספרי. נא להזין מספר בלבד.") from exc
    if value < 0:
        raise ValidationError("לא ניתן להזין ערך שלילי עבור בדיקת מעבדה.")
    if value > 100000:
        raise ValidationError("הערך שהוזן חורג מתחום סביר לבדיקת מעבדה. נא לבדוק את הנתון.")
    return value


# =============================================================================
# 6. RULE-BASED CLASSIFICATION ENGINE
# =============================================================================
# No language model is involved in deciding whether a value is normal,
# borderline or abnormal -- only explicit numeric comparisons below. The
# same input always produces the same classification.

def _resolve_thresholds(test_def: dict, sex: Optional[str]) -> dict:
    thresholds = test_def["thresholds"]
    if not test_def.get("sex_specific"):
        return thresholds
    if sex in ("male", "female"):
        return thresholds[sex]
    return thresholds.get("female", thresholds.get("male"))


def _resolve_reference_text(test_def: dict, sex: Optional[str]) -> str:
    ref_text = test_def.get("reference_text", "")
    if isinstance(ref_text, dict):
        if sex in ("male", "female"):
            return ref_text.get(sex, "")
        return ref_text.get("female", ref_text.get("male", ""))
    return ref_text


def classify_lab_value(test_key: str, value: float, sex: Optional[str] = None, age: Optional[int] = None, lab_config: dict = LAB_CONFIG) -> dict:
    """Classify a single validated numeric lab value against configured ranges.

    Returns: {"status", "direction", "value", "unit", "reference_text", "test_key"}
    """
    if test_key not in lab_config:
        raise ValidationError(f"בדיקה לא מוכרת במערכת: {test_key}")

    test_def = lab_config[test_key]
    th = _resolve_thresholds(test_def, sex)
    direction_support = test_def.get("direction", "both")

    status, direction = "normal", "normal"

    if direction_support in ("both", "low_only"):
        abnormal_low_max = th.get("abnormal_low_max")
        normal_min = th.get("normal_min")
        if abnormal_low_max is not None and value < abnormal_low_max:
            status, direction = "abnormal", "low"
        elif normal_min is not None and value < normal_min:
            status, direction = "borderline", "low"

    if status == "normal" and direction_support in ("both", "high_only"):
        normal_max = th.get("normal_max")
        abnormal_high_min = th.get("abnormal_high_min")
        if abnormal_high_min is not None and value >= abnormal_high_min:
            status, direction = "abnormal", "high"
        elif normal_max is not None and value > normal_max:
            status, direction = "borderline", "high"

    return {
        "status": status,
        "direction": direction,
        "value": value,
        "unit": test_def.get("unit", ""),
        "reference_text": _resolve_reference_text(test_def, sex),
        "test_key": test_key,
    }


def classify_patient_results(results: list, sex: Optional[str] = None, age: Optional[int] = None, lab_config: dict = LAB_CONFIG) -> list:
    """Classify a full set of a patient's entered/scenario lab results.

    Unknown test keys are skipped safely rather than raising, so a bad
    scenario entry never crashes the app.
    """
    classified = []
    for result in results:
        if result.test_key not in lab_config:
            continue
        test_def = lab_config[result.test_key]
        outcome = classify_lab_value(result.test_key, result.value, sex=sex, age=age, lab_config=lab_config)
        trend = compute_trend(result.value, result.previous_value)
        classified.append(
            ClassifiedLabResult(
                test_key=result.test_key,
                name_he=test_def.get("name_he", result.test_key),
                abbreviation=test_def.get("abbreviation", result.test_key.upper()),
                value=result.value,
                unit=outcome["unit"],
                status=outcome["status"],
                direction=outcome["direction"],
                reference_text=outcome["reference_text"],
                previous_value=result.previous_value,
                trend=trend,
            )
        )
    return classified


def sort_by_severity(classified: list) -> list:
    """Sort classified results: abnormal first, then borderline, then normal (stable)."""
    return sorted(classified, key=lambda r: STATUS_PRIORITY.get(r.status, 99))


# =============================================================================
# 7. TREND ANALYSIS
# =============================================================================
# These constants describe application behavior (which direction of change
# is worth flagging, and which follow-up question to add) rather than raw
# medical reference data, so they live here in Python rather than in
# lab_tests.json -- see README.md for the data-vs-logic rationale.

# The direction of change that represents a potential worsening for each
# trend-tracked test (used only to decide whether to add a trend-aware
# physician question -- never to imply causality or diagnose anything).
TREND_CONCERN_DIRECTION = {
    "hba1c": "up",
    "ldl": "up",
    "hemoglobin": "down",
    "ferritin": "down",
    "crp": "up",
}

TREND_QUESTIONS = {
    "hba1c": "האם השינוי לעומת הבדיקה הקודמת משנה את תדירות המעקב המומלצת?",
    "ldl": "האם המגמה לעומת הבדיקה הקודמת משנה את היעד או קצב המעקב המומלצים?",
    "hemoglobin": "האם המגמה לעומת הבדיקה הקודמת מצריכה בירור נוסף מעבר לערך הבודד?",
    "ferritin": "האם המגמה לעומת הבדיקה הקודמת מצריכה מעקב צמוד יותר?",
    "crp": "האם השינוי לעומת הבדיקה הקודמת משנה את מידת הדחיפות לפנייה לרופא/ה?",
}


def compute_trend(value: float, previous_value: Optional[float]) -> Optional[str]:
    """Return 'up' / 'down' / 'unchanged' relative to a previous result.

    Returns None when no previous value is available. Never implies
    causality -- only reports numeric direction.
    """
    if previous_value is None:
        return None
    if value > previous_value:
        return "up"
    if value < previous_value:
        return "down"
    return "unchanged"


def trend_sentence(result: ClassifiedLabResult) -> Optional[str]:
    """Build the 'בדיקה קודמת: X | בדיקה נוכחית: Y' sentence for display."""
    if result.trend is None or result.previous_value is None:
        return None
    return f"בדיקה קודמת: {result.previous_value} | בדיקה נוכחית: {result.value} — {TREND_TEXT_FULL[result.trend]}"


def trend_aware_question(result: ClassifiedLabResult) -> Optional[str]:
    """Return an extra physician question when the trend moves in the
    test's 'worsening' direction, or None otherwise. Never implies why the
    value changed -- only that the trend itself may be worth discussing.
    """
    if result.trend not in ("up", "down"):
        return None
    concern_direction = TREND_CONCERN_DIRECTION.get(result.test_key)
    if concern_direction is None or result.trend != concern_direction:
        return None
    return TREND_QUESTIONS.get(result.test_key)


# =============================================================================
# 8. COMBINATION / CONTEXT LOGIC
# =============================================================================
# Laboratory values are not interpreted in isolation: this section looks at
# how each result fits together with other flagged results, with its own
# trend, and with any clinical context notes attached to the patient.
#
# COMBO_DEFINITIONS describes application behavior (which test pairs to
# watch for, and what to say/ask when they co-occur) rather than raw
# medical reference data, so -- like the trend constants above -- it lives
# here in Python rather than in lab_tests.json.

COMBO_DEFINITIONS = [
    {
        "key": "hemoglobin_ferritin_low",
        "tests": {"hemoglobin": "low", "ferritin": "low"},
        "title": "המוגלובין נמוך + פריטין נמוך",
        "explanation": "מאחר שגם הפריטין וגם ההמוגלובין נמוכים, כדאי לדון בשתי התוצאות יחד עם הרופא/ה, ולא להתייחס לכל ערך בנפרד.",
        "questions": [
            "האם כדאי לפרש את שני הממצאים יחד?",
            "האם יש צורך לברר גורמים אפשריים לאובדן דם או ספיגה נמוכה?",
            "האם כדאי לבדוק B12 או חומצה פולית?",
            "מהו המעקב המתאים?",
        ],
    },
    {
        "key": "wbc_crp_high",
        "tests": {"wbc": "high", "crp": "high"},
        "title": "WBC גבוה + CRP גבוה",
        "explanation": "קיימים שני מדדים שיכולים להשתנות בתהליכים דלקתיים או זיהומיים, ולכן כדאי להתייחס אליהם יחד עם התסמינים וההקשר הקליני.",
        "questions": [
            "האם השילוב מצריך בירור נוסף?",
            "האם התסמינים שלי מתאימים לממצאים?",
            "האם כדאי לבצע בדיקות נוספות?",
            "מתי כדאי לחזור על הבדיקות?",
        ],
    },
    {
        "key": "ldl_hdl_risk",
        "tests": {"ldl": "high", "hdl": "low"},
        "title": "LDL גבוה + HDL נמוך",
        "explanation": "כדאי להעריך את פרופיל השומנים כתמונה כוללת ולא להתייחס לערך יחיד בלבד.",
        "questions": [
            "האם כדאי להסתכל על כל פרופיל השומנים יחד?",
            "מה יעד ה-LDL המתאים עבורי לפי גורמי הסיכון?",
            "אילו גורמי סיכון קרדיווסקולריים נוספים כדאי לקחת בחשבון?",
            "מתי כדאי לחזור על הבדיקה?",
        ],
    },
]


def detect_combinations(classified: list) -> list:
    """Return the COMBO_DEFINITIONS entries whose member tests are all
    flagged (borderline/abnormal) in the required direction.
    """
    by_key = {r.test_key: r for r in classified}
    matched = []
    for combo in COMBO_DEFINITIONS:
        ok = True
        for key, needed_direction in combo["tests"].items():
            r = by_key.get(key)
            if not r or r.status not in ("borderline", "abnormal") or r.direction != needed_direction:
                ok = False
                break
        if ok:
            matched.append(combo)
    return matched


def _notes_context_hint(test_key: str, notes: Optional[str]) -> Optional[tuple]:
    """Map scenario/patient notes to a (context_key, display_sentence) pair
    for a given test, using simple explicit keyword matching (never inferred
    silently). Returns None when no relevant context is found.
    """
    if not notes:
        return None
    lowered = notes.strip()
    if test_key == "wbc" and any(k in lowered for k in ("ויראלית", "מחלה", "שפעת")):
        return "recent_illness", f"בהתאם להקשר הקליני שצוין ({notes}), ייתכן קשר אפשרי לתהליך שחלף לאחרונה."
    if test_key == "ferritin" and any(k in lowered for k in ("צמחוני", "טבעוני")):
        return "vegetarian", f"בהתאם להקשר הקליני שצוין ({notes}), הרגלי התזונה עשויים להיות רלוונטיים לרמת הפריטין."
    return None


def build_integration_text(result: ClassifiedLabResult, matched_combos: list, notes: Optional[str]) -> str:
    """Build the per-card 'איך התוצאה משתלבת בתמונה הכוללת?' sentence.

    Combines (when relevant) an active combination, a meaningful trend, and
    scenario context -- personalized per test so cards do not repeat
    identical text. Falls back to a generic sentence when none apply.
    """
    snippets = []

    for combo in matched_combos:
        if result.test_key in combo["tests"]:
            other_keys = [k for k in combo["tests"] if k != result.test_key]
            other_names = " ו".join(LAB_CONFIG[k]["name_he"] for k in other_keys)
            snippets.append(f"מאחר שגם {other_names} מציג/ה ממצא בכיוון דומה, מומלץ לבחון את הממצאים יחד ולא כל אחד בנפרד.")

    if result.trend in ("up", "down"):
        snippets.append(f"בהשוואה לבדיקה הקודמת הערך {TREND_TEXT[result.trend]}, מה שיכול לתת הקשר נוסף מעבר לערך הבודד.")

    context_hint = _notes_context_hint(result.test_key, notes)
    if context_hint:
        snippets.append(context_hint[1])

    if not snippets:
        snippets.append("כרגע אין ממצאים נוספים או הקשר ידוע המשתלבים ישירות עם תוצאה זו, אך המשמעות הכוללת תמיד תלויה בתמונה הרפואית המלאה ובשיחה עם הרופא/ה.")

    return " ".join(snippets)


# =============================================================================
# 9. EXPLANATION ENGINE
# =============================================================================
# This section only *selects and assembles* pre-written, reviewed text
# fragments -- it never generates new medical claims at runtime, and it
# never states a diagnosis.

NORMAL_SUMMARY = "הערך נמצא בטווח הייחוס שהוגדר באב-הטיפוס."
BORDERLINE_LOW_SUMMARY = "הערך מעט מתחת לטווח הייחוס המקובל."
BORDERLINE_HIGH_SUMMARY = "הערך מעט מעל לטווח הייחוס המקובל."
ABNORMAL_LOW_SUMMARY = "הערך נמוך מהטווח המקובל."
ABNORMAL_HIGH_SUMMARY = "הערך גבוה מהטווח המקובל."


def _summary_sentence(result: ClassifiedLabResult) -> str:
    if result.status == "normal":
        return NORMAL_SUMMARY
    if result.status == "borderline":
        return BORDERLINE_LOW_SUMMARY if result.direction == "low" else BORDERLINE_HIGH_SUMMARY
    return ABNORMAL_LOW_SUMMARY if result.direction == "low" else ABNORMAL_HIGH_SUMMARY


def build_explanation(result: ClassifiedLabResult, matched_combos: list, notes: Optional[str], lab_config: dict = LAB_CONFIG) -> dict:
    """Build the full non-diagnostic explanation block for one classified result."""
    test_def = lab_config.get(result.test_key, {})

    possible_reasons = []
    if result.status in ("borderline", "abnormal") and result.direction in ("low", "high"):
        possible_reasons = test_def.get("possible_reasons", {}).get(result.direction, []) or []

    urgency_key = "borderline" if result.status == "borderline" else "abnormal"
    urgency_text = test_def.get("urgency", {}).get(urgency_key, "מומלץ לדון בתוצאה עם רופא/ת המשפחה.")

    return {
        "test_key": result.test_key,
        "name_he": result.name_he,
        "status": result.status,
        "direction": result.direction,
        "summary": _summary_sentence(result),
        "what_it_measures": test_def.get("what_it_measures", ""),
        "possible_reasons": possible_reasons,
        "integration_text": build_integration_text(result, matched_combos, notes),
        "urgency_text": urgency_text,
        "trend_sentence": trend_sentence(result),
    }


def build_explanations(results: list, notes: Optional[str], lab_config: dict = LAB_CONFIG) -> list:
    """Build explanation blocks for every non-normal result, severity-sorted."""
    flagged = [r for r in sort_by_severity(results) if r.status in ("borderline", "abnormal")]
    matched_combos = detect_combinations(results)
    return [build_explanation(r, matched_combos, notes, lab_config) for r in flagged]


# =============================================================================
# 10. PHYSICIAN-QUESTION ENGINE
# =============================================================================
# Questions are never generated freely at runtime -- they are selected
# deterministically from reviewed templates, based on test type, direction,
# severity, trend, other flagged tests, and scenario context.

def select_questions_for_result(result: ClassifiedLabResult, notes: Optional[str], lab_config: dict = LAB_CONFIG) -> list:
    """Select the base question list for a single flagged result (no combo),
    plus a trend-aware question when the trend is meaningful.
    """
    test_def = lab_config.get(result.test_key, {})
    templates = test_def.get("questions", {})
    context_hint = _notes_context_hint(result.test_key, notes)
    context_key = context_hint[0] if context_hint else None

    if context_key and context_key in templates:
        questions = list(templates[context_key])
    else:
        questions = list(templates.get("general", []))

    extra = trend_aware_question(result)
    if extra and extra not in questions:
        questions = questions + [extra]

    return questions


def select_questions_by_test(classified: list, notes: Optional[str] = None, lab_config: dict = LAB_CONFIG) -> dict:
    """Map test_key -> question list for every flagged result. Tests in an
    active combination share the combo's question set (plus a trend-aware
    question if relevant), instead of their individual list.
    """
    matched_combos = detect_combinations(classified)
    used_keys = set()
    by_test: dict = {}
    by_key = {r.test_key: r for r in classified}

    for combo in matched_combos:
        combo_questions = list(combo["questions"])
        for key in combo["tests"]:
            extra = trend_aware_question(by_key[key])
            questions = combo_questions + [extra] if extra and extra not in combo_questions else list(combo_questions)
            by_test[key] = questions
            used_keys.add(key)

    for result in classified:
        if result.status not in ("borderline", "abnormal"):
            continue
        if result.test_key in used_keys:
            continue
        by_test[result.test_key] = select_questions_for_result(result, notes, lab_config)

    return by_test


# =============================================================================
# 11. SUMMARY ENGINE
# =============================================================================

ALL_NORMAL_TEXT = "כל הערכים שנבדקו נמצאים בטווחי הייחוס שהוגדרו באב-הטיפוס."


def build_summary(results: list) -> PatientSummary:
    """Aggregate classified results into a structured, non-diagnostic summary.

    Never diagnoses, and never claims a normal panel proves general health.
    """
    total = len(results)
    normal = [r for r in results if r.status == "normal"]
    borderline = [r for r in results if r.status == "borderline"]
    abnormal = [r for r in results if r.status == "abnormal"]

    severity_sorted_flagged = sort_by_severity(abnormal + borderline)
    key_findings = [r.name_he for r in severity_sorted_flagged]
    non_normal_count = len(borderline) + len(abnormal)

    if total == 0:
        headline = "לא הוזנו תוצאות בדיקה לניתוח."
    elif non_normal_count == 0:
        headline = ALL_NORMAL_TEXT
    else:
        prefix = "מרבית התוצאות בטווח התקין. " if len(normal) > non_normal_count else ""
        if non_normal_count == 1:
            r = severity_sorted_flagged[0]
            word = "חריגה" if r.status == "abnormal" else "גבולית"
            headline = f"{prefix}נמצאה תוצאה אחת {word} שכדאי לדון בה עם רופא/ת המשפחה: {r.name_he}."
        else:
            names = " ו".join([", ".join(key_findings[:-1]), key_findings[-1]]) if len(key_findings) > 1 else key_findings[0]
            headline = f"{prefix}נמצאו {non_normal_count} תוצאות שכדאי לדון בהן עם רופא/ת המשפחה: {names}."

    return PatientSummary(total_tests=total, normal_count=len(normal), borderline_count=len(borderline), abnormal_count=len(abnormal), key_findings=key_findings, headline=headline)


# =============================================================================
# 12. FEEDBACK / EVALUATION LOGIC
# =============================================================================

FEEDBACK_QUESTIONS_CLARITY_OPTIONS = ["כן", "חלקית", "לא"]
FEEDBACK_ANXIETY_OPTIONS = ["כן מאוד", "במידה מסוימת", "לא", "לא רלוונטי"]
FEEDBACK_WOULD_USE_OPTIONS = ["כן", "אולי", "לא"]


def record_feedback(session_state, clarity: int, helpfulness: int, questions_clarity: str, anxiety: str, would_use: str, free_text: str) -> None:
    """Append one feedback submission to the session-only feedback log.

    No real persistence: this is a session-based academic demonstration,
    not a database. Data is lost when the browser session ends.
    """
    if "feedback_log" not in session_state:
        session_state["feedback_log"] = []
    session_state["feedback_log"].append(
        {"clarity": clarity, "helpfulness": helpfulness, "questions_clarity": questions_clarity, "anxiety": anxiety, "would_use": would_use, "free_text": free_text}
    )


# =============================================================================
# 13. UI / CSS HELPERS
# =============================================================================

def inject_global_css() -> None:
    """Inject the global RTL / healthcare visual identity once per page load."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;600;700;800&family=Assistant:wght@400;500;600;700&display=swap');

        :root {
            --me-bg: #EAF6F6; --me-bg-alt: #DCF1EF; --me-card: #FFFFFF; --me-border: #D9EEEC;
            --me-accent: #1E9E96; --me-accent-dark: #12746E; --me-text: #1F2D3D; --me-text-secondary: #5B6B79;
            --me-shadow: rgba(18, 116, 110, 0.10);
            --me-green-bg: #E5F6EC; --me-green-text: #1E7A46;
            --me-yellow-bg: #FFF6DF; --me-yellow-text: #8A6D1B;
            --me-red-bg: #FDEBEA; --me-red-text: #B3261E;
            --me-problem: #C6564E; --me-problem-bg: #FBEDEC;
        }
        html, body, [class*="css"] { direction: rtl; font-family: 'Assistant', 'Heebo', sans-serif; }
        .stApp { background: linear-gradient(180deg, var(--me-bg) 0%, var(--me-bg-alt) 100%); }
        h1, h2, h3, h4 { font-family: 'Heebo', sans-serif; color: var(--me-text); text-align: right; }
        p, span, div, label, li { text-align: right; }
        section[data-testid="stSidebar"] { background-color: #F4FBFA; border-left: 1px solid var(--me-border); }
        section[data-testid="stSidebar"] * { text-align: right; }
        .me-hero { text-align: center; padding: 1.4rem 1rem 0.6rem 1rem; }
        .me-hero h1 { font-size: 2.05rem; font-weight: 800; color: var(--me-accent-dark); line-height: 1.35; }
        .me-hero p { font-size: 1.05rem; color: var(--me-text-secondary); max-width: 640px; margin: 0.6rem auto 0 auto; text-align: center; }
        .me-app-title { font-family: 'Heebo', sans-serif; font-weight: 800; font-size: 1.55rem; color: var(--me-accent-dark); margin-bottom: 0.1rem; }
        .me-app-subtitle { color: var(--me-text-secondary); font-size: 0.95rem; margin-bottom: 1.2rem; }
        .me-card { background: var(--me-card); border: 1px solid var(--me-border); border-radius: 18px; padding: 1.4rem 1.6rem; margin-bottom: 1.1rem; box-shadow: 0 4px 18px var(--me-shadow); }
        .me-card h4 { margin-top: 0; margin-bottom: 0.6rem; font-size: 1.05rem; color: var(--me-accent-dark); }
        .me-section-label { font-weight: 700; font-size: 0.92rem; color: var(--me-accent-dark); margin-top: 0.9rem; margin-bottom: 0.3rem; }
        .me-badge { display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.25rem 0.75rem; border-radius: 999px; font-weight: 700; font-size: 0.85rem; }
        .me-badge-normal { background: var(--me-green-bg); color: var(--me-green-text); }
        .me-badge-borderline { background: var(--me-yellow-bg); color: var(--me-yellow-text); }
        .me-badge-abnormal { background: var(--me-red-bg); color: var(--me-red-text); }
        .me-synthetic-badge { display: inline-block; background: #EFF0FB; color: #4A4FBF; border-radius: 999px; padding: 0.2rem 0.7rem; font-size: 0.8rem; font-weight: 700; margin-inline-start: 0.5rem; }
        .me-trend { font-size: 0.82rem; color: var(--me-text-secondary); margin-right: 0.4rem; }
        .me-trend-line { font-size: 0.88rem; color: var(--me-text-secondary); background: #F4FBFA; border-radius: 10px; padding: 0.4rem 0.8rem; margin-top: 0.3rem; display: inline-block; }
        .me-disclaimer { background: #FFF9EC; border: 1px solid #F3E3B8; border-radius: 14px; padding: 0.9rem 1.2rem; color: #6B5A22; font-size: 0.88rem; margin-bottom: 1rem; }
        .me-question-item { background: #F4FBFA; border-radius: 12px; padding: 0.55rem 0.9rem; margin-bottom: 0.45rem; color: var(--me-text); font-size: 0.94rem; }
        table.me-results-table { width: 100%; border-collapse: separate; border-spacing: 0 8px; }
        table.me-results-table th { text-align: right; font-size: 0.82rem; color: var(--me-text-secondary); font-weight: 600; padding: 0 0.9rem; }
        table.me-results-table td { background: var(--me-card); padding: 0.85rem 0.9rem; font-size: 0.95rem; color: var(--me-text); border-top: 1px solid var(--me-border); border-bottom: 1px solid var(--me-border); }
        table.me-results-table tr td:first-child { border-radius: 0 14px 14px 0; border-right: 1px solid var(--me-border); }
        table.me-results-table tr td:last-child { border-radius: 14px 0 0 14px; border-left: 1px solid var(--me-border); }
        .me-flow-step { background: var(--me-card); border: 1px solid var(--me-border); border-radius: 14px; padding: 0.7rem 0.9rem; text-align: center; font-weight: 600; color: var(--me-accent-dark); box-shadow: 0 2px 10px var(--me-shadow); font-size: 0.92rem; }
        .me-flow-step-problem { background: var(--me-problem-bg); border: 1px solid #F2C9C5; color: var(--me-problem); }
        .me-flow-arrow { text-align: center; color: var(--me-accent); font-size: 1.3rem; line-height: 1.3rem; }
        .me-flow-arrow-problem { color: var(--me-problem); }
        div[data-testid="stMetric"] { background: var(--me-card); border: 1px solid var(--me-border); border-radius: 16px; padding: 0.8rem 1rem; box-shadow: 0 2px 10px var(--me-shadow); }
        .stButton>button { border-radius: 12px; font-weight: 700; background: var(--me-accent); color: white; border: none; }
        .stButton>button:hover { background: var(--me-accent-dark); color: white; }
        button[kind="secondary"] { background: white !important; color: var(--me-accent-dark) !important; border: 1.5px solid var(--me-accent) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header(title: str, subtitle: str = "") -> None:
    subtitle_html = f'<div class="me-app-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="me-app-title">{title}</div>{subtitle_html}', unsafe_allow_html=True)


def render_disclaimer(text: str) -> None:
    st.markdown(f'<div class="me-disclaimer">⚠️ {text}</div>', unsafe_allow_html=True)


def results_to_dataframe(results: list) -> pd.DataFrame:
    """Build a tidy, severity-sorted Pandas DataFrame for the results table."""
    ordered = sort_by_severity(results)
    rows = []
    for r in ordered:
        rows.append({"בדיקה": f"{r.name_he} ({r.abbreviation})", "תוצאה": r.value, "יחידות": r.unit, "טווח ייחוס": r.reference_text, "סטטוס": STATUS_LABELS.get(r.status, r.status), "_status_key": r.status, "_trend": r.trend})
    return pd.DataFrame(rows)


def render_results_table(results: list) -> None:
    """Render the lab results as a calm HTML table, abnormal-first ordering."""
    if not results:
        st.info("לא הוזנו תוצאות בדיקה להצגה.")
        return
    df = results_to_dataframe(results)
    rows_html = ""
    for _, row in df.iterrows():
        badge_class = f"me-badge-{row['_status_key']}"
        icon = STATUS_ICONS.get(row["_status_key"], "")
        trend_html = ""
        if row["_trend"] == "up":
            trend_html = '<span class="me-trend">↑ עלה</span>'
        elif row["_trend"] == "down":
            trend_html = '<span class="me-trend">↓ ירד</span>'
        rows_html += f"<tr><td>{row['בדיקה']}</td><td>{row['תוצאה']}{trend_html}</td><td>{row['יחידות']}</td><td>{row['טווח ייחוס']}</td><td><span class='me-badge {badge_class}'>{icon} {row['סטטוס']}</span></td></tr>"
    st.markdown(f'<table class="me-results-table"><thead><tr><th>בדיקה</th><th>תוצאה</th><th>יחידות</th><th>טווח ייחוס</th><th>סטטוס</th></tr></thead><tbody>{rows_html}</tbody></table>', unsafe_allow_html=True)


def render_explanation_card(explanation: dict, questions: Optional[list]) -> None:
    badge_class = f"me-badge-{explanation['status']}"
    icon = STATUS_ICONS.get(explanation["status"], "")
    label = STATUS_LABELS.get(explanation["status"], explanation["status"])

    if explanation["possible_reasons"]:
        reasons_html = "<ul>" + "".join(f"<li>{r}</li>" for r in explanation["possible_reasons"]) + "</ul>"
    else:
        reasons_html = "<p>אין מידע נוסף זמין עבור כיוון סטייה זה באב-הטיפוס.</p>"

    trend_html = f'<div class="me-trend-line">📈 {explanation["trend_sentence"]}</div>' if explanation.get("trend_sentence") else ""
    questions_html = "".join(f'<div class="me-question-item">🩺 {q}</div>' for q in (questions or []))

    st.markdown(
        f"""
        <div class="me-card">
            <h4>{explanation['name_he']} &nbsp;<span class="me-badge {badge_class}">{icon} {label}</span></h4>
            <div class="me-section-label">סיכום קצר</div>
            <p>{explanation['summary']}</p>
            {trend_html}
            <div class="me-section-label">מה הבדיקה מודדת?</div>
            <p>{explanation['what_it_measures']}</p>
            <div class="me-section-label">מה יכול להשפיע על הערך?</div>
            {reasons_html}
            <div class="me-section-label">איך התוצאה משתלבת בתמונה הכוללת?</div>
            <p>{explanation['integration_text']}</p>
            <div class="me-section-label">מתי כדאי לדבר עם הרופא?</div>
            <p>{explanation['urgency_text']}</p>
            <div class="me-section-label">מה כדאי לשאול את הרופא?</div>
            {questions_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_card(summary: PatientSummary) -> None:
    st.markdown(f'<div class="me-card"><h4>סיכום אישי</h4><p>{summary.headline}</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 תקינים", summary.normal_count)
    c2.metric("🟡 גבוליים", summary.borderline_count)
    c3.metric("🔴 חריגים", summary.abnormal_count)


def render_flow_diagram(steps: list, variant: str = "solution") -> None:
    """Render a simple vertical flow diagram. variant 'problem' uses a
    muted red palette; 'solution' uses the standard teal accent.
    """
    step_class = "me-flow-step me-flow-step-problem" if variant == "problem" else "me-flow-step"
    arrow_class = "me-flow-arrow me-flow-arrow-problem" if variant == "problem" else "me-flow-arrow"
    for i, step in enumerate(steps):
        st.markdown(f'<div class="{step_class}">{step}</div>', unsafe_allow_html=True)
        if i < len(steps) - 1:
            st.markdown(f'<div class="{arrow_class}">↓</div>', unsafe_allow_html=True)


# =============================================================================
# 14. PAGE RENDERING FUNCTIONS
# =============================================================================

def render_full_analysis(lab_results: list, sex: Optional[str], age: Optional[int], notes: Optional[str]) -> None:
    if not lab_results:
        st.info("לא הוזנו תוצאות בדיקה לניתוח. יש להזין לפחות ערך אחד.")
        return

    classified = classify_patient_results(lab_results, sex=sex, age=age)

    summary = build_summary(classified)
    render_summary_card(summary)

    st.markdown("#### תוצאות הבדיקה")
    render_results_table(classified)

    explanations = build_explanations(classified, notes)
    if explanations:
        questions_by_test = select_questions_by_test(classified, notes=notes)
        st.markdown("#### הסברים והמלצות לשיחה עם הרופא/ה")
        for explanation in explanations:
            questions = questions_by_test.get(explanation["test_key"], [])
            render_explanation_card(explanation, questions)
    else:
        st.success("לא נמצאו ממצאים גבוליים או חריגים הדורשים הסבר נוסף.")

    st.caption("המידע המוצג הוא הסבר כללי בלבד ואינו מהווה אבחנה או המלצה רפואית אישית. לכל שאלה לגבי המשמעות הקלינית של התוצאות יש לפנות לרופא/ת המשפחה.")


def render_home_page() -> None:
    st.markdown(
        """
        <div class="me-hero">
            <h1>בדיקות הדם הגיעו. עכשיו אפשר גם להבין אותן.</h1>
            <p>שכבת הסבר חכמה שמתרגמת תוצאות מעבדה לשפה פשוטה, עוזרת להבין מה דורש תשומת לב, ומכינה אותך לשיחה עם הרופא.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_b:
        cta1, cta2 = st.columns(2)
        with cta1:
            if st.button("צפייה בתרחיש לדוגמה", type="primary", use_container_width=True):
                st.session_state["page"] = "תוצאות והסברים"
                st.session_state["dashboard_mode"] = "תרחיש הדגמה סינתטי"
                st.rerun()
        with cta2:
            if st.button("הזנת תוצאות ידנית", type="secondary", use_container_width=True):
                st.session_state["page"] = "תוצאות והסברים"
                st.session_state["dashboard_mode"] = "הזנה ידנית"
                st.rerun()

    st.markdown("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="me-card"><h4>🔍 מבינים את התוצאה</h4><p>הסבר פשוט לכל ערך חריג או גבולי, בלי ז׳רגון רפואי מיותר.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="me-card"><h4>🧩 רואים את התמונה הכוללת</h4><p>התייחסות לשילובים בין בדיקות ולמגמות לאורך זמן, לא רק למספר בודד.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="me-card"><h4>🩺 מגיעים מוכנים לרופא</h4><p>שאלות מותאמות למצב הספציפי שלך, שיעזרו לשיחה להיות ממוקדת יותר.</p></div>', unsafe_allow_html=True)

    render_disclaimer(
        "זהו פרויקט גמר אקדמי להדגמה בלבד (Proof of Concept). המערכת אינה מכשיר רפואי, אינה מספקת אבחנה או ייעוץ רפואי, ומבוססת כולה על נתונים סינתטיים. " + REFERENCE_RANGE_DISCLAIMER
    )


def render_scenario_mode() -> None:
    labels = [f"{s['title']} — {s['name']}" for s in SCENARIOS]
    choice = st.selectbox("בחירת תרחיש הדגמה", options=labels)
    scenario = SCENARIOS[labels.index(choice)]

    sex_label = "גבר" if scenario["sex"] == "male" else "אישה"
    st.markdown(
        f"""
        <div class="me-card">
            <h4>{scenario['name']} <span class="me-synthetic-badge">🧪 תרחיש סינתטי</span></h4>
            <p>גיל: {scenario['age']} &nbsp;|&nbsp; מין: {sex_label}</p>
            <p>{scenario['context']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("מדובר בדמות ובנתונים סינתטיים לחלוטין, שנוצרו לצורך הדגמה בלבד.")

    lab_results = [LabResult(test_key=k, value=v, previous_value=scenario["previous_values"].get(k)) for k, v in scenario["values"].items() if k in LAB_CONFIG]
    render_full_analysis(lab_results, scenario["sex"], scenario["age"], scenario["notes"])


def render_manual_mode() -> None:
    st.markdown('<div class="me-card"><p>ניתן להזין רק את הבדיקות שבוצעו בפועל — כל השדות אופציונליים, והמערכת תנתח רק את מה שהוזן.</p></div>', unsafe_allow_html=True)

    with st.form("manual_input_form"):
        col_age, col_sex = st.columns(2)
        with col_age:
            age_raw = st.text_input("גיל (אופציונלי)", value="")
        with col_sex:
            sex_label = st.selectbox("מין (נדרש עבור בדיקות תלויות מין)", options=["לא צוין", "אישה", "גבר"])

        st.markdown("##### ערכי בדיקות")
        raw_values: dict = {}
        raw_previous: dict = {}

        for key in TEST_ORDER:
            test_def = LAB_CONFIG[key]
            label = f"{test_def['name_he']} ({test_def['abbreviation']}) — {test_def['unit']}"
            if key in TREND_TEST_KEYS:
                c1, c2 = st.columns([2, 1])
                with c1:
                    raw_values[key] = st.text_input(label, value="", key=f"cur_{key}")
                with c2:
                    raw_previous[key] = st.text_input("ערך קודם (אופציונלי)", value="", key=f"prev_{key}")
            else:
                raw_values[key] = st.text_input(label, value="", key=f"cur_{key}")

        notes = st.text_area("הערות רלוונטיות (אופציונלי, לדוגמה: 'מחלה ויראלית לאחרונה')", value="")
        submitted = st.form_submit_button("נתח תוצאות", type="primary")

    if not submitted:
        return

    sex_map = {"אישה": "female", "גבר": "male", "לא צוין": None}
    sex = sex_map[sex_label]

    age: Optional[int] = None
    if age_raw.strip():
        try:
            age = int(float(age_raw.strip()))
        except ValueError:
            st.error("הגיל שהוזן אינו תקין. נא להזין מספר בלבד.")
            return

    lab_results, errors = [], []
    for key in TEST_ORDER:
        raw = raw_values.get(key, "").strip()
        if not raw:
            continue
        try:
            value = validate_value(key, raw)
        except ValidationError as exc:
            errors.append(f"{LAB_CONFIG[key]['name_he']}: {exc}")
            continue

        previous_value = None
        if key in TREND_TEST_KEYS:
            raw_prev = raw_previous.get(key, "").strip()
            if raw_prev:
                try:
                    previous_value = validate_value(key, raw_prev)
                except ValidationError as exc:
                    errors.append(f"{LAB_CONFIG[key]['name_he']} (ערך קודם): {exc}")

        lab_results.append(LabResult(test_key=key, value=value, previous_value=previous_value))

    for err in errors:
        st.error(err)

    if not lab_results and not errors:
        st.warning("לא הוזן אף ערך בדיקה. נא להזין לפחות בדיקה אחת.")
        return

    if lab_results:
        render_full_analysis(lab_results, sex, age, notes)


def render_results_page() -> None:
    render_app_header("תוצאות והסברים", "בחרו תרחיש הדגמה, או הזינו ערכים באופן ידני")

    default_mode = st.session_state.get("dashboard_mode", "תרחיש הדגמה סינתטי")
    options = ["תרחיש הדגמה סינתטי", "הזנה ידנית"]
    mode = st.radio("אופן השימוש", options=options, index=options.index(default_mode), horizontal=True, label_visibility="collapsed")
    st.session_state["dashboard_mode"] = mode

    if mode == "תרחיש הדגמה סינתטי":
        render_scenario_mode()
    else:
        render_manual_mode()


def render_how_it_works_page() -> None:
    render_app_header("איך זה עובד", "מהתוצאה הגולמית ועד לשיחה מוכנה עם הרופא")
    steps = ["תוצאות הבדיקה שלך", "בדיקה שהנתונים תקינים וברורים", "סיווג לפי כללים רפואיים מוגדרים", "הסבר בשפה פשוטה, כולל הקשר ושילובים בין בדיקות", "שאלות מותאמות אישית", "שיחה עם הרופא/ה"]
    render_flow_diagram(steps, variant="solution")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="me-card"><h4>🧪 תרחישי הדגמה</h4><p>12 מטופלים סינתטיים המדגימים מצבים שונים, כולל שילובי ממצאים ומגמות.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="me-card"><h4>✍️ הזנה ידנית</h4><p>כל שדה אופציונלי; המערכת מנתחת רק את מה שהוזן בפועל.</p></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="me-card">
        <h4>📈 מעקב מגמות</h4>
        <p>עבור HbA1c, LDL, המוגלובין, פריטין ו-CRP ניתן להזין גם ערך קודם. המערכת מציגה
        האם הערך עלה או ירד לעומת הבדיקה הקודמת, ומוסיפה שאלה מותאמת כשהמגמה רלוונטית —
        תמיד ללא טענה לגבי הסיבה לשינוי.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_why_it_matters_page() -> None:
    render_app_header("למה זה חשוב?", "הפער בין קבלת תוצאה לבין הבנתה")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p style="text-align:center; font-weight:700; color:#C6564E;">המצב היום</p>', unsafe_allow_html=True)
        render_flow_diagram(["תוצאה חריגה ללא הסבר", "חיפוש עצמאי ברשת", "בלבול / חרדה", "שיחה רפואית פחות ממוקדת"], variant="problem")
    with col2:
        st.markdown('<p style="text-align:center; font-weight:700; color:#12746E;">עם MedExplain</p>', unsafe_allow_html=True)
        render_flow_diagram(["תוצאה", "הסבר פשוט", "הקשר ושילובים", "שאלות ממוקדות", "שיחה רפואית טובה יותר"], variant="solution")

    st.markdown(
        """
        <div class="me-card">
        <p>
        כשמטופל/ת רואה ערך מסומן כחריג בלי הקשר, התגובה הטבעית היא לחפש הסברים באופן
        עצמאי — לעיתים תוך הדבקת מידע רפואי אישי בכלים שלא נועדו לכך. המטרה של
        MedExplain היא לקצר את הפער הזה: לספק הסבר רגוע ומדויק מספיק כדי שהשיחה עם
        הרופא/ה תהיה ממוקדת יותר, ולא להחליף אותה.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_why_not_chatgpt_page() -> None:
    render_app_header("למה לא ChatGPT?", "הבדל בין הדבקת תוצאות בכלי AI כללי לבין כלי ייעודי")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="me-card">
            <h4>🌐 כלי AI ציבורי כללי</h4>
            <ul>
                <li>המטופל מזין מידע ידנית</li>
                <li>תשובה פתוחה ומשתנה</li>
                <li>אין בהכרח מבנה קבוע לבדיקות מעבדה</li>
                <li>ההקשר הרפואי עשוי להיות חלקי</li>
                <li>אין workflow מובנה מול הרופא</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="me-card">
            <h4>🩺 קונספט MedExplain</h4>
            <ul>
                <li>נתוני מעבדה מובנים</li>
                <li>סיווג דטרמיניסטי</li>
                <li>תבנית הסבר מבוקרת</li>
                <li>שאלות המכוונות לשיחה עם רופא</li>
                <li>יכול להשתלב בעתיד בתוך מערכת בריאות</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    render_disclaimer("ה-PoC הנוכחי עצמו אינו מערכת ייצור של קופת חולים, ואינו טוען לרמת אבטחה עדיפה על פני כלים כלליים. מדובר בקונספט עיצובי למערכת עתידית ומשולבת.")


def render_safety_privacy_page() -> None:
    render_app_header("בטיחות ופרטיות", "הגבולות המפורשים של אב-הטיפוס")

    cards = [
        ("🚫 לא מאבחנת", "המערכת אינה קובעת אבחנה רפואית מכל סוג."),
        ("💊 לא ממליצה על טיפול", "אין המלצות על תרופות, מינונים או טיפולים."),
        ("👩‍⚕️ לא מחליפה רופא", "הרופא/ה נותר/ת הסמכות הקלינית הבלעדית."),
        ("📏 טווחי הייחוס משתנים", "הערכים המוצגים הם לדוגמה בלבד, ומשתנים בין מעבדות."),
        ("🧪 נתונים סינתטיים בלבד", "כל התרחישים והשמות באפליקציה בדויים לחלוטין."),
        ("✅ נדרשת ולידציה קלינית", "התוכן הרפואי טרם אושר על-ידי גורם רפואי מוסמך."),
    ]
    cols = st.columns(3)
    for i, (title, text) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(f'<div class="me-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="me-card">
        <h4>סיכונים ומגבלות ידועים</h4>
        <ul>
            <li><b>סיכון הזיה:</b> כל שימוש עתידי ברכיבי AI גנרטיביים דורש בקרה קפדנית.</li>
            <li><b>הרגעת יתר:</b> ניסוח רגוע מדי עלול לגרום להמעטה בחשיבות ממצא.</li>
            <li><b>ניסוח מבהיל שלא לצורך:</b> נמנעים ממנו במכוון, אך יש לוודא זאת בהתמדה.</li>
            <li><b>הבדלים בין אוכלוסיות:</b> טווחי ייחוס עשויים להשתנות בהתאם לגיל, מין ורקע קליני.</li>
            <li><b>אוריינות בריאותית:</b> מטופלים שונים זקוקים לרמות פירוט שונות.</li>
            <li><b>שונות בין מעבדות:</b> טווחי הייחוס בפועל תלויים במעבדה ובשיטת המדידה.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="me-card">
        <h4>פרטיות — מצב נוכחי</h4>
        <ul>
            <li>נתונים סינתטיים בלבד — אין מידע על מטופלים אמיתיים</li>
            <li>אין חיבור לתיק רפואי אמיתי (EHR)</li>
            <li>אין שליחת מידע רפואי לשירות AI חיצוני בזמן ריצת האפליקציה</li>
            <li>אין טענה לרמת אבטחה של מערכת ייצור</li>
        </ul>
        </div>
        <div class="me-card">
        <h4>דרישות עתידיות לפריסה אמיתית (עדיין לא מיושמות)</h4>
        <ul>
            <li>אימות משתמשים והרשאות מבוססות תפקיד</li>
            <li>הצפנה בתעבורה ובמנוחה</li>
            <li>לוגים לבקרה וניטור שוטף</li>
            <li>צמצום מידע מזהה למינימום הנדרש</li>
            <li>ולידציה קלינית פורמלית וסקירת אבטחה</li>
            <li>עמידה בדרישות הפרטיות והרגולציה הרפואית בישראל</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feedback_page() -> None:
    render_app_header("משוב", "עזרו לנו לשפר את אב-הטיפוס")

    with st.form("feedback_form"):
        clarity = st.slider("עד כמה ההסבר היה ברור?", 1, 5, 3)
        helpfulness = st.slider("האם ההסבר עזר לך להבין את משמעות התוצאה?", 1, 5, 3)
        questions_clarity = st.radio("האם ברור לך יותר מה כדאי לשאול את הרופא?", FEEDBACK_QUESTIONS_CLARITY_OPTIONS, horizontal=True)
        anxiety = st.radio("האם ההסבר הפחית את רמת החשש שלך?", FEEDBACK_ANXIETY_OPTIONS, horizontal=True)
        would_use = st.radio("האם היית משתמש/ת בכלי כזה באפליקציית קופת החולים?", FEEDBACK_WOULD_USE_OPTIONS, horizontal=True)
        free_text = st.text_area("מה עדיין לא היה ברור?", value="")
        submitted = st.form_submit_button("שליחת משוב", type="primary")

    if submitted:
        record_feedback(st.session_state, clarity, helpfulness, questions_clarity, anxiety, would_use, free_text)
        st.success("תודה. המשוב נשמר במסגרת ההדגמה.")
        st.markdown(
            f"""
            <div class="me-card">
                <h4>סיכום התשובות שלך</h4>
                <p>בהירות ההסבר: {clarity}/5 &nbsp;|&nbsp; מידת העזרה בהבנה: {helpfulness}/5</p>
                <p>בהירות השאלות לרופא: {questions_clarity} &nbsp;|&nbsp; הפחתת חשש: {anxiety}</p>
                <p>שימוש עתידי באפליקציית קופת חולים: {would_use}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("ביישום עתידי ניתן להשתמש במדדים אלה להערכת בהירות, שימושיות ותחושת מוכנות לשיחה עם הרופא.")

    if st.session_state.get("feedback_log"):
        st.caption(f"משובים שנשלחו בהפעלה הנוכחית: {len(st.session_state['feedback_log'])} (הדגמה מבוססת session בלבד, ללא שמירה קבועה)")

    with st.expander("איך נמדדת הצלחה של כלי כזה?"):
        for title, text in SUCCESS_METRICS:
            st.markdown(f"**{title}** — {text}")


# =============================================================================
# 15. MAIN NAVIGATION / ROUTING
# =============================================================================

def main() -> None:
    st.set_page_config(page_title="MedExplain AI", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")
    inject_global_css()

    if "page" not in st.session_state:
        st.session_state["page"] = PAGES[0]

    with st.sidebar:
        st.markdown("### 🩺 MedExplain AI")
        st.caption("שכבת הסבר לתוצאות בדיקות דם")
        page = st.radio("ניווט", options=PAGES, index=PAGES.index(st.session_state["page"]))
        st.session_state["page"] = page
        st.divider()
        st.caption("נתונים סינתטיים בלבד · אינו מכשיר רפואי · אינו מהווה ייעוץ רפואי")

    if page == "עמוד הבית":
        render_home_page()
    elif page == "תוצאות והסברים":
        render_results_page()
    elif page == "איך זה עובד":
        render_how_it_works_page()
    elif page == "למה זה חשוב?":
        render_why_it_matters_page()
    elif page == "למה לא ChatGPT?":
        render_why_not_chatgpt_page()
    elif page == "בטיחות ופרטיות":
        render_safety_privacy_page()
    elif page == "משוב":
        render_feedback_page()


if __name__ == "__main__":
    main()
