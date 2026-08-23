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

Structured medical data pipeline (see README.md for the full diagram):
    lab_tests.json + scenarios.json
            -> input validation
            -> rule-based classification
            -> Pandas result model
            -> trend + combination analysis
            -> context-aware explanation + personalized questions
            -> patient summary / visit brief
            -> Streamlit presentation

Sections:
    1.  Imports
    2.  Global configuration
    3.  Data loading (lab_tests.json, scenarios.json)
    4.  Data models / dataclasses
    5.  Input validation
    6.  Rule-based classification engine
    7.  Trend analysis
    8.  Combination / context logic
    9.  Pandas result pipeline + summary engine
    10. Explanation engine
    11. Physician-question engine
    12. Visit brief engine
    13. Feedback / evaluation logic
    14. UI / CSS helpers
    15. Page rendering functions
    16. Main navigation / routing
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
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

# Trend outputs: "increased" | "decreased" | "stable" | "no_previous_value"
TREND_TEXT = {"increased": "עלה", "decreased": "ירד", "stable": "ללא שינוי"}
TREND_TEXT_FULL = {
    "increased": "עלה לעומת הבדיקה הקודמת",
    "decreased": "ירד לעומת הבדיקה הקודמת",
    "stable": "ללא שינוי לעומת הבדיקה הקודמת",
}
TREND_ARROWS = {"increased": "↑", "decreased": "↓", "stable": "→"}

REFERENCE_RANGE_DISCLAIMER = (
    "הטווחים המספריים במערכת זו הם ערכי ייחוס לדוגמה למטרות אב-טיפוס לימודי בלבד. "
    "טווחי הייחוס בפועל משתנים בין מעבדות שונות ותלויים בשיטת המדידה, ואינם תחליף "
    "לפרשנות של רופא/ה."
)

VISIT_BRIEF_CHECKLIST = [
    "בדיקות קודמות (לצורך השוואה)",
    "רשימת תרופות ותוספים בשימוש",
    "מידע על תסמינים רלוונטיים, אם קיימים",
]

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
# of application logic. Paths are resolved with pathlib, relative to this
# file's own location, so loading works regardless of the current working
# directory the app or tests are launched from.
#
# lab_tests.json direction / threshold conventions:
#   direction: "both" | "high_only" | "low_only"
#   thresholds: numeric boundaries. Sex-specific tests use "male"/"female"
#     sub-dicts with the same boundary keys:
#       abnormal_low_max   -> below this = abnormal-low
#       normal_min         -> below this (and >= abnormal_low_max) = borderline-low
#       normal_max         -> above this (and < abnormal_high_min) = borderline-high
#       abnormal_high_min  -> at/above this = abnormal-high

BASE_DIR = Path(__file__).resolve().parent


def _load_json(filename: str):
    """Load a required JSON data file from the same directory as app.py.

    Raises a clear, descriptive error if the file is missing or malformed --
    both lab_tests.json and scenarios.json are required for the application
    to function, so failing loudly here is preferable to a confusing error
    deep inside the UI later.
    """
    path = BASE_DIR / filename
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
    status: str        # "normal" | "borderline" | "abnormal"
    direction: str      # "low" | "high" | "normal"
    reference_text: str
    previous_value: Optional[float] = None
    trend: str = "no_previous_value"   # "increased" | "decreased" | "stable" | "no_previous_value"


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
# borderline or abnormal -- only explicit numeric comparisons below, reading
# thresholds from lab_tests.json. The same input always produces the same
# classification. This function is deliberately kept separate from
# generate_explanation() (Section 10) and generate_questions() (Section 11).

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
    "hba1c": "increased",
    "ldl": "increased",
    "hemoglobin": "decreased",
    "ferritin": "decreased",
    "crp": "increased",
}

TREND_QUESTIONS = {
    "hba1c": "האם השינוי לעומת הבדיקה הקודמת משנה את תדירות המעקב המומלצת?",
    "ldl": "האם המגמה לעומת הבדיקה הקודמת משנה את היעד או קצב המעקב המומלצים?",
    "hemoglobin": "האם המגמה לעומת הבדיקה הקודמת מצריכה בירור נוסף מעבר לערך הבודד?",
    "ferritin": "האם המגמה לעומת הבדיקה הקודמת מצריכה מעקב צמוד יותר?",
    "crp": "האם השינוי לעומת הבדיקה הקודמת משנה את מידת הדחיפות לפנייה לרופא/ה?",
}


def compute_trend(value: float, previous_value: Optional[float]) -> str:
    """Return 'increased' / 'decreased' / 'stable' / 'no_previous_value'.

    A purely numeric comparison that never implies why a value changed.
    Deterministic: the same two numbers always produce the same output.
    """
    if previous_value is None:
        return "no_previous_value"
    if value > previous_value:
        return "increased"
    if value < previous_value:
        return "decreased"
    return "stable"


def trend_sentence(result: ClassifiedLabResult) -> Optional[str]:
    """Build the 'בדיקה קודמת: X | בדיקה נוכחית: Y' sentence for display."""
    if result.trend == "no_previous_value" or result.previous_value is None:
        return None
    return f"בדיקה קודמת: {result.previous_value} | בדיקה נוכחית: {result.value} — {TREND_TEXT_FULL[result.trend]}"


def trend_aware_question(result: ClassifiedLabResult) -> Optional[str]:
    """Return an extra physician question when the trend moves in the
    test's 'worsening' direction, or None otherwise. Never implies why the
    value changed -- only that the trend itself may be worth discussing.
    """
    if result.trend not in ("increased", "decreased"):
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
        "explanation": "שני המדדים קשורים להערכת מצב הדם ומאגרי הברזל, ולכן כדאי לדון בשתי התוצאות יחד עם הרופא/ה.",
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
        "explanation": "שני המדדים עשויים להשתנות בהקשרים דלקתיים או זיהומיים, ולכן חשוב לפרש אותם יחד עם התסמינים וההיסטוריה הקלינית.",
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
        "explanation": "פרופיל השומנים נבחן כתמונה כוללת ולא על סמך ערך יחיד בלבד.",
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

    if result.trend in ("increased", "decreased"):
        snippets.append(f"בהשוואה לבדיקה הקודמת הערך {TREND_TEXT[result.trend]}, מה שיכול לתת הקשר נוסף מעבר לערך הבודד.")

    context_hint = _notes_context_hint(result.test_key, notes)
    if context_hint:
        snippets.append(context_hint[1])

    if not snippets:
        snippets.append("כרגע אין ממצאים נוספים או הקשר ידוע המשתלבים ישירות עם תוצאה זו, אך המשמעות הכוללת תמיד תלויה בתמונה הרפואית המלאה ובשיחה עם הרופא/ה.")

    return " ".join(snippets)


# =============================================================================
# 9. PANDAS RESULT PIPELINE + SUMMARY ENGINE
# =============================================================================
# This is where Pandas does real work, not just a listed dependency: the
# DataFrame built here is the single source of truth for severity ordering,
# status counts, and the results table -- every one of those downstream
# consumers reads from this DataFrame rather than recomputing its own view
# of the classified results.

RESULT_COLUMNS = [
    "test_key", "display_name", "abbreviation", "value", "unit",
    "reference_range", "classification", "direction", "previous_value", "trend",
]

ALL_NORMAL_TEXT = "כל הערכים שנבדקו נמצאים בטווחי הייחוס שהוגדרו באב-הטיפוס."


def build_results_dataframe(classified: list) -> pd.DataFrame:
    """Build the structured result table: one row per classified test.

    Columns: test_key, display_name, abbreviation, value, unit,
    reference_range, classification, direction, previous_value, trend.
    The DataFrame is returned already sorted abnormal -> borderline ->
    normal (stable sort, so tests keep their relative order within a
    severity tier) -- this sort is the single source of truth for display
    order used by both the results table and the explanation cards.
    """
    rows = []
    for r in classified:
        rows.append(
            {
                "test_key": r.test_key,
                "display_name": r.name_he,
                "abbreviation": r.abbreviation,
                "value": r.value,
                "unit": r.unit,
                "reference_range": r.reference_text,
                "classification": r.status,
                "direction": r.direction,
                "previous_value": r.previous_value,
                "trend": r.trend,
                "_status_priority": STATUS_PRIORITY.get(r.status, 99),
            }
        )
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS + ["_status_priority"])
    if not df.empty:
        df = df.sort_values("_status_priority", kind="stable").reset_index(drop=True)
    return df


def status_counts(df: pd.DataFrame) -> dict:
    """Count results per classification tier using pandas value_counts()."""
    counts = {"normal": 0, "borderline": 0, "abnormal": 0}
    if df.empty:
        return counts
    tallied = df["classification"].value_counts()
    for status in counts:
        counts[status] = int(tallied.get(status, 0))
    return counts


def sorted_test_keys(df: pd.DataFrame) -> list:
    """Return test_key values in the DataFrame's current (severity) order."""
    return [] if df.empty else df["test_key"].tolist()


def build_summary(df: pd.DataFrame) -> PatientSummary:
    """Aggregate the results DataFrame into a structured, non-diagnostic
    patient summary. Never diagnoses, and never claims a normal panel
    proves general health.
    """
    total = len(df)
    counts = status_counts(df)
    non_normal_count = counts["borderline"] + counts["abnormal"]

    flagged_df = df[df["classification"] != "normal"] if not df.empty else df
    key_findings = flagged_df["display_name"].tolist() if not flagged_df.empty else []

    if total == 0:
        headline = "לא הוזנו תוצאות בדיקה לניתוח."
    elif non_normal_count == 0:
        headline = ALL_NORMAL_TEXT
    else:
        prefix = "מרבית התוצאות בטווח התקין. " if counts["normal"] > non_normal_count else ""
        if non_normal_count == 1:
            row = flagged_df.iloc[0]
            word = "חריגה" if row["classification"] == "abnormal" else "גבולית"
            headline = f"{prefix}נמצאה תוצאה אחת {word} שכדאי לדון בה עם רופא/ת המשפחה: {row['display_name']}."
        else:
            names = " ו".join([", ".join(key_findings[:-1]), key_findings[-1]]) if len(key_findings) > 1 else key_findings[0]
            headline = f"{prefix}נמצאו {non_normal_count} תוצאות שכדאי לדון בהן עם רופא/ת המשפחה: {names}."

    return PatientSummary(
        total_tests=total,
        normal_count=counts["normal"],
        borderline_count=counts["borderline"],
        abnormal_count=counts["abnormal"],
        key_findings=key_findings,
        headline=headline,
    )


# =============================================================================
# 10. EXPLANATION ENGINE
# =============================================================================
# This section only *selects and assembles* pre-written, reviewed text
# fragments -- it never generates new medical claims at runtime, and it
# never states a diagnosis. Deliberately kept separate from
# classify_lab_value() (Section 6) and generate_questions()-equivalent
# logic (Section 11).

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


def generate_explanation(result: ClassifiedLabResult, matched_combos: list, notes: Optional[str], lab_config: dict = LAB_CONFIG) -> dict:
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
        "abbreviation": result.abbreviation,
        "value": result.value,
        "unit": result.unit,
        "status": result.status,
        "direction": result.direction,
        "summary": _summary_sentence(result),
        "what_it_measures": test_def.get("what_it_measures", ""),
        "possible_reasons": possible_reasons,
        "integration_text": build_integration_text(result, matched_combos, notes),
        "urgency_text": urgency_text,
        "trend_sentence": trend_sentence(result),
        "trend": result.trend,
        "previous_value": result.previous_value,
    }


# Backward-compatible alias matching the section's original internal name.
build_explanation = generate_explanation


def build_explanations(classified: list, df: pd.DataFrame, notes: Optional[str], lab_config: dict = LAB_CONFIG) -> list:
    """Build explanation blocks for every non-normal result.

    Ordering comes from the results DataFrame (Section 9) -- the single
    source of truth for severity order -- rather than being recomputed here,
    so the explanation cards and the results table can never disagree about
    display order.
    """
    by_key = {r.test_key: r for r in classified}
    matched_combos = detect_combinations(classified)
    ordered_keys = sorted_test_keys(df)
    flagged_ordered = [by_key[k] for k in ordered_keys if k in by_key and by_key[k].status in ("borderline", "abnormal")]
    return [generate_explanation(r, matched_combos, notes, lab_config) for r in flagged_ordered]


# =============================================================================
# 11. PHYSICIAN-QUESTION ENGINE
# =============================================================================
# Questions are never generated freely at runtime -- they are selected
# deterministically from reviewed templates, based on test type, direction,
# severity, trend, other flagged tests, and scenario context. Deliberately
# kept separate from classify_lab_value() (Section 6) and
# generate_explanation() (Section 10) above.

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


def generate_questions(classified: list, notes: Optional[str] = None, lab_config: dict = LAB_CONFIG) -> dict:
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


# Backward-compatible alias.
select_questions_by_test = generate_questions


# =============================================================================
# 12. VISIT BRIEF ENGINE
# =============================================================================
# The visit brief is not a new source of medical content -- it is a
# deterministic re-assembly of output already produced by Sections 6-11
# (classification, trend, combination, question engines) into a single,
# patient-facing "ready for my appointment" summary. No external LLM call
# happens here or anywhere else at runtime.

def _visit_brief_findings(df: pd.DataFrame) -> list:
    """Section 1: מה בלט בתוצאות -- non-normal findings in severity order."""
    if df.empty:
        return []
    flagged = df[df["classification"] != "normal"]
    return [{"name": row["display_name"], "status": row["classification"]} for _, row in flagged.iterrows()]


def _visit_brief_combinations(matched_combos: list) -> list:
    """Section 2: אילו ממצאים כדאי לשקול יחד -- active combination rules."""
    return [{"title": c["title"], "explanation": c["explanation"]} for c in matched_combos]


def _visit_brief_trends(ordered_classified: list) -> list:
    """Section 3: מה השתנה לעומת בדיקות קודמות -- meaningful trends only."""
    items = []
    for r in ordered_classified:
        if r.trend in ("increased", "decreased") and r.previous_value is not None:
            items.append({"name": r.name_he, "previous": r.previous_value, "current": r.value, "trend": r.trend, "unit": r.unit})
    return items


def _visit_brief_questions(ordered_classified: list, questions_by_test: dict, matched_combos: list, max_questions: int = 5) -> list:
    """Section 4: 3-5 שאלות מותאמות אישית -- deduplicated, combo questions
    first (broadest relevance), capped at max_questions. Never padded with
    invented questions if fewer than 3 genuinely apply.
    """
    seen = set()
    ordered = []
    for combo in matched_combos:
        for q in combo["questions"]:
            if q not in seen:
                seen.add(q)
                ordered.append(q)
    for r in ordered_classified:
        if r.status not in ("borderline", "abnormal"):
            continue
        for q in questions_by_test.get(r.test_key, []):
            if q not in seen:
                seen.add(q)
                ordered.append(q)
    return ordered[:max_questions]


def build_visit_brief(classified: list, df: pd.DataFrame, notes: Optional[str]) -> dict:
    """Assemble the full visit brief from existing deterministic engines.

    Returns a dict with the five sections requested by the product spec:
    findings, combinations, trends, questions, checklist.
    """
    matched_combos = detect_combinations(classified)
    questions_by_test = generate_questions(classified, notes)

    by_key = {r.test_key: r for r in classified}
    ordered_keys = sorted_test_keys(df)
    ordered_classified = [by_key[k] for k in ordered_keys if k in by_key]

    return {
        "findings": _visit_brief_findings(df),
        "combinations": _visit_brief_combinations(matched_combos),
        "trends": _visit_brief_trends(ordered_classified),
        "questions": _visit_brief_questions(ordered_classified, questions_by_test, matched_combos, max_questions=5),
        "checklist": list(VISIT_BRIEF_CHECKLIST),
    }


# =============================================================================
# 13. FEEDBACK / EVALUATION LOGIC
# =============================================================================

FEEDBACK_QUESTIONS_CLARITY_OPTIONS = ["כן", "חלקית", "לא"]
FEEDBACK_ANXIETY_OPTIONS = ["כן מאוד", "במידה מסוימת", "לא", "לא רלוונטי"]
FEEDBACK_WOULD_USE_OPTIONS = ["כן", "אולי", "לא"]


def record_feedback(session_state, clarity: int, helpfulness: int, questions_clarity: str, anxiety: str, would_use: str, free_text: str) -> None:
    """Append one feedback submission to the session-only feedback log.

    No real persistence: this is a session-based academic demonstration,
    not a database, and does not represent a completed clinical usability
    study. Data is lost when the browser session ends.
    """
    if "feedback_log" not in session_state:
        session_state["feedback_log"] = []
    session_state["feedback_log"].append(
        {"clarity": clarity, "helpfulness": helpfulness, "questions_clarity": questions_clarity, "anxiety": anxiety, "would_use": would_use, "free_text": free_text}
    )


# =============================================================================
# 14. UI / CSS HELPERS
# =============================================================================

def _md_html(html: str) -> None:
    """Render a block of HTML via st.markdown, safely.

    BUG THIS PREVENTS: Streamlit's Markdown renderer follows the CommonMark
    spec, under which any line indented 4 or more spaces is treated as a
    literal *code block* -- its content is HTML-escaped and displayed as
    visible text (e.g. "<div class=...>" shown literally on the page)
    instead of being parsed as HTML, even with unsafe_allow_html=True.
    Writing HTML as an indented, triple-quoted Python string (indented to
    match the surrounding code for readability) triggers exactly this rule.

    This was verified empirically with a CommonMark-compliant parser
    (markdown-it-py): the same HTML string, indented, parses as a
    `code_block` and renders as escaped text; dedented so the first
    character is `<` at column 0, it parses as an `html_block` and renders
    normally. See test_classifier.py for a regression test that parses
    every HTML string this app builds and asserts none of them produce a
    code/fence block.

    Every call site in this file that builds multi-line HTML must go
    through this helper (not st.markdown(..., unsafe_allow_html=True)
    directly) so this bug class cannot silently reappear.
    """
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


def inject_global_css() -> None:
    """Inject the global RTL / healthcare visual identity once per page load."""
    _md_html("""<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;600;700;800&family=Assistant:wght@400;500;600;700&display=swap');

:root {
    --me-bg: #EAF6F6; --me-bg-alt: #DCF1EF; --me-card: #FFFFFF; --me-border: #D9EEEC;
    --me-accent: #1E9E96; --me-accent-dark: #12746E; --me-text: #1F2D3D; --me-text-secondary: #5B6B79;
    --me-shadow: rgba(18, 116, 110, 0.10);
    --me-green-bg: #E5F6EC; --me-green-text: #1E7A46;
    --me-yellow-bg: #FFF6DF; --me-yellow-text: #8A6D1B;
    --me-red-bg: #FDEBEA; --me-red-text: #B3261E;
    --me-problem: #C6564E; --me-problem-bg: #FBEDEC;
    --me-combo-bg: #EFF3FF; --me-combo-border: #C9D6FA; --me-combo-text: #33418C;
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

/* Attention section header */
.me-attention-header { font-family: 'Heebo', sans-serif; font-weight: 800; font-size: 1.25rem; color: var(--me-text); margin: 1.1rem 0 0.7rem 0; display: flex; align-items: center; gap: 0.5rem; }

/* "התמונה הכוללת" combination callout -- visually distinct from a regular card */
.me-combo-callout { background: var(--me-combo-bg); border: 1.5px solid var(--me-combo-border); border-radius: 16px; padding: 1.1rem 1.4rem; margin-bottom: 1rem; }
.me-combo-callout h4 { color: var(--me-combo-text); margin-top: 0; margin-bottom: 0.4rem; font-size: 1.0rem; }
.me-combo-callout p { color: var(--me-combo-text); margin: 0; }
.me-combo-icon { font-size: 1.1rem; margin-left: 0.3rem; }

/* Compact visual trend widget */
.me-trend-widget { display: flex; align-items: center; gap: 0.7rem; background: #F4FBFA; border-radius: 14px; padding: 0.7rem 1rem; margin: 0.5rem 0 0.7rem 0; }
.me-trend-box { text-align: center; }
.me-trend-box .me-trend-label { font-size: 0.72rem; color: var(--me-text-secondary); }
.me-trend-box .me-trend-value { font-size: 1.05rem; font-weight: 700; color: var(--me-text); }
.me-trend-arrow-big { font-size: 1.4rem; color: var(--me-accent-dark); font-weight: 800; }
.me-trend-caption { font-size: 0.85rem; color: var(--me-text-secondary); margin-right: auto; }

/* Progressive-disclosure expander styling */
div[data-testid="stExpander"] { background: var(--me-card); border: 1px solid var(--me-border); border-radius: 14px; margin-top: 0.6rem; }

/* Visit brief */
.me-visit-brief { background: var(--me-card); border: 1.5px solid var(--me-accent); border-radius: 18px; padding: 1.5rem 1.7rem; margin-top: 1.2rem; box-shadow: 0 6px 22px var(--me-shadow); }
.me-visit-brief h3 { color: var(--me-accent-dark); margin-top: 0; }
.me-visit-brief-section { margin-top: 1rem; }
.me-visit-brief-section h5 { font-size: 0.95rem; color: var(--me-accent-dark); margin-bottom: 0.4rem; }
.me-checklist-item { padding: 0.3rem 0; color: var(--me-text); }
</style>""")


def render_app_header(title: str, subtitle: str = "") -> None:
    subtitle_html = f'<div class="me-app-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="me-app-title">{title}</div>{subtitle_html}', unsafe_allow_html=True)


def render_disclaimer(text: str) -> None:
    st.markdown(f'<div class="me-disclaimer">⚠️ {text}</div>', unsafe_allow_html=True)


def render_results_table(df: pd.DataFrame) -> None:
    """Render the results DataFrame as a calm HTML table (already
    severity-sorted by build_results_dataframe).
    """
    if df.empty:
        st.info("לא הוזנו תוצאות בדיקה להצגה.")
        return
    rows_html = ""
    for _, row in df.iterrows():
        badge_class = f"me-badge-{row['classification']}"
        icon = STATUS_ICONS.get(row["classification"], "")
        label = STATUS_LABELS.get(row["classification"], row["classification"])
        trend_html = ""
        if row["trend"] == "increased":
            trend_html = '<span style="font-size:0.82rem; color:var(--me-text-secondary); margin-right:0.4rem;">↑ עלה</span>'
        elif row["trend"] == "decreased":
            trend_html = '<span style="font-size:0.82rem; color:var(--me-text-secondary); margin-right:0.4rem;">↓ ירד</span>'
        rows_html += (
            f"<tr><td>{row['display_name']} ({row['abbreviation']})</td>"
            f"<td>{row['value']}{trend_html}</td><td>{row['unit']}</td><td>{row['reference_range']}</td>"
            f"<td><span class='me-badge {badge_class}'>{icon} {label}</span></td></tr>"
        )
    st.markdown(
        f'<table class="me-results-table"><thead><tr><th>בדיקה</th><th>תוצאה</th><th>יחידות</th><th>טווח ייחוס</th><th>סטטוס</th></tr></thead><tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )


def render_combo_callouts(matched_combos: list) -> None:
    """Render the prominent, visually distinct 'התמונה הכוללת' section.
    Appears only when at least one combination rule has fired.
    """
    if not matched_combos:
        return
    st.markdown('<div class="me-attention-header">🧩 התמונה הכוללת</div>', unsafe_allow_html=True)
    for combo in matched_combos:
        _md_html(f"""<div class="me-combo-callout">
    <h4><span class="me-combo-icon">🔗</span>{combo['title']}</h4>
    <p>{combo['explanation']}</p>
</div>""")


def render_trend_widget(explanation: dict) -> str:
    """Build the compact visual trend component HTML for one result, or an
    empty string when no previous value was supplied.

    Returns dedented, stripped HTML (no leading blank line or indentation)
    because this string gets spliced into the middle of a larger HTML block
    built by render_explanation_card(). A leading blank line here would
    terminate that outer HTML block early (per CommonMark, a blank line
    ends an HTML block), and the leftover indentation on the following
    lines would then be misread as a code block -- exactly the bug this
    file works hard to avoid elsewhere. See _md_html() above for the full
    explanation of the underlying rule.
    """
    if explanation.get("trend") not in ("increased", "decreased", "stable") or explanation.get("previous_value") is None:
        return ""
    arrow = TREND_ARROWS.get(explanation["trend"], "→")
    caption = TREND_TEXT_FULL.get(explanation["trend"], "")
    unit = explanation.get("unit", "")
    html = f"""
    <div class="me-trend-widget">
        <div class="me-trend-box">
            <div class="me-trend-label">בדיקה קודמת</div>
            <div class="me-trend-value">{explanation['previous_value']} {unit}</div>
        </div>
        <div class="me-trend-arrow-big">{arrow}</div>
        <div class="me-trend-box">
            <div class="me-trend-label">בדיקה נוכחית</div>
            <div class="me-trend-value">{explanation['value']} {unit}</div>
        </div>
        <div class="me-trend-caption">📈 {caption}</div>
    </div>
    """
    return textwrap.dedent(html).strip()



def render_explanation_card(explanation: dict, questions: Optional[list]) -> None:
    """Render one explanation card with progressive disclosure: primary
    content (result, status, summary, bigger-picture, timing, questions) is
    always visible; secondary educational detail ("what does this measure",
    "what can affect the value") lives inside a collapsed expander.

    Built by joining a list of HTML fragments rather than a single static
    f-string template with an embedded placeholder. That matters: a static
    template like f"...{trend_widget_html}..." leaves a line containing
    only the surrounding indentation whenever trend_widget_html is empty
    (no previous value supplied) -- and a whitespace-only line is a *blank
    line* under CommonMark, which terminates the HTML block right there and
    turns everything after it into a misread code block. Only appending
    trend_widget_html when it is non-empty avoids ever producing that line.
    """
    badge_class = f"me-badge-{explanation['status']}"
    icon = STATUS_ICONS.get(explanation["status"], "")
    label = STATUS_LABELS.get(explanation["status"], explanation["status"])
    trend_widget_html = render_trend_widget(explanation)
    questions_html = "".join(f'<div class="me-question-item">🩺 {q}</div>' for q in (questions or []))

    parts = [
        '<div class="me-card">',
        f'<h4>{explanation["name_he"]} ({explanation["abbreviation"]}) &nbsp;<span class="me-badge {badge_class}">{icon} {label}</span></h4>',
        f'<p style="font-size:1.1rem; font-weight:700; margin:0.2rem 0;">{explanation["value"]} {explanation["unit"]}</p>',
    ]
    if trend_widget_html:
        parts.append(trend_widget_html)
    parts += [
        '<div class="me-section-label">סיכום קצר</div>',
        f'<p>{explanation["summary"]}</p>',
        '<div class="me-section-label">איך התוצאה משתלבת בתמונה הכוללת?</div>',
        f'<p>{explanation["integration_text"]}</p>',
        '<div class="me-section-label">מתי כדאי לדבר עם הרופא?</div>',
        f'<p>{explanation["urgency_text"]}</p>',
        '<div class="me-section-label">מה כדאי לשאול את הרופא?</div>',
        questions_html,
        "</div>",
    ]
    _md_html("".join(parts))

    with st.expander("מידע נוסף"):
        st.markdown(f"**מה הבדיקה מודדת?**  \n{explanation['what_it_measures']}")
        if explanation["possible_reasons"]:
            reasons_html = "".join(f"<li>{r}</li>" for r in explanation["possible_reasons"])
            st.markdown(f"**מה יכול להשפיע על הערך?**", unsafe_allow_html=True)
            st.markdown(f"<ul>{reasons_html}</ul>", unsafe_allow_html=True)
        else:
            st.markdown("אין מידע נוסף זמין עבור כיוון סטייה זה באב-הטיפוס.")


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


def render_visit_brief(brief: dict) -> None:
    """Render the 'הכינו אותי לשיחה עם הרופא/ה' visit brief. Purely a
    presentation of data already computed by build_visit_brief() -- no new
    content is generated here.
    """
    parts = ['<div class="me-visit-brief"><h3>לקראת הפגישה שלך</h3>']

    parts.append('<div class="me-visit-brief-section"><h5>1. מה בלט בתוצאות</h5>')
    if brief["findings"]:
        items = "".join(f"<li>{f['name']} ({STATUS_LABELS.get(f['status'], f['status'])})</li>" for f in brief["findings"])
        parts.append(f"<ul>{items}</ul>")
    else:
        parts.append("<p>כל התוצאות שנבדקו היו בטווח התקין.</p>")
    parts.append("</div>")

    parts.append('<div class="me-visit-brief-section"><h5>2. אילו ממצאים כדאי לשקול יחד</h5>')
    if brief["combinations"]:
        items = "".join(f"<li><b>{c['title']}:</b> {c['explanation']}</li>" for c in brief["combinations"])
        parts.append(f"<ul>{items}</ul>")
    else:
        parts.append("<p>לא זוהה שילוב ממצאים המצריך התייחסות משותפת בבדיקה הנוכחית.</p>")
    parts.append("</div>")

    parts.append('<div class="me-visit-brief-section"><h5>3. מה השתנה לעומת בדיקות קודמות</h5>')
    if brief["trends"]:
        items = "".join(
            f"<li>{t['name']}: {t['previous']} ← {t['current']} {t['unit']} ({TREND_TEXT_FULL[t['trend']]})</li>"
            for t in brief["trends"]
        )
        parts.append(f"<ul>{items}</ul>")
    else:
        parts.append("<p>לא הוזנו ערכים קודמים להשוואה.</p>")
    parts.append("</div>")

    parts.append('<div class="me-visit-brief-section"><h5>4. שאלות מותאמות אישית לרופא</h5>')
    if brief["questions"]:
        items = "".join(f"<li>{q}</li>" for q in brief["questions"])
        parts.append(f"<ol>{items}</ol>")
    else:
        parts.append("<p>לא נמצאו ממצאים המצריכים שאלות ייעודיות מעבר לשיחה השגרתית.</p>")
    parts.append("</div>")

    parts.append('<div class="me-visit-brief-section"><h5>5. מה כדאי להביא לפגישה</h5>')
    items = "".join(f"<li>{c}</li>" for c in brief["checklist"])
    parts.append(f"<ul>{items}</ul></div>")

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# =============================================================================
# 15. PAGE RENDERING FUNCTIONS
# =============================================================================

def render_full_analysis(lab_results: list, sex: Optional[str], age: Optional[int], notes: Optional[str]) -> None:
    """The core results experience. Hierarchy:
    personalized summary -> status metrics -> "התמונה הכוללת" (if a
    combination fired) -> "מה דורש תשומת לב?" (detailed, progressively
    disclosed cards for non-normal results) -> compact full results table
    (secondary reference) -> visit brief CTA.
    """
    if not lab_results:
        st.info("לא הוזנו תוצאות בדיקה לניתוח. יש להזין לפחות ערך אחד.")
        return

    classified = classify_patient_results(lab_results, sex=sex, age=age)
    df = build_results_dataframe(classified)
    summary = build_summary(df)
    matched_combos = detect_combinations(classified)

    render_summary_card(summary)
    render_combo_callouts(matched_combos)

    explanations = build_explanations(classified, df, notes)
    if explanations:
        st.markdown('<div class="me-attention-header">🔎 מה דורש תשומת לב?</div>', unsafe_allow_html=True)
        questions_by_test = generate_questions(classified, notes=notes)
        for explanation in explanations:
            questions = questions_by_test.get(explanation["test_key"], [])
            render_explanation_card(explanation, questions)
    else:
        st.success("לא נמצאו ממצאים גבוליים או חריגים הדורשים הסבר נוסף.")

    with st.expander("📋 כל התוצאות (טבלה מלאה)", expanded=False):
        render_results_table(df)

    st.markdown("")
    if st.button("🗒️ הכינו אותי לשיחה עם הרופא/ה", type="primary"):
        st.session_state["show_visit_brief"] = True

    if st.session_state.get("show_visit_brief"):
        brief = build_visit_brief(classified, df, notes)
        render_visit_brief(brief)

    st.caption("המידע המוצג הוא הסבר כללי בלבד ואינו מהווה אבחנה או המלצה רפואית אישית. לכל שאלה לגבי המשמעות הקלינית של התוצאות יש לפנות לרופא/ת המשפחה.")


def render_home_page() -> None:
    _md_html("""<div class="me-hero">
    <h1>בדיקות הדם הגיעו. עכשיו אפשר גם להבין אותן.</h1>
    <p>שכבת הסבר חכמה שמתרגמת תוצאות מעבדה לשפה פשוטה, עוזרת להבין מה דורש תשומת לב, ומכינה אותך לשיחה עם הרופא.</p>
</div>""")

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_b:
        cta1, cta2 = st.columns(2)
        with cta1:
            if st.button("צפייה בתרחיש לדוגמה", type="primary", use_container_width=True):
                st.session_state["page"] = "תוצאות והסברים"
                st.session_state["dashboard_mode"] = "תרחיש הדגמה סינתטי"
                st.session_state["show_visit_brief"] = False
                st.rerun()
        with cta2:
            if st.button("הזנת תוצאות ידנית", type="secondary", use_container_width=True):
                st.session_state["page"] = "תוצאות והסברים"
                st.session_state["dashboard_mode"] = "הזנה ידנית"
                st.session_state["show_visit_brief"] = False
                st.rerun()

    st.markdown("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="me-card"><h4>🔍 מבינים את התוצאה</h4><p>הסבר פשוט לכל ערך חריג או גבולי, בלי ז׳רגון רפואי מיותר.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="me-card"><h4>🧩 רואים את התמונה הכוללת</h4><p>התייחסות לשילובים בין בדיקות ולמגמות לאורך זמן, לא רק למספר בודד.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="me-card"><h4>🩺 מגיעים מוכנים לרופא</h4><p>שאלות מותאמות למצב הספציפי שלך, ותדריך קצר לקראת הפגישה.</p></div>', unsafe_allow_html=True)

    render_disclaimer(
        "זהו פרויקט גמר אקדמי להדגמה בלבד (Proof of Concept). המערכת אינה מכשיר רפואי, אינה מספקת אבחנה או ייעוץ רפואי, ומבוססת כולה על נתונים סינתטיים. " + REFERENCE_RANGE_DISCLAIMER
    )


def render_scenario_mode() -> None:
    labels = [f"{s['title']} — {s['name']}" for s in SCENARIOS]
    choice = st.selectbox("בחירת תרחיש הדגמה", options=labels)
    scenario = SCENARIOS[labels.index(choice)]

    sex_label = "גבר" if scenario["sex"] == "male" else "אישה"
    _md_html(f"""<div class="me-card">
    <h4>{scenario['name']} <span class="me-synthetic-badge">🧪 תרחיש סינתטי</span></h4>
    <p>גיל: {scenario['age']} &nbsp;|&nbsp; מין: {sex_label}</p>
    <p>{scenario['context']}</p>
</div>""")
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
    if mode != st.session_state.get("dashboard_mode"):
        st.session_state["show_visit_brief"] = False
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

    _md_html("""<div class="me-card">
<h4>📈 מעקב מגמות</h4>
<p>עבור HbA1c, LDL, המוגלובין, פריטין ו-CRP ניתן להזין גם ערך קודם. המערכת מציגה
באופן חזותי האם הערך עלה או ירד לעומת הבדיקה הקודמת, ומוסיפה שאלה מותאמת כשהמגמה
רלוונטית — תמיד ללא טענה לגבי הסיבה לשינוי.</p>
</div>
<div class="me-card">
<h4>🗒️ תדריך לפני הביקור</h4>
<p>בסיום הניתוח ניתן ללחוץ על "הכינו אותי לשיחה עם הרופא/ה" ולקבל תקציר קצר של
הממצאים המרכזיים, השילובים והמגמות, ורשימת שאלות ממוקדת לקראת הפגישה.</p>
</div>""")


def render_why_it_matters_page() -> None:
    render_app_header("למה זה חשוב?", "הפער בין קבלת תוצאה לבין הבנתה")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p style="text-align:center; font-weight:700; color:#C6564E;">המצב היום</p>', unsafe_allow_html=True)
        render_flow_diagram(["תוצאה חריגה ללא הסבר", "חיפוש עצמאי ברשת", "בלבול / חרדה", "שיחה רפואית פחות ממוקדת"], variant="problem")
    with col2:
        st.markdown('<p style="text-align:center; font-weight:700; color:#12746E;">עם MedExplain</p>', unsafe_allow_html=True)
        render_flow_diagram(["תוצאה", "הסבר פשוט", "הקשר ושילובים", "שאלות ממוקדות", "שיחה רפואית טובה יותר"], variant="solution")

    _md_html("""<div class="me-card">
<p>
כשמטופל/ת רואה ערך מסומן כחריג בלי הקשר, התגובה הטבעית היא לחפש הסברים באופן
עצמאי — לעיתים תוך הדבקת מידע רפואי אישי בכלים שלא נועדו לכך. המטרה של
MedExplain היא לקצר את הפער הזה: לספק הסבר רגוע ומדויק מספיק כדי שהשיחה עם
הרופא/ה תהיה ממוקדת יותר, ולא להחליף אותה.
</p>
</div>""")


def render_why_not_chatgpt_page() -> None:
    render_app_header("למה לא ChatGPT?", "הבדל בין הדבקת תוצאות בכלי AI כללי לבין כלי ייעודי")
    col1, col2 = st.columns(2)
    with col1:
        _md_html("""<div class="me-card">
<h4>🌐 כלי AI ציבורי כללי</h4>
<ul>
    <li>המטופל מזין מידע ידנית</li>
    <li>תשובה פתוחה ומשתנה</li>
    <li>אין בהכרח מבנה קבוע לבדיקות מעבדה</li>
    <li>ההקשר הרפואי עשוי להיות חלקי</li>
    <li>אין workflow מובנה מול הרופא</li>
</ul>
</div>""")
    with col2:
        _md_html("""<div class="me-card">
<h4>🩺 קונספט MedExplain</h4>
<ul>
    <li>נתוני מעבדה מובנים</li>
    <li>סיווג דטרמיניסטי</li>
    <li>תבנית הסבר מבוקרת</li>
    <li>שאלות המכוונות לשיחה עם רופא</li>
    <li>יכול להשתלב בעתיד בתוך מערכת בריאות</li>
</ul>
</div>""")
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

    _md_html("""<div class="me-card">
<h4>סיכונים ומגבלות ידועים</h4>
<ul>
    <li><b>סיכון הזיה:</b> כל שימוש עתידי ברכיבי AI גנרטיביים דורש בקרה קפדנית.</li>
    <li><b>הרגעת יתר:</b> ניסוח רגוע מדי עלול לגרום להמעטה בחשיבות ממצא.</li>
    <li><b>ניסוח מבהיל שלא לצורך:</b> נמנעים ממנו במכוון, אך יש לוודא זאת בהתמדה.</li>
    <li><b>הבדלים בין אוכלוסיות:</b> טווחי ייחוס עשויים להשתנות בהתאם לגיל, מין ורקע קליני.</li>
    <li><b>אוריינות בריאותית:</b> מטופלים שונים זקוקים לרמות פירוט שונות.</li>
    <li><b>שונות בין מעבדות:</b> טווחי הייחוס בפועל תלויים במעבדה ובשיטת המדידה.</li>
</ul>
</div>""")

    _md_html("""<div class="me-card">
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
    <li>ניהול סודות מאובטח</li>
    <li>לוגים לבקרה וניטור שוטף</li>
    <li>צמצום מידע מזהה למינימום הנדרש</li>
    <li>ולידציה קלינית פורמלית וסקירת אבטחה</li>
    <li>עמידה בדרישות הפרטיות והרגולציה הרפואית בישראל</li>
</ul>
</div>""")


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
        _md_html(f"""<div class="me-card">
    <h4>סיכום התשובות שלך</h4>
    <p>בהירות ההסבר: {clarity}/5 &nbsp;|&nbsp; מידת העזרה בהבנה: {helpfulness}/5</p>
    <p>בהירות השאלות לרופא: {questions_clarity} &nbsp;|&nbsp; הפחתת חשש: {anxiety}</p>
    <p>שימוש עתידי באפליקציית קופת חולים: {would_use}</p>
</div>""")
        st.caption("ביישום עתידי ניתן להשתמש במדדים אלה להערכת בהירות, שימושיות ותחושת מוכנות לשיחה עם הרופא — לא כתחליף למחקר קליני מלא.")

    if st.session_state.get("feedback_log"):
        st.caption(f"משובים שנשלחו בהפעלה הנוכחית: {len(st.session_state['feedback_log'])} (הדגמה מבוססת session בלבד, ללא שמירה קבועה)")

    with st.expander("איך נמדדת הצלחה של כלי כזה?"):
        for title, text in SUCCESS_METRICS:
            st.markdown(f"**{title}** — {text}")


# =============================================================================
# 16. MAIN NAVIGATION / ROUTING
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
