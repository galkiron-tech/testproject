"""
test_classifier.py
--------------------
Unit tests for MedExplain AI's deterministic classification, trend,
combination, results-pipeline, and visit-brief engines.

Flat, root-level test file (not a tests/ package), matching the flat
repository layout. Imports directly from app.py -- importing app.py never
triggers any Streamlit UI code, because all Streamlit calls live inside
main() / page functions, guarded by `if __name__ == "__main__"`. Only the
JSON data loading (lab_tests.json, scenarios.json) happens at import time,
which is plain file I/O with no UI dependency.

Run with:
    pytest test_classifier.py -v
"""

import json
from pathlib import Path

import pytest
from markdown_it import MarkdownIt
from streamlit.testing.v1 import AppTest

from app import (
    LAB_CONFIG,
    SCENARIOS,
    BASE_DIR,
    ValidationError,
    LabResult,
    classify_lab_value,
    classify_patient_results,
    compute_trend,
    validate_value,
    detect_combinations,
    build_results_dataframe,
    status_counts,
    sorted_test_keys,
    build_summary,
    generate_questions,
    build_integration_text,
    trend_aware_question,
    build_visit_brief,
)


# ---------------------------------------------------------------------------
# JSON data integrity
# ---------------------------------------------------------------------------

def test_lab_tests_json_file_loads_independently():
    with open(BASE_DIR / "lab_tests.json", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_scenarios_json_file_loads_independently():
    with open(BASE_DIR / "scenarios.json", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)


def test_lab_config_has_all_eight_supported_tests():
    expected = {"wbc", "hemoglobin", "ferritin", "hba1c", "ldl", "hdl", "triglycerides", "crp"}
    assert set(LAB_CONFIG.keys()) == expected


def test_all_scenarios_have_required_fields():
    required = {"id", "name", "age", "sex", "context", "notes", "values"}
    for scenario in SCENARIOS:
        assert required.issubset(scenario.keys())


def test_all_twelve_scenarios_are_valid_and_classify_without_error():
    assert len(SCENARIOS) == 12
    for scenario in SCENARIOS:
        results = [
            LabResult(test_key=k, value=v, previous_value=scenario["previous_values"].get(k))
            for k, v in scenario["values"].items()
        ]
        classified = classify_patient_results(results, sex=scenario["sex"], age=scenario["age"])
        assert len(classified) == len(scenario["values"])


# ---------------------------------------------------------------------------
# HbA1c classification (including exact boundaries)
# ---------------------------------------------------------------------------

def test_hba1c_normal():
    assert classify_lab_value("hba1c", 5.2)["status"] == "normal"


def test_hba1c_borderline():
    result = classify_lab_value("hba1c", 6.0)
    assert result["status"] == "borderline"
    assert result["direction"] == "high"


def test_hba1c_abnormal():
    result = classify_lab_value("hba1c", 7.1)
    assert result["status"] == "abnormal"
    assert result["direction"] == "high"


def test_hba1c_exact_abnormal_boundary_is_abnormal():
    # 6.5 is the configured abnormal_high_min -- boundary is inclusive (>=).
    assert classify_lab_value("hba1c", 6.5)["status"] == "abnormal"


def test_hba1c_just_below_abnormal_boundary_is_borderline():
    assert classify_lab_value("hba1c", 6.4)["status"] == "borderline"


# ---------------------------------------------------------------------------
# WBC classification
# ---------------------------------------------------------------------------

def test_wbc_normal():
    assert classify_lab_value("wbc", 6.5)["status"] == "normal"


def test_wbc_borderline_high():
    result = classify_lab_value("wbc", 10.4)
    assert result["status"] == "borderline"
    assert result["direction"] == "high"


def test_wbc_abnormal_high():
    result = classify_lab_value("wbc", 12.8)
    assert result["status"] == "abnormal"
    assert result["direction"] == "high"


def test_wbc_exact_normal_max_boundary_is_normal():
    # 10.0 is configured normal_max -- boundary is inclusive for "normal".
    assert classify_lab_value("wbc", 10.0)["status"] == "normal"


# ---------------------------------------------------------------------------
# Hemoglobin: sex-specific classification
# ---------------------------------------------------------------------------

def test_hemoglobin_sex_specific_classification_changes_outcome():
    # 12.2 g/dL is within the normal female range but below the normal male
    # range -- proves sex-specific reference ranges genuinely change the
    # classification outcome, not just the displayed reference text.
    female_result = classify_lab_value("hemoglobin", 12.2, sex="female")
    male_result = classify_lab_value("hemoglobin", 12.2, sex="male")
    assert female_result["status"] == "normal"
    assert male_result["status"] in ("borderline", "abnormal")


def test_hemoglobin_female_abnormal_low():
    result = classify_lab_value("hemoglobin", 10.4, sex="female")
    assert result["status"] == "abnormal"
    assert result["direction"] == "low"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_validate_value_missing_input_raises():
    with pytest.raises(ValidationError):
        validate_value("wbc", "")
    with pytest.raises(ValidationError):
        validate_value("wbc", None)


def test_validate_value_negative_input_raises():
    with pytest.raises(ValidationError):
        validate_value("wbc", "-5")


def test_validate_value_non_numeric_input_raises():
    with pytest.raises(ValidationError):
        validate_value("wbc", "abc")


def test_validate_value_implausible_value_raises():
    with pytest.raises(ValidationError):
        validate_value("wbc", "999999")


def test_validate_value_unknown_test_raises():
    with pytest.raises(ValidationError):
        validate_value("not_a_real_test", "5")


def test_classify_lab_value_unknown_test_raises():
    with pytest.raises(ValidationError):
        classify_lab_value("not_a_real_test", 5.0)


# ---------------------------------------------------------------------------
# Trend: increased / decreased / stable / no_previous_value
# ---------------------------------------------------------------------------

def test_trend_increased():
    assert compute_trend(7.0, 6.5) == "increased"


def test_trend_decreased():
    assert compute_trend(5.5, 6.0) == "decreased"


def test_trend_stable():
    assert compute_trend(5.5, 5.5) == "stable"


def test_trend_no_previous_value():
    assert compute_trend(5.5, None) == "no_previous_value"


def test_trend_aware_question_added_when_worsening():
    results = [LabResult(test_key="hba1c", value=7.1, previous_value=6.6)]
    classified = classify_patient_results(results)
    question = trend_aware_question(classified[0])
    assert question == "האם השינוי לעומת הבדיקה הקודמת משנה את תדירות המעקב המומלצת?"


def test_trend_aware_question_none_when_improving():
    # LDL trending down is NOT the "worsening" direction (concern = increased),
    # so no trend-aware question should be added.
    results = [LabResult(test_key="ldl", value=178, previous_value=200)]
    classified = classify_patient_results(results)
    assert trend_aware_question(classified[0]) is None


def test_trend_aware_question_none_without_previous_value():
    results = [LabResult(test_key="hba1c", value=7.1)]
    classified = classify_patient_results(results)
    assert classified[0].trend == "no_previous_value"
    assert trend_aware_question(classified[0]) is None


# ---------------------------------------------------------------------------
# Combination rules
# ---------------------------------------------------------------------------

def test_combination_hemoglobin_ferritin_low():
    results = [LabResult(test_key="hemoglobin", value=10.4), LabResult(test_key="ferritin", value=8)]
    classified = classify_patient_results(results, sex="female")
    keys = [c["key"] for c in detect_combinations(classified)]
    assert "hemoglobin_ferritin_low" in keys


def test_combination_wbc_crp_high():
    results = [LabResult(test_key="wbc", value=12.8), LabResult(test_key="crp", value=18)]
    classified = classify_patient_results(results)
    keys = [c["key"] for c in detect_combinations(classified)]
    assert "wbc_crp_high" in keys


def test_combination_ldl_hdl_risk():
    results = [LabResult(test_key="ldl", value=178), LabResult(test_key="hdl", value=34)]
    classified = classify_patient_results(results, sex="male")
    keys = [c["key"] for c in detect_combinations(classified)]
    assert "ldl_hdl_risk" in keys


def test_combination_none_when_not_flagged():
    results = [LabResult(test_key="wbc", value=6.5)]
    classified = classify_patient_results(results)
    assert detect_combinations(classified) == []


def test_combination_questions_shared_across_member_tests():
    results = [LabResult(test_key="wbc", value=12.8), LabResult(test_key="crp", value=18)]
    classified = classify_patient_results(results)
    by_test = generate_questions(classified)
    assert by_test["wbc"] == by_test["crp"]


def test_combination_integration_text_differs_per_test():
    # Each card in a combination should describe the relationship from its
    # own perspective, not repeat identical text.
    results = [LabResult(test_key="hemoglobin", value=10.4), LabResult(test_key="ferritin", value=8)]
    classified = classify_patient_results(results, sex="female")
    combos = detect_combinations(classified)
    hb = next(r for r in classified if r.test_key == "hemoglobin")
    fe = next(r for r in classified if r.test_key == "ferritin")
    assert build_integration_text(hb, combos, None) != build_integration_text(fe, combos, None)


# ---------------------------------------------------------------------------
# Personalized questions differ across patients
# ---------------------------------------------------------------------------

def test_questions_differ_between_isolated_and_combination_ferritin():
    # Low ferritin alone should generate different questions than low
    # ferritin combined with low hemoglobin.
    isolated = classify_patient_results([LabResult(test_key="ferritin", value=8)], sex="female")
    combined = classify_patient_results(
        [LabResult(test_key="ferritin", value=8), LabResult(test_key="hemoglobin", value=10.4)], sex="female"
    )
    isolated_questions = generate_questions(isolated)["ferritin"]
    combined_questions = generate_questions(combined)["ferritin"]
    assert isolated_questions != combined_questions


def test_questions_differ_with_recent_illness_context():
    plain = classify_patient_results([LabResult(test_key="wbc", value=10.4)])
    with_context = classify_patient_results([LabResult(test_key="wbc", value=10.4)])
    plain_questions = generate_questions(plain, notes=None)["wbc"]
    context_questions = generate_questions(with_context, notes="מחלה ויראלית לאחרונה")["wbc"]
    assert plain_questions != context_questions


# ---------------------------------------------------------------------------
# Determinism and skip-unknown behavior
# ---------------------------------------------------------------------------

def test_classification_is_deterministic_and_repeatable():
    results = [LabResult(test_key="hba1c", value=7.1), LabResult(test_key="wbc", value=6.5)]
    first_pass = classify_patient_results(results, sex="female")
    second_pass = classify_patient_results(results, sex="female")
    assert [r.status for r in first_pass] == [r.status for r in second_pass]


def test_unknown_test_key_skipped_not_raised():
    results = [LabResult(test_key="wbc", value=6.5), LabResult(test_key="not_a_real_test", value=1.0)]
    classified = classify_patient_results(results, sex="male")
    assert len(classified) == 1
    assert classified[0].test_key == "wbc"


# ---------------------------------------------------------------------------
# Pandas result pipeline
# ---------------------------------------------------------------------------

def test_results_dataframe_has_expected_columns():
    results = [LabResult(test_key="wbc", value=6.5)]
    classified = classify_patient_results(results)
    df = build_results_dataframe(classified)
    expected_columns = {
        "test_key", "display_name", "abbreviation", "value", "unit",
        "reference_range", "classification", "direction", "previous_value", "trend",
    }
    assert expected_columns.issubset(set(df.columns))


def test_results_dataframe_sorts_abnormal_before_borderline_before_normal():
    results = [
        LabResult(test_key="wbc", value=6.5),      # normal
        LabResult(test_key="ldl", value=178),       # abnormal
        LabResult(test_key="hba1c", value=6.0),      # borderline
    ]
    classified = classify_patient_results(results)
    df = build_results_dataframe(classified)
    assert df["classification"].tolist() == ["abnormal", "borderline", "normal"]


def test_status_counts_from_dataframe():
    results = [
        LabResult(test_key="ldl", value=178),    # abnormal
        LabResult(test_key="hba1c", value=6.0),   # borderline
        LabResult(test_key="wbc", value=6.5),     # normal
    ]
    classified = classify_patient_results(results)
    df = build_results_dataframe(classified)
    counts = status_counts(df)
    assert counts == {"normal": 1, "borderline": 1, "abnormal": 1}


def test_sorted_test_keys_matches_dataframe_order():
    results = [LabResult(test_key="wbc", value=6.5), LabResult(test_key="ldl", value=178)]
    classified = classify_patient_results(results)
    df = build_results_dataframe(classified)
    assert sorted_test_keys(df) == df["test_key"].tolist()


def test_build_summary_all_normal():
    results = [LabResult(test_key="wbc", value=6.5)]
    classified = classify_patient_results(results)
    df = build_results_dataframe(classified)
    summary = build_summary(df)
    assert summary.abnormal_count == 0
    assert summary.borderline_count == 0


def test_build_summary_counts_multiple_findings():
    results = [
        LabResult(test_key="ldl", value=178),    # abnormal
        LabResult(test_key="hba1c", value=6.0),   # borderline
        LabResult(test_key="wbc", value=6.5),     # normal
    ]
    classified = classify_patient_results(results)
    df = build_results_dataframe(classified)
    summary = build_summary(df)
    assert summary.abnormal_count == 1
    assert summary.borderline_count == 1
    assert summary.normal_count == 1


# ---------------------------------------------------------------------------
# Visit brief engine
# ---------------------------------------------------------------------------

def test_visit_brief_has_five_sections():
    results = [LabResult(test_key="wbc", value=12.8), LabResult(test_key="crp", value=18)]
    classified = classify_patient_results(results)
    df = build_results_dataframe(classified)
    brief = build_visit_brief(classified, df, None)
    assert set(brief.keys()) == {"findings", "combinations", "trends", "questions", "checklist"}


def test_visit_brief_questions_capped_at_five():
    results = [
        LabResult(test_key="wbc", value=12.8),
        LabResult(test_key="crp", value=18),
        LabResult(test_key="ldl", value=178),
        LabResult(test_key="hdl", value=34),
        LabResult(test_key="hba1c", value=7.1),
    ]
    classified = classify_patient_results(results, sex="male")
    df = build_results_dataframe(classified)
    brief = build_visit_brief(classified, df, None)
    assert len(brief["questions"]) <= 5


def test_visit_brief_includes_active_combination():
    results = [LabResult(test_key="hemoglobin", value=10.4), LabResult(test_key="ferritin", value=8)]
    classified = classify_patient_results(results, sex="female")
    df = build_results_dataframe(classified)
    brief = build_visit_brief(classified, df, None)
    titles = [c["title"] for c in brief["combinations"]]
    assert "המוגלובין נמוך + פריטין נמוך" in titles


def test_visit_brief_includes_trend_when_present():
    results = [LabResult(test_key="hba1c", value=7.1, previous_value=6.6)]
    classified = classify_patient_results(results)
    df = build_results_dataframe(classified)
    brief = build_visit_brief(classified, df, None)
    assert len(brief["trends"]) == 1
    assert brief["trends"][0]["trend"] == "increased"


def test_visit_brief_checklist_is_non_empty():
    results = [LabResult(test_key="wbc", value=6.5)]
    classified = classify_patient_results(results)
    df = build_results_dataframe(classified)
    brief = build_visit_brief(classified, df, None)
    assert len(brief["checklist"]) > 0


# ---------------------------------------------------------------------------
# HTML rendering regression tests
# ---------------------------------------------------------------------------
# These tests exist because of a real bug found in a deployed build: HTML
# strings written as indented, triple-quoted Python f-strings (indented to
# match the surrounding code) were misclassified by Streamlit's CommonMark
# markdown renderer as literal *code blocks* rather than HTML, causing raw
# tags like <div class="..."> to render as visible text on the page instead
# of being interpreted as HTML.
#
# IMPORTANT: this is NOT something a naive substring search on rendered text
# (e.g. asserting '<div' not in some_string) can validate -- Streamlit's
# AppTest exposes the *source* string passed to st.markdown(), which
# legitimately contains '<div', 'class=', etc. even when everything is
# working correctly (that's the whole point of building HTML for
# unsafe_allow_html rendering). A substring check would "fail" on every
# correctly-functioning card.
#
# The only way to meaningfully test for this bug is to run the exact same
# string through a CommonMark-compliant parser (matching what Streamlit's
# frontend does) and confirm it is recognized as a single html_block, not a
# code_block/fence -- exactly what the app_html functions below do.

_MD = MarkdownIt("commonmark", {"html": True})


def _find_radio(at, label_substr):
    for r in at.radio:
        if label_substr in (r.label or ""):
            return r
    return None


def _find_text_input(at, label_substr):
    for ti in at.text_input:
        if label_substr in (ti.label or ""):
            return ti
    return None


def _assert_no_misrendered_html(at, label):
    """Fail with a clear message if any markdown block on the page would
    render as an escaped code block instead of real HTML.
    """
    for m in at.markdown:
        tokens = _MD.parse(m.value)
        for t in tokens:
            assert t.type not in ("code_block", "fence"), (
                f"[{label}] A markdown block would render as literal escaped "
                f"HTML instead of real HTML (indentation/blank-line bug). "
                f"Offending content starts with: {m.value[:120]!r}"
            )


@pytest.mark.parametrize(
    "page",
    ["עמוד הבית", "תוצאות והסברים", "איך זה עובד", "למה זה חשוב?", "למה לא ChatGPT?", "בטיחות ופרטיות", "משוב"],
)
def test_no_html_rendering_bug_on_page(page):
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    _find_radio(at, "ניווט").set_value(page).run(timeout=30)
    assert not at.exception
    _assert_no_misrendered_html(at, f"page:{page}")


@pytest.mark.parametrize("scenario_index", list(range(12)))
def test_no_html_rendering_bug_on_scenario(scenario_index):
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    _find_radio(at, "ניווט").set_value("תוצאות והסברים").run(timeout=30)
    options = at.selectbox[0].options
    at.selectbox[0].set_value(options[scenario_index]).run(timeout=30)
    assert not at.exception
    _assert_no_misrendered_html(at, f"scenario:{options[scenario_index]}")

    # Also check the visit brief, which appends more HTML onto the same page.
    for b in at.button:
        if "הכינו אותי" in b.label:
            b.click().run(timeout=30)
            assert not at.exception
            _assert_no_misrendered_html(at, f"visit_brief:{options[scenario_index]}")
            break


def test_no_html_rendering_bug_in_manual_mode_with_combination_and_trend():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    _find_radio(at, "ניווט").set_value("תוצאות והסברים").run(timeout=30)
    _find_radio(at, "אופן השימוש").set_value("הזנה ידנית").run(timeout=30)
    _find_text_input(at, "ספירת תאי דם לבנים").set_value("12.8")
    _find_text_input(at, "חלבון C-reactive").set_value("18")
    _find_text_input(at, "המוגלובין מסוכרר").set_value("7.1")
    at.button[0].click().run(timeout=30)
    assert not at.exception
    _assert_no_misrendered_html(at, "manual_mode_combo_and_trend")


@pytest.mark.parametrize(
    "field_label,bad_value",
    [
        ("ספירת תאי דם לבנים", "abc"),
        ("ספירת תאי דם לבנים", "-5"),
        ("גיל", "not-a-number"),
    ],
)
def test_manual_mode_invalid_input_shows_error_without_crashing(field_label, bad_value):
    """End-to-end UI check (not just the unit-level validate_value test
    above): confirms the actual Streamlit manual-input form catches invalid
    values, displays a calm error, and does not raise an uncaught exception.
    """
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    _find_radio(at, "ניווט").set_value("תוצאות והסברים").run(timeout=30)
    _find_radio(at, "אופן השימוש").set_value("הזנה ידנית").run(timeout=30)
    _find_text_input(at, field_label).set_value(bad_value)
    at.button[0].click().run(timeout=30)
    assert not at.exception
    assert len(at.error) > 0


def test_manual_mode_empty_submission_warns_without_crashing():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    _find_radio(at, "ניווט").set_value("תוצאות והסברים").run(timeout=30)
    _find_radio(at, "אופן השימוש").set_value("הזנה ידנית").run(timeout=30)
    at.button[0].click().run(timeout=30)
    assert not at.exception
    assert len(at.warning) > 0
