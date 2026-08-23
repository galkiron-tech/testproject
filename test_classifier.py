"""
test_classifier.py
--------------------
Unit tests for MedExplain AI's deterministic classification engine.

This is a flat, root-level test file (not a tests/ package), matching the
flat repository layout. It imports directly from app.py -- importing app.py
never triggers any Streamlit UI code, because all Streamlit calls live
inside main() / page functions, guarded by `if __name__ == "__main__"`.
Only the JSON data loading (lab_tests.json, scenarios.json) happens at
import time, which is plain file I/O with no UI dependency.

Run with:
    pytest test_classifier.py -v
"""

import pytest

from app import (
    LAB_CONFIG,
    SCENARIOS,
    ValidationError,
    LabResult,
    classify_lab_value,
    classify_patient_results,
    compute_trend,
    validate_value,
    sort_by_severity,
    detect_combinations,
    build_summary,
    select_questions_by_test,
    build_integration_text,
    trend_aware_question,
)


# ---------------------------------------------------------------------------
# HbA1c classification
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


# ---------------------------------------------------------------------------
# Hemoglobin: sex-specific classification
# ---------------------------------------------------------------------------

def test_hemoglobin_sex_specific_classification():
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


def test_validate_value_unknown_test_raises():
    with pytest.raises(ValidationError):
        validate_value("not_a_real_test", "5")


def test_classify_lab_value_unknown_test_raises():
    with pytest.raises(ValidationError):
        classify_lab_value("not_a_real_test", 5.0)


# ---------------------------------------------------------------------------
# Trend: up / down
# ---------------------------------------------------------------------------

def test_trend_up():
    assert compute_trend(7.0, 6.5) == "up"


def test_trend_down():
    assert compute_trend(5.5, 6.0) == "down"


def test_trend_unchanged():
    assert compute_trend(5.5, 5.5) == "unchanged"


def test_trend_no_previous_value_returns_none():
    assert compute_trend(5.5, None) is None


def test_trend_aware_question_added_when_worsening():
    results = [LabResult(test_key="hba1c", value=7.1, previous_value=6.6)]
    classified = classify_patient_results(results)
    question = trend_aware_question(classified[0])
    assert question == "האם השינוי לעומת הבדיקה הקודמת משנה את תדירות המעקב המומלצת?"


def test_trend_aware_question_none_when_improving():
    # LDL trending down is NOT the "worsening" direction (concern = up),
    # so no trend-aware question should be added.
    results = [LabResult(test_key="ldl", value=178, previous_value=200)]
    classified = classify_patient_results(results)
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
    by_test = select_questions_by_test(classified)
    assert by_test["wbc"] == by_test["crp"]


def test_combination_integration_text_differs_per_test():
    # Each card in a combination should describe the relationship from its
    # own perspective, not repeat identical text.
    results = [LabResult(test_key="hemoglobin", value=10.4), LabResult(test_key="ferritin", value=8)]
    classified = classify_patient_results(results, sex="female")
    combos = detect_combinations(classified)
    hb = next(r for r in classified if r.test_key == "hemoglobin")
    fe = next(r for r in classified if r.test_key == "ferritin")
    text_hb = build_integration_text(hb, combos, None)
    text_fe = build_integration_text(fe, combos, None)
    assert text_hb != text_fe


# ---------------------------------------------------------------------------
# Determinism, sorting, and skip-unknown behavior
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


def test_severity_sort_abnormal_before_borderline_before_normal():
    results = [
        LabResult(test_key="wbc", value=6.5),     # normal
        LabResult(test_key="ldl", value=178),      # abnormal
        LabResult(test_key="hba1c", value=6.0),     # borderline
    ]
    classified = classify_patient_results(results)
    ordered = sort_by_severity(classified)
    assert [r.status for r in ordered] == ["abnormal", "borderline", "normal"]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def test_summary_all_normal():
    results = [LabResult(test_key="wbc", value=6.5)]
    classified = classify_patient_results(results)
    summary = build_summary(classified)
    assert summary.abnormal_count == 0
    assert summary.borderline_count == 0


def test_summary_counts_multiple_findings():
    results = [
        LabResult(test_key="ldl", value=178),    # abnormal
        LabResult(test_key="hba1c", value=6.0),   # borderline
        LabResult(test_key="wbc", value=6.5),     # normal
    ]
    classified = classify_patient_results(results)
    summary = build_summary(classified)
    assert summary.abnormal_count == 1
    assert summary.borderline_count == 1
    assert summary.normal_count == 1


# ---------------------------------------------------------------------------
# Data integrity: lab_tests.json / scenarios.json loaded correctly
# ---------------------------------------------------------------------------

def test_lab_config_has_all_eight_supported_tests():
    expected = {"wbc", "hemoglobin", "ferritin", "hba1c", "ldl", "hdl", "triglycerides", "crp"}
    assert set(LAB_CONFIG.keys()) == expected


def test_all_scenarios_have_required_fields():
    required = {"id", "name", "age", "sex", "context", "notes", "values"}
    for scenario in SCENARIOS:
        assert required.issubset(scenario.keys())


def test_all_twelve_scenarios_classify_without_error():
    assert len(SCENARIOS) == 12
    for scenario in SCENARIOS:
        results = [
            LabResult(test_key=k, value=v, previous_value=scenario["previous_values"].get(k))
            for k, v in scenario["values"].items()
        ]
        classified = classify_patient_results(results, sex=scenario["sex"], age=scenario["age"])
        assert len(classified) == len(scenario["values"])
        build_summary(classified)
        select_questions_by_test(classified, notes=scenario["notes"])
