# MedExplain AI

**מערכת AI להסבר והנגשת תוצאות בדיקות דם למטופלים**

> This prototype is for educational demonstration only and is not a medical device.

A university Proof of Concept for an Artificial Intelligence in Medicine course. This
README documents the technical depth intentionally kept out of the patient-facing
product itself — the website stays product-focused and free of developer terminology;
this document is where that terminology belongs.

---

## 1. Project Overview

MedExplain AI is a patient-facing explanation layer for blood test results. It is
designed as a concept for a feature that could plausibly exist inside an Israeli HMO
app: it translates raw lab values into plain-language explanations, shows how findings
relate to each other and to trends over time, and prepares the patient with specific
questions for their physician.

## 2. Clinical Problem

Patients increasingly receive digital lab results before speaking with a physician.
An abnormal flag without context can cause unnecessary anxiety, uncontrolled web
searches, personal medical data being pasted into public AI tools, and a less focused
physician conversation. MedExplain AI targets the gap between *receiving* a result and
*understanding* it.

## 3. Product Concept

The product is deliberately narrow in scope: it explains, contextualizes, and prepares
— it does not diagnose or treat. Every explanation is written to support the upcoming
physician conversation, not replace it. The three-benefit framing on the home page
(understand the result / see the full picture / arrive prepared) is the product's core
value proposition.

## 4. Target User

- A patient who received digital blood test results and wants to understand an
  abnormal or borderline value before their appointment.
- Secondarily, this repository serves as an academic demonstration for course
  evaluators assessing both product thinking and technical implementation.

## 5. Architecture

The repository is intentionally flat — six files, no folders — while still keeping a
real separation of concerns:

| File | Responsibility |
|---|---|
| `lab_tests.json` | Structured laboratory knowledge base: thresholds, units, sex-specific ranges, plain-language explanations, physician question templates. |
| `scenarios.json` | The 12 synthetic demonstration patients. |
| `app.py` | Rule-based application logic and the patient-facing Streamlit UI. |
| `test_classifier.py` | Unit tests for the classification engine. |
| `requirements.txt` | External dependencies. |
| `README.md` | This document. |

A flat layout was chosen for deployment simplicity on Streamlit Community Cloud — every
file uploads directly to the repository root with no import-path or folder-structure
concerns — while still keeping medical data, application logic, testing, and
presentation as genuinely separate, independently reviewable concerns rather than one
undifferentiated blob of code.

`app.py` loads both JSON files at startup (paths resolved relative to its own location,
so it works regardless of the invoking working directory) and is itself internally
organized into 15 numbered sections:

| # | Section |
|---|---|
| 1 | Imports |
| 2 | Global configuration |
| 3 | Data loading (`lab_tests.json`, `scenarios.json`) |
| 4 | Data models (`LabResult`, `ClassifiedLabResult`, `PatientSummary`) |
| 5 | Input validation (`validate_value`) |
| 6 | Rule-based classification engine (`classify_lab_value`) |
| 7 | Trend analysis (`compute_trend`, `trend_aware_question`) |
| 8 | Combination / context logic (`COMBO_DEFINITIONS`, `detect_combinations`, `build_integration_text`) |
| 9 | Explanation engine (`build_explanation`) |
| 10 | Physician-question engine (`select_questions_by_test`) |
| 11 | Summary engine (`build_summary`) |
| 12 | Feedback / evaluation logic (`record_feedback`) |
| 13 | UI / CSS helpers |
| 14 | Page rendering functions |
| 15 | Main navigation / routing (`main`) |

`COMBO_DEFINITIONS` and the trend-question constants stay in `app.py` rather than in
`lab_tests.json`: they describe *application behavior* (which test pairs to watch for,
which follow-up question to add when a trend worsens) rather than raw medical reference
data, so they are colocated in Python with the logic that uses them.

Data flow:

```
תוצאות בדיקה (תרחיש הדגמה או הזנה ידנית)
        ↓
בדיקת תקינות קלט
        ↓
סיווג דטרמיניסטי מבוסס-כללים (thresholds from lab_tests.json)
        ↓
זיהוי שילובים ומגמות רלוונטיים
        ↓
בחירת הסבר, אינטגרציה להקשר, ושאלות מותאמות
        ↓
תצוגה למטופל
```

## 6. Structured Knowledge Base

`lab_tests.json` stores, for each of the 8 supported tests (WBC, Hemoglobin, Ferritin,
HbA1c, LDL, HDL, Triglycerides, CRP): a `key`, Hebrew display name, abbreviation, unit,
numeric thresholds (sex-specific sub-objects where relevant), classification
`direction` (`both` / `high_only` / `low_only`), a plain-language description of what
the test measures, possible non-diagnostic reasons for deviation, urgency wording, and
physician question templates. `app.py` loads it with the standard library's `json`
module and never duplicates threshold numbers in Python — the classifier always reads
from this file.

**Reference ranges are illustrative, for demonstration purposes only.** See Section 8
for a full validation pass distinguishing genuine lab-dependent values from
guideline/diagnostic thresholds.

## 7. Synthetic Scenarios

`scenarios.json` holds 12 fictional patients, each with an `id`, `title`, `name`, `age`,
`sex`, `context`, `notes`, `values` (lab results), and optional `previous_values` (for
the trend feature). All data is synthetic; no real patient information is used anywhere
in this project.

## 8. Reference Range Validation

Every threshold in `lab_tests.json` was reviewed against general clinical/laboratory
medicine knowledge, distinguishing three categories: values with a genuine, universal
population reference range; values that are inherently laboratory- or assay-dependent
(no single correct number exists); and values that are diagnostic/guideline thresholds
rather than a population reference range at all. This review was performed from general
medical knowledge, not a live literature search, so it should be read as a plausibility
check rather than a citation-verified audit. No numeric threshold was changed as a
result of this pass, since none was found to be clearly incorrect; several explanatory
text fields were expanded to state these distinctions explicitly to the patient (see
`what_it_measures` for hemoglobin, ferritin, LDL, HDL, and CRP in `lab_tests.json`).

| Test | Range/threshold used | Unit | Category | Note |
|---|---|---|---|---|
| WBC | 4.0–10.0 normal (3.5/11.0 abnormal bounds) | ×10³/µL | Laboratory-dependent | No universal interval; commonly cited ranges span roughly 4.0–11.0. The values used sit within that commonly-cited band. |
| Hemoglobin (M) | 13.5–17.5 normal | g/dL | Sex-specific, commonly-cited standard | Widely used "textbook" range; distinct from WHO's anemia *diagnostic* cutoff (<13.0 g/dL men), which is a different concept. |
| Hemoglobin (F) | 12.0–15.5 normal | g/dL | Sex-specific, commonly-cited standard | WHO anemia cutoff for non-pregnant women is <12.0 g/dL, close to but not identical to this reference range's lower bound. |
| Ferritin (M) | 30–300 normal | ng/mL | Laboratory-dependent | Ferritin reference ranges vary more between labs/assays than most other tests here. |
| Ferritin (F) | 15–150 normal | ng/mL | Laboratory-dependent | Lower bound (15) aligns with a commonly cited iron-deficiency screening cutoff, but this is a reference range, not a diagnostic cutoff. |
| HbA1c | <5.7% normal / 5.7–6.4% borderline / ≥6.5% abnormal | % | Guideline-based (diagnostic), not a lab reference range | Matches commonly cited ADA-style diagnostic criteria for prediabetes/diabetes. High confidence, but not re-verified against a live current guideline document. |
| LDL | <130 normal / 130–159 borderline / ≥160 abnormal | mg/dL | Guideline-influenced, simplified | Approximates traditional (NCEP ATP III-style) categories. Modern cardiology guidance increasingly uses individualized, risk-based LDL targets rather than one fixed population cutoff — stated explicitly in-app. |
| HDL | <40(M)/<50(F) low | mg/dL | Mixed: guideline value is often unisex; sex split is a common education convention | Classic "low HDL" cardiovascular-risk cutoffs are often stated as <40 mg/dL for both sexes; the sex split shown here is a common patient-education convention, now noted explicitly in-app. |
| Triglycerides | <150 normal / 150–199 borderline / ≥200 abnormal | mg/dL | Guideline-based, well-established | Matches commonly cited (NCEP ATP III-style) categories closely; fasting status materially affects results, reflected in the in-app physician questions. |
| CRP | <5 normal / 5–9.9 borderline / ≥10 abnormal | mg/L | Laboratory- and assay-dependent | Applies to standard/conventional CRP only, **not** hs-CRP (a different, more sensitive assay used for cardiovascular risk with a much lower, distinct scale — stated explicitly in-app). |

**Flagged rather than asserted:** the exact numeric boundary between "borderline" and
"abnormal" for WBC, ferritin, and standard CRP cannot be stated as a single correct
value — different laboratories legitimately publish different numbers. A production
system must use the reporting laboratory's own reference interval rather than any fixed
number, including the ones in this file. Anyone relying on this table for something
beyond classroom demonstration should re-verify each figure against a current, primary
clinical source (e.g. current ADA Standards of Care, current AHA/ACC or ESC lipid
guidelines, and the specific reporting laboratory's published reference intervals).

## 9. Rule-Based Classification

`classify_lab_value()` is fully deterministic: it compares a validated numeric value
against thresholds loaded from `lab_tests.json` (resolving sex-specific ranges where
applicable) and returns `normal` / `borderline` / `abnormal` plus a direction
(`low`/`high`). **No language model is involved in this decision.** The same input
always produces the same output — this is intentionally the most auditable part of the
system, since it is the most clinically sensitive. Results are then severity-sorted
(abnormal → borderline → normal) for both the results table and explanation cards.

## 10. Combination / Context Logic

Lab values are not interpreted in isolation. `detect_combinations()` checks whether
predefined test pairs are jointly flagged in the clinically relevant direction:

- Hemoglobin low + Ferritin low
- WBC high + CRP high
- LDL high + HDL low

When a combination is active, the affected tests share a combined explanation and
question set instead of duplicating generic per-test content. Separately,
`build_integration_text()` builds a personalized "how does this fit the bigger picture"
sentence for every explanation card, combining (when relevant) an active combination, a
meaningful trend, and any scenario context notes (e.g. "recent viral illness" for WBC,
"vegetarian diet" for ferritin) — each card's text is generated from that card's own
signals, so related cards describe the same relationship from their own perspective
rather than repeating identical text.

## 11. Trend Analysis

For HbA1c, LDL, Hemoglobin, Ferritin and CRP, an optional previous value can be
supplied (manually, or pre-populated in several demo scenarios). `compute_trend()`
reports `up` / `down` / `unchanged` — a purely numeric comparison that never implies why
the value changed. When a trend moves in a test's predefined "worsening" direction
(e.g. HbA1c trending up, hemoglobin trending down), `trend_aware_question()` appends one
extra, test-specific physician question — trends can therefore change *which* questions
a patient sees, not just the displayed numbers.

## 12. Technology Stack

| Component | Role |
|---|---|
| Python | Core application logic, deterministic classification engine |
| Streamlit | Interactive product UI |
| Pandas | Structuring the results table |
| JSON | Structured medical knowledge (`lab_tests.json`) and demonstration data (`scenarios.json`), loaded via the standard library `json` module |
| CSS / RTL | Embedded styling for the Hebrew right-to-left healthcare visual identity |
| Rule-based classification | Deterministic threshold logic in `app.py`, reading configuration from `lab_tests.json` |
| Context / combination rules | `detect_combinations`, `build_integration_text` — cross-test pattern recognition |
| Trend logic | `compute_trend`, `trend_aware_question` — numeric trend detection and trend-aware follow-up questions |
| pytest | Unit tests in `test_classifier.py` |
| LLM-assisted content development | Used during development for Hebrew copywriting, explanation drafting, and code structuring — **not** used at runtime for clinical classification |
| GitHub/GitLab | Version control during development; not part of the running application |
| Base44 | An earlier visual-prototyping stage in this project's history, not part of the current runtime |

## 13. Role of AI / LLMs

A language model assisted in writing this codebase and its Hebrew content during
development. At runtime, no model receives patient data and no model decides
classification — every patient-facing sentence is selected deterministically from
pre-written, reviewed content in `lab_tests.json` and `COMBO_DEFINITIONS`. This is a
conscious design choice: it keeps the most clinically sensitive decision fully
auditable and reproducible.

## 14. Safety Boundaries

The system never states a diagnosis (e.g. "you have diabetes/anemia/an infection"),
never recommends treatment or medication, and never claims a normal panel proves
general health. It consistently uses hedged, non-diagnostic language ("the value is
above/below the typical range", "it's worth discussing with your family doctor") and
always frames the family doctor as the next step and final clinical authority. See the
in-app "בטיחות ופרטיות" page for the patient-facing version of this.

## 15. Privacy-by-Design

**Current PoC:** synthetic data only, no real EHR connection, no patient data sent to
any external LLM at runtime, and no claim of production-grade medical security.

**Future HMO-grade deployment would require** (none of this is implemented here):
authentication, role-based access control, encryption in transit and at rest, audit
logging, access logging, monitoring, data minimization, formal clinical validation,
formal security review, and compliance with relevant Israeli medical privacy
regulation.

## 16. Limitations

- Reference ranges are illustrative and not a substitute for a specific laboratory's
  own reference interval or a physician's interpretation (see Section 8).
- The system cannot see symptoms, medication history, or full clinical context beyond
  what is entered or included in a scenario.
- The Hebrew-only interface is not adapted for accessibility or health-literacy
  differences across the full population.
- No formal clinical validation of the content has been performed; this is an academic
  demonstration, not a validated clinical tool.

## 17. Future Implementation Requirements

Beyond the security/privacy items in Section 15: formal clinical review of all medical
content by qualified professionals, expansion of supported tests and scenarios,
accessibility work for a broader patient population, and a real evaluation study using
the metrics below before any claim of clinical usefulness.

## 18. Evaluation Metrics

A real deployment should measure, not assume, that this kind of tool helps. The
in-app feedback page collects signal toward each of these (session-only in this demo,
with no real persistence):

- **Explanation clarity** — how easy the wording was to understand.
- **Patient understanding** — whether the patient leaves with an accurate picture of
  the finding.
- **Preparedness for the physician conversation** — whether the patient feels they have
  focused questions going into the appointment.
- **Usability** — how straightforward the tool was to use.
- **Reduction in uncertainty** — whether initial worry or confusion decreased after
  reading the explanation.
- **Ability to identify appropriate physician questions** — whether the generated
  questions were actually relevant to the patient's specific situation.

## 19. How to Install

```bash
pip install -r requirements.txt
```

## 20. How to Run

```bash
streamlit run app.py
```

Runs identically locally and on Streamlit Community Cloud. Upload all six files —
`app.py`, `lab_tests.json`, `scenarios.json`, `test_classifier.py`, `requirements.txt`,
`README.md` — directly to the repository root. No folders required.

## 21. Running Tests

```bash
pytest test_classifier.py -v
```

Tests import directly from `app.py`; importing it never triggers Streamlit UI
execution (all UI code is guarded under `if __name__ == "__main__":`), so the
classification engine, trend logic, and combination logic can all be tested in
isolation.

---

**This prototype is for educational demonstration only and is not a medical device.**
