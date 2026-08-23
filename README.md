# MedExplain AI

**מערכת AI להסבר והנגשת תוצאות בדיקות דם למטופלים**

> This prototype is for educational demonstration only and is not a medical device.

A university Proof of Concept for an Artificial Intelligence in Medicine course. This
README documents the technical depth intentionally kept out of the patient-facing
product itself — the website stays product-focused and free of developer terminology;
this document is where that terminology belongs.

---

## 1. Clinical Problem

Patients increasingly receive digital lab results before speaking with a physician.
An abnormal flag without context can cause unnecessary anxiety, uncontrolled web
searches, personal medical data being pasted into public AI tools, and a less focused
physician conversation. MedExplain AI targets the gap between *receiving* a result and
*understanding* it — it explains, contextualizes, and prepares; it does not diagnose or
treat.

## 2. Target Users

- A patient who received digital blood test results and wants to understand an
  abnormal or borderline value before their appointment.
- Secondarily, this repository serves as an academic demonstration for course
  evaluators assessing both product thinking and technical implementation.

## 3. Product Workflow

From the patient's perspective: choose a demo scenario or enter values manually →
see a personalized summary and status overview → see which findings are worth
discussing together and how any trend fits in → read a plain-language explanation per
finding, with the reasoning and question always visible and deeper background tucked
into an optional "more information" section → optionally generate a short visit brief
to bring into the physician appointment.

## 4. Repository Architecture

The repository is intentionally flat — six files, no folders — while still keeping a
real separation of concerns:

| File | Responsibility |
|---|---|
| `lab_tests.json` | Structured laboratory knowledge base: thresholds, units, sex-specific ranges, plain-language explanations, physician question templates. |
| `scenarios.json` | The 12 synthetic demonstration patients. |
| `app.py` | Rule-based application logic, the Pandas result pipeline, and the patient-facing Streamlit UI. |
| `test_classifier.py` | Automated tests for the classification, trend, combination, results-pipeline, and visit-brief engines. |
| `requirements.txt` | External dependencies. |
| `README.md` | This document. |

A flat layout was chosen for deployment simplicity on Streamlit Community Cloud —
every file uploads directly to the repository root with no import-path or
folder-structure concerns — while still keeping medical data, application logic,
testing, and presentation as genuinely separate, independently reviewable concerns.
`app.py` loads both JSON files at startup using `pathlib`, resolved relative to its own
location (`BASE_DIR = Path(__file__).resolve().parent`), so loading works regardless of
the invoking working directory.

`app.py` is internally organized into 16 numbered sections:

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
| 9 | Pandas result pipeline + summary engine (`build_results_dataframe`, `build_summary`) |
| 10 | Explanation engine (`generate_explanation`) |
| 11 | Physician-question engine (`generate_questions`) |
| 12 | Visit brief engine (`build_visit_brief`) |
| 13 | Feedback / evaluation logic (`record_feedback`) |
| 14 | UI / CSS helpers |
| 15 | Page rendering functions |
| 16 | Main navigation / routing (`main`) |

`classify_lab_value()`, `generate_explanation()`, and `generate_questions()` are kept as
three separate functions rather than one large routine — deliberately, so classification
logic, explanation content, and question selection can each be read, tested, and
modified independently.

Pipeline (also the actual code flow, not only a diagram):

```
lab_tests.json + scenarios.json
              ↓
        Python validation
              ↓
     rule-based classifier
              ↓
      Pandas result model
              ↓
 trends + combination rules
              ↓
 explanations + questions
              ↓
        Streamlit UI
```

## 5. JSON Knowledge Layer

`lab_tests.json` stores, for each of the 8 supported tests (WBC, Hemoglobin, Ferritin,
HbA1c, LDL, HDL, Triglycerides, CRP): a `key`, Hebrew display name, abbreviation, unit,
numeric thresholds (sex-specific sub-objects where relevant), classification
`direction` (`both` / `high_only` / `low_only`), a plain-language description of what
the test measures, possible non-diagnostic reasons for deviation, urgency wording, and
physician question templates. `app.py` loads it with the standard library's `json`
module and never duplicates threshold numbers in Python — the classifier always reads
from this file. There is one source of truth for this information.

`COMBO_DEFINITIONS` and the trend-question constants deliberately stay in `app.py`
rather than in `lab_tests.json`: they describe *application behavior* (which test pairs
to watch for, which follow-up question to add when a trend worsens) rather than raw
medical reference data, so they are colocated in Python with the logic that uses them.

### Reference Range Validation

Every threshold in `lab_tests.json` was reviewed against general clinical/laboratory
medicine knowledge, distinguishing three categories: values with a genuine, universal
population reference range; values that are inherently laboratory- or assay-dependent
(no single correct number exists); and values that are diagnostic/guideline thresholds
rather than a population reference range at all. This review was performed from general
medical knowledge, not a live literature search, so it should be read as a plausibility
check rather than a citation-verified audit.

| Test | Range/threshold used | Unit | Category | Note |
|---|---|---|---|---|
| WBC | 4.0–10.0 normal (3.5/11.0 abnormal bounds) | ×10³/µL | Laboratory-dependent | No universal interval; commonly cited ranges span roughly 4.0–11.0. |
| Hemoglobin (M) | 13.5–17.5 normal | g/dL | Sex-specific, commonly-cited standard | Distinct from WHO's anemia *diagnostic* cutoff (<13.0 g/dL men), which is a different concept. |
| Hemoglobin (F) | 12.0–15.5 normal | g/dL | Sex-specific, commonly-cited standard | WHO anemia cutoff for non-pregnant women is <12.0 g/dL, close to but not identical to this range's lower bound. |
| Ferritin (M) | 30–300 normal | ng/mL | Laboratory-dependent | Varies more between labs/assays than most other tests here. |
| Ferritin (F) | 15–150 normal | ng/mL | Laboratory-dependent | Lower bound aligns with a commonly cited iron-deficiency screening cutoff, but this is a reference range, not a diagnostic cutoff. |
| HbA1c | <5.7% / 5.7–6.4% / ≥6.5% | % | Guideline-based (diagnostic), not a lab reference range | Matches commonly cited ADA-style diagnostic criteria for prediabetes/diabetes. |
| LDL | <130 / 130–159 / ≥160 | mg/dL | Guideline-influenced, simplified | Approximates traditional (NCEP ATP III-style) categories; modern guidance uses individualized, risk-based targets. |
| HDL | <40(M)/<50(F) low | mg/dL | Mixed | Classic "low HDL" cutoffs are often unisex (<40 for both sexes); the sex split shown is a common patient-education convention. |
| Triglycerides | <150 / 150–199 / ≥200 | mg/dL | Guideline-based, well-established | Matches commonly cited categories closely; fasting status materially affects results. |
| CRP | <5 / 5–9.9 / ≥10 | mg/L | Laboratory- and assay-dependent | Standard/conventional CRP only, **not** hs-CRP (a different, more sensitive assay for cardiovascular risk with a much lower scale). |

A production system must use the reporting laboratory's own reference interval rather
than any fixed number, including the ones in this file.

## 6. Synthetic Scenario Dataset

`scenarios.json` holds 12 fictional patients, each with an `id`, `title`, `name`, `age`,
`sex`, `context`, `notes`, `values` (lab results), and optional `previous_values` (for
the trend feature). The scenarios are clinically differentiated, not the same template
with different numbers: an all-normal profile, isolated borderline/abnormal findings
across different tests, two findings meant to be read together (each of the three
combination rules is demonstrated by exactly one scenario), a post-illness WBC bump
with matching context notes, and a mixed-borderline profile. All data is synthetic; no
real patient information is used anywhere in this project.

## 7. Rule-Based Classifier

`classify_lab_value()` is fully deterministic: it compares a validated numeric value
against thresholds loaded from `lab_tests.json` (resolving sex-specific ranges where
applicable) and returns `normal` / `borderline` / `abnormal` plus a direction
(`low`/`high`). **No language model is involved in this decision.** The same input
always produces the same output.

## 8. Pandas Result Pipeline

`build_results_dataframe()` turns the classified results into a DataFrame with columns
`test_key`, `display_name`, `abbreviation`, `value`, `unit`, `reference_range`,
`classification`, `direction`, `previous_value`, `trend` — sorted abnormal → borderline
→ normal at construction time. This DataFrame, not a separately recomputed list, is the
single source of truth used to: sort findings for display, drive the results table, and
compute the status counts (`status_counts()` uses `Series.value_counts()`) that feed
both the on-screen metrics and the summary headline (`build_summary()`). Explanation
card ordering also reads its order from this DataFrame, so the table and the cards can
never disagree about display order.

## 9. Trend Engine

For HbA1c, LDL, Hemoglobin, Ferritin and CRP, an optional previous value can be
supplied (manually, or pre-populated in several demo scenarios). `compute_trend()`
deterministically returns one of `increased` / `decreased` / `stable` /
`no_previous_value` — a purely numeric comparison that never implies why the value
changed. Trend influences three visible things: the result interpretation (a compact
visual "previous → current" widget with a direction arrow and caption), the physician
questions (`trend_aware_question()` appends one extra, test-specific question when the
trend moves in that test's predefined "worsening" direction), and the visit brief's
"what changed since your last test" section.

## 10. Combination / Context Engine

Lab values are not interpreted in isolation. `detect_combinations()` checks whether
three predefined test pairs are jointly flagged in the clinically relevant direction:

- Hemoglobin low + Ferritin low
- WBC high + CRP high
- LDL high + HDL low

When a combination is active, it surfaces twice, deliberately: once as a prominent,
visually distinct "התמונה הכוללת" callout shown as soon as the rule fires (so it is
impossible to miss), and once folded into each affected card's own "how does this fit
the bigger picture" text via `build_integration_text()`, which also incorporates trend
and scenario-context signals (e.g. "recent viral illness" for WBC, "vegetarian diet"
for ferritin) — personalized per card so related cards describe the relationship from
their own angle rather than repeating identical text. The combination engine is
separately unit-tested from the classifier and the question engine.

## 11. Explanation Engine

`generate_explanation()` only *selects and assembles* pre-written, reviewed text
fragments from `lab_tests.json` plus the integration text from Section 10 — it never
generates new medical claims at runtime, and it never states a diagnosis. It is
deliberately a separate function from `classify_lab_value()` and `generate_questions()`.

## 12. Physician-Question Generation

`generate_questions()` selects questions deterministically based on test type,
direction, severity, trend, other flagged tests, and scenario-note context — never
generated freely at runtime. Tests inside an active combination share that combination's
question set (plus a trend-aware question when relevant) instead of duplicating
individual lists. The visit brief's question section (`build_visit_brief()`) reuses this
same engine, deduplicates across combination and individual questions, and caps the
result at five — it never invents extra questions to pad the list.

## 13. Automated Testing

`test_classifier.py` covers: JSON files loading independently of `app.py`, all 8 tests
present, all 12 scenarios structurally valid and classifiable, normal/borderline/
abnormal classification including exact threshold boundaries, sex-specific hemoglobin
logic, invalid/missing/negative/implausible input handling, all four trend outputs, all
three combination rules (and their absence), personalized-question differentiation
(isolated vs. combined findings, with vs. without context notes), the Pandas result
pipeline (columns, sort order, status counts), the visit brief engine (five sections
present, question cap, combination/trend inclusion), end-to-end UI handling of invalid
manual input (via `AppTest`, not just the underlying `validate_value` unit), and a
dedicated HTML-rendering regression suite (see below). 74 tests, all executed and
passing at delivery time — not assumed.

### HTML-rendering regression tests

A deployed build once rendered raw HTML (`<div class="me-section-label">`, etc.) as
visible text inside result cards. Root cause: Streamlit's Markdown renderer follows the
CommonMark spec, under which a line indented 4+ spaces — or a line containing only
whitespace — is treated as a *code block* and its content is HTML-escaped, regardless
of `unsafe_allow_html=True`. HTML built as an indented, triple-quoted Python string
(indented to match the surrounding code for readability), or a static template with a
placeholder that is sometimes empty (leaving a whitespace-only line), triggers exactly
this rule. Fix: every HTML string is now dedented before rendering (`_md_html()`
helper), and `render_explanation_card()` was rebuilt to join a list of non-empty
fragments rather than fill blanks into a static template, so an empty trend widget can
never leave a stray blank line.

Critically, a naive substring check on rendered text (asserting `'<div' not in ...`)
cannot catch this bug class: `AppTest` exposes the *source* string passed to
`st.markdown()`, which legitimately contains `<div`, `class=`, etc. even when
everything renders correctly — that check would fail on every correctly-functioning
card. The real regression test parses each rendered string with a CommonMark-compliant
parser (`markdown-it-py`) and asserts it is recognized as a single `html_block`, never
a `code_block`/`fence` — the same distinction a real browser's renderer makes. This
runs across all 7 pages, all 12 scenarios (including their visit briefs), and manual
input with a combination and a trend active simultaneously.

## 14. Role of AI / LLMs During Development

A language model assisted in writing this codebase and its Hebrew content during
development. At runtime, no model receives patient data and no model decides
classification — every patient-facing sentence is selected deterministically from
pre-written, reviewed content in `lab_tests.json` and `COMBO_DEFINITIONS`. This is a
conscious design choice: it keeps the most clinically sensitive decision fully
auditable and reproducible.

## 15. Privacy-by-Design

**Current PoC:** synthetic data only, no real EHR connection, no patient data sent to
any external LLM at runtime, and no claim of production-grade medical security.

**Future HMO-grade deployment would require** (none of this is implemented here):
authentication, role-based access control (RBAC), encryption in transit and at rest,
audit logging, secure secrets management, monitoring, data minimization, formal
clinical validation, formal security review, and compliance with relevant Israeli
medical privacy/regulatory requirements.

## 16. Safety Boundaries

The system never states a diagnosis (e.g. "you have diabetes/anemia/an infection"),
never recommends treatment or medication, and never claims a normal panel proves
general health. It consistently uses hedged, non-diagnostic language and always frames
the family doctor as the next step and final clinical authority. The visit brief
explicitly avoids treatment advice and diagnosis in every one of its five sections. See
the in-app "בטיחות ופרטיות" page for the patient-facing version of this.

## 17. Evaluation Framework

A real deployment should measure, not assume, that this kind of tool helps. The
in-app feedback page collects: explanation clarity, understanding of the finding's
meaning, preparedness for the physician conversation, usability, reduction in
uncertainty, and willingness to use the tool, plus free text. This is session-only in
the current demo, with no real persistence and no completed clinical study behind it —
the README states this explicitly so the collected variables are understood as
candidate outcomes for future validation, not as evidence of proven effectiveness.

## 18. Limitations

- Reference ranges are illustrative and not a substitute for a specific laboratory's
  own reference interval or a physician's interpretation (see Section 5).
- The system cannot see symptoms, medication history, or full clinical context beyond
  what is entered or included in a scenario.
- The Hebrew-only interface is not adapted for accessibility or health-literacy
  differences across the full population.
- No formal clinical validation of the content has been performed; this is an academic
  demonstration, not a validated clinical tool, and the feedback page does not
  constitute a clinical usability study.

## 19. Future Clinical Deployment Requirements

Beyond the security/privacy items in Section 15: formal clinical review of all medical
content by qualified professionals, expansion of supported tests and scenarios,
accessibility work for a broader patient population, and a real evaluation study using
the metrics in Section 17 before any claim of clinical usefulness.

## Technology Stack

Python · Streamlit · Pandas · JSON · pytest · `markdown-it-py` (test-only, used to
verify HTML renders correctly — see Section 13) · CSS/RTL · rule-based classification ·
structured medical knowledge · context/combination logic · trend analysis ·
LLM-assisted content development · GitHub/GitLab version control · Base44 visual
prototyping (an earlier visual-prototyping stage in this project's history, not part of
the current runtime).

## How to Install

```bash
pip install -r requirements.txt
```

## How to Run

```bash
streamlit run app.py
```

Runs identically locally and on Streamlit Community Cloud. Upload all six files —
`app.py`, `lab_tests.json`, `scenarios.json`, `test_classifier.py`, `requirements.txt`,
`README.md` — directly to the repository root. No folders required.

## Running Tests

```bash
pytest test_classifier.py -v
```

Tests import directly from `app.py`; importing it never triggers Streamlit UI
execution (all UI code is guarded under `if __name__ == "__main__":`), so the
classification, trend, combination, results-pipeline, and visit-brief engines can all
be tested in isolation.

---

**This prototype is for educational demonstration only and is not a medical device.**
