# Extraction diagnostic methodology

Notebook references from the initial stage are superseded by root notebook 06. The diagnostic files below are preserved; current ranking evaluation is in `results/final/`.

# Russian validation and compatibility review

This is an additive, opt-in extension of the first-half prototype. Nothing in
notebooks 01–06, the default `dictionary.json`, extraction rules, M0–M8 outputs,
clean data, or embedding caches has been changed. Read `validation/compatibility_check.json`
and `validation/pre_change_compatibility.json` before any future merge.

## Compatibility established before changes

Both sides have the same complete ID sets as the existing pipeline, without
missing or duplicate entities, and normalized titles agree. Raw sidecar order
is different from clean/M8 catalog order: **never join by row number**.
`compatibility.align_sidecar` validates IDs, side, dictionary identity and row
uniqueness, then returns evidence in the requested ID order. All added columns
use `m9_cv_` or `m9_vacancy_` prefixes, preserving existing `cv_skills`, `job_skills`,
text inputs and model scores. Raw snapshots remain distinct from cleaned M8 text;
matching titles do not prove every source field is identical. The adapter does
not change region policy, train/test split, candidate pools, model input, ranking
or evaluation. Existing ranking experiments have not been rerun.

The real-data check aligns all 65,714 resume IDs and 43,756 vacancy IDs. A 300-row
integration sample attaches both sidecars and executes the original notebook 04
text functions; all original columns and produced lexical inputs are unchanged.
These are data-contract checks, not a promise of future ranking quality.

## Expanded dictionary

`dictionary.v2.json` is opt-in: 47 canonical skills versus the original 29.
All original skill IDs and definitions are unchanged. Eighteen activities were
added with 33 conservative, exactly attested aliases. Evidence records the raw
vacancy ID, source field, original phrase and title. Expansion sources exclude
all 40 corpus diagnostic entity IDs. No resume text or recruiter outcome was
used to choose new aliases. All ESCO links remain null.

The additions have **agent review**, with human approval pending. They have not
been approved by a Russian-speaking human. The original dictionary remains the
default until review. Exact aliases are intentionally narrow; modest coverage
gains are preferable to ambiguous broad terms introduced just to raise coverage.

## What was validated

`validation/russian_validation_cases.csv` contains 80 explicit linguistic cases:

- 40 real structured vacancy phrases: a deterministic sample of 40 unique example
  IDs among the top 400 frequency-ranked candidate phrases. One equipment phrase
  is ambiguous and excluded until a human reviews its context, leaving 39 scored.
- 40 authored, dictionary-informed challenge examples, including negation,
  punctuation, multiple skills, inflections, aspirations and coordinated negation.

Annotations were written before predictions. Corpus entity IDs are disjoint from
new dictionary source IDs; phrase/vocabulary overlap can still exist. This is a
frequency-enriched diagnostic set, not an unbiased, independent test or a measured
recommendation improvement. It includes 26 corpus phrases with activities/traits
outside the frozen inventory; these are documented gaps or exclusions, not all
skills that should automatically be added. Scores use expected IDs within the
47-skill inventory. No unsupported precision or recall claim for all D4 is made.

Both the original and expanded dictionaries are evaluated against the same
frozen expected IDs, so a new canonical skill counts as a miss for the original.
Affirmative skill-set precision/recall/F1 and assertion failures are separate.
The three remaining assertion failures in the authored suite are coordinated
negation and aspirations; the validation deliberately exposes them. No rule or
alias was tuned after seeing these diagnostic predictions.

## Run and manually review

Open `notebooks/06_pull_and_normalize_skills.ipynb`. It shows the before/after results,
charts, remaining errors, a custom Russian example cell, and ID-safe attachment.
No previous notebook needs editing.

```sh
python3 -m m9_skills.validation --cases m9_skills/validation/russian_validation_cases.csv --baseline m9_skills/dictionary.json --expanded m9_skills/dictionary.v2.json --output m9_skills/validation/my_review_run
```

For human validation, make a copy of `russian_validation_cases.csv` and inspect
text, expected IDs, unknown labels, source ID/field and annotation notes. Enter
`human_decision` as `approve`, `correct` or `exclude`, plus a named
`human_reviewer` and the review date in `human_note`. For a correction,
`human_expected_skill_ids` must be a JSON list such as `["excel", "sql"]` or `[]`.
If assertion labels need correction, update `expected_assertions` in the copied
file and explain the change in human_note. Keep the original diagnostic labels.
Pending human decisions continue to use the agent labels; human-only results
are reported separately and currently contain zero reviewed cases.

For dictionary review, use `validation/dictionary_expansion_review.csv` to inspect
all evidence, then copy `validation/dictionary_alias_decisions.csv`, fill the
reviewer's decision/name/date/source/reason, and compile a new version:

```sh
python3 -m m9_skills.review --dictionary m9_skills/dictionary.v2.json --review my_alias_decisions.csv --output m9_skills/dictionary.v3.json
```

The compiler records decisions without claiming the entire seed is human-approved.
Do not copy Codex's agent review into human reviewer fields. Existing original
aliases still need human review too. Use a fresh output directory for every run.
Full-data extraction with v2 is reproducible with:

```sh
python3 -m m9_skills.workflow extract --raw 'datasets/HSE_dataset(D4).csv' --dictionary m9_skills/dictionary.v2.json --output m9_skills/outputs/my_v2_run
```

## Visualizations

`validation/charts/validation_overview.png` compares corpus and authored
precision/recall/F1, full-data mention coverage, and remaining errors.
`coverage_by_sphere.png` shows coverage for the eight largest occupational areas.
The accompanying CSV/JSON includes all professional spheres. Denominators are
unique vacancies, not repeated interactions. Coverage is not accuracy.

Plotting uses matplotlib, already listed in project requirements. Regenerate the
overview with `python3 -m m9_skills.validation_plots` and the paths shown in notebook
07. Raw source paths/IDs and skill excerpts remain private with the D4 dataset.

This guide covers extraction validation. The separate completed ranking experiment is now in `recommender/experiment.py`, `results/final/` and root notebook 07. M0–M8 remain unchanged. Dictionary human review remains pending.
