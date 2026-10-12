# Protocol and architecture

Question: can a reproducible, audited bibliographic text workflow rank records
against a released review's final-inclusion labels, while keeping uncertainty,
errors, source corrections and reviewer decisions inspectable?

Pipeline: frozen CSV -> normalization/duplicate candidates -> canonical groups
and label-conflict quarantine -> train/validation/test -> training-only TF-IDF
and logistic regression -> validation threshold -> held-out errors/intervals
-> priority queue -> human review CSV -> append-only SQLite ledger.

This is not a completed systematic review, meta-analysis, extraction of effect
sizes or an expert assessment of the original articles' study validity.

## Prespecified choices

- Seed 42; one 60/20/20 stratified split of canonical labelled groups.
- TF-IDF word unigrams/bigrams, minimum document frequency 2, maximum 15,000
  features, sublinear term frequency. Fit vocabulary only on training records.
- Balanced logistic regression, liblinear, max_iter 1000. No test-guided tuning.
- Choose the highest validation score threshold meeting recall >= 0.90 there.
  Report actual test recall without altering the threshold after seeing it.
- Baselines: all high priority and all standard priority. Precision/recall with
  undefined denominators are null rather than perfect or invented scores.
- Wilson 95% intervals for conditional precision and recall proportions. They
  are approximate sampling intervals for this holdout, not predictive uncertainty
  for a specific article, proof of calibration or a guarantee across domains.
- Report Brier/AP/confusion/F1 and all held-out error IDs; no accuracy-only claim
  on this imbalanced corpus. Slice by abstract availability with explicit counts.
- All records remain `needs_manual_review`, regardless of score. Exported blind
  review templates omit benchmark labels and model scores.
- Do not conflate title/abstract and full-text reviewer stages. Preserve changes,
  explicit reasons, reviewer identifier and source checksum. Conflicts require
  manual adjudication rather than model tie-breaking.

## Traceability

| Requirement | Test/evidence |
|---|---|
| ESW-001 schema/field quality | test_invalid_rows_quarantined; DOI/schema tests |
| ESW-002 provenance/duplicate lineage | test_duplicate_lineage_and_conflict_quarantine; source digest test |
| ESW-003 train/test isolation | test_full_workflow_reproducible_splits_queue_and_no_text_leak |
| ESW-004 validation-only threshold | test_validation_threshold_recall_constraint |
| ESW-005 honest error/uncertainty metrics | metric denominator, confusion and Wilson tests |
| ESW-006 no automatic exclusion | full workflow queue/template test |
| ESW-007 atomic, revisioned reviewer records | ledger idempotency/history and atomic rejection tests |
| ESW-008 distinct stages/adjudication | test_disagreement_and_stage_separation |
| ESW-009 reproducible CLI/build | test_cli_prepare_search_benchmark_and_reviews; CI |

## Method references

CEE guidance describes protocol-based eligibility criteria, distinct screening
stages, transparent reviewer procedures and handling of unclear cases:
https://environmentalevidence.org/information-for-authors/5-eligibility-screening/.
This engineering workflow draws on those concepts but does not claim compliance
with all CEE standards. A real review needs trained reviewers, a reviewed search
protocol, full texts, extraction procedures and study-validity appraisal.

Training-only feature fitting uses scikit-learn Pipeline:
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html.
Information leakage guidance:
https://scikit-learn.org/stable/common_pitfalls.html#data-leakage.

## Limitations and next study

Final-inclusion labels are a proxy for prioritization, not abstract eligibility
ground truth. One review and a random retrospective split do not test external
generalization, work saved or review completeness. Duplicate/label quarantine can
introduce selection effects. Topic/year/English selection and missing abstracts
can bias performance. Reviewers may not be independent, and published labels are
not unquestionable truth. Scores have not been calibrated. LLMs and real scientific
result extraction are not used. A follow-up should prespecify environmental review
criteria, obtain independent initial-screening labels, measure disagreement and
hold out an external review before evaluating any change to human workload.
