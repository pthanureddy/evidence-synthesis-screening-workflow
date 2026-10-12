# Executed verification

Recorded 12 October 2026 against implementation commit
`eccaa1bb5e79a14f120324aaa3c7e65584bc5a05`. The next documentation-only
commit adds this record without changing the benchmark implementation.

## Checks

- Documented editable installation, `pip check`, Ruff lint/format checks and
  source/wheel builds completed successfully on Windows with Python 3.12.14.
- `pytest --cov=evidence_screening --cov-branch` passed **56 tests**.
  Combined statement/branch coverage was **96.39%** (306/313 statements,
  94/102 branches). Coverage describes exercised code, not research validity.
- Documented `prepare`, `search`, `benchmark` and `reviews` CLI flows completed.
  The untouched review template correctly created **zero human review events**.
- [Hosted CI](https://github.com/pthanureddy/evidence-synthesis-screening-workflow/actions/runs/38306714145)
  passed on Python 3.11 and 3.12, including lint, formatting, tests, package
  build and the documented real-data CLI flows. Its artifacts contain the
  full benchmark, source-quality, search and review outputs.

## Corpus and partitions

The pinned input contains 8,911 records and 104 released final-inclusion labels.
Normalization retained 8,812 canonical duplicate groups, collapsing 99 duplicate
records. One group had conflicting labels and was excluded from modeling but
retained for manual review. No source rows were rejected. All source records
in this snapshot have abstracts and years.

The remaining 8,811 labeled groups were split once with seed 42, stratified
60/20/20: training 5,286 (62 positives), validation 1,762 (20 positives),
test 1,763 (21 positives). Duplicate grouping precedes splitting. TF-IDF
vocabulary and logistic-regression weights use only the training partition.
The validation partition selected threshold `0.28336258297804384` for its
configured recall target of 0.90. The test partition did not tune that cutoff.

## Held-out priority benchmark

| Quantity | Executed result |
| --- | ---: |
| Records | 1,763 |
| Original-reviewer included records | 21 |
| True positives / false negatives | 20 / 1 |
| False positives / true negatives | 25 / 1,717 |
| Recall | 95.24% (20/21) |
| Precision | 44.44% (20/45) |
| F1 | 0.6061 |
| Recall Wilson 95% interval | 77.33%-99.15% |
| Precision Wilson 95% interval | 30.94%-58.82% |
| Average precision | 0.6280 |
| Brier score | 0.009626 |

Runtime versions: Python 3.12.14, scikit-learn 1.9.1, pandas 2.3.3,
NumPy 2.5.4. Exact record IDs, denominators, error rows, baselines and versions
are exported by the documented benchmark command.

The labels are the original authors' **final inclusion decisions**, not initial
title/abstract decisions or new judgments collected here. The subject is
software-engineering fault prediction. These numbers do not establish
environmental-domain accuracy, calibrated confidence, workload savings, review
completeness or reduced bias. All groups still require human review; the
false negative is retained in the standard-priority queue. The absent-abstract
slice has zero records, so it cannot establish performance on missing abstracts.

The lexical AND query `fault prediction` matched 218 imported records. This
searches the supplied corpus; it is not a comprehensive bibliographic-database
search. Unit-test review events are synthetic fixtures, separate from released
human labels and from real study decisions.

See [methods](methods.md) and [data provenance](data-provenance.md) for the
assumptions, input digest, attribution and reproducible commands.
