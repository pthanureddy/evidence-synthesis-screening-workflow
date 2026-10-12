# Evidence Synthesis Screening Workflow

A Python workflow for audited bibliographic import, lexical searching, screening
priority, validation against released reviewer decisions and local manual-review
records. It uses the real, published Hall 2012 systematic-review dataset.

## Scope

The model ranks imported titles/abstracts; **every record still requires manual
review**. Labels were released by the original review authors and represent final
inclusion. They are not the initial title/abstract decisions and were not generated
by this project, its owner or an LLM. This is a retrospective software-engineering
benchmark, not a new systematic review or evidence of environmental-domain accuracy.

## What it implements

- CSV schema/size validation, Unicode/text/year/optional-DOI normalization.
- Declared duplicate links and exact DOI/title-year candidates, retained lineage,
  malformed-row quarantine and conflicting-label groups excluded from training.
- Input SHA-256, normalized records, field-quality counts and a lexical search log.
- A TF-IDF/logistic-regression pipeline fitted on training text only, a fixed
  60/20/20 stratified split after duplicate grouping, and threshold choice on the
  validation partition only.
- Held-out confusion/error records, precision/recall/F1, average precision, Brier
  score, Wilson 95% intervals, missing-abstract slices and two trivial baselines.
- A priority queue and blank manual review template with no gold labels/scores.
- An append-only SQLite reviewer ledger with explicit title/abstract and full-text
  stages, source-digest checks, idempotent imports and disagreement flags.
- Tests for provenance, invalid inputs, duplicate/label handling, leakage,
  reproducibility, metric denominators, atomic manual reviews and CLI behavior.

## Setup

Python 3.11+ from the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m build
```

## Run

```bash
evidence-screening prepare --output artifacts
evidence-screening search --query 'fault prediction' --output artifacts
evidence-screening benchmark --seed 42 --target-recall 0.9 --output artifacts
```

The benchmark creates `benchmark.json`, source quality/lineage, normalized data,
held-out predictions, `review_queue.csv`, a blank `manual_review_template.csv`
and a trusted local `model.joblib`. CLI status `0` means successful execution;
`2` means an input/workflow error. A successful run does not establish scientific
validity or a required holdout-recall target. Scores are not calibrated confidence.

The default input is the bundled public dataset. Another CSV must contain
`record_id,title,abstract,year,label_included`, with optional `duplicate_record_id`
and `doi`. Labels are `0`, `1` or empty. Empty labels permit an unlabelled queue;
benchmarking needs both classes and sufficient positive records. Source correction
changes the digest and requires a new review dataset. Full-text results, authors
and scientific effect estimates are not inferred from absent fields.

## Manual review

Fill selected template rows with a reviewer identifier, `title_abstract` or
`full_text` stage, `include`, `exclude` or `uncertain`, and a reason. Keep all
unreviewed rows blank. Use independent reviewer copies when assessing agreement.

```bash
evidence-screening reviews --decisions artifacts/manual_review_template.csv --output artifacts
```

The unfilled template records no decisions. Filled rows are validated before
any transaction is committed. Updates preserve previous events; repeated unchanged
imports create no extra events. Different stages are never merged. Disagreement
requires adjudication by reviewers; the model cannot adjudicate it.

## Dataset and attribution

`data/hall_2012.csv` has 8,911 released records and 104 final-inclusion labels
before this workflow's quality/duplicate handling. It is the original metadata-v1
snapshot, not the later OpenAlex-enriched 8,793-record version quoted in some
SYNERGY documentation. See [data provenance and licence](docs/data-provenance.md).
Review: Hall, T., Beecham, S., Bowes, D., Gray, D., and Counsell, S. (2012),
*A Systematic Literature Review on Fault Prediction Performance in Software
Engineering*, https://doi.org/10.1109/TSE.2011.103.
Original dataset: https://doi.org/10.5281/zenodo.1162952.
SYNERGY metadata collection: https://github.com/asreview/synergy-dataset.

## Methods and limits

The training partition is used for vocabulary/weights. The validation partition
chooses the highest score cutoff meeting its configured recall target. The test
partition is evaluated once, including any false negatives. Class imbalance,
quarantine selection, final-inclusion versus initial screening labels, one review
topic, English-language selection and missing metadata limit interpretation.
No effort/time savings, systematic-review completeness, reduced bias, model
fine-tuning, LLM use or generalization to environmental research is claimed.

Duplicate groups may combine multiple articles with the same exact title/year;
lineage makes these candidates inspectable. Conflicting labels are quarantined
conservatively and remain in the manual queue. Review them before real study use.
Full imported text is retained only locally in artifacts; comply with the source
licence and restrict access for private corpora. Only load joblib files generated
in a trusted environment. Reviewer strings are not authenticated identities.

Design and research boundaries: [methods](docs/methods.md).
AI assistance: implementation/test/docs drafting used Codex. Executed checks and
published source establish bounded engineering behavior, not unassisted mastery.
