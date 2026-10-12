import csv
import json
import sqlite3

import pytest

from evidence_screening.cli import main
from evidence_screening.data import import_records
from evidence_screening.review import ReviewLedger


def decisions(tmp_path, result, rows):
    path = tmp_path / "decisions.csv"
    fields = ["group_id", "reviewer", "stage", "decision", "reason", "source_sha256"]
    defaults = {
        "group_id": result.frame.iloc[0].group_id,
        "reviewer": "illustrative-a",
        "stage": "title_abstract",
        "decision": "include",
        "reason": "Illustrative test criterion",
        "source_sha256": result.report["source_sha256"],
    }
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{**defaults, **row} for row in rows])
    return path


def test_idempotent_import_and_preserved_revision_history(input_csv, tmp_path):
    result = import_records(input_csv)
    ledger = ReviewLedger(tmp_path / "ledger.db")
    path = decisions(tmp_path, result, [{}])
    assert ledger.ingest(path, result)["events_added"] == 1
    assert ledger.ingest(path, result)["events_added"] == 0
    path = decisions(
        tmp_path, result, [{"decision": "exclude", "reason": "Corrected illustrative criterion"}]
    )
    assert ledger.ingest(path, result)["events_added"] == 1
    assert ledger.summary(result.report["source_sha256"])[0]["status"] == "exclude"
    with sqlite3.connect(ledger.path) as db:
        assert db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 2


def test_disagreement_and_stage_separation(input_csv, tmp_path):
    result = import_records(input_csv)
    ledger = ReviewLedger(tmp_path / "ledger.db")
    rows = [
        {},
        {"reviewer": "illustrative-b", "decision": "exclude"},
        {"stage": "full_text", "decision": "uncertain"},
    ]
    report = ledger.ingest(decisions(tmp_path, result, rows), result)
    assert {x["stage"]: x["status"] for x in report["summary"]} == {
        "title_abstract": "needs_adjudication",
        "full_text": "uncertain",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"reviewer": ""},
        {"stage": "automatic"},
        {"decision": "score-accepted"},
        {"reason": ""},
        {"source_sha256": "wrong"},
        {"group_id": "missing"},
    ],
)
def test_invalid_reviews_rejected_atomically(input_csv, tmp_path, change):
    result = import_records(input_csv)
    ledger = ReviewLedger(tmp_path / "ledger.db")
    with pytest.raises(ValueError):
        ledger.ingest(
            decisions(tmp_path, result, [{}, {"reviewer": "illustrative-b", **change}]), result
        )
    assert ledger.summary(result.report["source_sha256"]) == []


def test_duplicate_reviewer_decisions_rejected(input_csv, tmp_path):
    result = import_records(input_csv)
    ledger = ReviewLedger(tmp_path / "ledger.db")
    with pytest.raises(ValueError):
        ledger.ingest(decisions(tmp_path, result, [{}, {}]), result)


def test_blank_template_rows_do_not_create_decisions(input_csv, tmp_path):
    result = import_records(input_csv)
    ledger = ReviewLedger(tmp_path / "ledger.db")
    path = decisions(
        tmp_path, result, [{"reviewer": "", "stage": "", "decision": "", "reason": ""}]
    )
    assert ledger.ingest(path, result)["events_added"] == 0
    assert ledger.summary(result.report["source_sha256"]) == []


def test_cli_prepare_search_benchmark_and_reviews(input_csv, tmp_path, capsys):
    output = tmp_path / "out"
    assert main(["prepare", "--input", str(input_csv), "--output", str(output)]) == 0
    assert (output / "normalized_records.csv").exists()
    assert (
        main(
            [
                "search",
                "--input",
                str(input_csv),
                "--output",
                str(output),
                "--query",
                "fault prediction",
            ]
        )
        == 0
    )
    assert json.loads((output / "search.json").read_text())["matched_count"] == 20
    assert main(["benchmark", "--input", str(input_csv), "--output", str(output)]) == 0
    assert (
        main(
            [
                "reviews",
                "--input",
                str(input_csv),
                "--output",
                str(output),
                "--decisions",
                str(output / "manual_review_template.csv"),
            ]
        )
        == 0
    )
    assert json.loads((output / "reviews.json").read_text())["events_added"] == 0


@pytest.mark.parametrize("args", [["search"], ["reviews"], ["prepare", "--input", "missing.csv"]])
def test_cli_errors_exit_two(input_csv, tmp_path, args):
    command = args + ["--output", str(tmp_path / "out")]
    if "--input" not in args:
        command += ["--input", str(input_csv)]
    assert main(command) == 2
