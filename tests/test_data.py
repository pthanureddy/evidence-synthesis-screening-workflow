import csv

import pytest

from evidence_screening.data import clean_text, import_records, normalize_doi, search_records


def modify(path, change):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    change(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    "value", ["10.1234/ABCD", "https://doi.org/10.1234/ABCD", "doi: 10.1234/ABCD"]
)
def test_doi_normalization(value):
    assert normalize_doi(value) == "10.1234/abcd"


@pytest.mark.parametrize("value", ["abcd", "10.foo/x", "https://wrong.example/10.1234/x"])
def test_invalid_doi_rejected(value):
    with pytest.raises(ValueError):
        normalize_doi(value)


def test_unicode_and_whitespace_normalization():
    assert clean_text("Ａ  title\n with\tspaces") == "A title with spaces"


@pytest.mark.parametrize(
    "field,value",
    [
        ("record_id", ""),
        ("title", ""),
        ("year", "201x"),
        ("year", "1799"),
        ("label_included", "yes"),
        ("doi", "bad-doi"),
    ],
)
def test_invalid_rows_quarantined(input_csv, field, value):
    modify(input_csv, lambda rows: rows[0].update({field: value}))
    result = import_records(input_csv)
    assert result.report["accepted_rows"] == 79
    assert len(result.report["rejected_rows"]) == 1


def test_duplicate_id_quarantined(input_csv):
    modify(input_csv, lambda rows: rows[1].update(record_id="0"))
    result = import_records(input_csv)
    assert result.report["accepted_rows"] == 79


def test_duplicate_lineage_and_conflict_quarantine(input_csv):
    modify(input_csv, lambda rows: rows[1].update(duplicate_record_id="0"))
    result = import_records(input_csv)
    assert result.report["canonical_rows"] == 79
    assert result.report["label_conflict_groups"] == 1
    group = result.frame.loc[result.frame.group_id == "0"].iloc[0]
    assert group.source_ids == "0;1" and group.label_conflict
    assert group.label_included != group.label_included  # pandas NaN, excluded from fitting
    assert len(result.report["lineage"][0]["members"]) == 2


def test_exact_title_year_duplicate_and_missing_abstract(input_csv):
    modify(input_csv, lambda rows: rows[1].update(title=rows[5]["title"], abstract=""))
    result = import_records(input_csv)
    assert result.report["canonical_rows"] == 79
    assert (
        result.report["missing_abstracts"] == 0
    )  # Longest retained abstract, lineage preserves empty one


def test_unknown_duplicate_reference_reported(input_csv):
    modify(input_csv, lambda rows: rows[0].update(duplicate_record_id="unknown"))
    assert import_records(input_csv).report["unresolved_duplicate_references"] == [
        {"record_id": "0", "reference": "unknown"}
    ]


def test_metadata_missingness_and_unlabelled_rows(input_csv):
    modify(input_csv, lambda rows: rows[0].update(year="", abstract="", label_included=""))
    result = import_records(input_csv)
    assert result.report["missing_abstracts"] == result.report["missing_years"] == 1
    assert result.frame.label_included.isna().sum() == 1


def test_source_digest_changes(input_csv):
    before = import_records(input_csv).report["source_sha256"]
    modify(input_csv, lambda rows: rows[0].update(title="Corrected title"))
    assert import_records(input_csv).report["source_sha256"] != before


def test_and_lexical_search(input_csv):
    result = import_records(input_csv)
    assert len(search_records(result, "fault prediction")) == 20
    assert search_records(result, "fault marine") == []
    with pytest.raises(ValueError):
        search_records(result, " ")


def test_empty_or_missing_columns_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("wrong\nx\n")
    with pytest.raises(ValueError):
        import_records(path)
