import csv

import pytest


@pytest.fixture
def input_csv(tmp_path):
    path = tmp_path / "records.csv"
    fields = [
        "record_id",
        "title",
        "abstract",
        "year",
        "label_included",
        "duplicate_record_id",
        "doi",
    ]
    rows = []
    for i in range(80):
        positive = i % 4 == 0
        rows.append(
            {
                "record_id": str(i),
                "title": f"Study {i}",
                "abstract": "empirical software fault prediction model"
                if positive
                else "unrelated robotics control engineering",
                "year": "2011",
                "label_included": str(int(positive)),
                "duplicate_record_id": "",
                "doi": "",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path
