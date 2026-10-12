"""Bibliographic normalization with duplicate lineage and label-conflict quarantine."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def clean_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def normalize_doi(value: str) -> str:
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I)
    if not value:
        return ""
    if not re.fullmatch(r"10\.\d{4,9}/\S+", value):
        raise ValueError("invalid DOI")
    return value.casefold()


@dataclass
class ImportResult:
    frame: pd.DataFrame
    report: dict


def import_records(path: Path) -> ImportResult:
    raw = path.read_bytes()
    if len(raw) > 30_000_000:
        raise ValueError("input exceeds 30 MB bound")
    csv.field_size_limit(200_000)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"record_id", "title", "abstract", "year", "label_included"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("missing required bibliographic columns")
        rows = list(reader)
    if not rows or len(rows) > 100_000:
        raise ValueError("input requires 1 to 100000 rows")
    accepted, rejected = [], []
    seen = set()
    for number, row in enumerate(rows, 2):
        try:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("malformed CSV row")
            identifier = clean_text(row["record_id"])
            if not identifier or identifier in seen:
                raise ValueError("empty or repeated record_id")
            seen.add(identifier)
            title, abstract = clean_text(row["title"]), clean_text(row["abstract"])
            if not title or len(title) > 2000 or len(abstract) > 100000:
                raise ValueError("title required or text exceeds field bounds")
            year_text = row["year"].strip()
            if year_text and not re.fullmatch(r"\d{4}", year_text):
                raise ValueError("year must be four digits or empty")
            year = int(year_text) if year_text else None
            if year is not None and not 1800 <= year <= 2100:
                raise ValueError("year outside documented bounds")
            label_text = row["label_included"].strip()
            if label_text not in ("", "0", "1"):
                raise ValueError("inclusion label must be 0, 1 or empty")
            accepted.append(
                {
                    "record_id": identifier,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "doi": normalize_doi(row.get("doi", "")),
                    "label_included": int(label_text) if label_text else None,
                    "duplicate_record_id": row.get("duplicate_record_id", "").strip(),
                }
            )
        except ValueError as error:
            rejected.append(
                {"line": number, "record_id": row.get("record_id", ""), "reason": str(error)}
            )
    if not accepted:
        raise ValueError("no valid records")
    parent = {r["record_id"]: r["record_id"] for r in accepted}

    def find(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    keys = {}
    unresolved = []
    for row in accepted:
        # Equal DOI or exact normalized title/year is a candidate duplicate key.
        key = (
            "doi:" + row["doi"]
            if row["doi"]
            else "title:" + row["title"].casefold() + ":" + str(row["year"])
        )
        if key in keys:
            union(row["record_id"], keys[key])
        keys[key] = row["record_id"]
        for reference in re.split(r"[,;\s]+", row["duplicate_record_id"]):
            if not reference:
                continue
            if reference in parent:
                union(row["record_id"], reference)
            else:
                unresolved.append({"record_id": row["record_id"], "reference": reference})
    groups = {}
    for row in accepted:
        groups.setdefault(find(row["record_id"]), []).append(row)
    canonical, lineage = [], []
    for group_id, members in sorted(groups.items()):
        # Prefer informative abstracts. Original rows and labels remain in lineage.
        chosen = dict(sorted(members, key=lambda x: (-len(x["abstract"]), x["record_id"]))[0])
        labels = {x["label_included"] for x in members if x["label_included"] is not None}
        conflict = len(labels) > 1
        chosen.update(
            group_id=group_id,
            label_conflict=conflict,
            label_included=None if conflict or not labels else next(iter(labels)),
            source_ids=";".join(sorted(x["record_id"] for x in members)),
        )
        chosen.pop("duplicate_record_id")
        canonical.append(chosen)
        lineage.append(
            {
                "group_id": group_id,
                "chosen_record_id": chosen["record_id"],
                "members": members,
                "label_conflict": conflict,
            }
        )
    frame = pd.DataFrame(canonical)
    report = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_rows": len(rows),
        "accepted_rows": len(accepted),
        "rejected_rows": rejected,
        "canonical_rows": len(canonical),
        "duplicate_rows_collapsed": len(accepted) - len(canonical),
        "label_conflict_groups": sum(x["label_conflict"] for x in canonical),
        "missing_abstracts": int((frame["abstract"] == "").sum()),
        "missing_years": int(frame["year"].isna().sum()),
        "unresolved_duplicate_references": unresolved,
        "lineage": lineage,
        "scope": "normalized import with provenance, not full-text scientific fact extraction",
    }
    return ImportResult(frame, report)


def write_import(result: ImportResult, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    result.frame.to_csv(output / "normalized_records.csv", index=False)
    (output / "quality_and_lineage.json").write_text(
        json.dumps(result.report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def search_records(result: ImportResult, query: str) -> list[str]:
    terms = re.findall(r"\w+", clean_text(query).casefold())
    if not terms or len(terms) > 20:
        raise ValueError("search requires 1 to 20 lexical terms")
    return [
        r.group_id
        for r in result.frame.itertuples()
        if all(term in (r.title + " " + r.abstract).casefold() for term in terms)
    ]
