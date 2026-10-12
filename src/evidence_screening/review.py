"""Local append-only manual review ledger; never creates decisions from scores."""

from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from .data import ImportResult


class ReviewLedger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS decisions (
                sequence INTEGER PRIMARY KEY, dataset_sha TEXT NOT NULL,
                group_id TEXT NOT NULL, reviewer TEXT NOT NULL, stage TEXT NOT NULL,
                decision TEXT NOT NULL, reason TEXT NOT NULL, submitted_utc TEXT NOT NULL)""")

    def ingest(self, path: Path, result: ImportResult) -> dict:
        allowed = set(result.frame.group_id)
        digest = result.report["source_sha256"]
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = {"group_id", "reviewer", "stage", "decision", "reason", "source_sha256"}
            if not fields.issubset(reader.fieldnames or []):
                raise ValueError("review CSV lacks required fields")
            rows = list(reader)
        checked, keys = [], set()
        for row in rows:
            values = [row.get(k) for k in ("reviewer", "stage", "decision", "reason")]
            if all(v == "" for v in values):
                continue  # Unfilled rows in the generated template are not decisions.
            if any(v is None or not v.strip() for v in values):
                raise ValueError("reviewer, stage, decision and reason are required")
            reviewer, stage, decision, reason = (v.strip() for v in values)
            if len(reviewer) > 80 or len(reason) > 4000:
                raise ValueError("review fields exceed bounds")
            if row["group_id"] not in allowed or row["source_sha256"] != digest:
                raise ValueError("review references unknown group or a changed dataset")
            if stage not in {"title_abstract", "full_text"} or decision not in {
                "include",
                "exclude",
                "uncertain",
            }:
                raise ValueError("invalid review stage or decision")
            key = (row["group_id"], reviewer, stage)
            if key in keys:
                raise ValueError("multiple decisions for reviewer/group/stage in one import")
            keys.add(key)
            checked.append((*key, decision, reason))
        changed = 0
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            for group, reviewer, stage, decision, reason in checked:
                prior = db.execute(
                    """SELECT decision,reason FROM decisions WHERE dataset_sha=?
                    AND group_id=? AND reviewer=? AND stage=? ORDER BY sequence DESC LIMIT 1""",
                    (digest, group, reviewer, stage),
                ).fetchone()
                if prior == (decision, reason):
                    continue
                db.execute(
                    "INSERT INTO decisions(dataset_sha,group_id,reviewer,stage,decision,reason,submitted_utc) VALUES(?,?,?,?,?,?,?)",
                    (
                        digest,
                        group,
                        reviewer,
                        stage,
                        decision,
                        reason,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                changed += 1
        return {
            "filled_rows": len(checked),
            "events_added": changed,
            "summary": self.summary(digest),
        }

    def summary(self, digest: str) -> list[dict]:
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute(
                """SELECT group_id,reviewer,stage,decision,reason FROM decisions
                WHERE dataset_sha=? AND sequence IN (
                    SELECT MAX(sequence) FROM decisions WHERE dataset_sha=? GROUP BY group_id,reviewer,stage)
                ORDER BY group_id,stage,reviewer""",
                (digest, digest),
            ).fetchall()
        grouped = defaultdict(list)
        for group, reviewer, stage, decision, reason in rows:
            grouped[(group, stage)].append(
                {"reviewer": reviewer, "decision": decision, "reason": reason}
            )
        return [
            {
                "group_id": group,
                "stage": stage,
                "reviews": values,
                "status": values[0]["decision"]
                if len({x["decision"] for x in values}) == 1
                else "needs_adjudication",
            }
            for (group, stage), values in grouped.items()
        ]
