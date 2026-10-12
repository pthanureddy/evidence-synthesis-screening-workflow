from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .data import import_records, search_records, write_import
from .review import ReviewLedger
from .validation import benchmark

LOG = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audited import, screening-priority benchmark and manual review ledger"
    )
    parser.add_argument("command", choices=["prepare", "benchmark", "search", "reviews"])
    parser.add_argument("--input", type=Path, default=Path("data/hall_2012.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-recall", type=float, default=0.9)
    parser.add_argument("--query")
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        result = import_records(args.input)
        LOG.info(
            "source_sha256=%s raw=%d canonical=%d conflicts=%d",
            result.report["source_sha256"],
            result.report["source_rows"],
            result.report["canonical_rows"],
            result.report["label_conflict_groups"],
        )
        args.output.mkdir(parents=True, exist_ok=True)
        if args.command == "prepare":
            write_import(result, args.output)
            response = {
                k: v for k, v in result.report.items() if k not in {"lineage", "rejected_rows"}
            }
        elif args.command == "benchmark":
            report = benchmark(result, args.output, args.seed, args.target_recall)
            response = {
                k: report[k]
                for k in (
                    "split_counts",
                    "positive_counts",
                    "threshold",
                    "validation",
                    "holdout",
                    "scope",
                )
            }
        elif args.command == "search":
            if not args.query:
                raise ValueError("search requires --query")
            matched = search_records(result, args.query)
            response = {
                "query": args.query,
                "matched_count": len(matched),
                "group_ids": matched,
                "source_sha256": result.report["source_sha256"],
                "scope": "AND lexical search of imported titles/abstracts, not comprehensive database searching",
            }
            (args.output / "search.json").write_text(
                json.dumps(response, indent=2) + "\n", encoding="utf-8"
            )
        else:
            if not args.decisions:
                raise ValueError("reviews requires --decisions")
            response = ReviewLedger(args.output / "reviews.db").ingest(args.decisions, result)
            (args.output / "reviews.json").write_text(
                json.dumps(response, indent=2) + "\n", encoding="utf-8"
            )
        print(json.dumps(response, indent=2, default=lambda x: x.item(), allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError) as error:
        LOG.error("workflow failed: %s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
