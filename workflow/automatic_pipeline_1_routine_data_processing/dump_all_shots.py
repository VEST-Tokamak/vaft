#!/usr/bin/env python3
"""Backfill raw SQL dumps for every available VEST shot into FileDB.

Discovers shots via ``vaft.database.raw.list_shots()`` (issue #126) --
filterable by shot range, date range, or their intersection -- and calls
``generate_raw_db_dump.dump_shot()`` for each one, writing to the canonical
FileDB layout via ``PipelinePaths(..., layout="filedb")`` so every dump lands
exactly where the rest of the pipeline (``run --config layout=filedb``)
already expects to find it.

A single bad shot does not abort a large backfill: failures are caught,
logged, and collected into the final summary instead of stopping the run.

Examples
--------
Dump every shot recorded in May 2026 into a FileDB root::

    python dump_all_shots.py --filedb-root /srv/vest.filedb/public \\
        --start-date 2026-05-01 --end-date 2026-05-31

Dump a specific shot-number range, skipping shots already dumped::

    python dump_all_shots.py --filedb-root /srv/vest.filedb/public \\
        --shot-min 45000 --shot-max 46000 --skip-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from paths import FILEDB, PipelinePaths  # noqa: E402

from generate_raw_db_dump import dump_shot  # noqa: E402
from vaft.database.raw import list_shots  # noqa: E402


LOGGER = logging.getLogger("vaft.dump_all_shots")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--filedb-root", required=True, help="FileDB root directory.")
    parser.add_argument("--shot-min", type=int, default=None, help="Inclusive shot-number lower bound.")
    parser.add_argument("--shot-max", type=int, default=None, help="Inclusive shot-number upper bound.")
    parser.add_argument("--start-date", default=None, help="Inclusive, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Inclusive, YYYY-MM-DD.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip shots whose FileDB dump already exists, instead of re-exporting them.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N discovered shots (for smoke-testing a backfill before running it in full).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be dumped without touching SQL or FileDB.",
    )
    parser.add_argument("--summary", type=Path, default=None, help="Write a batch-level JSON summary here.")
    args = parser.parse_args()

    # force=True: vaft.database.raw (imported above) already calls
    # logging.basicConfig() itself at import time, and whichever call runs
    # first normally wins -- silently dropping this script's own INFO-level
    # progress/summary output if some other import got there first. force=True
    # makes this call authoritative regardless of import order.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

    paths = PipelinePaths(args.filedb_root, FILEDB)

    LOGGER.info(
        "Discovering shots (shot_min=%s, shot_max=%s, start_date=%s, end_date=%s)",
        args.shot_min,
        args.shot_max,
        args.start_date,
        args.end_date,
    )
    shots = list_shots(
        shot_min=args.shot_min,
        shot_max=args.shot_max,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if args.limit is not None:
        shots = shots[: args.limit]
    LOGGER.info("Discovered %d shot(s)", len(shots))

    completed: list[int] = []
    skipped: list[int] = []
    failed: list[dict] = []

    for shot, record_datetime in shots:
        output = Path(paths.raw_dump(shot))
        metadata = Path(paths.raw_manifest(shot))

        if args.skip_existing and output.exists():
            LOGGER.info("shot %s: dump already exists at %s, skipping", shot, output)
            skipped.append(shot)
            continue

        if args.dry_run:
            LOGGER.info("shot %s (recorded %s): would dump to %s", shot, record_datetime, output)
            continue

        try:
            dump_shot(shot, output, metadata)
        except Exception as exc:
            LOGGER.error("shot %s: dump failed: %s", shot, exc)
            failed.append({"shot": shot, "reason": str(exc)})
            continue

        completed.append(shot)

    LOGGER.info(
        "Backfill complete: %d completed, %d skipped, %d failed (of %d discovered)",
        len(completed),
        len(skipped),
        len(failed),
        len(shots),
    )
    for entry in failed:
        LOGGER.error("  shot %s: %s", entry["shot"], entry["reason"])

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(
                {
                    "discovered": len(shots),
                    "completed": completed,
                    "skipped": skipped,
                    "failed": failed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        LOGGER.info("Batch summary saved to %s", args.summary)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
