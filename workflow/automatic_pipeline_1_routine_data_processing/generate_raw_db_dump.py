#!/usr/bin/env python3
"""Export one VEST shot's raw DAQ signals to a canonical dump.

Both modes go through :func:`vaft.database.raw.dump_all_raw_signals_for_shot`,
which resolves an archived source with the same ``sample_opt`` convention as
``load_raw``. In archive mode the dump is re-derived from the archive rather
than copied, so the product is a canonical ``.json.gz`` regardless of how the
source was written, and a shot mismatch or a missing ``fields`` mapping is an
error instead of a silently accepted input.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path

from vaft.database.raw import dump_all_raw_signals_for_shot
from vaft.omas.vest_upstream import sha256_file, write_manifest


LOGGER = logging.getLogger("vaft.generate_raw_db_dump")


def _inventory(output_path: Path) -> list[int]:
    """Return the field codes the written dump actually carries."""
    with gzip.open(output_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return sorted(int(code) for code in payload.get("fields", {}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--output", required=True, type=Path, help="Output raw dump .json.gz.")
    parser.add_argument("--metadata", type=Path, help="Output stage manifest JSON.")
    parser.add_argument(
        "--sample",
        default="",
        help="Archived raw source. When empty the shot is exported from the VEST SQL database.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sample = args.sample.strip()
    source_kind = "archive" if sample else "vest-sql"
    LOGGER.info("Exporting raw signals for shot %s from %s", args.shot, source_kind)
    if not dump_all_raw_signals_for_shot(
        shot=args.shot,
        output_path=str(args.output),
        sample_opt=sample if sample else False,
    ):
        raise RuntimeError(f"Failed to export VEST raw data for shot {args.shot}")

    if args.metadata:
        field_codes = _inventory(args.output)
        write_manifest(
            {
                "schema_version": 1,
                "stage": "raw",
                "shot": args.shot,
                "status": "success",
                "source": {"kind": source_kind, "name": Path(sample).name if sample else None},
                "inventory": {"field_count": len(field_codes), "field_codes": field_codes},
                "output": {"name": args.output.name, "sha256": sha256_file(args.output)},
            },
            args.metadata,
        )
        LOGGER.info("Raw manifest saved to %s", args.metadata)

    LOGGER.info("Raw db dump saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
