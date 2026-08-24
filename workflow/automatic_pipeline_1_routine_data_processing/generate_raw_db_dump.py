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


def _read_dump(output_path: Path) -> dict:
    with gzip.open(output_path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _inventory(payload: dict) -> list[int]:
    """Return the field codes the written dump actually carries."""
    return sorted(int(code) for code in payload.get("fields", {}))


def build_raw_manifest(shot: int, output: Path, source_kind: str, sample_name: str | None) -> dict:
    """Build the raw-stage manifest dict for an already-written dump.

    Shared by the single-shot CLI below and any batch driver (e.g.
    ``dump_all_shots.py``) so both produce byte-identical manifest shapes.
    """
    dump_payload = _read_dump(output)
    field_codes = _inventory(dump_payload)
    field_quality = dump_payload.get("field_quality", {})
    return {
        "schema_version": 1,
        "stage": "raw",
        "shot": shot,
        "status": "success",
        "source": {"kind": source_kind, "name": sample_name},
        "inventory": {"field_count": len(field_codes), "field_codes": field_codes},
        "pulse_datetime": dump_payload.get("pulse_datetime"),
        "quality_summary": {
            "flagged_field_count": len(field_quality),
            "all_zero": sorted(
                (code for code, flag in field_quality.items() if flag == "all_zero"), key=int
            ),
            "all_nan": sorted(
                (code for code, flag in field_quality.items() if flag == "all_nan"), key=int
            ),
            "empty": sorted(
                (code for code, flag in field_quality.items() if flag == "empty"), key=int
            ),
        },
        "output": {"name": output.name, "sha256": sha256_file(output)},
    }


def dump_shot(shot: int, output: Path, metadata: Path | None, sample: str = "") -> None:
    """Export one shot's raw signals and, if requested, its stage manifest.

    Shared by ``main()`` below and any batch driver -- one place owns the
    dump-then-manifest sequence so a single-shot Snakemake rule and a
    multi-shot backfill behave identically.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    sample = sample.strip()
    source_kind = "archive" if sample else "vest-sql"
    LOGGER.info("Exporting raw signals for shot %s from %s", shot, source_kind)
    if not dump_all_raw_signals_for_shot(
        shot=shot,
        output_path=str(output),
        sample_opt=sample if sample else False,
    ):
        raise RuntimeError(f"Failed to export VEST raw data for shot {shot}")

    if metadata is not None:
        metadata.parent.mkdir(parents=True, exist_ok=True)
        manifest = build_raw_manifest(shot, output, source_kind, Path(sample).name if sample else None)
        write_manifest(manifest, metadata)
        LOGGER.info("Raw manifest saved to %s", metadata)

    LOGGER.info("Raw db dump saved to %s", output)


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

    # force=True: vaft.database.raw (imported above) already calls
    # logging.basicConfig() at import time, and whichever call runs first
    # normally wins -- silently dropping this script's own INFO-level
    # progress log lines depending on import order otherwise.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    dump_shot(args.shot, args.output, args.metadata, args.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
