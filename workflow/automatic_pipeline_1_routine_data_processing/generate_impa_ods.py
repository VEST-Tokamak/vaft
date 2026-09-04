#!/usr/bin/env python3
"""Generate the standalone IMPA OMAS ODS from an archived VEST raw DAQ dump.

IMPA is an optional, campaign-dependent diagnostic, so this stage is opt-in and
is never a dependency of the baseline products. It also never fails the run:
every error is recorded in the stage manifest as ``failed`` and the script exits
0, so a broken IMPA branch cannot change the exit state of an otherwise
successful `main` publication (issue #305). Whether the product is eligible to
be published is decided by its manifest status, not by this exit code.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import traceback

from omas import ODS

from vaft.machine_mapping.dataset_description import dataset_description
from vaft.omas.vest_upstream import build_impa_ods, write_stage_product


LOGGER = logging.getLogger("vaft.generate_impa_ods")


def _failed_product(shot: int, run: int, error: BaseException) -> tuple[ODS, dict]:
    """A product carrying provenance only, and a manifest saying what broke."""
    ods = ODS(consistency_check=False)
    dataset_description(
        ods,
        shot,
        {
            "source_type": "shot",
            "run": run,
            "machine": "VEST",
            "user": "vaft",
            "description": "VEST IMPA; stage failed",
        },
    )
    manifest = {
        "schema_version": 1,
        "stage": "impa",
        "shot": int(shot),
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
        "quality_summary": {
            "missing": [],
            "repaired": [],
            "disabled": [],
            "rejected": [],
            "unavailable": ["impa"],
        },
    }
    return ods, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--raw-dump", required=True, type=Path, help="Archived raw DAQ JSON gzip file.")
    parser.add_argument("--output", required=True, type=Path, help="Output IMPA ODS JSON file.")
    parser.add_argument("--metadata", required=True, type=Path, help="Output stage manifest JSON.")
    # Unset means "use the IMPA window configured in vest.yaml".
    parser.add_argument("--tstart", default=None, type=float, help="Start time of the analysis window.")
    parser.add_argument("--tend", default=None, type=float, help="End time of the analysis window.")
    parser.add_argument("--dt", default=None, type=float, help="Analysis window time step.")
    parser.add_argument("--run", default=1, type=int, help="Dataset run number.")
    parser.add_argument(
        "--time-policies-json",
        default="",
        help="JSON object overriding the diagnostics time policies configured in vest.yaml.",
    )
    args = parser.parse_args()

    # force=True: see generate_diagnostics_ods.py -- vaft.database.raw calls
    # logging.basicConfig() at import time and would otherwise win.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    LOGGER.info("Generating IMPA ODS for shot %s from %s", args.shot, args.raw_dump)
    try:
        ods, manifest = build_impa_ods(
            shot=args.shot,
            raw_source=args.raw_dump,
            tstart=args.tstart,
            tend=args.tend,
            dt=args.dt,
            run=args.run,
            time_policies=json.loads(args.time_policies_json)
            if args.time_policies_json.strip()
            else None,
        )
    except Exception as error:  # noqa: BLE001 - the containment is the point
        LOGGER.exception("IMPA stage failed for shot %s; recording and continuing", args.shot)
        ods, manifest = _failed_product(args.shot, args.run, error)

    write_stage_product(ods, manifest, output=args.output, metadata=args.metadata)
    LOGGER.info("IMPA ODS saved to %s (status=%s)", args.output, manifest["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
