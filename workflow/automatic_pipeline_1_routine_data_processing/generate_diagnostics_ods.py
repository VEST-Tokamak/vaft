#!/usr/bin/env python3
"""Generate the diagnostics OMAS ODS from an archived VEST raw DAQ dump.

Components are mapped independently by
:func:`vaft.omas.vest_upstream.build_diagnostics_ods`: a diagnostic whose raw
signals are absent is recorded as ``unavailable`` in the stage manifest instead
of aborting the shot or, as the legacy workflow did, silently contributing a
zero-filled waveform. Valid siblings survive either way.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from vaft.omas.vest_upstream import build_diagnostics_ods, write_stage_product


LOGGER = logging.getLogger("vaft.generate_diagnostics_ods")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--raw-dump", required=True, type=Path, help="Archived raw DAQ JSON gzip file.")
    parser.add_argument("--static-ods", required=True, type=Path, help="Versioned static machine ODS.")
    parser.add_argument("--output", required=True, type=Path, help="Output diagnostics ODS JSON file.")
    parser.add_argument("--metadata", required=True, type=Path, help="Output stage manifest JSON.")
    parser.add_argument("--tstart", default=0.26, type=float, help="Start time for mapped diagnostics.")
    parser.add_argument("--tend", default=0.36, type=float, help="End time for mapped diagnostics.")
    parser.add_argument("--dt", default=4e-5, type=float, help="Mapped diagnostics time step.")
    parser.add_argument("--run", default=1, type=int, help="Dataset run number.")
    parser.add_argument(
        "--vest-magnetics-processing-json",
        default="",
        help="JSON object overriding VEST magnetics processing defaults.",
    )
    args = parser.parse_args()

    # force=True: vaft.database.raw (imported transitively via build_diagnostics_ods)
    # already calls logging.basicConfig() at import time, and whichever call runs
    # first normally wins -- silently dropping this script's own INFO-level log
    # lines depending on import order otherwise.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    LOGGER.info("Generating diagnostics ODS for shot %s from %s", args.shot, args.raw_dump)
    ods, manifest = build_diagnostics_ods(
        shot=args.shot,
        raw_source=args.raw_dump,
        static_ods=args.static_ods,
        tstart=args.tstart,
        tend=args.tend,
        dt=args.dt,
        run=args.run,
        vest_magnetics_processing=json.loads(args.vest_magnetics_processing_json)
        if args.vest_magnetics_processing_json.strip()
        else None,
    )

    write_stage_product(ods, manifest, output=args.output, metadata=args.metadata)
    unavailable = manifest["quality_summary"]["unavailable"]
    if unavailable:
        LOGGER.warning(
            "Shot %s has unavailable diagnostic components: %s",
            args.shot,
            ", ".join(unavailable),
        )
    LOGGER.info(
        "Diagnostics ODS saved to %s (status=%s)", args.output, manifest["status"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
