#!/usr/bin/env python3
"""Generate the VEST eddy-current OMAS ODS from the diagnostics ODS.

Passive geometry and the electromagnetic coupling matrices come from the
versioned static machine ODS, which is the VAFT-native form of the legacy
`emcoupling_rules` selection. Missing eddy inputs are reported explicitly rather
than surfacing as a bare key error deep inside the solver.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="vaft-mpl-"))

from vaft.omas.vest_upstream import build_eddy_ods, write_stage_product


LOGGER = logging.getLogger("vaft.generate_eddy_ods")


def _csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--diagnostics-ods", "--input", required=True, type=Path, help="Input diagnostics ODS JSON.")
    parser.add_argument("--static-ods", required=True, type=Path, help="Versioned static machine ODS.")
    parser.add_argument("--output", required=True, type=Path, help="Output eddy ODS JSON.")
    parser.add_argument("--metadata", required=True, type=Path, help="Output stage manifest JSON.")
    parser.add_argument("--filament-r", default="0.35,0.35,0.35", help="Comma-separated plasma filament R positions.")
    parser.add_argument("--filament-z", default="0.25,0.0,-0.25", help="Comma-separated plasma filament Z positions.")
    parser.add_argument("--filament-fraction", default="0.3333333,0.3333333,0.3333333", help="Comma-separated plasma current fractions.")
    parser.add_argument("--dt-sub", default=5e-5, type=float, help="Sub-step for eddy-current integration.")
    args = parser.parse_args()

    # force=True: vaft.database.raw installs a root handler at import time, which makes
    # basicConfig() a no-op without this, silently dropping our INFO logs.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    LOGGER.info("Generating eddy ODS for shot %s from %s", args.shot, args.diagnostics_ods)
    ods, manifest = build_eddy_ods(
        shot=args.shot,
        diagnostics_ods=args.diagnostics_ods,
        static_ods=args.static_ods,
        filament_r=_csv_floats(args.filament_r),
        filament_z=_csv_floats(args.filament_z),
        filament_fraction=_csv_floats(args.filament_fraction),
        dt_sub=args.dt_sub,
    )

    write_stage_product(ods, manifest, output=args.output, metadata=args.metadata)
    LOGGER.info("Eddy ODS saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
