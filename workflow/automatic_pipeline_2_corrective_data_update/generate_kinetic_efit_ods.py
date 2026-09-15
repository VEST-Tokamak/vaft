#!/usr/bin/env python3
"""Reconstruct one shot's kinetic-pressure equilibrium into its own lineage.

`--stage` is `electron_efit` or `kinetic_efit`, and it must agree with what the
core-profile product actually measured: the ion diagnostic is what separates a
measured Ti from one assumed through a Ti/Te ratio. Both stages are asked for
every shot and at most one can apply, so a mismatch is recorded as unavailable
rather than raised -- the other stage is the one that will answer.

Nothing is read from outside the FileDB. The base magnetic kfile is built from
the EFIT stage's own constraints product, and the psi map from the magnetic
solution.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import traceback

from vaft.omas.vest_upstream import build_kinetic_efit_ods, write_stage_product

LOGGER = logging.getLogger("vaft.generate_kinetic_efit_ods")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument(
        "--stage", required=True, choices=("electron_efit", "kinetic_efit"),
        help="Which lineage this run is for.",
    )
    parser.add_argument("--core-profiles-product", required=True, type=Path)
    parser.add_argument("--constraints-product", required=True, type=Path)
    parser.add_argument("--efit-product", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output equilibrium ODS JSON.")
    parser.add_argument("--metadata", required=True, type=Path, help="Output stage manifest JSON.")
    parser.add_argument("--time-ms", default=None, type=float, help="Reconstruction time; default is the best-aligned slice.")
    parser.add_argument("--workdir", default="", help="EFIT working directory; a temporary one is used and removed when empty.")
    parser.add_argument("--executable", default="", help="EFIT binary; resolved from $EFIT when empty.")
    parser.add_argument("--encoding", default="raw6", help="Pressure-point encoding: raw6, raw5 or spline.")
    parser.add_argument("--run", default=1, type=int, help="Dataset run number.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    try:
        ods, manifest = build_kinetic_efit_ods(
            shot=args.shot,
            stage=args.stage,
            core_profiles_product=args.core_profiles_product,
            constraints_product=args.constraints_product,
            efit_product=args.efit_product,
            time_ms=args.time_ms,
            workdir=args.workdir or None,
            executable=args.executable or None,
            encoding=args.encoding,
            run=args.run,
        )
    except Exception as error:  # noqa: BLE001 - the containment is the point
        LOGGER.exception("%s stage failed for shot %s; recording", args.stage, args.shot)
        from omas import ODS

        ods = ODS(consistency_check=False)
        manifest = {
            "schema_version": 1,
            "stage": args.stage,
            "shot": int(args.shot),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }

    write_stage_product(ods, manifest, output=args.output, metadata=args.metadata)
    LOGGER.info(
        "%s ODS saved to %s (status=%s, reconstruction=%s)",
        args.stage, args.output, manifest["status"], manifest.get("reconstruction"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
