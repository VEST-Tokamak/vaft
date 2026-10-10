#!/usr/bin/env python3
"""Add the VEST impurity composition, diluted n_i and Z_eff(rho) to one shot's core profiles.

Reads the measured ``core_profiles`` product and writes a product that is the
same plus the assumed C/O composition of ``vest.yaml``'s ``impurity_model``:
``ion[]`` with charge states, the diluted main ion and ``zeff``, each with an
``origin=assumed|derived`` provenance record. It is a separate stage with its
own manifest so a shot that cannot be composed (no model, no plasma age, a
measured Z_eff already) is *recorded* and keeps its measured product in HSDS.

Every outcome is written and the exit status is 0, as for the other stages of
this pipeline: read the manifest ``status``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import traceback

from vaft.omas.vest_upstream import build_composition_ods, write_stage_product

LOGGER = logging.getLogger("vaft.generate_composition_ods")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--core-profiles-product", required=True, type=Path, help="Measured core_profiles stage product.")
    parser.add_argument("--efit-product", default="", help="Magnetic EFIT product: volumes and lineage.")
    parser.add_argument("--diagnostics-product", default="", help="Diagnostics product: plasma onset.")
    parser.add_argument("--ionization", default="transient", choices=("transient", "coronal"))
    parser.add_argument("--normalization", default="ne_weighted_mean")
    parser.add_argument("--adf11-cache-dir", default="", help="OPEN-ADAS ADF11 cache; empty uses the user cache.")
    parser.add_argument("--output", required=True, type=Path, help="Output composed core_profiles ODS JSON.")
    parser.add_argument("--metadata", required=True, type=Path, help="Output stage manifest JSON.")
    parser.add_argument("--run", default=1, type=int, help="Dataset run number.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    try:
        ods, manifest = build_composition_ods(
            shot=args.shot,
            core_profiles_product=args.core_profiles_product,
            efit_product=args.efit_product or None,
            diagnostics_product=args.diagnostics_product or None,
            ionization=args.ionization,
            normalization=args.normalization,
            cache_dir=args.adf11_cache_dir or None,
            run=args.run,
        )
    except Exception as error:  # noqa: BLE001 - the containment is the point
        LOGGER.exception("composition stage failed for shot %s; recording", args.shot)
        from omas import ODS

        ods = ODS(consistency_check=False)
        manifest = {
            "schema_version": 1,
            "stage": "composition",
            "shot": int(args.shot),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }

    write_stage_product(ods, manifest, output=args.output, metadata=args.metadata)
    LOGGER.info(
        "composition ODS saved to %s (status=%s, slices=%s)",
        args.output, manifest["status"], manifest.get("slices"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
