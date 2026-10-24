#!/usr/bin/env python3
"""Infer an ion temperature on the electron-only slices from the magnetic pressure partition.

Reads the *composed* core_profiles product and pipeline 1's magnetic EFIT and
writes the same product with ``T_i = p_eq / (e f n_e) - T_e / f`` on the slices
that have no measured ion temperature: ``origin=inferred`` in its record, NaN
where the ion pressure is not positive and significant, and the slice
pressures recomputed from it. Slices with a measured T_i are left alone.

Only the ``efit/magnetic`` lineage is accepted, read from where the EFIT
product lives. This stage's product must never feed the kinetic EFIT stages:
their pressure would then be partitioned back into the T_i it was built from.

Every outcome is written and the exit status is 0: read the manifest ``status``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import traceback

from vaft.omas.vest_upstream import (
    TI_INFERENCE_FLOOR,
    TI_INFERENCE_WINDOW_S,
    build_ti_inferred_ods,
    write_stage_product,
)

LOGGER = logging.getLogger("vaft.generate_ti_inferred_ods")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--core-profiles-product", required=True, type=Path, help="Composed core_profiles stage product.")
    parser.add_argument("--efit-product", required=True, type=Path, help="Pipeline 1's magnetic EFIT product.")
    parser.add_argument("--floor", default=TI_INFERENCE_FLOOR, type=float, help="Relative sigma floor of p_EFIT (#874).")
    parser.add_argument("--window-s", default=TI_INFERENCE_WINDOW_S, type=float, help="Ensemble window for sigma(p_EFIT) [s].")
    parser.add_argument("--output", required=True, type=Path, help="Output core_profiles ODS JSON.")
    parser.add_argument("--metadata", required=True, type=Path, help="Output stage manifest JSON.")
    parser.add_argument("--run", default=1, type=int, help="Dataset run number.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    try:
        ods, manifest = build_ti_inferred_ods(
            shot=args.shot,
            core_profiles_product=args.core_profiles_product,
            efit_product=args.efit_product,
            floor=args.floor,
            window_s=args.window_s,
            run=args.run,
        )
    except Exception as error:  # noqa: BLE001 - the containment is the point
        LOGGER.exception("ti_inferred stage failed for shot %s; recording", args.shot)
        from omas import ODS

        from vaft.database._local import load_ods

        # Nothing is replicated for a failed stage; keep the composed profiles in
        # the product anyway, so it is never an empty file that looks like data.
        try:
            ods, _ = load_ods(args.core_profiles_product)
        except Exception:  # noqa: BLE001 - an unreadable input leaves nothing to pass on
            ods = ODS(consistency_check=False)
        manifest = {
            "schema_version": 1,
            "stage": "ti_inferred",
            "shot": int(args.shot),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }

    write_stage_product(ods, manifest, output=args.output, metadata=args.metadata)
    LOGGER.info(
        "ti_inferred ODS saved to %s (status=%s, slices=%s)",
        args.output, manifest["status"], manifest.get("slices"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
