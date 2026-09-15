#!/usr/bin/env python3
"""Map one shot's Thomson (and CES, when present) onto an equilibrium grid.

A slice the ion diagnostic covers becomes kinetic; every other stays
electron-only and is stripped of slice-total pressure, because a total computed
with a phantom Ti=Te would read as measured to anything that loads the product.

The manifest records how many of each were built. That count routes the
reconstruction afterwards -- a product with no kinetic slice can only support
the `electron-efit` lineage -- so it is written down rather than re-derived.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import traceback

from vaft.omas.vest_upstream import build_core_profiles_ods, write_stage_product

LOGGER = logging.getLogger("vaft.generate_core_profiles_ods")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--thomson-product", required=True, type=Path, help="Thomson stage product.")
    parser.add_argument("--ces-product", default="", help="CES stage product; empty when the shot has none.")
    parser.add_argument("--efit-product", default="", help="EFIT stage product supplying the psi map.")
    parser.add_argument("--geqdsk-dir", default="", help="CHEASE g-file directory, for shots predating the stage.")
    parser.add_argument("--output", required=True, type=Path, help="Output core_profiles ODS JSON.")
    parser.add_argument("--metadata", required=True, type=Path, help="Output stage manifest JSON.")
    parser.add_argument("--run", default=1, type=int, help="Dataset run number.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    try:
        ods, manifest = build_core_profiles_ods(
            shot=args.shot,
            thomson_product=args.thomson_product,
            ces_product=args.ces_product or None,
            efit_product=args.efit_product or None,
            geqdsk_dir=args.geqdsk_dir or None,
            run=args.run,
        )
    except Exception as error:  # noqa: BLE001 - the containment is the point
        LOGGER.exception("core_profiles stage failed for shot %s; recording", args.shot)
        from omas import ODS

        ods = ODS(consistency_check=False)
        manifest = {
            "schema_version": 1,
            "stage": "core_profiles",
            "shot": int(args.shot),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }

    write_stage_product(ods, manifest, output=args.output, metadata=args.metadata)
    LOGGER.info(
        "core_profiles ODS saved to %s (status=%s, slices=%s)",
        args.output, manifest["status"], manifest.get("slices"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
