#!/usr/bin/env python3
"""Store one shot's Thomson scattering upload as a stage product.

Thomson is an externally produced diagnostic: a `.mat` file is uploaded by hand
and most shots never get one. That is a normal outcome, not a failure, so a shot
with no upload yields a provenance-only product whose manifest says
``unavailable`` and the stage exits 0. Whether the product is eligible to be
replicated is decided by that status, never by this exit code.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import traceback

from vaft.omas.vest_upstream import build_thomson_ods, write_stage_product

LOGGER = logging.getLogger("vaft.generate_thomson_ods")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--output", required=True, type=Path, help="Output Thomson ODS JSON.")
    parser.add_argument("--metadata", required=True, type=Path, help="Output stage manifest JSON.")
    parser.add_argument("--data-root", default=None, help="Directory the uploads are searched under.")
    parser.add_argument("--mat-file", default=None, help="Explicit .mat file, overriding the search.")
    parser.add_argument("--run", default=1, type=int, help="Dataset run number.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    try:
        ods, manifest = build_thomson_ods(
            shot=args.shot,
            data_root=args.data_root or None,
            mat_file=args.mat_file or None,
            run=args.run,
        )
    except Exception as error:  # noqa: BLE001 - the containment is the point
        LOGGER.exception("Thomson stage failed for shot %s; recording and continuing", args.shot)
        from omas import ODS

        ods = ODS(consistency_check=False)
        manifest = {
            "schema_version": 1,
            "stage": "thomson",
            "shot": int(args.shot),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }

    write_stage_product(ods, manifest, output=args.output, metadata=args.metadata)
    LOGGER.info("Thomson ODS saved to %s (status=%s)", args.output, manifest["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
