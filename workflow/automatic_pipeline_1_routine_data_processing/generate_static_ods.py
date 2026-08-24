#!/usr/bin/env python3
"""Build one finalized, versioned VEST static machine ODS.

The legacy workflow selected a static electromagnetic-coupling file by shot
number (`emcoupling_rules` in its `config.yaml`, split at 43017 and 45967) and
kept those files under `static_file_dir`. This stage is the VAFT-native
equivalent: `vaft.omas.vest_upstream` owns the machine-era table, so the
boundaries live in one place and the product is built rather than hand-curated.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vaft.omas.vest_upstream import build_static_ods, write_stage_product


LOGGER = logging.getLogger("vaft.generate_static_ods")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-version", required=True, help="VEST machine era name.")
    parser.add_argument("--output", required=True, type=Path, help="Output static ODS JSON.")
    parser.add_argument("--metadata", required=True, type=Path, help="Output stage manifest JSON.")
    args = parser.parse_args()

    # force=True: vaft.database.raw installs a root handler at import time, which makes
    # basicConfig() a no-op without this, silently dropping our INFO logs.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    LOGGER.info("Building static ODS for machine era %s", args.machine_version)
    ods, manifest = build_static_ods(args.machine_version)
    write_stage_product(ods, manifest, output=args.output, metadata=args.metadata)
    LOGGER.info("Static ODS saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
