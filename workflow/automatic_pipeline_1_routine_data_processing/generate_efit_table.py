"""Build the EFIT Green table for one machine era (rule ``generate_efit_table``).

The packaged table under ``efit.table_dir`` was built for one era; a shot of
another era needs its own, because a Green table is a projection of that era's
conductors (#805).  This builds it with the base table's EFUND configuration --
grid, flags, quadrature -- so the two tables differ only in machine geometry.
See :func:`vaft.code.efit.efund.generate_era_table` for the chain.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vaft.code.efit.efund import (
    TableExistsError,
    efund_config_from_manifest,
    generate_era_table,
    read_table_manifest,
    table_identity,
)

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--era", required=True, help="machine era name (vaft.omas.vest_upstream)")
    parser.add_argument("--base-table-dir", required=True, type=Path, help="the table whose EFUND configuration to reuse")
    parser.add_argument("--output-dir", required=True, type=Path, help="table directory to build")
    parser.add_argument("--efund", default="", help="efund executable (default: $EFITHOME/bin/efund)")
    parser.add_argument("--timeout", default="", help="seconds before EFUND is stopped")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    base = read_table_manifest(args.base_table_dir)
    if base is None:
        LOGGER.error("%s has no table manifest to take the EFUND configuration from", args.base_table_dir)
        return 2
    overrides = {}
    if args.efund:
        overrides["executable"] = args.efund
    if args.timeout:
        overrides["timeout"] = float(args.timeout)
    config = efund_config_from_manifest(base, **overrides)
    LOGGER.info("building the %s table (%s x %s) into %s", args.era, config.nw, config.nh, args.output_dir)

    try:
        result = generate_era_table(
            args.era,
            args.output_dir,
            config=config,
            base_table_dir=args.base_table_dir,
            # The directory is this rule's own output: Snakemake's cleanup before a
            # rerun removes only the manifest, so what is left is ours to replace.
            replace_incomplete=True,
            extra={
                "generator": "workflow/automatic_pipeline_1_routine_data_processing/generate_efit_table.py",
                "configuration_from": table_identity(args.base_table_dir),
            },
        )
    except TableExistsError:
        # Another pipeline process sharing this FileDB built it while this one
        # waited for the lock: the output exists and is complete.
        LOGGER.info("table already built by another process: %s", args.output_dir)
        return 0
    if not result.ok:
        LOGGER.error("efund %s: %s", result.status, result.reason)
        for path in result.logs:
            LOGGER.error("  log: %s", path)
        return 1
    LOGGER.info("table ready: %s", result.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
