#!/usr/bin/env python3
"""Replicate one completed canonical FileDB stage product into HSDS.

The pipeline's stages own different parts of the IMAS tree and finish at
different times -- and for different sets of shots. A vacuum shot has
diagnostics and eddy and no equilibrium; a shot whose EFIT never converged still
has everything upstream of it. So each stage is replicated on its own, carrying
only the IDS subtree it owns, rather than one cumulative product being assembled
first and sent once.

Where each stage goes, and what of it travels, is
`vaft.database.sources.STAGE_REPLICATION`. This script does not decide either --
duplicating that table in the workflow is how the two would drift apart.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="vaft-mpl-"))

from vaft.database.filedb import FileDB
from vaft.database.replication import replicate_stage


LOGGER = logging.getLogger("vaft.replicate_to_hsds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument(
        "--stage", required=True, help="Canonical FileDB OMAS stage to replicate."
    )
    parser.add_argument(
        "--filedb-root", required=True, help="Canonical FileDB root directory."
    )
    parser.add_argument(
        "--attempts",
        default=3,
        type=int,
        help="Bounded retries for transient remote failures.",
    )
    parser.add_argument(
        "--retry-delay", default=5.0, type=float, help="Seconds between attempts."
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the read-back comparison. The record then says 'replicated' "
        "rather than 'validated', so a later run can still complete it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-send even when the recorded product hash still matches.",
    )
    args = parser.parse_args()

    # force=True: vaft.database.raw installs a root handler at import time, which
    # makes basicConfig() a no-op without this, silently dropping our INFO logs.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    record = replicate_stage(
        args.stage,
        args.shot,
        filedb=FileDB(args.filedb_root),
        attempts=args.attempts,
        retry_delay=args.retry_delay,
        validate=not args.no_validate,
        force=args.force,
    )
    LOGGER.info(
        "shot %s %s -> %s (%s): %s",
        record.shot,
        record.stage,
        record.source,
        record.state,
        ", ".join(record.ids),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
