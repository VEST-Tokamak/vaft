#!/usr/bin/env python3
"""Generate EFIT OMAS ODS from collected EFIT files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omas import ODS, save_omas_json

from vaft.code.efit import EFITConfig, collect_efit_outputs


LOGGER = logging.getLogger("vaft.generate_efit_ods")


def _minimal_efit_ods(shot: int, run: int, status: str) -> ODS:
    ods = ODS()
    ods["dataset_description.data_entry.machine"] = "VEST"
    ods["dataset_description.data_entry.pulse"] = int(shot)
    ods["dataset_description.data_entry.run"] = int(run)
    ods["equilibrium.ids_properties.comment"] = f"EFIT output unavailable: {status}"
    return ods


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--gfile-manifest", required=True, type=Path, help="Input gfile manifest.")
    parser.add_argument("--status", required=True, type=Path, help="Input EFIT status file.")
    parser.add_argument("--output", required=True, type=Path, help="Output EFIT ODS JSON.")
    parser.add_argument("--run", default=1, type=int, help="Dataset run number.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    workdir = args.gfile_manifest.parent.parent
    status_text = args.status.read_text(encoding="utf-8").strip() if args.status.exists() else "unknown"
    result = collect_efit_outputs(workdir, EFITConfig(workdir=workdir, shot=args.shot))

    ods = result.ods if result.ods is not None else _minimal_efit_ods(args.shot, args.run, status_text)
    if result.ods is not None:
        ods["dataset_description.data_entry.machine"] = "VEST"
        ods["dataset_description.data_entry.pulse"] = int(args.shot)
        ods["dataset_description.data_entry.run"] = int(args.run)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_omas_json(ods, str(args.output))
    LOGGER.info("EFIT ODS saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
