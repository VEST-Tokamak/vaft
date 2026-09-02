#!/usr/bin/env python3
"""Generate EFIT OMAS ODS from collected EFIT files."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from omas import ODS, load_omas_json, save_omas_json

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
    parser.add_argument("--constraints-ods", required=True, type=Path, help="Submitted EFIT constraints ODS.")
    parser.add_argument("--kfile-manifest", required=True, type=Path, help="Input kfile manifest.")
    parser.add_argument("--artifact-manifest", required=True, type=Path, help="Structured EFIT artifact manifest.")
    parser.add_argument("--output", required=True, type=Path, help="Output EFIT ODS JSON.")
    parser.add_argument("--run", default=1, type=int, help="Dataset run number.")
    args = parser.parse_args()

    # force=True: vaft.database.raw installs a root handler at import time, which makes
    # basicConfig() a no-op without this, silently dropping our INFO logs.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    workdir = args.gfile_manifest.parent.parent
    status_text = args.status.read_text(encoding="utf-8").strip() if args.status.exists() else "unknown"
    constraints_ods = load_omas_json(str(args.constraints_ods), consistency_check=False)
    kfiles = tuple(Path(line.strip()) for line in args.kfile_manifest.read_text(encoding="utf-8").splitlines() if line.strip())
    result = collect_efit_outputs(
        workdir, EFITConfig(workdir=workdir, shot=args.shot),
        expected_kfiles=kfiles, constraints_ods=constraints_ods,
    )

    ods = result.ods if result.ods is not None else _minimal_efit_ods(args.shot, args.run, status_text)
    if result.ods is not None:
        ods["dataset_description.data_entry.machine"] = "VEST"
        ods["dataset_description.data_entry.pulse"] = int(args.shot)
        ods["dataset_description.data_entry.run"] = int(args.run)
        ods["equilibrium.code.parameters.efit_collection.status"] = status_text
        ods["equilibrium.code.parameters.efit_collection.slice_statuses"] = [status.to_dict() for status in result.slice_statuses]
        ods["equilibrium.code.parameters.efit_collection.mapping_diagnostics"] = list(result.mapping_diagnostics)
        # Keep arbitrary case/path keys opaque. Numeric labels such as
        # ``039915.00316`` otherwise become ODS path components during reload.
        ods["equilibrium.code.parameters.efit_collection.artifact_hashes_json"] = json.dumps(
            dict(result.artifact_hashes), sort_keys=True
        )
        ods["equilibrium.code.parameters.efit_collection.artifact_manifest_json"] = json.dumps(
            json.loads(args.artifact_manifest.read_text(encoding="utf-8")),
            sort_keys=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_omas_json(ods, str(args.output))
    LOGGER.info("EFIT ODS saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
