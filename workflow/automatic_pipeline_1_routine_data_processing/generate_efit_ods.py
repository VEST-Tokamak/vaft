#!/usr/bin/env python3
"""Generate EFIT OMAS ODS from collected EFIT files."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from omas import ODS, load_omas_json, save_omas_json

from vaft.code.efit import EFITConfig, collect_efit_outputs
from vaft.omas.vest_upstream import write_manifest


LOGGER = logging.getLogger("vaft.generate_efit_ods")


def _minimal_efit_ods(shot: int, run: int, status: str) -> ODS:
    ods = ODS()
    ods["dataset_description.data_entry.machine"] = "VEST"
    ods["dataset_description.data_entry.pulse"] = int(shot)
    ods["dataset_description.data_entry.run"] = int(run)
    ods["equilibrium.ids_properties.comment"] = f"EFIT output unavailable: {status}"
    return ods


def efit_collection_parameters(
    *,
    status: str,
    slice_statuses,
    mapping_diagnostics,
    artifact_hashes,
    artifact_manifest,
) -> str:
    """Serialize the EFIT collection payload for `equilibrium.code.parameters`.

    The DD types that field as a single string, so everything the collection
    records has to travel inside one serialized document. Kept as a function so
    a reader can recover the payload with `json.loads` and a test can pin the
    round trip without running the stage.
    """
    return json.dumps(
        {
            "efit_collection": {
                "status": status,
                "slice_statuses": [s.to_dict() for s in slice_statuses],
                "mapping_diagnostics": list(mapping_diagnostics),
                "artifact_hashes": dict(artifact_hashes),
                "artifact_manifest": artifact_manifest,
            }
        },
        sort_keys=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--gfile-manifest", required=True, type=Path, help="Input gfile manifest.")
    parser.add_argument("--status", required=True, type=Path, help="Input EFIT status file.")
    parser.add_argument("--constraints-ods", required=True, type=Path, help="Submitted EFIT constraints ODS.")
    parser.add_argument("--kfile-manifest", required=True, type=Path, help="Input kfile manifest.")
    parser.add_argument("--artifact-manifest", required=True, type=Path, help="Structured EFIT artifact manifest.")
    parser.add_argument("--output", required=True, type=Path, help="Output EFIT ODS JSON.")
    parser.add_argument("--metadata", required=True, type=Path, help="Output stage manifest.")
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
        # `code.parameters` is a STR_0D in the DD, so the whole payload is one
        # serialized string. Writing it as a nested tree looks right locally and
        # is dropped entirely on the way through the Access Layer -- 4096 paths
        # of collection provenance vanished between the FileDB product and its
        # HSDS replica before this was a string (issue #380). CHEASE has always
        # serialized this field; EFIT now does the same.
        #
        # Arbitrary case and path keys stay inside the JSON rather than becoming
        # ODS path components: a numeric label such as `039915.00316` is not a
        # path segment.
        ods["equilibrium.code.parameters"] = efit_collection_parameters(
            status=status_text,
            slice_statuses=result.slice_statuses,
            mapping_diagnostics=result.mapping_diagnostics,
            artifact_hashes=result.artifact_hashes,
            artifact_manifest=json.loads(
                args.artifact_manifest.read_text(encoding="utf-8")
            ),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_omas_json(ods, str(args.output))

    # The stage manifest is what tells a consumer whether this product is a
    # result or a placeholder. A run that collected nothing still leaves its
    # ODS on disk for inspection, but says so here rather than letting a hollow
    # equilibrium look like a reconstruction.
    write_manifest(
        {
            "schema_version": 1,
            "stage": "efit",
            "shot": int(args.shot),
            "run": int(args.run),
            "status": "success" if result.ods is not None else "no_output",
            "efit_status": status_text,
            "slice_statuses": [status.to_dict() for status in result.slice_statuses],
            "mapping_diagnostics": list(result.mapping_diagnostics),
        },
        args.metadata,
    )
    LOGGER.info("EFIT ODS saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
