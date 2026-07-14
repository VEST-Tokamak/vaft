#!/usr/bin/env python3
"""Generate a CHEASE OMAS ODS from refined CHEASE g-files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omas import ODS, save_omas_json

from vaft.data.eqdsk import read_geqdsk


LOGGER = logging.getLogger("vaft.generate_chease_ods")


def _minimal_chease_ods(shot: int, run: int, status: str) -> ODS:
    ods = ODS()
    ods["dataset_description.data_entry.machine"] = "VEST"
    ods["dataset_description.data_entry.pulse"] = int(shot)
    ods["dataset_description.data_entry.run"] = int(run)
    ods["equilibrium.ids_properties.comment"] = f"CHEASE output unavailable: {status}"
    return ods


def _read_manifest(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        return ()
    return tuple(Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--refined-gfile-manifest", required=True, type=Path, help="Input CHEASE refined gfile manifest.")
    parser.add_argument("--status", required=True, type=Path, help="Input CHEASE status file.")
    parser.add_argument("--output", required=True, type=Path, help="Output CHEASE ODS JSON.")
    parser.add_argument("--run", default=1, type=int, help="Dataset run number.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    status_text = args.status.read_text(encoding="utf-8").strip() if args.status.exists() else "unknown"
    gfiles = _read_manifest(args.refined_gfile_manifest)

    ods = None
    parse_errors = []
    for time_index, gfile in enumerate(gfiles):
        try:
            ods = read_geqdsk(gfile).to_omas(ods=ods, time_index=time_index)
        except Exception as exc:
            parse_errors.append(f"{gfile}: {exc}")
            LOGGER.warning("Could not parse CHEASE gfile %s: %s", gfile, exc)

    if ods is None:
        ods = _minimal_chease_ods(args.shot, args.run, status_text)
    else:
        ods["dataset_description.data_entry.machine"] = "VEST"
        ods["dataset_description.data_entry.pulse"] = int(args.shot)
        ods["dataset_description.data_entry.run"] = int(args.run)
        if parse_errors:
            ods["equilibrium.ids_properties.comment"] = "CHEASE parse warnings: " + "; ".join(parse_errors[:5])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_omas_json(ods, str(args.output))
    LOGGER.info("CHEASE ODS saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
