#!/usr/bin/env python3
"""Build `mhd_linear` + `coils_non_axisymmetric` from ideal-GPEC run trees.

Companion to `generate_mhd_linear_ods.py` for the `gpec` (ideal-GPEC)
module: it reads every `run_gpec_module`-produced ideal-GPEC cell already on
disk for a shot and folds the control/cylindrical `.nc` output into one
`gpec_ideal` stage product via `vaft.omas.vest_upstream.build_gpec_ideal_ods`,
including the canonical 3D coil geometry and each run's coil excitation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vaft.omas.vest_upstream import build_gpec_ideal_ods, write_stage_product

LOGGER = logging.getLogger("vaft.generate_gpec_ideal_ods")


def _read_manifest(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        raise FileNotFoundError(f"Refined gfile manifest not found: {path}")
    return tuple(Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _parse_csv(text: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in str(text).split(",") if item.strip())


def _time_label(path: Path) -> str:
    if "." in path.name:
        return path.name.rsplit(".", maxsplit=1)[-1]
    return path.stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--refined-gfile-manifest", required=True, type=Path, help="CHEASE refined gfile manifest.")
    parser.add_argument("--workdir", type=Path, help="Shared ideal-GPEC run tree root.")
    parser.add_argument(
        "--mode-workdir", nargs="+", default=[],
        help="Canonical mode:path ideal-GPEC work trees (repeatable).",
    )
    parser.add_argument("--modes", default="1", help="Comma-separated toroidal mode numbers to fold in.")
    parser.add_argument("--output", required=True, type=Path, help="Output gpec_ideal ODS JSON.")
    parser.add_argument("--metadata", required=True, type=Path, help="Output stage manifest JSON.")
    args = parser.parse_args()
    if args.workdir is None and not args.mode_workdir:
        parser.error("one of --workdir or --mode-workdir is required")

    mode_workdirs: dict[int, Path] = {}
    for value in args.mode_workdir:
        for entry in value.split(","):
            mode, path = entry.split(":", maxsplit=1)
            mode_workdirs[int(mode)] = Path(path)

    # force=True: vaft.database.raw installs a root handler at import time, which makes
    # basicConfig() a no-op without this, silently dropping our INFO logs.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )

    gfiles = _read_manifest(args.refined_gfile_manifest)
    time_values = [_time_label(gfile) for gfile in gfiles]

    LOGGER.info("Building gpec_ideal ODS for shot %s (%d time slice(s))", args.shot, len(time_values))
    ods, manifest = build_gpec_ideal_ods(
        shot=args.shot,
        time_values=time_values,
        workdir=args.workdir,
        mode_workdirs=mode_workdirs,
        modes=_parse_csv(args.modes, int),
    )
    write_stage_product(ods, manifest, output=args.output, metadata=args.metadata)
    LOGGER.info("gpec_ideal ODS saved to %s (status=%s)", args.output, manifest["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
