#!/usr/bin/env python3
"""Build the `mhd_linear` IDS from a completed GPEC-suite run tree.

The legacy server workflow's `generate_mhd_linear_ods` rule was disconnected
from `rule all` even in production (see `test/data/vest_reference/manifest.yaml`),
so no linear-stability results ever reached an ODS on either the legacy path
or (until now) the VAFT-native one. This script is that missing rule for the
VAFT-native path: it reads every `run_gpec_module`-produced `(code, mode)`
cell already on disk for a shot and folds their DCON/RDCON/STRIDE `.nc`
output into one `mhd_linear` product, via `vaft.omas.vest_upstream`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vaft.omas.vest_upstream import build_mhd_linear_ods, write_stage_product

LOGGER = logging.getLogger("vaft.generate_mhd_linear_ods")


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


def _module_workdirs(values: list[str]) -> dict[tuple[str, int], Path]:
    result = {}
    for value in values:
        for entry in value.split(","):
            code, mode, path = entry.split(":", maxsplit=2)
            result[(code, int(mode))] = Path(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--refined-gfile-manifest", required=True, type=Path, help="CHEASE refined gfile manifest.")
    parser.add_argument("--workdir", type=Path, help="Legacy shared GPEC-suite run tree.")
    parser.add_argument("--module-workdir", nargs="+", default=[], help="Canonical code:mode:path GPEC work trees.")
    parser.add_argument("--modules", default="dcon,rdcon,stride", help="Comma-separated suite modules to fold in.")
    parser.add_argument("--modes", default="1,2", help="Comma-separated toroidal mode numbers to fold in.")
    parser.add_argument("--output", required=True, type=Path, help="Output mhd_linear ODS JSON.")
    parser.add_argument("--metadata", required=True, type=Path, help="Output stage manifest JSON.")
    args = parser.parse_args()
    if args.workdir is None and not args.module_workdir:
        parser.error("one of --workdir or --module-workdir is required")

    # force=True: vaft.database.raw installs a root handler at import time, which makes
    # basicConfig() a no-op without this, silently dropping our INFO logs.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )

    gfiles = _read_manifest(args.refined_gfile_manifest)
    time_values = [_time_label(gfile) for gfile in gfiles]

    LOGGER.info("Building mhd_linear ODS for shot %s (%d time slice(s))", args.shot, len(time_values))
    ods, manifest = build_mhd_linear_ods(
        shot=args.shot,
        time_values=time_values,
        workdir=args.workdir,
        module_workdirs=_module_workdirs(args.module_workdir),
        modules=_parse_csv(args.modules, str),
        modes=_parse_csv(args.modes, int),
    )
    write_stage_product(ods, manifest, output=args.output, metadata=args.metadata)
    LOGGER.info("mhd_linear ODS saved to %s (status=%s)", args.output, manifest["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
