"""Run the raw, static, diagnostics, and eddy VEST OMAS stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from vaft.omas.vest_upstream import (
    archive_raw_source,
    build_diagnostics_ods,
    build_eddy_ods,
    build_static_ods,
    write_manifest,
    write_stage_product,
)


def _csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vaft.cli vest-upstream")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    raw = subparsers.add_parser("raw", help="archive one source/shot raw dump")
    raw.add_argument("--shot", required=True, type=int)
    raw.add_argument("--source", type=Path)
    raw.add_argument("--output", required=True, type=Path)
    raw.add_argument("--metadata", required=True, type=Path)

    static = subparsers.add_parser("static", help="build one machine-era static ODS")
    static.add_argument("--machine-version", required=True)
    static.add_argument("--output", required=True, type=Path)
    static.add_argument("--metadata", required=True, type=Path)

    diagnostics = subparsers.add_parser("diagnostics", help="build diagnostic ODS")
    diagnostics.add_argument("--shot", required=True, type=int)
    diagnostics.add_argument("--raw-source", required=True, type=Path)
    diagnostics.add_argument("--static-ods", required=True, type=Path)
    diagnostics.add_argument("--output", required=True, type=Path)
    diagnostics.add_argument("--metadata", required=True, type=Path)
    # Unset means "use the configured plasma-analysis window".
    diagnostics.add_argument("--tstart", default=None, type=float)
    diagnostics.add_argument("--tend", default=None, type=float)
    diagnostics.add_argument("--dt", default=None, type=float)
    diagnostics.add_argument("--run", default=1, type=int)
    diagnostics.add_argument("--vest-magnetics-processing-json", default="")
    diagnostics.add_argument(
        "--time-policies-json",
        default="",
        help=(
            "JSON object overriding the per-component diagnostics time policies "
            "configured in vest.yaml. --tstart/--tend/--dt retune the default "
            "(plasma-analysis) window only."
        ),
    )

    eddy = subparsers.add_parser("eddy", help="build eddy-current ODS")
    eddy.add_argument("--shot", required=True, type=int)
    eddy.add_argument("--diagnostics-ods", required=True, type=Path)
    eddy.add_argument("--static-ods", required=True, type=Path)
    eddy.add_argument("--output", required=True, type=Path)
    eddy.add_argument("--metadata", required=True, type=Path)
    eddy.add_argument("--filament-r", default="0.35,0.35,0.35")
    eddy.add_argument("--filament-z", default="0.25,0.0,-0.25")
    eddy.add_argument("--filament-fraction", default="0.3333333,0.3333333,0.3333333")
    eddy.add_argument("--dt-sub", default=5e-5, type=float)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.stage == "raw":
        manifest = archive_raw_source(shot=args.shot, source=args.source, output=args.output)
        write_manifest(manifest, args.metadata)
        return 0
    if args.stage == "static":
        ods, manifest = build_static_ods(args.machine_version)
    elif args.stage == "diagnostics":
        ods, manifest = build_diagnostics_ods(
            shot=args.shot,
            raw_source=args.raw_source,
            static_ods=args.static_ods,
            tstart=args.tstart,
            tend=args.tend,
            dt=args.dt,
            run=args.run,
            vest_magnetics_processing=(
                json.loads(args.vest_magnetics_processing_json)
                if args.vest_magnetics_processing_json.strip()
                else None
            ),
            time_policies=(
                json.loads(args.time_policies_json)
                if args.time_policies_json.strip()
                else None
            ),
        )
    else:
        ods, manifest = build_eddy_ods(
            shot=args.shot,
            diagnostics_ods=args.diagnostics_ods,
            static_ods=args.static_ods,
            filament_r=_csv_floats(args.filament_r),
            filament_z=_csv_floats(args.filament_z),
            filament_fraction=_csv_floats(args.filament_fraction),
            dt_sub=args.dt_sub,
        )
    write_stage_product(ods, manifest, output=args.output, metadata=args.metadata)
    return 0


__all__ = ["main"]
