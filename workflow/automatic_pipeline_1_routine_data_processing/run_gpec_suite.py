#!/usr/bin/env python3
"""Run the VEST GPEC suite from CHEASE-refined g-files."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

from vaft.code.gpec import (
    GPEC_HOME_ENV,
    CoilInputSpec,
    GPECCaseInputs,
    GPECSuiteConfig,
    GPECSuiteResult,
    IdealGPECOptions,
    run_gpec_suite_case,
)


def _read_manifest(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        raise FileNotFoundError(f"Refined gfile manifest not found: {path}")
    return tuple(Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _parse_csv(text: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in str(text).split(",") if item.strip())


def _runner_module(code: str) -> str:
    """Translate FileDB's unambiguous ideal-GPEC path code to the executable key."""
    return "gpec" if code == "ideal-gpec" else code


def _time_label(path: Path) -> str:
    if "." in path.name:
        return path.name.rsplit(".", maxsplit=1)[-1]
    return path.stem


def _path_text(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_path_text(item) for item in value]
    if isinstance(value, list):
        return [_path_text(item) for item in value]
    if isinstance(value, dict):
        return {key: _path_text(item) for key, item in value.items()}
    return value


def _result_to_dict(result: GPECSuiteResult) -> dict[str, Any]:
    payload = asdict(result)
    return _path_text(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--refined-gfile-manifest", required=True, type=Path, help="CHEASE refined gfile manifest.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON run manifest.")
    parser.add_argument("--status", required=True, type=Path, help="Output status text file.")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="GPEC-suite run tree root. Defaults to --output's directory (the whole-shot aggregate case).",
    )
    parser.add_argument(
        "--dcon-workdir",
        type=Path,
        default=None,
        help="Optional DCON work-tree root for ideal-GPEC when codes use separate work trees.",
    )
    parser.add_argument("--gpec-home", default="", help=f"GPEC source/install root. Defaults to ${GPEC_HOME_ENV}.")
    parser.add_argument("--run-mode", default="auto", help="auto, prepare_only, or strict.")
    parser.add_argument("--modules", default="dcon,rdcon,stride,gpec", help="Comma-separated suite modules.")
    parser.add_argument("--modes", default="1,2", help="Comma-separated toroidal mode numbers.")
    parser.add_argument(
        "--code", default="", help="Single GPEC-suite module for this target (e.g. dcon). Overrides --modules."
    )
    parser.add_argument(
        "--mode", default=None, type=int, help="Single toroidal mode for this target. Overrides --modes."
    )
    parser.add_argument("--psilow", default=0.01, type=float, help="Minimum normalized psi.")
    parser.add_argument("--psihigh", default=0.994, type=float, help="Maximum normalized psi.")
    parser.add_argument("--timeout", default=1200.0, type=float, help="Per executable timeout in seconds.")
    parser.add_argument("--templates-dir", default="", help="Optional GPEC namelist template directory.")
    parser.add_argument("--coil-data-dir", default="", help="Optional VEST GPEC coil data directory.")
    parser.add_argument(
        "--coil-set",
        action="append",
        default=[],
        metavar="NAME=I1,I2,...",
        help=(
            "Activate one canonical VEST 3D coil set with per-sector currents "
            "in amperes (repeatable), e.g. MID=200,200,0,-200,-200,0. The "
            "ideal-GPEC coil.in and staged .dat files are then generated from "
            "the canonical configuration instead of the packaged template."
        ),
    )
    args = parser.parse_args()

    coil_specs = []
    for entry in args.coil_set:
        if "=" not in entry:
            parser.error(f"--coil-set expects NAME=I1,I2,...; got {entry!r}")
        name, currents = entry.split("=", 1)
        coil_specs.append(
            CoilInputSpec(name=name.strip(), currents_a=_parse_csv(currents, float))
        )

    gfiles = _read_manifest(args.refined_gfile_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    # prepare_gpec_suite_case()/run_gpec_suite_case() already create workdir.
    workdir = args.workdir if args.workdir is not None else args.output.parent

    gpec_home = Path(args.gpec_home).expanduser() if args.gpec_home else None

    config = GPECSuiteConfig(
        gpec_home=gpec_home,
        modules=(_runner_module(args.code),) if args.code else _parse_csv(args.modules, str),
        modes=(args.mode,) if args.mode is not None else _parse_csv(args.modes, int),
        run_mode=args.run_mode,
        templates_dir=Path(args.templates_dir).expanduser() if args.templates_dir else None,
        coil_data_dir=Path(args.coil_data_dir).expanduser() if args.coil_data_dir else None,
        psilow=args.psilow,
        psihigh=args.psihigh,
        timeout=args.timeout,
        gpec=IdealGPECOptions(coil_specs=tuple(coil_specs) or None),
    )

    results = []
    for gfile in gfiles:
        case = GPECCaseInputs(
            shot=args.shot,
            time_ms=_time_label(gfile),
            geqdsk=gfile,
            workdir=workdir,
            dcon_workdir=args.dcon_workdir,
        )
        result = run_gpec_suite_case(case, config)
        results.append(result)

    payload = {
        "shot": int(args.shot),
        "refined_gfiles": [str(path) for path in gfiles],
        "workdir": str(workdir),
        "dcon_workdir": str(args.dcon_workdir) if args.dcon_workdir else "",
        "config": {
            "gpec_home": str(gpec_home) if gpec_home else os.environ.get(GPEC_HOME_ENV, ""),
            "modules": list(config.modules),
            "modes": [int(mode) for mode in config.modes],
            "run_mode": args.run_mode,
            "psilow": args.psilow,
            "psihigh": args.psihigh,
            "timeout": args.timeout,
            "coil_sets": [
                {"name": spec.name, "currents_a": list(spec.currents_a)}
                for spec in coil_specs
            ],
        },
        "results": [_result_to_dict(result) for result in results],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    records = [record for result in results for record in result.records]
    completed = [record for record in records if record.status == "completed"]
    skipped = [record for record in records if record.status == "skipped"]
    failed = [record for record in records if record.status == "failed"]
    status = f"completed={len(completed)}; skipped={len(skipped)}; failed={len(failed)}; cases={len(results)}"
    if failed:
        # A failed numerical time slice is data, not a workflow crash.  The
        # aggregate manifest retains it and mhd_linear incorporates the
        # successfully produced slices from this code/mode cell.
        status = "partial: " + status
    elif completed:
        status = "completed: " + status
    else:
        status = "skipped: " + status
    args.status.write_text(status + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
