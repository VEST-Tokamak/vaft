#!/usr/bin/env python3
"""Compose the canonical shot sample from VAFT eddy and EFIT stage products."""

from __future__ import annotations

import argparse
from pathlib import Path

from omas import ODS

import vaft


def compose(eddy_stage: Path, efit_stage: Path, *, shot: int) -> ODS:
    eddy = vaft.omas.load(eddy_stage)
    efit = vaft.omas.load(efit_stage)
    canonical = ODS(consistency_check=False)
    for ids_name in sorted(eddy.keys()):
        canonical[ids_name] = eddy[ids_name]
    if "equilibrium" not in efit:
        raise ValueError(f"EFIT stage has no equilibrium IDS: {efit_stage}")
    canonical["equilibrium"] = efit["equilibrium"]
    # The diagnostics-stage description contains the archived pulse time and
    # richer mapping provenance; do not overwrite it with EFIT's minimal one.
    if canonical.get("dataset_description.data_entry.pulse") != int(shot):
        raise ValueError(f"Canonical reference source must describe shot {shot}")
    return canonical


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eddy-stage", required=True, type=Path)
    parser.add_argument("--efit-stage", required=True, type=Path)
    parser.add_argument("--shot", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    canonical = compose(
        args.eddy_stage.resolve(), args.efit_stage.resolve(), shot=args.shot
    )
    vaft.omas.save(canonical, args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
