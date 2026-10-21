#!/usr/bin/env python3
"""Compose the canonical shot sample from VAFT static, diagnostics, eddy and EFIT stage products.

Each stage of the routine pipeline publishes only the IDS subtree it owns
(``vaft.database.sources.STAGE_REPLICATION``): the eddy product carries
``pf_passive`` alone, and the machine description lives in the static product
of the shot's machine version.  The canonical source is their union:

* diagnostics and eddy through :func:`vaft.database.composition.compose_stage_products`,
  which also checks that the passive currents were solved on this
  diagnostics product's ``pf_active.time``;
* ``wall`` and ``em_coupling`` from the static product;
* ``equilibrium`` (with its constraints) from the EFIT product.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import vaft
from vaft.database.composition import compose_stage_products

#: The IDSs the canonical source takes from the static product.
STATIC_IDS = ("wall", "em_coupling")


def compose(
    *,
    static_stage: Path,
    diagnostics_stage: Path,
    eddy_stage: Path,
    efit_stage: Path,
    shot: int,
    eddy_manifest: Path | None = None,
):
    canonical, _ = compose_stage_products(
        diagnostics=diagnostics_stage, eddy=eddy_stage, eddy_manifest=eddy_manifest
    )
    static = vaft.omas.load(static_stage)
    for ids_name in STATIC_IDS:
        if ids_name not in static:
            raise ValueError(f"Static stage has no {ids_name} IDS: {static_stage}")
        canonical[ids_name] = static[ids_name]
    efit = vaft.omas.load(efit_stage)
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
    parser.add_argument("--static-stage", required=True, type=Path)
    parser.add_argument("--diagnostics-stage", required=True, type=Path)
    parser.add_argument("--eddy-stage", required=True, type=Path)
    parser.add_argument("--eddy-manifest", type=Path)
    parser.add_argument("--efit-stage", required=True, type=Path)
    parser.add_argument("--shot", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    canonical = compose(
        static_stage=args.static_stage.resolve(),
        diagnostics_stage=args.diagnostics_stage.resolve(),
        eddy_stage=args.eddy_stage.resolve(),
        efit_stage=args.efit_stage.resolve(),
        shot=args.shot,
        eddy_manifest=args.eddy_manifest.resolve() if args.eddy_manifest else None,
    )
    vaft.omas.save(canonical, args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
