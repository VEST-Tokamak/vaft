#!/usr/bin/env python3
"""Generate VEST eddy-current OMAS ODS from diagnostics ODS."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
from omas import load_omas_json, save_omas_json

from vaft.machine_mapping.em_coupling import em_coupling
from vaft.machine_mapping.pf_passive import pf_passive

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="vaft-mpl-"))

from vaft.omas.process_wrapper import compute_eddy_currents


LOGGER = logging.getLogger("vaft.generate_eddy_ods")


def _csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def build_eddy_ods(
    *,
    diagnostics_ods: Path,
    reference_ods: Path,
    filament_r: list[float],
    filament_z: list[float],
    filament_fraction: list[float],
    dt_sub: float,
):
    if not diagnostics_ods.exists():
        raise FileNotFoundError(f"Diagnostics ODS not found: {diagnostics_ods}")
    if len(filament_r) != len(filament_z) or len(filament_r) != len(filament_fraction):
        raise ValueError("Filament r, z, and fraction lists must have the same length")

    ods = load_omas_json(str(diagnostics_ods), consistency_check=False)
    pf_passive(ods, reference_ods)
    em_coupling(ods, reference_ods)

    pf_time = np.asarray(ods["pf_active.time"], dtype=float)
    ip_time = np.asarray(ods["magnetics.ip.0.time"], dtype=float)
    ip_data = np.asarray(ods["magnetics.ip.0.data"], dtype=float)
    ip_on_pf_time = np.interp(pf_time, ip_time, ip_data)

    plasma = list(zip(filament_r, filament_z))
    plasma_currents = [ip_on_pf_time * fraction for fraction in filament_fraction]
    compute_eddy_currents(ods, plasma, plasma_currents, dt_sub=dt_sub)
    return ods


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--diagnostics-ods", "--input", required=True, type=Path, help="Input diagnostics ODS JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Output eddy ODS JSON.")
    parser.add_argument("--reference-ods", required=True, type=Path, help="Reference ODS containing wall/coupling data.")
    parser.add_argument("--filament-r", default="0.35,0.35,0.35", help="Comma-separated plasma filament R positions.")
    parser.add_argument("--filament-z", default="0.25,0.0,-0.25", help="Comma-separated plasma filament Z positions.")
    parser.add_argument("--filament-fraction", default="0.3333333,0.3333333,0.3333333", help="Comma-separated plasma current fractions.")
    parser.add_argument("--dt-sub", default=5e-5, type=float, help="Sub-step for eddy-current integration.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    LOGGER.info("Generating eddy ODS for shot %s from %s", args.shot, args.diagnostics_ods)
    ods = build_eddy_ods(
        diagnostics_ods=args.diagnostics_ods,
        reference_ods=args.reference_ods,
        filament_r=_csv_floats(args.filament_r),
        filament_z=_csv_floats(args.filament_z),
        filament_fraction=_csv_floats(args.filament_fraction),
        dt_sub=args.dt_sub,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_omas_json(ods, str(args.output))
    LOGGER.info("Eddy ODS saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
