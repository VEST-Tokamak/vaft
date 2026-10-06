#!/usr/bin/env python3
"""Integrated pressure and profile shape of every ``weight_scan`` member against Thomson (#579).

    python workflow/efit_uncertainty_calibration/sensitivity_pressure.py \\
        --tables table_42962.json ... --scan-dirs shot_42962 ... \\
        --core-profiles <dir with <shot>/output/core_profiles.json.gz> --output pressure.json

``criteria.thomson`` compares pressures *at the Thomson channels*
(``ln(sum p_e / sum p_recon)``).  That point sum mixes two things: how much
pressure there is, and where it sits relative to the channels.  This script
separates them, for each member's own g-file:

* **amount** -- ``R_V = int p_EFIT dV / int p_e dV`` over the whole plasma, with
  ``p_e`` from the fitted ``core_profiles`` electron profile
  (``vaft.validation.kinetic_state.integrated_pressure_ratio``).  The physical
  band is ``1 <= R_V <= 2`` (no fast ions, T_i <= T_e, n_i <= n_e).
* **shape** -- the peaking ``p(0) / <p>_V`` of the EFIT pressure and of the
  fitted ``p_e``, and the RMS difference of the two profiles each normalised
  to its own volume average, on rho_tor_norm in [0, 1].

The shape is what tells the profile bases apart; the amount is what the
consistency band is about.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _load_gz(path: Path):
    from omas import load_omas_json

    with gzip.open(path, "rt", encoding="utf-8") as handle, \
            tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as staged:
        staged.write(handle.read())
        name = staged.name
    try:
        return load_omas_json(name, consistency_check=False)
    finally:
        Path(name).unlink(missing_ok=True)


def member_pressure(gfile: Path, rho: np.ndarray, p_e: np.ndarray) -> dict[str, Any]:
    from vaft.data import read_geqdsk
    from vaft.validation.kinetic_state import integrated_pressure_ratio, rho_tor_norm_of

    ods = read_geqdsk(str(gfile)).to_omas(ods=None, time_index=0)
    full = integrated_pressure_ratio(ods, 0, rho, p_e)
    if not full.get("available"):
        return {"available": False, "reason": full.get("reason")}
    coordinate = rho_tor_norm_of(ods, 0)
    pressure = np.asarray(ods["equilibrium.time_slice.0.profiles_1d.pressure"], float)
    grid_rho = np.asarray(coordinate["rho_tor_norm"], float)
    volume = full["volume_m3"]
    mean_efit = full["w_efit_j"] / 1.5 / volume
    mean_e = full["w_e_j"] / 1.5 / volume
    order, by_rho = np.argsort(grid_rho), np.argsort(rho)
    axis = np.linspace(0.0, 1.0, 51)
    efit_on_axis = np.interp(axis, grid_rho[order], pressure[order])
    e_on_axis = np.interp(axis, rho[by_rho], p_e[by_rho])
    shape_rms = (float(np.sqrt(np.mean((efit_on_axis / mean_efit - e_on_axis / mean_e) ** 2)))
                 if mean_efit > 0 and mean_e > 0 else math.nan)
    return {
        "available": True,
        "r_v": full["ratio"], "w_efit_j": full["w_efit_j"], "w_e_j": full["w_e_j"],
        "peaking_efit": float(efit_on_axis[0] / mean_efit) if mean_efit > 0 else math.nan,
        "peaking_e": float(e_on_axis[0] / mean_e) if mean_e > 0 else math.nan,
        "shape_rms": shape_rms,
    }


def main(argv: Sequence[str] | None = None) -> int:
    from vaft.validation.kinetic_state import core_profiles_electron_pressure

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, nargs="+", required=True)
    parser.add_argument("--scan-dirs", type=Path, nargs="+", required=True,
                        help="the weight_scan --output directories, in the order of --tables")
    parser.add_argument("--core-profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rows, profiles = [], {}
    for table, scan in zip(args.tables, args.scan_dirs):
        for record in json.loads(table.read_text(encoding="utf-8"))["records"]:
            if "error" in record:
                continue
            shot, time_ms, setting = int(record["shot"]), int(record["time_ms"]), record["setting"]
            if shot not in profiles:
                path = args.core_profiles / str(shot) / "output" / "core_profiles.json.gz"
                profiles[shot] = _load_gz(path) if path.is_file() else None
            row = {"shot": shot, "time_ms": time_ms, "setting": setting}
            if profiles[shot] is None:
                rows.append({**row, "available": False, "reason": "no core_profiles product"})
                continue
            electron = core_profiles_electron_pressure(profiles[shot], time_s=time_ms / 1e3)
            gfiles = sorted((scan / f"shot_{shot}" / f"t{time_ms * 1000:07d}" / setting).glob("g0*"))
            if not electron["available"] or not gfiles:
                rows.append({**row, "available": False,
                             "reason": electron.get("reason") or "no g-file (EFIT produced none)"})
                continue
            try:
                result = member_pressure(gfiles[0], np.asarray(electron["rho_tor_norm"], float),
                                         np.asarray(electron["p_e"], float))
            except Exception as error:  # a malformed g-file is a member result, not a crash
                result = {"available": False, "reason": repr(error)}
            rows.append({**row, **result})
    args.output.write_text(json.dumps({"rows": rows}, indent=1, allow_nan=True) + "\n", encoding="utf-8")
    print(f"{len(rows)} members, {sum(r['available'] for r in rows)} with an integrated ratio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
