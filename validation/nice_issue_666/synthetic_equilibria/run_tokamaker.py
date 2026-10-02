"""Issue #666: feed NICE noiseless diagnostics from the local TokaMaker solve."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon

from vaft.formula.constants import MU0
from vaft.process.equilibrium import as_equilibrium

from validation.nice_issue_666.synthetic_equilibria.run_solovev import (
    BASE, ROOT, plasma_response, run_case,
)

GFILE = Path("/tmp/tokamaker-41672-331-20260914/g041672.00331")


def tokamaker_filaments():
    eq = as_equilibrium(GFILE, convention=2)
    rr, zz = np.meshgrid(eq.r, eq.z, indexing="ij")
    inside = Polygon(np.column_stack((eq.lcfs.r, eq.lcfs.z))).contains_points(
        np.column_stack((rr.ravel(), zz.ravel()))
    ).reshape(rr.shape)
    pprime = np.interp(eq.psi, eq.psi_1d, eq.pprime)
    ffprime = np.interp(eq.psi, eq.psi_1d, eq.ffprime)
    jphi = rr * pprime + ffprime / (MU0 * rr)
    # Trapezoidal nodal area, then normalize away g-file interpolation error.
    dr, dz = float(eq.r[1] - eq.r[0]), float(eq.z[1] - eq.z[0])
    raw = jphi[inside] * dr * dz
    currents = raw * (float(eq.ip) / raw.sum())
    return rr[inside], zz[inside], currents, {
        "gfile": str(GFILE), "gfile_cocos": 2,
        "axis_m": list(eq.magnetic_axis), "ip_A": float(eq.ip),
        "r_geo_m": float((np.max(eq.lcfs.r) + np.min(eq.lcfs.r)) / 2),
        "a_geo_m": float((np.max(eq.lcfs.r) - np.min(eq.lcfs.r)) / 2),
        "z_extent_m": [float(np.min(eq.lcfs.z)), float(np.max(eq.lcfs.z))],
        "quadrature_filaments": int(inside.sum()),
        "raw_integrated_current_A": float(raw.sum()),
        "normalized_current_A": float(currents.sum()),
    }


def main():
    manifest = json.loads((BASE / "nice_case_manifest.json").read_text())
    channels = [c for c in manifest["diagnostic_channels"]
                if c["enabled"] and c["family"] in ("bpol_probe", "flux_loop")]
    src_r, src_z, currents, truth = tokamaker_filaments()
    plasma = plasma_response(channels, src_r, src_z, currents)
    native_active = np.asarray([
        next(a["native"] for a in manifest["active_response_audit"] if a["ods_path"] == c["ods_path"])
        for c in channels
    ])
    measurements = plasma + native_active
    results = [
        run_case("tokamaker_vacth_only", {"algoVacTHonly": 1}, measurements, channels),
        run_case("tokamaker_full_default", {}, measurements, channels),
        run_case("tokamaker_direct_10", {"iterMaxDirInitRecon": 10}, measurements, channels),
    ]
    payload = {
        "description": "Noiseless local TokaMaker plasma plus NICE-native active-coil response",
        "truth": truth,
        "plasma_signal_norms": {
            family: float(np.linalg.norm([y for c, y in zip(channels, plasma) if c["family"] == family]))
            for family in ("bpol_probe", "flux_loop")
        },
        "results": results,
    }
    (ROOT / "tokamaker_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
