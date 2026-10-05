"""Public-API OFT fixed-boundary PF fit benchmark (#1608, stage 2).

Run one coarse case locally; run multiple resolutions/topologies on a server.
Every fixed solve receives a new isolated working directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omas import ODS

from vaft.code.tokamaker import TokaMakerConfig, fit_free_boundary_coils_vfixed
from vaft.machine_mapping.pf_active import vfit_pf_active_static
from vaft.process.equilibrium import (
    guazzotto_freidberg_to_equilibrium,
    solovev_example,
    solve_guazzotto_freidberg,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--resolutions", type=float, nargs="+", default=[.05])
    parser.add_argument("--families", nargs="+", default=["solovev", "guazzotto_freidberg"])
    parser.add_argument("--topologies", nargs="+", default=["limited"])
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=False)
    machine = ODS(consistency_check=False)
    vfit_pf_active_static(machine)
    records = []
    for family in args.families:
        for topology in args.topologies:
            if family == "solovev":
                eq = solovev_example("single_null" if topology == "lower_single_null" else topology,
                                     resolution=33)
            elif family == "guazzotto_freidberg":
                options = dict(inverse_aspect_ratio=.33, nu=1., elongation=1.6, triangularity=.3)
                if topology != "limited":
                    options.update(x_point_elongation=2., x_point_triangularity=.5)
                eq = guazzotto_freidberg_to_equilibrium(
                    solve_guazzotto_freidberg(topology, **options), major_radius=.4,
                    toroidal_field=.1, resolution=33)
            else:
                raise ValueError(f"unknown family {family!r}")
            for dx in args.resolutions:
                case = args.workdir / f"{family}_{topology}_dx{dx:g}"
                fit = fit_free_boundary_coils_vfixed(
                    eq, machine, case, config=TokaMakerConfig(dx_plasma=dx, order=2),
                    regularization=1e-5)
                record = {"family": family, "topology": topology, "dx_plasma_m": dx,
                          "profile_mode": "power_law", "target_Ip_A": eq.ip,
                          "fixed_Ip_A": float(fit.fixed_stats["Ip"]),
                          "samples": len(fit.boundary_points_m), "currents_A": fit.currents_A,
                          "rms_relative_flux": fit.rms_relative_flux,
                          "max_relative_flux": fit.max_relative_flux,
                          "rank": fit.rank, "condition_number": fit.condition_number,
                          "regularization_norm": fit.regularization_norm,
                          "bounds_complete": fit.bounds_complete, "status": fit.status}
                records.append(record)
                (args.workdir / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
