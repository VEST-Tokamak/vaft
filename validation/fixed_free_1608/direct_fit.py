"""Reproduce the small direct PF-current fits for issue #1608.

Run from the repository root with
``PYTHONPATH=. python validation/fixed_free_1608/direct_fit.py``.
No TokaMaker solve or measured discharge is required.
"""

from __future__ import annotations

import json

from omas import ODS

from vaft.machine_mapping.pf_active import vfit_pf_active_static
from vaft.process.equilibrium import (
    find_stationary_points,
    fit_free_boundary_coils,
    guazzotto_freidberg_to_equilibrium,
    solovev_example,
    solve_guazzotto_freidberg,
)


def main() -> None:
    machine = ODS(consistency_check=False)
    vfit_pf_active_static(machine)
    records = []
    for family in ("solovev", "guazzotto_freidberg"):
        for topology in ("limited", "lower_single_null", "double_null"):
            if family == "solovev":
                eq = solovev_example("single_null" if topology == "lower_single_null" else topology,
                                     resolution=33)
                x_points = eq.metadata["x_points_requested"]
            else:
                kwargs = dict(inverse_aspect_ratio=.33, nu=1.0,
                              elongation=1.6, triangularity=.3)
                if topology != "limited":
                    kwargs.update(x_point_elongation=2.0, x_point_triangularity=.5)
                model = solve_guazzotto_freidberg(topology, **kwargs)
                eq = guazzotto_freidberg_to_equilibrium(
                    model, major_radius=.4, toroidal_field=.1, resolution=33)
                x_points = tuple((p.r, p.z) for p in find_stationary_points(eq, kind="x")
                                 if abs(p.psi_n - 1.0) < .02)
            fit = fit_free_boundary_coils(eq, machine, method="flux_normal",
                                          boundary_samples=40, x_points=x_points,
                                          regularization=1e-5)
            records.append({
                "family": family, "topology": topology,
                "target_ip_A": eq.ip, "integrated_ip_A": fit.integrated_ip_A,
                "currents_A": fit.currents_A,
                "rms_relative_flux": fit.rms_relative_flux,
                "max_relative_flux": fit.max_relative_flux,
                "rms_normal_field_T": fit.rms_normal_field_T,
                "max_saddle_field_T": fit.max_saddle_field_T,
                "rank": fit.rank, "condition_number": fit.condition_number,
                "regularization_norm": fit.regularization_norm,
                "active_bounds": fit.active_bounds,
                "bounds_complete": fit.bounds_complete, "status": fit.status,
            })
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
