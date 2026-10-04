"""Small solver-free checks of the fixed-boundary PF inverse problem (#1608)."""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace
from omas import ODS

from vaft.process.equilibrium import (
    convert_cocos,
    find_stationary_points,
    fit_free_boundary_coils,
    guazzotto_freidberg_to_equilibrium,
    solovev_example,
    solve_guazzotto_freidberg,
)


def _machine() -> ODS:
    ods = ODS(consistency_check=False)
    positions = ((.12, .50), (.12, -.50), (.75, .50), (.75, -.50),
                 (.80, 0), (.13, 0), (.40, .65), (.40, -.65))
    for i, (r, z) in enumerate(positions):
        base = f"pf_active.coil.{i}"
        ods[f"{base}.name"] = f"PF{i + 1}"
        ods[f"{base}.element.0.geometry.rectangle.r"] = r
        ods[f"{base}.element.0.geometry.rectangle.z"] = z
        ods[f"{base}.element.0.turns_with_sign"] = 10.0
    return ods


def _guazzotto(topology: str):
    options = dict(inverse_aspect_ratio=.33, nu=1.0, elongation=1.6, triangularity=.3)
    if topology != "limited":
        options.update(x_point_elongation=2.0, x_point_triangularity=.5)
    model = solve_guazzotto_freidberg(topology, **options)
    return guazzotto_freidberg_to_equilibrium(
        model, major_radius=.4, toroidal_field=.1, resolution=33,
    )


def _x_points(eq):
    return tuple((p.r, p.z) for p in find_stationary_points(eq, kind="x")
                 if abs(p.psi_n - 1.0) < .02)


@pytest.mark.parametrize("topology", ("limited", "single_null", "double_null"))
def test_solovev_direct_fit(topology):
    eq = solovev_example(topology, resolution=33)
    xp = eq.metadata["x_points_requested"]
    fit = fit_free_boundary_coils(eq, _machine(), method="flux_normal",
                                  boundary_samples=32, x_points=xp, regularization=1e-5)
    assert fit.optimizer_success
    assert fit.rank == len(fit.coil_names)
    assert abs(fit.integrated_ip_A / eq.ip - 1) < .03
    assert fit.rms_relative_flux < .06
    assert fit.rms_normal_field_T < .01
    assert fit.max_saddle_field_T is None or fit.max_saddle_field_T < .02
    assert fit.plasma_psi_Wb.shape == fit.coil_psi_Wb.shape
    assert len(fit.x_points_m) == len(xp)


@pytest.mark.parametrize("topology", ("limited", "lower_single_null", "double_null"))
def test_guazzotto_part1_direct_fit(topology):
    eq = _guazzotto(topology)
    xp = _x_points(eq)
    assert len(xp) == (0 if topology == "limited" else 2 if topology == "double_null" else 1)
    fit = fit_free_boundary_coils(eq, _machine(), method="flux_normal",
                                  boundary_samples=32, x_points=xp, regularization=1e-5)
    assert fit.optimizer_success
    assert abs(fit.integrated_ip_A / eq.ip - 1) < .03
    assert fit.rms_relative_flux < .03
    assert fit.rms_normal_field_T < .01
    assert fit.max_saddle_field_T is None or fit.max_saddle_field_T < .01


def test_flux_gauge_cocos_and_bounds():
    eq = solovev_example(resolution=33)
    machine = _machine()
    reference = fit_free_boundary_coils(eq, machine, boundary_samples=32)
    converted = fit_free_boundary_coils(convert_cocos(eq, 1), machine, boundary_samples=32)
    np.testing.assert_allclose(list(converted.currents_A.values()),
                               list(reference.currents_A.values()), rtol=1e-9)
    offset = .013
    shifted_eq = replace(eq, psi=eq.psi + offset, psi_1d=eq.psi_1d + offset,
                         psi_axis=eq.psi_axis + offset, psi_boundary=eq.psi_boundary + offset)
    shifted = fit_free_boundary_coils(shifted_eq, machine, boundary_samples=32)
    np.testing.assert_allclose(shifted.residuals["relative_flux_Wb"],
                               reference.residuals["relative_flux_Wb"], atol=1e-13)
    np.testing.assert_allclose(list(shifted.currents_A.values()), list(reference.currents_A.values()), rtol=1e-9)
    bounded = fit_free_boundary_coils(eq, machine, boundary_samples=32,
                                      current_bounds={"PF1": (-10.0, 10.0)},
                                      regularization=1e-5,
                                      current_penalties={"PF2": 2.0})
    assert abs(bounded.currents_A["PF1"]) <= 10.0 + 1e-6
    assert "PF1" in bounded.active_bounds
    assert bounded.regularization_norm > 0


def test_bad_inputs_fail_before_the_inverse_solve():
    eq = solovev_example(resolution=33)
    machine = _machine()
    with pytest.raises(ValueError, match="method"):
        fit_free_boundary_coils(eq, machine, method="unknown")
    with pytest.raises(ValueError, match="unknown current bounds"):
        fit_free_boundary_coils(eq, machine, current_bounds={"missing": (-1, 1)})
    with pytest.raises(ValueError, match="regularization"):
        fit_free_boundary_coils(eq, machine, regularization=-1)


def test_rank_deficiency_is_reported_not_called_accepted():
    eq = solovev_example(resolution=33)
    machine = _machine()
    for field in ("geometry.rectangle.r", "geometry.rectangle.z", "turns_with_sign"):
        machine[f"pf_active.coil.8.element.0.{field}"] = machine[f"pf_active.coil.0.element.0.{field}"]
    machine["pf_active.coil.8.name"] = "PF9"
    fit = fit_free_boundary_coils(eq, machine, boundary_samples=32)
    assert fit.optimizer_success
    assert fit.rank < len(fit.coil_names)
    assert not fit.accepted


def test_guazzotto_surface_current_cannot_be_silently_lost():
    eq = _guazzotto("limited")
    model = replace(eq.metadata["model"], pressure_pedestal=.2)
    unsupported = replace(eq, metadata={**eq.metadata, "model": model})
    with pytest.raises(ValueError, match="surface-current or flow"):
        fit_free_boundary_coils(unsupported, _machine())
