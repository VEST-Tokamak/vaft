"""Guazzotto-Freidberg Part 2: pedestals, surface currents and toroidal flow (#1149).

L. Guazzotto and J. P. Freidberg, J. Plasma Phys. 87, 905870305 (2021).
Table 2 publishes beta_P and q* at q0 = 1 for four variations of Part 1's
spherical double null (eps 0.75, kappa_X 2.4, delta_X 0.8, nu 1.04):

* A: current pedestal f_J = 0.25;
* B: A plus pressure pedestal f_P = 0.2 and edge bootstrap fraction f_B = 0.35;
* C: B plus adiabatic (gamma = 2) flow at M0 = 0.4;
* D: B plus incompressible (gamma = inf) flow at M0 = 0.4.

The implementation reproduces beta_P to 0.5 % and q* to 1.1 %; the
Part 1 row it starts from is itself 0.2 % low in q*.  Two readings are
pinned:

* Cases C and D keep nu = 1.04, above the flow-reduced nu_max = -1/G(-1) =
  1.025 of Eq. (5.7): the inboard edge current reverses slightly.  The paper
  runs them anyway, so reproducing them needs ``allow_current_reversal``.
* The surface-current jump uses Eq. (3.9), whose toroidal-field term carries
  R0**2/R**2; Eq. (6.2) prints it without.  The two differ by 0.2 % in q*.
"""

from __future__ import annotations

import numpy as np
import pytest

#: Each pedestal case scans alpha twice (the eigenvalue, then the pedestal root).
pytestmark = pytest.mark.slow

from vaft.process.equilibrium import (  # noqa: E402
    evaluate_guazzotto_freidberg,
    guazzotto_freidberg_parameters,
    guazzotto_freidberg_to_equilibrium,
    solve_guazzotto_freidberg,
)

SPHERICAL = dict(inverse_aspect_ratio=0.75, nu=1.04, x_point_elongation=2.4, x_point_triangularity=0.8)
PEDESTALS = dict(current_pedestal=0.25, pressure_pedestal=0.2, bootstrap_fraction=0.35)
CASES = {
    "A": (dict(current_pedestal=0.25), (0.873, 4.00)),
    "B": (PEDESTALS, (1.60, 2.62)),
    "C": (dict(PEDESTALS, mach_number=0.4, adiabatic_index=2.0, allow_current_reversal=True), (1.59, 3.18)),
    "D": (dict(PEDESTALS, mach_number=0.4, adiabatic_index=np.inf, allow_current_reversal=True), (1.60, 2.89)),
}


@pytest.fixture(scope="module")
def solved():
    return {name: solve_guazzotto_freidberg("double_null", **SPHERICAL, **extra)
            for name, (extra, _) in CASES.items()}


@pytest.fixture(scope="module")
def reference():
    return solve_guazzotto_freidberg("double_null", **SPHERICAL)


@pytest.mark.parametrize("name", sorted(CASES))
def test_table_2_is_reproduced(solved, name):
    beta_p, q_star = CASES[name][1]
    got = guazzotto_freidberg_parameters(solved[name], q0=1.0)
    assert got["beta_p"] == pytest.approx(beta_p, rel=0.01), (name, got["beta_p"])
    assert got["q_star"] == pytest.approx(q_star, rel=0.015), (name, got["q_star"])


def test_flow_cases_exceed_the_flow_reduced_nu_max(solved):
    assert solved["C"].status == solved["D"].status == "current_reversal"
    assert solved["A"].status == "converged"
    with pytest.raises(ValueError, match="Part 2, Eq. 5.7"):
        solve_guazzotto_freidberg("double_null", **SPHERICAL, mach_number=0.4, adiabatic_index=2.0)


def test_all_part_2_inputs_at_zero_are_part_1(reference):
    zero = solve_guazzotto_freidberg("double_null", **SPHERICAL, current_pedestal=0.0, pressure_pedestal=0.0,
                                     bootstrap_fraction=0.0, mach_number=0.0)
    assert zero.alpha == reference.alpha
    np.testing.assert_array_equal(zero.coefficients, reference.coefficients)
    par = guazzotto_freidberg_parameters(reference)
    assert par["surface_field_ratio"] == 1.0 and par["total_current_ratio"] == 1.0


def test_pedestal_eigenvalue_approaches_part_1_as_the_pedestal_vanishes(reference):
    alphas = [solve_guazzotto_freidberg("double_null", **SPHERICAL, current_pedestal=f).alpha
              for f in (0.2, 0.05, 0.005)]
    assert alphas[0] < alphas[1] < alphas[2] < reference.alpha
    assert reference.alpha - alphas[2] < 0.02


@pytest.mark.parametrize("name", sorted(CASES))
def test_grad_shafranov_and_the_inhomogeneous_conditions_hold(solved, name):
    model = solved[name]
    x, y = np.meshgrid(np.linspace(-0.95, 0.95, 9), np.linspace(-2.0, 2.0, 9))
    values = evaluate_guazzotto_freidberg(model, x, y)
    assert np.max(np.abs(values["gs_residual"])) < 1e-10*model.alpha**2
    # Eqs. (4.8)-(4.9): psi = 0 (psi_J = f_J/(1 - f_J)) on the surface, psi = 1 on the axis.
    assert evaluate_guazzotto_freidberg(model, *model.magnetic_axis)["psi"] == pytest.approx(1.0, abs=1e-9)
    for x0 in (-1.0, 1.0):
        assert abs(float(evaluate_guazzotto_freidberg(model, x0, 0.0)["psi"])) < 1e-9
    f_j = model.current_pedestal
    assert float(evaluate_guazzotto_freidberg(model, 1.0, 0.0)["psi_j"]) == pytest.approx(f_j/(1 - f_j))


def test_flow_shifts_the_axis_outward(solved):
    assert solved["C"].magnetic_axis[0] > solved["D"].magnetic_axis[0] > solved["B"].magnetic_axis[0]


@pytest.mark.parametrize("name", ["A", "C"])
def test_amperes_law_closes_on_the_surface(solved, name):
    par = guazzotto_freidberg_parameters(solved[name])
    assert par["core_current_line_integral"] == pytest.approx(par["core_current"], rel=5e-3)


def test_bootstrap_fraction_sets_the_exterior_current(solved):
    # Surface currents change q* only through Delta_B and I_hat/I (Eq. 6.11).
    a = guazzotto_freidberg_parameters(solved["A"])
    b = guazzotto_freidberg_parameters(solved["B"])
    assert b["total_current_ratio"] == pytest.approx(1/(1 - 0.35))
    assert b["q_star"]/a["q_star"] == pytest.approx(b["surface_field_ratio"]*(1 - 0.35), rel=1e-6)
    assert a["surface_field_ratio"] == 1.0


def test_pedestal_export_carries_the_edge_and_the_current():
    from matplotlib.path import Path
    from scipy.constants import mu_0

    model = solve_guazzotto_freidberg("limited", inverse_aspect_ratio=0.33, nu=1.0, elongation=1.6,
                                      triangularity=0.3, current_pedestal=0.2, pressure_pedestal=0.3)
    eq = guazzotto_freidberg_to_equilibrium(model, major_radius=1.0, toroidal_field=2.0, resolution=201)
    p0 = eq.metadata["p0"]
    assert eq.pressure[0] == pytest.approx(p0) and eq.pressure[-1] == pytest.approx(0.3*p0)
    assert eq.ffprime[-1] != 0.0 and eq.pprime[-1] != 0.0                # the edge current pedestal
    rm, zm = np.meshgrid(eq.r, eq.z, indexing="ij")
    psi_n = (eq.psi - eq.psi_axis)/(eq.psi_boundary - eq.psi_axis)
    order = np.argsort((eq.psi_1d - eq.psi_axis)/(eq.psi_boundary - eq.psi_axis))
    x1d = ((eq.psi_1d - eq.psi_axis)/(eq.psi_boundary - eq.psi_axis))[order]
    pp = np.interp(psi_n, x1d, eq.pprime[order]); ff = np.interp(psi_n, x1d, eq.ffprime[order])
    j = -2*np.pi*(rm*pp + ff/(mu_0*rm))
    inside = Path(eq.lcfs.points).contains_points(np.c_[rm.ravel(), zm.ravel()]).reshape(rm.shape)
    ip = np.sum(j*inside)*(eq.r[1] - eq.r[0])*(eq.z[1] - eq.z[0])
    assert ip == pytest.approx(eq.ip, rel=0.02)


def test_flow_export_is_refused(solved):
    with pytest.raises(ValueError, match="not a flux function"):
        guazzotto_freidberg_to_equilibrium(solved["C"], major_radius=1.0, toroidal_field=1.0)


@pytest.mark.parametrize("kwargs, match", [
    (dict(current_pedestal=1.0), "current_pedestal"),
    (dict(bootstrap_fraction=-0.1), "bootstrap_fraction"),
    (dict(mach_number=0.3), "adiabatic_index"),
    (dict(mach_number=0.3, adiabatic_index=3.0), "adiabatic_index"),
    (dict(mach_number=-0.1), "mach_number"),
])
def test_invalid_part_2_inputs_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        solve_guazzotto_freidberg("limited", inverse_aspect_ratio=0.33, nu=0.5, elongation=1.6, triangularity=0.3,
                                  **kwargs)
