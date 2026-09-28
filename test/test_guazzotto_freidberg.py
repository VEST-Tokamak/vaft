"""Guazzotto-Freidberg Part 1 analytic equilibria against the published paper (#1148).

L. Guazzotto and J. P. Freidberg, J. Plasma Phys. 87, 905870303 (2021).
Table 4 publishes, for nine configurations at q0 = 1, the eigenvalue alpha and
beta_T, beta_P, l_i, q95, q*. The implementation reproduces them to the
printed precision with three documented exceptions, each pinned here so a
change of reading shows up:

* "Elongated D": Table 1 and Fig. 3 give delta = 0.4, but the whole Table 4
  row (alpha 1.96, beta_T 0.0084, beta_P 0.27, l_i 0.85, q95 3.84, q* 2.90) is
  reproduced by delta = 0.6; delta = 0.4 gives alpha = 1.9206.
* "Inverse D": every column but l_i matches; the paper prints 0.97, the
  solution gives 0.866.
* Divertor q*: the paper's "kappa95" is not defined; with the elongation of
  the psi = 0.05 surface q* is 5-12 % above the table while every other
  column of those rows matches.
"""

from __future__ import annotations

import numpy as np
import pytest

#: Nine eigenvalue scans and their plasma integrals take a few minutes.
pytestmark = pytest.mark.slow

from vaft.process.equilibrium import (  # noqa: E402
    evaluate_guazzotto_freidberg,
    find_stationary_points,
    guazzotto_freidberg_parameters,
    guazzotto_freidberg_to_equilibrium,
    solve_guazzotto_freidberg,
)

LIMITED = "limited"; DN = "double_null"; SN = "lower_single_null"

#: (topology, eps, kappa, delta, kappa_X, delta_X, nu) and Table 4's (beta_T, beta_P, l_i, q95, q*, alpha).
TABLE = {
    "circle": ((LIMITED, 0.33, 1.0, 0.0, None, None, 1.0), (0.014, 1, 1.04, 3.21, 2.88, 2.38)),
    "ellipse": ((LIMITED, 0.25, 2.0, 0.0, None, None, 1.0), (0.018, 1, 0.85, 3.32, 3.04, 1.88)),
    "elongated_d_as_computed": ((LIMITED, 0.33, 1.8, 0.6, None, None, 0.3), (0.0084, 0.27, 0.85, 3.84, 2.90, 1.96)),
    "high_triangularity_d": ((LIMITED, 0.4, 2.0, 0.75, None, None, 1.0), (0.024, 1, 0.91, 6.82, 4.71, 2.05)),
    "inverse_d": ((LIMITED, 0.33, 1.9, -0.6, None, None, 0.5), (0.014, 0.49, 0.97, 3.12, 3.08, 1.91)),
    "double_null": ((DN, 0.33, None, None, 2.0, 0.5, 0.4), (0.011, 0.38, 0.86, 3.74, 2.65, 1.96)),
    "double_null_spherical": ((DN, 0.75, None, None, 2.4, 0.8, 1.04), (0.041, 1.05, 0.97, 18.2, 5.57, 1.90)),
    "single_null": ((SN, 0.33, 1.6, 0.4, 2.0, 0.5, 1.0), (0.020, 1, 0.93, 4.45, 3.05, 2.04)),
    "single_null_high_beta": ((SN, 0.25, 1.8, 0.6, 2.1, 0.8, 2.1), (0.018, 2.16, 1.00, 6.17, 3.71, 2.09)),
}
#: Published values this implementation does not reproduce, and why (see the module docstring).
KNOWN_TABLE_DISCREPANCIES = {("inverse_d", "li"), ("double_null", "q_star"), ("double_null_spherical", "q_star"),
                             ("single_null", "q_star"), ("single_null_high_beta", "q_star")}


def _solve(topology, eps, kappa, delta, kappa_x, delta_x, nu):
    return solve_guazzotto_freidberg(topology, inverse_aspect_ratio=eps, nu=nu, elongation=kappa,
                                     triangularity=delta, x_point_elongation=kappa_x, x_point_triangularity=delta_x)


@pytest.fixture(scope="module")
def solved():
    return {name: _solve(*inputs) for name, (inputs, _) in TABLE.items()}


@pytest.mark.parametrize("name", sorted(TABLE))
def test_table_4_is_reproduced(solved, name):
    published = dict(zip(("beta_t", "beta_p", "li", "q95", "q_star", "alpha"), TABLE[name][1]))
    got = guazzotto_freidberg_parameters(solved[name], q0=1.0)
    for key, value in published.items():
        if (name, key) in KNOWN_TABLE_DISCREPANCIES:
            continue
        # The table prints two or three significant figures.
        assert got[key] == pytest.approx(value, rel=0.035, abs=6e-4), (name, key, got[key], value)


def test_the_elongated_d_inputs_of_table_1_give_a_different_row():
    model = _solve(LIMITED, 0.33, 1.8, 0.4, None, None, 0.3)
    assert model.alpha == pytest.approx(1.9206, abs=1e-4)
    assert guazzotto_freidberg_parameters(model)["q95"] == pytest.approx(3.44, abs=0.02)   # not the table's 3.84


def test_known_discrepancies_stay_where_they_are(solved):
    assert guazzotto_freidberg_parameters(solved["inverse_d"])["li"] == pytest.approx(0.866, abs=0.005)
    for name, (_, published) in TABLE.items():
        if (name, "q_star") in KNOWN_TABLE_DISCREPANCIES:
            ratio = guazzotto_freidberg_parameters(solved[name])["q_star"]/published[4]
            assert 1.04 < ratio < 1.15, (name, ratio)


@pytest.mark.parametrize("name", sorted(TABLE))
def test_grad_shafranov_is_satisfied_to_round_off(solved, name):
    model = solved[name]
    x, y = np.meshgrid(np.linspace(-0.95, 0.95, 9), np.linspace(-1.0, 1.0, 9))
    values = evaluate_guazzotto_freidberg(model, x, y)
    assert np.max(np.abs(values["gs_residual"])) < 1e-10*model.alpha**2
    assert evaluate_guazzotto_freidberg(model, *model.magnetic_axis)["psi"] == pytest.approx(1.0)


@pytest.mark.parametrize("name", sorted(TABLE))
def test_matching_conditions_hold(solved, name):
    model = solved[name]
    for x0 in (-1.0, 1.0):                                             # inner and outer midplane on the surface
        assert abs(float(evaluate_guazzotto_freidberg(model, x0, 0.0)["psi"])) < 1e-8
    if model.topology != LIMITED:
        eps, dx = model.inverse_aspect_ratio, model.x_point_triangularity
        x_x = dx + eps/2*(1 - dx**2)
        heights = (model.x_point_elongation, -model.x_point_elongation) if model.topology == DN else (-model.x_point_elongation,)
        for z0 in heights:
            v = evaluate_guazzotto_freidberg(model, -x_x, z0)
            assert abs(float(v["psi"])) < 1e-8 and abs(float(v["psi_x"])) < 1e-8 and abs(float(v["psi_y"])) < 1e-8
            assert float(v["psi_xx"]*v["psi_yy"] - v["psi_xy"]**2) < 0      # a saddle, not an extremum


def test_single_null_has_no_mirrored_x_point(solved):
    model = solved["single_null"]
    eps, dx = model.inverse_aspect_ratio, model.x_point_triangularity
    mirrored = evaluate_guazzotto_freidberg(model, -(dx + eps/2*(1 - dx**2)), model.x_point_elongation)
    assert float(np.hypot(mirrored["psi_x"], mirrored["psi_y"])) > 1e-2
    eq = guazzotto_freidberg_to_equilibrium(model, major_radius=1.0, toroidal_field=1.0)
    saddles = [s for s in find_stationary_points(eq, kind="x") if abs(s.psi_n - 1) < 0.05]
    assert saddles and all(s.z < 0 for s in saddles)


def test_export_is_consistent_in_physical_units(solved):
    from matplotlib.path import Path
    from scipy.constants import mu_0

    model = solved["circle"]
    eq = guazzotto_freidberg_to_equilibrium(model, major_radius=1.0, toroidal_field=2.0, resolution=201)
    assert eq.convention.cocos == 11 and eq.lcfs.closed
    assert eq.psi_axis < 0 == eq.psi_boundary and eq.ip > 0
    # Pressure and current vanish at the surface: p ~ psi^2.
    assert eq.pressure[-1] == 0.0 and eq.pprime[-1] == 0.0 and eq.ffprime[-1] == 0.0
    # Ip of Eq. (6.8) against the grid integral of J_phi = -(R p' + FF'/(mu0 R)) in COCOS 11 (x 2 pi).
    rm, zm = np.meshgrid(eq.r, eq.z, indexing="ij")
    psi_n = (eq.psi - eq.psi_axis)/(eq.psi_boundary - eq.psi_axis)
    order = np.argsort((eq.psi_1d - eq.psi_axis)/(eq.psi_boundary - eq.psi_axis))
    x1d = ((eq.psi_1d - eq.psi_axis)/(eq.psi_boundary - eq.psi_axis))[order]
    pp = np.interp(psi_n, x1d, eq.pprime[order]); ff = np.interp(psi_n, x1d, eq.ffprime[order])
    j = -2*np.pi*(rm*pp + ff/(mu_0*rm))
    inside = Path(eq.lcfs.points).contains_points(np.c_[rm.ravel(), zm.ravel()]).reshape(rm.shape)
    ip = np.sum(j*inside)*(eq.r[1] - eq.r[0])*(eq.z[1] - eq.z[0])
    assert ip == pytest.approx(eq.ip, rel=0.01)


def test_q95_agrees_with_vafts_own_q_solver(solved):
    from vaft.process.equilibrium import calculate_q_profile_from_psi

    model = solved["ellipse"]
    eq = guazzotto_freidberg_to_equilibrium(model, major_radius=1.0, toroidal_field=1.0, resolution=257)
    q95 = calculate_q_profile_from_psi(eq.psi, eq.r, eq.z, (eq.psi_1d, eq.f), eq.psi_axis, eq.psi_boundary,
                                       np.array([0.95]), axis_rz=eq.magnetic_axis, cocos=11)
    assert abs(float(np.ravel(q95)[0])) == pytest.approx(guazzotto_freidberg_parameters(model)["q95"], rel=0.02)


@pytest.mark.parametrize("kwargs, match", [
    (dict(topology="bean"), "topology"),
    (dict(inverse_aspect_ratio=1.2), "invalid geometry"),
    (dict(nu=5.0), "current reversal"),
    (dict(topology=DN, x_point_elongation=1.5, x_point_triangularity=0.5), "Eq. 4.4"),
    (dict(elongation=None), "needs elongation"),
])
def test_invalid_inputs_are_refused_with_their_state(kwargs, match):
    base = dict(topology=LIMITED, inverse_aspect_ratio=0.33, nu=1.0, elongation=1.6, triangularity=0.3)
    base.update(kwargs)
    topology = base.pop("topology")
    with pytest.raises(ValueError, match=match):
        solve_guazzotto_freidberg(topology, **base)


def test_no_root_in_the_scan_range_is_an_explicit_failure():
    with pytest.raises(ValueError, match="no physical root"):
        solve_guazzotto_freidberg(LIMITED, inverse_aspect_ratio=0.33, nu=1.0, elongation=1.6, triangularity=0.3,
                                  alpha_range=(0.2, 0.6))
