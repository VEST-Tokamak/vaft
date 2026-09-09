"""Tests for calculating the safety factor profile q(psi) from psi map and F(psi).

Verifies Issue #647:
- Benchmark against analytic Solov'ev on-axis limit q_0 = |F_0| / (R_0 * sqrt(psi_RR * psi_ZZ))
- Benchmark against high-precision 1D adaptive contour quadrature across multiple flux surfaces
- Convergence with grid resolution
- COCOS invariance and orientation sign compliance
- Input flexibility: callable, scalar, array, tuple, and return_details
- Verification that flux_surface_quantities returns 'q'
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import root_scalar

from vaft.data.equilibrium import SolovevConstraint
from vaft.formula.equilibrium import q_from_flux_surface_averages
from vaft.process.equilibrium import (
    calculate_q_profile_from_psi,
    evaluate_solovev,
    flux_surface_quantities,
    solve_solovev_constraints,
)

R0, MINOR, ELONGATION = 1.0, 0.30, 1.6


@pytest.fixture(scope="module")
def solovev_model():
    theta = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    solved = solve_solovev_constraints(
        [
            SolovevConstraint(
                R0 + MINOR * np.cos(t), ELONGATION * MINOR * np.sin(t), "psi", 0.0
            )
            for t in theta
        ],
        pprime=-1.0e4,
        ffprime=0.05,
        rref=R0,
        psi_boundary=0.0,
    )
    assert solved.rank == 5
    return solved


def _evaluate_solovev_grid(model, size):
    r = np.linspace(R0 - 1.25 * MINOR, R0 + 1.25 * MINOR, size)
    z = np.linspace(-1.25 * ELONGATION * MINOR, 1.25 * ELONGATION * MINOR, size)
    grid_r, grid_z = np.meshgrid(r, z, indexing="ij")
    ev = evaluate_solovev(model, grid_r, grid_z)
    psi = np.asarray(ev["psi"], float)
    return r, z, psi


def _solovev_analytic_q0(model, f0: float) -> float:
    """Exact analytic on-axis safety factor from local Hessian."""
    eps = 1e-4
    ev_axis = evaluate_solovev(model, R0, 0.0)
    ev_r_p = evaluate_solovev(model, R0 + eps, 0.0)
    ev_r_m = evaluate_solovev(model, R0 - eps, 0.0)
    ev_z_p = evaluate_solovev(model, R0, eps)
    ev_z_m = evaluate_solovev(model, R0, -eps)

    psi_rr = (ev_r_p["psi"] + ev_r_m["psi"] - 2.0 * ev_axis["psi"]) / (eps**2)
    psi_zz = (ev_z_p["psi"] + ev_z_m["psi"] - 2.0 * ev_axis["psi"]) / (eps**2)

    return abs(float(f0)) / (R0 * np.sqrt(abs(float(psi_rr) * float(psi_zz))))


def _solovev_quadrature_q(model, psi_target: float, f0: float) -> float:
    """High-precision reference q via 1D adaptive contour quadrature."""

    def get_contour_point(t):
        def obj(rad):
            return (
                float(
                    evaluate_solovev(
                        model,
                        R0 + rad * np.cos(t),
                        ELONGATION * rad * np.sin(t),
                    )["psi"]
                )
                - psi_target
            )

        sol = root_scalar(obj, bracket=[0.0, MINOR * 1.5])
        rad_val = sol.root
        return R0 + rad_val * np.cos(t), ELONGATION * rad_val * np.sin(t)

    h = 1e-5

    def integrand(t):
        r_t, z_t = get_contour_point(t)
        r_p, z_p = get_contour_point(t + h)
        r_m, z_m = get_contour_point(t - h)
        dr_dt = (r_p - r_m) / (2.0 * h)
        dz_dt = (z_p - z_m) / (2.0 * h)
        dl_dt = np.hypot(dr_dt, dz_dt)
        ev = evaluate_solovev(model, r_t, z_t)
        grad_psi = np.hypot(float(ev["dpsi_dr"]), float(ev["dpsi_dz"]))
        return dl_dt / (r_t * grad_psi)

    integral = quad(integrand, 0.0, 2.0 * np.pi, epsabs=1e-8, epsrel=1e-8)[0]
    return abs(float(f0)) / (2.0 * np.pi) * integral


def test_q_from_flux_surface_averages_pure_formula():
    """Verify formula layer units and COCOS factor."""
    # COCOS 1 (Wb/rad): factor is 1 / (2*pi)^2
    q_rad = q_from_flux_surface_averages(1.0, 1.0, 1.0, cocos=1)
    assert np.isclose(q_rad, 1.0 / (4.0 * np.pi**2))

    # COCOS 11 (full Wb): factor is 1 / (2*pi)
    q_wb = q_from_flux_surface_averages(1.0, 1.0, 1.0, cocos=11)
    assert np.isclose(q_wb, 1.0 / (2.0 * np.pi))

    # Sign handling
    q_neg = q_from_flux_surface_averages(1.0, 1.0, 1.0, cocos=1, sigma_ip=-1, sigma_b0=1)
    assert q_neg < 0


def test_analytic_solovev_axis_limit(solovev_model):
    """Calculated q_0 matches the analytic Hessian formula on Solov'ev."""
    f0 = 1.2
    r, z, psi = _evaluate_solovev_grid(solovev_model, 129)
    psi_axis = float(psi.max())
    psi_edge = 0.0

    q_calc = calculate_q_profile_from_psi(
        psi,
        r,
        z,
        f0,
        psi_axis=psi_axis,
        psi_boundary=psi_edge,
        cocos=1,
        axis_rz=(R0, 0.0),
    )

    q0_expected = _solovev_analytic_q0(solovev_model, f0)
    rel_error = abs(q_calc[0] - q0_expected) / q0_expected

    # On a 129x129 grid, extrapolated q0 matches within 0.2%
    assert rel_error < 2.0e-3


def test_high_precision_quadrature_benchmark(solovev_model):
    """Calculated q(psi) matches adaptive 1D contour quadrature across surfaces."""
    f0 = 1.0
    r, z, psi = _evaluate_solovev_grid(solovev_model, 129)
    psi_axis = float(psi.max())
    psi_edge = 0.0

    levels = [0.2, 0.4, 0.6, 0.8]
    q_calc = calculate_q_profile_from_psi(
        psi,
        r,
        z,
        f0,
        psi_axis=psi_axis,
        psi_boundary=psi_edge,
        levels_norm=levels,
        cocos=1,
        axis_rz=(R0, 0.0),
    )

    for idx, lvl in enumerate(levels):
        psi_target = psi_axis + lvl * (psi_edge - psi_axis)
        q_ref = _solovev_quadrature_q(solovev_model, psi_target, f0)
        rel_err = abs(q_calc[idx] - q_ref) / q_ref
        assert rel_err < 5.0e-4, f"Level {lvl} rel_err={rel_err} exceeds 5e-4"


def test_mesh_convergence(solovev_model):
    """Refining grid resolution from 65 to 129 to 257 improves error monotonically."""
    f0 = 1.0
    psi_axis = float(evaluate_solovev(solovev_model, R0, 0.0)["psi"])
    psi_edge = 0.0
    lvl = 0.5
    psi_target = psi_axis + lvl * (psi_edge - psi_axis)
    q_ref = _solovev_quadrature_q(solovev_model, psi_target, f0)

    errors = []
    for size in (65, 129, 257):
        r, z, psi = _evaluate_solovev_grid(solovev_model, size)
        q_val = calculate_q_profile_from_psi(
            psi,
            r,
            z,
            f0,
            psi_axis=psi_axis,
            psi_boundary=psi_edge,
            levels_norm=[lvl],
            cocos=1,
            axis_rz=(R0, 0.0),
        )[0]
        errors.append(abs(q_val - q_ref) / q_ref)

    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 5.0e-5


def test_cocos_invariance(solovev_model):
    """The physical safety factor profile is invariant whether input is COCOS 1 or COCOS 11."""
    f0 = 1.0
    r, z, psi_rad = _evaluate_solovev_grid(solovev_model, 65)
    psi_axis_rad = float(psi_rad.max())
    psi_edge_rad = 0.0

    levels = np.linspace(0.0, 1.0, 21)

    # COCOS 1: Wb/rad
    q_cocos1 = calculate_q_profile_from_psi(
        psi_rad,
        r,
        z,
        f0,
        psi_axis=psi_axis_rad,
        psi_boundary=psi_edge_rad,
        levels_norm=levels,
        cocos=1,
        axis_rz=(R0, 0.0),
    )

    # COCOS 11: full Weber (multiplied by 2*pi)
    psi_wb = psi_rad * (2.0 * np.pi)
    psi_axis_wb = psi_axis_rad * (2.0 * np.pi)
    psi_edge_wb = psi_edge_rad * (2.0 * np.pi)

    q_cocos11 = calculate_q_profile_from_psi(
        psi_wb,
        r,
        z,
        f0,
        psi_axis=psi_axis_wb,
        psi_boundary=psi_edge_wb,
        levels_norm=levels,
        cocos=11,
        axis_rz=(R0, 0.0),
    )

    np.testing.assert_allclose(q_cocos1, q_cocos11, rtol=1e-12)


def test_cocos_signs(solovev_model):
    """Safety factor sign follows Sauter Eq. 23 under COCOS conventions."""
    r, z, psi = _evaluate_solovev_grid(solovev_model, 65)
    psi_axis = float(psi.max())
    psi_edge = 0.0

    # Normal orientations: Ip > 0, B0 > 0 -> q > 0
    q_pos = calculate_q_profile_from_psi(
        psi,
        r,
        z,
        1.0,
        psi_axis=psi_axis,
        psi_boundary=psi_edge,
        levels_norm=[0.5],
        cocos=11,
        sigma_ip=1,
        sigma_b0=1,
    )
    assert q_pos[0] > 0

    # Reversed current: Ip < 0, B0 > 0 -> q < 0
    q_neg_ip = calculate_q_profile_from_psi(
        psi,
        r,
        z,
        1.0,
        psi_axis=psi_axis,
        psi_boundary=psi_edge,
        levels_norm=[0.5],
        cocos=11,
        sigma_ip=-1,
        sigma_b0=1,
    )
    assert q_neg_ip[0] < 0

    # Both reversed: Ip < 0, B0 < 0 -> q > 0
    q_both_neg = calculate_q_profile_from_psi(
        psi,
        r,
        z,
        -1.0,
        psi_axis=psi_axis,
        psi_boundary=psi_edge,
        levels_norm=[0.5],
        cocos=11,
        sigma_ip=-1,
        sigma_b0=-1,
    )
    assert q_both_neg[0] > 0


def test_input_flexibility_and_details(solovev_model):
    """Tests callable f, tuple (psi, f), and return_details=True."""
    r, z, psi = _evaluate_solovev_grid(solovev_model, 65)
    psi_axis = float(psi.max())
    psi_edge = 0.0
    levels = np.linspace(0.0, 1.0, 21)

    # 1. Scalar F
    q_scalar = calculate_q_profile_from_psi(
        psi,
        r,
        z,
        2.0,
        psi_axis=psi_axis,
        psi_boundary=psi_edge,
        levels_norm=levels,
        cocos=1,
    )

    # 2. Callable F(psi)
    q_callable = calculate_q_profile_from_psi(
        psi,
        r,
        z,
        lambda p: 2.0,
        psi_axis=psi_axis,
        psi_boundary=psi_edge,
        levels_norm=levels,
        cocos=1,
    )
    np.testing.assert_allclose(q_scalar, q_callable, rtol=1e-12)

    # 3. Tuple (psi_f, f_values)
    psi_samples = np.linspace(min(psi_axis, psi_edge), max(psi_axis, psi_edge), 10)
    q_tuple = calculate_q_profile_from_psi(
        psi,
        r,
        z,
        (psi_samples, np.full_like(psi_samples, 2.0)),
        psi_axis=psi_axis,
        psi_boundary=psi_edge,
        levels_norm=levels,
        cocos=1,
    )
    np.testing.assert_allclose(q_scalar, q_tuple, rtol=1e-12)

    # 4. return_details=True
    details = calculate_q_profile_from_psi(
        psi,
        r,
        z,
        2.0,
        psi_axis=psi_axis,
        psi_boundary=psi_edge,
        levels_norm=levels,
        cocos=1,
        return_details=True,
    )
    assert isinstance(details, dict)
    assert "q" in details
    assert "levels_norm" in details
    assert "psi_levels" in details
    assert "q_axis" in details
    assert "q_95" in details
    assert "surfaces" in details
    assert np.isclose(details["q_axis"], details["q"][0])


def test_flux_surface_quantities_populates_q(solovev_model):
    """flux_surface_quantities populates 'q' when f_profile is given, and NaN when None."""
    r, z, psi = _evaluate_solovev_grid(solovev_model, 65)
    psi_axis = float(psi.max())
    psi_edge = 0.0
    levels = np.linspace(0.0, 1.0, 11)

    # Without f_profile -> q is all NaN
    surfaces_no_f = flux_surface_quantities(
        psi,
        r,
        z,
        psi_axis,
        psi_edge,
        levels,
        f_profile=None,
    )
    assert "q" in surfaces_no_f
    assert np.isnan(surfaces_no_f["q"]).all()

    # With f_profile -> q is finite
    f_vals = np.ones_like(levels) * 1.5
    surfaces_with_f = flux_surface_quantities(
        psi,
        r,
        z,
        psi_axis,
        psi_edge,
        levels,
        f_profile=f_vals,
    )
    assert "q" in surfaces_with_f
    assert np.isfinite(surfaces_with_f["q"]).all()
    assert (surfaces_with_f["q"] > 0).all()
