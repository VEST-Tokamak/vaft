"""Analytic/synthetic validation of the exact Green's-function API (#219).

Covers ``vaft.formula.green``: ``greens_function_exact`` (all modes),
``green_psi_exact`` / ``green_br_bz_exact``, rectangular-coil
``mutual_inductance`` / ``self_inductance``, and
``vaft.process.electromagnetics.compute_point_response_matrices``.
All references are analytic or cross-checks — no external fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ellipe, ellipk

from vaft.formula.green import (
    GREEN_EXACT_MODES,
    MU0,
    green_br_bz,
    green_br_bz_exact,
    green_psi_exact,
    green_r,
    greens_function_exact,
    mutual_inductance,
    self_inductance,
)
from vaft.process.electromagnetics import compute_point_response_matrices

RNG = np.random.default_rng(42)


def _random_points(n, r_range=(0.1, 1.5), z_range=(-0.8, 0.8)):
    r = RNG.uniform(*r_range, n)
    z = RNG.uniform(*z_range, n)
    return r, z


# ----------------------------------------------------------------------
# Elliptic-integral modes
# ----------------------------------------------------------------------


def test_k_e_modes_match_scipy_exactly():
    r, z = _random_points(50)
    r0, z0 = _random_points(50)
    m = 4.0 * r * r0 / ((r + r0) ** 2 + (z - z0) ** 2)
    np.testing.assert_array_equal(greens_function_exact(r, z, r0, z0, "K"), ellipk(m))
    np.testing.assert_array_equal(greens_function_exact(r, z, r0, z0, "E"), ellipe(m))


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        greens_function_exact(1.0, 0.0, 1.0, 0.1, "bogus")


# ----------------------------------------------------------------------
# G: reciprocity, axis limit, cross-checks
# ----------------------------------------------------------------------


def test_reciprocity():
    r1, z1 = _random_points(100)
    r2, z2 = _random_points(100)
    g12 = greens_function_exact(r1, z1, r2, z2, "psi")
    g21 = greens_function_exact(r2, z2, r1, z1, "psi")
    np.testing.assert_allclose(g12, g21, rtol=1e-12)


def test_on_axis_flux_is_zero():
    assert greens_function_exact(0.0, 0.3, 0.4, 0.0, "psi") == 0.0
    assert greens_function_exact(0.5, 0.3, 0.0, 0.0, "psi") == 0.0


def test_coincident_point_is_finite():
    for mode in GREEN_EXACT_MODES:
        val = greens_function_exact(0.4, 0.1, 0.4, 0.1, mode)
        assert np.all(np.isfinite(val)), mode


def test_psi_cross_check_vs_approximate_green_r():
    """The legacy polynomial-elliptic green_r agrees to ~1e-6."""
    r, z = _random_points(200)
    exact = green_psi_exact(r, z, 0.45, 0.05)
    approx = green_r(r, z, 0.45, 0.05)
    np.testing.assert_allclose(exact, approx, rtol=1e-6)


def test_br_bz_cross_check_vs_approximate():
    r, z = _random_points(200)
    br_e, bz_e = green_br_bz_exact(r, z, 0.45, 0.05)
    br_a, bz_a = green_br_bz(r, z, 0.45, 0.05)
    np.testing.assert_allclose(br_e, br_a, rtol=1e-5, atol=1e-13)
    np.testing.assert_allclose(bz_e, bz_a, rtol=1e-5, atol=1e-13)


# ----------------------------------------------------------------------
# Derivative modes vs central finite differences
# ----------------------------------------------------------------------


def _fd(fun, x, h):
    return (fun(x + h) - fun(x - h)) / (2.0 * h)


def test_dpsi_dr_matches_finite_difference():
    r, z = _random_points(50)
    r0, z0 = 0.45, 0.05
    h = 1e-6 * r
    fd = _fd(lambda rr: greens_function_exact(rr, z, r0, z0, "psi"), r, h)
    np.testing.assert_allclose(
        greens_function_exact(r, z, r0, z0, "dpsi_dr"), fd, rtol=1e-5
    )


def test_dpsi_dz_matches_finite_difference():
    r, z = _random_points(50)
    r0, z0 = 0.45, 0.05
    h = np.full_like(z, 1e-7)
    fd = _fd(lambda zz: greens_function_exact(r, zz, r0, z0, "psi"), z, h)
    np.testing.assert_allclose(
        greens_function_exact(r, z, r0, z0, "dpsi_dz"), fd, rtol=1e-5, atol=1e-10
    )


def test_d2psi_drdz_matches_finite_difference_of_dpsi_dz():
    r, z = _random_points(50)
    r0, z0 = 0.45, 0.05
    h = 1e-6 * r
    fd = _fd(lambda rr: greens_function_exact(rr, z, r0, z0, "dpsi_dz"), r, h)
    np.testing.assert_allclose(
        greens_function_exact(r, z, r0, z0, "d2psi_drdz"), fd, rtol=1e-4, atol=1e-9
    )


def test_d2psi_dr2_matches_finite_difference_of_dpsi_dr():
    r, z = _random_points(50)
    r0, z0 = 0.45, 0.05
    h = 1e-6 * r
    fd = _fd(lambda rr: greens_function_exact(rr, z, r0, z0, "dpsi_dr"), r, h)
    np.testing.assert_allclose(
        greens_function_exact(r, z, r0, z0, "d2psi_dr2"), fd, rtol=1e-4
    )


# ----------------------------------------------------------------------
# Analytic field limits
# ----------------------------------------------------------------------


def test_on_axis_bz_matches_circular_loop_formula():
    """Bz near the axis of a loop: mu0 a^2 / (2 (a^2 + z^2)^{3/2})."""
    a = 0.4
    z_obs = np.array([-0.5, -0.1, 0.0, 0.2, 0.6])
    r_obs = np.full_like(z_obs, 1e-4)
    _, bz = green_br_bz_exact(r_obs, z_obs, a, 0.0)
    analytic = MU0 * a**2 / (2.0 * (a**2 + z_obs**2) ** 1.5)
    np.testing.assert_allclose(bz, analytic, rtol=1e-4)


def test_far_field_flux_matches_dipole():
    """psi through a distant coaxial loop ~ mu0 pi a^2 b^2 / (2 d^3).

    The dipole reference carries O((a/d)^2, (b/d)^2) corrections, so the
    tolerance reflects the truncation of the reference, not the solver.
    """
    a, b, d = 0.3, 0.05, 8.0
    psi = green_psi_exact(np.array([b]), np.array([d]), a, 0.0)[0]
    analytic = MU0 * np.pi * a**2 * b**2 / (2.0 * d**3)
    np.testing.assert_allclose(psi, analytic, rtol=5e-3)


# ----------------------------------------------------------------------
# Mutual inductance
# ----------------------------------------------------------------------


def test_mutual_inductance_symmetry():
    m12 = mutual_inductance(0.3, 0.0, 0.02, 0.03, 0.5, 0.2, 0.04, 0.01)
    m21 = mutual_inductance(0.5, 0.2, 0.04, 0.01, 0.3, 0.0, 0.02, 0.03)
    np.testing.assert_allclose(m12, m21, rtol=1e-10)


def test_mutual_inductance_point_limit():
    """Point-point mutual equals mu0 * G exactly."""
    m = mutual_inductance(0.3, 0.0, 0.0, 0.0, 0.5, 0.2, 0.0, 0.0)
    expected = MU0 * greens_function_exact(0.3, 0.0, 0.5, 0.2, "psi")
    np.testing.assert_allclose(m, float(expected), rtol=1e-14)


def test_mutual_inductance_rect_converges_to_point():
    point = mutual_inductance(0.3, 0.0, 0.0, 0.0, 0.5, 0.2, 0.0, 0.0)
    small = mutual_inductance(0.3, 0.0, 1e-5, 1e-5, 0.5, 0.2, 1e-5, 1e-5)
    np.testing.assert_allclose(small, point, rtol=1e-6)


def test_mutual_inductance_far_coaxial_loops_dipole():
    """Coaxial loops far apart: M ~ mu0 pi a^2 b^2 / (2 d^3)."""
    a, b, d = 0.3, 0.2, 6.0
    m = mutual_inductance(a, 0.0, 0.0, 0.0, b, d, 0.0, 0.0)
    analytic = MU0 * np.pi * a**2 * b**2 / (2.0 * d**3)
    # Dipole reference truncates at O((a/d)^2, (b/d)^2) ~ 0.4%.
    np.testing.assert_allclose(m, analytic, rtol=1e-2)


def test_mutual_inductance_mu_r_and_turns_scaling():
    base = mutual_inductance(0.3, 0.0, 0.02, 0.03, 0.5, 0.2, 0.04, 0.01)
    scaled = mutual_inductance(
        0.3, 0.0, 0.02, 0.03, 0.5, 0.2, 0.04, 0.01,
        mu_r=1.04, turns1=3.0, turns2=2.0,
    )
    np.testing.assert_allclose(scaled, base * 1.04 * 6.0, rtol=1e-12)


# ----------------------------------------------------------------------
# Self-inductance
# ----------------------------------------------------------------------


def test_self_inductance_positive_finite():
    L = self_inductance(0.4, 0.01, 0.01)
    assert np.isfinite(L) and L > 0.0


def test_self_inductance_thin_loop_formula():
    """Thin loop: L ~ mu0 r (ln(8r/GMD) - 1.75), GMD = sqrt(dr dz / pi).

    The single-term reference uses the equivalent-circle GMD; the true
    rectangular-section GMD differs by a few percent inside the log, so
    this is a sanity check, not a precision claim.
    """
    r, dr, dz = 0.5, 0.004, 0.004
    L = self_inductance(r, dr, dz, n_div=8)
    gmd = np.sqrt(dr * dz / np.pi)
    analytic = MU0 * r * (np.log(8.0 * r / gmd) - 1.75)
    np.testing.assert_allclose(L, analytic, rtol=3e-2)


def test_self_inductance_invalid_input_raises():
    with pytest.raises(ValueError):
        self_inductance(0.0, 0.01, 0.01)


# ----------------------------------------------------------------------
# Vectorized response matrices
# ----------------------------------------------------------------------


def test_point_response_matrices_shapes_and_values():
    obs_r, obs_z = _random_points(7)
    src_r, src_z = _random_points(11, r_range=(0.2, 0.9))
    psi, bz, br = compute_point_response_matrices(obs_r, obs_z, src_r, src_z)
    assert psi.shape == bz.shape == br.shape == (7, 11)
    # Element-wise agreement with the scalar exact functions.
    np.testing.assert_allclose(
        psi[2, 3], float(green_psi_exact(obs_r[2], obs_z[2], src_r[3], src_z[3]))
    )
    br_s, bz_s = green_br_bz_exact(obs_r[4], obs_z[4], src_r[9], src_z[9])
    np.testing.assert_allclose(br[4, 9], float(br_s))
    np.testing.assert_allclose(bz[4, 9], float(bz_s))


def test_point_response_matrices_turns_and_groups():
    obs_r, obs_z = _random_points(5)
    src_r, src_z = _random_points(6, r_range=(0.2, 0.9))
    turns = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0])
    groups = np.array([0, 0, 1, 1, 2, 2])
    psi, bz, br = compute_point_response_matrices(
        obs_r, obs_z, src_r, src_z, turns=turns, groups=groups, n_groups=3
    )
    assert psi.shape == (5, 3)
    raw, _, _ = compute_point_response_matrices(obs_r, obs_z, src_r, src_z)
    expected_col0 = raw[:, 0] * 1.0 + raw[:, 1] * 2.0
    np.testing.assert_allclose(psi[:, 0], expected_col0, rtol=1e-12)


def test_point_response_matrices_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        compute_point_response_matrices([0.3, 0.4], [0.0], [0.5], [0.0])


@pytest.mark.perf
def test_point_response_matrix_performance_budget():
    """75 sensors x 965 sources psi+Bz+Br in < 50 ms (issue #219)."""
    import time

    obs_r, obs_z = _random_points(75)
    src_r, src_z = _random_points(965, r_range=(0.1, 0.9))
    compute_point_response_matrices(obs_r, obs_z, src_r, src_z)  # warm-up
    t0 = time.perf_counter()
    compute_point_response_matrices(obs_r, obs_z, src_r, src_z)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.050, f"response matrix took {elapsed * 1e3:.1f} ms"
