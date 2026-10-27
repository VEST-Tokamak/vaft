"""Separatrix safety-factor asymptotics and X-point geometry (#2152): the formulas called, not only catalogued."""

import numpy as np
import pytest
from scipy.integrate import quad

from vaft.formula import equilibrium as eq
from vaft.formula import geometry as geo


def test_the_log_asymptote_and_its_gradient():
    delta = np.array([1e-2, 1e-4, 1e-6])
    q = eq.separatrix_q_asymptotic(delta, 0.11, 2.3)
    assert q == pytest.approx(0.11 * np.log(1.0 / delta) + 2.3, rel=1e-15)
    # the gradient is d|q|/dpsi_N = -d|q|/d delta, A/delta
    h = 1e-9
    numeric = (eq.separatrix_q_asymptotic(1e-3 - h, 0.11, 2.3) - eq.separatrix_q_asymptotic(1e-3 + h, 0.11, 2.3)) / (2 * h)
    assert eq.separatrix_q_gradient_asymptotic(1e-3, 0.11) == pytest.approx(numeric, rel=1e-6)
    assert eq.separatrix_q_gradient_asymptotic(1e-3, 0.11) == pytest.approx(110.0)


@pytest.mark.parametrize("delta", [0.0, 1.0, -1e-3, np.nan, [1e-3, 1.5]])
def test_delta_outside_the_open_unit_interval_is_rejected(delta):
    with pytest.raises(ValueError, match="delta_psi_n"):
        eq.separatrix_q_asymptotic(delta, 0.1, 2.0)
    with pytest.raises(ValueError, match="delta_psi_n"):
        eq.separatrix_q_gradient_asymptotic(delta, 0.1)


@pytest.mark.parametrize("A", [0.0, -0.1, np.inf])
def test_a_non_positive_coefficient_is_rejected(A):
    with pytest.raises(ValueError, match="logarithmic coefficient"):
        eq.separatrix_q_asymptotic(1e-3, A, 2.0)
    with pytest.raises(ValueError, match="logarithmic coefficient"):
        eq.separatrix_rational_spacing(1, A)


def test_the_rational_offset_inverts_the_asymptote_and_spaces_geometrically():
    A, B = 0.11, 2.3
    for m, n in ((3, 1), (7, 2), (-10, 3), (10, -3)):
        delta = eq.separatrix_rational_offset(m, n, A, B)
        assert eq.separatrix_q_asymptotic(delta, A, B) == pytest.approx(abs(m) / abs(n), rel=1e-12)
    # C exp(-|m|/(|n| A)) with C = exp(B/A)
    assert eq.separatrix_rational_offset(4, 1, A, B) == pytest.approx(np.exp(B / A) * np.exp(-4 / A))
    for n in (1, 2, 5):
        ratio = eq.separatrix_rational_offset(12, n, A, B) / eq.separatrix_rational_offset(11, n, A, B)
        assert eq.separatrix_rational_spacing(n, A) == pytest.approx(ratio, rel=1e-12)
    with pytest.raises(ValueError, match="non-zero"):
        eq.separatrix_rational_offset(3, 0, A, B)
    with pytest.raises(ValueError, match="non-zero"):
        eq.separatrix_rational_spacing(0, A)


def _hessian(kappa_u, kappa_v, angle):
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    h = rot @ np.diag([kappa_u, -kappa_v]) @ rot.T
    return h[0, 0], h[1, 1], h[0, 1]


@pytest.mark.parametrize("kappa_u, kappa_v, angle", [(2.0, 0.5, 0.3), (1.0, 1.0, 0.0), (0.4, 3.0, -1.1)])
def test_the_saddle_expansion_recovers_a_rotated_saddle(kappa_u, kappa_v, angle):
    s = geo.xpoint_saddle_expansion(*_hessian(kappa_u, kappa_v, angle))
    assert (s.kappa_u, s.kappa_v) == pytest.approx((kappa_u, kappa_v), rel=1e-12)
    assert s.det_hessian == pytest.approx(-kappa_u * kappa_v, rel=1e-12)
    # the u axis is the rising direction, up to its sign
    u = np.array([np.cos(s.angle_u), np.sin(s.angle_u)])
    assert abs(u @ np.array([np.cos(angle), np.sin(angle)])) == pytest.approx(1.0, abs=1e-12)


def test_the_current_free_slab_x_point_is_the_symmetric_special_case():
    b_prime, h = 0.7, 1e-4
    psi = lambda x, y: geo.x_point_flux(x, y, b_prime)  # noqa: E731
    hxx = (psi(h, 0) - 2 * psi(0, 0) + psi(-h, 0)) / h**2
    hyy = (psi(0, h) - 2 * psi(0, 0) + psi(0, -h)) / h**2
    s = geo.xpoint_saddle_expansion(hxx, hyy, 0.0)
    assert s.kappa_u == pytest.approx(b_prime, rel=1e-6) and s.kappa_v == pytest.approx(b_prime, rel=1e-6)


def test_not_a_saddle_is_rejected():
    with pytest.raises(ValueError, match="O-point"):
        geo.xpoint_saddle_expansion(1.0, 2.0, 0.0)
    with pytest.raises(ValueError, match="degenerate"):
        geo.xpoint_saddle_expansion(1.0, -1e-9, 0.0)
    with pytest.raises(ValueError, match="degenerate"):
        geo.xpoint_saddle_expansion(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        geo.xpoint_saddle_expansion(np.nan, -1.0, 0.0)


def test_the_field_scale_is_the_gradient_of_the_saddle_over_R():
    ku, kv, R = 1.3, 0.6, 0.31
    u, v = np.array([0.01, -0.02, 0.0]), np.array([0.0, 0.015, -0.03])
    grad = np.hypot(ku * u, -kv * v)  # gradient of (ku u**2 - kv v**2)/2
    assert geo.xpoint_field_scale(u, v, ku, kv, R) == pytest.approx(grad / R, rel=1e-15)
    with pytest.raises(ValueError, match="R_X"):
        geo.xpoint_field_scale(0.0, 0.0, ku, kv, 0.0)


def test_the_log_coefficient_is_the_saddle_integral_slope():
    """oint dl/|grad psi| over the hyperbola branch near the saddle grows as ln(1/delta psi)/sqrt|det H|."""
    ku, kv, L = 1.7, 0.45, 0.2

    def branch(dpsi):
        # (ku u^2 - kv v^2)/2 = -|dpsi| on the core side: v = +-sqrt((ku u^2 + 2|dpsi|)/kv), and along it
        # dl/|grad psi| = du/|dpsi/dv| = du / sqrt(kv (ku u^2 + 2|dpsi|))
        f = lambda u: 1.0 / np.sqrt(kv * (ku * u * u + 2.0 * abs(dpsi)))  # noqa: E731
        return 2.0 * quad(f, 0.0, L, epsabs=1e-13, epsrel=1e-12, limit=200)[0]

    d1, d2 = 1e-7, 1e-9
    slope = (branch(d2) - branch(d1)) / np.log(d1 / d2)
    assert slope == pytest.approx(1.0 / np.sqrt(ku * kv), rel=1e-3)
    F, R = 0.04, 0.3
    det = -ku * kv
    assert geo.xpoint_log_q_coefficient(F, R, det) == pytest.approx(F / (2 * np.pi * R) * slope, rel=1e-3)


def test_the_log_coefficient_sums_x_points_and_converts_flux_families():
    F, R, det = 0.04, 0.3, -2.5e-3
    one = geo.xpoint_log_q_coefficient(F, R, det)
    assert one == pytest.approx(abs(F) / (2 * np.pi * R * np.sqrt(-det)))
    assert geo.xpoint_log_q_coefficient([F, -F], [R, R], [det, det]) == pytest.approx(2 * one)
    # full weber: the Hessian is 2 pi larger and the 2 pi in the prefactor goes
    assert geo.xpoint_log_q_coefficient(F, R, det * (2 * np.pi) ** 2, psi_per_radian=False) == pytest.approx(one)
    with pytest.raises(ValueError, match="negative"):
        geo.xpoint_log_q_coefficient(F, R, 1e-3)
    with pytest.raises(ValueError, match="matching"):
        geo.xpoint_log_q_coefficient([F, F, F], [R, R], det)
    with pytest.raises(ValueError, match="R_X"):
        geo.xpoint_log_q_coefficient(F, -R, det)


def test_resolution_length_and_flux_position_error():
    kappa, dpsi = 0.8, 2e-4
    ell = geo.xpoint_resolution_length(dpsi, kappa)
    assert 0.5 * kappa * ell**2 == pytest.approx(dpsi, rel=1e-14)
    assert geo.xpoint_resolution_length(-dpsi, kappa) == pytest.approx(ell)
    with pytest.raises(ValueError, match="kappa"):
        geo.xpoint_resolution_length(dpsi, 0.0)
    err = geo.flux_position_error([1e-3, 1e-3], [0.5, 0.0])
    assert err[0] == pytest.approx(2e-3) and np.isinf(err[1])
    with pytest.raises(ValueError, match="grad_psi"):
        geo.flux_position_error(1e-3, -0.1)
