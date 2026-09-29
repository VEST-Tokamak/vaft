"""Cold-plasma waves (#1113): Stix parameters, dispersion roots, conventions and classification."""

import numpy as np
import pytest

from vaft.formula.constants import EPS0, ME, QE
from vaft.formula.waves import (
    cma_coordinates,
    dielectric_tensor,
    cold_plasma_refractive_index_squared,
    perpendicular_refractive_index_squared,
    plasma_frequency,
    propagation_regime,
    stix_parameters,
)

MD = 2.0 * 1.67262192369e-27


def test_plasma_frequency_is_the_textbook_electron_value():
    # f_pe = 8.98 sqrt(n_e) Hz
    assert plasma_frequency(1e19, -QE, ME) / (2 * np.pi) == pytest.approx(8.98 * np.sqrt(1e19), rel=1e-3)
    assert plasma_frequency(1e19, QE, ME) == plasma_frequency(1e19, -QE, ME)
    with pytest.raises(ValueError):
        plasma_frequency(-1.0, -QE, ME)


def test_electron_stix_parameters_match_the_cma_closed_forms():
    omega = 2 * np.pi * 28e9
    for n_e, B in ((1e19, 0.5), (3e19, 0.8), (5e18, 1.5)):
        X, Y = cma_coordinates(omega, n_e, B)
        s = stix_parameters(omega, n_e, -QE, ME, B)
        assert s.P == pytest.approx(1 - X)
        assert s.R == pytest.approx(1 - X / (1 - Y))
        assert s.L == pytest.approx(1 - X / (1 + Y))
        assert s.S == pytest.approx(1 - X / (1 - Y**2))
        assert s.D == pytest.approx((s.R - s.L) / 2)


def test_the_electron_cyclotron_resonance_is_in_R_not_L():
    omega = 2 * np.pi * 28e9
    B_res = omega * ME / QE
    below, above = (stix_parameters(omega, 1e19, -QE, ME, B) for B in (B_res * 0.999, B_res * 1.001))
    assert abs(below.R) > 100 and np.sign(below.R) != np.sign(above.R)
    assert abs(below.L) < 10 and abs(above.L) < 10


def test_species_sum_and_broadcasting():
    omega = 2 * np.pi * 50e6
    n = np.array([1e19, 1e19])
    q, m = np.array([-QE, QE]), np.array([ME, MD])
    s = stix_parameters(omega, n, q, m, np.array([0.5, 1.0, 2.0]))
    assert np.shape(s.R) == (3,)
    # P sums both species regardless of B
    wp2 = n[0] * QE**2 / (EPS0 * ME) + n[1] * QE**2 / (EPS0 * MD)
    np.testing.assert_allclose(s.P, 1 - wp2 / omega**2)
    with pytest.raises(ValueError):
        stix_parameters(omega, n, q[:1], m, 1.0)
    with pytest.raises(ValueError):
        stix_parameters(0.0, n, q, m, 1.0)


def test_oblique_roots_solve_the_quartic_and_reduce_to_the_limits():
    omega = 2 * np.pi * 28e9
    s = stix_parameters(omega, 5e18, -QE, ME, 0.6)
    for theta in (0.2, 0.7, 1.3):
        for n2 in cold_plasma_refractive_index_squared(s.R, s.L, s.P, theta):
            A = s.S * np.sin(theta) ** 2 + s.P * np.cos(theta) ** 2
            B = s.R * s.L * np.sin(theta) ** 2 + s.P * s.S * (1 + np.cos(theta) ** 2)
            C = s.P * s.R * s.L
            assert A * n2**2 - B * n2 + C == pytest.approx(0.0, abs=1e-9 * max(abs(B), 1.0))
    # parallel: {R, L}; perpendicular: {P, RL/S}
    assert sorted(cold_plasma_refractive_index_squared(s.R, s.L, s.P, 0.0)) == pytest.approx(sorted([s.R, s.L]))
    n2_O, n2_X = perpendicular_refractive_index_squared(s.R, s.L, s.P)
    assert n2_O == s.P and n2_X == pytest.approx(s.R * s.L / s.S)
    assert sorted(cold_plasma_refractive_index_squared(s.R, s.L, s.P, np.pi / 2)) == pytest.approx(
        sorted([n2_O, n2_X]))


def test_the_finite_root_survives_the_resonance_cone():
    # S = 0 at perpendicular propagation: A ~ 0, one root infinite, the other exactly P (= C/B)
    R, L, P = 1.5, -1.5, 0.3
    roots = cold_plasma_refractive_index_squared(R, L, P, np.pi / 2)
    finite = [r for r in roots if abs(r) < 1e10]
    assert finite == [pytest.approx(P, rel=1e-12)]
    assert perpendicular_refractive_index_squared(R, L, P)[0] == pytest.approx(P)
    # an exact A = 0 away from pi/2: tan^2 theta = -P/S; the finite root is C/B
    R, L = 2.0, 0.5
    S = (R + L) / 2
    theta = 0.6
    P = -S * np.tan(theta) ** 2
    A = S * np.sin(theta) ** 2 + P * np.cos(theta) ** 2
    B = R * L * np.sin(theta) ** 2 + P * S * (1 + np.cos(theta) ** 2)
    roots = cold_plasma_refractive_index_squared(R, L, P, theta)
    assert abs(A) < 1e-12
    assert min(roots, key=abs) == pytest.approx(P * R * L / B, rel=1e-9)
    assert max(abs(r) for r in roots) > 1e10


def test_parallel_propagation_with_P_zero_returns_R_and_L():
    assert cold_plasma_refractive_index_squared(1.5, 0.4, 0.0, 0.0) == (1.5, 0.4)
    near = cold_plasma_refractive_index_squared(1.5, 0.4, 1e-12, 0.0)
    assert sorted(near) == pytest.approx([0.4, 1.5])


def test_trailing_axes_broadcast_like_a_loop_over_points():
    omega = 2 * np.pi * 28e9
    n = np.array([[1e19, 2e19, 3e19], [5e18, 1e19, 2e19]])
    q, m = np.array([-QE, QE]), np.array([ME, MD])
    B = np.array([[0.5, 1.0, 2.0], [0.7, 0.9, 1.1]])
    got = stix_parameters(omega, n, q, m, B)
    for field in ("R", "L", "P"):
        ref = np.array([[getattr(stix_parameters(omega, n[:, j], q, m, B[i, j]), field) for j in range(3)]
                        for i in range(2)])
        np.testing.assert_allclose(getattr(got, field), ref)


def test_an_empty_species_is_harmless_at_its_own_resonance():
    omega = 2 * np.pi * 28e9
    B = omega * MD / QE  # the ion resonance, with no ions present
    s = stix_parameters(omega, np.array([1e19, 0.0]), np.array([-QE, QE]), np.array([ME, MD]), B)
    alone = stix_parameters(omega, 1e19, -QE, ME, B)
    assert s.L == pytest.approx(alone.L) and s.R == pytest.approx(alone.R)


def test_the_dielectric_tensor_is_hermitian_and_carries_the_stix_parameters():
    eps = dielectric_tensor(np.array([1.5, 2.0]), np.array([0.4, -1.0]), np.array([0.2, 0.9]))
    assert eps.shape == (2, 3, 3)
    np.testing.assert_allclose(eps, np.conj(np.swapaxes(eps, -1, -2)))
    assert eps[0, 0, 0] == pytest.approx((1.5 + 0.4) / 2) and eps[0, 1, 0] == pytest.approx(1j * (1.5 - 0.4) / 2)


def test_classification():
    np.testing.assert_array_equal(propagation_regime(np.array([2.0, -0.5, 0.0, np.inf, -np.inf])),
                                  ["propagating", "evanescent", "cutoff", "resonance", "resonance"])
    assert propagation_regime(1e-4, atol=1e-3) == "cutoff"
    assert propagation_regime(0.5) == "propagating"
    with pytest.raises(ValueError):
        propagation_regime(np.nan)
    with pytest.raises(ValueError):
        propagation_regime(1.0, atol=-1.0)
