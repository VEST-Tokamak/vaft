"""Cold-plasma waves (#1113): Stix parameters, dispersion roots, conventions and classification."""

import numpy as np
import pytest

from vaft.formula.constants import EPS0, ME, QE
from vaft.formula.waves import (
    cma_coordinates,
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


def test_the_resonance_cone_is_infinite_n2():
    # A = S sin^2 + P cos^2 = 0 at tan^2 theta = -P/S
    R, L, P = 2.0, 0.5, -1.0
    S = (R + L) / 2
    theta = np.arctan(np.sqrt(-P / S))
    R2, L2, P2 = np.float64(R), np.float64(L), -S * np.tan(theta) ** 2  # exact A = 0 through P
    plus, minus = cold_plasma_refractive_index_squared(R2, L2, P2, theta)
    assert np.isinf(plus) or np.isinf(minus) or abs(plus) > 1e12 or abs(minus) > 1e12


def test_classification():
    np.testing.assert_array_equal(propagation_regime(np.array([2.0, -0.5, 0.0, np.inf, -np.inf])),
                                  ["propagating", "evanescent", "cutoff", "resonance", "resonance"])
    assert propagation_regime(1e-4, atol=1e-3) == "cutoff"
    assert propagation_regime(0.5) == "propagating"
    with pytest.raises(ValueError):
        propagation_regime(np.nan)
    with pytest.raises(ValueError):
        propagation_regime(1.0, atol=-1.0)
