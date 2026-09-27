"""TF ripple formulas (#1070): the analytic limits the issue asks for."""

import numpy as np
import pytest

from vaft.formula.equilibrium import vacuum_toroidal_field
from vaft.formula.particle import parallel_speed_from_mu
from vaft.formula.ripple import (
    gwb_stochastic_threshold,
    gwb_stochasticity_parameter,
    ripple_amplitude,
    ripple_trapping_pitch,
    ripple_well_parameter,
    toroidal_ripple_field,
)


def test_without_ripple_the_field_is_the_expanded_one_over_r():
    B0, R0, r = 2.0, 3.0, 0.1
    theta = np.linspace(0, 2 * np.pi, 13)
    smooth = toroidal_ripple_field(B0, r / R0, theta, 0.0, 16, 0.3)
    exact = vacuum_toroidal_field(B0, R0, R0 + r * np.cos(theta))
    np.testing.assert_allclose(smooth, exact, atol=1.1 * B0 * (r / R0) ** 2)


@pytest.mark.parametrize("delta", [0.001, 0.01, 0.05])
def test_the_amplitude_of_the_ripple_term_is_delta(delta):
    phi = np.linspace(0, 2 * np.pi, 20001)
    for eps, theta in ((0.0, 0.0), (0.3, 0.0), (0.3, 2.0)):  # the local amplitude, wherever on the surface
        B = toroidal_ripple_field(1.5, eps, theta, delta, 18, phi)
        assert ripple_amplitude(B.max(), B.min()) == pytest.approx(delta, rel=1e-6)
    with pytest.raises(ValueError):
        ripple_amplitude(1.0, 1.1)


def test_the_continuous_coil_limit_is_axisymmetric():
    # more coils with the ripple they produce falling away: B(phi) flattens
    phi = np.linspace(0, 2 * np.pi, 5001)
    spreads = [np.ptp(toroidal_ripple_field(1.0, 0.1, 0.5, 0.02 * np.exp(-n / 4), n, phi)) for n in (8, 16, 32)]
    assert spreads[0] > spreads[1] > spreads[2]


@pytest.mark.parametrize("theta", [0.3, 0.8, 1.4])
def test_wells_along_the_field_line_appear_exactly_below_alpha_star_one(theta):
    eps, q, n = 0.25, 2.0, 16
    delta_c = eps * abs(np.sin(theta)) / (n * q)  # alpha* = 1 here
    assert ripple_well_parameter(eps, theta, q, delta_c, n) == pytest.approx(1.0)
    # locally the smooth part is a straight slope eps sin(theta); the ripple rides on it
    x = np.linspace(-0.5, 0.5, 200001)

    def has_local_minimum(delta):
        smooth = toroidal_ripple_field(1.0, eps, theta, 0.0, n, 0.0) + eps * np.sin(theta) * x
        B = smooth - delta * np.cos(n * q * (theta + x))
        return np.any((B[1:-1] < B[:-2]) & (B[1:-1] < B[2:]))

    assert has_local_minimum(1.05 * delta_c)
    assert not has_local_minimum(0.95 * delta_c)


def test_the_trapping_pitch_is_the_mirror_condition_and_sqrt_two_delta_when_small():
    for delta in (1e-4, 1e-3, 0.01):
        assert ripple_trapping_pitch(delta) == pytest.approx(np.sqrt(2 * delta), rel=1.01 * delta)
    delta = 0.02
    xi = ripple_trapping_pitch(delta)
    B_min, B_max = 1 - delta, 1 + delta
    # a particle with exactly that pitch at the well bottom just stops at the barrier
    assert parallel_speed_from_mu(1.0, xi, B_min, B_max) == pytest.approx(0.0, abs=1e-7)
    assert np.isnan(parallel_speed_from_mu(1.0, 0.9 * xi, B_min, B_max))


def test_parallel_speed_bounces_where_the_mirror_says():
    v, xi, B_ref = 3.0, 0.6, 1.0
    B_bounce = B_ref / (1 - xi**2)
    assert parallel_speed_from_mu(v, xi, B_ref, B_ref) == pytest.approx(v * xi)
    assert parallel_speed_from_mu(v, xi, B_ref, B_bounce) == pytest.approx(0.0, abs=1e-12)
    assert parallel_speed_from_mu(v, xi, B_ref, 0.5 * B_ref) > v * xi
    with pytest.raises(ValueError):
        parallel_speed_from_mu(v, 1.2, B_ref, B_ref)


def test_the_gwb_threshold_falls_with_gyroradius_shear_and_coil_count():
    base = dict(epsilon=0.2, q=2.0, dq_dr=4.0, rho=0.02, n_tf=16)
    d0 = gwb_stochastic_threshold(**base)
    assert d0 == pytest.approx((0.2 / (np.pi * 16 * 2.0)) ** 1.5 / (0.02 * 4.0))
    assert gwb_stochastic_threshold(**{**base, "rho": 0.04}) < d0
    assert gwb_stochastic_threshold(**{**base, "dq_dr": 8.0}) < d0
    assert gwb_stochastic_threshold(**{**base, "epsilon": 0.3}) > d0
    assert gwb_stochasticity_parameter(2 * d0, **base) == pytest.approx(2.0)
    for bad in ({"rho": 0.0}, {"dq_dr": -1.0}, {"n_tf": 0}):
        with pytest.raises(ValueError):
            gwb_stochastic_threshold(**{**base, **bad})
