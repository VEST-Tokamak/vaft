"""Neoclassical orbit scales and NTV relations (#1111): orderings, conventions and signs."""

import numpy as np
import pytest

from vaft.formula.constants import QE
from vaft.formula.neoclassical import (
    banana_width,
    collisions_per_transit,
    deeply_trapped_bounce_frequency,
    neoclassical_regime_boundaries,
    transit_frequency,
    trapped_particle_effective_collision_frequency,
)
from vaft.formula.ntv import nonambipolar_torque_density, ntv_precession_frequency
from vaft.formula.particle import bounce_harmonic_detuning


def test_bounce_is_slower_than_transit_by_root_epsilon():
    v, q, R0 = 2.0e5, 2.0, 0.4
    for eps in (0.05, 0.3, 0.6):
        ratio = deeply_trapped_bounce_frequency(v, q, R0, eps) / transit_frequency(v, q, R0)
        assert ratio == pytest.approx(np.sqrt(eps / 2.0))
    assert transit_frequency(v, q, R0) == pytest.approx(v / (q * R0))


def test_bounce_frequency_matches_a_mirror_orbit():
    # integrate the parallel motion in B = B0 (1 - eps cos(s / qR0)) for a deeply trapped particle
    q, R0, eps, v_perp = 1.5, 1.0, 0.2, 1.0
    s, v_par, dt = 0.0, 1e-3 * np.sqrt(eps) * v_perp, 1e-2
    crossings, t, last = [], 0.0, v_par
    for _ in range(20000):
        accel = -0.5 * v_perp**2 * eps * np.sin(s / (q * R0)) / (q * R0)
        v_par += accel * dt
        s += v_par * dt
        t += dt
        if last > 0 >= v_par:
            crossings.append(t)
        last = v_par
    period = np.mean(np.diff(crossings))
    assert 2 * np.pi / period == pytest.approx(deeply_trapped_bounce_frequency(v_perp, q, R0, eps), rel=1e-3)


def test_effective_collision_frequency_and_banana_width_scalings():
    assert trapped_particle_effective_collision_frequency(1e3, 0.1) == pytest.approx(1e4)
    assert trapped_particle_effective_collision_frequency(0.0, 0.1) == 0.0
    assert banana_width(1e-3, 2.0, 0.25) == pytest.approx(4e-3)
    # banana width grows as 1/sqrt(eps) toward the axis
    widths = banana_width(1e-3, 2.0, np.array([0.04, 0.16]))
    assert widths[0] / widths[1] == pytest.approx(2.0)


def test_regime_boundaries_meet_the_trapped_particle_ordering():
    eps = np.array([0.1, 0.3])
    banana_plateau, plateau_ps = neoclassical_regime_boundaries(eps)
    np.testing.assert_allclose(banana_plateau, eps**1.5)
    np.testing.assert_allclose(plateau_ps, 1.0)
    # at the banana-plateau boundary nu_eff equals the bounce frequency, up to the sqrt(2) of the exact omega_b
    v, q, R0 = 1e5, 2.0, 1.0
    for e in eps:
        nu = neoclassical_regime_boundaries(e)[0] * transit_frequency(v, q, R0)
        assert collisions_per_transit(nu, v, q, R0) == pytest.approx(e**1.5)
        ratio = trapped_particle_effective_collision_frequency(nu, e) / deeply_trapped_bounce_frequency(v, q, R0, e)
        assert ratio == pytest.approx(np.sqrt(2.0))


def test_orbit_formulas_refuse_invalid_inputs():
    for bad in (0.0, 1.0, -0.1, np.nan):
        with pytest.raises(ValueError):
            banana_width(1e-3, 2.0, bad)
        with pytest.raises(ValueError):
            neoclassical_regime_boundaries(bad)
    with pytest.raises(ValueError):
        transit_frequency(-1.0, 2.0, 1.0)
    with pytest.raises(ValueError):
        trapped_particle_effective_collision_frequency(-1.0, 0.1)
    with pytest.raises(ValueError):
        collisions_per_transit(1.0, 1.0, 0.0, 1.0)


def test_scalar_and_array_parity():
    eps = np.array([0.1, 0.2, 0.4])
    arr = deeply_trapped_bounce_frequency(1e5, 2.0, 1.0, eps)
    assert isinstance(arr, np.ndarray)
    assert [deeply_trapped_bounce_frequency(1e5, 2.0, 1.0, e) for e in eps] == pytest.approx(list(arr))
    assert isinstance(banana_width(1e-3, 2.0, 0.1), float)


def test_precession_vanishes_at_the_superbanana_resonance_and_flips_with_rotation():
    assert ntv_precession_frequency(-3.0e3, 3.0e3) == 0.0
    assert ntv_precession_frequency(1.0e3, 2.0e3) == pytest.approx(3.0e3)
    # reversing both the E x B rotation and the drift reverses the precession
    assert ntv_precession_frequency(-1.0e3, -2.0e3) == pytest.approx(-3.0e3)
    np.testing.assert_allclose(ntv_precession_frequency(np.array([-1.0, 0.0, 1.0]), 1.0), [0.0, 1.0, 2.0])
    # the l = 0 bounce harmonic of the particle module, with the sign of n that makes it a precession
    assert ntv_precession_frequency(1.2e3, 0.7e3) == pytest.approx(
        bounce_harmonic_detuning(5.0e4, 1.2e3, 0.7e3, l=0, n=-1))
    with pytest.raises(ValueError):
        ntv_precession_frequency(np.nan, 1.0)


def test_torque_vanishes_for_ambipolar_fluxes_and_has_the_jxb_sign():
    Z = np.array([1.0, -1.0])
    assert nonambipolar_torque_density(Z, np.array([2.0e19, 2.0e19]), -0.5) == 0.0
    # outward ion flux, current along +phi (dpsi/dV < 0): torque along +phi
    torque = nonambipolar_torque_density(Z, np.array([1.0e19, 0.0]), -0.5)
    assert torque > 0.0
    assert torque == pytest.approx(0.5 * QE * 1.0e19)
    # species along axis 0, radius along axis 1
    flux = np.array([[1.0e19, 2.0e19], [0.0, 0.0]])
    np.testing.assert_allclose(nonambipolar_torque_density(Z, flux, np.array([-0.5, -1.0])),
                               [0.5 * QE * 1e19, 1.0 * QE * 2e19])
    with pytest.raises(ValueError):
        nonambipolar_torque_density(Z, np.array([1.0, 2.0, 3.0]), -0.5)
