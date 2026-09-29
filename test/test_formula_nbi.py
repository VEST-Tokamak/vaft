"""NBI formulas and the reduced attenuation process (#1136): limits, conservation and refusals."""

import numpy as np
import pytest

from vaft.formula.constants import QE
from vaft.formula.nbi import (
    beam_birth_probability_density,
    beam_particle_rate_from_power_energy,
    injected_toroidal_angular_momentum_rate,
    neutral_beam_optical_depth,
    neutral_survival_fraction_from_optical_depth,
    shine_through_fraction,
)
from vaft.process.nbi import neutral_beam_attenuation_along_path

S = np.linspace(0.0, 1.2, 1201)


def test_particle_rate_is_power_over_energy_per_component():
    assert beam_particle_rate_from_power_energy(1e6, 2e4) == pytest.approx(1e6 / (2e4 * QE))
    # a half-energy component at the same power injects twice the particles
    full, half = beam_particle_rate_from_power_energy(1e6, np.array([2e4, 1e4]))
    assert half == pytest.approx(2 * full)
    for bad in ((-1.0, 2e4), (1e6, 0.0), (np.inf, 2e4)):
        with pytest.raises(ValueError):
            beam_particle_rate_from_power_energy(*bad)


def test_uniform_attenuation_is_exact_exponential_decay():
    alpha = np.full_like(S, 2.0)
    tau = neutral_beam_optical_depth(S, alpha)
    np.testing.assert_allclose(tau, 2.0 * S, atol=1e-12)
    np.testing.assert_allclose(neutral_survival_fraction_from_optical_depth(tau), np.exp(-2.0 * S))
    assert shine_through_fraction(S, alpha) == pytest.approx(np.exp(-2.4))
    np.testing.assert_allclose(beam_birth_probability_density(S, alpha), 2.0 * np.exp(-2.0 * S))


def test_zero_depth_shines_through_and_more_depth_shines_less():
    assert shine_through_fraction(S, np.zeros_like(S)) == 1.0
    fractions = [shine_through_fraction(S, np.full_like(S, a)) for a in (0.5, 1.0, 2.0, 4.0)]
    assert np.all(np.diff(fractions) < 0)


def test_births_plus_shine_through_is_one():
    alpha = 3.0 * np.exp(-((S - 0.6) / 0.3) ** 2)
    total = np.trapezoid(beam_birth_probability_density(S, alpha), S) + shine_through_fraction(S, alpha)
    assert total == pytest.approx(1.0, abs=1e-5)


def test_leading_axes_are_independent_beams():
    alpha = np.stack([np.full_like(S, 1.0), np.full_like(S, 2.0)])
    np.testing.assert_allclose(shine_through_fraction(S, alpha), np.exp(-np.array([1.2, 2.4])))


def test_invalid_paths_and_coefficients_are_refused():
    with pytest.raises(ValueError, match="increase"):
        neutral_beam_optical_depth(S[::-1], np.ones_like(S))
    with pytest.raises(ValueError, match="non-negative"):
        neutral_beam_optical_depth(S, -np.ones_like(S))
    with pytest.raises(ValueError):
        neutral_beam_optical_depth(S, np.ones(5))
    with pytest.raises(ValueError):
        neutral_survival_fraction_from_optical_depth(-0.1)


def test_angular_momentum_rate_follows_the_tangency_radius_sign():
    m, v = 3.344e-27, 1.4e6
    co = injected_toroidal_angular_momentum_rate(1e20, m, 0.3, v)
    assert co == pytest.approx(1e20 * m * 0.3 * v)
    assert injected_toroidal_angular_momentum_rate(1e20, m, -0.3, v) == pytest.approx(-co)
    with pytest.raises(ValueError):
        injected_toroidal_angular_momentum_rate(1e20, 0.0, 0.3, v)


def test_process_conserves_particles_and_power_on_any_grid():
    for n in (3, 5, 1201):
        s = np.linspace(0.0, 1.0, n)
        alpha = np.full(n, 5.0)  # alpha * ds up to 2.5: a grid far too coarse for the point density
        r = neutral_beam_attenuation_along_path(s, alpha, beam_power=1.5e6, beam_energy_eV=2.5e4)
        width = np.diff(s)
        assert r.birth_fraction_per_cell.sum() + r.shine_through_fraction == pytest.approx(1.0, abs=1e-14)
        assert np.sum(r.power_birth_profile * width) + r.shine_through_power == pytest.approx(1.5e6, rel=1e-14)
        assert np.sum(r.birth_rate_density * width) == pytest.approx(r.particle_rate * r.absorbed_neutral_fraction,
                                                                     rel=1e-14)
        assert np.all(r.birth_fraction_per_cell >= 0.0)
        assert r.survival_fraction[0] == 1.0 and np.all(np.diff(r.survival_fraction) <= 0)
    # on a fine grid the cell density converges to the formula's point density at the cell centres
    s = np.linspace(0.0, 1.2, 1201)
    alpha = 2.5 * np.clip(1.0 - (2.0 * s / 1.2 - 1.0) ** 2, 0.0, None)
    r = neutral_beam_attenuation_along_path(s, alpha)
    point = np.interp(r.cell_centers, s, beam_birth_probability_density(s, alpha))
    np.testing.assert_allclose(r.birth_fraction_density, point, atol=2e-5)


def test_process_without_power_has_no_bookkeeping_and_refuses_bad_input():
    s = np.linspace(0.0, 1.0, 11)
    r = neutral_beam_attenuation_along_path(s, np.ones_like(s))
    assert r.particle_rate is None and r.power_birth_profile is None
    with pytest.raises(ValueError, match="both"):
        neutral_beam_attenuation_along_path(s, np.ones_like(s), beam_power=1e6)
    with pytest.raises(ValueError):
        neutral_beam_attenuation_along_path(s, np.ones((2, s.size)))
    for power in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            neutral_beam_attenuation_along_path(s, np.ones_like(s), beam_power=power, beam_energy_eV=2e4)
    with pytest.raises(ValueError):
        neutral_beam_attenuation_along_path(s, np.full_like(s, np.nan))


def test_formula_refusals():
    with pytest.raises(ValueError):
        neutral_beam_optical_depth(S, 1.0)  # a scalar alpha is not a profile
    with pytest.raises(ValueError):
        neutral_survival_fraction_from_optical_depth(np.inf)
    with pytest.raises(ValueError):
        beam_particle_rate_from_power_energy(1e6, np.nan)
    with pytest.raises(ValueError):
        injected_toroidal_angular_momentum_rate(1e20, 3.3e-27, np.inf, 1e6)
    with pytest.raises(ValueError):
        injected_toroidal_angular_momentum_rate(1e20, 3.3e-27, 0.3, -1.0)
    assert np.all(beam_birth_probability_density(S, np.full_like(S, 2.0)) >= 0.0)
    assert isinstance(neutral_survival_fraction_from_optical_depth(0.5), float)
