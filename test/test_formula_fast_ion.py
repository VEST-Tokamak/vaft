"""Classical fast-ion slowing down and multi-species collision times (#1606): limits, coefficients, closed forms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from vaft.formula.fast_ion import (
    critical_energy_from_T_e_A_b_n_species,
    critical_velocity_from_T_e_n_species,
    fast_ion_density_from_source,
    fast_ion_energy_density_from_source,
    fast_ion_pressure_from_energy_density,
    slowing_down_distribution,
    slowing_down_time_between_speeds,
    slowing_down_time_from_T_e_n_e_A_b_Z_b,
)
from vaft.formula.kinetic import (
    electron_collision_time_from_T_e_n_species,
    electron_ion_energy_exchange_time_from_T_e_n_species,
)

AMU = 1.66053906660e-27
NE = 1e19


def test_critical_energy_is_the_14_8_law():
    # pure deuterium, deuterium beam: 14.8 A_b T_e (1/A)^{2/3}, to the rounding of 14.8
    ec = critical_energy_from_T_e_A_b_n_species(100.0, 2.0, NE, [NE], [1.0], [2.0])
    assert ec == pytest.approx(14.8 * 2 * 100.0 * 0.5 ** (2 / 3), rel=3e-3)
    # linear in T_e
    assert critical_energy_from_T_e_A_b_n_species(200.0, 2.0, NE, [NE], [1.0], [2.0]) == pytest.approx(2 * ec)
    # quasi-neutral C6+ in D leaves sum n Z^2/A = n_e/2 (Z^2/A = Z/2 for both): E_c unchanged
    with_c = critical_energy_from_T_e_A_b_n_species(100.0, 2.0, NE, [0.8 * NE, NE / 30], [1.0, 6.0], [2.0, 12.0])
    assert with_c == pytest.approx(ec, rel=1e-12)
    # in hydrogen, and with partly ionised carbon in D, it falls
    h = critical_energy_from_T_e_A_b_n_species(100.0, 2.0, NE, [NE], [1.0], [1.0])
    h_c = critical_energy_from_T_e_A_b_n_species(100.0, 2.0, NE, [0.8 * NE, NE / 30], [1.0, 6.0], [1.0, 12.0])
    c4 = critical_energy_from_T_e_A_b_n_species(100.0, 2.0, NE, [0.8 * NE, NE / 20], [1.0, 4.0], [2.0, 12.0])
    assert h_c < h and c4 < ec


def test_critical_velocity_does_not_depend_on_the_beam():
    v = critical_velocity_from_T_e_n_species(100.0, NE, [NE], [1.0], [2.0])
    ec = critical_energy_from_T_e_A_b_n_species(100.0, 1.0, NE, [NE], [1.0], [2.0])
    assert 0.5 * AMU * v**2 / 1.602176634e-19 == pytest.approx(ec)


def test_slowing_down_time_matches_wesson():
    tau = slowing_down_time_from_T_e_n_e_A_b_Z_b(100.0, 1e19, 2.0, 1.0, 15.0)
    assert tau == pytest.approx(6.27e14 * 2 * 100.0**1.5 / (1e19 * 15.0), rel=2e-3)
    assert slowing_down_time_from_T_e_n_e_A_b_Z_b(100.0, 1e19, 2.0, 2.0, 15.0) == pytest.approx(tau / 4)


def test_electron_collision_time_matches_wesson_and_sums_species():
    tau = electron_collision_time_from_T_e_n_species(1000.0, [1e20], [1.0], 17.0)
    assert tau == pytest.approx(1.09e16 / (1e20 * 17.0), rel=3e-3)
    # sum n Z^2: a carbon fraction equals more hydrogen with the same sum
    mix = electron_collision_time_from_T_e_n_species(1000.0, [0.8e20, 1e20 / 30], [1.0, 6.0], 17.0)
    assert mix == pytest.approx(electron_collision_time_from_T_e_n_species(1000.0, [2e20], [1.0], 17.0))


def test_energy_exchange_carries_one_over_mass():
    tau_e = electron_collision_time_from_T_e_n_species(100.0, [1e19], [1.0], 15.0)
    tau_eq = electron_ion_energy_exchange_time_from_T_e_n_species(100.0, [1e19], [1.0], [2.0], 15.0)
    assert tau_eq == pytest.approx(tau_e * 2.0 * AMU / (2.0 * 9.10938356e-31))
    heavy = electron_ion_energy_exchange_time_from_T_e_n_species(100.0, [1e19 / 36], [6.0], [12.0], 15.0)
    assert heavy == pytest.approx(6.0 * tau_eq)       # same sum n Z^2, six times the mass


def test_distribution_integrates_to_the_closed_forms():
    s, tau, vb, vc, a_b = 1e20, 1e-3, 3e6, 1e6, 2.0
    n_q = quad(lambda v: slowing_down_distribution(v, s, tau, vb, vc) * 4 * np.pi * v**2, 0, vb)[0]
    w_q = quad(lambda v: 0.5 * a_b * AMU * v**2 * slowing_down_distribution(v, s, tau, vb, vc) * 4 * np.pi * v**2, 0, vb)[0]
    assert fast_ion_density_from_source(s, tau, vb, vc) == pytest.approx(n_q, rel=1e-10)
    assert fast_ion_energy_density_from_source(s, tau, a_b, vb, vc) == pytest.approx(w_q, rel=1e-10)
    assert slowing_down_distribution(1.01 * vb, s, tau, vb, vc) == 0.0


def test_limits_of_the_slowing_down_chain():
    s, tau, a_b = 1e20, 1e-3, 2.0
    vb = 3e6
    e_b = 0.5 * a_b * AMU * vb**2
    # v_b >> v_c: electron drag only, W -> S tau_s E_b / 2 and n -> S tau_s ln(v_b/v_c)
    w = fast_ion_energy_density_from_source(s, tau, a_b, vb, vb / 1e3)
    assert w == pytest.approx(s * tau * e_b / 2, rel=1e-5)
    # the time to slow to rest is n_f / S
    t = slowing_down_time_between_speeds(tau, 1e6, vb)
    assert fast_ion_density_from_source(s, tau, vb, 1e6) == pytest.approx(s * t)
    assert slowing_down_time_between_speeds(tau, 1e6, vb, vb) == 0.0
    assert fast_ion_pressure_from_energy_density(3.0) == pytest.approx(2.0)
    # v_b << v_c: ion drag only, W -> S tau_s E_b (v_b/v_c)^3 / 5, no cancellation error
    for x in (1e-4, 1e-2, 0.049, 0.051):
        w = fast_ion_energy_density_from_source(s, tau, a_b, vb, vb / x)
        assert w == pytest.approx(s * tau * e_b * (x**5 / 5 - x**8 / 8) / x**2, rel=1e-7)


@pytest.mark.parametrize("bad", [
    lambda: critical_velocity_from_T_e_n_species(-1.0, NE, [NE], [1.0], [2.0]),
    lambda: critical_velocity_from_T_e_n_species(100.0, NE, NE, 1.0, 2.0),
    lambda: slowing_down_time_between_speeds(1e-3, 1e6, 1e6, 2e6),
    lambda: electron_collision_time_from_T_e_n_species(100.0, [0.0], [1.0], 15.0),
    lambda: fast_ion_pressure_from_energy_density(-1.0),
])
def test_invalid_input_is_refused(bad):
    with pytest.raises(ValueError):
        bad()
