"""Disruption reference relations (#1041): the formulas evaluated against independent numbers and each other."""

import math

import numpy as np
import pytest

from vaft.formula.constants import C_LIGHT, EPS0, ME, QE
from vaft.formula.disruption import (
    avalanche_efolds_from_current_drop,
    avalanche_growth_rate,
    connor_hastie_critical_field,
    current_quench_current,
    dreicer_field,
    dreicer_generation_rate,
    inductive_parallel_electric_field,
    relativistic_collision_time,
    runaway_critical_momentum,
    runaway_current_from_density,
    thermal_quench_temperature,
)


def test_the_critical_field_is_the_textbook_value():
    # E_c ~ 0.08 V/m at 1e20 m^-3, ln Lambda = 15 (Breizman et al. 2019); linear in n_e ln Lambda
    assert connor_hastie_critical_field(1e20, 15.0) == pytest.approx(0.0765, rel=5e-3)
    assert connor_hastie_critical_field(2e20, 15.0) == pytest.approx(2 * connor_hastie_critical_field(1e20, 15.0))


def test_the_dreicer_field_is_the_critical_field_times_mc2_over_T():
    for T in (5.0, 100.0, 2000.0):
        ratio = dreicer_field(1e20, T, 15.0) / connor_hastie_critical_field(1e20, 15.0)
        assert ratio == pytest.approx(ME * C_LIGHT**2 / (QE * T), rel=1e-12)


def test_the_collision_time_is_mc_over_e_Ec():
    tau = relativistic_collision_time(5e19, 14.0)
    assert tau == pytest.approx(4 * math.pi * EPS0**2 * ME**2 * C_LIGHT**3 / (5e19 * QE**4 * 14.0))


def test_critical_momentum_diverges_at_Ec_and_falls_with_field():
    assert runaway_critical_momentum(0.5, 1.0) == math.inf
    assert runaway_critical_momentum(2.0, 1.0) == pytest.approx(1.0)
    assert runaway_critical_momentum(101.0, 1.0) == pytest.approx(0.1)


def test_the_avalanche_needs_Ec_and_is_linear_above():
    E_c, tau = 0.1, 1e-3
    assert avalanche_growth_rate(0.05, E_c, 1.0, tau, 15.0) == 0.0
    g = [avalanche_growth_rate(E, E_c, 1.0, tau, 15.0) for E in (0.2, 0.3)]
    assert g[1] == pytest.approx(2 * g[0])
    assert g[0] == pytest.approx(1 / (tau * 15.0) * math.sqrt(math.pi / 18.0))


def test_the_avalanche_efolds_equal_the_time_integral_of_the_rate():
    # an L/R current quench, E >> E_c: integrate gamma_av (E/E_c) and compare
    L, R0, I0, tau_cq, Z, lnL, n_e = 3e-6, 1.7, 1e6, 5e-3, 1.0, 15.0, 5e19
    t = np.linspace(0, 20 * tau_cq, 200001)
    I = current_quench_current(t, I0, tau_cq)
    E = inductive_parallel_electric_field(L, np.gradient(I, t), R0)
    E_c = connor_hastie_critical_field(n_e, lnL)
    rate = E / E_c / (relativistic_collision_time(n_e, lnL) * lnL) * math.sqrt(math.pi / (3 * (Z + 5)))
    integral = np.trapezoid(rate, t)
    assert avalanche_efolds_from_current_drop(I0 - I[-1], L, R0, Z, lnL) == pytest.approx(integral, rel=1e-4)
    with pytest.raises(ValueError):
        avalanche_efolds_from_current_drop(-1.0, L, R0, Z, lnL)


def test_a_decaying_current_induces_a_field_along_it():
    assert inductive_parallel_electric_field(2e-6, -1e8, 1.0) == pytest.approx(2e-6 * 1e8 / (2 * math.pi))
    assert inductive_parallel_electric_field(2e-6, 1e8, 1.0) < 0


def test_dreicer_is_exponentially_small_at_small_field_and_zero_without_field():
    n, T, Z, lnL = 5e19, 100.0, 1.0, 12.0
    E_D = dreicer_field(n, T, lnL)
    small, larger = (dreicer_generation_rate(n, T, f * E_D, Z, lnL) for f in (0.02, 0.05))
    assert 0.0 < small < 1e-4 * larger  # exp(-E_D/4E) dominates: about 2e-5 here
    assert dreicer_generation_rate(n, T, 0.0, Z, lnL) == 0.0
    assert dreicer_generation_rate(n, T, 0.05 * E_D, Z, lnL, prefactor=0.35) == pytest.approx(0.35 * larger)


def test_the_quench_models_start_and_end_where_they_should():
    assert thermal_quench_temperature(-1.0, 2000.0, 5.0, 1e-3) == 2000.0
    assert thermal_quench_temperature(0.0, 2000.0, 5.0, 1e-3) == pytest.approx(2000.0)
    assert thermal_quench_temperature(1.0, 2000.0, 5.0, 1e-3) == pytest.approx(5.0)
    assert current_quench_current(5e-3, 1e6, 5e-3) == pytest.approx(1e6 / math.e)
    with pytest.raises(ValueError):
        thermal_quench_temperature(0.0, 5.0, 2000.0, 1e-3)


def test_the_runaway_current_is_e_c_n_A():
    assert runaway_current_from_density(1e16, 1.0) == pytest.approx(QE * C_LIGHT * 1e16)
    with pytest.raises(ValueError):
        runaway_current_from_density(-1.0, 1.0)
