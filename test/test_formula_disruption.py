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
    E_c = 0.1
    tau = relativistic_collision_time(1e20, 15.0) * connor_hastie_critical_field(1e20, 15.0) / E_c
    assert avalanche_growth_rate(0.05, E_c, 1.0, 15.0) == 0.0
    g = [avalanche_growth_rate(E, E_c, 1.0, 15.0) for E in (0.2, 0.3)]
    assert g[1] == pytest.approx(2 * g[0])
    assert g[0] == pytest.approx(1 / (tau * 15.0) * math.sqrt(math.pi / 18.0))
    # more ion charge scatters more: slower avalanche at the same field
    assert avalanche_growth_rate(0.3, E_c, 5.0, 15.0) < avalanche_growth_rate(0.3, E_c, 1.0, 15.0)


def test_the_avalanche_efolds_are_the_rosenbluth_putvinski_current_scaling():
    # with L_p = mu0 R0 X: N = 2 X dI / (I_A ln Lambda) sqrt(pi / 3(Z + 5)), I_A = 4 pi eps0 m_e c^3 / e ~ 17 kA
    from vaft.formula.constants import MU0

    R0, a, l_i, Z, lnL, dI = 6.2, 2.0, 0.8, 1.0, 15.0, 15e6
    X = math.log(8 * R0 / a) - 2 + l_i / 2
    I_A = 4 * math.pi * EPS0 * ME * C_LIGHT**3 / QE
    assert I_A == pytest.approx(17.05e3, rel=1e-3)
    expected = 2 * X * dI / (I_A * lnL) * math.sqrt(math.pi / (3 * (Z + 5)))
    assert avalanche_efolds_from_current_drop(dI, MU0 * R0 * X, R0, Z, lnL) == pytest.approx(expected, rel=1e-6)
    assert expected > 20  # tens of e-folds at reactor current
    with pytest.raises(ValueError):
        avalanche_efolds_from_current_drop(-1.0, 1e-6, R0, Z, lnL)


def test_a_decaying_current_induces_a_field_along_it():
    assert inductive_parallel_electric_field(2e-6, -1e8, 1.0) == pytest.approx(2e-6 * 1e8 / (2 * math.pi))
    assert inductive_parallel_electric_field(2e-6, 1e8, 1.0) < 0


def test_dreicer_rises_with_field_falls_with_charge_and_needs_its_prefactor():
    n, T, lnL = 5e19, 100.0, 12.0
    E_D = dreicer_field(n, T, lnL)
    rates = [dreicer_generation_rate(n, T, f * E_D, 1.0, lnL, prefactor=1.0) for f in (0.02, 0.04, 0.08)]
    assert 0.0 < rates[0] < rates[1] < rates[2]
    assert rates[0] / rates[1] < 0.1  # doubling E from 2 % to 4 % of E_D: exponentially sensitive
    assert dreicer_generation_rate(n, T, 0.05 * E_D, 3.0, lnL, prefactor=1.0) < dreicer_generation_rate(
        n, T, 0.05 * E_D, 1.0, lnL, prefactor=1.0)
    assert dreicer_generation_rate(n, T, 0.0, 1.0, lnL, prefactor=1.0) == 0.0
    assert dreicer_generation_rate(n, T, 0.05 * E_D, 1.0, lnL, prefactor=0.35) == pytest.approx(
        0.35 * dreicer_generation_rate(n, T, 0.05 * E_D, 1.0, lnL, prefactor=1.0))
    with pytest.raises(TypeError):
        dreicer_generation_rate(n, T, 0.05 * E_D, 1.0, lnL)  # no hidden prefactor


def test_the_dreicer_exponent_at_one_point_by_hand():
    # the exponent at E = E_D/25, Z = 1: -25/4 - sqrt(2*25), and the power (25)^(6/16)
    n, T, lnL = 1e19, 50.0, 13.0
    E_D = dreicer_field(n, T, lnL)
    v_te = math.sqrt(2 * QE * T / ME)
    nu = n * QE**4 * lnL / (4 * math.pi * EPS0**2 * ME**2 * v_te**3)
    by_hand = n * nu * 25 ** (6 / 16) * math.exp(-25 / 4 - math.sqrt(50))
    assert dreicer_generation_rate(n, T, E_D / 25, 1.0, lnL, prefactor=1.0) == pytest.approx(by_hand, rel=1e-12)


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
