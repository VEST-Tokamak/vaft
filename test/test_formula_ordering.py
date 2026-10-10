"""Asymptotic ordering parameters (#1627): values against the NRL formulary and their identities."""

import math

import numpy as np
import pytest

from vaft.formula import ordering
from vaft.formula.constants import ME, MI_P, QE
from vaft.formula.ordering import (
    alfven_time,
    braginskii_electron_collision_time,
    braginskii_ion_collision_time,
    evolution_time,
    inertial_length,
    knudsen_number,
    lundquist_number,
    magnetic_reynolds_number,
    magnetization,
    mean_free_path,
    resistive_diffusion_time,
    sound_gyroradius,
    thermal_speed,
)
from vaft.formula.particle import gyrofrequency


def test_lundquist_is_resistive_over_alfven_time_and_scales_with_length():
    L, v_A, eta = 0.3, 2.0e6, 1.0e-6
    S = lundquist_number(L, v_A, eta)
    assert S == pytest.approx(resistive_diffusion_time(L, eta) / alfven_time(L, v_A), rel=1e-12)
    assert lundquist_number(0.01 * L, v_A, eta) == pytest.approx(0.01 * S)  # a layer S is not the global S
    assert magnetic_reynolds_number(v_A, L, eta) == pytest.approx(S)


def test_inertial_lengths_match_the_nrl_formulary():
    n = 1.0e20  # 1e14 cm^-3
    # NRL: c/omega_pi = 2.28e7 mu^(1/2) n^(-1/2) cm, c/omega_pe = 5.31e5 n^(-1/2) cm
    assert inertial_length(n, MI_P) == pytest.approx(2.28e7 / 1e7 * 1e-2, rel=2e-3)
    assert inertial_length(n, ME) == pytest.approx(5.31e5 / 1e7 * 1e-2, rel=2e-3)
    # deuterium: sqrt(2) longer
    assert inertial_length(n, 2 * MI_P) / inertial_length(n, MI_P) == pytest.approx(np.sqrt(2.0))


def test_sound_gyroradius_is_c_s_over_omega_i():
    T_e, m_i, B = 50.0, 2 * MI_P, 0.15
    c_s = np.sqrt(T_e * QE / m_i)
    omega_i = abs(gyrofrequency(QE, m_i, B))
    assert sound_gyroradius(T_e, m_i, B) == pytest.approx(c_s / omega_i, rel=1e-12)


def test_thermal_speed_matches_the_nrl_formulary():
    # NRL: v_Te = 4.19e7 T^(1/2) cm/s (T in eV)
    assert thermal_speed(100.0, ME) == pytest.approx(4.19e7 * 10.0 * 1e-2, rel=1e-3)


def test_braginskii_collision_times():
    assert braginskii_electron_collision_time(1e19, 100.0, 15.0) == pytest.approx(3.44e11 * 1e3 / 1.5e20)
    assert braginskii_electron_collision_time(1e19, 100.0, 15.0, Z=2.0) == pytest.approx(
        braginskii_electron_collision_time(1e19, 100.0, 15.0) / 2.0)
    tau_i = braginskii_ion_collision_time(1e19, 100.0, 15.0)
    assert tau_i == pytest.approx(2.09e13 * 1e3 / 1.5e20)
    assert braginskii_ion_collision_time(1e19, 100.0, 15.0, mass_number=2.0) == pytest.approx(np.sqrt(2.0) * tau_i)
    # T^(3/2)
    assert braginskii_electron_collision_time(1e19, 400.0, 15.0) == pytest.approx(
        8.0 * braginskii_electron_collision_time(1e19, 100.0, 15.0))


def test_charge_dependences():
    tau_i = braginskii_ion_collision_time(1e19, 100.0, 15.0)
    assert braginskii_ion_collision_time(1e19, 100.0, 15.0, Z=2.0) == pytest.approx(tau_i / 16.0)  # Z^4
    rho = sound_gyroradius(50.0, 2 * MI_P, 0.15)
    assert sound_gyroradius(50.0, 2 * MI_P, 0.15, Z=2.0) == pytest.approx(rho / np.sqrt(2.0))


def test_knudsen_and_magnetization_compose_from_the_kernels():
    n, T, B, lnL = 1e19, 100.0, 0.1, 15.0
    tau_e = braginskii_electron_collision_time(n, T, lnL)
    lam = mean_free_path(thermal_speed(T, ME), tau_e)
    assert knudsen_number(lam, 0.1) == pytest.approx(lam / 0.1)
    chi = magnetization(gyrofrequency(-QE, ME, B), tau_e)
    assert chi > 1e3  # electrons are strongly magnetized at these parameters
    assert chi == pytest.approx(QE * B / ME * tau_e)


def test_evolution_time_is_positive_and_infinite_when_stationary():
    out = evolution_time(np.array([100e3, 100e3, -2.0]), np.array([1e6, 0.0, 4.0]))
    np.testing.assert_allclose(out, [0.1, np.inf, 0.5])


@pytest.mark.parametrize("call", [
    lambda: alfven_time(0.0, 1.0),
    lambda: resistive_diffusion_time(1.0, -1.0),
    lambda: lundquist_number(1.0, np.nan, 1.0),
    lambda: magnetic_reynolds_number(-1.0, 1.0, 1.0),
    lambda: inertial_length(1e19, MI_P, q=0.0),
    lambda: sound_gyroradius(10.0, MI_P, 0.0),
    lambda: thermal_speed(-1.0, ME),
    lambda: braginskii_ion_collision_time(1e19, 10.0, 0.0),
    lambda: knudsen_number(1.0, 0.0),
    lambda: magnetization(0.0, 1.0),
    lambda: evolution_time(np.inf, 1.0),
])
def test_bad_inputs_raise(call):
    with pytest.raises(ValueError):
        call()


def test_debye_length_matches_the_nrl_formulary():
    # NRL: lambda_De = 7.43e2 T^1/2 n^-1/2 cm with n in cm^-3
    assert ordering.debye_length(1e19, 50.0) == pytest.approx(7.43e2 * math.sqrt(50.0 / 1e13) * 1e-2, rel=1e-3)
    # quadrupling the density halves it
    assert ordering.debye_length(4e19, 50.0) == pytest.approx(0.5 * ordering.debye_length(1e19, 50.0))
    with pytest.raises(ValueError):
        ordering.debye_length(0.0, 50.0)


def test_mach_number_is_the_flow_magnitude_over_the_reference_speed():
    assert ordering.mach_number(-3.0e4, 6.0e4) == pytest.approx(0.5)
    v_ti = ordering.thermal_speed(100.0, MI_P)
    assert ordering.mach_number(v_ti, v_ti) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ordering.mach_number(1.0, 0.0)


def test_pressure_anisotropy_is_relative_to_the_scalar_pressure():
    assert ordering.pressure_anisotropy(2.0, 2.0) == 0.0
    # p = (2 p_perp + p_par)/3: p_perp = 2, p_par = 1 gives p = 5/3 and Delta = 3/5
    assert ordering.pressure_anisotropy(2.0, 1.0) == pytest.approx(0.6)
    assert ordering.pressure_anisotropy(1.0, 2.0) == pytest.approx(-0.75)
    # the bounds: p_par -> 0 gives 3/2, p_perp -> 0 gives -3
    assert ordering.pressure_anisotropy(1.0, 1e-12) == pytest.approx(1.5)
    assert ordering.pressure_anisotropy(1e-12, 1.0) == pytest.approx(-3.0)
