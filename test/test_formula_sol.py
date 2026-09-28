"""Scrape-off-layer formulas (#951): closures explicit, sheath bookkeeping, conduction, Eich profile."""

import numpy as np
import pytest

from vaft.formula.constants import QE
from vaft.formula.sol import (
    KAPPA_0E,
    eich_integral_width,
    eich_target_heat_flux_profile,
    ion_saturation_current_density,
    ion_sound_speed,
    sheath_heat_flux,
    sheath_particle_flux,
    spitzer_harm_parallel_heat_flux,
    two_point_upstream_temperature,
)
from vaft.formula.stability import c_s_from_Te_Ti_mi

MD = 3.3435837724e-27


def test_sound_speed_generalises_the_isothermal_one():
    # Z = gamma_i = 1 in eV is the stability module's keV form
    assert ion_sound_speed(20.0, 30.0, MD) == pytest.approx(c_s_from_Te_Ti_mi(0.02, 0.03, MD))
    iso, adiabatic = ion_sound_speed(20.0, 20.0, MD), ion_sound_speed(20.0, 20.0, MD, gamma_i=3.0)
    assert adiabatic / iso == pytest.approx(np.sqrt(2.0))
    assert ion_sound_speed(20.0, 0.0, MD, Z=2.0) == pytest.approx(np.sqrt(2.0) * ion_sound_speed(20.0, 0.0, MD))
    for bad in ((-1.0, 10.0, MD), (10.0, 10.0, 0.0)):
        with pytest.raises(ValueError):
            ion_sound_speed(*bad)


def test_sheath_fluxes_share_one_particle_flux():
    n, c_s = 1e19, ion_sound_speed(15.0, 15.0, MD)
    gamma_t = sheath_particle_flux(n, c_s)
    assert gamma_t == pytest.approx(n * c_s)
    assert sheath_particle_flux(n, c_s, mach=1.5) == pytest.approx(1.5 * gamma_t)
    assert ion_saturation_current_density(n, c_s) == pytest.approx(QE * gamma_t)
    # the heat flux is gamma T_e per particle
    q = sheath_heat_flux(7.0, n, 15.0, c_s)
    assert q / gamma_t == pytest.approx(7.0 * 15.0 * QE)
    with pytest.raises(ValueError):
        sheath_heat_flux(0.0, n, 15.0, c_s)


def test_conduction_integrates_to_the_two_point_relation():
    # integrate dT/ds = -q / (kappa_0 T^{5/2}) from the target back to upstream at constant q
    q, L, T_t = 5e7, 15.0, 8.0
    s = np.linspace(L, 0.0, 200001)
    T = np.empty_like(s)
    T[0] = T_t
    for i in range(1, s.size):
        ds = s[i] - s[i - 1]  # negative: walking upstream
        # midpoint rule on dT/ds = -q / (kappa_0 T^2.5)
        half = T[i - 1] - 0.5 * ds * q / (KAPPA_0E * T[i - 1] ** 2.5)
        T[i] = T[i - 1] - ds * q / (KAPPA_0E * half**2.5)
    assert T[-1] == pytest.approx(two_point_upstream_temperature(T_t, q, L), rel=1e-6)
    # and the local flux of that profile is q everywhere
    dT = np.gradient(T, s)
    np.testing.assert_allclose(spitzer_harm_parallel_heat_flux(T, dT)[10:-10], q, rtol=1e-4)


def test_upstream_temperature_is_robust_to_the_heat_flux():
    # the 2/7 power: doubling q L raises T_u by 2^{2/7} once T_t is negligible
    T1 = two_point_upstream_temperature(0.0, 1e8, 20.0)
    T2 = two_point_upstream_temperature(0.0, 2e8, 20.0)
    assert T2 / T1 == pytest.approx(2.0 ** (2.0 / 7.0))
    assert two_point_upstream_temperature(12.0, 0.0, 20.0) == pytest.approx(12.0)


def test_eich_profile_limits_and_integral_width():
    s = np.linspace(-0.05, 0.3, 200001)
    lam, S, fx = 0.004, 1e-5, 4.0
    q = eich_target_heat_flux_profile(s, 2.0, lam, S, flux_expansion=fx)
    # S -> 0: the unspread exponential, q0 exp(-s / lambda_q f_x) on the SOL side, zero in the private flux
    sol_side = s > 5 * S
    np.testing.assert_allclose(q[sol_side], 2.0 * np.exp(-s[sol_side] / (lam * fx)), rtol=1e-6)
    assert np.all(q[s < -5 * S] < 1e-12)
    # the Makowski integral width is within a few per cent of the exact integral
    for lam, S, fx in ((0.003, 0.001, 5.0), (0.005, 0.002, 4.0)):
        q = eich_target_heat_flux_profile(s, 1.0, lam, S, flux_expansion=fx)
        exact = np.trapezoid(q, s) / q.max() / fx
        assert eich_integral_width(lam, S, flux_expansion=fx) == pytest.approx(exact, rel=0.03)
    # background and strike point
    shifted = eich_target_heat_flux_profile(s + 0.01, 1.0, 0.003, 0.001, s0=0.01, q_bg=0.2)
    np.testing.assert_allclose(shifted, eich_target_heat_flux_profile(s, 1.0, 0.003, 0.001) + 0.2)
    with pytest.raises(ValueError):
        eich_target_heat_flux_profile(s, 1.0, 0.0, 0.001)
