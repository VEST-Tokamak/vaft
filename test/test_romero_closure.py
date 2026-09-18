"""Romero's first-order closure (#781, child C): rates and integration.

Two of the three state equations are exact identities already in
vaft.formula.transformer, so the rates are pinned against those; the third is
the closure, pinned by its exact k = 0 solution.  The integrator is pinned
against the independent balance evaluator: a closure trajectory, turned back
into flux histories, must satisfy romero_flux_balance to differencing error.
"""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from vaft.formula.transformer import (
    internal_inductance_rate_from_I_p_V_R_V_C,
    plasma_current_rate_from_L_i_V_B_V_C_V_R,
    romero_closure_rates_from_I_p_L_i_V_CB_V_B_V_R_k_tau,
)
from vaft.process.equilibrium import integrate_romero_closure, romero_flux_balance

STATE = dict(I_p_A=1.2e5, L_i_H=1.3e-7, V_CB_V=-0.4, V_B_V=2.0, V_R_V=1.1)


def test_the_current_and_inductance_rates_are_the_exact_identities():
    d_ip, d_li, _ = romero_closure_rates_from_I_p_L_i_V_CB_V_B_V_R_k_tau(**STATE, k=0.6, tau_s=0.01)
    v_c = STATE["V_B_V"] + STATE["V_CB_V"]
    assert d_ip == pytest.approx(
        plasma_current_rate_from_L_i_V_B_V_C_V_R(STATE["L_i_H"], STATE["V_B_V"], v_c, STATE["V_R_V"]),
        rel=1e-14,
    )
    assert d_li == pytest.approx(
        internal_inductance_rate_from_I_p_V_R_V_C(STATE["I_p_A"], STATE["V_R_V"], v_c), rel=1e-14
    )


def test_a_flat_loop_voltage_profile_is_the_fixed_point():
    rates = romero_closure_rates_from_I_p_L_i_V_CB_V_B_V_R_k_tau(
        1.0e5, 1.0e-7, 0.0, 1.5, 1.5, 0.7, 0.02
    )
    assert rates == (0.0, 0.0, 0.0)


T = np.linspace(0.0, 0.05, 501)


def test_with_zero_gain_the_relative_voltage_decays_exponentially():
    out = integrate_romero_closure(
        T, 1.0, 1.0e-5, 0.0, I_p0=5.0e4, L_i0=1.2e-7, V_CB0=0.3, k=0.0, tau=0.008
    )
    np.testing.assert_allclose(out["V_CB"], 0.3 * np.exp(-T / 0.008), rtol=1e-6, atol=1e-12)


def test_a_constant_drive_relaxes_to_the_ohmic_current():
    out = integrate_romero_closure(
        np.linspace(0.0, 0.5, 201), 1.0, 1.0e-5, 0.0, I_p0=5.0e4, L_i0=1.2e-7, k=0.5, tau=0.01
    )
    assert out["I_p"][-1] == pytest.approx(1.0e5, rel=1e-4)  # V_B / R_p
    assert abs(out["V_CB"][-1]) < 1e-5
    assert out["V_R"][-1] == pytest.approx(1.0, rel=1e-4)


def _ramp():
    v_b = 1.5 + 0.8 * np.sin(2 * np.pi * T / 0.03)
    r_p = 1.2e-5 * (1.0 - 3.0 * T)
    return v_b, r_p, integrate_romero_closure(
        T, v_b, r_p, 2.0e3, I_p0=8.0e4, L_i0=1.2e-7, V_CB0=0.1, k=0.4, tau=0.006
    )


def test_the_trajectory_conserves_the_internal_field_energy_balance():
    v_b, r_p, out = _ramp()
    d_ip = np.gradient(out["I_p"], T)
    d_li = np.gradient(out["L_i"], T)
    lhs = out["L_i"] * d_ip + 0.5 * out["I_p"] * d_li
    np.testing.assert_allclose(lhs[5:-5], (v_b - out["V_R"])[5:-5], rtol=0, atol=5e-4)


def test_the_balance_evaluator_accepts_a_closure_trajectory():
    # Turn the trajectory into Romero-sign flux histories and hand them to the
    # independent evaluator: psi_B = -int V_B dt, psi_C = psi_B + L_i I_p.
    v_b, r_p, out = _ramp()
    psi_b = -cumulative_trapezoid(v_b, T, initial=0.0) + 0.01
    psi_c = psi_b + out["L_i"] * out["I_p"]
    balance = romero_flux_balance(T, out["I_p"], psi_b, psi_c, r_p, 2.0e3)
    interior = slice(5, -5)
    scale = np.max(np.abs(v_b))
    assert np.max(np.abs(balance["balance_residual"][interior])) < 2e-4 * scale
    np.testing.assert_allclose(balance["V_C"][interior], out["V_C"][interior], rtol=0, atol=2e-4 * scale)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"I_p0": 0.0}, "I_p0"),
        ({"L_i0": -1e-7}, "L_i0"),
        ({"tau": 0.0}, "tau_s"),
        ({"time": T[::-1]}, "increasing"),
        ({"V_B": np.ones(3)}, "match time"),
    ],
)
def test_a_malformed_setup_is_refused(kwargs, match):
    call = dict(time=T, V_B=1.0, R_p=1e-5, I_ni=0.0, I_p0=5e4, L_i0=1.2e-7, k=0.5, tau=0.01)
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        integrate_romero_closure(**call)


def test_a_current_reversal_is_refused_where_it_happens():
    # A negative boundary voltage drives the current through zero; the solver
    # never samples exactly zero there, so the crossing must be caught as an
    # event, not left to the rate function's non-zero check.
    with pytest.raises(ValueError, match="reaches zero at t ="):
        integrate_romero_closure(
            np.linspace(0.0, 0.05, 51), -3.0, 1.0e-5, 0.0,
            I_p0=2.0e3, L_i0=1.2e-7, k=0.5, tau=0.01,
        )
