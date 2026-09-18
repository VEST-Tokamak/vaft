"""Romero's balance over a sampled discharge (#781, child B machinery).

The histories are analytic, so every derivative and integral has a reference.
What is pinned is the docstring's claim about what each residual measures:
with the resistance consistent they all vanish to differencing error, with it
wrong they all move by exactly that error, and the two boundary-flux budgets
agree whatever the resistance is.
"""

import numpy as np
import pytest

from vaft.process.equilibrium import romero_flux_balance

T = np.linspace(0.0, 1.0, 4001)


# Both fluxes fall, psi_B faster: a transformer driving a positive current,
# with R_p = -(dpsi_B + dpsi_C)/(2 I_p) - L_i dI_p/(2 I_p) > 0 throughout.
def psi_c(t):
    return 0.05 - 0.1 * t - 0.01 * t**2


def psi_b(t):
    return 0.03 - 0.15 * t + 0.005 * np.sin(3.0 * t)


def i_p(t):
    return 1.0e5 * (1.0 + 0.4 * t)


def _analytic(t):
    """L_i, V_B, V_C and V_I from the closed forms, for a consistent R_p."""
    ip, dip = i_p(t), 4.0e4
    l_i = (psi_c(t) - psi_b(t)) / ip
    dpsi_c = -0.1 - 0.02 * t
    dpsi_b = -0.15 + 0.015 * np.cos(3.0 * t)
    dl_i = ((dpsi_c - dpsi_b) * ip - (psi_c(t) - psi_b(t)) * dip) / ip**2
    v_b, v_c = -dpsi_b, -dpsi_c
    v_i = l_i * dip + 0.5 * ip * dl_i
    return l_i, v_b, v_c, v_i


def _consistent_resistance():
    # Romero's eq. (23), V_B = V_R + V_I, solved for R_p with I_ni = 0.
    _, v_b, _, v_i = _analytic(T)
    resistance = (v_b - v_i) / i_p(T)
    assert np.all(resistance > 0.0)
    return resistance


def _run(r_p, i_ni=0.0, psi_sign=1.0):
    return romero_flux_balance(T, i_p(T), psi_sign * psi_b(T), psi_sign * psi_c(T), r_p, i_ni)


def test_with_a_consistent_resistance_every_residual_is_differencing_error():
    out = _run(_consistent_resistance())
    interior = slice(5, -5)  # the end samples use one-sided differences
    l_i, v_b, v_c, v_i = _analytic(T)
    np.testing.assert_allclose(out["L_i"], l_i, rtol=1e-12)
    np.testing.assert_allclose(out["V_B"][interior], v_b[interior], rtol=1e-6, atol=1e-9)
    scale = np.max(np.abs(v_b))
    assert np.max(np.abs(out["balance_residual"][interior])) < 1e-5 * scale
    for key in ("V_C_from_L_i", "V_C_from_I_p"):
        assert np.max(np.abs(out[key][interior] - out["V_C"][interior])) < 1e-5 * scale
    assert abs(out["Phi_closure"][-1]) < 1e-5 * np.max(np.abs(out["Phi_B"]))


def test_a_wrong_resistance_moves_every_residual_by_exactly_its_error():
    r_true = _consistent_resistance()
    good, bad = _run(r_true), _run(1.2 * r_true)
    error = 0.2 * r_true * i_p(T)  # the extra resistive voltage
    np.testing.assert_allclose(
        bad["balance_residual"] - good["balance_residual"], -error, rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(bad["V_C_from_L_i"] - good["V_C_from_L_i"], error, rtol=1e-12)
    np.testing.assert_allclose(bad["V_C_from_I_p"] - good["V_C_from_I_p"], 2.0 * error, rtol=1e-12)


def test_the_boundary_budget_matches_the_direct_flux_change_whatever_the_physics():
    out = _run(0.0)  # a resistance that makes no physical sense
    assert np.max(np.abs(out["Phi_B"] - out["Phi_B_direct"])) < 1e-6 * np.max(
        np.abs(out["Phi_B_direct"])
    )


def test_non_inductive_current_enters_only_through_the_resistive_voltage():
    r_true = _consistent_resistance()
    out = _run(r_true, i_ni=2.0e4)
    np.testing.assert_allclose(out["V_R"], r_true * (i_p(T) - 2.0e4), rtol=1e-12)


def test_an_inverted_flux_convention_is_refused():
    with pytest.raises(ValueError, match="Romero"):
        _run(1e-6, psi_sign=-1.0)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda a: {**a, "time": a["time"][::-1]}, "increasing"),
        (lambda a: {**a, "time": np.r_[a["time"][:2], a["time"][1:-1]]}, "increasing"),
        (lambda a: {**a, "I_p": a["I_p"][:-1]}, "shape"),
        (lambda a: {**a, "I_p": np.where(np.arange(T.size) == 7, 0.0, a["I_p"])}, "non-zero"),
        (lambda a: {**a, "I_p": np.where(np.arange(T.size) > 2000, -a["I_p"], a["I_p"])},
         "changes sign"),
        (lambda a: {**a, "R_p": np.ones(3)}, "match time"),
        (lambda a: {k: v[:2] if isinstance(v, np.ndarray) and v.shape == T.shape else v
                    for k, v in a.items()}, "three"),
    ],
    ids=["reversed", "duplicate_time", "short_series", "zero_current", "current_reversal",
         "bad_R_p", "two_samples"],
)
def test_a_malformed_history_is_refused(mutate, match):
    base = {"time": T, "I_p": i_p(T), "psi_boundary": psi_b(T),
            "psi_equilibrium": psi_c(T), "R_p": 1e-6, "I_ni": 0.0}
    with pytest.raises(ValueError, match=match):
        romero_flux_balance(**mutate(base))
