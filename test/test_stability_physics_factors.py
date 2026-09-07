"""Three stability kernels that dropped a factor (issues #348, #363, #364).

Each was wrong by a factor rather than a detail, and none had a single test
calling it -- which is why all three survived. The negative controls here keep
the old expressions around as the thing the fix must *not* reproduce.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.formula.constants import ME, MU0, QE
from vaft.formula.equilibrium import normalized_larmor_radius_from_M_T_a_Bt
from vaft.formula.stability import (
    ballooning_alpha_from_p_B_R,
    beta_pol_from_beta_tor,
    beta_tor_from_beta_pol,
    rhostar_from_Te_a_Bt,
)

# A VEST-like spherical tokamak: R0 = 0.4 m, a = 0.28 m, so epsilon = 0.7.
EPSILON = 0.7
Q95 = 3.0


# --- #348 -------------------------------------------------------------------

def test_rho_star_is_dimensionless_and_carries_the_right_constant():
    r"""rho_e/a with rho_e = sqrt(2 m_e e T_e)/(e B), constant sqrt(2 m_e/e)."""
    Te, a, B = 500.0, 0.28, 0.2
    expected = np.sqrt(2.0 * ME * QE * Te) / (QE * B * a)
    assert rhostar_from_Te_a_Bt(Te, a, B) == pytest.approx(expected)
    assert np.sqrt(2.0 * ME / QE) == pytest.approx(3.372e-6, rel=1e-3)


def test_rho_star_agrees_with_the_equilibrium_definition():
    """One definition, so the two cannot drift apart (this is what #353 wants)."""
    Te, a, B = 500.0, 0.28, 0.2
    assert rhostar_from_Te_a_Bt(Te, a, B) == pytest.approx(
        normalized_larmor_radius_from_M_T_a_Bt(ME, Te, a, B)
    )


def test_rho_star_falls_when_the_machine_grows():
    """The defect: multiplying by `a` made rho* rise with machine size."""
    Te, B = 500.0, 0.2
    small, large = rhostar_from_Te_a_Bt(Te, 0.1, B), rhostar_from_Te_a_Bt(Te, 1.0, B)
    assert large < small
    old_expression = np.sqrt(Te) / B  # times a, as it used to be
    assert old_expression * 1.0 > old_expression * 0.1, "the old form rose with a"


# --- #363 -------------------------------------------------------------------

def test_beta_p_uses_q_over_epsilon_not_q_alone():
    beta_t = 0.05
    assert beta_pol_from_beta_tor(beta_t, Q95, EPSILON) == pytest.approx(
        beta_t * (Q95 / EPSILON) ** 2
    )


@pytest.mark.parametrize("epsilon", [0.7, 0.3, 0.1])
def test_the_missing_epsilon_squared_was_the_whole_error(epsilon):
    """The old form is the new one times epsilon^2 -- a factor of 100 at eps=0.1."""
    beta_t = 0.05
    old = beta_t * Q95**2
    new = beta_pol_from_beta_tor(beta_t, Q95, epsilon)
    assert old == pytest.approx(new * epsilon**2)


def test_the_two_directions_are_exact_inverses():
    """They round-tripped before too -- both were wrong by the same factor."""
    beta_t = 0.05
    beta_p = beta_pol_from_beta_tor(beta_t, Q95, EPSILON)
    assert beta_tor_from_beta_pol(beta_p, Q95, EPSILON) == pytest.approx(beta_t)


@pytest.mark.parametrize("bad", [0.0, -0.3, np.nan])
def test_a_non_positive_inverse_aspect_ratio_is_refused(bad):
    with pytest.raises(ValueError, match="epsilon"):
        beta_pol_from_beta_tor(0.05, Q95, bad)


# --- #364 -------------------------------------------------------------------

def _parabolic_pressure(n: int = 41):
    R = np.linspace(0.4, 0.9, n)
    return R, 1.0e4 * (1.0 - ((R - 0.4) / 0.5) ** 2)


def test_ballooning_alpha_carries_the_q_squared():
    R, p = _parabolic_pressure()
    B, q = 0.3, 3.0
    expected = -2.0 * MU0 * R * q**2 * np.gradient(p, R) / B**2
    np.testing.assert_allclose(ballooning_alpha_from_p_B_R(p, B, R, q), expected)


def test_alpha_scales_as_q_squared():
    R, p = _parabolic_pressure()
    at_one = ballooning_alpha_from_p_B_R(p, 0.3, R, 1.0)
    at_three = ballooning_alpha_from_p_B_R(p, 0.3, R, 3.0)
    np.testing.assert_allclose(at_three, 9.0 * at_one)


def test_the_old_form_was_alpha_over_q_squared():
    """At q ~ 3 that is an order of magnitude, read against a 0.6*s boundary."""
    R, p = _parabolic_pressure()
    B, q = 0.3, 3.0
    old = -2.0 * MU0 * R * np.gradient(p, R) / B**2
    np.testing.assert_allclose(old, ballooning_alpha_from_p_B_R(p, B, R, q) / q**2)


def test_alpha_is_positive_where_the_pressure_falls_outward():
    """Sign convention: a decreasing profile drives ballooning."""
    R, p = _parabolic_pressure()
    alpha = ballooning_alpha_from_p_B_R(p, 0.3, R, 3.0)
    assert np.all(alpha[1:] > 0.0)
