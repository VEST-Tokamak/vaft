"""The eleven stability kernels that nothing called (#711 follow-up).

`test_formula_docstrings` and `test_formula_catalog` parametrize over every
formula in the package, which makes the suite look complete while never
evaluating one: they only `getattr` the function.  Eleven of the nineteen in
`vaft.formula.stability` had no caller anywhere in `test/`, and the three
beside them had just been found to have dropped factors (#363, #364, #348).

So these check numbers, against the definitions the docstrings state and
against independent arithmetic, not merely that a call returns without
raising.  Where a routine is documented as *not* being the textbook quantity
its name suggests, the test pins what it actually computes and says why, so a
later correction has to change the test deliberately.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.formula.constants import MI_P, MU0, QE
from vaft.formula.stability import (
    ballooning_stability_criterion,
    c_s_from_Te_Ti_mi,
    collisionality_from_n_T_B_R,
    empirical_li_qa,
    greenwald_density,
    li_from_qa_empirical,
    plasma_stability_margins,
    power_limit_from_beta,
    power_limit_from_q,
    sawtooth_stability_criterion,
    v_alfven_from_B_n_mi,
)


# --- speeds, against the NRL formulary ------------------------------------------


def test_the_alfven_speed_matches_its_definition_and_the_formulary():
    B, n = 0.35, 5.0e18
    assert v_alfven_from_B_n_mi(B, n) == pytest.approx(B / np.sqrt(MU0 * n * MI_P))

    # NRL Plasma Formulary (2019) p.29: v_A = 2.18e11 * B[G] / sqrt(mu * n_i[cm^-3]) cm/s.
    nrl_cm_s = 2.18e11 * (B * 1.0e4) / np.sqrt(1.0 * (n * 1.0e-6))
    assert v_alfven_from_B_n_mi(B, n) == pytest.approx(nrl_cm_s * 1.0e-2, rel=2e-3)


def test_the_alfven_speed_scales_as_the_field_over_the_root_density():
    base = v_alfven_from_B_n_mi(0.3, 1.0e19)
    assert v_alfven_from_B_n_mi(0.6, 1.0e19) == pytest.approx(2.0 * base)
    assert v_alfven_from_B_n_mi(0.3, 4.0e19) == pytest.approx(0.5 * base)
    # A heavier ion slows it by the root of the mass ratio.
    assert v_alfven_from_B_n_mi(0.3, 1.0e19, m_i=4.0 * MI_P) == pytest.approx(base / 2.0)


def test_the_sound_speed_is_isothermal_and_uses_both_temperatures():
    assert c_s_from_Te_Ti_mi(1.0, 1.0) == pytest.approx(
        np.sqrt(2.0e3 * QE / MI_P)
    )
    # NRL p.29 for the electron-only, gamma = 1 case: 9.79e5 sqrt(T_e[eV]) cm/s.
    assert c_s_from_Te_Ti_mi(1.0, 0.0) == pytest.approx(
        9.79e5 * np.sqrt(1.0e3) * 1.0e-2, rel=2e-3
    )
    # Isothermal, not adiabatic: gamma_i = 3 would be larger by sqrt((Te+3Ti)/(Te+Ti)).
    assert c_s_from_Te_Ti_mi(1.0, 1.0) < np.sqrt((1.0 + 3.0) * 1.0e3 * QE / MI_P)


# --- the Greenwald limit ---------------------------------------------------------


def test_the_greenwald_density_is_the_literature_value_in_this_module_s_units():
    # n_G [1e20 m^-3] = I_p[MA] / (pi a^2); this returns 1e19 units, so ten times that.
    I_p, a = 1.0, 0.5
    literature_1e20 = I_p / (np.pi * a**2)
    assert greenwald_density(I_p, a) == pytest.approx(10.0 * literature_1e20)


def test_the_greenwald_density_falls_as_the_inverse_square_of_the_minor_radius():
    assert greenwald_density(1.0, 1.0) == pytest.approx(10.0 / np.pi)
    assert greenwald_density(1.0, 2.0) == pytest.approx(greenwald_density(1.0, 1.0) / 4.0)
    assert greenwald_density(2.0, 1.0) == pytest.approx(2.0 * greenwald_density(1.0, 1.0))


# --- the two criteria, both of which report a margin and a threshold --------------


def test_the_ballooning_criterion_reports_the_distance_to_its_own_threshold():
    margin, crit = ballooning_stability_criterion(0.9, 1.0)
    assert crit == pytest.approx(0.6)
    assert margin == pytest.approx(0.3), "positive margin means above the boundary"

    stable_margin, _ = ballooning_stability_criterion(0.3, 1.0)
    assert stable_margin < 0.0

    # Vectorised over both arguments, and the threshold is linear in the shear.
    alpha = np.array([0.0, 0.6, 1.2])
    margins, crits = ballooning_stability_criterion(alpha, 1.0)
    np.testing.assert_allclose(crits, 0.6)
    np.testing.assert_allclose(margins, alpha - 0.6)


def test_the_sawtooth_threshold_goes_negative_above_unit_q0_where_sawteeth_do_not_occur():
    margin, crit = sawtooth_stability_criterion(0.8, 0.5)
    assert crit == pytest.approx(0.3 * 0.2)
    assert margin == pytest.approx(0.5 - 0.06)

    # Documented limitation, pinned so a fix has to change it: above q_0 = 1 the
    # threshold is negative, so every poloidal beta "exceeds" it (#350).
    _, crit_above = sawtooth_stability_criterion(1.4, 0.5)
    assert crit_above < 0.0


# --- the two "limits" that are not powers ----------------------------------------


def test_the_beta_power_limit_is_the_stored_energy_it_is_documented_to_be():
    beta_N, B0, V = 0.03, 0.4, 1.5
    assert power_limit_from_beta(beta_N, B0, V) == pytest.approx(
        beta_N * B0**2 * V / (2.0 * MU0)
    )
    # Magnetic energy density times volume: an energy in joules, not a power.
    assert power_limit_from_beta(0.0, B0, V) == 0.0


def test_the_q_power_limit_is_the_ampere_squared_expression_it_is_documented_to_be():
    q_95, I_p, R0 = 3.0, 1.0e5, 0.4
    assert power_limit_from_q(q_95, I_p, R0) == pytest.approx(
        2.0 * np.pi * R0 * I_p / (MU0 * q_95)
    )
    # Inverse in q_95, linear in the current: the scaling the rearrangement implies.
    assert power_limit_from_q(6.0, I_p, R0) == pytest.approx(
        power_limit_from_q(3.0, I_p, R0) / 2.0
    )


# --- the collisionality figure, which is deliberately not anyone's nu_star --------


def test_the_collisionality_figure_is_this_module_s_own_normalisation():
    n_e, T_e, B_t, R0 = 3.0, 1.0, 0.3, 0.4
    # abs=0 deliberately. The figure is of order 1e-17, and `approx`'s default
    # absolute tolerance of 1e-12 swamps it: without this the prefactor could be
    # wrong by any factor and the comparison would still pass.
    assert collisionality_from_n_T_B_R(n_e, T_e, B_t, R0) == pytest.approx(
        6.921e-18 * n_e * R0 / (T_e**2 * B_t), rel=1e-12, abs=0.0
    )
    # Falls as the square of the temperature, which is the one trend it is for.
    assert collisionality_from_n_T_B_R(n_e, 2.0, B_t, R0) == pytest.approx(
        collisionality_from_n_T_B_R(n_e, 1.0, B_t, R0) / 4.0
    )


# --- the JET survey band ----------------------------------------------------------


def test_the_survey_returns_two_band_edges_at_each_integer_safety_factor():
    qa, li = empirical_li_qa()
    assert qa.size == li.size == 18
    np.testing.assert_array_equal(qa, np.repeat(np.arange(2, 11), 2))
    # Upper edge first, lower second, at every q_a.
    assert np.all(li[0::2] > li[1::2])
    # The documented saturation: the lower branch flattens above q_a ~ 6.
    np.testing.assert_allclose(li[1::2][-4:], 0.30)


def test_the_interpolation_returns_the_lower_edge_at_the_surveyed_points():
    _, li = empirical_li_qa()
    for index, qa_value in enumerate(range(2, 11)):
        assert li_from_qa_empirical(float(qa_value)) == pytest.approx(li[2 * index + 1])


def test_the_interpolation_rises_between_the_surveyed_points_which_the_physics_does_not():
    """A known defect, pinned so a fix has to change it deliberately (#755).

    The survey gives two l_i per q_a, and `np.interp` over those duplicated
    abscissae runs from one band's lower edge to the next band's upper edge.
    The documented physics is monotone -- low q_a goes with high l_i -- and the
    returned curve is not.
    """
    assert li_from_qa_empirical(6.5) > li_from_qa_empirical(6.0)
    assert li_from_qa_empirical(7.0) < li_from_qa_empirical(6.5)

    sampled = li_from_qa_empirical(np.linspace(2.0, 10.0, 81))
    rising = int(np.sum(np.diff(sampled) > 0))
    assert rising == 72, (
        "the sawtooth of #755: 72 of 80 intervals rise against the documented "
        "physics; change this number only together with the interpolation"
    )


def test_the_interpolation_holds_its_end_values_outside_the_surveyed_range():
    _, li = empirical_li_qa()
    assert li_from_qa_empirical(1.0) == pytest.approx(li[0])
    assert li_from_qa_empirical(12.0) == pytest.approx(li[-1])


# --- the aggregate, whose three margins use three sign conventions ----------------


def test_the_margins_carry_the_three_conventions_their_docstring_names():
    beta_margin, q_margin, density_margin = plasma_stability_margins(
        beta_N=2.0, q_95=3.0, n_e=5.0, n_G=10.0
    )
    assert q_margin == pytest.approx(1.0), "a difference: q_95 - 2"
    assert density_margin == pytest.approx(0.5), "a ratio: n_e / n_G"

    from vaft.formula.stability import beta_stability_boundary

    assert beta_margin == pytest.approx(beta_stability_boundary(2.0, 3.0)[0])


def test_a_safety_factor_below_two_reports_a_negative_margin():
    _, q_margin, _ = plasma_stability_margins(beta_N=2.0, q_95=1.5, n_e=5.0, n_G=10.0)
    assert q_margin < 0.0


def test_a_density_above_the_greenwald_limit_reports_a_fraction_above_one():
    _, _, density_margin = plasma_stability_margins(
        beta_N=2.0, q_95=3.0, n_e=12.0, n_G=10.0
    )
    assert density_margin > 1.0
