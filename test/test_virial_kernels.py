"""Numeric coverage for the ``vaft.formula.virial`` kernels.

Every assertion here recomputes the quantity independently of the
implementation, either from the closed form in the docstring or from an
identity that must hold whatever the implementation is (a uniform field
giving :math:`l_i = 1`, the Lao bundle agreeing with its three parts, the
energy forms in :mod:`~vaft.formula.equilibrium` agreeing with the volume
integrals here).  Tolerances are ``rel``-only with ``abs=0.0`` wherever the
expected value is small, because ``pytest.approx`` passes when *either*
tolerance is met and its default ``abs=1e-12`` silently accepts an arbitrary
prefactor error on a small number.
"""

import numpy as np
import pytest

from vaft.formula import (
    eK_from_K,
    kinetic_energy_from_beta_p_B_pa_V_p,
    magnetic_energy_from_li_B_pa_V_p,
    virial_D0_boundary_from_bp_li_eK,
    virial_S1_approx,
    virial_S2_approx_from_D0_a_R0,
    virial_S3_approx_from_eK_d,
    virial_beta_p_from_S_alpha_mu,
    virial_beta_p_from_S_li,
    virial_beta_p_from_volume,
    virial_beta_p_lao_from_S_mu_rt,
    virial_beta_p_li_from_S_alpha_mu_rt,
    virial_beta_pd_from_S_mu_rt,
    virial_kinetic_energy,
    virial_li_from_S_alpha_rt,
    virial_li_from_volume,
    virial_magnetic_energy,
    virial_stability_criterion,
    virial_theorem,
    virial_thermal_energy,
)
from vaft.formula.constants import MU0


# --------------------------------------------------------------------------
# Shafranov integrals
# --------------------------------------------------------------------------

def test_S1_is_the_leading_order_constant_two():
    assert virial_S1_approx() == 2.0


def test_S2_approx_matches_its_closed_form():
    eK, D0, a, R0 = 0.6, -0.12, 0.4, 1.8
    assert virial_S2_approx_from_D0_a_R0(eK, D0, a, R0) == pytest.approx(
        -(2 * a / R0) * (D0 + 1) * (1 + eK / 2), rel=1e-13, abs=0.0
    )


def test_S2_vanishes_when_the_shift_cancels_the_unit_offset():
    # D0 = -1 removes the (D0 + 1) factor for any shape or aspect ratio.
    assert virial_S2_approx_from_D0_a_R0(0.6, -1.0, 0.4, 1.8) == 0.0


def test_S3_approx_matches_its_closed_form():
    eK, d = 0.55, 0.3
    assert virial_S3_approx_from_eK_d(eK, d) == pytest.approx(
        1 - 0.5 * eK - d * (1 - 0.5 * eK**2), rel=1e-13, abs=0.0
    )


def test_S3_reduces_to_one_minus_half_eK_for_a_zero_d_parameter():
    assert virial_S3_approx_from_eK_d(0.55, 0.0) == pytest.approx(1 - 0.5 * 0.55)


def test_D0_boundary_matches_its_closed_form_and_is_inward_signed():
    beta_p, li, eK, b, R = 0.8, 0.9, 0.5, 0.35, 1.7
    expected = -(b / (2 * R)) * (2 * beta_p + li + 0.5 * eK) / (1 + 0.5 * eK)
    got = virial_D0_boundary_from_bp_li_eK(beta_p, li, eK, b, R)
    assert got == pytest.approx(expected, rel=1e-13, abs=0.0)
    # The prefactor is negative and the numerator positive for any physical
    # (beta_p, li, eK >= 0), so the sign cannot flip with the shape.
    assert got < 0.0


def test_D0_grows_in_magnitude_with_beta_p():
    args = (0.9, 0.5, 0.35, 1.7)
    low = virial_D0_boundary_from_bp_li_eK(0.2, *args)
    high = virial_D0_boundary_from_bp_li_eK(2.0, *args)
    assert abs(high) > abs(low)


def test_eK_is_zero_for_a_circle_and_approaches_one_for_a_thin_ellipse():
    assert eK_from_K(1.0) == 0.0
    assert eK_from_K(2.0) == pytest.approx(3.0 / 5.0, rel=1e-13, abs=0.0)
    assert eK_from_K(1e6) == pytest.approx(1.0, rel=1e-11)


# --------------------------------------------------------------------------
# Closures
# --------------------------------------------------------------------------

def test_beta_p_from_S_alpha_mu_matches_its_closed_form():
    S1, S2, S3, alpha, mui = 2.0, -0.4, 0.3, 1.6, 0.25
    expected = ((S1 + S2) * (alpha - 1) + alpha * mui + S3) / (3 * (alpha - 1) + 1)
    assert virial_beta_p_from_S_alpha_mu(S1, S2, S3, alpha, mui) == pytest.approx(
        expected, rel=1e-13, abs=0.0
    )


def test_beta_p_from_S_li_matches_its_closed_form():
    assert virial_beta_p_from_S_li(2.0, -0.4, 0.9) == pytest.approx(
        0.25 * 2.0 + 0.5 * -0.4 + -0.5 * 0.9, rel=1e-13, abs=0.0
    )


def test_beta_p_from_S_li_trades_one_for_one_against_half_of_li():
    base = virial_beta_p_from_S_li(2.0, -0.4, 0.9)
    assert virial_beta_p_from_S_li(2.0, -0.4, 1.1) == pytest.approx(base - 0.1)


def test_li_from_S_alpha_rt_matches_its_closed_form():
    S1, S2, S3, alpha, rt = 2.0, -0.4, 0.3, 1.6, 1.05
    expected = (0.5 * S1 + 0.5 * S2 * (1.0 - rt) - S3) / (alpha - 1.0)
    assert virial_li_from_S_alpha_rt(S1, S2, S3, alpha, rt) == pytest.approx(
        expected, rel=1e-13, abs=0.0
    )


def test_li_from_S_alpha_rt_rejects_an_alpha_at_the_singularity():
    with pytest.raises(ValueError, match="alpha"):
        virial_li_from_S_alpha_rt(2.0, -0.4, 0.3, 1.0, 1.05)


def test_li_from_S_alpha_rt_accepts_an_alpha_just_outside_the_guard():
    # The guard is |alpha - 1| <= eps, so 2*eps must pass and stay finite.
    value = virial_li_from_S_alpha_rt(2.0, -0.4, 0.3, 1.0 + 2e-12, 1.05, eps=1e-12)
    assert np.isfinite(value)


def test_lao_bundle_equals_its_three_component_closures():
    S1, S2, S3, alpha, mui, rt = 2.0, -0.4, 0.3, 1.6, 0.25, 1.05
    beta_p, li, beta_pd = virial_beta_p_li_from_S_alpha_mu_rt(
        S1, S2, S3, alpha, mui, rt
    )
    assert beta_p == virial_beta_p_lao_from_S_mu_rt(S1, S2, mui, rt)
    assert li == virial_li_from_S_alpha_rt(S1, S2, S3, alpha, rt)
    assert beta_pd == virial_beta_pd_from_S_mu_rt(S1, S2, mui, rt)


def test_lao_beta_p_delegate_matches_its_closed_form():
    # The bundle test above only asserts that the two sides agree, which a
    # mutation of both survives; these two pin the coefficients themselves.
    S1, S2, mui, rt = 2.0, -0.4, 0.25, 1.05
    assert virial_beta_p_lao_from_S_mu_rt(S1, S2, mui, rt) == pytest.approx(
        0.5 * S1 + 0.5 * S2 * (1.0 - rt) + mui, rel=1e-13, abs=0.0
    )


def test_lao_beta_pd_delegate_matches_its_closed_form():
    S1, S2, mui, rt = 2.0, -0.4, 0.25, 1.05
    assert virial_beta_pd_from_S_mu_rt(S1, S2, mui, rt) == pytest.approx(
        0.5 * S1 - mui + 0.5 * S2 * (1.0 - rt), rel=1e-13, abs=0.0
    )


def test_the_two_lao_delegates_differ_only_in_the_sign_of_mui():
    S1, S2, mui, rt = 2.0, -0.4, 0.25, 1.05
    beta_p = virial_beta_p_lao_from_S_mu_rt(S1, S2, mui, rt)
    beta_pd = virial_beta_pd_from_S_mu_rt(S1, S2, mui, rt)
    assert beta_p - beta_pd == pytest.approx(2.0 * mui, rel=1e-13, abs=0.0)


def test_li_from_S_alpha_rt_honours_an_injected_guard_width():
    # The default eps is 1e-12; a wider guard must reject an alpha the
    # default would accept, which pins the default rather than the branch.
    alpha = 1.0 + 1e-9
    assert np.isfinite(virial_li_from_S_alpha_rt(2.0, -0.4, 0.3, alpha, 1.05))
    with pytest.raises(ValueError, match="alpha"):
        virial_li_from_S_alpha_rt(2.0, -0.4, 0.3, alpha, 1.05, eps=1e-6)


def test_lao_bundle_propagates_the_singular_alpha_guard():
    with pytest.raises(ValueError, match="alpha"):
        virial_beta_p_li_from_S_alpha_mu_rt(2.0, -0.4, 0.3, 1.0, 0.25, 1.05)


# --------------------------------------------------------------------------
# Volume integrals
# --------------------------------------------------------------------------

def test_li_from_volume_is_exactly_one_for_a_field_at_the_reference_value():
    # l_i is B_p^2 volume-averaged and normalised to B_pa^2, so a uniform
    # field equal to B_pa must give exactly 1 whatever the cell layout.
    B_pa = 0.21
    dV = np.array([0.4, 1.0, 2.0])
    B_p = np.full(dV.shape, B_pa)
    assert virial_li_from_volume(B_p, dV, B_pa, dV.sum()) == pytest.approx(
        1.0, rel=1e-13, abs=0.0
    )


def test_li_from_volume_normalises_by_the_caller_supplied_volume():
    # Omega is the caller's reference volume, not the sum of the cells: a
    # caller passing twice the cell sum must get half the inductance.
    B_pa = 0.21
    dV = np.array([0.4, 1.0, 2.0])
    B_p = np.full(dV.shape, B_pa)
    at_cell_sum = virial_li_from_volume(B_p, dV, B_pa, dV.sum())
    at_double = virial_li_from_volume(B_p, dV, B_pa, 2.0 * dV.sum())
    assert at_double == pytest.approx(0.5 * at_cell_sum, rel=1e-13, abs=0.0)


def test_li_from_volume_scales_quadratically_with_the_field():
    dV = np.array([0.4, 1.0, 2.0])
    B_pa = 0.21
    single = virial_li_from_volume(np.full(3, B_pa), dV, B_pa, dV.sum())
    doubled = virial_li_from_volume(np.full(3, 2 * B_pa), dV, B_pa, dV.sum())
    assert doubled == pytest.approx(4.0 * single, rel=1e-13, abs=0.0)


def test_beta_p_from_volume_matches_the_uniform_pressure_closed_form():
    p0, B_pa = 1.2e4, 0.21
    dV = np.array([0.4, 1.0, 2.0])
    got = virial_beta_p_from_volume(np.full(3, p0), dV, B_pa, dV.sum())
    assert got == pytest.approx(2 * MU0 * p0 / B_pa**2, rel=1e-13, abs=0.0)


def test_beta_p_from_volume_honours_an_injected_mu0():
    dV = np.array([1.0, 1.0])
    p = np.array([1.0, 3.0])
    default = virial_beta_p_from_volume(p, dV, 0.5, dV.sum())
    doubled = virial_beta_p_from_volume(p, dV, 0.5, dV.sum(), mu0=2 * MU0)
    assert doubled == pytest.approx(2 * default, rel=1e-13, abs=0.0)


# --------------------------------------------------------------------------
# Energies
# --------------------------------------------------------------------------

def test_thermal_energy_matches_three_halves_n_T_V():
    n = np.array([1.0e19, 2.0e19])
    T = np.array([300.0, 500.0])
    V = 2.5
    assert virial_thermal_energy(n, T, V) == pytest.approx(
        1.5 * float(np.sum(n * T)) * V, rel=1e-13, abs=0.0
    )


def test_kinetic_energy_matches_half_n_m_v_squared_V():
    n = np.array([1.0e19, 2.0e19])
    v = np.array([1.0e4, 3.0e4])
    m, V = 3.34e-27, 2.5
    assert virial_kinetic_energy(n, v, m, V) == pytest.approx(
        0.5 * float(np.sum(n * m * v**2)) * V, rel=1e-13, abs=0.0
    )


def test_kinetic_energy_is_quadratic_in_the_velocity():
    n = np.array([1.0e19, 2.0e19])
    v = np.array([1.0e4, 3.0e4])
    single = virial_kinetic_energy(n, v, 3.34e-27, 2.5)
    doubled = virial_kinetic_energy(n, 2 * v, 3.34e-27, 2.5)
    assert doubled == pytest.approx(4.0 * single, rel=1e-13, abs=0.0)


def test_magnetic_energy_matches_B_squared_V_over_two_mu0():
    B = np.array([0.2, 0.5])
    V = 2.5
    assert virial_magnetic_energy(B, V) == pytest.approx(
        float(np.sum(B**2)) * V / (2 * MU0), rel=1e-13, abs=0.0
    )


def test_virial_theorem_sums_the_three_energies_and_ratios_the_last_two():
    W_mag, W_kin, W_th = 4.0e5, 1.0e5, 3.0e5
    total, ratio = virial_theorem(W_mag, W_kin, W_th)
    assert total == pytest.approx(W_mag + W_kin + W_th, rel=1e-13, abs=0.0)
    assert ratio == pytest.approx((W_kin + W_th) / W_mag, rel=1e-13, abs=0.0)


def test_stability_criterion_is_the_virial_ratio_offset_by_the_threshold():
    W_mag, W_kin, W_th = 4.0e5, 1.0e5, 3.0e5
    margin, critical = virial_stability_criterion(W_mag, W_kin, W_th)
    assert critical == 0.5
    assert margin == pytest.approx(virial_theorem(W_mag, W_kin, W_th)[1] - 0.5)


def test_stability_margin_is_zero_exactly_at_the_threshold():
    # W_kin + W_th = W_mag / 2 puts the plasma on the heuristic boundary.
    margin, _ = virial_stability_criterion(4.0e5, 1.0e5, 1.0e5)
    assert margin == pytest.approx(0.0, abs=1e-15)


# --------------------------------------------------------------------------
# Cross-consistency with the equilibrium energy forms
# --------------------------------------------------------------------------

def test_beta_p_volume_integral_reproduces_the_equilibrium_thermal_energy():
    # W_K = (3/2) beta_p B_pa^2 V_p / (2 mu0) with beta_p = 2 mu0 p / B_pa^2
    # collapses to (3/2) p V, which is the volume form's thermal energy for a
    # uniform plasma.  The two modules must therefore agree exactly.
    p0, B_pa = 1.2e4, 0.21
    dV = np.array([0.4, 1.0, 2.0])
    Omega = float(dV.sum())
    beta_p = virial_beta_p_from_volume(np.full(3, p0), dV, B_pa, Omega)
    assert kinetic_energy_from_beta_p_B_pa_V_p(beta_p, B_pa, Omega) == pytest.approx(
        1.5 * p0 * Omega, rel=1e-12, abs=0.0
    )


def test_li_volume_integral_reproduces_the_equilibrium_magnetic_energy():
    B_pa = 0.21
    dV = np.array([0.4, 1.0, 2.0])
    Omega = float(dV.sum())
    B_p = np.full(3, B_pa)
    li = virial_li_from_volume(B_p, dV, B_pa, Omega)
    assert magnetic_energy_from_li_B_pa_V_p(li, B_pa, Omega) == pytest.approx(
        virial_magnetic_energy(np.array([B_pa]), Omega), rel=1e-12, abs=0.0
    )
