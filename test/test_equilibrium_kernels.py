"""Numeric coverage for the ``vaft.formula.equilibrium`` kernels.

The catalog and docstring suites never call these functions, so a wrong
coefficient has been invisible to CI.  Each assertion below recomputes the
value from the closed form in the docstring, or pins an identity that must
hold whatever the implementation is: the three bremsstrahlung variants
agreeing, the flux/safety-factor chain round-tripping, the aspect-ratio pair
being reciprocal.

``pytest.approx`` passes when *either* its relative or its absolute tolerance
is met, and its default ``abs=1e-12`` swamps anything smaller than that, so
every comparison of a small number passes ``abs=0.0`` explicitly.
"""

import numpy as np
import pytest

from vaft.formula import (
    alpha_heating_power,
    alpha_heating_power_from_n_D_n_T_T_keV_V,
    aspect_ratio_from_a_R,
    auxiliary_heating_power,
    bootstrap_current_fraction,
    bremsstrahlung_power_density_from_T_e_p_Z_eff,
    bremsstrahlung_power_density_from_Z_eff_n_e_T_e,
    bremsstrahlung_radiation_power_from_z_eff_n_e_t_e,
    confinement_time_from_P_loss_W_th,
    current_density_from_B,
    current_density_from_psi,
    current_drive_efficiency,
    current_limit_from_beta,
    current_limit_from_q,
    cyclotron_synchrotron_power_density_scaling_from_n_e_B_t_T_e,
    cylindrical_safety_factor_from_R_B_epsilon_I_f_kappa_delta,
    decay_index_from_bz,
    ec_heating_power_from_I_ec_V_ec,
    elongation_from_RZ_boundary,
    heating_power_from_p_ohm_p_aux,
    inductive_voltage_from_dW_magdt_I_p,
    inverse_aspect_ratio_from_a_R,
    kinetic_energy_from_beta_p_B_pa_V_p,
    kink_safety_factor,
    loss_power_from_p_heat_dWdt_p_rad,
    magnetic_energy_from_li_B_pa_V_p,
    nbi_heating_power_from_I_nbi_V_nbi,
    normalized_collisionality_from_a_n_q_epsilon_T,
    normalized_collisionality_from_nu_ii_T_i_M_i_R_a_q,
    normalized_plasma_current,
    ohmic_heating_power_from_I_p_V_res,
    phi_from_Bphi,
    poloidal_field_factor,
    poloidal_field_magnitude,
    psi_from_RBtheta,
    q_from_phi,
    q_from_rhoN,
    radial_magnetic_field_from_psi,
    rho_tor_from_phi,
    rhoN_from_phi,
    shear_from_r_q,
    stored_energy_from_beta_V,
    stored_energy_from_p_V,
    surface_poloidal_flux_from_psi_boundary,
    toroidal_electric_field,
    toroidal_flux_from_q_psi,
    r_at_z_extremum_from_RZ_contour,
    triangularity_from_RZ_boundary,
    triangularity_lower_from_RZ_boundary,
    triangularity_upper_from_RZ_boundary,
    verify_kadomtsev_constraint,
    vertical_magnetic_field_from_psi,
)
from vaft.formula.constants import E_ALPHA, MU0, QE, SIGMA_V_COEF

BOLTZMANN_J_PER_EV = 1.602176634e-19


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def test_aspect_ratio_and_its_inverse_are_reciprocal():
    a, R = 0.4, 1.8
    assert aspect_ratio_from_a_R(a, R) == pytest.approx(R / a, rel=1e-13, abs=0.0)
    assert inverse_aspect_ratio_from_a_R(a, R) == pytest.approx(
        a / R, rel=1e-13, abs=0.0
    )
    assert aspect_ratio_from_a_R(a, R) * inverse_aspect_ratio_from_a_R(a, R) == (
        pytest.approx(1.0, rel=1e-13, abs=0.0)
    )


def test_elongation_of_an_analytic_ellipse_is_its_axis_ratio():
    theta = np.linspace(0.0, 2 * np.pi, 2001)
    a, kappa, R0 = 0.4, 1.8, 1.5
    R = R0 + a * np.cos(theta)
    Z = kappa * a * np.sin(theta)
    assert elongation_from_RZ_boundary(R, Z) == pytest.approx(kappa, rel=1e-6)


def test_elongation_is_one_for_a_circle():
    theta = np.linspace(0.0, 2 * np.pi, 2001)
    R = 1.5 + 0.4 * np.cos(theta)
    Z = 0.4 * np.sin(theta)
    assert elongation_from_RZ_boundary(R, Z) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("delta", [0.0, 0.3, 0.6, -0.4])
def test_triangularity_of_a_miller_boundary_is_the_sine_of_its_delta(delta):
    # A Miller boundary R = R0 + a*cos(theta + delta*sin(theta)) reaches its
    # highest point at theta = pi/2, where R = R0 - a*sin(delta).  The
    # geometric triangularity (R0 - R_at_max_Z)/a is therefore sin(delta)
    # exactly, not delta -- the parameter is defined through an arcsine.  That
    # makes this an exact identity rather than a tolerance on a shape fit.
    theta = np.linspace(0.0, 2 * np.pi, 2001)
    a, R0, kappa = 0.4, 1.5, 1.8
    R = R0 + a * np.cos(theta + delta * np.sin(theta))
    Z = kappa * a * np.sin(theta)
    assert triangularity_from_RZ_boundary(R, Z, R0) == pytest.approx(
        np.sin(delta), abs=1e-6
    )


def test_a_symmetric_boundary_has_equal_upper_and_lower_triangularity():
    theta = np.linspace(0.0, 2 * np.pi, 2001)
    a, R0, kappa, delta = 0.4, 1.5, 1.8, 0.4
    R = R0 + a * np.cos(theta + delta * np.sin(theta))
    Z = kappa * a * np.sin(theta)
    upper = triangularity_upper_from_RZ_boundary(R, Z, R0)
    lower = triangularity_lower_from_RZ_boundary(R, Z, R0)
    assert upper == pytest.approx(lower, abs=1e-9)
    assert upper == pytest.approx(np.sin(delta), abs=1e-6)
    # The mean is exactly the mean, not an independent measurement.
    assert triangularity_from_RZ_boundary(R, Z, R0) == pytest.approx(
        0.5 * (upper + lower), rel=1e-13, abs=0.0
    )


def test_upper_and_lower_triangularity_separate_on_an_asymmetric_boundary():
    # Different delta above and below the midplane, which is the only case
    # where IMAS's two values carry more than their mean does.
    theta = np.linspace(0.0, 2 * np.pi, 4001)
    a, R0, kappa = 0.4, 1.5, 1.8
    delta_up, delta_low = 0.5, -0.2
    delta = np.where(np.sin(theta) >= 0.0, delta_up, delta_low)
    R = R0 + a * np.cos(theta + delta * np.sin(theta))
    Z = kappa * a * np.sin(theta)
    assert triangularity_upper_from_RZ_boundary(R, Z, R0) == pytest.approx(
        np.sin(delta_up), abs=1e-5
    )
    assert triangularity_lower_from_RZ_boundary(R, Z, R0) == pytest.approx(
        np.sin(delta_low), abs=1e-5
    )
    assert triangularity_from_RZ_boundary(R, Z, R0) == pytest.approx(
        0.5 * (np.sin(delta_up) + np.sin(delta_low)), abs=1e-5
    )


def test_triangularity_is_zero_for_a_circle_at_its_own_centre():
    theta = np.linspace(0.0, 2 * np.pi, 2001)
    R = 1.5 + 0.4 * np.cos(theta)
    Z = 0.4 * np.sin(theta)
    assert triangularity_from_RZ_boundary(R, Z, 1.5) == pytest.approx(0.0, abs=1e-9)


def test_r_at_z_extremum_beats_the_nearest_vertex_and_wraps():
    # Three points around a parabolic top whose true maximum sits between the
    # samples; the nearest vertex would report 1.4, the parabola 1.5.
    R = np.array([1.4, 1.6, 2.0, 1.0])
    Z = np.array([0.99, 0.99, 0.0, 0.0])
    assert r_at_z_extremum_from_RZ_contour(R, Z, upper=True) == pytest.approx(1.5)
    # A flat neighbourhood makes the parabola degenerate; fall back to the vertex.
    flat = np.array([0.5, 0.5, 0.5, 0.5])
    assert r_at_z_extremum_from_RZ_contour(R, flat, upper=True) == pytest.approx(
        R[int(np.argmax(flat))]
    )
    # Fewer than three points cannot carry a parabola.
    assert r_at_z_extremum_from_RZ_contour(
        np.array([1.0, 2.0]), np.array([0.0, 1.0]), upper=True
    ) == pytest.approx(2.0)
    # Exactly three can: the threshold is < 3, not <= 3, and at three points
    # the parabola still moves the answer off the vertex.
    assert r_at_z_extremum_from_RZ_contour(
        np.array([1.4, 1.6, 2.0]), np.array([0.99, 0.99, 0.0]), upper=True
    ) == pytest.approx(1.5)
    # The |shift| > 1 arm of the same guard is unreachable: when the middle
    # sample is the extremum, |shift| <= 1/2 for every pair of neighbours.
    # Sampled over 2e6 random neighbour pairs, the largest was 0.4999998.


def test_the_formula_and_process_extremum_helpers_are_one_implementation():
    from vaft.process.equilibrium import r_at_z_extremum

    theta = np.linspace(0.0, 2 * np.pi, 501)
    R = 1.5 + 0.4 * np.cos(theta + 0.3 * np.sin(theta))
    Z = 0.7 * np.sin(theta)
    for upper in (True, False):
        assert r_at_z_extremum(R, Z, upper=upper) == (
            r_at_z_extremum_from_RZ_contour(R, Z, upper=upper)
        )


def test_triangularity_agrees_with_the_process_contour_shape_parameters():
    # vaft.process computes the same two numbers against the geometric centre;
    # passing that centre as R0 must reproduce them exactly, or the package
    # holds two definitions of one quantity again (#365).
    from vaft.process.equilibrium import contour_shape_parameters

    theta = np.linspace(0.0, 2 * np.pi, 1001)
    a, R0, kappa, delta = 0.4, 1.5, 1.8, 0.35
    R = R0 + a * np.cos(theta + delta * np.sin(theta))
    Z = kappa * a * np.sin(theta)
    shape = contour_shape_parameters(R, Z)
    r_geo = 0.5 * (R.max() + R.min())
    assert triangularity_upper_from_RZ_boundary(R, Z, r_geo) == pytest.approx(
        shape["triangularity_upper"], rel=1e-13, abs=0.0
    )
    assert triangularity_lower_from_RZ_boundary(R, Z, r_geo) == pytest.approx(
        shape["triangularity_lower"], rel=1e-13, abs=0.0
    )
    assert elongation_from_RZ_boundary(R, Z) == pytest.approx(
        shape["elongation"], rel=1e-13, abs=0.0
    )


# --------------------------------------------------------------------------
# Fields, flux and current
# --------------------------------------------------------------------------

def test_poloidal_field_magnitude_is_the_euclidean_norm():
    b_r = np.array([3.0, -0.5])
    b_z = np.array([4.0, 1.2])
    assert poloidal_field_magnitude(b_r, b_z) == pytest.approx(
        np.array([5.0, np.hypot(-0.5, 1.2)]), rel=1e-13, abs=0.0
    )


def test_vertical_field_from_a_linear_psi_cut_matches_the_cocos_prefactor():
    R = np.linspace(1.0, 2.0, 41)
    slope = 0.07
    psi = 0.3 + slope * R
    expected = -poloidal_field_factor(None) / R * slope
    assert vertical_magnetic_field_from_psi(psi, R, np.zeros_like(R)) == (
        pytest.approx(expected, rel=1e-10, abs=0.0)
    )


def test_radial_field_from_a_linear_psi_cut_matches_the_cocos_prefactor():
    Z = np.linspace(-0.5, 0.5, 41)
    R = np.full_like(Z, 1.4)
    slope = -0.02
    psi = 0.3 + slope * Z
    expected = poloidal_field_factor(None) / R * slope
    assert radial_magnetic_field_from_psi(psi, R, Z) == pytest.approx(
        expected, rel=1e-10, abs=0.0
    )


def test_an_explicit_cocos_index_rescales_the_field_by_the_factor_ratio():
    R = np.linspace(1.0, 2.0, 41)
    psi = 0.3 + 0.07 * R
    default = vertical_magnetic_field_from_psi(psi, R, np.zeros_like(R))
    cocos11 = vertical_magnetic_field_from_psi(psi, R, np.zeros_like(R), cocos=11)
    ratio = poloidal_field_factor(11) / poloidal_field_factor(None)
    assert cocos11 == pytest.approx(default * ratio, rel=1e-12, abs=0.0)


def test_current_density_from_psi_is_the_documented_B_z_over_mu0():
    # Documented in #355: this expression is B_Z / mu0, a current per unit
    # length, not a current density.  Pin the relation so the defect cannot
    # be "fixed" silently in one place and not the other.
    R = np.linspace(1.0, 2.0, 41)
    psi = 0.3 + 0.07 * R
    b_z = vertical_magnetic_field_from_psi(psi, R, np.zeros_like(R))
    assert current_density_from_psi(psi, R) == pytest.approx(
        b_z / (MU0 * poloidal_field_factor(None)), rel=1e-10, abs=0.0
    )


def test_current_density_from_B_is_the_radial_derivative_over_mu0():
    R = np.linspace(1.0, 2.0, 41)
    slope = 0.11
    B = 0.02 + slope * R
    assert current_density_from_B(B, R) == pytest.approx(
        np.full_like(R, slope / MU0), rel=1e-10, abs=0.0
    )


def test_current_density_from_B_vanishes_for_a_uniform_field():
    R = np.linspace(1.0, 2.0, 41)
    assert current_density_from_B(np.full_like(R, 0.3), R) == pytest.approx(
        np.zeros_like(R), abs=1e-9
    )


def test_decay_index_of_a_linear_field_matches_its_closed_form():
    # np.gradient is exact on a linear profile, so this pins the prefactor
    # rather than the differencing scheme.
    r = np.linspace(0.5, 1.5, 51)
    c, k = 1.0, 0.3
    b_z = c - k * r
    assert decay_index_from_bz(r, b_z) == pytest.approx(
        r * k / (c - k * r), rel=1e-10, abs=0.0
    )


def test_decay_index_broadcasts_the_radius_along_the_requested_axis():
    # The 2-D path reshapes r to line up with `axis`; every row of a stack of
    # identical profiles must reproduce the 1-D answer, and a transposed stack
    # with axis=0 must give the transpose of it.
    r = np.linspace(0.5, 1.5, 21)
    c, k = 1.0, 0.3
    b_row = c - k * r
    expected = r * k / (c - k * r)

    stack = np.vstack([b_row, 2.0 * b_row, -0.5 * b_row])
    got = decay_index_from_bz(r, stack)
    assert got.shape == stack.shape
    for row in got:
        # n is a logarithmic derivative, so scaling B_Z leaves it unchanged.
        assert row == pytest.approx(expected, rel=1e-10, abs=0.0)

    got_axis0 = decay_index_from_bz(r, stack.T, axis=0)
    assert got_axis0 == pytest.approx(got.T, rel=1e-10, abs=0.0)


def test_decay_index_is_zero_for_a_uniform_vertical_field():
    r = np.linspace(0.5, 1.5, 51)
    assert decay_index_from_bz(r, np.full_like(r, 0.02)) == pytest.approx(
        np.zeros_like(r), abs=1e-9
    )


def test_toroidal_electric_field_matches_minus_dpsi_dt_over_two_pi_R():
    r = np.array([0.8, 1.2])
    dpsi_dt = np.array([0.4, -0.6])
    assert toroidal_electric_field(r, dpsi_dt) == pytest.approx(
        -dpsi_dt / (2.0 * np.pi * r), rel=1e-13, abs=0.0
    )


def test_surface_poloidal_flux_converts_per_radian_to_full_weber():
    assert surface_poloidal_flux_from_psi_boundary(0.017) == pytest.approx(
        0.017 * 2 * np.pi, rel=1e-13, abs=0.0
    )


def test_psi_from_RBtheta_integrates_a_constant_integrand_exactly():
    l = np.linspace(0.0, 2.0, 101)
    R = np.full_like(l, 1.4)
    B_theta = np.full_like(l, 0.05)
    assert psi_from_RBtheta(R, B_theta, l) == pytest.approx(
        1.4 * 0.05 * 2.0, rel=1e-12, abs=0.0
    )


def test_psi_from_RBtheta_adds_the_axis_offset():
    l = np.linspace(0.0, 2.0, 101)
    R = np.full_like(l, 1.4)
    B_theta = np.full_like(l, 0.05)
    base = psi_from_RBtheta(R, B_theta, l)
    assert psi_from_RBtheta(R, B_theta, l, psi_axis=0.25) == pytest.approx(
        base + 0.25, rel=1e-12, abs=0.0
    )


def test_phi_from_Bphi_is_the_area_weighted_sum():
    B_phi = np.array([0.18, 0.20, 0.22])
    dA = np.array([0.01, 0.02, 0.03])
    assert phi_from_Bphi(B_phi, dA) == pytest.approx(
        float(np.sum(B_phi * dA)), rel=1e-13, abs=0.0
    )


# --------------------------------------------------------------------------
# The flux / safety-factor / radial-coordinate chain
# --------------------------------------------------------------------------

def test_toroidal_flux_and_q_from_phi_round_trip():
    # Phi = int q dpsi, then q = dPhi/dpsi must return the original profile.
    # np.gradient is only first-order at the two end points, so compare the
    # interior where the second-order stencil applies.
    psi = np.linspace(0.0, 0.4, 401)
    q = 1.0 + 4.0 * psi
    phi = toroidal_flux_from_q_psi(q, psi)
    recovered = q_from_phi(psi, phi)
    # Both steps are exact on these profiles (trapezoid on a linear q,
    # central difference on the resulting quadratic over a uniform grid),
    # so the interior must agree to near machine precision.
    assert recovered[1:-1] == pytest.approx(q[1:-1], rel=1e-12, abs=0.0)


def test_toroidal_flux_starts_at_zero_and_integrates_a_constant_q():
    psi = np.linspace(0.0, 0.4, 401)
    phi = toroidal_flux_from_q_psi(np.full_like(psi, 2.5), psi)
    assert phi[0] == 0.0
    assert phi[-1] == pytest.approx(2.5 * 0.4, rel=1e-12, abs=0.0)


def test_q_from_rhoN_matches_its_closed_form_on_a_linear_label():
    # rhoN = psiN makes drhoN/dpsiN exactly 1, so q = C * psiN.
    psiN = np.linspace(0.1, 1.0, 91)
    assert q_from_rhoN(psiN, psiN, C=1.5) == pytest.approx(
        1.5 * psiN, rel=1e-10, abs=0.0
    )


def test_q_from_rhoN_defaults_its_scale_factor_to_one():
    psiN = np.linspace(0.1, 1.0, 91)
    assert q_from_rhoN(psiN, psiN) == pytest.approx(
        q_from_rhoN(psiN, psiN, C=1.0), rel=1e-13, abs=0.0
    )
    assert q_from_rhoN(psiN, psiN)[10] == pytest.approx(psiN[10], rel=1e-10, abs=0.0)


def test_rhoN_from_phi_is_the_square_root_of_the_normalised_flux():
    phi = np.array([0.0, 0.25, 1.0])
    assert rhoN_from_phi(phi, 4.0) == pytest.approx(
        np.sqrt(phi / 4.0), rel=1e-13, abs=0.0
    )
    assert rhoN_from_phi(4.0, 4.0) == pytest.approx(1.0, rel=1e-13, abs=0.0)


def test_rho_tor_normalised_by_its_boundary_value_equals_rhoN():
    # rho_tor ~ sqrt(|Phi|), so the ratio must drop B0 entirely and coincide
    # with the normalised label whatever the field is.
    phi = np.array([0.02, 0.10, 0.31])
    phi_b = 0.31
    for B0 in (0.18, -1.7):
        ratio = rho_tor_from_phi(phi, B0) / rho_tor_from_phi(phi_b, B0)
        assert ratio == pytest.approx(rhoN_from_phi(phi, phi_b), rel=1e-12, abs=0.0)


def test_rho_tor_matches_its_closed_form_and_ignores_the_sign_of_B0():
    assert rho_tor_from_phi(0.31, -1.7) == pytest.approx(
        np.sqrt(0.31 / (np.pi * 1.7)), rel=1e-13, abs=0.0
    )
    assert rho_tor_from_phi(0.31, 1.7) == rho_tor_from_phi(0.31, -1.7)


def test_shear_of_a_power_law_q_profile_is_its_exponent():
    # q ~ r^2 has s = (r/q) dq/dr = 2 everywhere.  np.gradient drops to a
    # one-sided first-order stencil at the two end points, so the exponent is
    # only recovered on the interior.
    r = np.linspace(0.05, 0.4, 801)
    assert shear_from_r_q(r, r**2)[1:-1] == pytest.approx(
        np.full(r.size - 2, 2.0), rel=1e-12, abs=0.0
    )


def test_shear_is_zero_for_a_flat_q_profile():
    r = np.linspace(0.05, 0.4, 101)
    assert shear_from_r_q(r, np.full_like(r, 1.3)) == pytest.approx(
        np.zeros_like(r), abs=1e-9
    )


# --------------------------------------------------------------------------
# Safety factor and operational limits
# --------------------------------------------------------------------------

def test_cylindrical_safety_factor_matches_its_closed_form():
    R, B, eps, I, fs = 1.8, 0.2, 0.25, 1.0e5, 1.3
    assert cylindrical_safety_factor_from_R_B_epsilon_I_f_kappa_delta(
        R, B, eps, I, fs
    ) == pytest.approx(
        2.0 * np.pi * eps**2 * R * B / (MU0 * I * fs), rel=1e-12, abs=0.0
    )


def test_cylindrical_safety_factor_is_inverse_in_the_plasma_current():
    args = (1.8, 0.2, 0.25)
    single = cylindrical_safety_factor_from_R_B_epsilon_I_f_kappa_delta(
        *args, 1.0e5, 1.3
    )
    doubled = cylindrical_safety_factor_from_R_B_epsilon_I_f_kappa_delta(
        *args, 2.0e5, 1.3
    )
    assert doubled == pytest.approx(0.5 * single, rel=1e-12, abs=0.0)


def test_kink_safety_factor_circular_branch_matches_its_closed_form():
    R, a, kappa, Ip, Bt = 1.8, 0.4, 1.6, 1.0e5, 0.2
    q, q_min, beta_max, beta_crit, ip_max = kink_safety_factor(
        R, a, kappa, Ip, Bt, "circular"
    )
    assert q == pytest.approx(
        2 * np.pi * a**2 * Bt / (MU0 * Ip * R), rel=1e-12, abs=0.0
    )
    assert beta_max is None and beta_crit is None
    assert q_min == pytest.approx(1 + kappa / 2, rel=1e-13, abs=0.0)
    assert ip_max == pytest.approx(q * Ip * 2 / (1 + kappa), rel=1e-12, abs=0.0)


def test_kink_safety_factor_conventional_branch_adds_the_shape_factor():
    R, a, kappa, Ip, Bt = 1.8, 0.4, 1.6, 1.0e5, 0.2
    q, _, beta_max, beta_crit, _ = kink_safety_factor(
        R, a, kappa, Ip, Bt, "conventional"
    )
    g = 1 / kappa * (1 + 4 / np.pi**2 * (kappa**2 - 1))
    expected_q = 2 * np.pi * a**2 * kappa * Bt / (MU0 * Ip * R) * g
    assert q == pytest.approx(expected_q, rel=1e-12, abs=0.0)
    eps = a / R
    assert beta_max == pytest.approx(
        np.pi**2 / 16 * kappa * eps / expected_q**2, rel=1e-12, abs=0.0
    )
    assert beta_crit == pytest.approx(
        0.14 * eps * kappa / expected_q, rel=1e-12, abs=0.0
    )


def test_kink_safety_factor_ST_branch_matches_its_closed_form():
    R, a, kappa, Ip, Bt = 0.4, 0.3, 1.8, 1.0e5, 0.18
    q, _, beta_max, beta_crit, _ = kink_safety_factor(R, a, kappa, Ip, Bt, "ST")
    expected_q = 2 * np.pi * a**2 * Bt / (MU0 * Ip * R) * (1 + kappa**2 / 2)
    assert q == pytest.approx(expected_q, rel=1e-12, abs=0.0)
    eps = a / R
    assert beta_max == pytest.approx(
        0.072 * (1 + kappa**2) / 2 * eps, rel=1e-12, abs=0.0
    )
    braket = 0.03 * (expected_q - 1) / ((3 / 4) ** 4 + (expected_q - 1) ** 4) ** 0.25
    assert beta_crit == pytest.approx(
        5 * braket * (1 + kappa**2) / 2 * eps / expected_q, rel=1e-12, abs=0.0
    )


def test_kink_safety_factor_rejects_an_unknown_shape():
    with pytest.raises(ValueError, match="Invalid type"):
        kink_safety_factor(1.8, 0.4, 1.6, 1.0e5, 0.2, "spheromak")


def test_current_limits_from_q_and_from_beta_share_one_closed_form():
    # The beta form is the q form with beta_N substituted for q_95, which is
    # exactly what its docstring says; equal arguments must give equal results.
    a, B0 = 0.4, 0.2
    assert current_limit_from_q(3.0, a, B0) == pytest.approx(
        2 * np.pi * a**2 * B0 / (MU0 * 3.0), rel=1e-12, abs=0.0
    )
    assert current_limit_from_beta(3.0, a, B0) == current_limit_from_q(3.0, a, B0)


def test_normalized_plasma_current_converts_amps_to_megaamps():
    assert normalized_plasma_current(1.0e5, 1.8, 0.4, 0.2) == pytest.approx(
        0.1 / (0.4 * 0.2), rel=1e-13, abs=0.0
    )


def test_normalized_plasma_current_ignores_the_major_radius():
    # R is in the signature but unused; pin that so a change is deliberate.
    assert normalized_plasma_current(1.0e5, 1.8, 0.4, 0.2) == (
        normalized_plasma_current(1.0e5, 99.0, 0.4, 0.2)
    )


# --------------------------------------------------------------------------
# Energies and confinement
# --------------------------------------------------------------------------

def test_stored_energy_from_pressure_is_p_times_V():
    assert stored_energy_from_p_V(1.2e4, 2.5) == pytest.approx(
        3.0e4, rel=1e-13, abs=0.0
    )


def test_stored_energy_from_beta_matches_its_closed_form():
    beta, B0, V = 0.04, 0.2, 2.5
    assert stored_energy_from_beta_V(beta, B0, V) == pytest.approx(
        beta * B0**2 * V / (2 * MU0), rel=1e-12, abs=0.0
    )


def test_the_two_stored_energy_forms_agree_at_the_defining_beta():
    # beta = 2 mu0 p / B0^2 is the definition of toroidal beta, so the two
    # expressions must return the same energy for the same plasma.
    p, B0, V = 1.2e4, 0.2, 2.5
    beta = 2 * MU0 * p / B0**2
    assert stored_energy_from_beta_V(beta, B0, V) == pytest.approx(
        stored_energy_from_p_V(p, V), rel=1e-12, abs=0.0
    )


def test_kinetic_and_magnetic_energy_forms_differ_only_by_the_three_halves():
    # W_K = (3/2) beta_p B_pa^2 V / (2 mu0) and W_M = l_i B_pa^2 V / (2 mu0).
    B_pa, V = 0.21, 2.5
    assert kinetic_energy_from_beta_p_B_pa_V_p(1.0, B_pa, V) == pytest.approx(
        1.5 * magnetic_energy_from_li_B_pa_V_p(1.0, B_pa, V), rel=1e-12, abs=0.0
    )
    assert magnetic_energy_from_li_B_pa_V_p(0.9, B_pa, V) == pytest.approx(
        0.9 * B_pa**2 * V / (2 * MU0), rel=1e-12, abs=0.0
    )


def test_confinement_time_is_stored_energy_over_loss_power():
    assert confinement_time_from_P_loss_W_th(2.0e5, 3.0e3) == pytest.approx(
        0.015, rel=1e-13, abs=0.0
    )


def test_heating_and_loss_powers_compose_into_a_consistent_balance():
    P_ohm, P_aux, dWdt, P_rad = 1.5e5, 4.0e5, 2.0e4, 8.0e4
    P_heat = heating_power_from_p_ohm_p_aux(P_ohm, P_aux)
    assert P_heat == pytest.approx(5.5e5, rel=1e-13, abs=0.0)
    assert loss_power_from_p_heat_dWdt_p_rad(P_heat, dWdt, P_rad) == pytest.approx(
        P_heat - dWdt - P_rad, rel=1e-13, abs=0.0
    )


def test_inductive_voltage_is_the_magnetic_power_per_ampere():
    assert inductive_voltage_from_dW_magdt_I_p(3.0e4, 1.0e5) == pytest.approx(
        0.3, rel=1e-13, abs=0.0
    )


@pytest.mark.parametrize(
    "func", [ec_heating_power_from_I_ec_V_ec, nbi_heating_power_from_I_nbi_V_nbi,
             ohmic_heating_power_from_I_p_V_res]
)
def test_the_three_power_products_are_current_times_voltage(func):
    assert func(12.0, 2.5e4) == pytest.approx(3.0e5, rel=1e-13, abs=0.0)


def test_auxiliary_power_split_conserves_the_input_and_matches_its_ratio():
    P_aux, eta = 4.0e5, 0.6
    P_heat, P_CD = auxiliary_heating_power(P_aux, eta)
    assert P_heat + P_CD == pytest.approx(P_aux, rel=1e-12, abs=0.0)
    assert P_CD == pytest.approx(P_aux / (1 + eta), rel=1e-12, abs=0.0)
    # Recorded, not endorsed: the split runs the wrong way.  P_CD grows as
    # eta_CD falls, so a plasma with no current-drive efficiency at all is
    # told every watt went to current drive and none to heating.  Pinned so
    # a correction is a deliberate change rather than a silent one.
    assert auxiliary_heating_power(P_aux, 0.0) == (0.0, P_aux)


# --------------------------------------------------------------------------
# Radiation
# --------------------------------------------------------------------------

def test_the_three_bremsstrahlung_variants_agree_on_the_same_plasma():
    # First principles, the NRL engineering fit and the pressure form are
    # three routes to one quantity; NRL's rounded 1.69e-38 prefactor sets the
    # 0.3 % spread.  This is the strongest available check on all three.
    for n_e, T_eV in [(1e19, 1000.0), (5e19, 300.0), (2e20, 3000.0)]:
        p = 2 * n_e * T_eV * BOLTZMANN_J_PER_EV  # p = 2 n_e k T, T_i = T_e
        first = bremsstrahlung_power_density_from_Z_eff_n_e_T_e(n_e, T_eV, 2.0)
        nrl = bremsstrahlung_radiation_power_from_z_eff_n_e_t_e(2.0, n_e, T_eV)
        press = bremsstrahlung_power_density_from_T_e_p_Z_eff(T_eV, p, 2.0)
        assert first == pytest.approx(nrl, rel=3e-3)
        assert press == pytest.approx(nrl, rel=3e-3)


def test_both_bremsstrahlung_forms_default_to_a_Z_eff_of_two():
    assert bremsstrahlung_power_density_from_Z_eff_n_e_T_e(1e19, 1000.0) == (
        bremsstrahlung_power_density_from_Z_eff_n_e_T_e(1e19, 1000.0, 2.0)
    )
    assert bremsstrahlung_power_density_from_T_e_p_Z_eff(1000.0, 3204.0) == (
        bremsstrahlung_power_density_from_T_e_p_Z_eff(1000.0, 3204.0, 2.0)
    )
    # ... and are linear in it, so the default is a factor, not an offset.
    assert bremsstrahlung_power_density_from_Z_eff_n_e_T_e(
        1e19, 1000.0, 4.0
    ) == pytest.approx(
        2.0 * bremsstrahlung_power_density_from_Z_eff_n_e_T_e(1e19, 1000.0),
        rel=1e-12,
        abs=0.0,
    )


def test_nrl_bremsstrahlung_matches_its_published_coefficient():
    assert bremsstrahlung_radiation_power_from_z_eff_n_e_t_e(
        2.0, 1e19, 1000.0
    ) == pytest.approx(1.69e-38 * 2.0 * 1e19**2 * np.sqrt(1000.0), rel=1e-12, abs=0.0)


def test_bremsstrahlung_scales_as_density_squared_and_root_temperature():
    base = bremsstrahlung_radiation_power_from_z_eff_n_e_t_e(2.0, 1e19, 1000.0)
    assert bremsstrahlung_radiation_power_from_z_eff_n_e_t_e(
        2.0, 2e19, 1000.0
    ) == pytest.approx(4.0 * base, rel=1e-12, abs=0.0)
    assert bremsstrahlung_radiation_power_from_z_eff_n_e_t_e(
        2.0, 1e19, 4000.0
    ) == pytest.approx(2.0 * base, rel=1e-12, abs=0.0)


def test_pressure_form_bremsstrahlung_matches_its_documented_coefficient():
    T_eV, p, Z = 1000.0, 3204.0, 2.0
    assert bremsstrahlung_power_density_from_T_e_p_Z_eff(T_eV, p, Z) == pytest.approx(
        Z * 0.052 * (p / 1e5) ** 2 / (T_eV * 1e-3) ** 1.5 * 1e6, rel=1e-12, abs=0.0
    )


def test_cyclotron_power_density_matches_its_closed_form():
    # Regression for #753: the body referred to four undefined lower-case
    # names, so the function raised NameError on every call until #368's
    # constants move was finished off.
    from vaft.formula.constants import C_LIGHT, EPS0, ME

    n_e, B_t, T_eV = 1e19, 0.2, 1000.0
    coeff = QE**4 / (3.0 * np.pi * EPS0 * ME**3 * C_LIGHT**3)
    assert cyclotron_synchrotron_power_density_scaling_from_n_e_B_t_T_e(
        n_e, B_t, T_eV
    ) == pytest.approx(coeff * n_e * B_t**2 * T_eV * QE, rel=1e-12, abs=0.0)


def test_cyclotron_power_density_is_quadratic_in_the_toroidal_field():
    base = cyclotron_synchrotron_power_density_scaling_from_n_e_B_t_T_e(
        1e19, 0.2, 1000.0
    )
    assert cyclotron_synchrotron_power_density_scaling_from_n_e_B_t_T_e(
        1e19, 0.4, 1000.0
    ) == pytest.approx(4.0 * base, rel=1e-12, abs=0.0)


# --------------------------------------------------------------------------
# Heuristic scalings
# --------------------------------------------------------------------------

def test_alpha_heating_power_is_the_same_object_as_its_long_name():
    assert alpha_heating_power is alpha_heating_power_from_n_D_n_T_T_keV_V


def test_alpha_heating_power_matches_its_closed_form():
    n_D, n_T, T, V = 0.5, 0.5, 10.0, 2.5
    assert alpha_heating_power(n_D, n_T, T, V) == pytest.approx(
        (n_D * 1e19) * (n_T * 1e19) * SIGMA_V_COEF * T**2 * E_ALPHA * V,
        rel=1e-12,
        abs=0.0,
    )


def test_alpha_heating_power_is_quadratic_in_the_temperature_fit():
    base = alpha_heating_power(0.5, 0.5, 10.0, 2.5)
    assert alpha_heating_power(0.5, 0.5, 20.0, 2.5) == pytest.approx(
        4.0 * base, rel=1e-12, abs=0.0
    )


def test_bootstrap_fraction_matches_its_closed_form():
    n_e, T_e, R0, a, q95 = 1.0, 1.0, 1.8, 0.4, 3.0
    beta_p = 0.4 * n_e * T_e * a / (R0 * q95**2)
    assert bootstrap_current_fraction(n_e, T_e, R0, a, q95) == pytest.approx(
        0.3 * np.sqrt(beta_p), rel=1e-12, abs=0.0
    )


def test_bootstrap_fraction_falls_with_the_edge_safety_factor():
    low = bootstrap_current_fraction(1.0, 1.0, 1.8, 0.4, 2.0)
    high = bootstrap_current_fraction(1.0, 1.0, 1.8, 0.4, 6.0)
    assert high < low


def test_current_drive_efficiency_matches_its_closed_form():
    assert current_drive_efficiency(1.0, 4.0, 2.0) == pytest.approx(
        0.3 * np.sqrt(1.0 * 4.0 / 2.0), rel=1e-12, abs=0.0
    )


def test_current_drive_efficiency_defaults_to_a_hydrogenic_Z_eff():
    assert current_drive_efficiency(1.0, 4.0) == current_drive_efficiency(
        1.0, 4.0, 1.0
    )


def test_poloidal_field_factor_distinguishes_the_per_radian_conventions():
    # The psi_per_radian switch is the branch the cocos-index path never
    # reaches; full-weber storage carries the extra 1/(2 pi).
    assert poloidal_field_factor(None, psi_per_radian=True) == pytest.approx(
        -1.0, rel=1e-13, abs=0.0
    )
    assert poloidal_field_factor(None, psi_per_radian=None) == pytest.approx(
        -1.0, rel=1e-13, abs=0.0
    )
    assert poloidal_field_factor(None, psi_per_radian=False) == pytest.approx(
        -1.0 / (2.0 * np.pi), rel=1e-13, abs=0.0
    )
    # An explicit index carries both halves and overrides the switch.
    assert poloidal_field_factor(11, psi_per_radian=False) == (
        poloidal_field_factor(11, psi_per_radian=True)
    )


def test_collisionality_scaling_defaults_its_scale_factor_to_one():
    args = (0.4, 1e19, 3.0, 0.25, 300.0)
    assert normalized_collisionality_from_a_n_q_epsilon_T(*args) == (
        normalized_collisionality_from_a_n_q_epsilon_T(*args, C=1.0)
    )


def test_collisionality_scaling_matches_its_closed_form():
    a, n, q, eps, T = 0.4, 1e19, 3.0, 0.25, 300.0
    assert normalized_collisionality_from_a_n_q_epsilon_T(
        a, n, q, eps, T, C=2.0
    ) == pytest.approx(2.0 * a * n * q / (eps**2.5 * T**2), rel=1e-12, abs=0.0)


@pytest.mark.parametrize("eps,T", [(0.0, 300.0), (-0.1, 300.0)])
def test_collisionality_scaling_rejects_a_non_positive_epsilon(eps, T):
    with pytest.raises(ValueError, match="epsilon"):
        normalized_collisionality_from_a_n_q_epsilon_T(0.4, 1e19, 3.0, eps, T)


def test_collisionality_scaling_rejects_a_non_positive_temperature():
    with pytest.raises(ValueError, match="T_eV"):
        normalized_collisionality_from_a_n_q_epsilon_T(0.4, 1e19, 3.0, 0.25, 0.0)


def test_collisionality_from_nu_ii_matches_its_closed_form():
    nu, T_i, M_i, R, a, q = 1.0e4, 300.0, 3.34e-27, 1.8, 0.4, 3.0
    assert normalized_collisionality_from_nu_ii_T_i_M_i_R_a_q(
        nu, T_i, M_i, R, a, q
    ) == pytest.approx(
        nu * np.sqrt(M_i / (QE * T_i)) * (R / a) ** 1.5 * q * R, rel=1e-12, abs=0.0
    )


def test_collisionality_from_nu_ii_is_linear_in_the_collision_frequency():
    args = (300.0, 3.34e-27, 1.8, 0.4, 3.0)
    single = normalized_collisionality_from_nu_ii_T_i_M_i_R_a_q(1.0e4, *args)
    doubled = normalized_collisionality_from_nu_ii_T_i_M_i_R_a_q(2.0e4, *args)
    assert doubled == pytest.approx(2.0 * single, rel=1e-12, abs=0.0)


def test_kadomtsev_constraint_matches_its_closed_form():
    mu_rho, mu_beta, mu_nu, a_P = 0.4, 0.2, 0.1, 0.5
    expected = 5.0 + (
        mu_rho * (1 + a_P) - (3 * (mu_rho + 2 * mu_beta - 4 * mu_nu - 2) / 2)
    )
    assert verify_kadomtsev_constraint(mu_rho, mu_beta, mu_nu, a_P) == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


def test_kadomtsev_constraint_returns_five_when_the_bracket_vanishes():
    # The offset from 5 is mu_rho*(1 + a_P) - (3/2)(mu_rho + 2 mu_beta
    # - 4 mu_nu - 2).  With mu_rho = mu_nu = 0 the bracket closes at
    # mu_beta = 1, which pins the factor on mu_beta and the -2 together.
    assert verify_kadomtsev_constraint(0.0, 1.0, 0.0, 0.0) == pytest.approx(
        5.0, rel=1e-12, abs=0.0
    )
    # mu_nu enters the bracket with a factor -4, so -0.5 closes it too and
    # the two cases together fix both coefficients and their signs.
    assert verify_kadomtsev_constraint(0.0, 0.0, -0.5, 0.0) == pytest.approx(
        5.0, rel=1e-12, abs=0.0
    )
