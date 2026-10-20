"""Transport and turbulence reference relations (#1112): analytic limits, signs and gyro-Bohm units."""

import numpy as np
import pytest

from vaft.formula import transport as tr
from vaft.formula import turbulence as tb
from vaft.formula.constants import QE
from vaft.formula.ordering import sound_gyroradius
from vaft.formula.particle import exb_drift_velocity
from vaft.formula.sol import ion_sound_speed

M_D = 3.3435837768e-27  # deuteron mass [kg], as GACODE normalises


def test_mean_square_displacement_reproduces_a_random_walk():
    rng = np.random.default_rng(1112)
    step, tau, n_steps = 2e-3, 1e-5, 400
    walks = np.cumsum(rng.choice([-step, step], size=(4000, n_steps)), axis=1)
    t = tau * np.arange(1, n_steps + 1)
    rate = np.polyfit(t, np.mean(walks ** 2, axis=0), 1)[0]
    assert tr.diffusivity_from_mean_square_displacement(rate) == pytest.approx(step ** 2 / (2 * tau), rel=0.05)
    assert tr.diffusivity_from_mean_square_displacement(6.0, dimensions=3) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        tr.diffusivity_from_mean_square_displacement(1.0, dimensions=0)


def test_taylor_green_kubo_reproduces_the_exponential_correlation():
    variance, tau_c = 4.0e6, 2.5e-6
    lag = np.linspace(0.0, 40 * tau_c, 4001)
    d = tr.diffusivity_from_velocity_autocorrelation(lag, variance * np.exp(-lag / tau_c))
    assert d == pytest.approx(variance * tau_c, rel=1e-5)
    with pytest.raises(ValueError):
        tr.diffusivity_from_velocity_autocorrelation(lag[1:], np.ones(lag.size - 1))


def test_particle_flux_signs_follow_the_gradient_and_convection_convention():
    assert tr.particle_flux_from_D_grad_n_V_n(1.0, -1e20, 1e19, 0.0) == pytest.approx(1e20)  # peaked: outward
    assert tr.particle_flux_from_D_grad_n_V_n(0.0, -1e20, 1e19, -2.0) == pytest.approx(-2e19)  # pinch: inward
    assert tr.conductive_heat_flux_from_n_chi_grad_T(1e19, 1.0, -1e3) == pytest.approx(1e22 * QE)
    with pytest.raises(ValueError):
        tr.particle_flux_from_D_grad_n_V_n(-1.0, 0.0, 1e19, 0.0)


def test_bohm_and_gyro_bohm_units():
    # T/(eB) in m^2/s for T in eV: 100 eV at 1 T gives 100 m^2/s, Bohm's 1/16 of it
    assert tr.bohm_diffusivity_from_T_B(100.0, 1.0) == pytest.approx(100.0 / 16.0)
    assert tr.bohm_diffusivity_from_T_B(100.0, 1.0, coefficient=1.0) == pytest.approx(100.0)
    assert tr.classical_diffusivity_from_nu_rho(1e5, 1e-3) == pytest.approx(0.1)
    rho_s, c_s, a = 2e-3, 1e5, 0.4
    chi = tr.gyrobohm_diffusivity_from_rho_s_c_s_L_ref(rho_s, c_s, a)
    assert chi == pytest.approx(rho_s ** 2 * c_s / a)
    assert tr.gyrobohm_diffusivity_from_rho_s_c_s_L_ref(-rho_s, c_s, a) == pytest.approx(chi)  # signed B_unit


def test_generic_gyro_bohm_units_match_the_tglf_normalisation():
    from vaft.code.gacode.tglf.inputs import TGLFNormalisation

    n_e, t_e_ev, b_unit, a = 2e19, 150.0, -0.3, 0.4
    c_s = ion_sound_speed(t_e_ev, 0.0, M_D)
    rho_s = -sound_gyroradius(t_e_ev, M_D, abs(b_unit))  # signed as B_unit is
    norm = TGLFNormalisation(electron_density=n_e, electron_temperature=t_e_ev * QE, sound_speed=c_s,
                             gyroradius=rho_s, minor_radius=a, b_unit=b_unit)
    assert tr.gyrobohm_particle_flux_from_n_c_s_rho_s_L_ref(n_e, c_s, rho_s, a) == pytest.approx(norm.particle_flux)
    assert tr.gyrobohm_energy_flux_from_n_T_c_s_rho_s_L_ref(n_e, t_e_ev, c_s, rho_s, a) == pytest.approx(
        norm.energy_flux)


def test_mixing_length_scales_as_gamma_over_k_perp_squared():
    d = tb.mixing_length_diffusivity_from_gamma_k_perp(1e5, 100.0)
    assert d == pytest.approx(10.0)
    assert tb.mixing_length_diffusivity_from_gamma_k_perp(2e5, 100.0) == pytest.approx(2 * d)
    assert tb.mixing_length_diffusivity_from_gamma_k_perp(1e5, 200.0) == pytest.approx(d / 4)


def test_cross_phase_flux_vanishes_in_phase_and_reverses_with_the_phase():
    args = (1e17, 10.0, 50.0, 1.0)
    assert tb.coherent_particle_flux_from_amplitudes_phase(*args, 0.0) == pytest.approx(0.0, abs=1e-6)
    plus = tb.coherent_particle_flux_from_amplitudes_phase(*args, 0.6)
    assert plus > 0 and tb.coherent_particle_flux_from_amplitudes_phase(*args, -0.6) == pytest.approx(-plus)


def test_cross_phase_flux_is_the_period_average_of_the_exb_correlation():
    n_amp, phi_amp, k_y, b, alpha_n, alpha_phi = 1e17, 10.0, 50.0, 1.0, 0.2, 1.1
    y = np.linspace(0.0, 2 * np.pi / k_y, 2000, endpoint=False)
    n_fluct = n_amp * np.cos(k_y * y + alpha_n)
    grad_phi = np.zeros((y.size, 3))
    grad_phi[:, 1] = -k_y * phi_amp * np.sin(k_y * y + alpha_phi)
    v_x = tb.exb_velocity_from_potential_gradient(grad_phi, [0.0, 0.0, b])[:, 0]
    assert tb.turbulent_particle_flux_from_fluctuations(n_fluct, v_x) == pytest.approx(
        tb.coherent_particle_flux_from_amplitudes_phase(n_amp, phi_amp, k_y, b, alpha_phi - alpha_n), rel=1e-9)


def test_exb_velocity_reuses_the_particle_drift_owner():
    grad_phi, b = np.array([3.0, -1.0, 0.5]), np.array([0.1, 0.2, 1.5])
    np.testing.assert_allclose(tb.exb_velocity_from_potential_gradient(grad_phi, b),
                               exb_drift_velocity(-grad_phi, b))


def test_diamagnetic_drift_is_opposite_for_ions_and_electrons():
    grad_p, b = [-1e3, 0.0, 0.0], [0.0, 0.0, 1.0]  # pressure falls outward along +x
    v_i = tb.diamagnetic_velocity_from_grad_p_q_n_B(grad_p, QE, 1e19, b)
    v_e = tb.diamagnetic_velocity_from_grad_p_q_n_B(grad_p, -QE, 1e19, b)
    np.testing.assert_allclose(v_i, -v_e)
    assert v_i[1] == pytest.approx(-1e3 / (QE * 1e19))  # B x grad p = z x (-x) = -y
    assert tb.diamagnetic_frequency_from_k_v_star([0.0, 2.0, 0.0], v_i) == pytest.approx(2 * v_i[1])


def test_heat_flux_is_three_halves_of_the_pressure_correlation():
    p, v = np.array([1.0, -1.0, 2.0]), np.array([2.0, -2.0, 1.0])
    assert tb.turbulent_heat_flux_from_fluctuations(p, v) == pytest.approx(1.5 * np.mean(p * v))


def test_reynolds_stress_sign_follows_the_velocity_convention():
    v_r, v_theta = np.array([1.0, -1.0]), np.array([2.0, -2.0])
    stress = tb.reynolds_stress_from_velocity_fluctuations(v_r, v_theta)
    assert stress == pytest.approx(2.0)
    assert tb.reynolds_stress_from_velocity_fluctuations(v_r, -v_theta) == pytest.approx(-stress)
    assert tb.zonal_flow_drive_from_reynolds_stress_gradient(-3.0) == pytest.approx(3.0)
    assert tb.zonal_flow_drive_from_reynolds_stress_gradient(-3.0, 2.0, 0.5) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        tb.zonal_flow_drive_from_reynolds_stress_gradient(0.0, 1.0, -1.0)


def test_shearing_rate_is_a_magnitude_and_the_ratio_is_not_a_classifier():
    assert tb.exb_shearing_rate_from_velocity_gradient(-4e4) == pytest.approx(4e4)
    np.testing.assert_allclose(tb.exb_shearing_rate_from_velocity_gradient([1.0, -2.0]), [1.0, 2.0])
    ratio = tb.growth_to_shear_rate_ratio(4e4, 2e4)
    assert ratio == pytest.approx(2.0) and isinstance(ratio, float)
    with pytest.raises(ValueError):
        tb.growth_to_shear_rate_ratio(1.0, 0.0)
