"""Turbulence--zonal-flow process layer (#1820) on synthetic spectra and Lotka--Volterra orbits."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vaft.formula import turbulence as pp
from vaft.process import gyrokinetics as gk

TRUE = dict(gamma_eff=0.3, coupling_suppression=0.5, coupling_drive=0.2, gamma_zonal=0.1)
NORMALIZATION = {
    "turbulence_definition": "sum_{ky!=0} |phi|^2", "zonal_definition": "sum_{ky=0} |phi|^2",
    "potential_normalization": "synthetic", "time_normalization": "arbitrary",
    "spectral_coordinates": "none",
}


def orbit(n0, e0, t_end, n_samples=1500, params=TRUE):
    t = np.linspace(0.0, t_end, n_samples)
    sol = solve_ivp(lambda _, y: pp.predator_prey_rhs(y[0], y[1], **params), (0, t_end),
                    [n0, e0], t_eval=t, method="LSODA", rtol=1e-10, atol=1e-12)
    return t, sol.y[0], sol.y[1]


# -- intensities ---------------------------------------------------------------------


def spectrum():
    ky = np.array([0.0, 0.1, 0.2])
    kx = np.array([-0.2, 0.0, 0.2, 0.4])
    phi = np.zeros((4, 3, 2), dtype=complex)
    phi[:, 0, :] = 1.0 + 1.0j          # |phi|^2 = 2 on every zonal kx
    phi[:, 1, :] = 0.5                 # 0.25
    phi[:, 2, 1] = 1.0                 # 1 at the second time only
    return phi, kx, ky


def test_the_intensity_split_counts_ky0_as_zonal_and_the_rest_as_turbulence():
    phi, _, ky = spectrum()
    out = gk.zonal_turbulence_intensity(phi, ky)
    assert out.zonal.tolist() == pytest.approx([8.0, 8.0])
    assert out.turbulence.tolist() == pytest.approx([1.0, 5.0])
    assert out.ratio.tolist() == pytest.approx([8.0, 1.6])
    assert "proxy" in out.weighting


def test_weights_enter_and_are_labelled():
    phi, kx, ky = spectrum()
    weights = (kx[:, None] ** 2 + ky[None, :] ** 2)
    out = gk.zonal_turbulence_intensity(phi, ky, weights=weights, weighting="k_perp^2")
    assert out.zonal[0] == pytest.approx(2.0 * np.sum(kx ** 2))
    assert out.weighting == "k_perp^2"


def test_no_zonal_column_and_wrong_shapes_are_refused():
    phi, kx, ky = spectrum()
    with pytest.raises(ValueError, match="ky = 0"):
        gk.zonal_turbulence_intensity(phi, ky + 0.05)
    with pytest.raises(ValueError, match="n_kx, n_ky, n_time"):
        gk.zonal_turbulence_intensity(phi[..., 0], ky)
    with pytest.raises(ValueError):
        gk.zonal_turbulence_intensity(phi, ky[:2])


def test_zonal_shear_proxy_weights_kx_squared_and_to_the_fourth():
    phi, kx, ky = spectrum()
    out = gk.zonal_shear_proxy(phi, kx, ky)
    assert out.flow[0] == pytest.approx(2.0 * np.sum(kx ** 2))
    assert out.shear[0] == pytest.approx(2.0 * np.sum(kx ** 4))
    assert "proxy" in out.normalization


# -- cycles --------------------------------------------------------------------------


def test_a_small_orbit_gives_the_linearized_period_and_quarter_period_lags():
    n_star, e_star = pp.predator_prey_fixed_point(**TRUE)
    period = pp.predator_prey_period(TRUE["gamma_eff"], TRUE["gamma_zonal"])
    t, n, e = orbit(n_star * 1.05, e_star, 5 * period, n_samples=5000)
    m = gk.predator_prey_cycle_metrics(t, n, e)
    assert m.qualified and m.n_cycles >= 3
    assert m.period_mean == pytest.approx(period, rel=0.01)
    assert m.peak_response_lag_mean == pytest.approx(period / 4, rel=0.03)
    assert m.correlation_lag == pytest.approx(period / 4, rel=0.03)


def test_lags_are_negative_when_the_roles_are_swapped():
    n_star, e_star = pp.predator_prey_fixed_point(**TRUE)
    period = pp.predator_prey_period(TRUE["gamma_eff"], TRUE["gamma_zonal"])
    t, n, e = orbit(n_star * 1.05, e_star, 5 * period, n_samples=5000)
    swapped = gk.predator_prey_cycle_metrics(t, e, n)      # "zonal" now leads
    assert swapped.correlation_lag < 0


def test_too_few_cycles_is_not_qualified_and_says_why():
    t, n, e = orbit(1.0, 0.3, 30.0)
    m = gk.predator_prey_cycle_metrics(t, n, e, min_cycles=5)
    assert not m.qualified
    assert any("min_cycles" in reason for reason in m.reasons)


def test_transport_is_summarised_per_cycle_not_used_as_the_turbulence():
    n_star, e_star = pp.predator_prey_fixed_point(**TRUE)
    period = pp.predator_prey_period(TRUE["gamma_eff"], TRUE["gamma_zonal"])
    t, n, e = orbit(n_star * 1.5, e_star, 4 * period, n_samples=3000)
    flux = 2.0 * n
    m = gk.predator_prey_cycle_metrics(t, n, e, transport=flux)
    cycles = m.transport_cycles["cycles"]
    assert len(cycles) == m.n_cycles
    assert cycles[0]["max"] == pytest.approx(2.0 * n[(t >= cycles[0]["start"]) & (t <= cycles[0]["stop"])].max())


# -- fit -----------------------------------------------------------------------------


def test_the_trajectory_fit_recovers_the_coefficients_from_a_noisy_orbit():
    rng = np.random.default_rng(1820)
    t, n, e = orbit(1.2, 0.3, 120.0, n_samples=600)
    n_obs = n * np.exp(0.02 * rng.standard_normal(n.size))
    e_obs = e * np.exp(0.02 * rng.standard_normal(e.size))
    fit = gk.fit_predator_prey_model(t, n_obs, e_obs, normalization=NORMALIZATION)
    assert fit.success, fit.reasons
    assert fit.gamma_eff == pytest.approx(TRUE["gamma_eff"], rel=0.05)
    assert fit.coupling_suppression == pytest.approx(TRUE["coupling_suppression"], rel=0.05)
    assert fit.coupling_drive == pytest.approx(TRUE["coupling_drive"], rel=0.05)
    assert fit.gamma_zonal == pytest.approx(TRUE["gamma_zonal"], rel=0.05)
    assert fit.rms_turbulence < 0.1 and fit.rms_zonal < 0.1
    assert fit.normalization == NORMALIZATION and fit.reasons == ()


def test_rescaling_the_intensities_moves_only_the_couplings():
    t, n, e = orbit(1.2, 0.3, 120.0, n_samples=600)
    base = gk.fit_predator_prey_model(t, n, e, normalization=NORMALIZATION)
    scaled = gk.fit_predator_prey_model(t, 10.0 * n, 4.0 * e, normalization=NORMALIZATION)
    assert scaled.gamma_eff == pytest.approx(base.gamma_eff, rel=1e-3)
    assert scaled.gamma_zonal == pytest.approx(base.gamma_zonal, rel=1e-3)
    assert scaled.coupling_drive == pytest.approx(base.coupling_drive / 10.0, rel=1e-3)
    assert scaled.coupling_suppression == pytest.approx(base.coupling_suppression / 4.0, rel=1e-3)


def test_a_missing_normalisation_record_is_reported():
    t, n, e = orbit(1.2, 0.3, 60.0, n_samples=300)
    fit = gk.fit_predator_prey_model(t, n, e)
    assert any("normalisation not recorded" in reason for reason in fit.reasons)


def test_a_window_restricts_the_fit():
    t, n, e = orbit(1.2, 0.3, 120.0, n_samples=600)
    fit = gk.fit_predator_prey_model(t, n, e, window=(20.0, 80.0), normalization=NORMALIZATION)
    assert fit.time_window[0] >= 20.0 and fit.time_window[1] <= 80.0
    assert fit.turbulence_fit.size == np.count_nonzero((t >= 20.0) & (t <= 80.0))


def test_invalid_inputs_are_refused():
    t, n, e = orbit(1.2, 0.3, 30.0, n_samples=100)
    with pytest.raises(ValueError, match="positive"):
        gk.fit_predator_prey_model(t, n - n.max(), e)
    with pytest.raises(ValueError, match="method"):
        gk.fit_predator_prey_model(t, n, e, method="per_capita")
    with pytest.raises(ValueError, match="increasing"):
        gk.predator_prey_cycle_metrics(t[::-1], n, e)


def test_log_scale_finds_bursts_a_linear_range_misses():
    """A large burst followed by smaller ones (a trace spanning decades): with a
    prominence fraction of the linear range only the large one passes."""
    t = np.linspace(0.0, 400.0, 4001)
    amp = np.array([100.0, 3.0, 3.0, 3.0])
    n = 0.01 + sum(a * np.exp(-((t - 50 - 100 * k) / 8.0) ** 2) for k, a in enumerate(amp))
    e = 0.01 + sum(a * np.exp(-((t - 75 - 100 * k) / 8.0) ** 2) for k, a in enumerate(amp))
    linear = gk.predator_prey_cycle_metrics(t, n, e, scale="linear")
    log = gk.predator_prey_cycle_metrics(t, n, e, scale="log")
    assert linear.n_cycles == 0
    assert log.n_cycles == 3
    assert log.period_mean == pytest.approx(100.0, rel=0.01)
    assert log.peak_response_lag_mean == pytest.approx(25.0, rel=0.02)
