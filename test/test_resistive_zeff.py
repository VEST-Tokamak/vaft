"""Resistive Z_eff on synthetic states: #1214 validation Levels A-C, and the guards.

The formula catalog tests never call a formula (they check docstrings), so
every claim here is numeric: a known Z_eff is put in and must come back out
(Level A), the Spitzer limit must reduce to the uniform-ellipse resistance
(Level B), and the neoclassical models must order and scale as the trapped
fraction says (Level C, on an analytic state; the VEST 48224 version is in
``test_resistive_zeff_ods.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.formula.constants import MU0
from vaft.process.resistive_zeff import (
    ROMERO_VOLTAGE_CONVENTION,
    FluxSurfaceState,
    RomeroBoundaryFlux,
    Smoothing,
    TabulatedConductivity,
    infer_resistive_zeff,
    model_resistance,
    observed_resistance,
    parallel_conductivity,
    resistive_zeff_sensitivity,
    smooth_local_polynomial,
)

R0, A, KAPPA, B0 = 0.4, 0.25, 1.6, 0.1


def _state(*, time=0.0, flat=False, t_e0=100.0, n_e0=1e19, ip=1e5, points=81,
           aspect_scale=1.0):
    """A circular-in-rho, elliptic-in-volume analytic state.

    ``flat`` gives uniform T_e, n_e and current, the uniform-ellipse limit.
    ``aspect_scale`` shrinks the minor radius at fixed R0 (large-aspect limit).
    """
    a = A * aspect_scale
    psi = np.linspace(0.0, 1.0, points)
    rho = np.sqrt(psi)
    volume = 2.0 * np.pi**2 * R0 * a**2 * KAPPA * rho**2
    if flat:
        t_e = np.full(points, t_e0)
        n_e = np.full(points, n_e0)
        shape = np.ones(points)
    else:
        t_e = t_e0 * (1.0 - 0.9 * psi)
        n_e = n_e0 * (1.0 - 0.5 * psi)
        shape = 1.0 - psi
    area = volume / (2.0 * np.pi * R0)
    # Normalise the shape so that int j dA = ip with <J.B> = j B0.
    j_unit = shape
    total = np.trapezoid(j_unit, area) if hasattr(np, "trapezoid") else np.trapz(j_unit, area)
    j = j_unit * ip / total
    return FluxSurfaceState(
        time=time, psi_norm=psi, volume=volume, T_e=t_e, n_e=n_e,
        q=1.0 + 2.0 * psi, r_inboard=R0 - a * rho, r_outboard=R0 + a * rho,
        j_dot_b=j * B0, b2_average=np.full(points, B0**2), I_p=ip,
    )


def _flux(time, ip, psi_b, li_3=0.5):
    return RomeroBoundaryFlux(
        time=time, I_p=ip, psi_boundary=psi_b, li_3=np.full(time.size, li_3), R0=R0,
        flux_normalization="synthetic full Wb", flux_sign=1.0,
    )


def _flat_top(r_p, ip=1e5, samples=11, dt=1e-3):
    t = np.arange(samples) * dt
    return _flux(t, np.full(samples, ip), 0.05 - r_p * ip * t)


# --- Level B: the Spitzer limit ----------------------------------------------------


def test_the_flat_large_aspect_limit_is_the_uniform_ellipse_resistance():
    from vaft.formula.equilibrium import spitzer_resistivity_from_T_e_Z_eff_ln_Lambda
    from vaft.formula.startup import plasma_resistance_uniform_ellipse_from_eta_R0_a_kappa

    state = _state(flat=True)
    got = model_resistance(state, 2.0, model="spitzer_nrl", ln_lambda=17.0).R_p
    eta = spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(100.0, 2.0, 17.0)
    expected = plasma_resistance_uniform_ellipse_from_eta_R0_a_kappa(eta, R0, A, KAPPA)
    assert got == pytest.approx(expected, rel=1e-3)


def test_spitzer_resistance_scales_as_z_over_te_to_the_three_halves():
    state = _state()
    base = model_resistance(state, 1.0, model="spitzer_nrl", ln_lambda=17.0).R_p
    double_z = model_resistance(state, 2.0, model="spitzer_nrl", ln_lambda=17.0).R_p
    hot = model_resistance(state.scaled(T_e=4.0), 1.0, model="spitzer_nrl", ln_lambda=17.0).R_p
    assert double_z / base == pytest.approx(2.0, rel=1e-12)
    assert hot / base == pytest.approx(4.0**-1.5, rel=1e-12)


@pytest.mark.parametrize("model", ["spitzer_nrl", "sauter_spitzer", "sauter", "redl"])
def test_the_resistance_rises_monotonically_with_the_charge(model):
    state = _state()
    values = [model_resistance(state, z, model=model, ln_lambda="sauter").R_p
              for z in np.geomspace(1.0, 6.0, 15)]
    assert np.all(np.diff(values) > 0.0)


def test_the_resistance_is_an_ohm_sized_number_for_vest():
    r = model_resistance(_state(), 2.0, model="spitzer_nrl", ln_lambda=17.0).R_p
    assert 1e-7 < r < 1e-3


# --- Level C: analytic neoclassical comparison ---------------------------------------


def test_trapped_particles_raise_the_resistance_and_vanish_at_large_aspect():
    tight = _state()
    loose = _state(aspect_scale=0.02)
    for state, low, high in ((tight, 1.3, 6.0), (loose, 0.95, 1.25)):
        sp = model_resistance(state, 2.0, model="sauter_spitzer", ln_lambda="sauter").R_p
        for model in ("sauter", "redl"):
            ratio = model_resistance(state, 2.0, model=model, ln_lambda="sauter").R_p / sp
            assert low < ratio < high, (model, ratio)


def test_redl_and_sauter_agree_to_the_refit_difference():
    state = _state()
    sauter = model_resistance(state, 2.0, model="sauter", ln_lambda="sauter").R_p
    redl = model_resistance(state, 2.0, model="redl", ln_lambda="sauter").R_p
    assert abs(redl / sauter - 1.0) < 0.2


def test_a_bootstrap_current_lowers_the_resistive_voltage():
    state = _state()
    with_bs = FluxSurfaceState(**{**state.__dict__, "j_bootstrap_dot_b": 0.1 * state.j_dot_b,
                                  "bootstrap_model": "synthetic 10%"})
    plain = model_resistance(state, 2.0, model="redl", ln_lambda="sauter").R_p
    reduced = model_resistance(with_bs, 2.0, model="redl", ln_lambda="sauter").R_p
    assert reduced == pytest.approx(0.9 * plain, rel=1e-12)


def test_spitzer_nrl_is_the_parallel_coefficient():
    """#1188: 1/eta_par with 5.2e-5, not the 1.98x larger perpendicular value."""
    state = _state(flat=True)
    sigma = parallel_conductivity(state, 1.0, model="spitzer_nrl", ln_lambda=10.0)
    assert sigma[0] == pytest.approx(100.0**1.5 / (5.2e-5 * 10.0), rel=1e-12)


# --- Level A: synthetic algebraic recovery ---------------------------------------------


@pytest.mark.parametrize("model", ["spitzer_nrl", "sauter", "redl"])
@pytest.mark.parametrize("z_true", [1.3, 2.0, 3.7])
def test_a_known_charge_is_recovered_on_a_flat_top(model, z_true):
    states = [_state(time=t) for t in (0.003, 0.005, 0.007)]
    r_true = model_resistance(states[0], z_true, model=model, ln_lambda="sauter").R_p
    observed = observed_resistance(_flat_top(r_true), I_ni=0.0, smoothing=Smoothing("none"))
    result = infer_resistive_zeff(observed, states, model=model, ln_lambda="sauter",
                                  bounds=(1.0, 6.0), weights="uniform")
    assert result.status == "ok"
    assert result.zeff == pytest.approx(z_true, rel=1e-6)
    assert result.quality["monotonic"]


def test_a_known_charge_is_recovered_through_a_current_ramp():
    """V_I is non-zero: the inductive correction must be taken off exactly."""
    z_true = 2.4
    t = np.arange(21) * 1e-3
    ip = 8e4 + 2e6 * t  # 2 MA/s ramp
    li = 0.5
    l_i = 0.5 * MU0 * R0 * li
    states = [_state(time=tk, ip=float(np.interp(tk, t, ip))) for tk in t[3:-3:3]]
    r_true = model_resistance(states[0], z_true, model="redl", ln_lambda=15.0).R_p
    # R_p is profile-shape only here (I_p scales out), so one value holds throughout.
    v_b = r_true * ip + l_i * 2e6
    psi_b = 0.05 - np.concatenate([[0.0], np.cumsum(0.5 * (v_b[1:] + v_b[:-1]) * np.diff(t))])
    observed = observed_resistance(_flux(t, ip, psi_b, li), I_ni=0.0, smoothing=Smoothing("none"))
    assert np.max(observed.inductive_fraction[1:-1]) > 0.05
    result = infer_resistive_zeff(observed, states, model="redl", ln_lambda=15.0,
                                  bounds=(1.0, 6.0), weights="uniform")
    assert result.zeff == pytest.approx(z_true, rel=1e-4)


def test_a_solution_beyond_the_bounds_is_reported_not_clipped():
    states = [_state(time=0.005)]
    r_true = model_resistance(states[0], 5.0, model="spitzer_nrl", ln_lambda=17.0).R_p
    observed = observed_resistance(_flat_top(r_true), I_ni=0.0, smoothing=Smoothing("none"))
    result = infer_resistive_zeff(observed, states, model="spitzer_nrl", ln_lambda=17.0,
                                  bounds=(1.0, 3.0), weights="uniform")
    assert result.status == "bound_hit"
    assert result.quality["bound_hit"]
    assert result.zeff == pytest.approx(3.0, rel=2e-3)


@pytest.mark.parametrize("z_true", [1.02, 7.9])
def test_a_minimum_inside_an_end_scan_interval_is_not_a_bound_hit(z_true):
    """Default bounds (1, 8) and 41 log-spaced points put the first interval at
    1.0-1.053 and the last at 7.6-8.0; a clean hydrogen plasma minimises in the
    first one. The scan's end point being the lowest J is a bracketing fact,
    not a bound hit (cold review 0.8.0 delta-absorb-13-physics F1)."""
    states = [_state(time=0.005)]
    r_true = model_resistance(states[0], z_true, model="spitzer_nrl", ln_lambda=17.0).R_p
    observed = observed_resistance(_flat_top(r_true), I_ni=0.0, smoothing=Smoothing("none"))
    result = infer_resistive_zeff(observed, states, model="spitzer_nrl", ln_lambda=17.0,
                                  bounds=(1.0, 8.0), weights="uniform")
    assert result.zeff == pytest.approx(z_true, rel=1e-6)
    assert result.status == "ok"
    assert not result.quality["bound_hit"]
    assert result.reason is None


def test_a_window_with_no_positive_resistance_is_not_identifiable():
    states = [_state(time=0.005)]
    observed = observed_resistance(_flat_top(-1e-6), I_ni=0.0, smoothing=Smoothing("none"))
    result = infer_resistive_zeff(observed, states, model="redl", ln_lambda="sauter",
                                  bounds=(1.0, 6.0), weights="uniform")
    assert result.status == "not_identifiable"
    assert result.zeff is None and result.reason


def test_noise_gives_a_finite_uncertainty_that_covers_the_truth():
    rng = np.random.default_rng(1214)
    states = [_state(time=t) for t in np.arange(2, 19) * 1e-3]
    r_true = model_resistance(states[0], 2.0, model="redl", ln_lambda=15.0).R_p
    flux = _flat_top(r_true, samples=21)
    noisy = RomeroBoundaryFlux(**{**flux.__dict__,
                                  "psi_boundary": flux.psi_boundary + rng.normal(0, 2e-6, 21)})
    observed = observed_resistance(noisy, I_ni=0.0, smoothing=Smoothing("none"))
    result = infer_resistive_zeff(observed, states, model="redl", ln_lambda=15.0,
                                  bounds=(1.0, 6.0), weights="uniform")
    assert result.estimate["uncertainty"] is not None and result.estimate["uncertainty"] > 0
    assert abs(result.zeff - 2.0) < 4.0 * result.estimate["uncertainty"]


def test_the_result_flattens_to_one_row_and_names_itself_inferred():
    states = [_state(time=0.005)]
    r_true = model_resistance(states[0], 2.0, model="redl", ln_lambda="sauter").R_p
    observed = observed_resistance(_flat_top(r_true), I_ni=0.0, smoothing=Smoothing("none"))
    result = infer_resistive_zeff(observed, states, model="redl", ln_lambda="sauter",
                                  bounds=(1.0, 6.0), weights="uniform")
    row = result.as_row()
    assert "inferred" in row["quantity"]
    assert row["conductivity_model" if "conductivity_model" in row else "model_conductivity_model"] == "redl"
    assert row["provenance_current_source_assumption"].startswith("ohmic")
    assert row["provenance_voltage_convention"] == ROMERO_VOLTAGE_CONVENTION
    assert all(np.ndim(v) == 0 for v in row.values())


# --- Phase F: sensitivity ----------------------------------------------------------------


def test_the_sensitivity_table_moves_the_way_the_physics_says():
    states = [_state(time=t) for t in (0.003, 0.005, 0.007)]
    r_true = model_resistance(states[0], 2.0, model="redl", ln_lambda=15.0).R_p
    nominal, rows = resistive_zeff_sensitivity(
        _flat_top(r_true), states, I_ni=0.0, smoothing=Smoothing("none"),
        model="redl", ln_lambda=15.0, bounds=(1.0, 8.0), weights="uniform",
    )
    assert nominal.zeff == pytest.approx(2.0, rel=1e-5)
    by = {(r["input"], r["setting"]): r for r in rows}
    # Hotter plasma conducts better, so more charge is needed for the same R_p.
    assert by[("T_e", "x1.1")]["delta"] > 0.0 > by[("T_e", "x0.9")]["delta"]
    # Spitzer has no trapped-particle penalty: it needs a larger charge.
    assert by[("conductivity_model", "spitzer_nrl")]["delta"] > 0.5
    assert "max_abs_delta_T_e" in nominal.sensitivity


# --- guards -------------------------------------------------------------------------------


def test_a_bare_voltage_array_is_refused():
    with pytest.raises(TypeError, match="RomeroBoundaryFlux"):
        observed_resistance(np.ones(5), I_ni=0.0, smoothing=Smoothing("none"))


def test_a_flux_in_another_convention_cannot_be_built():
    t = np.arange(5) * 1e-3
    with pytest.raises(ValueError, match="not Romero"):
        RomeroBoundaryFlux(time=t, I_p=np.ones(5), psi_boundary=t, li_3=np.ones(5), R0=R0,
                           flux_normalization="x", flux_sign=1.0,
                           voltage_convention="ejima:+2pi dpsi/dt")


def test_the_noninductive_current_and_smoothing_have_no_default():
    flux = _flat_top(1e-5)
    with pytest.raises(TypeError):
        observed_resistance(flux, smoothing=Smoothing("none"))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        observed_resistance(flux, I_ni=0.0)  # type: ignore[call-arg]


def test_states_are_matched_by_time_never_by_index():
    observed = observed_resistance(_flat_top(1e-5), I_ni=0.0, smoothing=Smoothing("none"))
    with pytest.raises(ValueError, match="matched by time"):
        infer_resistive_zeff(observed, [_state(time=0.0105)], model="redl", ln_lambda=15.0,
                             bounds=(1.0, 6.0), weights="uniform")


def test_bounds_and_weights_are_explicit():
    observed = observed_resistance(_flat_top(1e-5), I_ni=0.0, smoothing=Smoothing("none"))
    with pytest.raises(ValueError, match="bounds"):
        infer_resistive_zeff(observed, [_state(time=0.005)], model="redl", ln_lambda=15.0,
                             bounds=(0.5, 6.0), weights="uniform")
    with pytest.raises(ValueError, match="weights"):
        infer_resistive_zeff(observed, [_state(time=0.005)], model="redl", ln_lambda=15.0,
                             bounds=(1.0, 6.0), weights="inverse")


def test_local_polynomial_smoothing_keeps_a_polynomial_and_refuses_a_thin_window():
    t = np.sort(np.random.default_rng(0).uniform(0.0, 0.02, 40))
    x = 3.0 + 2.0 * t - 50.0 * t**2
    np.testing.assert_allclose(smooth_local_polynomial(t, x, window_s=0.004, order=2), x,
                               rtol=1e-10)
    with pytest.raises(ValueError, match="widen the window"):
        smooth_local_polynomial(t[:4], x[:4], window_s=1e-5, order=2)


def test_a_tabulated_conductivity_is_held_to_its_own_charge():
    state = _state()
    sigma = parallel_conductivity(state, 1.0, model="sauter", ln_lambda=15.0)
    table = TabulatedConductivity("table", state.psi_norm, sigma, z_eff=1.0)
    same = model_resistance(state, 1.0, model=table, ln_lambda=15.0)
    assert same.R_p == pytest.approx(
        model_resistance(state, 1.0, model="sauter", ln_lambda=15.0).R_p, rel=1e-12)
    assert same.conductivity_model == "table"
    with pytest.raises(ValueError, match="cannot be evaluated"):
        model_resistance(state, 2.0, model=table, ln_lambda=15.0)


# --- #1188: no hidden Spitzer inputs ----------------------------------------------------


def test_omitting_the_spitzer_inputs_warns_and_names_the_issue():
    from vaft.formula.equilibrium import spitzer_resistivity_from_T_e_Z_eff_ln_Lambda

    with pytest.warns(FutureWarning, match="#1188"):
        implied = spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(100.0)
    assert implied == pytest.approx(spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(100.0, 2.0, 17.0))


def test_the_spitzer_coefficient_is_the_nrl_parallel_one():
    from vaft.formula.equilibrium import spitzer_resistivity_from_T_e_Z_eff_ln_Lambda

    eta = spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(1000.0, 1.0, 1.0)
    assert eta == pytest.approx(1.65e-9, rel=0.01)  # NRL eta_par, T in keV
    assert eta < 1.03e-4 / 1000.0**1.5 / 1.5  # far below the perpendicular value


def test_the_ohmic_power_wrapper_warns_without_its_spitzer_inputs():
    pytest.importorskip("omas")
    pytest.importorskip("skimage")
    from vaft.omas.process_wrapper import compute_ohmic_heating_power_from_core_profiles
    from vaft.omas.sample import sample_ods

    with pytest.warns(FutureWarning, match="#1188"):
        compute_ohmic_heating_power_from_core_profiles(sample_ods(48224), time_slice=0)


def test_non_positive_samples_are_kept_and_counted_not_dropped():
    """Dropping only the negative side of the noise would bias Z upward (cold review)."""
    states = [_state(time=t) for t in (0.002, 0.003, 0.005, 0.007)]
    r_true = model_resistance(states[0], 2.0, model="redl", ln_lambda=15.0).R_p
    flux = _flat_top(r_true)
    v = r_true * 1e5
    kick = np.zeros(flux.time.size)
    kick[4] = 2.5 * v * 1e-3  # central differences: V_B at 3 ms negative, at 5 ms raised
    bent = RomeroBoundaryFlux(**{**flux.__dict__, "psi_boundary": flux.psi_boundary + kick})
    observed = observed_resistance(bent, I_ni=0.0, smoothing=Smoothing("none"))
    assert "R_p_nonpositive" in observed.flags[3]
    result = infer_resistive_zeff(observed, states, model="redl", ln_lambda=15.0,
                                  bounds=(1.0, 8.0), weights="uniform")
    assert result.quality["n_nonpositive_samples"] == 1
    assert result.quality["n_samples"] == 4


def test_the_parameters_variant_warns_at_the_callers_line():
    import warnings

    from vaft.process.equilibrium import resistive_layer_parameters

    psi = np.linspace(0.0, 1.0, 32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resistive_layer_parameters(psi, 2.0 + 10.0 * psi**2, 1, t_e=100.0 * (1.1 - psi),
                                   n_e=np.full(32, 1e19))
    deprecations = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert deprecations and deprecations[0].filename == __file__
