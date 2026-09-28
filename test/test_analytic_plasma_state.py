"""Analytic L-/H-mode/ITB plasma states on a fixed geometry (#1045)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vaft.data import AnalyticPlasmaState, BarrierStep, KineticProfiles
from vaft.formula.constants import QE
from vaft.process.equilibrium import solovev_example
from vaft.process.profile import (
    analytic_hmode_itb_state,
    analytic_hmode_state,
    analytic_itb_state,
    analytic_lmode_state,
    compose_analytic_profile,
    compose_plasma_state,
    evaluate_analytic_profile,
    evaluate_plasma_state,
    project_flux_function,
    project_plasma_state,
)

PRESETS = (analytic_lmode_state, analytic_hmode_state, analytic_itb_state, analytic_hmode_itb_state)
FINE = np.linspace(0.0, 1.0, 4001)


@pytest.fixture(scope="module")
def limited():
    return solovev_example("limited", resolution=65)


@pytest.fixture(scope="module")
def single_null():
    return solovev_example("single_null", resolution=65, major_radius=0.45, elongation=1.9)


# --- requested values ---------------------------------------------------------------


@pytest.mark.parametrize("preset", PRESETS)
def test_axis_and_separatrix_values_hold_exactly(preset):
    state = preset(te_axis=300.0, te_sep=15.0, ne_axis=3e19, ne_sep=4e18)
    assert state.T_e[0] == pytest.approx(300.0, rel=1e-12)
    assert state.T_e[-1] == pytest.approx(15.0, rel=1e-12)
    assert state.n_e[0] == pytest.approx(3e19, rel=1e-12)
    assert state.n_e[-1] == pytest.approx(4e18, rel=1e-12)
    assert state.psi_norm[0] == 0.0 and state.psi_norm[-1] == 1.0


@pytest.mark.parametrize("preset", (analytic_hmode_state, analytic_hmode_itb_state))
def test_pedestal_top_holds_at_the_knee(preset):
    state = preset(pedestal_position=0.92, pedestal_width=0.06, te_ped=90.0, ne_ped=1.1e19, ti_ped=70.0)
    knee = 0.92 - 0.03
    for name, value in (("T_e", 90.0), ("n_e", 1.1e19), ("T_i", 70.0)):
        assert evaluate_plasma_state(state, knee, name) == pytest.approx(value, rel=1e-12)
        assert state.profiles[name].pedestal.knee == pytest.approx(knee)


def test_barrier_position_is_the_steepest_point_and_width_is_the_full_width():
    step = BarrierStep(position=0.45, width=0.08, height=1.0)
    profile = compose_analytic_profile(
        "T_e", axis_value=2.0, separatrix_value=1.0, itb_height=step.height,
        itb_position=step.position, itb_width=step.width,
    )
    # the whole axis-to-separatrix drop is the barrier, so the profile is the step alone
    assert profile.core_amplitude == 0.0
    gradient = evaluate_analytic_profile(profile, FINE, derivative=True)
    assert FINE[np.argmin(gradient)] == pytest.approx(0.45, abs=FINE[1])
    values = evaluate_analytic_profile(profile, FINE) - 1.0
    # Groebner full width: tanh(+-1) at the knee and foot, i.e. 88% and 12% of the rise
    upper = 0.5 * (1 + np.tanh(1.0))
    knee = np.interp(-upper, -values, FINE)
    foot = np.interp(-(1 - upper), -values, FINE)
    assert foot - knee == pytest.approx(0.08, rel=2e-3)
    assert 0.5 * (knee + foot) == pytest.approx(0.45, abs=1e-4)


def test_itb_is_independent_per_channel():
    state = analytic_itb_state(
        itb_position={"n_e": 0.25, "T_e": 0.35, "T_i": 0.5},
        itb_width={"n_e": 0.05, "T_e": 0.08, "T_i": 0.1},
        ne_itb_height=0.0,
    )
    assert state.profiles["n_e"].itb is None
    assert state.profiles["T_e"].itb.position == 0.35
    assert state.profiles["T_i"].itb.position == 0.5 and state.profiles["T_i"].itb.width == 0.1
    for name, where in (("T_e", 0.35), ("T_i", 0.5)):
        smooth = analytic_lmode_state()
        excess = (np.abs(evaluate_analytic_profile(state.profiles[name], FINE, derivative=True))
                  - np.abs(evaluate_analytic_profile(smooth.profiles[name], FINE, derivative=True)))
        assert FINE[np.argmax(excess)] == pytest.approx(where, abs=0.03)


# --- continuity, derivatives, positivity ------------------------------------------------


@pytest.mark.parametrize("preset", PRESETS)
def test_profiles_are_finite_positive_and_continuous(preset):
    state = preset(psi_norm=FINE)
    for name in ("n_e", "n_i", "T_e", "T_i", "p_e", "p_i", "p_total", "dp_total_dpsi_norm"):
        values = getattr(state, name)
        assert np.all(np.isfinite(values)), name
    for name in ("n_e", "n_i", "T_e", "T_i", "p_total"):
        assert np.all(getattr(state, name) > 0.0), name
        # a bounded analytic gradient bounds the jump between neighbouring samples
        bound = np.max(np.abs(np.gradient(getattr(state, name), FINE))) * FINE[1] * 1.01
        assert np.max(np.abs(np.diff(getattr(state, name)))) <= bound


@pytest.mark.parametrize("preset", PRESETS)
def test_analytic_derivatives_match_finite_differences(preset):
    state = preset()
    x = np.linspace(0.02, 0.98, 97)
    h = 1e-6
    for name, profile in state.profiles.items():
        numeric = (evaluate_analytic_profile(profile, x + h) - evaluate_analytic_profile(profile, x - h)) / (2 * h)
        analytic = evaluate_analytic_profile(profile, x, derivative=True)
        scale = np.max(np.abs(analytic))
        np.testing.assert_allclose(analytic, numeric, rtol=0, atol=1e-6 * scale, err_msg=name)
    numeric = (evaluate_plasma_state(state, x + h) - evaluate_plasma_state(state, x - h)) / (2 * h)
    analytic = evaluate_plasma_state(state, x, "dp_total_dpsi_norm")
    np.testing.assert_allclose(analytic, numeric, rtol=0, atol=1e-6 * np.max(np.abs(analytic)))


def test_non_positive_profiles_are_refused():
    with pytest.raises(ValueError, match="not positive"):
        compose_analytic_profile("T_e", axis_value=100.0, separatrix_value=-1.0)
    with pytest.raises(ValueError, match="not positive"):
        # an ITB rise larger than the whole axis-to-edge drop digs below zero outside it
        analytic_itb_state(te_axis=100.0, te_sep=5.0, te_itb_height=300.0, itb_position=0.5)


def test_malformed_barriers_are_refused():
    with pytest.raises(ValueError, match="pedestal needs"):
        compose_analytic_profile("T_e", axis_value=100.0, separatrix_value=1.0, pedestal_top_value=50.0)
    with pytest.raises(ValueError, match="knee"):
        compose_analytic_profile("T_e", axis_value=100.0, separatrix_value=1.0, pedestal_top_value=50.0,
                                 pedestal_position=0.02, pedestal_width=0.1)
    with pytest.raises(ValueError, match="itb_position"):
        compose_analytic_profile("T_e", axis_value=100.0, separatrix_value=1.0, itb_height=10.0,
                                 itb_position=1.2, itb_width=0.05)
    with pytest.raises(ValueError, match="at least one"):
        compose_analytic_profile("T_e", axis_value=100.0, separatrix_value=1.0, core_alpha=0.5)


# --- pressure is derived from the stored state ---------------------------------------------


@pytest.mark.parametrize("z_eff", (1.0, 1.8))
def test_pressure_is_n_k_t_of_the_stored_profiles(z_eff):
    state = analytic_hmode_itb_state(z_eff=z_eff, impurity_charge=6.0)
    np.testing.assert_allclose(state.p_e, QE * state.n_e * state.T_e, rtol=1e-12)
    np.testing.assert_allclose(state.p_i, QE * (state.n_i + state.n_impurity) * state.T_i, rtol=1e-12)
    np.testing.assert_allclose(state.p_total, state.p_e + state.p_i, rtol=1e-12)
    # quasi-neutrality and the effective charge of the declared composition
    np.testing.assert_allclose(state.n_i + 6.0 * state.n_impurity, state.n_e, rtol=1e-12)
    np.testing.assert_allclose((state.n_i + 36.0 * state.n_impurity) / state.n_e, z_eff, rtol=1e-12)
    if z_eff == 1.0:
        np.testing.assert_array_equal(state.n_i, state.n_e)


def test_state_is_sealed_and_converts_to_the_kinetic_container():
    state = analytic_hmode_state()
    with pytest.raises(ValueError):
        state.T_e[0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.label = "other"
    kinetic = state.to_kinetic_profiles()
    assert isinstance(kinetic, KineticProfiles)
    np.testing.assert_array_equal(kinetic.p_total, state.p_total)
    np.testing.assert_array_equal(kinetic.n_z, state.n_impurity)
    np.testing.assert_allclose(state.rho_pol_norm ** 2, state.psi_norm)
    np.testing.assert_allclose(state.dp_total_drho_pol_norm,
                               2 * np.sqrt(state.psi_norm) * state.dp_total_dpsi_norm)


def test_compose_refuses_a_profile_in_the_wrong_slot():
    t = compose_analytic_profile("T_e", axis_value=100.0, separatrix_value=5.0)
    n = compose_analytic_profile("n_e", axis_value=1e19, separatrix_value=1e18)
    with pytest.raises(ValueError, match="slot"):
        compose_plasma_state(t, n, t)
    with pytest.raises(ValueError):
        compose_plasma_state(n, t, compose_analytic_profile("T_i", axis_value=50.0, separatrix_value=5.0),
                             z_eff=7.0, impurity_charge=6.0)


# --- limiting cases ---------------------------------------------------------------------------


def test_vanishing_barriers_recover_the_smooth_state():
    smooth = analytic_lmode_state(psi_norm=FINE)
    no_itb = analytic_itb_state(ne_itb_height=0.0, te_itb_height=0.0, ti_itb_height=0.0, psi_norm=FINE)
    for name in ("n_e", "T_e", "T_i", "p_total", "dp_total_dpsi_norm"):
        np.testing.assert_array_equal(getattr(no_itb, name), getattr(smooth, name))
    # A pedestal whose height tends to zero: tiny ITB heights converge linearly.
    for height in (1e-3, 1e-6):
        weak = analytic_itb_state(ne_itb_height=height * 1e19, te_itb_height=height * 100,
                                  ti_itb_height=height * 50, psi_norm=FINE)
        assert np.max(np.abs(weak.T_e - smooth.T_e)) < 2 * height * 100


def test_vanishing_pedestal_amplitude_recovers_the_smooth_profile():
    smooth = compose_analytic_profile("T_e", axis_value=250.0, separatrix_value=10.0)
    for height in (1.0, 1e-3, 1e-6):
        # choose the top value that corresponds to a pedestal of this height
        knee = 0.93 - 0.025
        top = float(evaluate_analytic_profile(smooth, knee))
        profile = compose_analytic_profile("T_e", axis_value=250.0, separatrix_value=10.0,
                                           pedestal_top_value=top + height, pedestal_position=0.93,
                                           pedestal_width=0.05)
        difference = evaluate_analytic_profile(profile, FINE) - evaluate_analytic_profile(smooth, FINE)
        assert np.max(np.abs(difference)) < 2.0 * height
    exact = compose_analytic_profile("T_e", axis_value=250.0, separatrix_value=10.0,
                                     pedestal_top_value=top, pedestal_position=0.93, pedestal_width=0.05)
    assert abs(exact.pedestal.height) < 1e-9


def test_composition_is_deterministic():
    a = analytic_hmode_itb_state()
    b = analytic_hmode_itb_state()
    for name in ("n_e", "T_e", "T_i", "p_total", "dp_total_dpsi_norm"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.profiles == b.profiles


# --- geometry separation --------------------------------------------------------------------------


def test_projection_matches_the_1d_definition_on_any_geometry(limited, single_null):
    state = analytic_hmode_itb_state()
    before = dict(state.profiles)
    for eq in (limited, single_null):
        te = project_plasma_state(state, eq, "T_e")
        psi_n = (eq.psi - eq.psi_axis) / (eq.psi_boundary - eq.psi_axis)
        inside = np.isfinite(te)
        assert inside.sum() > 100
        np.testing.assert_allclose(te[inside], evaluate_plasma_state(state, psi_n[inside], "T_e"),
                                   rtol=1e-12)
        assert np.all(psi_n[inside] <= 1.0 + 1e-12) and np.all(psi_n[inside] >= -1e-12)
    assert dict(state.profiles) == before


def test_projection_excludes_the_private_flux_region(single_null):
    state = analytic_lmode_state()
    projected = project_plasma_state(state, single_null, "n_e")
    psi_n = (single_null.psi - single_null.psi_axis) / (single_null.psi_boundary - single_null.psi_axis)
    below = single_null.z[None, :] < single_null.lcfs.z.min() - 1e-3
    below = np.broadcast_to(below, psi_n.shape)
    # below the X-point there are points with psi_n < 1 (private flux), all excluded
    assert np.any(below & (psi_n < 1.0))
    assert np.all(np.isnan(projected[below]))


def test_projection_edge_fill_and_generic_function(limited):
    state = analytic_hmode_state()
    edge = project_plasma_state(state, limited, "T_e", outside="edge")
    assert np.all(np.isfinite(edge))
    assert np.nanmin(edge) == pytest.approx(state.T_e[-1])
    ones = project_flux_function(limited, lambda x: np.ones_like(x))
    assert np.nanmax(ones) == 1.0 and np.isnan(ones).any()
    with pytest.raises(ValueError, match="outside"):
        project_flux_function(limited, lambda x: x, outside="zero")


def test_projection_is_cocos_independent(limited):
    state = analytic_hmode_state()
    flipped = dataclasses.replace(limited, psi=-2 * np.pi * limited.psi, psi_axis=-2 * np.pi * limited.psi_axis,
                                  psi_boundary=-2 * np.pi * limited.psi_boundary)
    np.testing.assert_allclose(project_plasma_state(state, flipped, "p_total"),
                               project_plasma_state(state, limited, "p_total"), rtol=1e-12)


def test_state_records_that_it_is_not_an_equilibrium():
    state = analytic_lmode_state()
    assert isinstance(state, AnalyticPlasmaState)
    assert state.metadata["grad_shafranov_consistent"] is False
    assert state.coordinate == "psi_norm"


def test_projection_view_model_carries_the_projected_field(limited):
    from vaft.plot import plasma_state_projection_model

    state = analytic_hmode_state()
    model = plasma_state_projection_model(state, limited, "T_e")
    np.testing.assert_array_equal(model.values, project_plasma_state(state, limited, "T_e").T)
    assert model.filled and "eV" in model.value_label
    assert any(layer.label == "LCFS" for layer in model.overlays)
