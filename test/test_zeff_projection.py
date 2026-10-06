"""Z_eff(rho) -> R_p -> Z_eff^res,equiv (#1566): flat recovery, Spitzer analytic, window objective."""

from __future__ import annotations

import numpy as np
import pytest

from test_resistive_zeff import _state  # Lane Z's analytic flux-surface state
from vaft.process.zeff_projection import (
    flat_profile_from_resistive,
    profile_conductivity_model,
    project_window_to_resistive_scalar,
    project_zeff_profile_to_resistive_scalar,
    spitzer_resistive_equivalent_zeff,
)

MODELS = ("spitzer_nrl", "sauter", "redl")


@pytest.mark.parametrize("model", MODELS)
def test_a_flat_profile_projects_to_itself(model):
    state = _state()
    for value in (1.0, 2.0, 3.3):
        p = project_zeff_profile_to_resistive_scalar(state, np.full(81, value), model=model)
        assert p.zeff_equivalent == pytest.approx(value, rel=1e-8)


@pytest.mark.parametrize("model", MODELS)
def test_the_profile_closure_reproduces_the_scalar_model(model):
    from vaft.process.resistive_zeff import model_resistance

    state = _state()
    closure = profile_conductivity_model(np.full(81, 2.0), base_model=model)
    assert model_resistance(state, 1.0, model=closure, ln_lambda="sauter").R_p == pytest.approx(
        model_resistance(state, 2.0, model=model, ln_lambda="sauter").R_p, rel=1e-12)


def test_spitzer_numeric_projection_is_the_analytic_dissipation_mean():
    state = _state()
    rho = np.sqrt(state.psi_norm)
    profile = 1.5 + 1.0 * rho**2
    analytic = spitzer_resistive_equivalent_zeff(state, profile)
    numeric = project_zeff_profile_to_resistive_scalar(state, profile, model="spitzer_nrl").zeff_equivalent
    assert numeric == pytest.approx(analytic, rel=1e-8)


def test_a_peaked_current_weights_the_core_not_the_volume():
    """Z0 + dZ rho^2 with a peaked j: the resistive scalar sits below the volume mean."""
    state = _state()
    profile = 1.5 + 1.0 * state.psi_norm
    p = project_zeff_profile_to_resistive_scalar(state, profile, model="redl")
    assert 1.5 < p.zeff_equivalent < p.profile_volume_mean


def test_the_window_scalar_uses_lane_z_s_objective_and_recovers_flat():
    states = [_state(time=t, t_e0=100.0 + 200 * t) for t in (0.0, 1e-3, 2e-3)]
    p = project_window_to_resistive_scalar(states, [np.full(81, 2.0)] * 3, model="redl")
    assert p.convergence["status"] == "ok"
    assert p.zeff_equivalent == pytest.approx(2.0, rel=1e-4)
    rho_profiles = [1.5 + s.psi_norm for s in states]
    q = project_window_to_resistive_scalar(states, rho_profiles, model="redl")
    singles = [project_zeff_profile_to_resistive_scalar(s, z).zeff_equivalent for s, z in zip(states, rho_profiles)]
    assert min(singles) - 1e-3 <= q.zeff_equivalent <= max(singles) + 1e-3


def test_the_vest_preset_projects_to_two():
    from vaft.process.impurity import resolve_impurity_composition

    resolved = resolve_impurity_composition(machine_preset="vest")
    state = _state()
    p = project_zeff_profile_to_resistive_scalar(state, np.full(81, resolved.zeff), model="redl")
    assert p.zeff_equivalent == pytest.approx(2.0, rel=1e-8)


def test_a_flat_profile_from_a_resistive_scalar_needs_the_stated_assumption():
    state = _state()
    np.testing.assert_allclose(flat_profile_from_resistive(1.74, state, assumption="flat"), 1.74)
    with pytest.raises(ValueError, match="assumption='flat'"):
        flat_profile_from_resistive(1.74, state, assumption="measured")


def test_bad_profiles_are_refused():
    state = _state()
    with pytest.raises(ValueError, match="surfaces"):
        project_zeff_profile_to_resistive_scalar(state, np.full(10, 2.0))
    with pytest.raises(ValueError, match="at least 1"):
        project_zeff_profile_to_resistive_scalar(state, np.full(81, 0.5))
    with pytest.raises(ValueError, match="no constant Z_eff"):
        project_zeff_profile_to_resistive_scalar(state, np.full(81, 9.0), bounds=(1.0, 3.0))


# --- cold-review findings ------------------------------------------------------------------


def test_a_numeric_ln_lambda_and_a_radial_profile_still_match_the_analytic_spitzer():
    state = _state()
    profile = 1.2 + 1.5 * state.psi_norm
    analytic = spitzer_resistive_equivalent_zeff(state, profile, ln_lambda=12.0)
    numeric = project_zeff_profile_to_resistive_scalar(state, profile, model="spitzer_nrl", ln_lambda=12.0)
    assert numeric.zeff_equivalent == pytest.approx(analytic, rel=1e-8)


def test_the_profile_model_refuses_a_different_ln_lambda():
    from vaft.process.resistive_zeff import model_resistance

    closure = profile_conductivity_model(np.full(81, 2.0), base_model="redl", ln_lambda="sauter")
    with pytest.raises(ValueError, match="different ln_lambda"):
        model_resistance(_state(), 1.0, model=closure, ln_lambda=15.0)


@pytest.mark.parametrize("model", ["sauter", "redl"])
def test_the_profile_model_is_surface_local_for_a_radial_profile(model):
    from vaft.process.resistive_zeff import parallel_conductivity

    state = _state()
    profile = 1.0 + 2.0 * state.psi_norm
    sigma = profile_conductivity_model(profile, base_model=model)(state, 1.0, parallel_conductivity.__globals__["_ln_lambda_profile"](state, "sauter")[0])
    for i in (0, 20, 40, 80):
        assert sigma[i] == pytest.approx(parallel_conductivity(state, profile[i], model=model, ln_lambda="sauter")[i], rel=1e-12)


def test_the_window_scalar_does_not_depend_on_the_current_sign():
    states = [_state(time=t, ip=-1e5) for t in (0.0, 1e-3, 2e-3)]
    p = project_window_to_resistive_scalar(states, [np.full(81, 2.0)] * 3, model="redl")
    assert p.convergence["status"] == "ok" and p.zeff_equivalent == pytest.approx(2.0, rel=1e-4)


def test_states_sharing_a_time_are_refused():
    states = [_state(time=0.0), _state(time=0.0)]
    with pytest.raises(ValueError, match="share a time"):
        project_window_to_resistive_scalar(states, [np.full(81, 2.0)] * 2)
