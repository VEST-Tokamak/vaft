"""Fixed-boundary CHEASE equilibria synthesized from 0D descriptors (#120).

The boundary constructor, the source model and the refusal states are tested
offline; the solves need a CHEASE binary and are skipped without one, like the
other CHEASE integration tests.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from vaft.code.chease_synthesis import (
    ZeroDimensionalEquilibriumSpec as Spec,
    build_input_equilibrium,
    construct_boundary,
    synthesize_equilibrium_from_0d,
)
from vaft.process.equilibrium import contour_shape_parameters

needs_chease = pytest.mark.skipif(
    not (os.environ.get("CHEASEHOME") or os.environ.get("CHEASE_EXEC_DIR") or os.environ.get("CHEASE")),
    reason="CHEASE synthesis integration test requires CHEASEHOME, CHEASE_EXEC_DIR or CHEASE",
)

VEST = Spec(major_radius=0.4, minor_radius=0.235, elongation=1.7, triangularity=0.35,
            toroidal_field=0.1, plasma_current=1.0e5)


@pytest.mark.parametrize("kappa, delta, z0", [(1.0, 0.0, 0.0), (1.8, 0.0, 0.0), (1.6, 0.4, 0.0),
                                                (1.6, -0.4, 0.0), (1.7, 0.3, 0.05), (2.2, 0.8, 0.0)])
def test_boundary_recovers_the_requested_shape(kappa, delta, z0):
    spec = dataclasses.replace(VEST, elongation=kappa, triangularity=delta, vertical_position=z0)
    boundary = construct_boundary(spec)
    shape = contour_shape_parameters(boundary.r, boundary.z)
    assert shape["elongation"] == pytest.approx(kappa, rel=2e-3)
    assert shape["triangularity_upper"] == pytest.approx(delta, abs=3e-3)
    assert shape["triangularity_lower"] == pytest.approx(delta, abs=3e-3)
    assert 0.5*(boundary.z.max() + boundary.z.min()) == pytest.approx(z0, abs=1e-9)
    area = 0.5*np.sum(boundary.r*np.roll(boundary.z, -1) - np.roll(boundary.r, -1)*boundary.z)
    assert area > 0                                                   # counter-clockwise


def test_source_model_carries_the_requested_pressure_share():
    spec = dataclasses.replace(VEST, pressure_fraction=0.3)
    eq = build_input_equilibrium(spec)
    from scipy.constants import mu_0

    on_axis = (mu_0*spec.major_radius**2*eq.pprime[0], eq.ffprime[0])
    assert on_axis[0]/(on_axis[0] + on_axis[1]) == pytest.approx(0.3)
    assert eq.pprime[-1] == pytest.approx(0.0) and np.all(eq.pprime <= 0) and np.all(eq.ffprime <= 0)
    assert eq.convention.cocos == 11 and eq.lcfs.closed


def test_q95_request_reaches_the_input_q_profile():
    eq = build_input_equilibrium(VEST, q95=5.0)
    from scipy.interpolate import interp1d

    # The adapter samples q at the constraint surface with a cubic interpolation.
    q_at = interp1d(np.linspace(0, 1, eq.q.size), eq.q, kind="cubic")(0.95)
    assert float(q_at) == pytest.approx(5.0, rel=1e-10)


@pytest.mark.parametrize("change, status", [
    (dict(minor_radius=0.5), "invalid_geometry"),
    (dict(triangularity=1.0), "invalid_geometry"),
    (dict(plasma_current=0.0), "invalid_geometry"),
    (dict(pressure_fraction=1.0), "invalid_source_model"),
    (dict(current_beta=0.0), "invalid_source_model"),
])
def test_invalid_specifications_come_back_as_states(change, status):
    result = synthesize_equilibrium_from_0d(dataclasses.replace(VEST, **change))
    assert result.status == status and not result.ok and result.chease is None and result.reason


@pytest.mark.parametrize("target, q95", [("beta_p", None), ("li", None), ("q95", None), ("q95", 0.8)])
def test_targets_chease_cannot_impose_are_refused(target, q95):
    result = synthesize_equilibrium_from_0d(VEST, target=target, q95=q95)
    assert result.status == "unsupported_target" and result.chease is None


@needs_chease
def test_current_target_is_met_and_the_boundary_preserved(tmp_path):
    from vaft.code.chease import CHEASEConfig

    result = synthesize_equilibrium_from_0d(VEST, config=CHEASEConfig(workdir=tmp_path))
    assert result.ok, result.reason
    assert abs(result.residuals["plasma_current"]) < 1e-3
    for name in ("elongation", "minor_radius", "triangularity_upper", "triangularity_lower"):
        assert abs(result.residuals[name]) < 0.01, name
    assert result.refined_geqdsk is not None and result.refined_ods is not None
    assert result.achieved["q95"] > result.achieved["q0"] > 0          # achieved, not requested


@needs_chease
def test_q95_target_is_met_and_the_current_becomes_an_output(tmp_path):
    from vaft.code.chease import CHEASEConfig

    result = synthesize_equilibrium_from_0d(VEST, target="q95", q95=6.0, config=CHEASEConfig(workdir=tmp_path))
    assert result.ok, result.reason
    assert abs(result.residuals["q95"]) < 0.02
    assert result.achieved["ip"] < VEST.plasma_current                  # q95 = 6 needs less current


@needs_chease
def test_descriptors_respond_to_their_controls(tmp_path):
    from vaft.code.chease import CHEASEConfig

    base = synthesize_equilibrium_from_0d(VEST, config=CHEASEConfig(workdir=tmp_path/"base"))
    taller = synthesize_equilibrium_from_0d(dataclasses.replace(VEST, elongation=1.9), config=CHEASEConfig(workdir=tmp_path/"tall"))
    higher_p = synthesize_equilibrium_from_0d(dataclasses.replace(VEST, pressure_fraction=0.7), config=CHEASEConfig(workdir=tmp_path/"p"))
    assert base.ok and taller.ok and higher_p.ok
    assert taller.achieved["elongation"] > base.achieved["elongation"] + 0.15
    assert taller.achieved["q95"] > base.achieved["q95"]                 # more elongation, same Ip: higher q
    assert higher_p.achieved["beta_t"] > base.achieved["beta_t"]
    assert higher_p.achieved["shafranov_shift"] > base.achieved["shafranov_shift"]


@needs_chease
def test_achieved_descriptors_are_stable_under_mesh_refinement(tmp_path):
    from vaft.code.chease import CHEASEConfig

    coarse = synthesize_equilibrium_from_0d(VEST, config=CHEASEConfig(workdir=tmp_path/"coarse", ns=80, nt=80))
    fine = synthesize_equilibrium_from_0d(VEST, config=CHEASEConfig(workdir=tmp_path/"fine", ns=150, nt=150))
    assert coarse.ok and fine.ok
    for name in ("q95", "beta_t", "li_virial", "shafranov_shift"):
        assert coarse.achieved[name] == pytest.approx(fine.achieved[name], rel=0.02), name


@pytest.mark.parametrize("ip, bt", [(1e5, 0.1), (-1e5, 0.1), (1e5, -0.1), (-1e5, -0.1)])
def test_input_signs_are_cocos_11_consistent(ip, bt):
    eq = build_input_equilibrium(dataclasses.replace(VEST, plasma_current=ip, toroidal_field=bt))
    assert np.all(np.sign(eq.q) == np.sign(ip*bt))                     # sigma_rho_theta_phi = +1
    assert np.sign(eq.psi_boundary - eq.psi_axis) == np.sign(ip)          # psi rises outward for Ip > 0
    assert np.all(np.sign(eq.f) == np.sign(bt))


@needs_chease
def test_negative_current_keeps_its_sign_through_chease(tmp_path):
    from vaft.code.chease import CHEASEConfig

    spec = dataclasses.replace(VEST, plasma_current=-1.0e5, vertical_position=0.05)
    result = synthesize_equilibrium_from_0d(spec, config=CHEASEConfig(workdir=tmp_path))
    assert result.ok, result.reason
    assert float(result.refined_ods["equilibrium.time_slice.0.global_quantities.ip"]) < 0
    assert abs(result.residuals["vertical_position"]) < 0.01


@needs_chease
def test_default_run_does_not_write_into_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = synthesize_equilibrium_from_0d(VEST)
    assert result.ok and list(tmp_path.iterdir()) == []


def test_a_stopped_solve_is_a_timeout_not_a_non_convergence(tmp_path, monkeypatch):
    """#1016: CHEASE returns a timeout; synthesis names it rather than 'non_converged'."""
    import vaft.code.chease_synthesis as module
    from vaft.code.chease import CHEASEConfig, CHEASEResult

    monkeypatch.setattr(
        module, "refine_equilibrium",
        lambda source, config: CHEASEResult(
            returncode=None, workdir=tmp_path, runtime_status="timeout",
            stderr="CHEASE timed out after 60 s of running",
        ),
    )
    result = synthesize_equilibrium_from_0d(VEST, config=CHEASEConfig(workdir=tmp_path))
    assert (result.status, result.reason) == ("timeout", "CHEASE timed out after 60 s of running")
    assert not result.ok and result.chease.runtime_status == "timeout"


# --- barrier-profile sources (#1166 scope B) ------------------------------------


def _hmode_pressure():
    from vaft.process.profile import compose_analytic_profile

    return compose_analytic_profile("p", unit="Pa", axis_value=1.0e3, separatrix_value=20.0,
                                    pedestal_top_value=450.0, pedestal_position=0.93, pedestal_width=0.05)


def _itb_state():
    from vaft.process.profile import analytic_itb_state

    return analytic_itb_state()


def _profile_gradient(profile, x):
    from vaft.data import AnalyticPlasmaState
    from vaft.process.profile import evaluate_analytic_profile, evaluate_plasma_state

    if isinstance(profile, AnalyticPlasmaState):
        return evaluate_plasma_state(profile, x, "dp_total_dpsi_norm")
    return evaluate_analytic_profile(profile, x, derivative=True)


@pytest.mark.parametrize("make_profile", [_hmode_pressure, _itb_state], ids=["hmode", "itb"])
def test_a_barrier_profile_sets_the_input_pprime_shape(make_profile):
    from scipy.constants import mu_0

    profile = make_profile()
    spec = dataclasses.replace(VEST, pressure_profile=profile, pressure_fraction=0.4)
    eq = build_input_equilibrium(spec, resolution=257)
    x = np.linspace(0.0, 1.0, eq.pprime.size)
    want = _profile_gradient(profile, x)
    # Shape only: the input p' is dp/dpsi_N of the profile times one constant.
    np.testing.assert_allclose(eq.pprime/eq.pprime.min(), want/want.min(), rtol=1e-12, atol=1e-14)
    # pressure_fraction is the pressure's share of the psi_N-integrated source.
    pp_int = np.trapezoid(mu_0*spec.major_radius**2*eq.pprime, x)
    ff_int = np.trapezoid(eq.ffprime, x)
    assert pp_int/(pp_int + ff_int) == pytest.approx(0.4, rel=2e-3)
    assert np.all(eq.pprime <= 0) and np.all(eq.ffprime <= 0)


def test_the_hmode_pedestal_sits_where_requested_in_the_input():
    spec = dataclasses.replace(VEST, pressure_profile=_hmode_pressure())
    eq = build_input_equilibrium(spec, resolution=257)
    x = np.linspace(0.0, 1.0, eq.pprime.size)
    edge = x > 0.6
    assert x[edge][np.argmin(eq.pprime[edge])] == pytest.approx(0.93, abs=0.5/256)


def test_an_edge_current_profile_puts_an_ffprime_bump_on_the_pedestal():
    from vaft.process.profile import compose_analytic_profile

    g = compose_analytic_profile("g", unit="", axis_value=1.0, separatrix_value=0.01, pedestal_top_value=0.3,
                                 pedestal_position=0.93, pedestal_width=0.05)
    eq = build_input_equilibrium(dataclasses.replace(VEST, pressure_profile=_hmode_pressure(), current_profile=g),
                                 resolution=257)
    x = np.linspace(0.0, 1.0, eq.ffprime.size)
    edge = x > 0.6
    peak = x[edge][np.argmin(eq.ffprime[edge])]
    assert peak == pytest.approx(0.93, abs=0.5/256)
    assert abs(eq.ffprime[edge]).max() > abs(eq.ffprime[x < 0.6]).max()      # a localized edge current


def test_without_profiles_the_source_model_is_unchanged():
    """The generalized-parabolic model, reproduced term by term: bit-identical input arrays."""
    from scipy.constants import mu_0

    from vaft.formula.equilibrium import generalized_parabolic_profile

    spec = dataclasses.replace(VEST, pressure_fraction=0.3, current_alpha=2.0)
    eq = build_input_equilibrium(spec)
    x = np.linspace(0.0, 1.0, eq.pprime.size)
    pp = 0.3*generalized_parabolic_profile(x, alpha=1.0, beta=2.0)
    ff = 0.7*generalized_parabolic_profile(x, alpha=2.0, beta=1.0)
    r0, a, kappa = spec.major_radius, spec.minor_radius, spec.elongation
    amplitude = mu_0*spec.plasma_current*r0/(np.pi*a**2*kappa*float(np.mean(pp + ff)))
    assert np.array_equal(eq.pprime, -amplitude*pp/(mu_0*r0**2)/(2*np.pi))
    assert np.array_equal(eq.ffprime, -amplitude*ff/(2*np.pi))
    explicit_none = build_input_equilibrium(dataclasses.replace(spec, pressure_profile=None, current_profile=None))
    assert np.array_equal(explicit_none.pprime, eq.pprime) and np.array_equal(explicit_none.psi, eq.psi)


def test_requested_pressure_shape_of_the_parabolic_model_is_analytic():
    from vaft.code.chease_synthesis import requested_pressure_shape

    x = np.linspace(0.0, 1.0, 51)
    p, dp = requested_pressure_shape(VEST, x)                  # p' ~ (1 - x)**2, so p ~ (1 - x)**3
    np.testing.assert_allclose(p, (1 - x)**3, atol=1e-8)
    np.testing.assert_allclose(dp, -3*(1 - x)**2, atol=1e-12)


def test_requested_pressure_shape_of_a_profile_is_the_normalized_profile():
    from vaft.code.chease_synthesis import requested_pressure_shape
    from vaft.process.profile import evaluate_analytic_profile

    profile = _hmode_pressure()
    x = np.linspace(0.0, 1.0, 51)
    p, dp = requested_pressure_shape(dataclasses.replace(VEST, pressure_profile=profile), x)
    np.testing.assert_allclose(p, (evaluate_analytic_profile(profile, x) - 20.0)/980.0, rtol=1e-12)
    np.testing.assert_allclose(dp, evaluate_analytic_profile(profile, x, derivative=True)/980.0, rtol=1e-12)


def _hollow_pressure():
    from vaft.process.profile import compose_analytic_profile

    return compose_analytic_profile("p", unit="Pa", axis_value=300.0, separatrix_value=20.0,
                                    pedestal_top_value=600.0, pedestal_position=0.9, pedestal_width=0.05)


def _rising_current():
    from vaft.process.profile import compose_analytic_profile

    return compose_analytic_profile("g", unit="", axis_value=0.1, separatrix_value=1.0)


@pytest.mark.parametrize("change, message", [
    (lambda: dict(pressure_profile=_hmode_pressure(), pressure_beta=3.0), "both a generalized-parabolic pressure"),
    (lambda: dict(pressure_profile=_hmode_pressure(), pressure_alpha=2.0), "both a generalized-parabolic pressure"),
    (lambda: dict(current_profile=_hmode_pressure(), current_beta=2.0), "both a generalized-parabolic current"),
    (lambda: dict(pressure_profile=_hollow_pressure()), "rises outward"),
    (lambda: dict(pressure_profile=np.ones(5)), "must be an AnalyticProfile"),
    (lambda: dict(current_profile=_itb_state()), "current_profile must be an AnalyticProfile"),
    (lambda: dict(current_profile=_rising_current()), "must fall from axis to separatrix"),
    (lambda: dict(pressure_profile=_hmode_pressure(), pressure_fraction=0.0), "would be discarded"),
])
def test_inconsistent_profile_sources_are_refused(change, message):
    result = synthesize_equilibrium_from_0d(dataclasses.replace(VEST, **change()))
    assert result.status == "invalid_source_model" and result.chease is None
    assert message in result.reason


@needs_chease
@pytest.mark.parametrize("make_profile, barrier", [(_hmode_pressure, 0.93), (_itb_state, 0.3)], ids=["hmode", "itb"])
def test_a_barrier_pressure_profile_is_achieved_through_chease(tmp_path, make_profile, barrier):
    from vaft.code.chease import CHEASEConfig

    spec = dataclasses.replace(VEST, pressure_profile=make_profile(), pressure_fraction=0.3)
    result = synthesize_equilibrium_from_0d(spec, config=CHEASEConfig(workdir=tmp_path))
    assert result.ok, result.reason
    # The state tolerance is 0.02 in normalized pressure; CHEASE tracks the shape far closer.
    assert result.residuals["pressure_shape"] < 0.005
    solved = result.achieved_profiles
    x = solved["psi_n"]
    near = np.abs(x - barrier) < 0.15
    # The barrier's steepest pressure gradient is where it was requested, on the solved mesh.
    assert x[near][np.argmin(solved["pprime_norm"][near])] == pytest.approx(barrier, abs=0.01)
    requested = np.interp(x, result.source_profiles["psi_n"], result.source_profiles["pprime_norm"])
    assert solved["pprime_norm"][near].min() == pytest.approx(requested[near].min(), rel=0.05)
    assert solved["q"][-1] > solved["q"][0] > 0 and result.achieved["q95"] > result.achieved["q0"]
