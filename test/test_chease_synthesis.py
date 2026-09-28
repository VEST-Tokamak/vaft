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
