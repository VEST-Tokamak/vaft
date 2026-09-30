"""Several 0D targets at once around the CHEASE synthesis (#120).

The loop logic is tested offline against a smooth fake of the single-target
synthesis; the CHEASE solves are skipped without a binary.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

import vaft.code.chease_synthesis as synthesis
from vaft.code.chease import CHEASEConfig, find_chease_executable
from vaft.code.chease_synthesis import SyntheticEquilibriumResult, ZeroDimensionalEquilibriumSpec as Spec
from vaft.code.chease_synthesis_targets import (
    DEFAULT_KNOBS,
    MULTI_TARGET_STATUSES,
    TargetKnob,
    synthesize_equilibrium_to_targets,
)

VEST = Spec(major_radius=0.4, minor_radius=0.235, elongation=1.7, triangularity=0.35,
            toroidal_field=0.1, plasma_current=1.0e5)

needs_chease = pytest.mark.skipif(find_chease_executable() is None,
                                  reason="multi-target CHEASE synthesis needs a CHEASE executable")

#: Coarse mesh: 2 s a solve, descriptors equal to the default mesh to 3 digits.
COARSE = dict(ns=40, nt=40, nw=129)


def _fake_response(spec, *, target="plasma_current", q95=None):
    """A smooth stand-in for CHEASE's descriptors, shaped like the VEST scan."""
    cb, pf = spec.current_beta, spec.pressure_fraction
    peaking = 1 - math.exp(-(cb - 0.3)/1.5)
    shape_q95 = 2.44 - 0.45*peaking - 0.1*pf                 # q95 at 100 kA
    ip = abs(spec.plasma_current)
    if target == "q95":
        ip = ip*shape_q95*(1e5/ip)/q95                        # q95 ~ 1/Ip at fixed shapes
    q95_value = shape_q95*1e5/ip
    return {"ip": ip, "q95": q95_value, "q0": 0.8 - 0.6*peaking - 0.3*pf,
            "beta_p": 1.3*pf*(0.5 + peaking), "beta_n": 9*pf*(0.5 + peaking)*ip/1e5,
            "li_virial": 0.7 + 1.5*peaking + 0.6*pf}


class FakeSynthesis:
    """Records every call; returns fake achieved descriptors, optionally failing."""

    def __init__(self, fail_at=None, fail_status="non_converged", drop=()):
        self.calls, self.fail_at, self.fail_status, self.drop = [], fail_at, fail_status, drop

    def __call__(self, spec, *, config=None, target="plasma_current", q95=None, **kwargs):
        self.calls.append(dict(spec=spec, target=target, q95=q95, config=config, **kwargs))
        requested = {"plasma_current": spec.plasma_current}
        if len(self.calls) - 1 == self.fail_at:
            return SyntheticEquilibriumResult(self.fail_status, "fake failure", spec, target, requested)
        achieved = {k: v for k, v in _fake_response(spec, target=target, q95=q95).items() if k not in self.drop}
        return SyntheticEquilibriumResult("success", None, spec, target, requested, achieved=achieved)


@pytest.fixture
def fake(monkeypatch):
    stub = FakeSynthesis()
    monkeypatch.setattr(synthesis, "synthesize_equilibrium_from_0d", stub)
    return stub


def _within_bounds(result):
    for record in result.history:
        for target, knob in result.assignment.items():
            assert knob.lower <= record.knobs[knob.name] <= knob.upper


def test_ip_and_q95_converge_through_the_current_shape(fake, tmp_path):
    result = synthesize_equilibrium_to_targets(VEST, {"plasma_current": 1e5, "q95": 2.35},
                                               config=CHEASEConfig(workdir=tmp_path))
    assert result.status == "converged", result.reason
    assert result.imposed == "plasma_current" and result.assignment["q95"].name == "current_beta"
    assert abs(result.residuals["q95"]) <= 0.01
    assert all(call["target"] == "plasma_current" for call in fake.calls)
    assert len(result.history) == len(fake.calls) <= 8
    # Each solve has its own directory; the knob moved away from the start.
    assert len({str(call["config"].workdir) for call in fake.calls}) == len(fake.calls)
    assert result.final_spec.current_beta != VEST.current_beta
    _within_bounds(result)
    rows = {row["target"]: row for row in result.table()}
    assert rows["plasma_current"]["control"] == "CHEASE normalization"
    assert rows["q95"]["achieved"] == result.record.achieved["q95"]


def test_achieved_values_are_never_the_requested_ones(fake):
    result = synthesize_equilibrium_to_targets(VEST, {"plasma_current": 1e5, "q95": 2.35})
    for record in result.history:
        response = _fake_response(record.result.spec)
        assert record.achieved["q95"] == response["q95"]          # what that solve gave
        assert record.requested["q95"] == 2.35
    assert sum(record.achieved["q95"] == 2.35 for record in result.history) == 0


def test_a_descriptor_the_solve_did_not_produce_is_absent_not_requested(monkeypatch):
    stub = FakeSynthesis(drop=("q95",))
    monkeypatch.setattr(synthesis, "synthesize_equilibrium_from_0d", stub)
    result = synthesize_equilibrium_to_targets(VEST, {"plasma_current": 1e5, "q95": 2.35})
    assert result.status == "validation_failed" and "q95" in result.reason
    assert "q95" not in result.history[0].achieved and "q95" not in result.history[0].residuals
    assert len(stub.calls) == 1


def test_q95_far_from_the_geometric_value_is_not_reachable(fake):
    result = synthesize_equilibrium_to_targets(VEST, {"plasma_current": 1e5, "q95": 6.0})
    assert result.status == "not_reachable", result.reason
    knob = DEFAULT_KNOBS["current_shape"]
    assert any(r.knobs["current_beta"] == knob.lower for r in result.history)   # the bound was solved
    low, high = result.reachable["q95"]
    assert high < 6.0 and high == pytest.approx(_fake_response(dataclasses.replace(VEST, current_beta=knob.lower))["q95"])
    assert not result.ok and result.record is not None                           # best solve kept, labelled
    assert result.achieved["q95"] == high
    _within_bounds(result)


def test_beta_p_with_ip_moves_the_pressure_share(fake):
    result = synthesize_equilibrium_to_targets(VEST, {"plasma_current": 1e5, "beta_p": 0.5})
    assert result.status == "converged", result.reason
    assert result.assignment["beta_p"].name == "pressure_fraction"
    assert result.final_spec.current_beta == VEST.current_beta                  # the other knob untouched
    assert abs(result.residuals["beta_p"]) <= 0.01


def test_an_unreachable_beta_hits_the_pressure_bound(fake):
    result = synthesize_equilibrium_to_targets(VEST, {"plasma_current": 1e5, "beta_p": 5.0})
    assert result.status == "not_reachable" and "pressure_fraction" in result.reason
    assert max(r.knobs["pressure_fraction"] for r in result.history) == DEFAULT_KNOBS["pressure_share"].upper


def test_q95_imposed_with_a_beta_target(fake):
    result = synthesize_equilibrium_to_targets(VEST, {"q95": 4.0, "beta_n": 2.0})
    assert result.status == "converged", result.reason
    assert result.imposed == "q95"
    assert all(call["target"] == "q95" and call["q95"] == 4.0 for call in fake.calls)


def test_three_targets_by_damped_newton(fake):
    targets = {"plasma_current": 1e5, "q95": 2.3, "beta_p": 0.6}
    result = synthesize_equilibrium_to_targets(VEST, targets)
    assert result.status == "converged", result.reason
    assert {k.name for k in result.assignment.values()} == {"current_beta", "pressure_fraction"}
    assert all(abs(result.residuals[n]) <= 0.01 for n in ("q95", "beta_p"))
    assert {r.purpose for r in result.history} >= {"initial", "jacobian", "newton"}
    _within_bounds(result)


def test_two_free_targets_out_of_the_box_are_not_reachable(fake):
    result = synthesize_equilibrium_to_targets(VEST, {"plasma_current": 1e5, "q95": 3.0, "beta_p": 0.4},
                                               max_iterations=8)
    assert result.status == "not_reachable", result.reason
    assert "bounds" in result.reason and result.reachable["q95"][1] < 3.0
    _within_bounds(result)


def test_the_iteration_cap_is_an_end_state(fake):
    result = synthesize_equilibrium_to_targets(VEST, {"plasma_current": 1e5, "q95": 2.35},
                                               max_iterations=1, tolerance=1e-9)
    assert result.status == "max_iterations"
    assert result.iterations == 1 and result.record is not None


@pytest.mark.parametrize("fail_at, fail_status, state", [
    (0, "non_converged", "chease_failed"), (1, "timeout", "chease_failed"),
    (0, "boundary_mismatch", "validation_failed"), (1, "invalid_equilibrium", "validation_failed"),
])
def test_failed_solves_end_the_iteration_with_their_state(monkeypatch, fail_at, fail_status, state):
    stub = FakeSynthesis(fail_at=fail_at, fail_status=fail_status)
    monkeypatch.setattr(synthesis, "synthesize_equilibrium_from_0d", stub)
    result = synthesize_equilibrium_to_targets(VEST, {"plasma_current": 1e5, "q95": 2.35})
    assert result.status == state and fail_status in result.reason
    assert len(result.history) == fail_at + 1 and result.history[-1].status == fail_status


@pytest.mark.parametrize("targets, knobs, spec, fragment", [
    ({"plasma_current": 1e5, "beta_p": 0.3, "beta_n": 2.0}, None, VEST, "pressure share"),
    ({"plasma_current": 1e5, "q95": 2.3, "li": 1.0}, None, VEST, "current shape"),
    ({"plasma_current": 1e5, "shafranov_shift": 0.02}, None, VEST, "not descriptors"),
    ({"plasma_current": 1e5, "q95": 2.3}, {"q95": ("pressure_fraction", 0.1, 0.5)}, VEST, "moved by"),
])
def test_combinations_without_a_knob_are_refused(fake, targets, knobs, spec, fragment):
    result = synthesize_equilibrium_to_targets(spec, targets, knobs=knobs)
    assert result.status == "unsupported_target" and fragment in result.reason
    assert fake.calls == [] and result.history == []


def test_a_current_profile_leaves_no_current_shape_knob(fake):
    from vaft.process.profile import compose_analytic_profile

    g = compose_analytic_profile("g", unit="", axis_value=1.0, separatrix_value=0.01)
    result = synthesize_equilibrium_to_targets(dataclasses.replace(VEST, current_profile=g),
                                               {"plasma_current": 1e5, "li": 1.0})
    assert result.status == "unsupported_target" and "current_profile" in result.reason and not fake.calls


@pytest.mark.parametrize("targets, knobs", [
    ({"plasma_current": 1e5, "q95": 0.9}, None),
    ({"plasma_current": 0.0, "q95": 2.3}, None),
    ({"plasma_current": 1e5, "beta_p": 0.3}, {"beta_p": TargetKnob("pressure_fraction", 0.5, 0.2)}),
    ({"plasma_current": 1e5, "beta_p": 0.3}, {"beta_p": TargetKnob("pressure_fraction", 0.1, 1.0)}),
    ({}, None),
])
def test_invalid_requests_are_refused_before_solving(fake, targets, knobs):
    result = synthesize_equilibrium_to_targets(VEST, targets, knobs=knobs)
    assert result.status == "invalid_request" and result.reason and not fake.calls


def test_a_single_imposed_target_is_one_solve(fake):
    result = synthesize_equilibrium_to_targets(VEST, {"plasma_current": 8e4})
    assert result.status == "converged" and len(fake.calls) == 1
    assert fake.calls[0]["spec"].plasma_current == 8e4


def test_statuses_are_declared(fake):
    seen = {synthesize_equilibrium_to_targets(VEST, t).status for t in (
        {"plasma_current": 1e5, "q95": 2.35}, {"plasma_current": 1e5, "q95": 6.0}, {"li": 1.0, "q0": 0.5})}
    assert seen <= set(MULTI_TARGET_STATUSES)


# --- CHEASE ------------------------------------------------------------------------


@needs_chease
@pytest.mark.parametrize("ip, q95", [(4e4, 6.0), (1e5, 2.4)])
def test_chease_reaches_ip_and_q95_together(tmp_path, ip, q95):
    """At fixed Ip, q95 above the default-source value needs a broader current (lower li)."""
    result = synthesize_equilibrium_to_targets(VEST, {"plasma_current": ip, "q95": q95},
                                               config=CHEASEConfig(workdir=tmp_path, **COARSE))
    assert result.status == "converged", result.reason
    assert abs(result.residuals["plasma_current"]) < 0.01 and abs(result.residuals["q95"]) < 0.01
    assert result.result.ok and result.final_spec.current_beta < VEST.current_beta   # broader current
    assert result.achieved["q95"] == result.result.achieved["q95"]
    assert result.result.achieved["li_virial"] < result.history[0].result.achieved["li_virial"]
    assert len(result.history) <= 8


@needs_chease
def test_chease_reaches_three_targets(tmp_path):
    result = synthesize_equilibrium_to_targets(VEST, {"plasma_current": 1e5, "q95": 2.35, "beta_p": 0.4},
                                               config=CHEASEConfig(workdir=tmp_path, **COARSE))
    assert result.status == "converged", result.reason
    assert all(abs(result.residuals[n]) < 0.01 for n in ("plasma_current", "q95", "beta_p"))
    assert len(result.history) <= 16


@needs_chease
def test_chease_reaches_beta_p_with_ip(tmp_path):
    result = synthesize_equilibrium_to_targets(VEST, {"plasma_current": 1e5, "beta_p": 0.5},
                                               config=CHEASEConfig(workdir=tmp_path, **COARSE))
    assert result.status == "converged", result.reason
    assert abs(result.residuals["beta_p"]) < 0.01 and abs(result.residuals["plasma_current"]) < 0.01
    assert result.final_spec.pressure_fraction > VEST.pressure_fraction


@needs_chease
def test_chease_refuses_q95_6_at_100_ka_as_not_reachable(tmp_path):
    result = synthesize_equilibrium_to_targets(VEST, {"plasma_current": 1e5, "q95": 6.0},
                                               config=CHEASEConfig(workdir=tmp_path, **COARSE))
    assert result.status == "not_reachable", result.reason
    assert result.reachable["q95"][1] < 3.0
