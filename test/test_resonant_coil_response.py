"""Resonant-field coil response: the verified operator and stages A-D.

Synthetic operators test the algebra; ``test/data/gpec/resonant_diiid_like_1884/basis.json``
(13 GPEC runs plus 6 confirmations on upstream's *artificial* DIII-D-like
example, not DIII-D #147131) tests the claims the study made: GPEC's
resonant flux is linear in conj(c), and every optimum the operator predicts
is what GPEC returned.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vaft.process.ntv_coil_response import (
    budgeted_torque_extrema,
    coil_phasors,
    fixed_amplitude_phase_extrema,
    two_group_phase_extrema,
)
from vaft.process.perturbation import ResonantWindow, rms_resonant_field
from vaft.process.resonant_coil_response import (
    constrained_window_extremum,
    match_resonant_target,
    qualify_response_operator,
    resonant_operating_space,
    resonant_response_operator,
    window_weights,
)

RNG = np.random.default_rng(1884)
PSI = np.array([0.3, 0.6, 0.85, 0.95, 0.99])
CORE = ResonantWindow("core", 0.0, 0.9, "synthetic")
EDGE = ResonantWindow("edge", 0.9, 1.0, "synthetic")


def _operator(h, field="Phi_res"):
    return resonant_response_operator(list(h.T), field=field, psi_n_rational=PSI, m_rational=[2, 3, 4, 5, 6],
                                      n=1, groups=("a", "b", "c")[: h.shape[1]], amplitude_unit=1000.0)


H = RNG.normal(size=(5, 3)) + 1j * RNG.normal(size=(5, 3))
OP = _operator(H)


def test_the_response_is_linear_in_conj_c_and_the_quadratic_follows():
    c = RNG.normal(size=3) + 1j * RNG.normal(size=3)
    np.testing.assert_allclose(OP.response(c), H @ np.conj(c))
    w = window_weights(OP, EDGE)
    q = OP.quadratic(w)
    direct = OP.response(c)
    assert np.real(c.conj() @ q @ c) == pytest.approx(np.real(direct.conj() @ w @ direct))
    # The window's mean square is the RMS the perturbation layer reports, squared.
    assert np.sqrt(np.real(c.conj() @ q @ c)) == pytest.approx(rms_resonant_field(PSI, direct, window=EDGE))
    # Using H^H W H instead would be a different (mirrored) form for complex c.
    wrong = H.conj().T @ w @ H
    assert not np.isclose(np.real(c.conj() @ wrong @ c), np.real(c.conj() @ q @ c))


def test_qualification_needs_complex_held_out_runs_and_rejects_the_wrong_relation():
    held = np.array([[1j, 0, 0], [0.4, 0.9j, -0.3]])
    good = qualify_response_operator(OP, held, OP.response(held), sample_ids=["x", "y"], rtol=1e-9)
    assert good.status == "qualified"
    linear_in_c = held @ H.T  # a response linear in c, not conj(c)
    assert qualify_response_operator(OP, held, linear_in_c, sample_ids=["x", "y"], rtol=1e-3).status == "rejected"
    real_only = np.array([[1.0, 1.0, 0.0]])
    assert qualify_response_operator(OP, real_only, OP.response(real_only), sample_ids=["r"],
                                     rtol=1e-9).status == "insufficient"
    failed = qualify_response_operator(OP, held, np.vstack([OP.response(held[0]), np.full(5, np.nan)]),
                                       sample_ids=["x", "y"], rtol=1e-9)
    assert failed.status == "insufficient" and "y" not in failed.relative_errors


def test_an_empty_window_is_refused_not_a_zero():
    with pytest.raises(ValueError, match="holds no rational surface"):
        window_weights(OP, ResonantWindow("none", 0.1, 0.2, "synthetic"))


def test_target_matching_recovers_a_reachable_excitation_with_its_conjugate_phase():
    c_true = np.array([0.7 * np.exp(0.4j), 0.0, 1.2 * np.exp(-1.1j)])
    out = match_resonant_target(OP, OP.response(c_true), weights=np.eye(5), regularization=0.0, metric=np.eye(3))
    np.testing.assert_allclose(out["c"], c_true, atol=1e-10)
    assert out["relative_residual"] < 1e-12 and out["rank"] == 3 and out["null_space"].shape == (3, 0)
    # Rotating the target's phase by +chi rotates c by -chi: Phi = H conj(c).
    rotated = match_resonant_target(OP, OP.response(c_true) * np.exp(0.3j), weights=np.eye(5),
                                    regularization=0.0, metric=np.eye(3))
    np.testing.assert_allclose(rotated["c"], c_true * np.exp(-0.3j), atol=1e-10)
    # A penalty shrinks the excitation and leaves a residual.
    shrunk = match_resonant_target(OP, OP.response(c_true), weights=np.eye(5), regularization=1.0, metric=np.eye(3))
    assert np.linalg.norm(shrunk["c"]) < np.linalg.norm(c_true) and shrunk["relative_residual"] > 0


def test_a_rank_deficient_response_reports_its_null_space():
    h = np.column_stack([H[:, 0], H[:, 1], H[:, 0] + H[:, 1]])  # group c is a and b combined
    op = _operator(h)
    out = match_resonant_target(op, op.response(np.array([1.0, 0.5, 0.0])), weights=np.eye(5),
                                regularization=0.0, metric=np.eye(3))
    assert out["rank"] == 2 and out["null_space"].shape == (3, 1)
    np.testing.assert_allclose(op.response(out["null_space"][:, 0]), 0.0, atol=1e-10)
    assert out["relative_residual"] < 1e-10


def test_the_core_cap_is_met_and_no_feasible_excitation_does_better():
    qe, qc = OP.quadratic(window_weights(OP, EDGE)), OP.quadratic(window_weights(OP, CORE))
    free = constrained_window_extremum(qe, qc, metric=np.eye(3), budget=2.0, limit=1e9)
    assert free["status"] == "unconstrained"
    limit = 0.2 * free["constraint"]
    capped = constrained_window_extremum(qe, qc, metric=np.eye(3), budget=2.0, limit=limit)
    assert capped["status"] == "active" and capped["constraint"] == pytest.approx(limit, rel=1e-6)
    assert np.real(capped["c"].conj() @ capped["c"]) == pytest.approx(2.0)
    samples = RNG.normal(size=(20000, 3)) + 1j * RNG.normal(size=(20000, 3))
    samples *= np.sqrt(2.0 / np.sum(np.abs(samples) ** 2, axis=1))[:, None]
    core = np.real(np.einsum("ni,ij,nj->n", samples.conj(), qc, samples))
    edge = np.real(np.einsum("ni,ij,nj->n", samples.conj(), qe, samples))
    assert edge[core <= limit].max() <= capped["objective"] * (1 + 1e-9)
    floor = np.linalg.eigvalsh(qc)[0] * 2.0
    assert constrained_window_extremum(qe, qc, metric=np.eye(3), budget=2.0,
                                       limit=0.5 * floor)["status"] == "infeasible"


def test_the_builder_reports_rms_per_window_and_never_zero_for_a_failure():
    held = np.array([[1j, 0, 0], [0.4, 0.9j, -0.3]])
    qualification = qualify_response_operator(OP, held, OP.response(held), sample_ids=["x", "y"], rtol=1e-9)
    responses = np.vstack([OP.response(held[0]), np.full(5, np.nan)])
    empty = ResonantWindow("gap", 0.1, 0.2, "synthetic")
    result = resonant_operating_space(
        ("a", "b", "c"), 1, held, {"Phi_res": responses}, psi_n_rational=PSI,
        windows={"core": CORE, "edge": EDGE, "gap": empty}, sample_ids=["x", "y"], roles=["scan", "scan"],
        amplitude_unit=1000.0, frame="synthetic", provenance={"synthetic": True},
        operators={"Phi_res": (OP, qualification)},
    )
    values = result.values.set_index(["sample_id", "metric"])
    assert values.loc[("x", "phi_res_rms_edge"), "value"] == pytest.approx(
        rms_resonant_field(PSI, responses[0], window=EDGE))
    assert values.loc[("y", "phi_res_rms_core"), "status"] == "failed"
    assert values.loc[("x", "phi_res_rms_gap"), "status"] == "undefined"
    assert values.loc[("y", "phi_res_rms_core_model"), "status"] == "valid"  # the model still predicts
    assert result.provenance["phi_res_rms_edge"]["window"] == [0.9, 1.0]


# ------------------------------------------------- the GPEC reference

DATA = json.loads((Path(__file__).parent / "data" / "gpec" / "resonant_diiid_like_1884" / "basis.json").read_text())
SAMPLES = {row["sample_id"]: row for row in DATA["samples"]}
WINDOWS = {k: ResonantWindow(k, v["low"], v["high"], v["source"]) for k, v in DATA["windows"].items()}


def _c(row):
    return np.asarray(row["c_re"]) + 1j * np.asarray(row["c_im"])


def _phi(row, field):
    return np.asarray(row[field]["re"]) + 1j * np.asarray(row[field]["im"])


def _real_operator(field):
    return resonant_response_operator(
        [_phi(SAMPLES[f"e{k}"], field) for k in range(3)], field=field,
        psi_n_rational=DATA["surfaces"]["psi_n_rational"], m_rational=DATA["surfaces"]["m_rational"],
        n=DATA["case"]["n"], groups=DATA["case"]["groups"], amplitude_unit=DATA["case"]["phasor_unit_a_per_turn"],
    )


@pytest.mark.parametrize("field", ["Phi_res", "Phi_res_v"])
def test_gpec_resonant_flux_is_linear_in_conj_c(field):
    op = _real_operator(field)
    held = [s for s in SAMPLES if s not in ("e0", "e1", "e2")]
    result = qualify_response_operator(op, np.array([_c(SAMPLES[s]) for s in held]),
                                       [_phi(SAMPLES[s], field) for s in held], sample_ids=held, rtol=1e-4)
    assert result.status == "qualified" and max(result.relative_errors.values()) < 1e-5
    # And not in c: Phi(i e_k) = -i Phi(e_k), so H c is wrong by its whole magnitude.
    ie0 = SAMPLES["ie0"]
    assert np.linalg.norm(op.H @ _c(ie0) - _phi(ie0, field)) / np.linalg.norm(_phi(ie0, field)) == pytest.approx(2.0, abs=1e-4)


def test_core_and_edge_are_split_by_the_fitted_pedestal():
    psi = np.asarray(DATA["surfaces"]["psi_n_rational"])
    assert list(np.asarray(DATA["surfaces"]["m_rational"])[WINDOWS["core"].mask(psi)]) == [2, 3, 4]
    assert list(np.asarray(DATA["surfaces"]["m_rational"])[WINDOWS["edge"].mask(psi)]) == [5, 6]


def test_every_predicted_optimum_is_what_gpec_returned():
    op = _real_operator("Phi_res")
    qc, qe = op.quadratic(window_weights(op, WINDOWS["core"])), op.quadratic(window_weights(op, WINDOWS["edge"]))
    stage_a_core = two_group_phase_extrema(qc[:2, :2], (1, 1))
    stage_a_edge = two_group_phase_extrema(qe[:2, :2], (1, 1))
    three_core = fixed_amplitude_phase_extrema(qc, (1, 1, 1), grid_points=72, refine=3)
    three_edge = fixed_amplitude_phase_extrema(qe, (1, 1, 1), grid_points=72, refine=3)
    budget = budgeted_torque_extrema(qe, np.eye(3), 3.0)
    expected = {
        "A_core_min": coil_phasors((1, 1, 0), (0, stage_a_core["delta_phi_min"], 0)),
        "A_edge_max": coil_phasors((1, 1, 0), (0, stage_a_edge["delta_phi_max"], 0)),
        "F3_core_min": coil_phasors((1, 1, 1), three_core["phases_min"]),
        "F3_edge_max": coil_phasors((1, 1, 1), three_edge["phases_max"]),
    }
    psi = op.psi_n_rational
    rms = {}
    for row in DATA["confirmations"]:
        c, direct = _c(row), _phi(row, "Phi_res")
        if row["sample_id"] in expected:  # the run is the optimum this code predicts today
            np.testing.assert_allclose(c, expected[row["sample_id"]], atol=1e-6)
        for window in ("core", "edge"):
            model = rms_resonant_field(psi, op.response(c), window=WINDOWS[window])
            measured = rms_resonant_field(psi, direct, window=WINDOWS[window])
            assert model == pytest.approx(measured, rel=1e-4), (row["sample_id"], window)
            rms[row["sample_id"], window] = measured
    # The core minimum suppresses the core far below what the edge maximum drives there.
    assert rms["F3_core_min", "core"] < 0.03 * rms["F3_edge_max", "core"]
    # The free edge maximum on the budget beats the fixed-amplitude one, as it must.
    assert rms["C_edge_max", "edge"] >= rms["F3_edge_max", "edge"]
    assert np.sqrt(budget["T_max"]) == pytest.approx(rms["C_edge_max", "edge"], rel=1e-4)
    # Stage D kept the core at its cap and still drove the edge.
    assert rms["D_edge_capped", "edge"] > rms["D_edge_capped", "core"]


def _slsqp_reference(qo, qc, budget, limit, starts=40, seed=7):
    """An independent answer: multistart SLSQP on the real parametrization."""
    from scipy.optimize import minimize

    k = qo.shape[0]
    rng = np.random.default_rng(seed)

    def vec(x):
        return x[:k] + 1j * x[k:]

    best = -np.inf
    for _ in range(starts):
        x0 = rng.normal(size=2 * k)
        x0 *= np.sqrt(budget) / np.linalg.norm(x0)
        found = minimize(lambda x: -np.real(vec(x).conj() @ qo @ vec(x)), x0, method="SLSQP", constraints=[
            {"type": "eq", "fun": lambda x: np.real(vec(x).conj() @ vec(x)) - budget},
            {"type": "ineq", "fun": lambda x: limit - np.real(vec(x).conj() @ qc @ vec(x))},
        ], options={"ftol": 1e-12, "maxiter": 500})
        if found.success:
            best = max(best, -found.fun)
    return best


@pytest.mark.parametrize("case", ["diagonal", "more_groups_than_surfaces"])
def test_a_degenerate_top_eigenspace_still_lands_on_the_cap(case):
    """Cold review: when null(Q_c) lies inside null(Q_o) the top eigenvalue is degenerate at mu*."""
    if case == "diagonal":
        qo, qc, limit = np.diag([2.0, 1.0, 0.0]), np.diag([1.0, 1.0, 0.0]), 0.3
        expected = 0.6
    else:
        rng = np.random.default_rng(3)
        h = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
        op = resonant_response_operator(list(h.T), field="Phi_res", psi_n_rational=[0.5, 0.9, 0.97],
                                        m_rational=[2, 3, 4], n=1, groups=("a", "b", "c", "d"), amplitude_unit=1.0)
        qo = op.quadratic(window_weights(op, ResonantWindow("edge", 0.95, 1.0, "synthetic")))
        qc = op.quadratic(window_weights(op, ResonantWindow("total", 0.0, 1.0, "synthetic")))
        limit = 0.443
        expected = _slsqp_reference(qo, qc, 1.0, limit)
    out = constrained_window_extremum(qo, qc, metric=np.eye(qo.shape[0]), budget=1.0, limit=limit)
    assert out["status"] == "active"
    assert out["constraint"] == pytest.approx(limit, rel=1e-6)
    assert out["objective"] == pytest.approx(expected, rel=1e-5)


def test_a_zero_cap_on_a_singular_constraint_is_feasible_and_lives_in_its_null_space():
    rng = np.random.default_rng(11)
    for _ in range(5):
        b = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
        qc = b.conj().T @ b  # rank 2 of 3
        qo = np.diag(rng.uniform(0.5, 2.0, 3)).astype(complex)
        out = constrained_window_extremum(qo, qc, metric=np.eye(3), budget=1.0, limit=0.0)
        assert out["status"] in ("unconstrained", "active", "at_floor")
        assert out["constraint"] == pytest.approx(0.0, abs=1e-9)
        null = np.linalg.eigh(qc)[1][:, 0]
        assert out["objective"] == pytest.approx(np.real(null.conj() @ qo @ null), rel=1e-6)


def test_an_infeasible_cap_returns_the_least_constraint_excitation():
    qo, qc = np.diag([1.0, 2.0]), np.diag([1.0, 3.0])
    out = constrained_window_extremum(qo, qc, metric=np.eye(2), budget=1.0, limit=0.5)
    assert out["status"] == "infeasible" and out["constraint"] == pytest.approx(1.0)


def test_stage_d_and_stage_b_on_the_gpec_operator():
    op = _real_operator("Phi_res")
    qc, qe = op.quadratic(window_weights(op, WINDOWS["core"])), op.quadratic(window_weights(op, WINDOWS["edge"]))
    budget_max = budgeted_torque_extrema(qe, np.eye(3), 3.0)["c_max"]
    cap = (0.2 * np.sqrt(np.real(budget_max.conj() @ qc @ budget_max))) ** 2  # 20 % of the free core RMS
    out = constrained_window_extremum(qe, qc, metric=np.eye(3), budget=3.0, limit=cap)
    confirmed = _c(next(r for r in DATA["confirmations"] if r["sample_id"] == "D_edge_capped"))
    assert out["status"] == "active"
    assert abs(np.vdot(out["c"], confirmed)) == pytest.approx(np.vdot(confirmed, confirmed).real, rel=1e-6)
    # Stage B: a reachable complex target comes back with the conjugate phase.
    c_true = np.array([0.6 * np.exp(0.2j), 0.9 * np.exp(-1.0j), 0.3])
    match = match_resonant_target(op, op.response(c_true), weights=np.eye(5), regularization=0.0, metric=np.eye(3))
    np.testing.assert_allclose(match["c"], c_true, atol=1e-9)
    assert match["rank"] == 3


def test_the_builder_refuses_an_operator_for_a_field_it_was_not_given():
    held = np.array([[1j, 0, 0]])
    qualification = qualify_response_operator(OP, held, OP.response(held), sample_ids=["x"], rtol=1e-9)
    with pytest.raises(ValueError, match="without responses"):
        resonant_operating_space(("a", "b", "c"), 1, held, {"Phi_res_v": OP.response(held)}, psi_n_rational=PSI,
                                 windows={"edge": EDGE}, sample_ids=["x"], roles=["scan"], amplitude_unit=1.0,
                                 frame="-", provenance={}, operators={"Phi_res": (OP, qualification)})
