"""Quadratic coil-response algebra of a signed NTV torque (#1887).

Synthetic Hermitian matrices only -- positive, negative and indefinite -- so
these verify the algebra, not PENTRC physics.  No solver, no Optuna.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.ntv_coil_response import (
    budgeted_torque_extrema,
    coil_phasors,
    fit_inhomogeneous_torque,
    fixed_amplitude_phase_extrema,
    hermitian_torque_matrix,
    quadratic_invariance_checks,
    quadratic_probe_design,
    quadratic_torque,
    qualify_torque_matrix,
    two_group_phase_extrema,
)

RNG = np.random.default_rng(1887)


def _hermitian(k, eigenvalues):
    raw = RNG.normal(size=(k, k)) + 1j * RNG.normal(size=(k, k))
    unitary, _ = np.linalg.qr(raw)
    return unitary @ np.diag(eigenvalues) @ unitary.conj().T


INDEFINITE = _hermitian(3, [-2.0, 0.5, 3.0])


def _probe_torques(q):
    return {label: float(quadratic_torque(q, c)) for label, c in quadratic_probe_design(q.shape[0])}


@pytest.mark.parametrize("eigenvalues", [[1.0, 2.0, 4.0], [-3.0, -1.0, -0.2], [-2.0, 0.5, 3.0]])
def test_k_squared_probes_recover_q_with_the_declared_phasor_sign(eigenvalues):
    q = _hermitian(3, eigenvalues)
    probes = quadratic_probe_design(3)
    assert len(probes) == 9  # K + 2 C(K, 2) = K^2
    recovered = hermitian_torque_matrix(_probe_torques(q), 3)
    np.testing.assert_allclose(recovered, q, atol=1e-12)
    # Probes run in the opposite phasor sign (a physical +90 deg is e_i - i e_j)
    # recover conj(Q): the convention is load-bearing.
    flipped = {label: float(quadratic_torque(q, c.conj())) for label, c in probes}
    np.testing.assert_allclose(hermitian_torque_matrix(flipped, 3), q.conj(), atol=1e-12)


def test_signed_torque_of_an_indefinite_q_takes_both_signs():
    torques = quadratic_torque(INDEFINITE, RNG.normal(size=(200, 3)) + 1j * RNG.normal(size=(200, 3)))
    assert torques.min() < 0 < torques.max()
    assert np.isrealobj(torques)


def test_held_out_synthetic_samples_qualify_the_reconstructed_matrix():
    q = hermitian_torque_matrix(_probe_torques(INDEFINITE), 3)
    held_out = np.array([coil_phasors([1.0, 0.7, 1.3], [0.0, 1.1, -2.0]),
                         coil_phasors([0.4, 1.5, 0.2], [2.5, 0.3, 4.0]),
                         coil_phasors([0.9, 0.0, 1.1], [1.0, 0.0, 5.5])])
    result = qualify_torque_matrix(q, held_out, quadratic_torque(INDEFINITE, held_out), atol=1e-9, rtol=1e-9)
    assert result.status == "qualified"
    assert result.eigenvalues[0] < 0 < result.eigenvalues[-1]


def test_a_non_quadratic_response_is_rejected_not_promoted():
    def response(c):  # a quartic piece: not c^H Q c
        return float(quadratic_torque(INDEFINITE, c) + 0.5 * np.sum(np.abs(c)) ** 4)

    q = hermitian_torque_matrix({label: response(c) for label, c in quadratic_probe_design(3)}, 3)
    held_out = np.array([coil_phasors([2.0, 1.0, 0.5], [0.3, 2.0, 4.0])])
    result = qualify_torque_matrix(q, held_out, [response(held_out[0])], atol=1e-6, rtol=1e-3)
    assert result.status == "rejected" and "held_out" in result.reasons[-1]


def test_failed_held_out_evaluations_never_qualify_and_are_counted_not_zeroed():
    q = hermitian_torque_matrix(_probe_torques(INDEFINITE), 3)
    result = qualify_torque_matrix(q, np.ones((1, 3)), [np.nan], atol=1e-9, rtol=1e-9)
    assert result.status == "insufficient"
    assert any("failed" in reason for reason in result.reasons)
    # One good sample beside two failures is not a validation.
    held_out = np.ones((3, 3)) * np.array([[1.0], [2.0], [0.5]])
    partial = qualify_torque_matrix(q, held_out, [float(quadratic_torque(q, held_out[0])), np.nan, np.nan],
                                    atol=1e-9, rtol=1e-9)
    assert partial.status == "insufficient" and partial.provenance["held_out_failed"] == 2


def test_a_failed_probe_cannot_identify_q():
    torques = _probe_torques(INDEFINITE)
    torques["e0+ie2"] = np.nan
    with pytest.raises(ValueError, match="e0\\+ie2"):
        hermitian_torque_matrix(torques, 3)


def test_invariances_hold_for_a_quadratic_and_fail_for_an_offset():
    c = coil_phasors([1.0, 0.4, 0.8], [0.0, 0.9, 2.5])

    def evaluate(t0):
        return lambda x: float(quadratic_torque(INDEFINITE, x)) + t0

    for t0, expected in ((0.0, True), (0.3, False)):
        f = evaluate(t0)
        checks = quadratic_invariance_checks(
            f(c), scaled=[(0.5, f(0.5 * c)), (2.0, f(2.0 * c))], reversed_torque=f(-c),
            rotated=[(1.0, f(np.exp(1j) * c))], atol=1e-9, rtol=1e-9,
        )
        names = {check.name: check.passed for check in checks}
        assert set(names) == {"amplitude_scaling", "sign_reversal", "common_phase"}
        assert names["sign_reversal"] and names["common_phase"]  # an offset survives these
        assert names["amplitude_scaling"] is expected  # but not scaling


def test_a_linear_background_fails_sign_reversal():
    b = np.array([0.3, -0.2j, 0.1])
    c = coil_phasors([1.0, 0.4, 0.8], [0.0, 0.9, 2.5])

    def f(x):
        return float(quadratic_torque(INDEFINITE, x) + 2 * np.real(b.conj() @ x))

    (check,) = quadratic_invariance_checks(f(c), reversed_torque=f(-c), atol=1e-9, rtol=1e-9)
    assert check.name == "sign_reversal" and not check.passed


def test_a_fixed_amplitude_scan_cannot_separate_a_background_and_says_so():
    phases = RNG.uniform(0, 2 * np.pi, size=(80, 3))
    phasors = np.exp(1j * phases)  # every |c_k| = 1: T0 and Q_ii are collinear
    with pytest.raises(ValueError, match="vary the group amplitudes"):
        fit_inhomogeneous_torque(phasors, quadratic_torque(INDEFINITE, phasors))


def test_the_inhomogeneous_fit_finds_a_background_term():
    b = np.array([0.2 - 0.1j, -0.3j, 0.05])
    phasors = RNG.normal(size=(40, 3)) + 1j * RNG.normal(size=(40, 3))
    torques = quadratic_torque(INDEFINITE, phasors) + 2 * np.real(phasors @ b.conj()) + 0.7
    fit = fit_inhomogeneous_torque(phasors, torques)
    assert fit["T0"] == pytest.approx(0.7)
    np.testing.assert_allclose(fit["b"], b, atol=1e-10)
    np.testing.assert_allclose(fit["Q"], INDEFINITE, atol=1e-10)
    assert fit["residual_rms"] < 1e-10 and fit["rank"] == fit["unknowns"] == 16
    with pytest.raises(ValueError, match="unknowns"):
        fit_inhomogeneous_torque(phasors[:10], torques[:10])


def test_two_group_analytic_extrema_match_a_phase_grid():
    q = np.array([[1.0, 0.6 * np.exp(0.8j)], [0.6 * np.exp(-0.8j), -0.5]])
    extrema = two_group_phase_extrema(q, (1.2, 0.9))
    grid = np.linspace(0, 2 * np.pi, 20001)
    sweep = quadratic_torque(q, np.stack([np.full(grid.size, 1.2), 0.9 * np.exp(1j * grid)], axis=1))
    assert extrema["T_max"] == pytest.approx(sweep.max(), rel=1e-6)
    assert extrema["T_min"] == pytest.approx(sweep.min(), rel=1e-6)
    assert extrema["delta_phi_max"] == pytest.approx(grid[np.argmax(sweep)], abs=1e-3)
    assert extrema["delta_phi_max"] == pytest.approx(np.mod(-0.8, 2 * np.pi))
    assert 0 <= extrema["delta_phi_min"] < 2 * np.pi  # wrapped, not -arg + pi unwrapped


def test_a_small_cross_term_under_cancelling_diagonals_still_has_a_phase():
    q = np.array([[1e6, 1e-7], [1e-7, -1e6]])
    extrema = two_group_phase_extrema(q, (1.0, 1.0))
    assert not extrema["phase_independent"]
    assert extrema["T_max"] - extrema["T_min"] == pytest.approx(4e-7, rel=1e-3)


@pytest.mark.parametrize("q, amplitudes", [(np.diag([1.0, -2.0]), (1.0, 1.0)), (INDEFINITE[:2, :2], (0.0, 1.0))])
def test_no_cross_term_or_a_dead_group_has_no_preferred_phase(q, amplitudes):
    extrema = two_group_phase_extrema(q, amplitudes)
    assert extrema["phase_independent"] and extrema["delta_phi_max"] is None
    assert extrema["T_max"] == extrema["T_min"]


def test_budgeted_extrema_meet_the_budget_and_bound_every_feasible_excitation():
    d = np.diag([1.0, 2.0, 0.5])
    extrema = budgeted_torque_extrema(INDEFINITE, d, budget=3.0)
    for key in ("c_max", "c_min"):
        c = extrema[key]
        assert np.real(c.conj() @ d @ c) == pytest.approx(3.0)
    assert extrema["T_min"] < 0 < extrema["T_max"]
    samples = RNG.normal(size=(500, 3)) + 1j * RNG.normal(size=(500, 3))
    samples *= np.sqrt(3.0 / np.real(np.einsum("ni,ij,nj->n", samples.conj(), d, samples)))[:, None]
    torques = quadratic_torque(INDEFINITE, samples)
    assert torques.max() <= extrema["T_max"] + 1e-9 and torques.min() >= extrema["T_min"] - 1e-9
    with pytest.raises(ValueError, match="positive definite"):
        budgeted_torque_extrema(INDEFINITE, np.diag([1.0, -1.0, 1.0]), budget=1.0)


def test_fixed_amplitude_extrema_reduce_to_the_analytic_two_group_answer():
    q = np.array([[1.0, 0.6 * np.exp(0.8j)], [0.6 * np.exp(-0.8j), -0.5]])
    numeric = fixed_amplitude_phase_extrema(q, (1.2, 0.9), grid_points=36, refine=2)
    analytic = two_group_phase_extrema(q, (1.2, 0.9))
    assert numeric["T_max"] == pytest.approx(analytic["T_max"], rel=1e-8)
    assert numeric["T_min"] == pytest.approx(analytic["T_min"], rel=1e-8)
    assert numeric["phases_max"][0] == 0.0


def test_three_group_fixed_amplitudes_beat_every_grid_sample():
    amplitudes = (1.0, 0.8, 1.1)
    found = fixed_amplitude_phase_extrema(INDEFINITE, amplitudes, grid_points=12, refine=3)
    phases = RNG.uniform(0, 2 * np.pi, size=(2000, 2))
    samples = np.asarray(amplitudes) * np.exp(1j * np.column_stack([np.zeros(2000), phases]))
    torques = quadratic_torque(INDEFINITE, samples)
    assert found["T_max"] >= torques.max() - 1e-9 and found["T_min"] <= torques.min() + 1e-9


def test_inputs_are_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="Hermitian"):
        quadratic_torque(np.array([[1.0, 1.0], [0.0, 1.0]]), np.ones(2))
    with pytest.raises(ValueError, match="negative"):
        coil_phasors([-1.0], [0.0])
    with pytest.raises(ValueError, match="tolerances"):
        quadratic_invariance_checks(1.0, atol=-1.0, rtol=0.0)
    with pytest.raises(ValueError, match="positive"):
        quadratic_invariance_checks(1.0, scaled=[(0.0, 0.0)], atol=0.0, rtol=0.0)
    with pytest.raises(ValueError, match="not finite"):
        quadratic_torque(np.full((2, 2), np.nan), np.ones(2))
    with pytest.raises(ValueError, match="exceeds"):
        fixed_amplitude_phase_extrema(np.eye(6), np.ones(6), grid_points=36, refine=0)
