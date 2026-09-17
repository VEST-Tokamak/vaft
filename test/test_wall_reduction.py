"""Reduced wall dynamics and response-ranked selection (vaft #494, vfit #10).

Synthetic systems whose answers are closed forms; the packaged machine lives
in test_wall_reduction_packaged.py.  The question these tests fix is the one
the order study rests on: a reduced wall must reproduce the FULL wall's
response under the same integrator, and the modes that matter are the ones
the drive excites and the diagnostics see, not the slowest ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.electromagnetics import solve_eddy_currents
from vaft.process.wall_modes import (
    WallModeError,
    allocate_per_segment,
    build_wall_mode_basis,
    combined_operators,
    mode_scores,
    moment_patterns,
    orthonormalize_r,
    project,
    reduced_operators,
    select_all,
    select_by_score,
    select_slowest,
    solve_reduced_eddy,
)


def _system(n: int = 12, n_src: int = 2, seed: int = 494):
    """An SPD wall with two decades of resistance and decay times from
    ~10 ms to ~1 s, three interleaved segments, and a source coupling
    (test_wall_modes' pencil, scaled to physical units and driven)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(1.0, 2.0, size=(n, n))
    M = 1.0e-4 * ((base + base.T) / 2.0 + n * np.eye(n))   # physics L [H] (code M_mat)
    R = np.diag(np.logspace(-3.0, -1.0, n))                 # [Ohm]
    L = 1.0e-4 * rng.normal(size=(n, n_src))                # source coupling [H] (code L_mat)
    idx = np.arange(n)
    segments = [("A", idx[0::3]), ("B", idx[1::3]), ("C", idx[2::3])]
    basis = build_wall_mode_basis(R, M, segments)
    time = np.linspace(0.0, 3.0, 601)
    drive = np.stack([np.sin(np.pi * time), (time > 0.3) * 1.0], axis=1)
    return R, M, L, basis, time, drive


@pytest.fixture(scope="module")
def system():
    return _system()


# --- the reduced circuit is the full circuit in other coordinates ------------

def test_full_rank_reduced_dynamics_reproduce_the_full_solve(system):
    R, M, L, basis, time, drive = system
    I_full = solve_eddy_currents(R, L, M, drive, time, dt_sub=5e-3)
    ops = reduced_operators(basis, R, M, L, select_all(basis))
    a, I_red = solve_reduced_eddy(ops, drive, time, V=basis.V(), dt_sub=5e-3)
    np.testing.assert_allclose(I_red, I_full, rtol=0, atol=1e-10 * np.abs(I_full).max())
    # and the amplitudes are the projection of the full solution
    np.testing.assert_allclose(a, project(basis, I_full, R), rtol=0, atol=1e-10 * np.abs(a).max())


def test_reduced_solve_refuses_operators_without_source_coupling(system):
    R, M, L, basis, time, drive = system
    with pytest.raises(WallModeError, match="M_r"):
        solve_reduced_eddy(reduced_operators(basis, R, M), drive, time)


def test_amplitudes_alone_are_returned_without_a_basis(system):
    R, M, L, basis, time, drive = system
    a, I_w = solve_reduced_eddy(reduced_operators(basis, R, M, L), drive, time, dt_sub=5e-3)
    assert I_w is None and a.shape == (time.size, basis.n_elements)


# --- rankings ---------------------------------------------------------------

def test_scores_are_one_per_mode_and_tau_ranking_is_the_spectrum(system):
    R, M, L, basis, time, drive = system
    scores = mode_scores(basis, R, M, L)
    assert set(scores) == {"tau", "drive_gain"}
    np.testing.assert_array_equal(scores["tau"], basis.tau())
    assert all(v.shape == (basis.n_elements,) for v in scores.values())


def test_response_rankings_need_a_drive_and_a_time_base(system):
    R, M, L, basis, time, drive = system
    with pytest.raises(WallModeError, match="time"):
        mode_scores(basis, R, M, L, drive=drive)
    scores = mode_scores(basis, R, M, L, drive=drive, time=time, dt_sub=5e-3)
    assert {"response_energy", "output_weight"} <= set(scores)
    # without an observation matrix the two coincide
    np.testing.assert_allclose(scores["output_weight"], scores["response_energy"])


def test_response_energy_is_the_rms_projected_amplitude(system):
    R, M, L, basis, time, drive = system
    I_full = solve_eddy_currents(R, L, M, drive, time, dt_sub=5e-3)
    expected = np.sqrt(np.mean(project(basis, I_full, R) ** 2, axis=0))
    scores = mode_scores(basis, R, M, L, drive=drive, time=time, dt_sub=5e-3)
    np.testing.assert_allclose(scores["response_energy"], expected, rtol=1e-12)


def test_observability_weights_a_mode_nobody_sees_to_zero(system):
    R, M, L, basis, time, drive = system
    G = np.zeros((1, basis.n_elements))
    scores = mode_scores(basis, R, M, L, G=G, drive=drive, time=time, dt_sub=5e-3)
    assert np.all(scores["drive_gain"] == 0.0) and np.all(scores["output_weight"] == 0.0)
    assert np.any(scores["response_energy"] > 0.0)


def test_a_mode_the_sources_do_not_couple_to_scores_zero_gain():
    """Segment C decoupled from the sources: its drive gain must vanish
    while its decay times are unremarkable."""
    R, M, L, basis, time, drive = _system()
    L = L.copy()
    L[basis.segment("C").index, :] = 0.0
    scores = mode_scores(basis, R, M, L)
    labels = basis.labels()
    in_c = np.array([seg == "C" for seg, _ in labels])
    assert np.all(scores["drive_gain"][in_c] == 0.0)
    assert np.all(scores["drive_gain"][~in_c] > 0.0)


# --- selection --------------------------------------------------------------

def test_select_by_score_keeps_the_top_modes_with_segment_identity(system):
    R, M, L, basis, time, drive = system
    score = np.arange(basis.n_elements, dtype=float)[::-1]   # first labels score highest
    keep = select_by_score(basis, score, 5)
    assert sum(k.size for k in keep) == 5
    assert basis.labels(keep) == basis.labels()[:5]
    with pytest.raises(WallModeError, match="entries"):
        select_by_score(basis, score[:-1], 3)


def test_select_by_score_with_tau_equals_select_slowest(system):
    R, M, L, basis, time, drive = system
    for M_keep in (1, 4, 7):
        by_score = select_by_score(basis, basis.tau(), M_keep)
        slowest = select_slowest(basis, M_keep)
        assert all(np.array_equal(a, b) for a, b in zip(by_score, slowest))


def test_response_ranked_selection_beats_decay_time_ranking(system):
    """The point of the study: for the same total the drive-excited modes
    reproduce the response better than the slowest ones."""
    R, M, L, basis, time, drive = system
    I_full = solve_eddy_currents(R, L, M, drive, time, dt_sub=5e-3)
    scores = mode_scores(basis, R, M, L, drive=drive, time=time, dt_sub=5e-3)

    def error(keep):
        ops = reduced_operators(basis, R, M, L, keep)
        _, I_red = solve_reduced_eddy(ops, drive, time, V=basis.V(keep), dt_sub=5e-3)
        return np.linalg.norm(I_red - I_full) / np.linalg.norm(I_full)

    e_energy = error(select_by_score(basis, scores["response_energy"], 4))
    e_tau = error(select_by_score(basis, scores["tau"], 4))
    assert e_energy < e_tau


def test_allocation_stops_at_the_tolerance_and_reports_every_round(system):
    R, M, L, basis, time, drive = system
    keep, history = allocate_per_segment(basis, R, M, L, drive, time, tolerance=0.2, dt_sub=5e-3)
    assert history[0]["M_total"] == 0 and history[0]["dissipation"] == pytest.approx(1.0)
    assert history[-1]["dissipation"] <= 0.2
    assert history[-1]["M_repr"] == tuple(int(k.size) for k in keep)
    totals = [row["M_total"] for row in history]
    assert totals == sorted(totals) and 0 < totals[-1] < basis.n_elements
    assert set(history[-1]["by_segment"]) == {"A", "B", "C"}


def test_allocation_never_gives_modes_to_a_segment_without_error(system):
    """A segment the sources do not reach carries no error and gets no modes."""
    R, M, L, basis, time, drive = system
    L = L.copy()
    L[basis.segment("C").index, :] = 0.0
    keep, history = allocate_per_segment(basis, R, M, L, drive, time, tolerance=0.2, dt_sub=5e-3)
    # C is only reached through mutual inductance with A and B, a much smaller effect
    assert keep[2].size <= min(keep[0].size, keep[1].size)


def test_allocation_in_output_norm_needs_a_response(system):
    R, M, L, basis, time, drive = system
    with pytest.raises(WallModeError, match="needs the response"):
        allocate_per_segment(basis, R, M, L, drive, time, tolerance=0.1, metric="output")
    G = np.ones((2, basis.n_elements))
    keep, history = allocate_per_segment(basis, R, M, L, drive, time, tolerance=0.1, metric="output", G=G, dt_sub=5e-3)
    assert history[-1]["output"] <= 0.1


def test_allocation_with_an_unreachable_tolerance_uses_every_mode(system):
    R, M, L, basis, time, drive = system
    keep, history = allocate_per_segment(basis, R, M, L, drive, time, tolerance=0.0, step=3, dt_sub=5e-3)
    assert history[-1]["M_total"] == basis.n_elements
    assert history[-1]["dissipation"] < 1e-8


# --- moment patterns --------------------------------------------------------

def test_orthonormalize_r_returns_an_r_orthonormal_basis_and_drops_dependence(system):
    R, M, L, basis, time, drive = system
    X = np.hstack([L, 2.0 * L[:, :1]])     # a repeated direction
    V = orthonormalize_r(X, R)
    assert V.shape == (basis.n_elements, 2)
    np.testing.assert_allclose(V.T @ R @ V, np.eye(2), atol=1e-12)
    # sign rule: the largest component of every column is positive
    assert np.all(V[np.argmax(np.abs(V), axis=0), np.arange(2)] > 0)


def test_resistive_patterns_are_exact_for_a_constant_source_ramp(system):
    """Under a constant dI/dt the wall settles into I_w = -R^{-1} M dI/dt,
    which the first moment block spans exactly."""
    R, M, L, basis, time, drive = system
    tau_max = basis.tau().max()
    t = np.linspace(0.0, 40.0 * tau_max, 4001)
    ramp = np.outer(t, [1.0, -0.5])
    I_full = solve_eddy_currents(R, L, M, ramp, t, dt_sub=t[1] - t[0])
    steady = -np.linalg.solve(R, L @ np.array([1.0, -0.5]))
    np.testing.assert_allclose(I_full[-1], steady, rtol=1e-6)
    V1 = moment_patterns(R, M, L, 1)
    residual = steady - V1 @ (V1.T @ (R @ steady))
    assert np.linalg.norm(residual) < 1e-10 * np.linalg.norm(steady)


def test_higher_moments_nest_and_stay_r_orthonormal(system):
    R, M, L, basis, time, drive = system
    V1, V2 = moment_patterns(R, M, L, 1), moment_patterns(R, M, L, 2)
    assert V1.shape[1] == 2 and V2.shape[1] == 4
    np.testing.assert_allclose(V2.T @ R @ V2, np.eye(4), atol=1e-12)
    # span(V1) is inside span(V2)
    residual = V1 - V2 @ (V2.T @ (R @ V1))
    assert np.abs(residual).max() < 1e-10
    with pytest.raises(WallModeError, match="order"):
        moment_patterns(R, M, L, 0)


def test_combined_operators_project_like_reduced_operators(system):
    R, M, L, basis, time, drive = system
    V = basis.V(select_slowest(basis, 5))
    ops = combined_operators(V, R, M, L, label="eigen")
    ref = reduced_operators(basis, R, M, L, select_slowest(basis, 5))
    np.testing.assert_allclose(ops.L_r, ref.L_r, atol=1e-14)
    np.testing.assert_allclose(ops.M_r, ref.M_r, atol=1e-14)
    assert ops.labels == tuple(("eigen", k) for k in range(5)) and ops.n_modes == (5,)
    with pytest.raises(WallModeError, match="shape"):
        combined_operators(V[:-1], R, M, L)


def test_moment_patterns_beat_the_same_number_of_slowest_modes_for_a_slow_drive(system):
    R, M, L, basis, time, drive = system
    slow = np.stack([np.sin(0.1 * np.pi * time), 0.5 * time], axis=1)   # slow relative to tau
    I_full = solve_eddy_currents(R, L, M, slow, time, dt_sub=5e-3)
    V = moment_patterns(R, M, L, 1)
    _, I_m = solve_reduced_eddy(combined_operators(V, R, M, L), slow, time, V=V, dt_sub=5e-3)
    keep = select_slowest(basis, V.shape[1])
    _, I_t = solve_reduced_eddy(reduced_operators(basis, R, M, L, keep), slow, time, V=basis.V(keep), dt_sub=5e-3)
    err = lambda I: np.linalg.norm(I - I_full) / np.linalg.norm(I_full)
    assert err(I_m) < err(I_t)


# --- finding plasma-free shots ---------------------------------------------

def test_shots_are_classified_by_plasma_and_coil_peaks_with_evidence():
    from vaft.database.raw import RawSignalUnavailableError
    from vaft.validation.wall_reduction import find_plasma_free_shots

    t = np.linspace(0.0, 0.5, 11)
    waveforms = {
        1: (np.zeros(11), [5.0e3 * np.sin(t)]),          # coil drive, no plasma
        2: (8.0e4 * np.sin(t), [5.0e3 * np.sin(t)]),     # a plasma shot
        3: (np.zeros(11), [2.0e2 * np.ones(11)]),        # nothing fired
        4: (3.5e3 * np.random.default_rng(0).normal(size=11), [1.0e4 * np.ones(11)]),  # failed-breakdown residual
    }

    def plasma_current(shot):
        if shot == 5:
            raise RawSignalUnavailableError(5, 102, "no DAQ")
        return t, waveforms[shot][0]

    def pf(shot):
        return t, waveforms[shot][1]

    rows = find_plasma_free_shots([1, 2, 3, 4, 5], loaders={"plasma_current": plasma_current, "pf": pf})
    assert [row["class"] for row in rows] == ["plasma_free", "plasma", "undriven", "plasma_free", "daq_missing"]
    assert rows[1]["plasma_current_peak"] == pytest.approx(8.0e4 * np.sin(0.5))
    assert rows[0]["coil_current_peak"] == pytest.approx(5.0e3 * np.sin(0.5))
    assert "reason" in rows[4]
