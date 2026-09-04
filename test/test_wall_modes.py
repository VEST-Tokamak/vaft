"""The segment-wise wall eigenbasis: local eigenbasis, global dynamics (vaft #473).

Synthetic systems first -- every property the reduced-wall contract (vfit #8)
asks the basis to guarantee is checked on matrices whose answer is known --
and the packaged machine lives in test_wall_mode_basis_ods.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.formula.green import green_r
from vaft.process.electromagnetics import wall_propagator
from vaft.process.wall_modes import (
    WallModeBasis,
    WallModeError,
    build_wall_mode_basis,
    canonical_sign,
    check_wall_mode_basis,
    global_time_constants,
    project,
    reconstruct,
    reconstruction_error,
    reduce_response,
    reduced_operators,
    segment_eigenmodes,
    select_all,
    select_slowest,
    select_tau_range,
    subspace_angles_r,
    symmetrize_inductance,
)

MU0 = 4.0e-7 * np.pi


def _self_inductance(major: float, minor: float = 0.02) -> float:
    return MU0 * major * (np.log(8.0 * major / minor) - 2.0)


def _spd_system(n: int = 12, seed: int = 308):
    """An SPD pencil with four decades of resistance (test_fast_objective's)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(1.0, 2.0, size=(n, n))
    M = (base + base.T) / 2.0 + n * np.eye(n)
    R = np.diag(np.logspace(-4.0, 0.0, n))
    return R, M, rng


def _three_segments(n: int = 12):
    """Three interleaved segments over ``n`` elements, so assembly must
    scatter by index rather than slice."""
    idx = np.arange(n)
    return [("A", idx[0::3]), ("B", idx[1::3]), ("C", idx[2::3])]


# --- single conductor and closed forms --------------------------------------

def test_a_single_conductor_decays_with_tau_equal_to_l_over_r():
    tau, V, residual = segment_eigenmodes(np.array([[2.0e-3]]), np.array([[5.0e-6]]))
    assert tau[0] == pytest.approx(5.0e-6 / 2.0e-3)
    assert V[0, 0] == pytest.approx(1.0 / np.sqrt(2.0e-3))  # v^T R v = 1
    assert residual < 1e-15


def test_two_coupled_loops_reproduce_the_closed_form_pencil():
    """The same two-loop fixture test_vacuum_benchmark drives its solver with."""
    loops = ((0.2, 0.9), (0.95, 0.0))
    L = np.array([[_self_inductance(ri) if i == j else green_r(ri, zi, rj, zj) for j, (rj, zj) in enumerate(loops)]
                  for i, (ri, zi) in enumerate(loops)])
    R = np.diag([3.0e-4, 3.0e-4])
    tau, V, residual = segment_eigenmodes(R, L)
    expected = np.sort(np.linalg.eigvals(np.linalg.solve(R, L)).real)[::-1]
    np.testing.assert_allclose(tau, expected, rtol=1e-12)
    assert residual < 1e-14


# --- the basis on a three-segment SPD system --------------------------------

@pytest.fixture
def system():
    R, M, _ = _spd_system()
    basis = build_wall_mode_basis(R, M, _three_segments())
    return R, M, basis


def test_segment_eigenpairs_satisfy_the_pencil_to_machine_precision(system):
    _, _, basis = system
    assert all(seg.residual < 1e-12 for seg in basis.segments)


def test_the_basis_is_r_orthonormal_and_diagonalizes_each_block(system):
    R, M, basis = system
    r = np.diag(R)
    for seg in basis.segments:
        Rg = np.diag(r[seg.index])
        Lg = M[np.ix_(seg.index, seg.index)]
        np.testing.assert_allclose(seg.V.T @ Rg @ seg.V, np.eye(seg.size), atol=1e-13)
        np.testing.assert_allclose(seg.V.T @ Lg @ seg.V, np.diag(seg.tau), atol=1e-13 * seg.tau.max())


def test_modes_are_ordered_by_descending_tau_within_each_segment(system):
    _, _, basis = system
    for seg in basis.segments:
        assert np.all(np.diff(seg.tau) < 0)


def test_the_largest_component_of_every_mode_is_positive(system):
    _, _, basis = system
    for seg in basis.segments:
        pivots = np.argmax(np.abs(seg.V), axis=0)
        assert np.all(seg.V[pivots, np.arange(seg.size)] > 0)
    assert np.array_equal(canonical_sign(np.array([[-2.0, 1.0], [1.0, 3.0]])), np.array([[2.0, 1.0], [-1.0, 3.0]]))


def test_normalization_and_sign_are_deterministic_across_runs(system):
    R, M, basis = system
    again = build_wall_mode_basis(R, M, _three_segments())
    for a, b in zip(basis.segments, again.segments):
        assert np.array_equal(a.V, b.V) and np.array_equal(a.tau, b.tau)
    assert basis.digest() == again.digest()


def test_full_basis_round_trip_is_exact(system):
    R, M, basis = system
    rng = np.random.default_rng(1)
    I = rng.standard_normal(12)
    np.testing.assert_allclose(reconstruct(basis, project(basis, I, R)), I, atol=1e-12)
    series = rng.standard_normal((7, 12))
    np.testing.assert_allclose(reconstruct(basis, project(basis, series, R)), series, atol=1e-12)


def test_projection_is_exact_inside_the_retained_subspace(system):
    R, M, basis = system
    keep = select_slowest(basis, [2, 1, 2])
    a = np.array([0.3, -1.2, 0.7, 2.0, -0.4])
    I = reconstruct(basis, a, keep)
    np.testing.assert_allclose(project(basis, I, R, keep), a, atol=1e-12)


def test_projection_does_not_assume_euclidean_orthonormality(system):
    R, M, basis = system
    V = basis.V()
    I = np.random.default_rng(2).standard_normal(12)
    a = project(basis, I, R)
    assert not np.allclose(V.T @ I, a)             # the naive projection is wrong here ...
    np.testing.assert_allclose(V.T @ (np.diag(R) * I), a)   # ... the R-weighted one is exact


def test_response_reduction_commutes_with_reconstruction(system):
    R, M, basis = system
    G = np.random.default_rng(3).standard_normal((5, 12))
    keep = select_slowest(basis, 6)
    a = np.random.default_rng(4).standard_normal(6)
    np.testing.assert_allclose(reduce_response(G, basis, keep) @ a, G @ reconstruct(basis, a, keep), atol=1e-13)


def test_reduced_resistance_is_the_identity_and_inductance_keeps_off_diagonal_coupling(system):
    R, M, basis = system
    ops = reduced_operators(basis, R, M)
    np.testing.assert_allclose(ops.R_r, np.eye(12), atol=1e-13)
    metrics = check_wall_mode_basis(basis, R, M)
    assert all(value > 1e-3 for value in metrics["coupling"].values())
    # a block-diagonal inductance has no inter-segment coupling to keep
    block = np.zeros_like(M)
    for _, idx in _three_segments():
        block[np.ix_(idx, idx)] = M[np.ix_(idx, idx)]
    decoupled = build_wall_mode_basis(R, block, _three_segments())
    assert all(value < 1e-14 for value in check_wall_mode_basis(decoupled, R, block)["coupling"].values())


def test_full_rank_reduced_system_has_the_global_spectrum(system):
    R, M, basis = system
    expected = np.sort(np.linalg.eigvals(np.linalg.solve(R, M)).real)[::-1]
    np.testing.assert_allclose(global_time_constants(basis, M), expected, rtol=1e-12)


def test_reduced_dynamics_match_the_full_propagator(system):
    """With R_r = I the reduced homogeneous dynamics are a' = -L_r^-1 a, and
    lifting its propagator back must equal the solver's own."""
    from scipy.linalg import expm

    R, M, basis = system
    dt = 2.0e-5
    ops = reduced_operators(basis, R, M)
    V = basis.V()
    lifted = V @ expm(-np.linalg.solve(ops.L_r, np.eye(12)) * dt) @ V.T @ R
    np.testing.assert_allclose(lifted, wall_propagator(R, M, dt, method="eigh"), atol=1e-12)


def test_source_coupling_is_projected_with_the_same_basis(system):
    R, M, basis = system
    coupling = np.random.default_rng(5).standard_normal((12, 3))
    ops = reduced_operators(basis, R, M, coupling)
    np.testing.assert_allclose(ops.M_r, basis.V().T @ coupling)
    assert ops.n_modes == (4, 4, 4)


# --- pathologies refuse explicitly -------------------------------------------

def test_zero_resistance_is_refused():
    with pytest.raises(WallModeError, match="finite and positive"):
        segment_eigenmodes(np.diag([1.0e-3, 0.0]), np.eye(2) * 1e-6)


def test_non_diagonal_resistance_is_refused():
    with pytest.raises(WallModeError, match="must be diagonal"):
        segment_eigenmodes(np.array([[1e-3, 1e-5], [1e-5, 1e-3]]), np.eye(2) * 1e-6)


def test_singular_inductance_is_refused():
    with pytest.raises(WallModeError, match="not positive definite"):
        segment_eigenmodes(np.diag([1e-3, 1e-3]), np.ones((2, 2)) * 1e-6)


def test_asymmetric_inductance_above_tolerance_is_refused_and_below_is_symmetrized():
    R, M, _ = _spd_system(4)
    skewed = M.copy(); skewed[0, 1] *= 1.0 + 1e-3
    with pytest.raises(WallModeError, match="asymmetric"):
        build_wall_mode_basis(R, skewed, [("A", np.arange(4))])
    slightly = M.copy(); slightly[0, 1] *= 1.0 + 1e-7
    with pytest.warns(RuntimeWarning, match="symmetrized"):
        basis = build_wall_mode_basis(R, slightly, [("A", np.arange(4))])
    assert float(basis.provenance["input_asymmetry"]) > 1e-8
    rounding = M.copy(); rounding[0, 1] *= 1.0 + 1e-10
    _sym, asym = symmetrize_inductance(rounding)
    assert asym < 1e-8


def test_degenerate_pair_is_refused_by_default_and_reported_on_request():
    L = np.diag([2.0e-6, 2.0e-6, 1.0e-6])   # two identical uncoupled loops
    R = np.diag([1e-3, 1e-3, 1e-3])
    with pytest.raises(WallModeError, match="near-degenerate"):
        build_wall_mode_basis(R, L, [("A", np.arange(3))])
    with pytest.warns(RuntimeWarning, match="near-degenerate"):
        basis = build_wall_mode_basis(R, L, [("A", np.arange(3))], on_cluster="warn")
    assert "A:0-1" in basis.provenance["degenerate_pairs"]


def test_ill_conditioned_block_is_refused():
    R = np.diag([1.0, 1.0])
    L = np.diag([1.0, 1e-14])
    with pytest.raises(WallModeError, match="condition number|positive definite"):
        segment_eigenmodes(R, L)


def test_a_segment_map_that_misses_or_repeats_an_element_is_refused():
    R, M, _ = _spd_system(4)
    with pytest.raises(WallModeError, match="belong to no segment"):
        build_wall_mode_basis(R, M, [("A", [0, 1])])
    with pytest.raises(WallModeError, match="shares elements"):
        build_wall_mode_basis(R, M, [("A", [0, 1, 2]), ("B", [2, 3])])


# --- selection, inspection, identity ----------------------------------------

def test_select_slowest_across_segments_and_per_segment(system):
    R, M, basis = system
    keep = select_slowest(basis, 3)
    assert sum(k.size for k in keep) == 3
    chosen = basis.tau(keep)
    assert np.all(chosen >= np.sort(basis.tau())[::-1][2])
    per = select_slowest(basis, [1, 0, 2])
    assert [k.size for k in per] == [1, 0, 2]
    assert [k.tolist() for k in per] == [[0], [], [0, 1]]


def test_select_tau_range_and_labels_carry_segment_identity(system):
    R, M, basis = system
    lo = float(np.median(basis.tau()))
    keep = select_tau_range(basis, lo)
    assert all(basis.segment(seg_id).tau[k] >= lo for seg_id, k in basis.labels(keep))
    assert basis.labels(select_all(basis))[:2] == (("A", 0), ("A", 1))


def test_subspace_angles_compare_eigenspaces_not_vectors():
    R = np.diag([1e-3, 2e-3])
    V = np.linalg.qr(np.random.default_rng(6).standard_normal((2, 2)))[0]
    rotation = np.array([[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]])
    angles = subspace_angles_r(V, V @ rotation, R)
    np.testing.assert_allclose(angles, 0.0, atol=1e-12)


def test_digest_changes_with_resistance_and_not_with_call_order(system):
    R, M, basis = system
    doubled = build_wall_mode_basis(2.0 * R, M, _three_segments())
    assert doubled.digest() != basis.digest()
    assert build_wall_mode_basis(R, M, _three_segments()).digest() == basis.digest()


def test_npz_round_trip_preserves_basis_and_provenance(system, tmp_path):
    R, M, basis = system
    path = tmp_path / "basis.npz"
    basis.to_npz(path)
    back = WallModeBasis.from_npz(path)
    assert back.digest() == basis.digest()
    assert dict(back.provenance) == dict(basis.provenance)
    assert back.n_modes() == basis.n_modes()


def test_reconstruction_error_reports_the_norms_this_basis_is_built_in(system):
    R, M, basis = system
    I = np.random.default_rng(7).standard_normal(12)
    keep = select_slowest(basis, 5)
    err = reconstruction_error(I, reconstruct(basis, project(basis, I, R, keep), keep), R, basis)
    assert 0.0 < err["relative_dissipation"] < 1.0 and 0.0 < err["relative_l2"] < 1.0
    assert set(err["segments"]) == {"A", "B", "C"}
    exact = reconstruction_error(I, reconstruct(basis, project(basis, I, R)), R)
    assert exact["relative_dissipation"] < 1e-12
