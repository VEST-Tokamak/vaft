"""Singular coupling, the energy norm, and the edge-overlap metric."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.perturbation import (
    COUPLING_FAMILIES,
    NEGATIVE_EIGENVALUE_POLICIES,
    edge_overlap_metric,
    energy_norm_matrix,
)


def hermitian(eigenvalues, seed=0):
    """A Hermitian matrix with the given spectrum, and its decomposition."""
    rng = np.random.default_rng(seed)
    size = len(eigenvalues)
    raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    vectors, _ = np.linalg.qr(raw)          # unitary: columns orthonormal
    rows = vectors.conj().T                  # one eigenvector per row
    matrix = rows.conj().T @ np.diag(eigenvalues) @ rows
    return matrix, np.asarray(eigenvalues, dtype=float), rows


# --- the energy norm ----------------------------------------------------------


def test_the_norm_is_the_inverse_square_root_of_the_matrix():
    matrix, values, vectors = hermitian([4.0, 1.0, 0.25])
    norm = energy_norm_matrix(values, vectors, negative_eigenvalues="raise")
    np.testing.assert_allclose(norm @ matrix @ norm, np.eye(3), atol=1e-12)


def test_the_negative_eigenvalue_policy_has_no_default():
    _, values, vectors = hermitian([4.0, 1.0])
    with pytest.raises(TypeError, match="negative_eigenvalues"):
        energy_norm_matrix(values, vectors)


def test_each_policy_does_what_it_says():
    """The three constructions this replaces answered this three ways and
    gave no way to tell which had been used."""
    _, values, vectors = hermitian([4.0, -1.0, 0.25])
    with pytest.raises(ValueError, match="not positive"):
        energy_norm_matrix(values, vectors, negative_eigenvalues="raise")
    taken = energy_norm_matrix(values, vectors, negative_eigenvalues="abs")
    dropped = energy_norm_matrix(values, vectors, negative_eigenvalues="drop")
    assert taken.shape == dropped.shape == (3, 3)
    assert not np.allclose(taken, dropped)
    # Dropping loses the mode; taking the magnitude keeps it.
    assert np.linalg.matrix_rank(dropped, tol=1e-10) < np.linalg.matrix_rank(taken, tol=1e-10)
    assert set(NEGATIVE_EIGENVALUE_POLICIES) == {"raise", "abs", "drop"}


def test_a_matrix_a_run_never_filled_is_refused_by_every_policy():
    """The ideal runs carry W_xe as identically zero, and there is no inverse
    square root of nothing."""
    zeros = np.zeros(4)
    vectors = np.eye(4, dtype=complex)
    for policy in NEGATIVE_EIGENVALUE_POLICIES:
        with pytest.raises(ValueError):
            energy_norm_matrix(zeros, vectors, negative_eigenvalues=policy)


def test_an_unknown_policy_and_a_shape_mismatch_are_refused():
    _, values, vectors = hermitian([4.0, 1.0])
    with pytest.raises(ValueError, match="must be one of"):
        energy_norm_matrix(values, vectors, negative_eigenvalues="clip")
    with pytest.raises(ValueError, match="one row per eigenvalue"):
        energy_norm_matrix([1.0, 2.0, 3.0], vectors, negative_eigenvalues="raise")


# --- the overlap metric -------------------------------------------------------


def test_the_metric_is_the_dominant_modes_share_per_unit_field():
    # A coupling matrix whose right singular vectors are the first two axes.
    coupling = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=complex)
    field = np.array([3.0, 4.0, 5.0], dtype=complex)
    result = edge_overlap_metric(coupling, field, b_t0=2.0)
    np.testing.assert_allclose(np.sort(result.singular_values), [1.0, 2.0])
    # The field projects 3 onto one axis and 4 onto the other; the dominant
    # mode is the one carrying the most field, not the largest singular value.
    np.testing.assert_allclose(np.sort(result.projection), [3.0, 4.0])
    assert result.delta_e == pytest.approx(4.0 / 2.0)
    assert result.projection[result.dominant_mode] == pytest.approx(4.0)


def test_the_dominant_mode_is_not_always_the_largest_singular_value():
    coupling = np.array([[5.0, 0.0], [0.0, 1.0]], dtype=complex)
    field = np.array([1.0, 9.0], dtype=complex)
    result = edge_overlap_metric(coupling, field, b_t0=1.0)
    assert result.singular_values[0] == pytest.approx(5.0)
    assert result.projection[result.dominant_mode] == pytest.approx(9.0)


def test_the_projection_is_a_magnitude_and_not_a_phase():
    """A right singular vector is fixed only up to a phase, so the complex
    overlap is a gauge -- two correct decompositions of one matrix differ by
    exactly that phase."""
    coupling = np.array([[1.0 + 1.0j, 0.0], [0.0, 2.0 - 1.0j]], dtype=complex)
    field = np.array([1.0 + 2.0j, 3.0 - 1.0j], dtype=complex)
    result = edge_overlap_metric(coupling, field, b_t0=1.0)
    assert np.isrealobj(result.projection)
    assert np.all(result.projection >= 0.0)
    # Rotating the whole matrix by a global phase cannot change the metric.
    rotated = edge_overlap_metric(coupling * np.exp(0.7j), field, b_t0=1.0)
    assert rotated.delta_e == pytest.approx(result.delta_e)


def test_the_toroidal_field_is_required_and_checked():
    """The code this replaces defaulted it to one, which reports a field in
    tesla as a dimensionless metric. A run carries it as an attribute."""
    coupling = np.eye(2, dtype=complex)
    field = np.ones(2, dtype=complex)
    with pytest.raises(TypeError, match="b_t0"):
        edge_overlap_metric(coupling, field)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="b_t0 must be positive"):
            edge_overlap_metric(coupling, field, b_t0=bad)
    assert edge_overlap_metric(coupling, field, b_t0=4.0).delta_e == pytest.approx(0.25)


def test_a_norm_is_applied_to_the_coupling_before_the_decomposition():
    coupling = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=complex)
    field = np.array([1.0, 0.0], dtype=complex)
    norm = np.diag([0.5, 1.0]).astype(complex)
    plain = edge_overlap_metric(coupling, field, b_t0=1.0)
    normed = edge_overlap_metric(coupling, field, b_t0=1.0, norm=norm)
    assert not np.allclose(np.sort(plain.singular_values), np.sort(normed.singular_values))
    np.testing.assert_allclose(np.sort(normed.singular_values), [1.0, 2.0])


def test_bases_that_do_not_line_up_are_refused():
    """The coupling matrix and the field have to be on one harmonic basis;
    GPEC writes some couplings on a different one."""
    with pytest.raises(ValueError, match="on one basis"):
        edge_overlap_metric(np.eye(2), np.ones(3), b_t0=1.0)
    with pytest.raises(ValueError, match="norm is"):
        edge_overlap_metric(np.eye(2), np.ones(2), b_t0=1.0, norm=np.eye(3))
    with pytest.raises(ValueError, match="dimensional"):
        edge_overlap_metric(np.ones(3), np.ones(3), b_t0=1.0)


def test_the_five_coupling_families_are_named_and_distinct():
    """The reader this builds on keeps them apart; the one it replaces
    flattened all five into one list of surfaces."""
    assert set(COUPLING_FAMILIES) == {"flux", "current", "island", "penetrated", "delta"}
    assert len(set(COUPLING_FAMILIES.values())) == 5
