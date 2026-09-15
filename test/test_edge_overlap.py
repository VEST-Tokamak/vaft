"""Singular coupling, the energy norm, and the edge-overlap metric."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.perturbation import (
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


def test_the_field_is_projected_onto_the_singular_vector_not_its_conjugate():
    """numpy returns V^H, so its rows are already conjugated; conjugating
    again gives v^T f, a bilinear form rather than a projection. A diagonal
    coupling cannot tell the two apart -- its singular vectors are real unit
    axes -- so this one is genuinely complex and non-diagonal."""
    coupling = np.array([[1.0 + 2.0j, 0.5 - 1.0j], [0.3 + 0.4j, 2.0 + 0.1j]])
    field = np.array([1.0 + 1.0j, 2.0 - 3.0j])
    _, _, right = np.linalg.svd(coupling, full_matrices=False)
    correct = np.abs(right @ field)
    wrong = np.abs(right.conj() @ field)
    assert not np.allclose(correct, wrong), "the fixture must distinguish them"
    np.testing.assert_allclose(
        edge_overlap_metric(coupling, field, b_t0=1.0).projection, correct
    )


def test_a_norm_that_does_not_commute_is_applied_on_the_right():
    """The coupling acts on the field, so the norm goes between them. With a
    norm that commutes with the coupling -- a scalar multiple, or two
    diagonals -- either side gives the same singular values and the test
    proves nothing."""
    coupling = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=complex)
    # Symmetric and positive definite, the shape energy_norm_matrix returns --
    # but it does not commute with the coupling, and coupling @ norm is not the
    # transpose of norm @ coupling either, so the two sides are distinguishable.
    norm = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=complex)
    assert not np.allclose(coupling @ norm, norm @ coupling), "must not commute"
    expected = np.linalg.svd(coupling @ norm, compute_uv=False)
    flipped = np.linalg.svd(norm @ coupling, compute_uv=False)
    assert not np.allclose(np.sort(expected), np.sort(flipped)), "sides must differ"
    field = np.array([1.0, 0.0], dtype=complex)
    got = edge_overlap_metric(coupling, field, b_t0=1.0, norm=norm)
    np.testing.assert_allclose(np.sort(got.singular_values), np.sort(expected))


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


def test_a_coupling_a_run_never_filled_is_refused():
    """The zero matrix decomposes with V = I, so the projection is just the
    field's own components and the metric comes out plausible for a run that
    computed no coupling at all."""
    with pytest.raises(ValueError, match="identically zero"):
        edge_overlap_metric(np.zeros((2, 2)), np.ones(2), b_t0=1.0)
    with pytest.raises(ValueError, match="identically zero"):
        edge_overlap_metric(np.zeros((0, 2)), np.ones(2), b_t0=1.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_coupling_or_field_is_refused(bad):
    """numpy's decomposition raises LinAlgError on a NaN, which is not the
    error this documents, and an infinity decomposes to a confident wrong
    answer."""
    coupling = np.array([[1.0, 0.0], [0.0, bad]])
    with pytest.raises(ValueError, match="non-finite"):
        edge_overlap_metric(coupling, np.ones(2), b_t0=1.0)
    with pytest.raises(ValueError, match="non-finite"):
        edge_overlap_metric(np.eye(2), np.array([1.0, bad]), b_t0=1.0)


def test_a_complex_eigenvalue_is_refused_rather_than_truncated():
    """float(complex) drops the imaginary part with a warning, and
    numpy.linalg.eig returns a complex dtype even for a real spectrum."""
    with pytest.raises(ValueError, match="complex"):
        energy_norm_matrix([1.0 + 1.0j, 2.0], np.eye(2), negative_eigenvalues="raise")
    # A complex dtype carrying a real spectrum is fine.
    got = energy_norm_matrix(
        np.array([4.0 + 0j, 1.0 + 0j]), np.eye(2), negative_eigenvalues="raise"
    )
    np.testing.assert_allclose(np.diag(got), [0.5, 1.0])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_eigenvalue_slips_past_every_policy_unless_refused(bad):
    """A NaN is not <= 0, so it would reach the norm; an infinity gives
    lambda**-0.5 = 0 and drops a mode in silence."""
    for policy in NEGATIVE_EIGENVALUE_POLICIES:
        with pytest.raises(ValueError, match="not finite"):
            energy_norm_matrix([1.0, bad], np.eye(2), negative_eigenvalues=policy)


def test_an_incomplete_eigenvector_set_is_refused():
    """It returns a norm of whatever dimension the vectors span, which is not
    the one the caller asked about."""
    with pytest.raises(ValueError, match="a complete set is square"):
        energy_norm_matrix([1.0, 2.0, 3.0], np.eye(5)[:3], negative_eigenvalues="raise")
