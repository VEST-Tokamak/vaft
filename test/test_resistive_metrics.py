"""Resistive-layer metrics: the Lundquist number, the layer width it sets,
the derivative jump across that width, and the area-weighted dominant mode.

On the normal path these reproduce the legacy numerically. What differs is
where the legacy was wrong: a width law that defaulted to one paper's fit, a
shielded field that silently became the derivative jump, and an area guard a
negative area slipped past.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.perturbation import (
    PAPER_JUMP_WIDTH,
    DominantMode,
    JumpWidthModel,
    dominant_mode,
    finite_width_delta,
    jump_width,
    lundquist_number,
    shielded_field,
)


def test_lundquist_is_the_ratio_of_the_two_times():
    np.testing.assert_allclose(lundquist_number([1.0, 2.0], [1e-6, 4e-6]), [1e6, 5e5])


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_a_non_positive_or_non_finite_time_is_refused(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        lundquist_number([1.0, bad], [1e-6, 1e-6])
    with pytest.raises(ValueError, match="finite and positive"):
        lundquist_number([1.0, 1.0], [1e-6, bad])


def test_mismatched_times_are_refused():
    with pytest.raises(ValueError, match="resistive times against"):
        lundquist_number([1.0, 2.0], [1e-6])


def test_the_paper_law_reproduces_its_formula():
    S = np.array([1e6, 1e8])
    np.testing.assert_allclose(jump_width(S, PAPER_JUMP_WIDTH), 3.4e-4 + 0.5 * S ** (-1.0 / 3.0))


def test_the_model_is_required():
    """The coefficients are one paper's fit; the legacy carried them as
    keyword defaults, so a caller who never chose a model got one anyway."""
    with pytest.raises(TypeError):
        jump_width([1e6])
    with pytest.raises(TypeError, match="JumpWidthModel"):
        jump_width([1e6], (3.4e-4, 0.5))


def test_the_width_falls_as_s_to_the_minus_one_third():
    model = JumpWidthModel(floor=0.0, scale=1.0)
    assert jump_width([8.0], model)[0] / jump_width([1.0], model)[0] == pytest.approx(0.5)


def test_the_floor_keeps_the_width_finite_as_s_grows():
    assert jump_width([1e30], PAPER_JUMP_WIDTH)[0] == pytest.approx(3.4e-4, rel=1e-6)


@pytest.mark.parametrize("floor, scale", [(-1e-4, 0.5), (3.4e-4, -0.5), (np.nan, 0.5)])
def test_a_bad_model_is_refused_at_construction(floor, scale):
    with pytest.raises(ValueError, match="finite and non-negative"):
        JumpWidthModel(floor=floor, scale=scale)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_a_non_positive_lundquist_number_is_refused(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        jump_width([1e6, bad], PAPER_JUMP_WIDTH)


GRID = np.linspace(0.0, 1.0, 2001)


def test_a_kink_returns_its_slope_difference():
    """A difference of derivatives at the edges -- not an integral, as the
    legacy docstring said."""
    c = 0.5
    field = np.where(GRID < c, 1.0 * (GRID - c), 4.0 * (GRID - c)).astype(complex)
    got = finite_width_delta(GRID, field, [c], [0.1])
    assert got[0].real == pytest.approx(3.0, rel=1e-3)
    assert got[0].imag == pytest.approx(0.0, abs=1e-9)


def test_a_smooth_field_gives_the_second_derivative_times_the_full_width():
    """f'' * w for a smooth field pins that the width enters, and that it is
    the full width rather than a half."""
    got = finite_width_delta(GRID, (GRID ** 2).astype(complex), [0.5], [0.1])
    assert got[0].real == pytest.approx(0.2, rel=1e-3)


def test_one_complex_value_per_surface():
    got = finite_width_delta(GRID, np.exp(1j * 6 * GRID), [0.3, 0.6], [0.02, 0.04])
    assert got.shape == (2,) and np.iscomplexobj(got)


COARSE = np.linspace(0.0, 1.0, 21)
COARSE_KINK = np.where(COARSE < 0.5, COARSE - 0.5, 4.0 * (COARSE - 0.5)).astype(complex)


def test_a_layer_resolved_by_two_nodes_each_side_is_exact():
    """The smallest window accepted still returns the slope difference."""
    got = finite_width_delta(COARSE, COARSE_KINK, [0.5], [0.1])
    assert got[0].real == pytest.approx(3.0, abs=1e-12)


@pytest.mark.parametrize("width", [0.02, 0.08])
def test_a_layer_narrower_than_the_grid_is_refused_rather_than_underestimated(width):
    """Once an edge's stencil reaches across the centre the answer is wrong
    (0.6 instead of 3 at width 0.02), so it is refused, not returned."""
    with pytest.raises(ValueError, match="refine the grid"):
        finite_width_delta(COARSE, COARSE_KINK, [0.5], [width])


def test_a_window_past_the_grid_is_refused_rather_than_extrapolated():
    with pytest.raises(ValueError, match="reaches past the grid"):
        finite_width_delta(GRID, GRID.astype(complex), [0.99], [0.1])


@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_a_non_positive_width_is_refused(bad):
    with pytest.raises(ValueError, match="width must be positive"):
        finite_width_delta(GRID, GRID.astype(complex), [0.5], [bad])


def test_a_grid_that_is_not_increasing_is_refused():
    with pytest.raises(ValueError, match="strictly increasing"):
        finite_width_delta(GRID[::-1], GRID.astype(complex), [0.5], [0.1])


def test_non_finite_input_is_refused():
    field = GRID.astype(complex)
    field[100] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        finite_width_delta(GRID, field, [0.5], [0.1])


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError, match="field values"):
        finite_width_delta(GRID, GRID[:-1].astype(complex), [0.5], [0.1])
    with pytest.raises(ValueError, match="surfaces against"):
        finite_width_delta(GRID, GRID.astype(complex), [0.5, 0.6], [0.1])


def test_the_shielded_field_is_the_inductance_times_the_jump():
    L = np.array([[2.0, 1.0], [0.5, 3.0]], dtype=complex)
    delta = np.array([1.0 + 1.0j, 2.0 - 1.0j])
    np.testing.assert_allclose(shielded_field(delta, L), L @ delta)


def test_there_is_no_identity_fallback():
    """The legacy returned delta itself when no inductance was given."""
    with pytest.raises(ValueError):
        shielded_field([1.0 + 1.0j], None)


def test_a_mismatched_inductance_is_refused():
    with pytest.raises(ValueError, match="one row per surface"):
        shielded_field([1.0, 2.0], np.eye(3))


RNG = np.random.default_rng(3)
COUPLING = RNG.normal(size=(4, 3)) + 1j * RNG.normal(size=(4, 3))


def test_the_dominant_pair_satisfies_the_weighted_decomposition():
    rows, cols = RNG.uniform(1, 3, 4), RNG.uniform(1, 3, 3)
    mode = dominant_mode(COUPLING, response_area=rows, control_area=cols)
    assert isinstance(mode, DominantMode)
    np.testing.assert_allclose(COUPLING @ mode.control_vector,
                               mode.singular_value * mode.response_vector)


def test_unweighted_is_the_ordinary_leading_singular_value():
    assert dominant_mode(COUPLING).singular_value == pytest.approx(
        np.linalg.svd(COUPLING, compute_uv=False)[0])


def test_the_control_vector_is_not_conjugated_twice():
    """#778 found `vh.conj() @ x` computing v^T x. The returned control
    vector must be the right singular vector itself."""
    mode = dominant_mode(COUPLING)
    u, s, _ = np.linalg.svd(COUPLING, full_matrices=False)
    np.testing.assert_allclose(COUPLING @ mode.control_vector, s[0] * u[:, 0])


@pytest.mark.parametrize("bad", [-4.0, 0.0, np.nan])
def test_a_bad_area_is_refused_by_name(bad):
    """The legacy checked sqrt(area) <= 0, which a negative area defeats:
    sqrt of it is nan and nan <= 0 is False."""
    with pytest.raises(ValueError, match="response area must be finite and positive"):
        dominant_mode(COUPLING, response_area=[1.0, bad, 2.0, 1.0])


def test_an_area_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="control areas against"):
        dominant_mode(COUPLING, control_area=[1.0, 2.0])


def test_a_non_finite_coupling_is_refused():
    broken = COUPLING.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        dominant_mode(broken)
