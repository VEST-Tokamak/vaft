"""Exact path-length operators for chord-integrated diagnostics (#886)."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.line_of_sight import (
    Sightlines,
    build_line_integral_operator,
    clip_segment_to_polygon,
    project_emissivity,
)

R = np.linspace(0.2, 0.8, 61)  # cell edges 0.195 .. 0.805
Z = np.linspace(-0.5, 0.5, 101)  # cell edges -0.505 .. 0.505


def _chords(*segments, phi=0.0):
    segments = np.asarray(segments, dtype=float)
    return Sightlines(segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3],
                      np.full(len(segments), phi))


def test_uniform_field_integrates_to_the_chord_length_inside_the_grid():
    chords = _chords(
        (0.0, 0.1, 1.0, 0.1),  # horizontal, crosses the whole grid
        (0.5, -1.0, 0.5, 1.0),  # vertical
        (0.3, -0.2, 0.6, 0.2),  # diagonal, entirely inside: length 0.5
        (0.0, 0.0, 0.2, 0.0),  # ends inside the first cell
        (0.0, 0.3, 1.0, 0.3),  # along a row of nodes
        (0.0, 0.305, 1.0, 0.305),  # exactly on a cell edge
    )
    G = build_line_integral_operator(R, Z, chords)
    lengths = np.asarray(G.sum(axis=1)).ravel()
    np.testing.assert_allclose(lengths, [0.61, 1.01, 0.5, 0.005, 0.61, 0.61], atol=1e-12)
    ones = np.ones((R.size, Z.size))
    np.testing.assert_allclose(project_emissivity(ones, G), lengths)


def test_each_entry_is_a_cell_length_not_a_mask():
    chords = _chords((0.3, -0.2, 0.6, 0.2))
    G = build_line_integral_operator(R, Z, chords).toarray()[0]
    assert G.max() <= np.hypot(0.01, 0.01) + 1e-12
    assert np.count_nonzero(G) > 30


def test_domain_clipping_keeps_only_the_inside_length():
    square = ([0.3, 0.7, 0.7, 0.3], [-0.3, -0.3, 0.3, 0.3])
    chords = _chords((0.0, 0.0, 1.0, 0.0), (0.2, -0.4, 0.8, 0.5))
    G = build_line_integral_operator(R, Z, chords, domain=square)
    lengths = np.asarray(G.sum(axis=1)).ravel()
    # Horizontal: 0.4. Diagonal: slope 1.5, inside for z in [-0.3, 0.3] -> x in
    # [0.2667, 0.6667], clipped at r = 0.3 -> z in [-0.25, 0.3].
    diagonal = np.hypot(0.55 / 1.5, 0.55)
    np.testing.assert_allclose(lengths, [0.4, diagonal], atol=1e-12)


def test_a_non_convex_domain_gives_several_intervals():
    u_shape = ([0.0, 1.0, 1.0, 0.7, 0.7, 0.3, 0.3, 0.0],
               [0.0, 0.0, 1.0, 1.0, 0.4, 0.4, 1.0, 1.0])
    intervals = clip_segment_to_polygon((-0.5, 0.7), (1.5, 0.7), *u_shape)
    np.testing.assert_allclose(intervals, [(0.25, 0.4), (0.6, 0.75)], atol=1e-12)
    assert clip_segment_to_polygon((-0.5, 2.0), (1.5, 2.0), *u_shape) == []
    with pytest.raises(ValueError):
        clip_segment_to_polygon((0, 0), (1, 1), [0, 1], [0, 1])


def test_projection_is_linear():
    rng = np.random.default_rng(886)
    chords = _chords(*rng.uniform([0.1, -0.6, 0.1, -0.6], [0.9, 0.6, 0.9, 0.6], size=(12, 4)))
    G = build_line_integral_operator(R, Z, chords)
    a, b = rng.normal(size=(2, R.size, Z.size))
    np.testing.assert_allclose(project_emissivity(2.0 * a - 3.0 * b, G),
                               2.0 * project_emissivity(a, G) - 3.0 * project_emissivity(b, G),
                               atol=1e-12)
    stacked = project_emissivity(np.stack([a, b]), G)
    assert stacked.shape == (2, 12)
    with pytest.raises(ValueError, match="cells"):
        project_emissivity(np.ones((5, 5)), G)


def test_brightness_converges_with_the_grid():
    """A Gaussian blob seen along a chord passing a distance d from its centre
    integrates to sqrt(pi) s exp(-d^2/s^2) exactly; the chord is oblique so it
    is not aligned with any row of cells."""
    s, d = 0.08, 0.03
    angle = np.deg2rad(37.0)
    direction = np.array([np.cos(angle), np.sin(angle)])
    foot = np.array([0.5, 0.0]) + d * np.array([-direction[1], direction[0]])
    start, end = foot - 0.45 * direction, foot + 0.45 * direction
    exact = np.sqrt(np.pi) * s * np.exp(-(d**2) / s**2)
    chords = _chords((*start, *end))
    errors = []
    for n in (31, 61, 121, 241):
        r = np.linspace(0.2, 0.8, n)
        z = np.linspace(-0.5, 0.5, n)
        rr, zz = np.meshgrid(r, z, indexing="ij")
        field = np.exp(-((rr - 0.5) ** 2 + zz**2) / s**2)
        brightness = project_emissivity(field, build_line_integral_operator(r, z, chords))
        errors.append(abs(brightness[0] - exact) / exact)
    assert errors[-1] < 2e-3
    assert errors[-1] < errors[-2] < errors[-3] < errors[-4]


def test_sightlines_validate_and_subset():
    chords = _chords((0, 0, 1, 0), (0, 1, 1, 1))
    assert chords.labels == ("0", "1")
    one = chords.subset(np.array([False, True]))
    assert len(one) == 1 and one.z1[0] == 1.0 and one.labels == ("1",)
    with pytest.raises(ValueError, match="disagree"):
        Sightlines([0, 1], [0], [1, 1], [0, 0], [0, 0])
    with pytest.raises(ValueError, match="labels"):
        Sightlines([0], [0], [1], [0], [0], labels=("a", "b"))
    with pytest.raises(ValueError, match="increasing"):
        build_line_integral_operator(R[::-1], Z, chords)
