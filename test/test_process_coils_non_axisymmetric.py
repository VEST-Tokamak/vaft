"""Plain-array kernels for non-axisymmetric coil sets: mode content and vacuum field."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.coils_non_axisymmetric import (
    biot_savart_filaments,
    toroidal_mode_decomposition,
)


def test_cosine_pattern_gives_half_amplitude_and_positive_phase():
    phi = np.deg2rad(np.arange(0, 360, 60))
    amplitude, delta = 100.0, np.deg2rad(30.0)
    currents = amplitude * np.cos(phi + delta)
    coeff = toroidal_mode_decomposition(phi, currents, modes=(0, 1, 2))
    assert abs(coeff[1]) == pytest.approx(amplitude / 2)
    assert np.angle(coeff[1]) == pytest.approx(delta)
    assert abs(coeff[0]) < 1e-12
    assert abs(coeff[2]) < 1e-12


def test_six_coil_n1_pattern_is_dominated_by_n_equals_one():
    phi = np.deg2rad(np.arange(0, 360, 60))
    currents = np.array([97.0, 26.0, -71.0, -97.0, -26.0, 71.0])
    coeff = toroidal_mode_decomposition(phi, currents, modes=(1, 2, 3))
    assert abs(coeff[1]) > 10 * abs(coeff[2])
    assert abs(coeff[1]) > 10 * abs(coeff[3])
    # e^{-i n phi} kernel: a pattern peaking before phi = 0 has a positive phase.
    assert 2 * abs(coeff[1]) == pytest.approx(np.max(np.abs(currents)), rel=0.05)


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="equal in length"):
        toroidal_mode_decomposition([0.0, 1.0], [1.0], modes=(1,))


def _circle(radius: float, n: int = 400, z: float = 0.0) -> np.ndarray:
    t = np.linspace(0.0, 2 * np.pi, n + 1)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t), np.full_like(t, z)])


def test_circular_loop_center_field_matches_mu0_i_over_2a():
    radius, current = 0.5, 1000.0
    field = biot_savart_filaments([_circle(radius)], [current], [[0.0, 0.0, 0.0]])
    expected = 4e-7 * np.pi * current / (2 * radius)
    assert field.shape == (1, 3)
    assert field[0, 2] == pytest.approx(expected, rel=1e-4)
    assert abs(field[0, 0]) < 1e-12 and abs(field[0, 1]) < 1e-12


def test_reversed_traversal_flips_the_sign_and_currents_superpose():
    loop = _circle(0.5)
    probe = [[0.1, 0.0, 0.2]]
    forward = biot_savart_filaments([loop], [1.0], probe)
    backward = biot_savart_filaments([loop[::-1]], [1.0], probe)
    both = biot_savart_filaments(np.stack([loop, loop]), [1.0, 2.0], probe)
    assert np.allclose(backward, -forward)
    assert np.allclose(both, 3 * forward)


def test_open_filament_is_refused():
    loop = _circle(0.5)[:-1]
    with pytest.raises(ValueError, match="not closed"):
        biot_savart_filaments([loop], [1.0], [[0.0, 0.0, 0.0]])


def test_probe_on_the_conductor_is_refused_rather_than_returning_nan():
    loop = _circle(0.5, n=8)
    midpoint = 0.5 * (loop[0] + loop[1])
    with pytest.raises(ValueError, match="singular"):
        biot_savart_filaments([loop], [1.0], [midpoint])


def test_empty_sector_set_is_refused():
    with pytest.raises(ValueError, match="at least one sector"):
        toroidal_mode_decomposition([], [], modes=(1,))
