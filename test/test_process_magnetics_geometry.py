"""Issue #486: a layout centre and a poloidal angle for a magnetic-sensor array."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.magnetics import magnetics_sensor_centre, magnetics_sensor_poloidal_angle


def _ring(r0, z0, a, n=12):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return r0 + a * np.cos(theta), z0 + a * np.sin(theta), theta


def test_a_ring_of_sensors_recovers_its_centre_and_angles():
    r, z, theta = _ring(0.45, 0.0, 0.35)
    centre = magnetics_sensor_centre(r, z)
    assert centre == pytest.approx((0.45, 0.0), abs=1e-9)
    angles = magnetics_sensor_poloidal_angle(r, z, centre)
    np.testing.assert_allclose(angles, theta, atol=1e-9)
    assert np.all((angles >= 0.0) & (angles < 2.0 * np.pi))
    # Quadrants: outboard midplane 0, top pi/2, inboard midplane pi, bottom 3pi/2.
    assert magnetics_sensor_poloidal_angle([0.8, 0.45, 0.1, 0.45], [0.0, 0.35, 0.0, -0.35], (0.45, 0.0)) == pytest.approx(
        [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    )


def test_a_vertically_shifted_layout_moves_the_centre_with_it():
    r, z, _ = _ring(0.45, 0.12, 0.35)
    assert magnetics_sensor_centre(r, z)[1] == pytest.approx(0.12, abs=1e-9)


def test_an_elongated_layout_uses_the_radial_clusters_not_the_height():
    r = np.array([0.1] * 10 + [0.8] * 10)
    z = np.concatenate([np.linspace(-1.0, 1.0, 10), np.linspace(-1.0, 1.0, 10)])
    assert magnetics_sensor_centre(r, z) == pytest.approx((0.45, 0.0), abs=1e-9)


def test_an_outboard_radial_scan_does_not_drag_the_centre():
    """The IMPA Hall array: a few probes far outboard of the wall array."""
    inboard = np.full(33, 0.09)
    wall = np.full(43, 0.79)
    scan = np.array([0.91, 0.96, 1.01, 1.06, 1.11, 1.16, 1.21, 1.26])
    z = np.zeros(inboard.size + wall.size + scan.size)
    with_scan = magnetics_sensor_centre(np.concatenate([inboard, wall, scan]), z)
    without = magnetics_sensor_centre(np.concatenate([inboard, wall]), z[: inboard.size + wall.size])
    assert with_scan[0] == pytest.approx(without[0], abs=0.01)
    assert with_scan[0] == pytest.approx(0.44, abs=0.01)
    # A plain centroid would have moved by several centimetres.
    assert abs(np.mean(np.concatenate([inboard, wall, scan])) - without[0]) > 0.03


def test_a_layout_without_a_radial_split_falls_back_to_the_mean():
    r = np.linspace(0.3, 0.6, 8)
    z = np.zeros(8)
    assert magnetics_sensor_centre(r, z)[0] == pytest.approx(np.mean(r))


def test_non_finite_positions_are_ignored_and_an_empty_layout_is_refused():
    r = np.array([0.1, np.nan, 0.8])
    z = np.array([0.0, 0.5, 0.0])
    assert magnetics_sensor_centre(r, z) == pytest.approx((0.45, 0.0))
    with pytest.raises(ValueError, match="at least one finite"):
        magnetics_sensor_centre([np.nan], [np.nan])


def test_the_angle_defaults_to_the_layout_centre():
    r, z, theta = _ring(0.5, 0.05, 0.3)
    np.testing.assert_allclose(magnetics_sensor_poloidal_angle(r, z), theta, atol=1e-9)
