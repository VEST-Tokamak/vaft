"""Single-particle motion: the drift formulas agree with the integrated orbit."""

import numpy as np
import pytest

from vaft.formula.particle import (
    boris_orbit,
    gyration_offset,
    curvature_drift_velocity,
    exb_drift_velocity,
    grad_b_drift_velocity,
    gyrofrequency,
    larmor_radius,
)

ZERO = lambda x: np.zeros(3)  # noqa: E731


def _uniform(vec):
    vec = np.asarray(vec, dtype=float)
    return lambda x: vec


def _periods(q, m, B, n_periods, steps_per_period=400):
    T = 2 * np.pi / abs(gyrofrequency(q, m, B))
    return T / steps_per_period, n_periods * steps_per_period


def _drift(x, dt):
    """Guiding-centre velocity: a least-squares line through the orbit, which averages the gyration."""
    t = np.arange(len(x)) * dt
    return np.polyfit(t, x, 1)[0]


def test_boris_conserves_energy_without_an_electric_field():
    dt, n = _periods(1.0, 1.0, 2.0, 50, 40)
    _, v = boris_orbit(1.0, 1.0, [0, 0, 0], [1.0, 0.3, 0.2], ZERO, _uniform([0.3, -0.4, 2.0]), dt, n)
    energy = np.sum(v * v, axis=1)
    assert np.ptp(energy) < 1e-12


@pytest.mark.parametrize("q, m", [(1.0, 1.0), (-1.0, 0.25), (2.0, 4.0)])
def test_the_orbit_radius_is_the_larmor_radius(q, m):
    B = 1.5
    dt, n = _periods(q, m, B, 3, 2000)
    x, _ = boris_orbit(q, m, [0, 0, 0], [0.7, 0, 0], ZERO, _uniform([0, 0, B]), dt, n)
    center = x.mean(axis=0)
    radius = np.hypot(*(x[:, :2] - center[:2]).T)
    assert radius.mean() == pytest.approx(larmor_radius(q, m, 0.7, B), rel=1e-4)


def test_ions_and_electrons_gyrate_in_opposite_senses():
    B = _uniform([0, 0, 1.0])
    senses = []
    for q in (1.0, -1.0):
        dt, n = _periods(q, 1.0, 1.0, 1)
        x, v = boris_orbit(q, 1.0, [0, 0, 0], [1.0, 0, 0], ZERO, B, dt, n)
        # z-component of angular momentum about the orbit centre
        senses.append(np.sign(np.mean(np.cross(x - x.mean(axis=0), v)[:, 2])))
        assert np.sign(gyrofrequency(q, 1.0, 1.0)) == q
    # an ion rotates clockwise seen with B towards the viewer (negative L_z)
    assert senses == [-1.0, 1.0]


@pytest.mark.parametrize("q, m", [(1.0, 4.0), (-1.0, 1.0)])
def test_every_species_drifts_at_the_exb_velocity(q, m):
    E, B = np.array([0.4, 0.1, 0.0]), np.array([0.0, 0.0, 2.0])
    dt, n = _periods(q, m, 2.0, 20)
    x, _ = boris_orbit(q, m, [0, 0, 0], [0, 0, 0], _uniform(E), _uniform(B), dt, n)
    assert np.allclose((x[-1] - x[0]) / (n * dt), exb_drift_velocity(E, B), atol=1e-4)


def test_the_grad_b_drift_matches_the_integrated_orbit():
    L = 200.0  # rho / L = 1/200
    B_of = lambda x: np.array([0.0, 0.0, 1.0 + x[0] / L])  # noqa: E731
    dt, n = _periods(1.0, 1.0, 1.0, 200, 100)
    x, _ = boris_orbit(1.0, 1.0, [0, 0, 0], [0, 1.0, 0], ZERO, B_of, dt, n)
    measured = _drift(x, dt)
    expected = grad_b_drift_velocity(1.0, 1.0, 1.0, [0, 0, 1.0], [1 / L, 0, 0])
    assert measured[1] == pytest.approx(expected[1], rel=0.05)
    assert abs(measured[0]) < 0.2 * abs(expected[1])


@pytest.mark.parametrize("q", [1.0, -1.0])
def test_the_vacuum_toroidal_field_drift_is_grad_b_plus_curvature(q):
    R0, B0 = 50.0, 1.0

    def B_of(x):
        R = np.hypot(x[0], x[1])
        return B0 * R0 / R * np.array([-x[1] / R, x[0] / R, 0.0])

    v_perp, v_par = 0.6, 0.8
    dt, n = _periods(q, 1.0, B0, 60)
    x, _ = boris_orbit(q, 1.0, [R0, 0, 0], [v_perp, v_par, 0.0], ZERO, B_of, dt, n)
    measured_vz = (x[-1, 2] - x[0, 2]) / (n * dt)
    b = np.array([0, B0, 0.0])
    expected = (grad_b_drift_velocity(q, 1.0, v_perp, b, [-B0 / R0, 0, 0])
                + curvature_drift_velocity(q, 1.0, v_par, b, [R0, 0, 0]))
    assert expected[2] * q > 0  # ions drift along +z in this field
    assert measured_vz == pytest.approx(expected[2], rel=0.05)


def test_drift_formulas_reject_bad_input():
    with pytest.raises(ValueError):
        exb_drift_velocity([1, 0], [0, 0, 1])
    with pytest.raises(ValueError):
        exb_drift_velocity([1, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError):
        grad_b_drift_velocity(0.0, 1.0, 1.0, [0, 0, 1], [1, 0, 0])
    with pytest.raises(ValueError):
        boris_orbit(1.0, 1.0, [0, 0, 0], [1, 0, 0], ZERO, ZERO, -1.0, 10)
    with pytest.raises(ValueError):
        larmor_radius(0.0, 1.0, 1.0, 1.0)


@pytest.mark.parametrize("q", [1.0, -1.0])
def test_gyration_offset_places_the_guiding_centre_at_the_orbit_centre(q):
    B, v = np.array([0.0, 0.0, 1.5]), np.array([0.8, 0.0, 0.0])
    dt, n = _periods(q, 1.0, 1.5, 1, 2000)
    x, _ = boris_orbit(q, 1.0, [0, 0, 0], v, ZERO, _uniform(B), dt, n)
    offset = gyration_offset(q, 1.0, B, v)
    assert np.allclose(x[:-1].mean(axis=0)[:2], -offset[:2], atol=2e-3)
    assert np.linalg.norm(offset) == pytest.approx(larmor_radius(q, 1.0, 0.8, 1.5))
    with pytest.raises(ValueError):
        gyration_offset(0.0, 1.0, B, v)
