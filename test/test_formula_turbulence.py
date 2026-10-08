"""Turbulence--zonal-flow predator--prey kernels (#1820) against a numerical orbit."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.formula import turbulence as pp

PARAMS = dict(gamma_eff=0.3, coupling_suppression=0.5, coupling_drive=0.2, gamma_zonal=0.1)
RATES = dict(gamma_eff=PARAMS["gamma_eff"], gamma_zonal=PARAMS["gamma_zonal"])


def integrate(n0, e0, t_end, dt=1e-3):
    """Classical RK4 of the model, with the formula module's own right-hand side."""
    def f(y):
        return np.array(pp.predator_prey_rhs(y[0], y[1], **PARAMS))
    steps = int(round(t_end / dt))
    y = np.array([n0, e0], dtype=float)
    out = np.empty((steps + 1, 2)); out[0] = y
    for k in range(steps):
        k1 = f(y); k2 = f(y + 0.5 * dt * k1); k3 = f(y + 0.5 * dt * k2); k4 = f(y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k + 1] = y
    return np.arange(steps + 1) * dt, out


def test_the_rhs_vanishes_at_both_fixed_points():
    n_star, e_star = pp.predator_prey_fixed_point(**PARAMS)
    assert (n_star, e_star) == pytest.approx((0.1 / 0.2, 0.3 / 0.5))
    assert pp.predator_prey_rhs(n_star, e_star, **PARAMS) == pytest.approx((0.0, 0.0), abs=1e-15)
    assert pp.predator_prey_rhs(0.0, 0.0, **PARAMS) == (0.0, 0.0)


def test_the_invariant_is_conserved_along_an_orbit_and_minimal_at_the_fixed_point():
    _, y = integrate(1.5, 0.3, 60.0)
    h = pp.predator_prey_invariant(y[:, 0], y[:, 1], **PARAMS)
    assert np.ptp(h) < 1e-9 * abs(h[0])
    n_star, e_star = pp.predator_prey_fixed_point(**PARAMS)
    h_star = pp.predator_prey_invariant(n_star, e_star, **PARAMS)
    assert h_star < h.min()


def test_a_small_orbit_has_the_linearized_period_and_a_quarter_period_lag():
    n_star, e_star = pp.predator_prey_fixed_point(**PARAMS)
    period = pp.predator_prey_period(**RATES)
    assert period == pytest.approx(2 * np.pi / np.sqrt(0.03))
    t, y = integrate(n_star * 1.01, e_star, 3 * period, dt=5e-3)
    dn = y[:, 0] - n_star
    up = np.where((dn[:-1] < 0) & (dn[1:] >= 0))[0]          # upward zero crossings of N
    assert np.diff(t[up]).mean() == pytest.approx(period, rel=2e-3)
    n_peaks = np.where((y[1:-1, 0] > y[:-2, 0]) & (y[1:-1, 0] >= y[2:, 0]))[0] + 1
    e_peaks = np.where((y[1:-1, 1] > y[:-2, 1]) & (y[1:-1, 1] >= y[2:, 1]))[0] + 1
    lag = t[e_peaks[e_peaks > n_peaks[0]][0]] - t[n_peaks[0]]
    assert lag == pytest.approx(pp.predator_prey_response_lag(**RATES), rel=5e-3)
    assert pp.predator_prey_response_lag(**RATES) == pytest.approx(period / 4)


def test_a_large_orbit_is_slower_than_the_linearized_period():
    period = pp.predator_prey_period(**RATES)
    t, y = integrate(5.0, 0.6, 6 * period, dt=5e-3)
    n_star, _ = pp.predator_prey_fixed_point(**PARAMS)
    dn = y[:, 0] - n_star
    up = np.where((dn[:-1] < 0) & (dn[1:] >= 0))[0]
    assert np.diff(t[up]).mean() > 1.05 * period


def test_arrays_broadcast():
    dn, de = pp.predator_prey_rhs(np.array([0.5, 1.0]), 0.6, **PARAMS)
    assert dn.shape == de.shape == (2,)
    assert pp.predator_prey_period(np.array([0.3, 1.2]), 0.1).shape == (2,)


@pytest.mark.parametrize("bad", [
    dict(PARAMS, gamma_eff=-0.1),
    dict(PARAMS, coupling_drive=np.nan),
])
def test_unphysical_coefficients_are_refused(bad):
    with pytest.raises(ValueError):
        pp.predator_prey_rhs(1.0, 1.0, **bad)


def test_strict_positivity_where_the_formula_needs_it():
    with pytest.raises(ValueError):
        pp.predator_prey_fixed_point(**dict(PARAMS, coupling_suppression=0.0))
    with pytest.raises(ValueError):
        pp.predator_prey_invariant(0.0, 1.0, **PARAMS)
    with pytest.raises(ValueError):
        pp.predator_prey_rhs(-1.0, 1.0, **PARAMS)
    with pytest.raises(ValueError, match="approximation"):
        pp.predator_prey_period(0.3, 0.1, approximation="exact")
