"""VDE reference quantities (#1042): evaluated, not just catalogued."""

import math

import numpy as np
import pytest

from vaft.formula.constants import MU0
from vaft.formula.vde import (
    halo_current_fraction,
    thin_wall_time,
    toroidal_peaking_factor,
    vde_growth_rate,
    vertical_velocity,
    wall_mode_decay_time,
)


def test_the_growth_rate_of_an_exponential_displacement_is_its_exponent():
    t = np.linspace(0.0, 0.01, 2001)
    Z = 0.05 + 0.002 * np.exp(300.0 * t)
    g = vde_growth_rate(t, Z, 0.05)
    np.testing.assert_allclose(g[5:-5], 300.0, rtol=1e-4)
    np.testing.assert_allclose(vertical_velocity(t, Z)[5:-5], 0.6 * np.exp(300.0 * t[5:-5]), rtol=1e-4)
    # a downward displacement grows at the same rate: |dZ|
    np.testing.assert_allclose(vde_growth_rate(t, 0.05 - (Z - 0.05), 0.05)[5:-5], 300.0, rtol=1e-4)
    with pytest.raises(ValueError):
        vde_growth_rate(t[::-1], Z, 0.05)


def test_the_thin_wall_time_and_its_harmonics():
    assert thin_wall_time(1.4e6, 0.02, 0.8) == pytest.approx(MU0 * 1.4e6 * 0.02 * 0.8)
    assert wall_mode_decay_time(1.0, 1) == pytest.approx(0.5)
    assert wall_mode_decay_time(1.0, 3) == pytest.approx(1 / 6)
    for bad in ((0.0, 0.02, 0.8), (1e6, -0.01, 0.8)):
        with pytest.raises(ValueError):
            thin_wall_time(*bad)
    with pytest.raises(ValueError):
        wall_mode_decay_time(1.0, 0)


def test_halo_fraction_and_peaking():
    assert halo_current_fraction(0.3e6, -1.0e6) == pytest.approx(0.3)
    phi = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    assert toroidal_peaking_factor(np.full(64, 5.0)) == pytest.approx(1.0)
    assert toroidal_peaking_factor(1.0 + 0.5 * np.cos(phi)) == pytest.approx(1.5)
    np.testing.assert_allclose(toroidal_peaking_factor(np.stack([np.ones(8), np.arange(8.0)])), [1.0, 7 / 3.5])
    for bad in (np.array([1.0]), np.array([1.0, -1.0]), np.zeros(4)):
        with pytest.raises(ValueError):
            toroidal_peaking_factor(bad)
    with pytest.raises(ValueError):
        halo_current_fraction(-1.0, 1e6)
