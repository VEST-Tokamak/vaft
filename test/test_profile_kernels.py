"""Analytic psi_N profile kernels: generalized parabolic and Groebner's mtanh (#552).

The catalog and docstring suites only look these up; this file calls them.
The mtanh tests pin the parameter semantics VAFT adopts -- full width,
symmetry point at the steepest gradient, height above the edge value -- so a
later change of convention cannot pass silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.formula.equilibrium import (
    generalized_parabolic_profile as gp,
    generalized_parabolic_profile_derivative as dgp,
    modified_tanh_profile as mtanh,
    modified_tanh_profile_derivative as dmtanh,
)

H = 1e-6


def _fd(f, x, **kw):
    return (f(x + H, **kw) - f(x - H, **kw)) / (2 * H)


# --- generalized parabolic ------------------------------------------------------


def test_gp_endpoints_and_shape():
    kw = dict(core_value=5.0, edge_value=0.5, alpha=2.0, beta=1.5)
    assert gp(0.0, **kw) == pytest.approx(5.0)
    assert gp(1.0, **kw) == pytest.approx(0.5)
    x = np.linspace(0, 1, 12).reshape(3, 4)
    assert gp(x, **kw).shape == (3, 4)
    assert np.ndim(gp(0.3, **kw)) == 0


def test_gp_linear_limit_and_parameter_dependence():
    x = np.linspace(0, 1, 11)
    np.testing.assert_allclose(gp(x), 1 - x)
    mid = 0.5
    assert gp(mid, alpha=4.0) > gp(mid, alpha=1.0)   # flatter core
    assert gp(mid, beta=3.0) < gp(mid, beta=1.0)     # more peaked


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (2.0, 1.5), (1.5, 3.0), (3.0, 2.0)])
def test_gp_derivative_matches_finite_differences(alpha, beta):
    x = np.linspace(0.02, 0.98, 25)
    kw = dict(core_value=3.0, edge_value=0.2, alpha=alpha, beta=beta)
    np.testing.assert_allclose(dgp(x, **kw), _fd(gp, x, **kw), rtol=1e-6, atol=1e-8)


def test_gp_derivative_endpoint_limits():
    assert dgp(0.0, alpha=1.0, beta=2.0) == pytest.approx(-2.0)
    assert dgp(0.0, alpha=2.0, beta=2.0) == 0.0
    assert np.isinf(dgp(0.0, alpha=0.5))
    assert dgp(1.0, beta=2.0) == 0.0
    assert dgp(1.0, alpha=3.0, beta=1.0) == pytest.approx(-3.0)
    assert np.isinf(dgp(1.0, beta=0.5))


@pytest.mark.parametrize("kw", [dict(alpha=0.0), dict(beta=-1.0), dict(core_value=np.nan), dict(alpha=np.inf)])
def test_gp_rejects_bad_parameters(kw):
    with pytest.raises(ValueError):
        gp(0.5, **kw)
    with pytest.raises(ValueError):
        dgp(0.5, **kw)


@pytest.mark.parametrize("x", [-0.1, 1.2, np.nan])
def test_gp_refuses_psi_n_outside_the_plasma(x):
    with pytest.raises(ValueError):
        gp(x)


# --- modified tanh -------------------------------------------------------------

PED = dict(pedestal_height=2.0, pedestal_position=0.95, pedestal_width=0.04, edge_value=0.1)


def test_mtanh_height_and_edge_semantics():
    assert mtanh(0.0, **PED) == pytest.approx(2.1, abs=1e-12)   # top = edge + height
    assert mtanh(2.0, **PED) == pytest.approx(0.1, abs=1e-12)   # outside -> edge value
    assert mtanh(0.95, **PED) == pytest.approx(1.1)             # halfway at the symmetry point


def test_mtanh_width_is_the_full_width_between_knee_and_foot():
    knee, foot = 0.95 - 0.02, 0.95 + 0.02
    fraction = (mtanh(knee, **PED) - 0.1) / 2.0
    assert fraction == pytest.approx((1 + np.tanh(1.0)) / 2)
    assert (mtanh(foot, **PED) - 0.1) / 2.0 == pytest.approx((1 - np.tanh(1.0)) / 2)


def test_mtanh_steepest_gradient_sits_at_the_symmetry_point():
    x = np.linspace(0.8, 1.05, 50001)
    gradient = dmtanh(x, **PED)
    assert x[np.argmin(gradient)] == pytest.approx(0.95, abs=1e-5)
    assert gradient.min() == pytest.approx(-2.0 / 0.04)


def test_mtanh_core_slope_continues_into_the_core():
    sloped = dict(PED, core_slope=0.05)
    assert mtanh(0.5, **sloped) > mtanh(0.5, **PED)
    # Deep in the core the rise is h s / width per unit psi_N.
    assert dmtanh(0.5, **sloped) == pytest.approx(-2.0 * 0.05 / 0.04, rel=1e-6)
    assert mtanh(2.0, **sloped) == pytest.approx(0.1, abs=1e-9)


@pytest.mark.parametrize("slope", [0.0, 0.08, -0.03])
def test_mtanh_derivative_matches_finite_differences(slope):
    x = np.linspace(0.0, 1.2, 61)
    kw = dict(PED, core_slope=slope)
    np.testing.assert_allclose(dmtanh(x, **kw), _fd(mtanh, x, **kw), rtol=1e-5, atol=1e-5)


def test_narrow_pedestal_stays_finite_and_step_like():
    narrow = dict(PED, pedestal_width=1e-4)
    x = np.array([0.9, 0.9499, 0.9501, 1.0])
    values = mtanh(x, **narrow)
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(dmtanh(x, **narrow)))
    np.testing.assert_allclose(values[[0, -1]], [2.1, 0.1], atol=1e-12)


@pytest.mark.parametrize("width", [0.0, -0.02, np.nan])
def test_mtanh_rejects_a_non_positive_width(width):
    with pytest.raises(ValueError):
        mtanh(0.5, **dict(PED, pedestal_width=width))
    with pytest.raises(ValueError):
        dmtanh(0.5, **dict(PED, pedestal_width=width))


def test_mtanh_matches_the_literal_groebner_form():
    z = np.linspace(-4, 4, 17); s = 0.2
    literal = ((1 + s*z)*np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    x = 0.95 - z*0.04/2
    np.testing.assert_allclose(mtanh(x, **dict(PED, core_slope=s)), 0.1 + 1.0*(1 + literal), rtol=1e-12)


def test_h_mode_is_a_composition_of_the_two_kernels():
    x = np.linspace(0, 1, 101)
    core = gp(x, core_value=3.0, edge_value=0.0, alpha=2.0, beta=2.0)
    h_mode = core + mtanh(x, **PED)
    assert h_mode[0] == pytest.approx(3.0 + 2.1)
    assert np.argmin(np.gradient(h_mode, x)) == pytest.approx(95, abs=1)
