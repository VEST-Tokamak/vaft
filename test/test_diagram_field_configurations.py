"""Canonical field configurations and reconnection topology (#1063): the drawn fields are the formulas'."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram._scene import Label
from vaft.formula.constants import MU0
from vaft.formula.geometry import (
    harris_sheet_current_density,
    harris_sheet_field,
    sheared_slab_field,
    x_point_flux,
)
from vaft.formula.stability import slab_perturbed_flux


def test_the_harris_current_is_ampere_of_the_harris_field():
    B0, a = 0.3, 0.02
    x = np.linspace(-5 * a, 5 * a, 2001)
    B = harris_sheet_field(x, B0, a)
    np.testing.assert_allclose(B, -harris_sheet_field(-x, B0, a))
    np.testing.assert_allclose(np.gradient(B, x) / MU0, harris_sheet_current_density(x, B0, a), rtol=2e-4, atol=1e2)
    assert harris_sheet_current_density(0.0, B0, a) == pytest.approx(B0 / (MU0 * a))
    # the sheet carries 2 B0 / mu0 per unit length
    xx = np.linspace(-40 * a, 40 * a, 40001)
    assert np.trapezoid(harris_sheet_current_density(xx, B0, a), xx) == pytest.approx(2 * B0 / MU0, rel=1e-6)
    for bad in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError):
            harris_sheet_field(0.1, B0, bad)
        with pytest.raises(ValueError):
            harris_sheet_current_density(0.1, B0, bad)


def test_the_x_point_is_a_current_free_null_with_right_angle_separatrices():
    Bp, h = 2.0, 1e-5
    x, y = 0.3, -0.2
    dpsi_dx = (x_point_flux(x + h, y, Bp) - x_point_flux(x - h, y, Bp)) / (2 * h)
    dpsi_dy = (x_point_flux(x, y + h, Bp) - x_point_flux(x, y - h, Bp)) / (2 * h)
    # B_perp = z x grad psi = (-dpsi/dy, dpsi/dx) = B'(y, x)
    assert (-dpsi_dy, dpsi_dx) == pytest.approx((Bp * y, Bp * x))
    lap = (x_point_flux(x + h, y, Bp) + x_point_flux(x - h, y, Bp) + x_point_flux(x, y + h, Bp)
           + x_point_flux(x, y - h, Bp) - 4 * x_point_flux(x, y, Bp)) / h**2
    assert lap == pytest.approx(0.0, abs=1e-4)
    s = np.linspace(-1, 1, 11)
    np.testing.assert_allclose(x_point_flux(s, s, Bp), 0.0)
    np.testing.assert_allclose(x_point_flux(s, -s, Bp), 0.0)
    # the same saddle as the tearing flux about its X-point (x = 0, cos k_y y = 1), to second order
    shear, amp, k = 1.0, 0.01, 1.0
    for xx, yy in ((0.01, 0.0), (0.0, 0.01), (0.007, -0.005)):
        tearing = slab_perturbed_flux(xx, yy, shear, amp, k) - amp
        assert tearing == pytest.approx(x_point_flux(xx, yy * math.sqrt(amp * k * k / shear), shear), rel=1e-3)
    with pytest.raises(ValueError):
        x_point_flux(0.0, 0.0, 0.0)


@pytest.mark.parametrize("kind", ["uniform", "sheared", "reversed", "guide"])
def test_the_sheets_carry_the_formula_fields(kind):
    m = vaft.diagram.slab_field_configuration(kind).model
    x, field = m["x"], m["field"]
    if kind == "sheared":
        np.testing.assert_allclose(field, sheared_slab_field(x, 1.0, 1.2)[:, 1:])
    if kind in ("reversed", "guide"):
        np.testing.assert_allclose(field[:, 0], harris_sheet_field(x, 1.0, 1.0))
        np.testing.assert_allclose(field[:, 0], -field[::-1, 0])  # B_y odd
    magnitude = np.hypot(field[:, 0], field[:, 1])
    if kind == "reversed":
        assert magnitude[2] == 0.0 and np.all(field[:, 1] == 0.0)
    if kind == "guide":
        assert magnitude.min() == pytest.approx(0.8)  # no null
    if kind == "uniform":
        np.testing.assert_allclose(field, [[0.0, 1.0]] * 5)


def test_an_unknown_configuration_is_refused():
    with pytest.raises(ValueError):
        vaft.diagram.slab_field_configuration("pinch")
    with pytest.raises(ValueError):
        vaft.diagram.current_sheet(guide_field="yes")


@pytest.mark.parametrize("guide", [False, True])
def test_the_current_sheet_lines_are_at_equal_flux_steps(guide):
    m = vaft.diagram.current_sheet(guide_field=guide).model
    xs = sorted(x for x, _ in m["field_lines"] if x > 0)
    flux = np.log(np.cosh(np.array(xs)))  # A_z / (B0 a)
    np.testing.assert_allclose(np.diff(flux), np.diff(flux)[0], rtol=1e-9)
    for x, b_y in m["field_lines"]:
        assert b_y == pytest.approx(float(harris_sheet_field(x, 1.0, 1.0)))
        assert np.sign(b_y) == np.sign(x)
    assert m["J0"] == pytest.approx(1.0 / MU0)
    assert m["B_g"] == (0.8 if guide else 0.0)


def test_the_x_point_contours_are_flux_contours():
    m = vaft.diagram.x_point().model
    for line in m["lines"]:
        psi = x_point_flux(line[:, 1], line[:, 0], 1.0)  # drawn as (y, x)
        assert np.ptp(psi) < 1e-2 * max(abs(v) for v in m["levels"])


def test_reconnection_separates_upstream_from_reconnected_flux():
    m = vaft.diagram.magnetic_reconnection().model
    for name, sign in (("upstream", 1.0), ("reconnected", -1.0)):
        for line in m[name]:
            psi = x_point_flux(line[:, 1], line[:, 0] / m["stretch"], 1.0)
            assert np.all(sign * psi > 0)


def test_the_island_grows_with_the_tearing_amplitude():
    m = vaft.diagram.island_formation().model
    amps, widths = m["amplitudes"], m["widths"]
    assert amps[0] == 0.0 and widths[0] == 0.0
    assert list(widths) == sorted(widths)
    for a, w in zip(amps, widths):
        assert w == pytest.approx(4 * math.sqrt(a))


@pytest.mark.parametrize("name", ["slab_field_configuration", "current_sheet", "harris_sheet", "x_point",
                                  "magnetic_reconnection", "island_formation"])
def test_every_configuration_diagram_is_deterministic_and_exported(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    n_labels = lambda d: sum(isinstance(i, Label) for i in d.scene.items)  # noqa: E731
    assert fn().scene.role("equations") or fn().scene.role("note")
    assert n_labels(fn(labels=False)) < n_labels(fn())
    with pytest.raises(ValueError):
        fn(labels="yes")
