"""The operational-boundary data model (#1067): values, margins, windows, curves, registry.

The catalog tests only read docstrings; everything here calls the functions
and checks numbers.
"""

import math

import numpy as np
import pytest

from vaft.formula import boundaries as B
from vaft.formula.stability import greenwald_density

_X = B.BoundaryQuantity("x", "x", "m")
_Y = B.BoundaryQuantity("y", "y", "T")
_P = B.BoundaryQuantity("power", "P", "MW")
_SRC = (B.BoundarySource("test source", equation="Eq. (0)"),)


def _power_law(side="below", ranges=None):
    return B.Boundary(
        key="toy", family="toy", target=_P, inputs=(_X, _Y), form="power_law",
        coefficient=2.0, exponents={"x": 1.0, "y": -2.0}, allowed_side=side, sources=_SRC,
        applicability=B.Applicability(ranges=ranges or {}),
    )


# ------------------------------------------------------------------
# Greenwald entry: references the existing function, unchanged
# ------------------------------------------------------------------

def test_greenwald_entry_is_the_existing_function():
    entry = B.get_boundary("greenwald")
    I_p = np.array([0.05, 0.08, 1.0, 15.0])
    a = np.array([0.24, 0.3, 0.5, 2.0])
    np.testing.assert_array_equal(
        B.boundary_value(entry, plasma_current=I_p, minor_radius=a), greenwald_density(I_p, a)
    )


def test_greenwald_literature_form_in_1e20():
    """n_G[1e20 m^-3] = I_p[MA]/(pi a^2): I_p = 1 MA over pi a^2 = 1 m^2 is 1e20 m^-3."""
    entry = B.get_boundary("greenwald")
    value = B.boundary_value(entry, plasma_current=1.0, minor_radius=1.0 / math.sqrt(math.pi))
    assert entry.target.unit == "1e19 m^-3"
    assert value == pytest.approx(10.0)


def test_greenwald_metadata():
    entry = B.get_boundary("greenwald")
    assert entry.target.name == "line_average_density"
    assert entry.allowed_side == "below"
    assert entry.hardness == "soft"
    assert entry.input("plasma_current").unit == "MA"
    assert entry.input("minor_radius").unit == "m"
    assert any("Eq. (1)" in s.equation for s in entry.sources)
    assert entry.uncertainty.coefficient is None  # the source quotes none; none is invented


def test_greenwald_fraction_above_one_is_reported_not_forbidden():
    entry = B.get_boundary("greenwald")
    n_G = greenwald_density(0.1, 0.25)
    result = B.evaluate_boundary(entry, 1.2 * n_G, plasma_current=0.1, minor_radius=0.25)
    assert result.ratio == pytest.approx(1.2)
    assert result.margin == pytest.approx(-0.2)
    assert result.allowed is False
    assert result.in_domain


# ------------------------------------------------------------------
# Margin and direction
# ------------------------------------------------------------------

@pytest.mark.parametrize("side", ["below", "above"])
def test_margin_is_positive_exactly_on_the_permitted_side(side):
    boundary = _power_law(side)
    b = B.boundary_value(boundary, x=3.0, y=0.5)
    assert b == pytest.approx(2.0 * 3.0 / 0.25)
    under, over = B.evaluate_boundary(boundary, 0.5 * b, x=3.0, y=0.5), B.evaluate_boundary(boundary, 1.5 * b, x=3.0, y=0.5)
    permitted, forbidden = (under, over) if side == "below" else (over, under)
    assert permitted.margin == pytest.approx(0.5) and permitted.allowed
    assert forbidden.margin == pytest.approx(-0.5) and not forbidden.allowed
    assert under.difference == pytest.approx(-0.5 * b)
    assert over.ratio == pytest.approx(1.5)


def test_a_state_on_the_boundary_is_not_counted_as_permitted():
    boundary = _power_law()
    b = B.boundary_value(boundary, x=1.0, y=1.0)
    result = B.evaluate_boundary(boundary, b, x=1.0, y=1.0)
    assert result.margin == 0.0 and result.allowed is False


def test_evaluation_broadcasts_over_arrays():
    boundary = _power_law()
    x = np.linspace(1.0, 2.0, 5)
    result = B.evaluate_boundary(boundary, np.full(5, 3.0), x=x, y=1.0)
    np.testing.assert_allclose(result.boundary_value, 2.0 * x)
    np.testing.assert_array_equal(result.allowed, 2.0 * x > 3.0)


def test_extrapolation_is_named_and_warned():
    boundary = _power_law(ranges={"x": (1.0, 2.0), "y": (None, 1.0)})
    inside = B.evaluate_boundary(boundary, 1.0, x=1.5, y=0.5)
    assert inside.in_domain and inside.extrapolated == () and inside.warnings == ()
    outside = B.evaluate_boundary(boundary, 1.0, x=np.array([1.5, 2.5]), y=2.0)
    assert outside.extrapolated == ("x", "y")
    assert not outside.in_domain
    assert len(outside.warnings) == 2 and "[m]" in outside.warnings[0]


def test_inputs_must_match_the_declaration():
    boundary = _power_law()
    with pytest.raises(TypeError, match="missing"):
        B.boundary_value(boundary, x=1.0)
    with pytest.raises(TypeError, match="unexpected"):
        B.boundary_value(boundary, x=1.0, y=1.0, z=1.0)


# ------------------------------------------------------------------
# Units: the model never converts, so a unit change is the caller's rescale
# ------------------------------------------------------------------

def test_power_law_in_rescaled_units_agrees():
    """The same law declared in mm and MW must agree after the caller converts."""
    in_m = _power_law()
    x_mm = B.BoundaryQuantity("x", "x", "mm")
    in_mm = B.Boundary(
        key="toy_mm", family="toy", target=_P, inputs=(x_mm, _Y), form="power_law",
        coefficient=2.0e-3, exponents={"x": 1.0, "y": -2.0}, allowed_side="below", sources=_SRC,
    )
    assert B.boundary_value(in_mm, x=1500.0, y=0.7) == pytest.approx(B.boundary_value(in_m, x=1.5, y=0.7))


# ------------------------------------------------------------------
# Threshold and window
# ------------------------------------------------------------------

def _threshold(value, side):
    return B.Boundary(key=f"t{value}", family="toy", target=_P, inputs=(), form="threshold",
                      coefficient=value, allowed_side=side, sources=_SRC)


def test_threshold_is_constant():
    assert B.boundary_value(_threshold(4.0, "above")) == 4.0


def test_window_is_permitted_only_between_its_edges():
    window = B.BoundaryWindow("w", "toy_regime", lower=_threshold(1.0, "above"), upper=_threshold(3.0, "below"))
    result = B.evaluate_window(window, np.array([0.5, 2.0, 3.5]))
    np.testing.assert_array_equal(result.inside, [False, True, False])
    assert B.evaluate_window(window, 2.0).inside is True


def test_window_rejects_inverted_edges_and_mixed_targets():
    with pytest.raises(ValueError, match="lower boundary"):
        B.BoundaryWindow("w", "r", lower=_threshold(1.0, "below"), upper=_threshold(3.0, "below"))
    other = B.Boundary(key="o", family="toy", target=_Y, inputs=(), form="threshold",
                       coefficient=3.0, allowed_side="below", sources=_SRC)
    with pytest.raises(ValueError, match="target"):
        B.BoundaryWindow("w", "r", lower=_threshold(1.0, "above"), upper=other)


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

@pytest.mark.parametrize("change, message", [
    ({"form": "spline"}, "form"),
    ({"allowed_side": "left"}, "allowed_side"),
    ({"hardness": "firm"}, "hardness"),
    ({"sources": ()}, "BoundarySource"),
    ({"exponents": {"x": 1.0}}, "one exponent per input"),
    ({"applicability": B.Applicability(ranges={"z": (0, 1)})}, "unknown inputs"),
])
def test_invalid_boundaries_are_rejected(change, message):
    kwargs = dict(key="bad", family="toy", target=_P, inputs=(_X, _Y), form="power_law",
                  coefficient=1.0, exponents={"x": 1.0, "y": 1.0}, allowed_side="below", sources=_SRC)
    kwargs.update(change)
    with pytest.raises(ValueError, match=message):
        B.Boundary(**kwargs)


def test_applicability_rejects_a_reversed_range():
    with pytest.raises(ValueError, match="low > high"):
        B.Applicability(ranges={"x": (2.0, 1.0)})


def test_entries_are_immutable():
    entry = B.get_boundary("greenwald")
    with pytest.raises(Exception):
        entry.allowed_side = "above"
    with pytest.raises(TypeError):
        entry.applicability.ranges["plasma_current"] = (0, 1)


# ------------------------------------------------------------------
# Curves
# ------------------------------------------------------------------

def test_greenwald_curve_against_current_is_linear_and_increasing():
    I_p = np.linspace(0.02, 0.15, 30)
    curve = B.boundary_curve(B.get_boundary("greenwald"), "plasma_current", I_p, minor_radius=0.25)
    assert curve.x_quantity.unit == "MA" and curve.y_quantity.name == "line_average_density"
    assert curve.allowed_side == "below"
    assert np.all(np.diff(curve.y) > 0)
    np.testing.assert_allclose(curve.y / curve.x, greenwald_density(1.0, 0.25))
    assert curve.xy.shape == (30, 2)
    assert dict(curve.fixed) == {"minor_radius": 0.25}


def test_greenwald_curve_against_minor_radius_decreases():
    a = np.linspace(0.15, 0.35, 20)
    curve = B.boundary_curve(B.get_boundary("greenwald"), "minor_radius", a, plasma_current=0.1)
    assert np.all(np.diff(curve.y) < 0)


def test_curve_of_a_threshold_is_flat():
    boundary = B.Boundary(key="flat", family="toy", target=_P, inputs=(_X,), form="power_law",
                          coefficient=5.0, exponents={"x": 0.0}, allowed_side="above", sources=_SRC)
    curve = B.boundary_curve(boundary, "x", np.linspace(0.0, 1.0, 4))
    np.testing.assert_allclose(curve.y, 5.0)


def test_curve_rejects_unknown_or_doubly_given_sweeps():
    entry = B.get_boundary("greenwald")
    with pytest.raises(KeyError):
        B.boundary_curve(entry, "density", [1.0], minor_radius=0.25)
    with pytest.raises(TypeError, match="swept"):
        B.boundary_curve(entry, "plasma_current", [0.1], plasma_current=0.1, minor_radius=0.25)


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

def test_registry_lists_and_resolves():
    assert "greenwald" in B.list_boundaries()
    assert "greenwald" in B.list_boundaries("density_limit")
    assert B.list_boundaries("no_such_family") == ()
    with pytest.raises(KeyError, match="registered"):
        B.get_boundary("no_such_boundary")


def test_registered_keys_are_unique():
    with pytest.raises(ValueError, match="already registered"):
        B._register(B.get_boundary("greenwald"))
