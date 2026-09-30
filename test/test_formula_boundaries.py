"""The operational-boundary data model (#1067): values, margins, windows, curves, registry.

The catalog tests only read docstrings; everything here calls the functions
and checks numbers.
"""

import dataclasses
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
# Signs: a negative current or a non-positive boundary is never "safe"
# ------------------------------------------------------------------

def test_greenwald_uses_the_current_magnitude():
    """OMAS stores I_p with its COCOS sign; the limit must not flip with it."""
    entry = B.get_boundary("greenwald")
    positive = B.evaluate_boundary(entry, 5.0, plasma_current=0.1, minor_radius=0.25)
    negative = B.evaluate_boundary(entry, 5.0, plasma_current=-0.1, minor_radius=0.25)
    assert negative.boundary_value > 0
    assert negative.boundary_value == pytest.approx(positive.boundary_value)
    assert negative.margin == pytest.approx(positive.margin)
    assert negative.allowed is positive.allowed is True


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan])
def test_non_positive_boundary_is_never_permitted(value):
    boundary = B.Boundary(key="bad", family="toy", target=_P, inputs=(), form="threshold",
                          coefficient=value, allowed_side="below", sources=_SRC)
    result = B.evaluate_boundary(boundary, -2.0)
    assert result.allowed is False
    assert math.isnan(result.margin) and math.isnan(result.ratio)
    assert any("non-positive" in w for w in result.warnings)


def test_zero_current_slices_are_flagged_in_an_array():
    entry = B.get_boundary("greenwald")
    result = B.evaluate_boundary(entry, np.array([1.0, 1.0]), plasma_current=np.array([0.08, 0.0]), minor_radius=0.24)
    np.testing.assert_array_equal(result.allowed, [True, False])
    assert np.isnan(result.margin[1]) and result.margin[0] > 0


def test_non_finite_inputs_are_out_of_domain():
    boundary = _power_law(ranges={"x": (0.0, 10.0)})
    result = B.evaluate_boundary(boundary, 1.0, x=np.nan, y=1.0)
    assert result.extrapolated == ("x",)


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


def test_window_routes_each_edge_its_own_inputs():
    lower = B.Boundary(key="lo", family="toy", target=_P, inputs=(_X,), form="power_law",
                       coefficient=1.0, exponents={"x": 1.0}, allowed_side="above", sources=_SRC)
    upper = B.Boundary(key="hi", family="toy", target=_P, inputs=(_Y,), form="power_law",
                       coefficient=10.0, exponents={"y": 1.0}, allowed_side="below", sources=_SRC)
    window = B.BoundaryWindow("w", "toy_regime", lower=lower, upper=upper)
    result = B.evaluate_window(window, 5.0, x=2.0, y=1.0)
    assert result.lower.boundary_value == 2.0 and result.upper.boundary_value == 10.0
    assert result.inside is True
    assert B.evaluate_window(window, 5.0, x=6.0, y=1.0).inside is False
    with pytest.raises(TypeError, match="does not use"):
        B.evaluate_window(window, 5.0, x=2.0, y=1.0, z=0.0)


def test_window_edges_match_by_quantity_identity_not_wording():
    reworded = B.BoundaryQuantity("power", "P_{\\rm loss}", "MW", "Same quantity, other words.")
    upper = B.Boundary(key="u", family="toy", target=reworded, inputs=(), form="threshold",
                       coefficient=3.0, allowed_side="below", sources=_SRC)
    B.BoundaryWindow("w", "r", lower=_threshold(1.0, "above"), upper=upper)


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


def test_entries_are_immutable_and_hashable():
    entry = B.get_boundary("greenwald")
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.allowed_side = "above"
    with pytest.raises(TypeError):
        entry.applicability.ranges["plasma_current"] = (0, 1)
    assert entry in {entry}
    applicability = B.Applicability(ranges={"x": [0.0, 1.0]})
    assert applicability.ranges["x"] == (0.0, 1.0)


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


def test_curve_arrays_are_read_only_and_scalars_become_one_point():
    curve = B.boundary_curve(B.get_boundary("greenwald"), "plasma_current", 0.1, minor_radius=0.25)
    assert curve.xy.shape == (1, 2)
    with pytest.raises(ValueError):
        curve.y[0] = 0.0


def test_curve_rejects_array_fixed_inputs():
    with pytest.raises(TypeError, match="scalars"):
        B.boundary_curve(B.get_boundary("greenwald"), "plasma_current", [0.1, 0.2, 0.3],
                         minor_radius=np.array([0.2, 0.3]))


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


# ------------------------------------------------------------------
# Hugill diagram (#1068): the Greenwald line in (nR/B, 1/q_cyl)
# ------------------------------------------------------------------

def test_greenwald_hugill_slope_is_50_kappa_over_pi():
    entry = B.get_boundary("greenwald_hugill")
    inv_q = np.linspace(0.05, 0.5, 10)
    curve = B.boundary_curve(entry, "inverse_cylindrical_q", inv_q, swap_axes=True, area_elongation=1.7)
    np.testing.assert_allclose(curve.x / curve.y, 50.0 * 1.7 / np.pi)
    assert curve.x_quantity.name == "murakami_parameter" and curve.y_quantity.name == "inverse_cylindrical_q"
    assert curve.allowed_side == "left"


@pytest.mark.parametrize("kappa", [1.0, 1.7])
def test_greenwald_hugill_is_the_diagram_line(kappa):
    from vaft.diagram._stability_space import hugill

    xy = hugill(elongation=kappa, labels=False).model.curves["greenwald"][1:]
    entry = B.get_boundary("greenwald_hugill")
    np.testing.assert_allclose(
        B.boundary_value(entry, inverse_cylindrical_q=xy[:, 1], area_elongation=kappa), xy[:, 0], rtol=1e-12
    )


@pytest.mark.parametrize("R, B_t, a, kappa", [(0.4, 0.15, 0.236, 1.52), (1.7, 2.0, 0.6, 1.8)])
def test_a_state_at_the_greenwald_density_lies_on_the_hugill_line(R, B_t, a, kappa):
    """Any machine size and shape: a, R, B_T and kappa_a cancel, so n = n_G(I_p, a) is exactly on the line."""
    I_p = np.linspace(0.03, 0.12, 7) * (a / 0.236) ** 2
    x, y = B.hugill_coordinates(greenwald_density(I_p, a), R, B_t, a, kappa, I_p)
    result = B.evaluate_boundary(B.get_boundary("greenwald_hugill"), x, inverse_cylindrical_q=y, area_elongation=kappa)
    np.testing.assert_allclose(result.ratio, 1.0)
    below = B.evaluate_boundary(B.get_boundary("greenwald_hugill"), 0.5 * x, inverse_cylindrical_q=y, area_elongation=kappa)
    assert np.all(below.allowed) and np.allclose(below.margin, 0.5)


def test_hugill_coordinates_drop_the_current_and_field_sign():
    plus = B.hugill_coordinates(3.0, 0.4, 0.15, 0.236, 1.52, 0.08)
    minus = B.hugill_coordinates(3.0, 0.4, -0.15, 0.236, 1.52, -0.08)
    assert plus == pytest.approx(minus)
    assert plus[0] == pytest.approx(3.0 * 0.4 / 0.15)


def test_hugill_y_is_the_inverse_of_the_repository_q_cyl():
    from vaft.formula.equilibrium import q_cyl_from_B_R_epsilon_kappa_I

    R, B_t, a, kappa, I_p = 0.4, 0.15, 0.236, 1.52, np.array([0.03, 0.08])
    _, y = B.hugill_coordinates(1.0, R, B_t, a, kappa, I_p)
    np.testing.assert_allclose(y, 1.0 / q_cyl_from_B_R_epsilon_kappa_I(B_t, R, a / R, kappa, I_p * 1e6), rtol=1e-12)
    assert B.hugill_coordinates(3.0, R, B_t, a, kappa, 0.08)[1] == pytest.approx(0.4 * 0.08 / (5 * 0.236**2 * 1.52 * 0.15))


def test_hugill_coordinates_map_zero_current_to_the_origin():
    """A VEST time trace starts and ends at I_p = 0; it must project, not raise."""
    x, y = B.hugill_coordinates(np.array([0.0, 2.0]), 0.4, 0.15, 0.236, 1.52, np.array([0.0, 0.08]))
    assert x[0] == 0.0 and y[0] == 0.0 and y[1] > 0


def test_hugill_coordinates_reject_unphysical_inputs():
    with pytest.raises(ValueError, match="n_e"):
        B.hugill_coordinates(-1.0, 0.4, 0.15, 0.236, 1.52, 0.08)
    with pytest.raises(ValueError, match="kappa_a"):
        B.hugill_coordinates(1.0, 0.4, 0.15, 0.236, 0.0, 0.08)


def test_swap_axes_maps_above_to_right():
    boundary = _power_law(side="above")
    curve = B.boundary_curve(boundary, "x", [1.0, 2.0], swap_axes=True, y=1.0)
    assert curve.allowed_side == "right"
    np.testing.assert_allclose(curve.x, [2.0, 4.0])
    np.testing.assert_allclose(curve.y, [1.0, 2.0])


# ------------------------------------------------------------------
# Murakami limit (#1068), checked against Murakami, Callen & Berry,
# Nucl. Fusion 16 (1976) 347, Table I: (device, R0 [m], B_T [T], n_e max [1e19 m^-3], gas injection)
# ------------------------------------------------------------------

_MURAKAMI_TABLE_I = (
    ("Alcator", 0.54, 7.5, 35.0, True),
    ("TM-3", 0.40, 3.5, 7.0, False),
    ("TFR", 0.98, 5.0, 6.3, False),
    ("T-4", 0.90, 4.5, 4.0, False),
    ("Pulsator", 0.70, 2.7, 10.0, True),
    ("ST", 1.09, 4.3, 5.7, False),
    ("T-3", 0.90, 3.4, 4.5, False),
    ("ORMAK", 0.80, 2.5, 3.7, False),
    ("ORMAK", 0.80, 1.8, 3.0, False),
    ("CLEO", 0.90, 1.9, 2.0, False),
    ("ATC", 0.90, 1.5, 2.1, False),
    ("JFT-2", 0.90, 1.0, 1.0, False),
    ("T-6", 0.70, 0.6, 1.0, False),
)


def test_murakami_is_B_over_R_in_1e19():
    entry = B.get_boundary("murakami")
    assert entry.target.name == "line_average_density" and entry.target.unit == "1e19 m^-3"
    assert B.boundary_value(entry, toroidal_field=7.5, major_radius=0.54) == pytest.approx(7.5 / 0.54)
    assert entry.allowed_side == "below" and entry.hardness == "soft"


def test_murakami_line_matches_table_I():
    """Stationary-fill devices scatter about the line; cold-gas injection sits ~2.5x above (Greenwald 2002: ~2x)."""
    entry = B.get_boundary("murakami")
    ratios = {True: [], False: []}
    for _, R0, B_T, n_max, injected in _MURAKAMI_TABLE_I:
        ratios[injected].append(n_max / B.boundary_value(entry, toroidal_field=B_T, major_radius=R0))
    stationary = np.array(ratios[False])
    assert np.all((stationary > 0.7) & (stationary < 1.6))
    assert 0.9 < np.median(stationary) < 1.3
    assert np.all(np.array(ratios[True]) > 2.0)


def test_murakami_ranges_are_table_I():
    entry = B.get_boundary("murakami")
    R0 = [row[1] for row in _MURAKAMI_TABLE_I]
    B_T = [row[2] for row in _MURAKAMI_TABLE_I]
    assert entry.applicability.ranges["major_radius"] == (min(R0), max(R0))
    assert entry.applicability.ranges["toroidal_field"] == (min(B_T), max(B_T))
    vest = B.evaluate_boundary(entry, 3.0, toroidal_field=0.15, major_radius=0.4)
    assert vest.extrapolated == ("toroidal_field",)  # VEST's field is below every Table I device


def test_murakami_on_the_hugill_diagram_is_the_same_limit():
    """n <= B/R  <=>  n R/B <= 1: both entries give the same margin for any state."""
    line, vertical = B.get_boundary("murakami"), B.get_boundary("murakami_hugill")
    for n, B_T, R0 in [(2.0, 1.5, 0.9), (0.5, 0.15, 0.4), (12.0, 3.0, 0.3)]:
        m1 = B.evaluate_boundary(line, n, toroidal_field=B_T, major_radius=R0).margin
        m2 = B.evaluate_boundary(vertical, n * R0 / B_T).margin
        assert m1 == pytest.approx(m2)


def test_greenwald_hugill_slope_is_greenwalds_circular_form():
    """Greenwald 1988, after Eq. (1): circular limit (5/pi) B/(qR) in 1e20 m^-3 -> 50/pi in 1e19."""
    entry = B.get_boundary("greenwald_hugill")
    assert entry.coefficient == pytest.approx(10 * 5 / np.pi)


def test_low_q_is_q_psi_above_two():
    entry = B.get_boundary("low_q")
    assert entry.target.name == "edge_safety_factor" and B.boundary_value(entry) == 2.0
    assert B.evaluate_boundary(entry, 3.0).allowed and not B.evaluate_boundary(entry, 1.8).allowed
    assert entry.family == "current_limit" and entry.event == "disruption"


def test_every_published_entry_cites_a_location_in_its_source():
    for key in B.list_boundaries():
        entry = B.get_boundary(key)
        if entry.origin == "published":
            assert all(source.equation for source in entry.sources), key


# ------------------------------------------------------------------
# Giacomin et al. 2022 edge density limit (#1068): the paper's worked predictions
# ------------------------------------------------------------------

def _giacomin(A, a, P, R, q, kappa, B_T):
    return B.boundary_value(B.get_boundary("giacomin_edge"), mass_number=A, minor_radius=a,
                            separatrix_power=P, major_radius=R, edge_safety_factor_95=q,
                            elongation=kappa, toroidal_field=B_T)


@pytest.mark.parametrize("A, a, P, R, q, kappa, B_T, expected, rel", [
    (2.0, 0.22, 5.0, 0.67, 4.0, 1.5, 8.0, 5.0, 0.10),     # Alcator C-Mod, p. 5: "n_lim = 5e20"
    (2.0, 2.0, 50.0, 6.2, 3.0, 1.8, 5.3, 2.5, 0.10),      # ITER, p. 5: "~2.5e20"
    (2.5, 0.57, 28.0, 1.85, 3.0, 2.0, 12.2, 8.7, 0.01),   # SPARC, p. 5: "~8.7e20"
])
def test_giacomin_reproduces_the_papers_predictions(A, a, P, R, q, kappa, B_T, expected, rel):
    """The paper states no mass number for these cases. SPARC pins it: A = 2.5 (D-T) gives 8.70
    exactly. C-Mod and ITER are quoted to one significant figure and fit within 10 % for A = 1-3,
    so they check the geometry and power dependence, not A."""
    assert _giacomin(A, a, P, R, q, kappa, B_T) == pytest.approx(expected, rel=rel)


def test_giacomin_rejects_signed_field_as_non_finite():
    entry = B.get_boundary("giacomin_edge")
    result = B.evaluate_boundary(entry, 0.3, mass_number=2.0, minor_radius=0.5, separatrix_power=2.0,
                                 major_radius=1.5, edge_safety_factor_95=4.0, elongation=1.5, toroidal_field=-2.0)
    assert result.allowed is False and any("non-positive or not finite" in w for w in result.warnings)


def test_giacomin_exponents_are_eq_12():
    base = dict(A=2.0, a=0.5, P=2.0, R=1.5, q=4.0, kappa=1.5, B_T=2.0)
    ref = _giacomin(**base)
    for name, factor, exponent in [("A", 2, 1 / 6), ("a", 2, 3 / 14), ("P", 2, 10 / 21),
                                   ("R", 2, -43 / 42), ("q", 2, -22 / 21), ("B_T", 2, 2 / 3)]:
        scaled = dict(base, **{name: base[name] * factor})
        assert _giacomin(**scaled) / ref == pytest.approx(factor ** exponent)
    assert _giacomin(**dict(base, kappa=2.0)) / ref == pytest.approx(((1 + 4.0) / (1 + 2.25)) ** (-1 / 3))


def test_giacomin_targets_edge_density_and_reports_extrapolation():
    entry = B.get_boundary("giacomin_edge")
    assert entry.target.name == "edge_density" and entry.target.unit == "1e20 m^-3"
    assert entry.uncertainty.coefficient == 0.3
    vest = B.evaluate_boundary(entry, 0.05, mass_number=1.0, minor_radius=0.24, separatrix_power=0.1,
                               major_radius=0.4, edge_safety_factor_95=6.0, elongation=1.5, toroidal_field=0.15)
    assert set(vest.extrapolated) == {"toroidal_field", "major_radius"}


# ------------------------------------------------------------------
# L-H threshold family (#1066): Martin et al. 2008 and Ryter et al. 2014
# ------------------------------------------------------------------

# ITER as the papers use it: B_T = 5.3 T and S = 678 m^2 (Martin 2008, p. 4); I_p = 15 MA
# (Martin 2008, p. 7); R0 = 6.2 m and a = 2 m (Giacomin et al. 2022, p. 5).
_ITER = dict(B_T=5.3, S=678.0, I_p=15.0, R=6.2, a=2.0)


@pytest.mark.parametrize("n_e20, expected", [(0.5, 52.0), (1.0, 86.0)])
def test_martin_reproduces_table_1_for_iter(n_e20, expected):
    entry = B.get_boundary("martin_2008_lh")
    P = B.boundary_value(entry, line_average_density=n_e20, toroidal_field=_ITER["B_T"],
                         plasma_surface_area=_ITER["S"])
    assert P == pytest.approx(expected, rel=0.01)


def test_martin_is_an_L_to_H_access_threshold_on_loss_power():
    entry = B.get_boundary("martin_2008_lh")
    assert (entry.source_regime, entry.target_regime, entry.branch) == ("L_mode", "H_mode", "high_density")
    assert entry.target.name == "loss_power" and entry.allowed_side == "above"
    assert entry.input("line_average_density").unit == "1e20 m^-3"
    below = B.evaluate_boundary(entry, 40.0, line_average_density=0.5, toroidal_field=5.3, plasma_surface_area=678.0)
    above = B.evaluate_boundary(entry, 60.0, line_average_density=0.5, toroidal_field=5.3, plasma_surface_area=678.0)
    assert not below.allowed and above.allowed and above.ratio == pytest.approx(60.0 / 52.3, rel=0.01)


def test_martin_uncertainty_is_the_published_fit_statistics():
    u = B.get_boundary("martin_2008_lh").uncertainty
    assert u.coefficient_factor == pytest.approx(np.exp(0.057))
    assert dict(u.exponents) == {"line_average_density": 0.035, "toroidal_field": 0.032, "plasma_surface_area": 0.019}
    assert u.rms_relative == 0.308


def _ryter(I_p, B_T, a, aspect_ratio):
    return B.boundary_value(B.get_boundary("ryter_2014_nmin"), plasma_current=I_p, toroidal_field=B_T,
                            minor_radius=a, aspect_ratio=aspect_ratio)


def test_ryter_reproduces_the_iter_minimum_density():
    """Ryter 2014, p. 8: ~4e19 m^-3 at full field and current, ~2.2e19 at half field and current."""
    full = _ryter(_ITER["I_p"], _ITER["B_T"], _ITER["a"], _ITER["R"] / _ITER["a"])
    half = _ryter(_ITER["I_p"] / 2, _ITER["B_T"] / 2, _ITER["a"], _ITER["R"] / _ITER["a"])
    assert full == pytest.approx(4.0, rel=0.05)
    assert half == pytest.approx(2.2, rel=0.10)


def test_martin_at_ryter_minimum_is_the_papers_minimum_power():
    """Ryter 2014, p. 8: inserting n_e,min in the threshold scaling gives ~41 MW for ITER at full field.
    Eq. (1) at n_e,min (4.0e19) gives ~44 MW; the printed Eq. (4) would give ~62 MW and is not registered."""
    n_min_1e20 = _ryter(_ITER["I_p"], _ITER["B_T"], _ITER["a"], _ITER["R"] / _ITER["a"]) / 10.0
    P_min = B.boundary_value(B.get_boundary("martin_2008_lh"), line_average_density=n_min_1e20,
                             toroidal_field=_ITER["B_T"], plasma_surface_area=_ITER["S"])
    assert P_min == pytest.approx(41.0, rel=0.10)


def test_ryter_minimum_bounds_the_martin_branch_from_below():
    entry = B.get_boundary("ryter_2014_nmin")
    assert entry.allowed_side == "above" and entry.branch == "low_density_boundary"
    assert entry.target.unit == "1e19 m^-3"
    assert "lh_threshold" == entry.family == B.get_boundary("martin_2008_lh").family
    assert set(B.list_boundaries("lh_threshold")) == {"martin_2008_lh", "ryter_2014_nmin"}


def test_transition_needs_both_regimes():
    with pytest.raises(ValueError, match="source_regime and target_regime"):
        B.Boundary(key="half", family="toy", target=_P, inputs=(), form="threshold", coefficient=1.0,
                   allowed_side="above", sources=_SRC, source_regime="L_mode")
