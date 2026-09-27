"""Tearing diagrams: one concept each, and the drawn slopes are the formula's index (#1039)."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _tearing
from vaft.diagram._chart import CHART_HEIGHT
from vaft.formula.stability import delta_prime_from_outer_derivatives

NAMES = ("rational_surface", "delta_prime", "tearing_layer_matching")


def _curves(diagram, role):
    return [np.asarray(it.points) for it in diagram.scene.role(role) if hasattr(it, "points")]


# --- the formula ----------------------------------------------------------------------


def test_delta_prime_is_the_slope_jump_over_psi_s():
    assert delta_prime_from_outer_derivatives(2.0, -1.0, 3.0) == pytest.approx(2.0)
    assert delta_prime_from_outer_derivatives(-2.0, -1.0, 3.0) == pytest.approx(-2.0)
    # independent of the normalisation of psi
    assert delta_prime_from_outer_derivatives(5.0, -2.5, 7.5) == pytest.approx(
        delta_prime_from_outer_derivatives(1.0, -0.5, 1.5))
    np.testing.assert_allclose(delta_prime_from_outer_derivatives(1.0, [0.0, 1.0], [1.0, 0.0]), [1.0, -1.0])
    for bad in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            delta_prime_from_outer_derivatives(bad, 0.0, 1.0)


# --- rational surface --------------------------------------------------------------------


@pytest.mark.parametrize("m, n", [(2, 1), (3, 2), (1, 1), (5, 1), (1, 2), (7, 3)])
def test_the_q_profile_crosses_m_over_n_once_at_r_s(m, n):
    d = vaft.diagram.rational_surface(m, n)
    chart = d.model
    r, q = chart.curves["q_profile"].T
    assert np.all(np.diff(q) > 0)  # monotonic, as the schematic promises
    assert np.count_nonzero(np.diff(q > m / n)) == 1
    r_s = chart.parameters["r_s"]
    assert 0.0 < r_s < 1.0
    assert np.interp(r_s, r, q) == pytest.approx(m / n, rel=1e-3)
    (marker,) = [it for it in d.scene.role("rational_surface") if hasattr(it, "kind")]
    assert np.allclose(marker.at, chart.to_cm(np.array([r_s, m / n])))


@pytest.mark.parametrize("m, n", [(2, 1), (3, 1), (5, 2), (1, 1), (1, 10), (11, 10), (50, 1), (1, 11)])
def test_the_labels_stay_clear_of_the_q_curve(m, n):
    d = vaft.diagram.rational_surface(m, n)
    curve = d.model.to_cm(d.model.curves["q_profile"])
    (level,) = [it for it in d.scene.role("rational_level") if hasattr(it, "text")]
    (pitch,) = d.scene.role("region_pitch")
    for label, size in ((level, _tearing.LEVEL_LABEL_SIZE), (pitch, (4.3, 0.6))):
        x0, y0, x1, y1 = _tearing.label_box(label.at, label.anchor, size)
        inside = (curve[:, 0] > x0) & (curve[:, 0] < x1) & (curve[:, 1] > y0) & (curve[:, 1] < y1)
        assert not inside.any(), (m, n, label.text)


@pytest.mark.parametrize("m, n", [(0, 1), (2, 0), (-1, 1), (2.0, 1), (True, 1), (4, 2)])
def test_mode_numbers_are_positive_integers_in_lowest_terms(m, n):
    with pytest.raises(ValueError):
        vaft.diagram.rational_surface(m, n)


# --- Delta-prime -------------------------------------------------------------------------


@pytest.mark.parametrize("sign, expected", [("positive", 1), ("zero", 0), ("negative", -1)])
def test_the_drawn_branches_meet_and_their_slopes_give_the_sign(sign, expected):
    d = vaft.diagram.delta_prime(sign)
    p = d.model.parameters
    left, right = d.model.curves["outer_left"], d.model.curves["outer_right"]
    # both outer solutions reach the same psi(r_s) and vanish on the axis and at the edge
    assert left[-1] == pytest.approx([p["r_s"], p["psi_s"]])
    assert right[0] == pytest.approx([p["r_s"], p["psi_s"]])
    assert left[0, 1] == pytest.approx(0.0, abs=1e-12) and right[-1, 1] == pytest.approx(0.0, abs=1e-12)
    assert left[:, 1].min() >= -1e-12 and right[:, 1].min() >= -1e-12
    # the slopes measured on the drawn curves are the ones the formula was given
    # (a quadratic through the last points differentiates the drawn curve exactly at its end)
    measured_minus = np.polyval(np.polyder(np.polyfit(left[-5:, 0], left[-5:, 1], 2)), p["r_s"])
    measured_plus = np.polyval(np.polyder(np.polyfit(right[:5, 0], right[:5, 1], 2)), p["r_s"])
    assert measured_minus == pytest.approx(p["dpsi_dr_minus"], abs=1e-6)
    assert measured_plus == pytest.approx(p["dpsi_dr_plus"], abs=1e-6)
    value = delta_prime_from_outer_derivatives(p["psi_s"], measured_minus, measured_plus)
    assert np.sign(round(value, 6)) == expected == np.sign(p["delta_prime"])
    # the dashed tangents are those slopes through psi(r_s)
    for role, key in (("slope_left", "dpsi_dr_minus"), ("slope_right", "dpsi_dr_plus")):
        (a, b) = d.model.curves[role]
        assert (b[1] - a[1]) / (b[0] - a[0]) == pytest.approx(p[key])
        assert np.interp(p["r_s"], [a[0], b[0]], [a[1], b[1]]) == pytest.approx(p["psi_s"])


def test_zero_removes_the_jump():
    p = vaft.diagram.delta_prime("zero").model.parameters
    assert p["dpsi_dr_minus"] == p["dpsi_dr_plus"] and p["delta_prime"] == 0.0


@pytest.mark.parametrize("sign, text", [("positive", ">"), ("zero", "="), ("negative", "<")])
def test_the_sign_label_carries_the_delta_prime_role(sign, text):
    (label,) = vaft.diagram.delta_prime(sign).scene.role("delta_prime")
    assert f"\\Delta' {text} 0" in label.text


@pytest.mark.parametrize("sign", ["positive", "negative"])
def test_the_tangent_labels_sit_at_the_upper_end_clear_of_the_curve(sign):
    d = vaft.diagram.delta_prime(sign)
    for role in ("slope_left", "slope_right"):
        (label,) = [it for it in d.scene.role(role) if hasattr(it, "text")]
        top = d.model.to_cm(d.model.curves[role][np.argmax(d.model.curves[role][:, 1])])
        assert abs(label.at[0] - top[0]) < 0.1 and label.at[1] == pytest.approx(top[1])


def test_the_figure_shows_the_formulas_own_definition():
    from vaft.diagram._equations import formula_equation

    d = vaft.diagram.delta_prime()
    (box,) = d.scene.role("equations")
    assert formula_equation(delta_prime_from_outer_derivatives) in box.text
    assert not vaft.diagram.delta_prime(labels=False).scene.role("equations")


def test_delta_prime_takes_only_a_sign():
    for bad in ("large", 1.0, "Positive", ["positive"]):
        with pytest.raises(ValueError, match="sign"):
            vaft.diagram.delta_prime(bad)


# --- layer matching ---------------------------------------------------------------------------


def test_the_layer_is_centred_on_r_s_and_the_outer_regions_lie_outside_it():
    d = vaft.diagram.tearing_layer_matching()
    chart = d.model
    lo, hi = chart.parameters["layer_edges"]
    r_s = chart.parameters["r_s"]
    assert (lo + hi) / 2 == pytest.approx(r_s) and hi - lo == pytest.approx(2 * _tearing._LAYER_HALF_WIDTH)
    assert chart.curves["outer_left"][:, 0].max() == pytest.approx(lo)
    assert chart.curves["outer_right"][:, 0].min() == pytest.approx(hi)
    band = chart.curves["inner_layer"]
    assert band[:, 0].min() == pytest.approx(lo) and band[:, 0].max() == pytest.approx(hi)
    (outline,) = _curves(d, "inner_layer")
    assert np.allclose(outline, chart.to_cm(band))


def test_the_layer_solution_matches_value_and_slope_at_both_edges():
    chart = vaft.diagram.tearing_layer_matching().model
    left, right, inner = chart.curves["outer_left"], chart.curves["outer_right"], chart.curves["inner_solution"]
    assert inner[0] == pytest.approx(left[-1]) and inner[-1] == pytest.approx(right[0])

    # the drawn curves are exact polynomials: fit them and differentiate at the edges
    lo, hi = chart.parameters["layer_edges"]
    d_inner = np.polyder(np.polyfit(inner[:, 0], inner[:, 1], 3))
    d_left = np.polyder(np.polyfit(left[-5:, 0], left[-5:, 1], 2))
    d_right = np.polyder(np.polyfit(right[:5, 0], right[:5, 1], 2))
    assert np.polyval(d_inner, lo) == pytest.approx(np.polyval(d_left, lo), abs=1e-6)
    assert np.polyval(d_inner, hi) == pytest.approx(np.polyval(d_right, hi), abs=1e-6)


def test_the_matching_arrows_run_from_the_outer_regions_to_the_layer_edges():
    d = vaft.diagram.tearing_layer_matching()
    lo, hi = d.model.parameters["layer_edges"]
    (left,) = d.scene.role("matching_left")
    (right,) = d.scene.role("matching_right")
    x_lo = float(d.model.to_cm(np.array([lo, 0.0]))[0])
    x_hi = float(d.model.to_cm(np.array([hi, 0.0]))[0])
    assert left.end[0] == pytest.approx(x_lo) and left.start[0] < x_lo
    assert right.end[0] == pytest.approx(x_hi) and right.start[0] > x_hi
    assert 0.0 < left.end[1] < CHART_HEIGHT


# --- common ------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_every_tearing_diagram_is_deterministic_exported_and_labelled_schematic(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    (note,) = fn().scene.role("note")
    assert "chematic" in note.text
    assert not fn(labels=False).scene.role("note")
    with pytest.raises(ValueError, match="labels"):
        fn(labels="yes")
