"""Stability and operational-space charts: their boundaries are the formulas'.

Everything here runs on the chart model (data coordinates) and needs no TeX.
"""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _stability_space as ss
from vaft.formula.equilibrium import q_cyl_from_B_R_epsilon_kappa_I
from vaft.formula.stability import (
    ballooning_stability_criterion,
    beta_N_from_beta_a_B0_Ip,
    greenwald_density,
    s_alpha_ballooning_stable,
    s_alpha_marginal_alpha,
)

# --- s-alpha formula -------------------------------------------------------------


def test_s_alpha_is_stable_without_pressure_gradient():
    assert s_alpha_ballooning_stable(np.linspace(0.1, 2.0, 7), 0.0).all()


def test_s_alpha_has_a_single_unstable_band_between_first_and_second_stability():
    s = np.array([0.3, 1.0, 1.5])
    first, second = s_alpha_marginal_alpha(s)
    assert np.all(second > first + 0.5)
    for si, a1, a2 in zip(s, first, second):
        assert s_alpha_ballooning_stable(si, 0.9 * a1)
        assert not s_alpha_ballooning_stable(si, 0.5 * (a1 + a2))
        assert s_alpha_ballooning_stable(si, 1.1 * a2)


def test_the_first_boundary_follows_the_linear_approximation_at_moderate_shear():
    s = np.array([0.8, 1.0, 1.2, 1.5])
    first, _ = s_alpha_marginal_alpha(s)
    _, alpha_crit = ballooning_stability_criterion(first, s)
    assert np.allclose(first, alpha_crit, rtol=0.08)


def test_the_boundaries_match_the_classic_s_alpha_diagram():
    # Regression values of this solver; consistent with, but tighter than
    # anything readable off, Connor, Hastie & Taylor (1978) Fig. 1
    # (alpha_1 ~ 0.6 and second stability near alpha ~ 2.6 at unit shear).
    first, second = s_alpha_marginal_alpha(1.0)
    assert first == pytest.approx(0.61, abs=0.02)
    assert second == pytest.approx(2.60, abs=0.05)


def test_the_boundaries_are_converged_in_step_and_range():
    s = np.array([0.3, 1.0])
    first = s_alpha_marginal_alpha(s)[0]
    for alpha in (first * 0.97, first * 1.03):
        ref = s_alpha_ballooning_stable(s, alpha)
        assert np.array_equal(s_alpha_ballooning_stable(s, alpha, step=0.01), ref)
        assert np.array_equal(s_alpha_ballooning_stable(s, alpha, theta_max=80 * np.pi), ref)


def test_no_boundary_without_shear():
    assert all(np.isnan(v) for v in s_alpha_marginal_alpha(-0.5))


@pytest.mark.parametrize("kw", [{"s": np.nan}, {"s": 1.0, "alpha_max": 0.01}, {"s": 1.0, "resolution": 0.0}])
def test_s_alpha_marginal_rejects_bad_input(kw):
    with pytest.raises(ValueError):
        s_alpha_marginal_alpha(**kw)


@pytest.mark.parametrize("kw", [{"theta_max": 0.0}, {"step": -1.0}])
def test_s_alpha_rejects_a_bad_integration(kw):
    with pytest.raises(ValueError):
        s_alpha_ballooning_stable(1.0, 0.5, **kw)


# --- s-alpha chart ------------------------------------------------------------------


def test_the_s_alpha_chart_draws_the_formula_boundaries():
    chart = vaft.diagram.s_alpha_ballooning().model
    rows = chart.curves["first"][::6, 1]
    first, second = s_alpha_marginal_alpha(rows)
    assert np.allclose(chart.curves["first"][::6, 0], first, atol=2e-3)
    ok = np.isfinite(second)
    got = dict(zip(np.round(chart.curves["second"][:, 1], 12), chart.curves["second"][:, 0]))
    assert np.allclose([got[round(r, 12)] for r in rows[ok]], second[ok], atol=2e-3)


@pytest.mark.parametrize("s_max, alpha_max", [
    (1.5, 3.5), (0.2, 6.0), (0.5, 6.0), (1.0, 6.0), (1.5, 1.0), (2.5, 1.0), (2.5, 2.0), (0.2, 2.0),
])
def test_the_s_alpha_region_labels_sit_in_their_regions(s_max, alpha_max):
    chart = vaft.diagram.s_alpha_ballooning(s_max=s_max, alpha_max=alpha_max).model
    labels = chart.labels
    assert "first" in labels
    for name, (a, s) in labels.items():
        assert 0 < a < alpha_max and 0 < s < s_max, name
        first, second = s_alpha_marginal_alpha(s)
        if name == "first":
            assert s_alpha_ballooning_stable(s, a) and a < first
        elif name == "unstable":
            assert not s_alpha_ballooning_stable(s, a)
        else:
            assert s_alpha_ballooning_stable(s, a) and a > second


# --- Hugill and Troyon -------------------------------------------------------------


@pytest.mark.parametrize("elongation", [1.0, 1.7, 5.0])
def test_the_hugill_line_is_the_greenwald_density(elongation):
    chart = vaft.diagram.hugill(elongation=elongation).model
    line = chart.curves["greenwald"][1:]
    A = ss._HUGILL_ASPECT_RATIO
    a = ss._R0 / A
    # recover the current from 1/q through the same formula, then check n = n_G(I)
    I_MA = np.linspace(1e-4, 20.0, 400001)
    inv_q = 1.0 / q_cyl_from_B_R_epsilon_kappa_I(ss._B0, ss._R0, 1 / A, elongation, I_MA * 1e6)
    I_at = np.interp(line[:, 1], inv_q, I_MA)
    assert np.allclose(line[:, 0], greenwald_density(I_at, a) * ss._R0 / ss._B0, rtol=1e-3)
    # the slope depends on the elongation alone: M / (1/q) = 50 kappa / pi
    assert np.allclose(line[:, 0] / line[:, 1], 50 * elongation / np.pi, rtol=1e-6)


def test_the_hugill_regions_are_on_the_right_sides():
    chart = vaft.diagram.hugill().model
    slope = chart.parameters["murakami_at_q_limit"] * chart.parameters["q_limit"]  # M per unit 1/q
    x, y = chart.labels["accessible"]
    assert y < 1 / chart.parameters["q_limit"] and x < slope * y
    x, y = chart.labels["density"]
    assert x > slope * y
    x, y = chart.labels["low_q"]
    assert y > 1 / chart.parameters["q_limit"]


def test_the_troyon_line_has_the_limiting_beta_n_everywhere():
    chart = vaft.diagram.troyon(beta_N_max=3.5, aspect_ratio=2.5).model
    x, beta = chart.curves["beta_limit"][1:].T
    a = ss._R0 / 2.5
    assert np.allclose(beta_N_from_beta_a_B0_Ip(beta, a, ss._B0, x * a * ss._B0), 3.5)


def test_the_troyon_low_q_cutoff_is_where_q_cyl_reaches_the_limit():
    chart = vaft.diagram.troyon(q_limit=2.5, aspect_ratio=3.0, elongation=1.7).model
    x_q = chart.curves["low_q"][0, 0]
    a = ss._R0 / 3.0
    q = q_cyl_from_B_R_epsilon_kappa_I(ss._B0, ss._R0, 1 / 3.0, 1.7, x_q * a * ss._B0 * 1e6)
    assert q == pytest.approx(2.5, rel=1e-4)


def test_the_troyon_regions_are_on_the_right_sides():
    chart = vaft.diagram.troyon().model
    beta_N_max, x_q = chart.parameters["beta_N_max"], chart.parameters["current_at_q_limit"]
    x, y = chart.labels["stable"]
    assert y < beta_N_max * x and x < x_q
    x, y = chart.labels["beta"]
    assert y > beta_N_max * x
    x, y = chart.labels["low_q"]
    assert x > x_q


@pytest.mark.parametrize("fn, kw", [
    (vaft.diagram.hugill, {"elongation": 0.0}),
    (vaft.diagram.hugill, {"q_limit": 0.0}),
    (vaft.diagram.troyon, {"beta_N_max": -1.0}),
    (vaft.diagram.troyon, {"elongation": 0.0}),
    (vaft.diagram.s_alpha_ballooning, {"s_max": 5.0}),
])
def test_invalid_chart_parameters_fail_explicitly(fn, kw):
    with pytest.raises(ValueError):
        fn(**kw)


# --- peeling-ballooning (schematic) -------------------------------------------------


def test_the_peeling_ballooning_corner_is_computed_on_the_boundary():
    model = ss.PeelingBallooningModel()
    alpha, J = model.corner()
    assert model.margin(alpha, J) == pytest.approx(0.0, abs=1e-12)
    assert model.peeling(alpha, J) == pytest.approx(model.ballooning(alpha, J))
    chart = vaft.diagram.peeling_ballooning().model
    assert chart.points["corner"] == pytest.approx((alpha, J))
    # and the traced boundary passes through it
    assert np.min(np.hypot(*(chart.curves["boundary"] - [alpha, J]).T)) < 5e-3


def test_the_peeling_ballooning_boundary_is_the_zero_margin_and_meets_both_axes():
    model = ss.PeelingBallooningModel()
    boundary = vaft.diagram.peeling_ballooning().model.curves["boundary"]
    assert np.allclose(model.margin(boundary[:, 0], boundary[:, 1]), 0.0, atol=1e-9)
    assert boundary[0, 1] == pytest.approx(0.0, abs=1e-12)  # on the alpha axis
    assert boundary[-1, 0] == pytest.approx(0.0, abs=1e-12)  # on the J axis
    # far from the corner each branch is its own linear limit
    assert boundary[-1, 1] == pytest.approx(model.j0, abs=0.01)
    assert boundary[0, 0] == pytest.approx(model.a0, abs=0.01)


def test_the_peeling_ballooning_labels_are_in_their_regions_and_it_says_schematic():
    d = vaft.diagram.peeling_ballooning()
    model, chart = ss.PeelingBallooningModel(), d.model
    assert model.margin(*chart.labels["stable"]) < 0
    a, j = chart.labels["peeling"]
    assert model.peeling(a, j) > 0 and model.peeling(a, j) > model.ballooning(a, j)
    a, j = chart.labels["ballooning"]
    assert model.ballooning(a, j) > 0 and model.ballooning(a, j) > model.peeling(a, j)
    assert "schematic" in d.tikz


@pytest.mark.parametrize("name", ["peeling_ballooning", "s_alpha_ballooning", "hugill", "troyon"])
def test_every_chart_is_deterministic_and_exposed_lazily(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__


@pytest.mark.parametrize("elongation, q_limit", [(1.0, 2.0), (1.8, 2.0), (5.0, 2.0), (1.0, 0.5)])
def test_the_hugill_limits_follow_the_closed_form_for_any_shape(elongation, q_limit):
    chart = vaft.diagram.hugill(elongation=elongation, q_limit=q_limit).model
    x_q = 50 * elongation / (np.pi * q_limit)
    assert chart.parameters["murakami_at_q_limit"] == pytest.approx(x_q, rel=1e-9)
    # the Greenwald line reaches the top of the chart, past the low-q line
    assert chart.curves["greenwald"][-1, 1] == pytest.approx(chart.y_range[1], rel=1e-9)
    x, y = chart.labels["accessible"]
    assert y < 1 / q_limit and x < x_q * y * q_limit
    x, y = chart.labels["density"]
    assert x > x_q * y * q_limit


@pytest.mark.parametrize("aspect_ratio, elongation, q_limit", [(3.0, 1.7, 2.0), (1.3, 1.8, 2.0), (1.3, 2.5, 2.0),
                                                                (1.2, 2.0, 1.0)])
def test_the_troyon_cutoff_follows_the_closed_form_for_any_shape(aspect_ratio, elongation, q_limit):
    chart = vaft.diagram.troyon(aspect_ratio=aspect_ratio, elongation=elongation, q_limit=q_limit).model
    x_q = 5 * elongation / (aspect_ratio * q_limit)  # 5 eps kappa / q
    assert chart.parameters["current_at_q_limit"] == pytest.approx(x_q, rel=1e-9)
    assert chart.curves["low_q"][0, 0] == pytest.approx(x_q, rel=1e-9)


def test_charts_always_get_readable_ticks():
    for kw in ({"q_limit": 10.0}, {"q_limit": 0.5}, {"beta_N_max": 0.5}):
        scene = vaft.diagram.troyon(**kw).scene
        ticks = [item for item in scene.role("ticks") if hasattr(item, "text")]
        assert len(ticks) >= 6  # at least three per axis
