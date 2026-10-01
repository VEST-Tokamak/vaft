"""The l_i-q family (#1422): Wesson 1989 (JET, empirical) and Cheng 1987 (theory) kept apart."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vaft.diagram
from vaft.diagram import _op_space as ops
from vaft.formula import boundaries as B
from vaft.formula.stability import empirical_li_qa

WESSON = ("wesson_1989_jet_li_qpsi_lower", "wesson_1989_jet_li_qpsi_upper")
CHENG = ("cheng_1987_li_qa_lower", "cheng_1987_li_qa_upper")


def _value(key, q):
    entry = B.get_boundary(key)
    return B.boundary_value(entry, **{entry.input_names[0]: np.asarray(q, dtype=float)})


# --- provenance and identity ---------------------------------------------------------------------


@pytest.mark.parametrize("key, figure", [(k, "Fig. 6") for k in WESSON] + [(k, "Fig. 4") for k in CHENG])
def test_every_curve_names_its_source_figure_and_is_a_reproduction(key, figure):
    entry = B.get_boundary(key)
    assert figure in entry.sources[0].equation
    assert entry.origin == "reproduced" and entry.family == "li_q"


def test_the_empirical_and_theoretical_references_use_different_quantities():
    w, c = B.get_boundary(WESSON[0]), B.get_boundary(CHENG[0])
    assert w.basis == "empirical" and c.basis != "empirical"
    assert not B.same_quantity(w.target, c.target)
    assert not B.same_quantity(w.inputs[0], c.inputs[0])
    # Wesson's x is the equilibrium edge q of the registered low_q limit, not q95
    assert B.same_quantity(w.inputs[0], B.get_boundary("low_q").target)
    assert w.inputs[0].name != "edge_safety_factor_95"


@pytest.mark.parametrize("key, side", [(WESSON[0], "above"), (WESSON[1], "below"), (CHENG[0], "above"),
                                       (CHENG[1], "below")])
def test_allowed_sides_put_the_operating_space_between_the_branches(key, side):
    assert B.get_boundary(key).allowed_side == side


def test_branches_do_not_cross_inside_their_ranges():
    q = np.linspace(2.0, 10.0, 801)
    assert np.all(_value(WESSON[1], q) >= _value(WESSON[0], q) - 1e-12)
    q = np.linspace(2.0, 7.75, 801)
    assert np.all(_value(CHENG[1], q) > _value(CHENG[0], q))


def test_outside_the_drawn_range_there_is_no_boundary():
    for key in WESSON + CHENG:
        assert np.isnan(_value(key, [1.5])).all() and np.isnan(_value(key, [10.5])).all()


# --- reproduction ----------------------------------------------------------------------------------


def test_the_legacy_arrays_are_the_wesson_lower_boundary():
    """#1422 audit: empirical_li_qa is the tooth vertices of Fig. 6's lower boundary, not Fig. 5."""
    qa, li = empirical_li_qa()
    tops, bottoms = li[0::2], li[1::2]
    q = qa[0::2].astype(float)
    # a tooth's top is the left limit at integer q, its bottom the value at q
    np.testing.assert_allclose(_value(WESSON[0], q[1:] - 1e-9), tops[1:], atol=0.035)
    np.testing.assert_allclose(_value(WESSON[0], q[:-1]), bottoms[:-1], atol=0.035)
    assert "Fig. 6" in empirical_li_qa.__doc__


def test_the_cheng_upper_bound_stays_under_the_printed_closed_form_maximum():
    """Fig. 4 prints MAX(l_i/2) = [1 + 2 ln(q(a)/q(0))]/4, q(0) = 1.01: the uniform-current-core profile."""
    q = np.linspace(2.0, 7.75, 200)
    l_max = 2.0 * (1.0 + 2.0 * np.log(q / 1.01)) / 4.0
    assert np.all(_value(CHENG[1], q) < l_max)
    assert np.all(l_max - _value(CHENG[1], q) < 0.35)


# --- projections, diagram, plot ---------------------------------------------------------------------


def test_each_projection_draws_only_its_own_reference():
    plan = ops.overlay_plan("li_qa_wesson", WESSON + CHENG, x_range=(1, 11), y_range=(0, 2))
    assert plan.keys == WESSON and {k for k, _ in plan.omitted} == set(CHENG)
    plan = ops.overlay_plan("li_qa_cheng", WESSON + CHENG, x_range=(1, 8), y_range=(0, 2.6))
    assert plan.keys == CHENG and {k for k, _ in plan.omitted} == set(WESSON)
    assert ops.overlay_plan("q95_li", WESSON + CHENG, x_range=(1, 11), y_range=(0, 2)).curves == ()


@pytest.mark.parametrize("reference", ["wesson_1989", "cheng_1987"])
def test_the_diagram_curves_are_the_registered_boundaries(reference):
    chart = vaft.diagram.li_qa(reference=reference).model
    keys = chart.parameters["boundaries"]
    for branch in ("lower", "upper"):
        key = next(k for k in keys if k.endswith(branch))
        xy = chart.curves[branch]
        np.testing.assert_allclose(_value(key, xy[:, 0]), xy[:, 1])
    edge = chart.curves["edge"]
    assert np.allclose(edge[:, 0], 2.0)


def test_unknown_reference_is_refused():
    with pytest.raises(ValueError):
        vaft.diagram.li_qa(reference="li_qa_2020")


def test_a_q95_table_gets_no_wesson_line():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from vaft.plot.operational_space import population_overlay

    t = pd.DataFrame({"edge_safety_factor_95": [5.0, 6.0], "internal_inductance_li3": [0.5, 0.6]})
    plan = population_overlay(t, "li_qa_wesson", x="edge_safety_factor_95")
    assert plan.curves == () and all("edge_safety_factor" in reason for _, reason in plan.omitted)
    t = t.rename(columns={"edge_safety_factor_95": "edge_safety_factor"})
    assert population_overlay(t, "li_qa_wesson").keys == WESSON + ("low_q",)


# A second, independent reading of the figures (cold review of #1473, 600 dpi, tick-calibrated) pins every
# branch, so a digitizing slip in one table cannot hide behind a self-consistent test.
SECOND_READING = {
    "wesson_1989_jet_li_qpsi_upper": {3.0: 1.11, 4.0: 1.26, 6.0: 1.52, 8.0: 1.72, 9.95: 1.86},
    "cheng_1987_li_qa_upper": {3.0: 2 * 0.699, 4.0: 2 * 0.814, 5.0: 2 * 0.914, 6.0: 2 * 1.001, 7.0: 2 * 1.072},
    # bottoms of the Cheng jig-saw at q = 2..6 (value at integer q), in l_i
    "cheng_1987_li_qa_lower": {2.0: 2 * 0.345, 3.0: 2 * 0.355, 4.0: 2 * 0.355, 5.0: 2 * 0.412, 6.0: 2 * 0.442},
}


@pytest.mark.parametrize("key", sorted(SECOND_READING))
def test_every_branch_matches_an_independent_reading_of_its_figure(key):
    points = SECOND_READING[key]
    np.testing.assert_allclose(_value(key, list(points)), list(points.values()), atol=0.025)


def test_the_last_wesson_tooth_drops_like_the_others():
    assert _value(WESSON[0], [10.0])[0] == pytest.approx(0.295)
    assert _value(WESSON[0], [10.0 - 1e-9])[0] == pytest.approx(0.678, abs=1e-6)
