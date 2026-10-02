"""Operational-space projections, boundary compatibility and the population renderer (#944, #1425)."""

from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vaft.diagram import _op_space as ops  # noqa: E402
from vaft.formula import boundaries as B  # noqa: E402
from vaft.plot.operational_space import operational_space_population, population_overlay  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _hugill_table(n=30, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({
        "shot": rng.integers(39900, 43000, n),
        "murakami_parameter": rng.uniform(0.5, 6.0, n),
        "inverse_cylindrical_q": rng.uniform(0.05, 0.4, n),
        "area_elongation": rng.uniform(1.4, 1.8, n),
        "efit_quality": rng.choice(["good", "admissible"], n),
    })
    t.attrs["units"] = {"murakami_parameter": "1e19 m^-2 T^-1"}
    return t


# --- quantity identity -------------------------------------------------------------


def test_same_quantity_compares_identity_and_unit_only():
    q = B.BoundaryQuantity("line_average_density", "n", "1e19 m^-3")
    assert B.same_quantity(q, B.BoundaryQuantity("line_average_density", "n_e", "1e19 m^-3", "other text"))
    assert not B.same_quantity(q, B.BoundaryQuantity("line_average_density", "n", "1e20 m^-3"))
    assert not B.same_quantity(q, B.BoundaryQuantity("edge_density", "n", "1e19 m^-3"))


def test_threshold_curve_draws_a_registered_threshold_on_either_axis():
    entry = B.get_boundary("murakami_hugill")
    other = ops.AXIS_QUANTITIES["inverse_cylindrical_q"]
    vertical = B.threshold_curve(entry, other, np.linspace(0, 0.5, 5), target_axis="x")
    assert np.all(vertical.x == 1.0) and vertical.allowed_side == "left"
    assert vertical.x_quantity == entry.target and vertical.y_quantity == other
    horizontal = B.threshold_curve(entry, other, np.linspace(0, 0.5, 5), target_axis="y")
    assert np.all(horizontal.y == 1.0) and horizontal.allowed_side == "below"
    with pytest.raises(TypeError):
        B.threshold_curve(B.get_boundary("greenwald_hugill"), other, [0.1])
    with pytest.raises(ValueError):
        B.threshold_curve(entry, entry.target, [0.1])


# --- projections and overlay selection ---------------------------------------------------------


def test_every_projection_default_is_drawable_on_its_own_axes():
    for key in ops.list_projections():
        proj = ops.get_projection(key)
        fixed = {"area_elongation": 1.6}
        plan = ops.overlay_plan(proj, x_range=(0.0, 5.0), y_range=(0.0, 1.0), fixed=fixed)
        assert plan.keys == proj.default_boundaries, (key, plan.omitted)
        for curve in plan.curves:
            if key in ("troyon",):  # declared transform: drawn on the projection's own quantities
                assert B.same_quantity(curve.x_quantity, proj.x) and B.same_quantity(curve.y_quantity, proj.y)
                continue
            assert {curve.x_quantity.name, curve.y_quantity.name} == {proj.x.name, proj.y.name}


@pytest.mark.parametrize("projection", ["hugill", "q95_li", "beta_n_li", "troyon"])
def test_the_q_psi_low_q_limit_is_never_drawn_on_a_q_cyl_or_q95_or_other_axis(projection):
    plan = ops.overlay_plan(projection, ["low_q"], x_range=(0, 1), y_range=(0, 1))
    assert plan.curves == ()
    assert plan.omitted[0][0] == "low_q" and "targets edge_safety_factor [-]" in plan.omitted[0][1]


def test_a_power_law_needs_its_off_axis_inputs():
    plan = ops.overlay_plan("hugill", ["greenwald_hugill"], x_range=(0, 5), y_range=(0, 0.5))
    assert plan.curves == () and "area_elongation" in plan.omitted[0][1]
    plan = ops.overlay_plan("hugill", ["greenwald_hugill"], x_range=(0, 5), y_range=(0, 0.5),
                            fixed={"area_elongation": 1.5})
    curve = plan.curves[0]
    on = curve.y > 0
    np.testing.assert_allclose(curve.x[on] / curve.y[on], 50 * 1.5 / np.pi)


def test_boundaries_false_and_explicit_lists():
    assert ops.overlay_plan("hugill", False, x_range=(0, 1), y_range=(0, 1)).curves == ()
    plan = ops.overlay_plan("hugill", ["murakami_hugill"], x_range=(0, 1), y_range=(0, 1))
    assert plan.keys == ("murakami_hugill",)
    with pytest.raises(KeyError):
        ops.overlay_plan("hugill", ["no_such_boundary"], x_range=(0, 1), y_range=(0, 1))
    with pytest.raises(ValueError):
        ops.overlay_plan("hugill", "all", x_range=(0, 1), y_range=(0, 1))
    assert ops.overlay_plan("hugill", True, x_range=(0, 1), y_range=(0, 1),
                            fixed={"area_elongation": 1.5}).keys == ("greenwald_hugill", "murakami_hugill")
    assert ops.overlay_plan("hugill", ["murakami_hugill"] * 2, x_range=(0, 1), y_range=(0, 1)).keys == (
        "murakami_hugill",)
    with pytest.raises(ValueError):
        population_overlay(_hugill_table(), "hugill", boundaries="all")


def test_the_troyon_transform_is_the_definition_of_beta_n():
    plan = ops.overlay_plan("troyon", x_range=(0.0, 2.0), y_range=(0.0, 6.0))
    curve = plan.curves[0]
    from vaft.formula.stability import beta_N_from_beta_a_B0_Ip
    a, b0 = 0.3, 0.5
    x = curve.x[1:]
    np.testing.assert_allclose(beta_N_from_beta_a_B0_Ip(curve.y[1:], a, b0, x * a * b0),
                               B.boundary_value(B.get_boundary("troyon")))


def test_unknown_projection_lists_the_registered_ones():
    with pytest.raises(KeyError, match="hugill"):
        ops.get_projection("nope")


# --- table overlay ------------------------------------------------------------------------------


def test_table_overlay_takes_off_axis_inputs_from_the_table_median():
    t = _hugill_table()
    plan = population_overlay(t, "hugill")
    assert plan.keys == ("greenwald_hugill", "murakami_hugill")
    assert plan.curves[0].fixed["area_elongation"] == pytest.approx(t["area_elongation"].median())


def test_table_overlay_refuses_a_substituted_column():
    t = _hugill_table().rename(columns={"inverse_cylindrical_q": "q95"})
    plan = population_overlay(t, "hugill", y="q95")
    assert plan.curves == ()
    assert all("not the projection's 'inverse_cylindrical_q'" in reason for _, reason in plan.omitted)


def test_table_overlay_needs_declared_matching_units():
    t = _hugill_table()
    t.attrs["units"] = {}
    plan = population_overlay(t, "hugill")
    assert plan.curves == () and "not declared" in plan.omitted[0][1]
    plan = population_overlay(t, "hugill", units={"murakami_parameter": "1e20 m^-2 T^-1"})
    assert plan.curves == () and "1e20" in plan.omitted[0][1]


# --- renderer -----------------------------------------------------------------------------------


def test_renderer_contract_and_drawn_boundaries():
    t = _hugill_table()
    fig, ax = operational_space_population(t, "hugill", marker="efit_quality", hollow=["admissible"])
    assert ax.figure is fig
    assert ax.vaft_overlay.keys == ("greenwald_hugill", "murakami_hugill")
    labels = [line.get_label() for line in ax.get_lines()]
    assert any(label.startswith("greenwald_hugill") for label in labels)
    # passed axes are authoritative
    fig2, ax2 = plt.subplots()
    out = operational_space_population(t, "hugill", boundaries=False, ax=ax2)
    assert out == (fig2, ax2) and ax2.vaft_overlay.curves == ()


def test_renderer_keeps_data_and_warns_when_the_boundary_does_not_apply():
    t = _hugill_table().rename(columns={"inverse_cylindrical_q": "q95"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, ax = operational_space_population(t, "hugill", y="q95")
    assert ax.vaft_overlay.curves == ()
    assert any("not drawn" in str(w.message) for w in caught)
    assert len(ax.collections) >= 1  # the data are still there


def test_renderer_numeric_colour_and_missing_column():
    t = _hugill_table()
    t["R_p"] = np.linspace(0, 1, len(t))
    fig, ax = operational_space_population(t, "hugill", color="R_p", marker="efit_quality")
    assert len(fig.axes) == 2  # colour bar
    with pytest.raises(KeyError):
        operational_space_population(t, "hugill", color="nope")


def test_the_population_renderer_imports_no_ods_or_database_layer():
    code = ("import sys, vaft.plot.operational_space; "
            "bad = [m for m in ('omas', 'vaft.omas', 'vaft.database', 'h5pyd') if m in sys.modules]; "
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_missing_colour_values_stay_visible_and_bools_are_categories():
    t = _hugill_table()
    t["R_p"] = np.where(np.arange(len(t)) % 2 == 0, np.nan, 1.0)
    fig, ax = operational_space_population(t, "hugill", color="R_p")
    faces = np.concatenate([c.get_facecolors() for c in ax.collections if len(c.get_offsets())])
    assert np.all(faces[:, 3] > 0)
    t["stable"] = np.arange(len(t)) % 2 == 0
    fig, ax = operational_space_population(t, "hugill", color="stable")
    assert len(fig.axes) == 1  # no colour bar for a bool column


def test_extrapolated_boundaries_are_warned_about():
    t = pd.DataFrame({"edge_safety_factor": [1.5, 6.0, 12.0], "internal_inductance_li3": [0.5, 0.6, 0.7]})
    from vaft.formula.boundaries import list_boundaries
    if "wesson_1989_jet_li_qpsi_lower" not in list_boundaries():
        pytest.skip("li-q family lands with #1422")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        operational_space_population(t, "li_qa_wesson")
    assert any("extrapolated" in str(w.message) for w in caught)


def test_a_far_threshold_is_warned_not_silently_off_screen():
    t = pd.DataFrame({"internal_inductance_li3": [0.4, 0.5, 0.6], "normalized_beta": [0.01, 0.02, 0.03]})
    t.attrs["units"] = {"normalized_beta": "% m T/MA"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, ax = operational_space_population(t, "beta_n_li")
    assert ax.get_ylim()[1] < 1.0
    assert any("not in view" in str(w.message) for w in caught)


def test_axis_ranges_and_colour_limits_are_honoured():
    t = _hugill_table()
    t["R_p"] = np.linspace(0.0, 40.0, len(t))
    fig, ax = operational_space_population(t, "hugill", color="R_p", color_limits=(0.0, 6.0),
                                           x_range=(0.0, 10.0), y_range=(0.0, 0.6))
    assert ax.get_xlim() == (0.0, 10.0) and ax.get_ylim() == (0.0, 0.6)
    bar = fig.axes[1]
    assert bar.get_ylim()[1] == pytest.approx(6.0)
