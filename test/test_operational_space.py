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
        fixed = {"area_elongation": 1.6, "elongation": 1.6, "minor_radius": 0.25, "major_radius": 0.4, "triangularity": 0.3,
                 "toroidal_field": 0.17, "plasma_surface_area": 4.0, "plasma_current": 0.1, "aspect_ratio": 1.6,
                 "effective_charge": 2.0}
        plan = ops.overlay_plan(proj, x_range=(0.0, 5.0), y_range=(0.0, 1.0), fixed=fixed)
        assert plan.keys == proj.default_boundaries, (key, plan.omitted)
        for curve in plan.curves:
            if key in ("troyon", "qstar_in"):  # transform or constant line: drawn on the projection's own quantities
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


def test_a_category_keeps_its_colour_across_figures():
    t = _hugill_table()
    order = ["stable", "marginal", "unresolved"]
    t["verdict"] = pd.Categorical(["stable"] * (len(t) - 1) + ["unresolved"], categories=order)
    _, ax1 = operational_space_population(t, "hugill", color="verdict")
    t["verdict"] = pd.Categorical(["marginal"] + ["unresolved"] * (len(t) - 1), categories=order)
    _, ax2 = operational_space_population(t, "hugill", color="verdict")

    def legend_colour(ax, name):
        handle = next(h for h, label in zip(*ax.get_legend_handles_labels()) if label == f"verdict={name}")
        return tuple(handle.get_facecolor()[0])

    assert legend_colour(ax1, "unresolved") == legend_colour(ax2, "unresolved")


# --- kink limits (Freidberg 2008) and L-H access ---------------------------------------------------


def test_freidberg_kink_limits_reproduce_the_book_and_each_other():
    cur = B.get_boundary("freidberg_2008_kink_current")
    qmin = B.get_boundary("freidberg_2008_kink_qstar")
    kappa = np.array([1.0, 1.5, 2.0])
    i_max = B.boundary_value(cur, minor_radius=0.3, major_radius=0.4, toroidal_field=0.18, elongation=kappa)
    # Eq. (13.163): 2 pi a^2 B0/(mu0 R0) = 0.2025 MA at kappa = 1, and the factor 2 kappa/(1 + kappa)
    np.testing.assert_allclose(i_max, 0.2025 * 2 * kappa / (1 + kappa), rtol=1e-3)
    # Eq. (13.162) through Eq. (13.160): q* at I_max is exactly the limit
    np.testing.assert_allclose(B.kink_coordinates(0.3, 0.4, 0.18, kappa, i_max),
                               B.boundary_value(qmin, elongation=kappa))
    assert "13.162" in qmin.sources[0].equation and "13.163" in cur.sources[0].equation


def test_a_boundary_with_every_input_off_axis_is_a_constant_line_only_when_fixed():
    plan = ops.overlay_plan("qstar_in", x_range=(0, 4), y_range=(0, 8), fixed={"elongation": 1.6})
    curve = plan.curves[0]
    assert np.allclose(curve.y, 1.3) and curve.allowed_side == "above" and curve.fixed["elongation"] == 1.6
    plan = ops.overlay_plan("qstar_in", x_range=(0, 4), y_range=(0, 8))
    assert plan.curves == () and "not fixed" in plan.omitted[0][1]


def test_the_lh_plane_draws_martin_and_refuses_ryter_on_unit():
    fixed = {"toroidal_field": 0.2, "plasma_surface_area": 4.0, "plasma_current": 0.1, "minor_radius": 0.25,
             "aspect_ratio": 1.6, "effective_charge": 2.0}
    plan = ops.overlay_plan("lh_threshold", ["martin_2008_lh", "ryter_2014_nmin"], x_range=(0, 0.5),
                            y_range=(0, 2), fixed=fixed)
    assert plan.keys == ("martin_2008_lh",)
    assert "1e19" in plan.omitted[0][1]


# --- cold review of #1478 ------------------------------------------------------------------


def test_kink_coordinates_propagate_nan_and_reject_bad_finite_values():
    q = B.kink_coordinates(np.array([0.25, np.nan]), 0.4, 0.17, np.array([1.5, 1.5]), 0.1)
    assert np.isfinite(q[0]) and np.isnan(q[1])
    assert B.kink_coordinates(0.25, 0.4, -0.17, 1.5, 0.1) == pytest.approx(B.kink_coordinates(0.25, 0.4, 0.17, 1.5, 0.1))
    with pytest.raises(ValueError):
        B.kink_coordinates(0.0, 0.4, 0.17, 1.5, 0.1)
    with pytest.raises(ValueError):
        B.kink_coordinates(np.inf, 0.4, 0.17, 1.5, 0.1)


def _legend_colour(ax, label):
    handle = next(h for h, text in zip(*ax.get_legend_handles_labels()) if text == label)
    return tuple(np.round(handle.get_facecolor()[0][:3], 3))


def test_missing_categories_are_grey_and_plain_strings_keep_slots_across_subsets():
    from matplotlib.colors import to_rgb
    from vaft.plot.operational_space import MISSING_COLOR

    t = _hugill_table()
    t["verdict"] = ["stable", "not available", "unstable"] * 10
    _, ax = operational_space_population(t, "hugill", color="verdict")
    assert _legend_colour(ax, "verdict=not available") == tuple(np.round(to_rgb(MISSING_COLOR), 3))
    # rows of one category that cannot be plotted do not shift the other categories' colours
    sub = t.copy()
    sub.loc[sub["verdict"] == "stable", "murakami_parameter"] = np.nan
    _, ax2 = operational_space_population(sub, "hugill", color="verdict")
    assert _legend_colour(ax, "verdict=unstable") == _legend_colour(ax2, "verdict=unstable")


def test_placement_says_where_a_boundary_lands_or_why_not():
    hugill = ops.placement("hugill", "murakami_hugill")
    assert hugill.drawable and hugill.kind == "vertical"
    troyon = ops.placement("troyon", "troyon")
    assert troyon.drawable and troyon.kind == "ratio"
    beta_li = ops.placement("beta_n_li", "troyon")
    assert beta_li.drawable and beta_li.kind == "horizontal"
    wesson = ops.placement("li_qa_wesson", "wesson_1989_jet_li_qpsi_lower")
    assert wesson.drawable and wesson.kind == "curve" and not wesson.swap_axes
    refused = ops.placement("q95_li", "wesson_1989_jet_li_qpsi_lower")
    assert not refused.drawable and refused.reason
    with pytest.raises(ops.IncompatibleBoundary):
        ops.placement("q95_li", "wesson_1989_jet_li_qpsi_lower", strict=True)


def _legend_marker(ax, label):
    handle = next(h for h, text in zip(*ax.get_legend_handles_labels()) if text == label)
    return handle.get_paths()[0].vertices.shape


def test_marker_shapes_keep_their_group_across_subsets():
    t = _hugill_table()
    t["efit_quality"] = ["good", "admissible"] * 15
    _, ax = operational_space_population(t, "hugill", marker="efit_quality", color="efit_quality")
    sub = t.copy()
    sub.loc[sub.index[0], "murakami_parameter"] = np.nan   # the first row (a 'good' one) cannot be plotted
    sub.loc[sub["efit_quality"] == "good", "murakami_parameter"] = np.nan
    sub.loc[sub.index[2], "murakami_parameter"] = 1.0      # one 'good' row plotted after 'admissible' rows
    _, ax2 = operational_space_population(sub, "hugill", marker="efit_quality", color="efit_quality")
    for label in ("efit_quality=good", "efit_quality=admissible"):
        assert _legend_marker(ax, label) == _legend_marker(ax2, label)


# --- inline boundary style, trajectories, category colours (#1456) -------------------------------------


def test_inline_style_names_the_limit_its_side_and_its_basis():
    t = _hugill_table()
    _, ax = operational_space_population(t, "hugill", boundary_style="inline", boundaries=["greenwald_hugill"])
    labels = ax.get_legend_handles_labels()[1] + [p.get_label() for p in ax.get_legend().get_patches()]
    legend = " ".join(text.get_text() for text in ax.get_legend().get_texts())
    assert "Greenwald/Hugill limit" in legend and "Unstable (Derived)" in legend.replace("\n  ", " ")
    texts = [t_.get_text().strip() for t_ in ax.texts]
    assert "Greenwald/Hugill limit" in texts and texts.count("Stable") == 1
    assert labels   # the legend holds the boundary patches


def test_inline_label_follows_the_line_in_display_coordinates():
    t = _hugill_table()
    fig, ax = operational_space_population(t, "hugill", boundary_style="inline", boundaries=["greenwald_hugill"])
    fig.canvas.draw()
    curve = ax.vaft_overlay.curves[0]
    label = next(t_ for t_ in ax.texts if t_.get_text().strip() == "Greenwald/Hugill limit")
    p = ax.transData.transform(np.c_[curve.x[[0, -1]], curve.y[[0, -1]]])
    expected = np.degrees(np.arctan2(*(p[1] - p[0])[::-1]))
    assert label.get_rotation() == pytest.approx(expected % 360, abs=1.0)


def test_regime_thresholds_name_their_regimes_not_instability():
    t = pd.DataFrame({"line_average_density": [0.05, 0.1, 0.2], "loss_power": [0.2, 0.5, 0.8],
                      "toroidal_field": 0.17, "plasma_surface_area": 4.7})
    t.attrs["units"] = {"line_average_density": "1e20 m^-3", "loss_power": "MW", "toroidal_field": "T",
                        "plasma_surface_area": "m^2"}
    _, ax = operational_space_population(t, "lh_threshold", boundary_style="inline", boundaries=["martin_2008_lh"])
    legend = " ".join(text.get_text() for text in ax.get_legend().get_texts())
    assert "L-mode" in legend and "Unstable" not in legend
    assert "H-mode accessible" in [t_.get_text() for t_ in ax.texts]


def test_shade_style_is_unchanged_by_default():
    _, ax = operational_space_population(_hugill_table(), "hugill")
    assert not [t_ for t_ in ax.texts if t_.get_text() == "Stable"]
    assert "greenwald_hugill" in " ".join(ax.get_legend_handles_labels()[1])


def test_trajectories_run_in_time_order_with_one_marker_size():
    t = _hugill_table()
    t["time_efit_s"] = np.linspace(0.30, 0.33, len(t))
    traj = t.iloc[[5, 1, 3]]          # given out of order
    _, ax = operational_space_population(t, "hugill", trajectories={"Shot A": traj})
    path = next(c for c in ax.collections if c.get_label() == "Shot A")
    order = traj.sort_values("time_efit_s")
    np.testing.assert_allclose(path.get_offsets()[:, 0], order["murakami_parameter"])
    sizes = path.get_sizes()
    assert np.ptp(sizes) == 0 and sizes[0] > 0


def test_category_colors_replace_palette_slots_but_not_missing_grey():
    from matplotlib.colors import to_rgb
    from vaft.plot.operational_space import MISSING_COLOR

    t = _hugill_table()
    t["verdict"] = ["Stable", "unknown", "Unstable"] * 10
    _, ax = operational_space_population(t, "hugill", color="verdict", legend_keys=False,
                                         category_colors={"Stable": "#8fd18f", "unknown": "#ff0000"})
    assert _legend_colour(ax, "Stable") == tuple(np.round(to_rgb("#8fd18f"), 3))
    assert _legend_colour(ax, "unknown") == tuple(np.round(to_rgb(MISSING_COLOR), 3))


def test_a_widened_threshold_is_shaded_on_its_forbidden_side():
    """f_G = 1 above every state: the hatch is above the line, not between it and the old axis top."""
    t = pd.DataFrame({"loss_power": [0.1, 0.5, 1.0], "greenwald_fraction": [0.2, 0.4, 0.6]})
    t.attrs["units"] = {"loss_power": "MW"}
    _, ax = operational_space_population(t, "greenwald_fraction_power", boundary_style="inline")
    lo, hi = ax.get_ylim()
    assert hi >= 1.0 + 0.11 * (1.0 - lo)   # room above the line for its name
    label = next(t_ for t_ in ax.texts if t_.get_text().strip() == "Greenwald limit")
    assert label.get_position()[1] == pytest.approx(1.0) and label.get_va() == "bottom"
    fills = [c for c in ax.collections if type(c).__name__ in ("PolyCollection", "FillBetweenPolyCollection")]
    ys = np.concatenate([p.vertices[:, 1] for c in fills for p in c.get_paths()])
    assert ys.min() >= 1.0 - 1e-9


def test_format_sizes_the_canvas_and_is_refused_beside_a_callers_axes():
    fig, _ = operational_space_population(_hugill_table(), "hugill", format="double_column", theme="technical")
    assert fig.get_size_inches()[0] == pytest.approx(7.0)
    _, ax0 = plt.subplots()
    with pytest.raises(TypeError):
        operational_space_population(_hugill_table(), "hugill", format="screen", ax=ax0)


# --- edge-q proxies from global shape (#1456) ---------------------------------------------------------


def test_menard_qstar_is_freidbergs_with_the_1_plus_kappa_squared_shape_factor():
    for kappa in (1.2, 1.6, 2.0):
        ratio = B.cylindrical_kink_coordinates(0.3, 0.4, 0.17, kappa, 0.1) / B.kink_coordinates(0.3, 0.4, 0.17, kappa, 0.1)
        assert ratio == pytest.approx((1 + kappa**2) / (2 * kappa))


def test_iter_q95_formula_reproduces_the_iter_design_point():
    # R = 6.2 m, a = 2.0 m, B = 5.3 T, 15 MA, kappa95 = 1.7, delta95 = 0.33: the ITER q95 = 3 design point
    assert B.iter_q95_coordinates(2.0, 6.2, 5.3, 1.7, 0.33, 15.0) == pytest.approx(3.0, abs=0.01)
    with pytest.raises(ValueError):
        B.iter_q95_coordinates(0.5, 0.4, 0.17, 1.5, 0.3, 0.1)   # a >= R0
    assert np.isnan(B.iter_q95_coordinates(np.nan, 0.4, 0.17, 1.5, 0.3, 0.1))


def test_shape_current_limits_are_the_q_limits_written_as_currents():
    shape = dict(minor_radius=0.27, major_radius=0.38, elongation=1.5, triangularity=0.3)
    bt = 0.17
    i_iter = B.boundary_value(B.get_boundary("iter_1991_q95_current"), toroidal_field=bt, **shape)
    assert B.iter_q95_coordinates(0.27, 0.38, bt, 1.5, 0.3, i_iter) == pytest.approx(2.1)
    i_menard = B.boundary_value(B.get_boundary("menard_2004_qstar_current"), toroidal_field=bt,
                                **{k: v for k, v in shape.items() if k != "triangularity"})
    assert B.cylindrical_kink_coordinates(0.27, 0.38, bt, 1.5, i_menard) == pytest.approx(1.0)


def test_ip_bt_carries_every_current_limit_as_a_line_in_the_field():
    fixed = {"minor_radius": 0.27, "major_radius": 0.38, "elongation": 1.5, "triangularity": 0.3}
    for key in ops.get_projection("ip_bt").default_boundaries:
        place = ops.placement("ip_bt", key, fixed)
        assert place.drawable and place.kind == "curve" and place.sweep == "toroidal_field"
    assert ops.placement("q95_li", "iter_1991_q95_min").kind == "vertical"
    # the estimate and the equilibrium q95 are different quantities
    assert not ops.placement("q95_li", "iter_1991_q95_estimate_min").drawable


def test_axis_units_are_typeset_with_powers_of_ten():
    from vaft.plot.operational_space import _label
    q = ops.AXIS_QUANTITIES["murakami_parameter"]
    assert _label(q, q.name).endswith("[10$^{19}$ m$^{-2}$ T$^{-1}$]")


def test_a_saw_tooth_boundary_is_named_along_its_overall_direction():
    t = pd.DataFrame({"edge_safety_factor": np.linspace(4, 15, 30), "internal_inductance_li3": np.linspace(0.4, 0.8, 30)})
    _, ax = operational_space_population(t, "li_qa_wesson", boundary_style="inline", x_range=(0, 18), y_range=(0, 2))
    label = next(t_ for t_ in ax.texts if t_.get_text().strip() == "Kink / double-tearing limit")
    assert abs(label.get_rotation() % 180) < 20 or abs(label.get_rotation() % 180) > 160   # near horizontal
    assert label.get_position()[1] <= 0.35   # below the teeth, on the forbidden side


def test_current_limits_validate_their_geometry():
    for key, extra in (("menard_2004_qstar_current", {}), ("iter_1991_q95_current", {"triangularity": 0.3})):
        with pytest.raises(ValueError):
            B.boundary_value(B.get_boundary(key), minor_radius=-0.3, major_radius=0.4, toroidal_field=0.17,
                             elongation=1.5, **extra)
    with pytest.raises(ValueError):
        B.boundary_value(B.get_boundary("iter_1991_q95_current"), minor_radius=0.45, major_radius=0.4,
                         toroidal_field=0.17, elongation=1.5, triangularity=0.3)


def test_start_scaling_is_akers_f_of_A_on_the_iter_shaping_factor():
    a, R0, B0, kappa, delta, ip = 0.27, 0.38, 0.17, 1.5, 0.3, 0.1
    A = R0 / a
    f_start = 1.17 * np.sqrt(A / (A - 1.0))
    f_iter = (1.17 - 0.65 / A) / (1.0 - 1.0 / A**2) ** 2
    ratio = B.start_q95_coordinates(a, R0, B0, kappa, delta, ip) / B.iter_q95_coordinates(a, R0, B0, kappa, delta, ip)
    assert ratio == pytest.approx(f_start / f_iter)
    dnd = B.start_q95_coordinates(a, R0, B0, kappa, delta, ip, configuration="double_null")
    assert dnd / B.start_q95_coordinates(a, R0, B0, kappa, delta, ip) == pytest.approx(0.77)
    with pytest.raises(ValueError):
        B.start_q95_coordinates(a, R0, B0, kappa, delta, ip, configuration="diverted")
    i_lim = B.boundary_value(B.get_boundary("akers_2000_q95_current"), minor_radius=a, major_radius=R0,
                             toroidal_field=B0, elongation=kappa, triangularity=delta)
    assert B.start_q95_coordinates(a, R0, B0, kappa, delta, i_lim) == pytest.approx(2.1)


def test_a_derived_boundary_says_derived_and_murakami_is_a_historical_reference():
    _, ax = operational_space_population(_hugill_table(), "hugill", boundary_style="inline")
    legend = " ".join(t_.get_text() for t_ in ax.get_legend().get_texts()).replace("\n  ", " ")
    assert "Murakami (historical conventional-tokamak reference) (Derived)" in legend
    along = [t_.get_text().strip() for t_ in ax.texts]
    assert "Murakami (reference)" in along   # the long name stays in the legend, a short one fits the line
    assert "(Derived)" in legend and "(Empirical)" not in legend
    murakami = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
    assert murakami, "the historical reference is a dashed line"
    # one hatched forbidden side (Greenwald), none for the reference
    hatched = [c for c in ax.collections if getattr(c, "get_hatch", lambda: None)()]
    assert len(hatched) == 1
    # the "Stable" label keeps clear of the reference line (Murakami at x = 1)
    stable = next(t_ for t_ in ax.texts if t_.get_text() == "Stable")
    x_stable = ax.transAxes.transform(stable.get_position())[0]
    x_line = ax.transData.transform((1.0, 0.0))[0]
    assert abs(x_stable - x_line) > 0.08 * ax.get_window_extent().width


# --- applicability status (#1628, plotting side) -------------------------------------------------


def _wesson_table(machine_class=None):
    t = pd.DataFrame({"edge_safety_factor": np.linspace(4, 15, 30), "internal_inductance_li3": np.linspace(0.4, 0.8, 30)})
    if machine_class:
        t.attrs["machine_class"] = machine_class
    return t


def test_a_conventional_boundary_is_outside_for_a_spherical_tokamak_population():
    from vaft.plot.operational_space import APPLICABILITY_STATUSES
    _, ax = operational_space_population(_wesson_table("spherical_tokamak"), "li_qa_wesson", boundary_style="inline",
                                         x_range=(0, 18), y_range=(0, 2))
    status, reasons = ax.vaft_applicability["wesson_1989_jet_li_qpsi_lower"]
    assert status == "OUTSIDE" and status in APPLICABILITY_STATUSES
    assert any(r.startswith("$q_\\psi$") and "outside 2-10" in r for r in reasons) and "calibrated on JET" in reasons
    legend = " ".join(t_.get_text() for t_ in ax.get_legend().get_texts()).replace("\n  ", " ")
    assert "[outside calibration:" in legend


def test_without_a_machine_class_a_generic_boundary_is_unassessed():
    _, ax = operational_space_population(_hugill_table(), "hugill", boundary_style="inline")
    status, reasons = ax.vaft_applicability["greenwald_hugill"]
    assert status == "UNASSESSED" and "no machine class" in reasons[0]


def test_a_class_that_covers_spherical_tokamaks_is_not_supported_without_a_tested_range():
    t = pd.DataFrame({"normalized_current": [50.0, 80.0], "kink_safety_factor_cylindrical": [0.3, 0.2]})
    t.attrs["units"] = {"normalized_current": "MA m^-1 T^-1"}
    t.attrs["machine_class"] = "spherical_tokamak"
    _, ax = operational_space_population(t, "qstar_cyl_in", boundary_style="inline")
    status, reasons = ax.vaft_applicability["menard_2004_qstar_min"]
    assert status == "UNASSESSED" and "no calibration range" in reasons[-1]   # Menard declares no range


def test_supported_and_outside_on_a_declared_range(monkeypatch):
    from dataclasses import replace
    from vaft.plot.operational_space import applicability_status
    entry = B.get_boundary("takizuka_2004_lh")   # its class covers spherical tokamaks; give it a test range
    ranged = replace(entry, applicability=replace(entry.applicability, ranges={"line_average_density": (0.05, 0.5)}))
    registry = dict(B._REGISTRY)
    monkeypatch.setattr(B, "get_boundary", lambda key: ranged if key == entry.key else registry[key])
    fixed = {"toroidal_field": 0.17, "plasma_current": 0.1, "minor_radius": 0.27, "aspect_ratio": 1.4,
             "plasma_surface_area": 4.7, "effective_charge": 2.0}
    curve = ops.overlay_plan("lh_threshold", ["takizuka_2004_lh"], x_range=(0.01, 0.6), y_range=(0, 1),
                             fixed=fixed).curves[0]
    axes = ("line_average_density", "loss_power")
    inside = pd.DataFrame({"line_average_density": [0.1, 0.2]})
    beyond = pd.DataFrame({"line_average_density": [0.1, 0.02]})
    assert applicability_status(curve, inside, axes, "spherical_tokamak")[0] == "SUPPORTED"
    status, reasons = applicability_status(curve, beyond, axes, "spherical_tokamak")
    assert status == "OUTSIDE" and "1/2 outside 0.05-0.5" in reasons[0]
    assert applicability_status(curve, inside.iloc[:0], axes, "spherical_tokamak")[0] == "UNASSESSED"   # no data
    assert applicability_status(curve, inside, axes, None)[0] == "UNASSESSED"   # no machine class


def test_jet_and_st_only_classes_exclude_each_others_populations():
    from vaft.plot.operational_space import _machine_coverage
    jet = "JET (conventional aspect ratio, R = 3 m), 1985-88 operation"
    assert _machine_coverage(jet, "spherical_tokamak") is False
    assert _machine_coverage(jet, "ST") is False
    assert _machine_coverage(jet, "conventional_tokamak") is None   # a descriptive phrase is not coverage
    assert _machine_coverage("spherical tokamak (START equilibria)", "conventional_tokamak") is False
    assert _machine_coverage("tokamak including spherical tokamaks", "conventional_tokamak") is None


def test_missing_inputs_are_unassessed_and_a_quantity_mismatch_is_not_applicable():
    t = pd.DataFrame({"murakami_parameter": [1.0, 2.0], "inverse_cylindrical_q": [0.2, 0.3]})   # unit undeclared
    _, ax = operational_space_population(t, "hugill")
    assert {s for s, _ in ax.vaft_applicability.values()} == {"UNASSESSED"}


def test_an_omitted_boundary_is_not_applicable():
    t = pd.DataFrame({"edge_safety_factor_95": [5.0, 6.0], "internal_inductance_li3": [0.5, 0.6]})
    _, ax = operational_space_population(t, "li_qa_wesson", x="edge_safety_factor_95")
    assert {s for s, _ in ax.vaft_applicability.values()} == {"NOT_APPLICABLE"}


# --- spherical-tokamak Hugill diagram (#1602) ----------------------------------------------------


def test_st_hugill_coordinates_put_the_greenwald_and_sykes_densities_on_their_lines():
    a, R, bt, kappa, ip = 0.27, 0.38, 0.17, 1.5, 0.12
    for density, key in ((10 * ip / (np.pi * a * a), "greenwald_hugill_st"),
                         (10 * ip / (np.pi * a * a * kappa), "sykes_2000_st_hugill")):
        x, y = B.hugill_coordinates_st(density, R, bt, a, kappa, ip)
        line = B.boundary_value(B.get_boundary(key), inverse_cylindrical_q_st=y, elongation=kappa)
        assert x == pytest.approx(float(line))
    # the ST q_cyl is Menard's q*, and differs from the conventional Hugill q_cyl by (1 + kappa^2)/(2 kappa_a)
    _, y_st = B.hugill_coordinates_st(1.0, R, bt, a, kappa, ip)
    assert 1.0 / y_st == pytest.approx(B.cylindrical_kink_coordinates(a, R, bt, kappa, ip))
    _, y_conv = B.hugill_coordinates(1.0, R, bt, a, kappa, ip)
    assert y_conv / y_st == pytest.approx((1 + kappa**2) / (2 * kappa))


def test_st_and_conventional_hugill_axes_are_never_substituted():
    assert not ops.placement("hugill", "sykes_2000_st_hugill", {"elongation": 1.5}).drawable
    assert not ops.placement("hugill", "greenwald_hugill_st", {"elongation": 1.5}).drawable
    assert not ops.placement("hugill_st", "greenwald_hugill", {"area_elongation": 1.5}).drawable
    assert ops.placement("hugill_st", "murakami_hugill").kind == "vertical"   # the x axis is shared


def test_sykes_boundary_is_supported_for_a_spherical_tokamak_and_murakami_stays_a_reference():
    x, y = B.hugill_coordinates_st(np.linspace(5, 30, 20), 0.38, 0.17, 0.27, 1.5, np.linspace(0.05, 0.25, 20))
    t = pd.DataFrame({"murakami_parameter": x, "inverse_cylindrical_q_st": y, "elongation": 1.5})
    t.attrs["units"] = {"murakami_parameter": "1e19 m^-2 T^-1"}
    t.attrs["machine_class"] = "spherical_tokamak"
    _, ax = operational_space_population(t, "hugill_st", boundary_style="inline")
    assert ax.vaft_applicability["sykes_2000_st_hugill"][0] == "SUPPORTED"
    legend = " ".join(t_.get_text() for t_ in ax.get_legend().get_texts()).replace("\n  ", " ")
    assert "historical conventional-tokamak reference" in legend and "Hugill limit (Sykes 2000, MAST)" in legend


def test_st_hugill_diagram_curves_are_the_registered_boundaries():
    import vaft.diagram
    chart = vaft.diagram.hugill_st(elongation=1.8).model
    xy = chart.curves["hugill"]
    expected = B.boundary_value(B.get_boundary("sykes_2000_st_hugill"), inverse_cylindrical_q_st=xy[:, 1], elongation=1.8)
    np.testing.assert_allclose(xy[:, 0], expected)
    assert chart.parameters["boundaries"] == ("sykes_2000_st_hugill", "greenwald_hugill_st", "murakami_hugill")
