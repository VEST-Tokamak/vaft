"""Several canonical plots composed into one figure (issue #1467)."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import vaft
from vaft.plot import AxisLink, FigureCell, FigureComposition
from vaft.plot.composition import build_composition, render_composition
from vaft.plot.models import LineSeries, Panels

STACK = ("plasma_current_time", "flux_loop_time_voltage", "equilibrium_time_q95")


@pytest.fixture(scope="module")
def ods():
    return vaft.omas.sample_ods(39915)


@pytest.fixture(scope="module")
def entries(ods):
    return vaft.omas.normalize_entries(ods, label="shot")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _equilibrium(**kwargs):
    return FigureComposition(
        shape=(2, 2),
        cells=(
            FigureCell("equilibrium_field_psi", 0, 0, rowspan=2),
            FigureCell("equilibrium_profile_pressure", 0, 1),
            FigureCell("equilibrium_profile_q", 1, 1),
        ),
        **kwargs,
    )


# --- the composition model ------------------------------------------------------


def test_cells_must_fit_and_not_overlap():
    with pytest.raises(ValueError, match="does not fit the 2x2 grid"):
        FigureComposition((2, 2), (FigureCell("plasma_current_time", 1, 1, colspan=2),))
    with pytest.raises(ValueError, match="overlap at grid cell"):
        FigureComposition((2, 2), (
            FigureCell("plasma_current_time", 0, 0, rowspan=2),
            FigureCell("equilibrium_time_q95", 1, 0),
        ))
    with pytest.raises(ValueError, match="distinct names"):
        FigureComposition((2, 1), (FigureCell("plasma_current_time", 0, 0), FigureCell("plasma_current_time", 1, 0)))
    with pytest.raises(ValueError, match="at least one row"):
        FigureComposition((0, 1), (FigureCell("plasma_current_time"),))
    with pytest.raises(ValueError, match="rowspan and colspan"):
        FigureCell("plasma_current_time", rowspan=0)
    # a plot used twice with distinct names is fine; an empty grid cell too
    composition = FigureComposition((2, 2), (
        FigureCell("plasma_current_time", 0, 0, name="ip_full"),
        FigureCell("plasma_current_time", 1, 1, name="ip_window", options={"time_range": (0.30, 0.33)}),
    ))
    assert [cell.name for cell in composition.cells] == ["ip_full", "ip_window"]


def test_links_name_existing_cells_and_groups_merge():
    with pytest.raises(ValueError, match="names no cell 'nope'"):
        FigureComposition.stack(STACK, links=(AxisLink("x", ("plasma_current_time", "nope")),))
    with pytest.raises(ValueError, match="at least two cells"):
        AxisLink("x", ("a", "a"))
    with pytest.raises(ValueError, match="axis must be one of"):
        AxisLink("z", ("a", "b"))
    assert FigureComposition.stack(STACK).resolved_links() == (("x", (0, 1, 2)),)
    assert FigureComposition.stack(STACK, share_x=False).resolved_links() == ()
    # share_x groups by column; an explicit link touching one joins it
    grid = FigureComposition.grid(["plasma_current_time", "equilibrium_time_q95", "flux_loop_time_voltage", None],
                                  (2, 2), share_x=True)
    assert grid.resolved_links() == (("x", (0, 2)),)
    joined = FigureComposition.grid(
        ["plasma_current_time", "equilibrium_time_q95", "flux_loop_time_voltage"], (2, 2), share_x=True,
        links=(AxisLink("x", ("flux_loop_time_voltage", "equilibrium_time_q95")),),
    )
    assert joined.resolved_links() == (("x", (0, 1, 2)),)


def test_shapes_and_their_reproducible_form():
    stack = FigureComposition.stack(STACK, panel_labels=True, title="")
    assert stack.shape == (3, 1) and [c.region for c in stack.cells] == [(0, 0, 1, 1), (1, 0, 1, 1), (2, 0, 1, 1)]
    side = FigureComposition.side_by_side(["equilibrium_profile_q", "equilibrium_profile_pressure"])
    assert side.shape == (1, 2) and not side.share_x
    grid = FigureComposition.grid(["plasma_current_time", None, "equilibrium_time_q95"], (2, 2))
    assert [c.region[:2] for c in grid.cells] == [(0, 0), (1, 0)]
    composition = _equilibrium(panel_labels=True, links=(AxisLink("x", ("equilibrium_profile_pressure", "equilibrium_profile_q")),))
    data = composition.to_dict()
    assert json.loads(json.dumps(data)) == data, "JSON-serialisable"
    assert FigureComposition.from_dict(data) == composition
    with pytest.raises(ValueError, match="takes no rows"):
        FigureComposition.from_dict({**data, "rows": 2})


# --- building -------------------------------------------------------------------


def test_cells_are_built_by_their_own_recipes(entries):
    model = build_composition(_equilibrium(), entries)
    assert isinstance(model, Panels) and (model.nrows, model.ncols) == (2, 2)
    assert model.spans == ((0, 0, 2, 1), (0, 1, 1, 1), (1, 1, 1, 1))
    assert model.suptitle == "#39915", "the shot is named, as an overview does"
    assert build_composition(_equilibrium(title=""), entries).suptitle == ""
    single = vaft.omas.extract_plasma_current_time(entries[0][1])
    stacked = build_composition(FigureComposition.stack(["plasma_current_time"]), entries).models[0]
    assert isinstance(stacked, LineSeries)
    assert [s.y.tolist() for s in stacked.series] == [s.y.tolist() for s in single.series]


def test_a_cell_holds_one_panel(entries):
    with pytest.raises(ValueError, match="draws 7 panels of its own"):
        build_composition(FigureComposition.stack(["equilibrium_overview"]), entries)
    with pytest.raises(ValueError, match="a composed cell holds one panel"):
        build_composition(FigureComposition((1, 1), (FigureCell("flux_loop_time_flux", options={"layout": "subplots"}),)),
                          entries)
    with pytest.raises(ValueError):
        build_composition(FigureComposition((1, 1), (FigureCell("plasma_current_time", options={"yunit": "furlong"}),)),
                          entries)


# --- Matplotlib -------------------------------------------------------------------


def test_matplotlib_stack_shares_time_and_labels_the_panels(ods):
    before = dict(plt.rcParams)
    figure, axes = vaft.omas.compose(FigureComposition.stack(STACK, panel_labels=True), ods)
    assert len(axes) == 3 and figure.axes[:3] == list(axes)
    assert all(axes[0].get_shared_x_axes().joined(axes[0], other) for other in axes[1:])
    axes[2].set_xlim(0.30, 0.33)
    assert axes[0].get_xlim() == pytest.approx((0.30, 0.33)), "zooming one zooms all"
    assert axes[0].get_xlabel() == "" and axes[2].get_xlabel()
    assert not any(label.get_visible() and label.get_text() for label in axes[0].get_xticklabels())
    marks = [[t.get_text() for t in axis.texts if t.get_text().startswith("(")] for axis in axes]
    assert marks == [["(a)"], ["(b)"], ["(c)"]]
    assert dict(plt.rcParams) == before, "no global rcParams change"


def test_matplotlib_spans_place_the_map_over_two_rows(ods):
    figure, axes = vaft.omas.compose(_equilibrium(), ods)
    assert len(axes) == 3
    # the map sits in a sub-grid with its colorbar; that sub-grid covers both rows
    spec = axes[0].get_subplotspec().get_topmost_subplotspec()
    assert (spec.rowspan.start, spec.rowspan.stop) == (0, 2)
    assert len(figure.axes) == 4, "the map's colorbar gets its own cell"
    assert axes[0].get_title() == "Poloidal Flux" and axes[2].get_title() == "Safety Factor q"


def test_matplotlib_takes_the_figure_presentation(ods):
    figure, _ = vaft.omas.compose(FigureComposition.side_by_side(["equilibrium_profile_q", "equilibrium_profile_pressure"]),
                                  ods, format="double_column")
    assert tuple(figure.get_size_inches()) == pytest.approx((7.0, 3.5))
    with pytest.raises(TypeError, match="plot options belong to its cells"):
        vaft.omas.compose(FigureComposition.stack(STACK), ods, yunit="MA")


def test_a_power_spectrum_and_a_trace_share_one_figure(ods):
    figure, axes = vaft.omas.compose(
        FigureComposition.side_by_side(["plasma_current_time", "mirnov_spectrum"]), ods,
    )
    assert len(axes) == 2 and axes[1].get_lines()


def test_native_imas_input_composes_the_same_way(ods):
    entry = vaft.imas.load(vaft.data.sample(39915, representation="imas"))
    _, from_imas = vaft.imas.compose(FigureComposition.stack(["plasma_current_time", "equilibrium_time_q95"]), entry)
    _, from_omas = vaft.omas.compose(FigureComposition.stack(["plasma_current_time", "equilibrium_time_q95"]), ods)
    assert [a.get_title() for a in from_imas] == [a.get_title() for a in from_omas]


# --- Plotly -----------------------------------------------------------------------


def test_plotly_matches_the_linked_axes(ods):
    pytest.importorskip("plotly")
    figure = vaft.omas.compose(FigureComposition.stack(STACK, panel_labels=True), ods, backend="plotly")
    layout = figure.layout
    assert layout.xaxis2.matches == "x" and layout.xaxis3.matches == "x"
    assert layout.xaxis.showticklabels is False and layout.xaxis3.showticklabels is not False
    assert [a.text for a in layout.annotations if a.text.startswith("<b>(")] == ["<b>(a)</b>", "<b>(b)</b>", "<b>(c)</b>"]
    assert layout.title.text == "#39915"


def test_plotly_spans_and_refusals(ods):
    pytest.importorskip("plotly")
    figure = vaft.omas.compose(_equilibrium(), ods, backend="plotly")
    tall = figure.layout.yaxis.domain
    short = figure.layout.yaxis2.domain
    assert tall[1] - tall[0] > 1.5 * (short[1] - short[0]), "the map spans both rows"
    with pytest.raises(TypeError, match="does not apply them"):
        vaft.omas.compose(FigureComposition.stack(STACK), ods, backend="plotly", format="screen")
    with pytest.raises(NotImplementedError, match="GeometryLayers"):
        vaft.omas.compose(FigureComposition.stack(["machine_geometry_poloidal", "plasma_current_time"]), ods,
                          backend="plotly")


def test_the_dict_form_draws_the_same_figure(ods):
    composition = FigureComposition.stack(STACK)
    _, axes = vaft.omas.compose(composition.to_dict(), ods)
    assert [a.get_title() for a in axes] == ["Plasma Current", "Flux Loop Voltage", "q95"]
    with pytest.raises(TypeError, match="FigureComposition or its to_dict"):
        render_composition(["plasma_current_time"], vaft.omas.normalize_entries(ods, label="shot"))


# --- the Panels contract the renderers rely on -----------------------------------


def test_panel_links_are_validated(entries):
    model = build_composition(FigureComposition.stack(STACK[:2]), entries)
    with pytest.raises(ValueError, match="'x' or 'y'"):
        Panels(models=model.models, links=(("z", (0, 1)),))
    with pytest.raises(ValueError, match="outside 0..1"):
        Panels(models=model.models, links=(("x", (0, 5)),))
    with pytest.raises(ValueError, match="fewer than two"):
        Panels(models=model.models, links=(("x", (1, 1)),))


# --- regressions from the cold review -------------------------------------------


def test_a_fixed_range_on_any_linked_cell_sets_the_linked_range(ods):
    window = (0.30, 0.32)
    for cells in (
        [FigureCell("plasma_current_time"), FigureCell("equilibrium_time_q95", options={"time_range": window})],
        [FigureCell("equilibrium_time_q95", options={"time_range": window}), FigureCell("plasma_current_time")],
    ):
        _, axes = vaft.omas.compose(FigureComposition.stack(cells), ods)
        assert [a.get_xlim() for a in axes] == [pytest.approx(window)] * 2, "order must not matter"
        figure = vaft.omas.compose(FigureComposition.stack(cells), ods, backend="plotly")
        ranges = [figure.layout.xaxis.range, figure.layout.xaxis2.range]
        anchored = next(r for r in ranges if r is not None)
        assert tuple(anchored) == pytest.approx(window)
        assert {figure.layout.xaxis.matches, figure.layout.xaxis2.matches} - {None} in ({"x"}, {"x2"})


def test_unfixed_links_span_every_linked_panel_s_data(ods):
    _, alone = vaft.omas.compose(FigureComposition.stack(["plasma_current_time"]), ods)
    _, axes = vaft.omas.compose(FigureComposition.stack(["equilibrium_time_q95", "plasma_current_time"]), ods)
    low, high = axes[0].get_xlim()
    assert low <= alone[0].get_xlim()[0] + 1e-9 and high >= alone[0].get_xlim()[1] - 1e-9, \
        "q95 first still shows the whole plasma current"


def test_a_colorbar_in_a_stack_keeps_the_time_axes_aligned():
    ods = vaft.omas.sample_ods(40600)
    _, axes = vaft.omas.compose(
        FigureComposition.stack(["plasma_current_time", "camera_visible_spectrogram", "flux_loop_time_voltage"]), ods,
    )
    boxes = [axis.get_position() for axis in axes]
    assert max(b.x1 for b in boxes) - min(b.x1 for b in boxes) < 1e-6
    assert max(b.x0 for b in boxes) - min(b.x0 for b in boxes) < 1e-6


def test_share_x_and_tick_labels_follow_every_column_a_cell_covers(ods):
    composition = FigureComposition(
        shape=(2, 2), share_x=True,
        cells=(
            FigureCell("plasma_current_time", 0, 0, colspan=2),
            FigureCell("equilibrium_time_q95", 1, 0),
            FigureCell("flux_loop_time_voltage", 1, 1),
        ),
    )
    assert composition.resolved_links() == (("x", (0, 1, 2)),)
    _, axes = vaft.omas.compose(composition, ods)
    assert axes[0].get_xlabel() == "" and axes[1].get_xlabel() and axes[2].get_xlabel()
    figure = vaft.omas.compose(composition, ods, backend="plotly")
    assert figure.layout.xaxis.showticklabels is False


def test_y_links_tie_the_rows(ods):
    composition = FigureComposition.side_by_side(["equilibrium_profile_q", "equilibrium_profile_pressure"],
                                                 links=(AxisLink("y", ("equilibrium_profile_q", "equilibrium_profile_pressure")),))
    _, axes = vaft.omas.compose(composition, ods)
    assert axes[0].get_shared_y_axes().joined(axes[0], axes[1])


def test_figure_options_belong_to_compose_not_to_a_cell():
    for option in ("format", "figsize", "save_path", "row_heights"):
        with pytest.raises(ValueError, match="set them on compose"):
            FigureCell("plasma_current_time", options={option: "x"})


def test_the_dict_form_is_plain_json_and_equal_after_a_round_trip():
    import numpy as np

    composition = FigureComposition(
        shape=(1, 2),
        cells=(
            {"plot": "plasma_current_time", "row": 0, "col": 0, "options": {"time_range": (0.30, np.float64(0.32))}},
            FigureCell("equilibrium_time_q95", 0, 1, options={"time_range": np.array([0.30, 0.32])}),
        ),
    )
    data = composition.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert FigureComposition.from_dict(data) == composition
    assert dict(composition.cells[0].options) == {"time_range": [0.30, 0.32]}
    with pytest.raises(TypeError):
        hash(composition)


def test_the_rcparams_stay_untouched_with_a_theme_and_format(ods):
    before = dict(plt.rcParams)
    vaft.omas.compose(FigureComposition.stack(STACK[:2]), ods, theme="technical", format="single_column")
    assert dict(plt.rcParams) == before
