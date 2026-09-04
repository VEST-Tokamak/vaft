"""Layout contract: overlay, subplots, grouped (issue #260).

Layout arranges an already-resolved channel set and never changes it.  The
figure's structure -- return type, axes count and order -- is a function of the
layout and the resolved selection alone, never of a count threshold and never
of which shot happened to be loaded.  Policy:
``notebooks/plotting_sample_using_vaft_plot_module.ipynb``.
"""

import contextlib
import io
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import omas
import pytest

import vaft
import vaft.omas
from vaft.plot.models import Panels


def _load(rel):
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path(rel)))


@pytest.fixture(scope="module")
def shots():
    return {
        39915: _load("samples/39915/omas.json.gz"),
        41524: _load("samples/41524/imas.nc"),
        41672: _load("samples/41672/imas.nc"),
    }


def _traces(axes):
    return [line for panel in np.asarray(axes).ravel() for line in panel.lines]


# ---------------------------------------------------------------------------
# Return shape follows the layout, not the data (sections 3, 9)
# ---------------------------------------------------------------------------

def test_overlay_is_the_default_and_returns_one_axes(shots):
    figure, axes = vaft.omas.plot_flux_loop_time_flux(shots[39915], selection="inboard")
    assert isinstance(axes, matplotlib.axes.Axes)
    figure2, axes2 = vaft.omas.plot_flux_loop_time_flux(shots[39915], selection="inboard", layout="overlay")
    assert isinstance(axes2, matplotlib.axes.Axes)
    plt.close("all")


def test_subplots_returns_an_axes_array_even_for_one_channel(shots):
    figure, axes = vaft.omas.plot_flux_loop_time_flux(
        shots[39915], selection="inboard_mid", layout="subplots"
    )
    assert isinstance(axes, np.ndarray) and axes.shape == (1, 1)
    plt.close(figure)


def test_grouped_returns_an_axes_array_in_canonical_region_order(shots):
    figure, axes = vaft.omas.plot_flux_loop_time_flux(shots[39915], layout="grouped")
    assert isinstance(axes, np.ndarray)
    assert [panel.get_title() for panel in axes.ravel()] == ["inboard", "outboard"]
    plt.close(figure)


def test_an_unknown_layout_is_refused(shots):
    with pytest.raises(ValueError, match="overlay, subplots, grouped"):
        vaft.omas.plot_flux_loop_time_flux(shots[39915], layout="stacked")


# ---------------------------------------------------------------------------
# Layout never changes selection (section 1)
# ---------------------------------------------------------------------------

def test_every_layout_draws_the_same_resolved_channels(shots):
    ods = shots[39915]
    overlay = vaft.omas.plot_flux_loop_time_flux(ods, selection="inboard")[1]
    subplots = vaft.omas.plot_flux_loop_time_flux(ods, selection="inboard", layout="subplots")[1]
    grouped = vaft.omas.plot_flux_loop_time_flux(ods, selection="inboard", layout="grouped")[1]
    reference = sorted(line.get_label() for line in overlay.lines)
    # A 7-panel grid has a hidden spare slot; only drawn panels carry channels.
    assert sorted(panel.get_title() for panel in subplots.ravel() if panel.get_visible()) == reference
    assert sorted(line.get_label() for line in _traces(grouped)) == reference
    plt.close("all")


def test_subplot_order_follows_the_resolved_selection(shots):
    figure, axes = vaft.omas.plot_flux_loop_time_flux(
        shots[39915], selection=[7, 3, 5], layout="subplots"
    )
    assert [panel.get_title()[:3] for panel in axes.ravel()] == ["[7]", "[3]", "[5]"]
    plt.close(figure)


# ---------------------------------------------------------------------------
# Structure is a function of layout + selection, never of a count (sections 5, 6)
# ---------------------------------------------------------------------------

def test_the_grid_is_a_deterministic_function_of_the_panel_count():
    from vaft.omas._plot_recipes import _layout_columns

    assert [_layout_columns(n) for n in (1, 6, 7, 16, 17, 36, 37, 43)] == [1, 1, 2, 2, 3, 3, 4, 4]
    assert _layout_columns(43, requested=2) == 2


def test_a_large_selection_keeps_every_channel(shots):
    figure, axes = vaft.omas.plot_b_field_probe_time_field(
        shots[39915], selection="inboard", layout="subplots"
    )
    drawn = [panel for panel in axes.ravel() if panel.get_visible() and panel.lines]
    assert axes.shape == (11, 4)
    assert len(drawn) == 42  # 43 inboard probes, one of which carries no data
    plt.close(figure)


def test_ncols_overrides_the_default_columns(shots):
    figure, axes = vaft.omas.plot_flux_loop_time_flux(
        shots[39915], selection="inboard", layout="subplots", ncols=1
    )
    assert axes.shape == (7, 1)
    plt.close(figure)


# ---------------------------------------------------------------------------
# ax= contract (section 8)
# ---------------------------------------------------------------------------

def test_supplied_axes_must_match_the_layout(shots):
    ods = shots[39915]
    figure, single = plt.subplots()
    with pytest.raises(ValueError, match="7 panels but received 1"):
        vaft.omas.plot_flux_loop_time_flux(ods, selection="inboard", layout="subplots", ax=single)
    figure, seven = plt.subplots(7, 1)
    _, out = vaft.omas.plot_flux_loop_time_flux(ods, selection="inboard", layout="subplots", ax=seven)
    assert all(panel.lines for panel in np.asarray(out).ravel())
    figure, three = plt.subplots(3, 1)
    with pytest.raises(ValueError, match="received 3 axes"):
        vaft.omas.plot_flux_loop_time_flux(ods, selection="inboard", layout="subplots", ax=three)
    plt.close("all")


# ---------------------------------------------------------------------------
# grouped uses canonical regions only (sections 5, 7)
# ---------------------------------------------------------------------------

def test_grouped_refuses_a_family_with_no_radial_split():
    ods = omas.ODS()
    ods["magnetics.time"] = np.linspace(0.0, 0.1, 4)
    for index in range(4):
        ods[f"magnetics.b_field_pol_probe.{index}.name"] = f"MP{index}"
        ods[f"magnetics.b_field_pol_probe.{index}.position.r"] = 0.796
        ods[f"magnetics.b_field_pol_probe.{index}.position.z"] = 0.1 * index
        ods[f"magnetics.b_field_pol_probe.{index}.voltage.data"] = np.ones(4)
    with pytest.raises(ValueError, match="grouped layout is unsupported"):
        vaft.omas.plot_mirnov_time_voltage(ods, layout="grouped")


def test_grouped_applies_to_multi_channel_plots_only(shots):
    with pytest.raises(ValueError, match="multi-channel plots only"):
        vaft.omas.plot_plasma_current_time(shots[39915], layout="grouped")


# ---------------------------------------------------------------------------
# Display policy survives layout (sections 15, 16, 17)
# ---------------------------------------------------------------------------

def test_an_invalid_channel_panel_is_marked_in_subplots(shots):
    figure, axes = vaft.omas.plot_mirnov_time_voltage(
        shots[39915], selection=[26, 70], layout="subplots"
    )
    flagged = [panel for panel in axes.ravel() if any(t.get_text() == "invalid" for t in panel.texts)]
    assert len(flagged) == 1
    assert flagged[0].lines[0].get_linestyle() == "--"
    plt.close(figure)


def test_multi_shot_subplot_puts_shots_inside_the_channel_panel(shots):
    figure, axes = vaft.omas.plot_flux_loop_time_flux(
        [shots[39915], shots[41524]], selection=[5], layout="subplots"
    )
    assert axes.shape == (1, 1)
    panel = axes.ravel()[0]
    assert [line.get_label() for line in panel.lines] == ["39915", "41524"]
    # The panel title already names the channel; the legend carries no title.
    assert panel.get_legend().get_title().get_text() == ""
    plt.close(figure)


def test_shared_x_is_dropped_across_mixed_time_bases(shots):
    figure, axes = vaft.omas.plot_mirnov_time_voltage(
        shots[39915], selection=[26, 70], layout="subplots"
    )
    a, b = axes.ravel()
    assert not a.get_shared_x_axes().joined(a, b)
    plt.close(figure)
    figure, axes = vaft.omas.plot_flux_loop_time_flux(shots[39915], selection="inboard", layout="subplots")
    a, b = axes.ravel()[:2]
    assert a.get_shared_x_axes().joined(a, b)
    plt.close(figure)


def test_a_composite_member_cannot_take_a_layout(shots):
    with pytest.raises(ValueError, match="cannot themselves take a layout"):
        vaft.omas.plot_current_overview(shots[39915], layout="subplots")


# ---------------------------------------------------------------------------
# Overviews keep one shape on every shot (sections 6, 18; C12)
# ---------------------------------------------------------------------------

def test_the_diagnostics_overview_has_the_same_shape_on_every_shot(shots):
    shapes, placeholders = set(), {}
    for shot, ods in shots.items():
        figure, axes = vaft.omas.plot_diagnostics_overview(ods)
        shapes.add(axes.shape)
        placeholders[shot] = sum(1 for panel in axes.ravel() if not panel.axison)
        plt.close(figure)
    assert shapes == {(5, 2)}
    assert all(0 < n < 10 for n in placeholders.values()), placeholders


def test_the_diagnostics_overview_excludes_flagged_channels_by_default(shots):
    figure, axes = vaft.omas.plot_diagnostics_overview(shots[39915])
    assert not any(line.get_linestyle() == "--" for line in _traces(axes))
    plt.close(figure)
    # The caller's own choice still wins over the overview's default: every
    # channel drawn (selection="all"), the flagged ones shown demoted.
    figure, axes = vaft.omas.plot_diagnostics_overview(shots[39915], validity="show", selection="all")
    assert any(line.get_linestyle() == "--" for line in _traces(axes))
    plt.close(figure)


def test_placeholders_keep_a_fixed_grid_and_name_the_missing_member():
    from vaft.plot.models import LineSeries, Series

    trace = Series(x=np.arange(3.0), y=np.arange(3.0))
    panels = Panels(models=(LineSeries(series=(trace,)),), ncols=2,
                    placeholders=((1, "x\nnot available"),))
    assert (panels.nrows, panels.ncols) == (1, 2)
    with pytest.raises(ValueError, match="outside the grid"):
        Panels(models=(LineSeries(series=(trace,)),), ncols=1, placeholders=((5, "x"),))


# ---------------------------------------------------------------------------
# Grouping classifies each entry against its own geometry (section 17)
# ---------------------------------------------------------------------------

def _probe_array(radii):
    """A Mirnov array with the given major radii, one sample row per probe."""
    ods = omas.ODS()
    ods["magnetics.time"] = np.linspace(0.0, 0.1, 4)
    for index, radius in enumerate(radii):
        ods[f"magnetics.b_field_pol_probe.{index}.name"] = f"MP{index}"
        ods[f"magnetics.b_field_pol_probe.{index}.position.r"] = radius
        ods[f"magnetics.b_field_pol_probe.{index}.position.z"] = 0.05 * index
        ods[f"magnetics.b_field_pol_probe.{index}.voltage.data"] = np.ones(4) * (index + 1)
    return ods


def test_grouped_classifies_each_shot_against_its_own_divider():
    # Shot A splits at 0.45 m; shot B's array was moved outward and splits at
    # 0.775 m.  Judged by A's divider every B probe would be outboard.
    shots = omas.ODC()
    shots["A"] = _probe_array([0.1, 0.1, 0.8, 0.8])
    shots["B"] = _probe_array([0.6, 0.6, 0.95, 0.95])
    figure, axes = vaft.omas.plot_mirnov_time_voltage(shots, layout="grouped", label="key")
    by_region = {panel.get_title(): len(panel.lines) for panel in axes.ravel()}
    assert by_region == {"inboard": 4, "outboard": 4}
    plt.close(figure)


def test_grouped_refuses_an_entry_without_a_split_beside_one_that_has():
    shots = omas.ODC()
    shots["A"] = _probe_array([0.1, 0.1, 0.8, 0.8])
    shots["C"] = _probe_array([0.796] * 4)
    with pytest.raises(ValueError, match="grouped layout is unsupported .* in entry 'C'"):
        vaft.omas.plot_mirnov_time_voltage(shots, layout="grouped", label="key")


def test_supplied_axes_must_be_axes(shots):
    with pytest.raises(TypeError, match="ax entries must be matplotlib Axes; got str"):
        vaft.omas.plot_flux_loop_time_flux(
            shots[39915], selection="inboard", layout="subplots", ax=["a"] * 7
        )


def test_panels_sharing_a_time_base_label_it_once_per_column(shots):
    figure, axes = vaft.omas.plot_pf_coil_time_current_turns(shots[39915], layout="subplots")
    labels = [axis.get_xlabel() for axis in axes.ravel() if axis.get_visible()]
    assert labels[:-1] == [""] * (len(labels) - 1) and labels[-1] == "Time [s]"
    assert all(axis.get_ylabel() == "Coil Ampere-turns [kA-turns]" for axis in axes.ravel() if axis.get_visible())
    plt.close(figure)
    figure, axes = vaft.omas.plot_pf_coil_time_current_turns(shots[39915], layout="subplots", ncols=2)
    columns = {}
    for axis in axes.ravel():
        if axis.get_visible():
            columns.setdefault(round(axis.get_position().x0, 3), []).append(axis.get_xlabel())
    assert all(labels[-1] == "Time [s]" and not any(labels[:-1]) for labels in columns.values())
    # The shorter column's last panel shows its tick numbers too.
    figure.canvas.draw()
    bottoms = [axis for axis in axes.ravel() if axis.get_visible() and axis.get_xlabel() == "Time [s]"]
    assert len(bottoms) == 2 and all(any(t.get_text() for t in axis.get_xticklabels()) for axis in bottoms)
    plt.close(figure)
