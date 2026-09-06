"""Issue #481: what a line plot is drawn against, and how it says so.

``x=`` is a per-plot vocabulary: ``time`` (the default), the sample
``index``, and the sibling quantities a recipe declares -- each sampled on
the ordinate's own grid, so nothing is interpolated.  An abscissa this input
cannot supply falls back to the index and the axis says index, never a time
label over sample numbers (the issue #276 contract, applied to lines).
"""

from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import (
    RECIPES,
    Abscissa,
    LineRecipe,
    abscissa_options,
    abscissa_options_for,
    build_model,
)
from vaft.plot.models import LineSeries, Panels


@pytest.fixture(scope="module")
def sample():
    return vaft.omas.load(vaft.data.sample(39915, representation="omas"))


@pytest.fixture(scope="module")
def entries(sample):
    return normalize_entries(sample)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# the vocabulary
# ---------------------------------------------------------------------------

def test_every_line_plot_offers_time_and_the_sample_index():
    assert abscissa_options(LineRecipe(y_path="x")) == ("time", "index")
    assert abscissa_options_for("flux_loop_time_flux") == ("time", "index")
    assert abscissa_options_for("equilibrium_field_psi") is None  # not a line plot


def test_the_equilibrium_scalars_declare_the_plasma_current_as_a_sibling():
    for name in ("equilibrium_time_q95", "equilibrium_time_li", "equilibrium_time_beta_n"):
        assert abscissa_options_for(name) == ("time", "ip", "index")
        declared = RECIPES[name].abscissae
        assert [entry.name for entry in declared] == ["ip"]
        assert declared[0].paths == ("equilibrium.time_slice.{i}.global_quantities.ip",)
    # A quantity against itself says nothing.
    assert abscissa_options_for("equilibrium_time_plasma_current") == ("time", "index")


def test_a_composite_offers_the_union_of_its_members():
    options = abscissa_options_for("equilibrium_overview_time")
    assert options is None or ("time" in options and "index" in options)


def test_an_unknown_abscissa_is_refused_naming_the_options(sample, entries):
    with pytest.raises(ValueError, match="x= one of time, ip, index; got 'rho'"):
        build_model("equilibrium_time_q95", entries, x="rho")
    with pytest.raises(ValueError, match="x must be one of time, index"):
        vaft.omas.plot_flux_loop_time_flux(sample, x="ip")


# ---------------------------------------------------------------------------
# what gets drawn
# ---------------------------------------------------------------------------

def test_the_sample_index_is_drawn_and_labelled_as_one(entries):
    time = build_model("flux_loop_time_flux", entries)
    index = build_model("flux_loop_time_flux", entries, x="index")
    assert time.x_label == "Time" and time.x_unit == "s"
    assert index.x_label == "Sample index" and index.x_unit == ""
    for drawn, stored in zip(index.series, time.series):
        np.testing.assert_allclose(drawn.x, np.arange(np.asarray(stored.y).size))
        np.testing.assert_allclose(drawn.y, stored.y)  # only the abscissa changes


def test_a_sibling_quantity_is_read_on_the_ordinates_own_grid(sample, entries):
    against_ip = build_model("equilibrium_time_q95", entries, x="ip")
    against_time = build_model("equilibrium_time_q95", entries)
    stored = np.asarray(
        [float(sample[f"equilibrium.time_slice.{i}.global_quantities.ip"]) for i in range(9)]
    )
    assert against_ip.x_label == "Plasma Current" and against_ip.x_unit == "kA"
    np.testing.assert_allclose(against_ip.series[0].x, stored * 1e-3)
    np.testing.assert_allclose(against_ip.series[0].y, against_time.series[0].y)
    assert against_ip.series[0].x.size == against_ip.series[0].y.size


def test_xunit_applies_to_the_chosen_abscissa(entries):
    assert build_model("equilibrium_time_q95", entries, xunit="ms").x_unit == "ms"
    in_amperes = build_model("equilibrium_time_q95", entries, x="ip", xunit="A")
    assert in_amperes.x_unit == "A" and in_amperes.series[0].x.max() > 1e4
    with pytest.raises(ValueError, match="no display conversions exist"):
        build_model("equilibrium_time_q95", entries, x="index", xunit="ms")


def test_an_input_without_the_axis_says_index_rather_than_time(sample):
    """A missing time array used to be drawn as a time axis of sample numbers."""
    timeless = copy.deepcopy(sample)
    del timeless["equilibrium.time"]
    model = build_model("equilibrium_time_q95", normalize_entries(timeless))
    assert model.x_label == "Sample index" and model.x_unit == ""
    np.testing.assert_allclose(model.series[0].x, np.arange(9))


# ---------------------------------------------------------------------------
# the other options are unaffected
# ---------------------------------------------------------------------------

def test_layout_selection_and_orientation_are_unaffected_by_the_choice(entries):
    panels = build_model("flux_loop_time_flux", entries, x="index", layout="subplots")
    assert isinstance(panels, Panels) and len(panels.models) == 11
    assert all(isinstance(m, LineSeries) and m.x_label == "Sample index" for m in panels.models)
    grouped = build_model("flux_loop_time_flux", entries, x="index", layout="grouped")
    assert isinstance(grouped, Panels) and len(grouped.models) == 2
    chosen = build_model("flux_loop_time_flux", entries, x="index", selection="outboard")
    assert len(chosen.series) == 4
    canonical = build_model("diamagnetic_flux_time", entries, x="index", orientation="canonical")
    intuitive = build_model("diamagnetic_flux_time", entries, x="index", orientation="intuitive")
    np.testing.assert_allclose(intuitive.series[0].y, -canonical.series[0].y)


def test_the_reconstruction_overlay_is_refused_off_the_time_axis(entries):
    with pytest.raises(ValueError, match="cannot be placed on the 'index' abscissa"):
        build_model("plasma_current_time", entries, x="index", synthetic="equilibrium")
    # On the time axis the overlay is built as before (this shot stores no
    # reconstruction constraints, so it adds no trace; the call still stands).
    overlay = build_model("plasma_current_time", entries, synthetic="equilibrium")
    assert overlay.x_label == "Time"


# ---------------------------------------------------------------------------
# discovery, controls, equivalence
# ---------------------------------------------------------------------------

def test_discovery_states_the_abscissae_and_narrows_them_to_the_input(sample):
    record = next(r for r in vaft.omas.available_plots(sample) if r.name == "equilibrium_time_q95")
    assert record.abscissa == {
        "default": "time", "options": ("time", "ip", "index"), "declared": ("time", "ip", "index"),
    }
    assert "abscissa: time (default) | ip | index" in str(
        vaft.omas.available_plots(sample, query="equilibrium", view="time")
    )
    assert "abscissa: time by default; time | ip | index" in str(
        vaft.omas.available_plots(sample, query="q95", detail=True)
    )
    without_ip = copy.deepcopy(sample)
    for index in range(9):
        del without_ip[f"equilibrium.time_slice.{index}.global_quantities.ip"]
    narrowed = next(r for r in vaft.omas.available_plots(without_ip) if r.name == "equilibrium_time_q95")
    assert narrowed.abscissa["options"] == ("time", "index")
    assert "ip" in narrowed.abscissa["declared"]


def test_the_control_layer_offers_the_abscissa(sample):
    from vaft.plot.controls import controls_for

    record = next(r for r in vaft.omas.available_plots(sample) if r.name == "equilibrium_time_q95")
    control = next(c for c in controls_for(record) if c.name == "x")
    assert control.options == ("time", "ip", "index") and control.default == "time"
    result = vaft.omas.plot_equilibrium_time_q95(sample, interactive=True, interaction_backend="none")
    result.state.set("x", "ip")
    assert result.axes.get_xlabel().startswith("Plasma Current")


def test_omas_and_imas_agree_on_every_abscissa(sample):
    from test_imas_omas_plot_equivalence import assert_models_equal
    from vaft.imas.access import IDSEntry

    with vaft.imas.load(vaft.data.sample(39915, representation="imas"), imas_version="3.41.0") as handle:
        entry = IDSEntry(handle)
        for name, x in (
            ("equilibrium_time_q95", "time"), ("equilibrium_time_q95", "ip"),
            ("equilibrium_time_q95", "index"), ("flux_loop_time_flux", "index"),
        ):
            expected = build_model(name, [("39915", sample)], x=x)
            actual = build_model(name, [("39915", entry)], x=x)
            assert_models_equal(actual, expected)


def test_both_renderers_label_the_chosen_abscissa(sample):
    figure, axes = vaft.omas.plot_equilibrium_time_q95(sample, x="ip")
    assert axes.get_xlabel() == "Plasma Current [kA]"
    plotly = vaft.omas.plot_equilibrium_time_q95(sample, x="index", backend="plotly")
    assert plotly.layout.xaxis.title.text == "Sample index"


def _with_a_broken_channel(sample):
    """Channel 0 flat (dropped by the ``active`` preset) and its time unusable."""
    broken = copy.deepcopy(sample)
    broken["magnetics.flux_loop.0.flux.data"] = np.zeros_like(
        np.asarray(sample["magnetics.flux_loop.0.flux.data"])
    )
    broken["magnetics.flux_loop.0.flux.time"] = np.arange(3, dtype=float)
    return broken


def test_a_channel_the_selection_drops_does_not_decide_the_abscissa(sample):
    """Review of #582: a dropped channel's failed reading used to put every
    surviving channel on a sample index, discarding their real times."""
    broken = _with_a_broken_channel(sample)
    drawn = build_model("flux_loop_time_flux", normalize_entries(broken))
    assert drawn.x_label == "Time" and len(drawn.series) == 10
    reference = build_model("flux_loop_time_flux", normalize_entries(sample))
    np.testing.assert_allclose(drawn.series[0].x, reference.series[1].x)


def test_a_channel_that_is_drawn_does_decide_it(sample):
    """Asked for every channel, the figure includes the broken one and says so."""
    broken = _with_a_broken_channel(sample)
    everything = build_model("flux_loop_time_flux", normalize_entries(broken), selection="all")
    assert everything.x_label == "Sample index" and len(everything.series) == 11
    for trace in everything.series:
        np.testing.assert_allclose(trace.x, np.arange(np.asarray(trace.y).size))


def test_a_sibling_missing_from_one_slice_is_still_offered(sample):
    """Discovery must not refuse an abscissa the builder would draw."""
    partial = copy.deepcopy(sample)
    del partial["equilibrium.time_slice.0.global_quantities.ip"]
    record = next(r for r in vaft.omas.available_plots(partial) if r.name == "equilibrium_time_q95")
    assert record.abscissa["options"] == ("time", "ip", "index")
    model = build_model("equilibrium_time_q95", normalize_entries(partial), x="ip")
    assert model.x_label == "Plasma Current" and np.isnan(model.series[0].x[0])


def test_the_abscissa_survives_a_grouped_layout_with_a_selection(entries):
    grouped = build_model(
        "flux_loop_time_flux", entries, x="index", layout="grouped", selection="valid"
    )
    assert isinstance(grouped, Panels)
    assert all(m.x_label == "Sample index" for m in grouped.models)
