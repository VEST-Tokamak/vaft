"""Issue #486: flux-loop and B-probe values against sensor position at one time.

A new ``spatial`` view: ``coordinate="z"`` draws the inboard and outboard
sensors as two panels split by the family's own radial divider; ``"theta"``
one panel against the poloidal angle about the layout centre.  ``time=``
snaps to a stored sample; ``time_slice=`` maps through a stored equilibrium
slice.
"""

from __future__ import annotations

import copy
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import build_model, resolve_time_sample
from vaft.plot.models import Panels, Profile1D

from _sample_fixtures import sample_ods


@pytest.fixture(scope="module")
def sample():
    return sample_ods(39915)


@pytest.fixture(scope="module")
def entries(sample):
    return normalize_entries(sample)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# time snapping
# ---------------------------------------------------------------------------

def test_a_time_snaps_to_the_nearest_stored_sample_and_says_so():
    times = np.linspace(0.26, 0.36, 2501)
    assert resolve_time_sample(times, 0.31)[:2] == (1250, pytest.approx(0.31))
    index, stored, reason = resolve_time_sample(times, 0.310015)
    assert index == 1250 and stored == pytest.approx(0.31) and reason == "nearest sample to t = 310.01 ms"
    assert resolve_time_sample(times, None) == (1250, pytest.approx(0.31), "middle sample")
    with pytest.warns(UserWarning, match="outside the stored samples"):
        assert resolve_time_sample(times, 0.5)[0] == 2500
    with pytest.raises(ValueError, match="finite"):
        resolve_time_sample(times, float("nan"))
    with pytest.raises(ValueError, match="time axis"):
        resolve_time_sample([], 0.3)


def test_the_view_and_the_names_are_registered():
    from vaft.plot import canonical_names
    from vaft.plot.registry import VIEWS, get_spec

    assert "spatial" in VIEWS
    for name, subject in (("flux_loop_spatial_flux", "flux_loop"), ("b_field_probe_spatial_field", "b_field_probe")):
        assert name in canonical_names()
        spec = get_spec(name)
        assert spec.view == "spatial" and spec.subject == subject and spec.model is Profile1D
    assert callable(vaft.omas.plot_flux_loop_spatial_flux) and callable(vaft.imas.plot_b_field_probe_spatial_field)


# ---------------------------------------------------------------------------
# the flux loops
# ---------------------------------------------------------------------------

def test_z_splits_the_loops_into_inboard_and_outboard_panels(entries):
    model = build_model("flux_loop_spatial_flux", entries, time=0.31)
    assert isinstance(model, Panels) and [p.title for p in model.models] == ["inboard", "outboard"]
    inboard, outboard = model.models
    assert len(inboard.series[0].x) == 7 and len(outboard.series[0].x) == 4
    for panel in model.models:
        assert panel.coordinate_label == "Z [m]"
        assert np.all(np.diff(panel.series[0].x) >= 0)
        assert panel.y_unit == "mWb" and panel.display.unit == "mWb"
    assert "at t = 310.00 ms (nearest sample to t = 310.00 ms)" in model.suptitle
    assert model.ncols == 2 and model.share_x is False


def test_theta_is_one_panel_in_degrees_about_the_layout_centre(sample, entries):
    from vaft.process.magnetics import magnetics_sensor_centre

    model = build_model("flux_loop_spatial_flux", entries, coordinate="theta")
    assert isinstance(model, Profile1D)
    x = model.series[0].x
    assert x.size == 11 and np.all((x >= 0) & (x < 360)) and np.all(np.diff(x) >= 0)
    assert model.x_limits == (0.0, 360.0)
    r = [float(sample[f"magnetics.flux_loop.{i}.position.0.r"]) for i in range(11)]
    z = [float(sample[f"magnetics.flux_loop.{i}.position.0.z"]) for i in range(11)]
    r0, z0 = magnetics_sensor_centre(r, z)
    assert model.coordinate_label == f"Poloidal angle θ [deg] about (R, Z) = ({r0:.3f}, {z0:.3f}) m"
    moved = build_model("flux_loop_spatial_flux", entries, coordinate="theta", centre=(0.3, 0.0))
    assert not np.allclose(moved.series[0].x, x)
    with pytest.raises(ValueError, match="poloidal_angle on every selected channel"):
        build_model("flux_loop_spatial_flux", entries, coordinate="theta", angle="stored")


def test_the_default_is_the_middle_sample_and_time_slice_maps_through_the_equilibrium(sample, entries):
    default = build_model("flux_loop_spatial_flux", entries)
    assert "(middle sample)" in default.suptitle
    by_slice = build_model("flux_loop_spatial_flux", entries, time_slice=4)
    stored = float(sample["equilibrium.time_slice.4.time"])
    assert f"t = {stored * 1e3:.2f} ms" in by_slice.suptitle
    no_equilibrium = copy.deepcopy(sample)
    del no_equilibrium["equilibrium"]
    with pytest.raises(ValueError, match="time_slice= needs a stored equilibrium"):
        build_model("flux_loop_spatial_flux", normalize_entries(no_equilibrium), time_slice=4)
    with pytest.warns(UserWarning, match="outside the stored samples"):
        build_model("flux_loop_spatial_flux", entries, time=0.5)


# ---------------------------------------------------------------------------
# the probes
# ---------------------------------------------------------------------------

def test_probes_split_by_their_own_divider_and_the_presets_apply(sample, entries):
    from vaft.plot.backend.recipes import _resolve_preset

    model = build_model("b_field_probe_spatial_field", entries)
    counts = {p.title: len(p.series[0].x) for p in model.models}
    assert set(counts) == {"inboard", "outboard"} and sum(counts.values()) <= 63
    outboard = build_model("b_field_probe_spatial_field", entries, selection="outboard")
    assert [p.title for p in outboard.models] == ["outboard"]
    expected = _resolve_preset(sample, "magnetics.b_field_pol_probe", 76, "outboard", ("magnetics.b_field_pol_probe.{i}.field.data",))
    assert len(outboard.models[0].series[0].x) <= len(expected)
    assert all(p.y_unit == "mT" for p in model.models)


def test_a_flagged_channel_is_masked_under_all_and_absent_under_active(entries):
    everything = build_model("b_field_probe_spatial_field", entries, selection="all")
    masks = [p.series[0].valid_mask for p in everything.models]
    assert any(m is not None and not m.all() for m in masks)
    active = build_model("b_field_probe_spatial_field", entries)
    assert all(p.series[0].valid_mask is None for p in active.models)
    assert sum(len(p.series[0].x) for p in active.models) < sum(len(p.series[0].x) for p in everything.models)


def test_the_stored_angle_collapses_on_vest_and_is_documented(entries):
    model = build_model("b_field_probe_spatial_field", entries, coordinate="theta", angle="stored")
    assert set(np.round(model.series[0].x, 6)) == {270.0}
    assert model.coordinate_label == "Stored poloidal angle [deg]"


# ---------------------------------------------------------------------------
# options, discovery, rendering
# ---------------------------------------------------------------------------

def test_unknown_coordinate_layout_and_angle_are_refused_by_name(sample):
    with pytest.raises(ValueError, match="coordinate must be one of z, theta; got 'r'"):
        vaft.omas.plot_flux_loop_spatial_flux(sample, coordinate="r")
    with pytest.raises(ValueError, match="takes no layout="):
        vaft.omas.plot_flux_loop_spatial_flux(sample, layout="subplots")
    with pytest.raises(ValueError, match="angle must be one of geometric, stored"):
        vaft.omas.plot_flux_loop_spatial_flux(sample, coordinate="theta", angle="magnetic")
    with pytest.raises(ValueError, match="pass coordinate='theta' with them"):
        vaft.omas.plot_flux_loop_spatial_flux(sample, angle="stored")
    with pytest.raises(ValueError, match="pass coordinate='theta' with them"):
        vaft.omas.plot_flux_loop_spatial_flux(sample, centre=(0.3, 0.0))
    for bad in ((0.3,), "0.3,0.0", (0.3, float("nan")), (1, 2, 3)):
        with pytest.raises(ValueError, match="centre="):
            vaft.omas.plot_flux_loop_spatial_flux(sample, coordinate="theta", centre=bad)
    # The profile vocabulary is untouched by the spatial one.
    with pytest.raises(ValueError, match="coordinate must be one of rho_tor_norm"):
        vaft.omas.plot_equilibrium_profile_q(sample, coordinate="theta")


def test_discovery_states_coordinates_channels_times_and_controls(sample):
    record = next(r for r in vaft.omas.available_plots(sample) if r.name == "flux_loop_spatial_flux")
    assert record.view == "spatial"
    assert record.coordinates == {"default": "z", "options": ("z", "theta"), "declared": ("z", "theta")}
    assert record.channels["total"] == 11 and record.channels["regions"] == {"inboard": 7, "outboard": 4}
    assert record.times["count"] == 2500 and record.times["start"] == pytest.approx(0.26)
    assert record.layouts == ()
    assert record.times["option"] == "time_index" and record.times["selected"] == 1250
    assert record.controls == ("time_index", "selection", "channels", "yunit", "coordinate", "validity", "theme")
    text = str(vaft.omas.available_plots(sample, query="flux loop"))
    assert "coordinates: z (default) | theta" in text and "2500 samples" in text
    result = vaft.omas.plot_flux_loop_spatial_flux(sample, interactive=True, interaction_backend="none")
    result.state.set("coordinate", "theta")
    assert not hasattr(result.axes, "shape")


def test_both_backends_draw_both_coordinates(sample):
    figure, axes = vaft.omas.plot_flux_loop_spatial_flux(sample, time=0.31)
    assert axes.shape == (1, 2) and axes[0, 0].get_title() == "inboard"
    figure, axis = vaft.omas.plot_b_field_probe_spatial_field(sample, coordinate="theta")
    assert axis.get_xlabel().startswith("Poloidal angle θ [deg] about")
    plotly = vaft.omas.plot_flux_loop_spatial_flux(sample, backend="plotly")
    assert sum(1 for k in plotly.layout if k.startswith("xaxis")) == 2
    plotly = vaft.omas.plot_b_field_probe_spatial_field(sample, coordinate="theta", backend="plotly")
    assert len(plotly.data) == 1


def test_the_impa_profile_is_unchanged_by_the_shared_time_snap(sample):
    """The two inline argmin snaps in the IMPA builder now go through resolve_time_sample."""
    # IMPA is its own stage since #305: compose it onto the sample first.
    from _synthetic_inputs import make_impa_composed

    entries = normalize_entries(make_impa_composed(sample))
    for time in (None, 0.3):
        model = build_model("impa_profile_field", entries, **({"time": time} if time else {}))
        assert model.series and model.series[0].label == "IMPA measurement"
        assert np.all(np.isfinite(model.series[0].y))


def test_omas_and_imas_agree_on_both_plots(sample):
    from test_imas_omas_plot_equivalence import assert_models_equal
    from vaft.imas.access import IDSEntry

    with vaft.imas.load(vaft.data.sample(39915, representation="imas"), imas_version="3.41.0") as handle:
        entry = IDSEntry(handle)
        for name, options in (
            ("flux_loop_spatial_flux", {"time": 0.31}),
            ("flux_loop_spatial_flux", {"coordinate": "theta"}),
            ("b_field_probe_spatial_field", {"selection": "all"}),
        ):
            expected = build_model(name, [("39915", sample)], **options)
            actual = build_model(name, [("39915", entry)], **options)
            assert_models_equal(actual, expected)


def test_two_shots_are_drawn_together_with_their_own_time_stamps(sample):
    """The second entry's sample is never hidden behind the first one's stamp."""
    other = copy.deepcopy(sample)
    other["magnetics.time"] = np.asarray(sample["magnetics.time"]) + 0.02
    for index in range(11):
        other[f"magnetics.flux_loop.{index}.flux.time"] = np.asarray(sample[f"magnetics.flux_loop.{index}.flux.time"]) + 0.02
    entries = [("a", sample), ("b", other)]
    same = build_model("flux_loop_spatial_flux", entries, time=0.31)
    # Both shots resolve to the same stamp: said once.
    assert same.suptitle.count("nearest sample") == 1 and "a: " not in same.suptitle
    for panel in same.models:
        assert [trace.label for trace in panel.series] == ["a", "b"]
    default = build_model("flux_loop_spatial_flux", entries)
    assert "a: " in default.suptitle and "b: " in default.suptitle  # middle samples differ by 20 ms
    theta = build_model("flux_loop_spatial_flux", entries, coordinate="theta")
    assert len(theta.series) == 2


# -- issue #1380 phase 1: dense time_index navigation --------------------------------------


def _probe_values(model):
    return {
        float(x): float(y)
        for panel in model.models for trace in panel.series for x, y in zip(trace.x, trace.y)
    }


@pytest.mark.parametrize("index", [0, 1250, 2499])
def test_time_index_draws_that_stored_sample_on_every_channel(sample, entries, index):
    model = build_model("flux_loop_spatial_flux", entries, time_index=index, selection="all")
    stored = float(np.asarray(sample["magnetics.time"])[index])
    assert f"t = {stored * 1e3:.2f} ms (sample {index})" in model.suptitle
    assert "[mWb]" in model.suptitle  # the stored Wb, displayed in mWb
    expected = sorted(
        1e3 * float(np.asarray(sample[f"magnetics.flux_loop.{i}.flux.data"])[index]) for i in range(11)
    )
    drawn = sorted(y for panel in model.models for trace in panel.series for y in trace.y)
    assert drawn == pytest.approx(expected)


def test_time_and_its_time_index_draw_the_same_sample(sample, entries):
    stored = float(np.asarray(sample["magnetics.time"])[777])
    by_time = build_model("b_field_probe_spatial_field", entries, time=stored)
    by_index = build_model("b_field_probe_spatial_field", entries, time_index=777)
    assert _probe_values(by_time) == _probe_values(by_index)


@pytest.mark.parametrize(
    "options",
    [{"time": 0.31, "time_index": 3}, {"time": 0.31, "time_slice": 2}, {"time_slice": 2, "time_index": 3}],
)
def test_two_state_selectors_are_refused_not_ranked(entries, options):
    with pytest.raises(ValueError, match="takes one of time=, time_slice=, time_index="):
        build_model("flux_loop_spatial_flux", entries, **options)


@pytest.mark.parametrize(("value", "match"), [(2500, "outside the 2500 stored samples"), (-2501, "outside"), (1.5, "an int")])
def test_a_time_index_off_the_axis_is_refused(entries, value, match):
    with pytest.raises(ValueError, match=match):
        build_model("flux_loop_spatial_flux", entries, time_index=value)


def test_the_slider_starts_at_the_static_default_and_steps_the_samples(sample):
    result = vaft.omas.plot_b_field_probe_spatial_field(sample, interactive=True, interaction_backend="none")
    slider = next(c for c in result.controls if c.name == "time_index")
    assert slider.kind == "range" and slider.options == (0, 2499, 1)
    assert result.state["time_index"] == 1250
    static = build_model("b_field_probe_spatial_field", normalize_entries(sample))
    assert "(middle sample)" in static.suptitle
    assert _probe_values(build_model("b_field_probe_spatial_field", normalize_entries(sample), time_index=1250)) == (
        _probe_values(static)
    )
    result.state.set("time_index", 100)
    assert any("(sample 100)" in text.get_text() for text in result.figure.findobj(matplotlib.text.Text))
    assert result.state["coordinate"] == "z"  # stepping time leaves the other controls alone


def _shifted(sample, channel, offset):
    other = copy.deepcopy(sample)
    path = f"magnetics.flux_loop.{channel}.flux.time"
    other[path] = np.asarray(sample[path]) + offset
    return other


def test_channels_off_the_shared_grid_get_no_slider_and_no_time_index(sample):
    other = _shifted(sample, 3, 2e-5)  # half a 40 us sample
    record = next(r for r in vaft.omas.available_plots(other) if r.name == "flux_loop_spatial_flux")
    assert record.times["shared"] is False and "option" not in record.times
    assert "time_index" not in record.controls
    with pytest.raises(ValueError, match="do not share one time base"):
        build_model("flux_loop_spatial_flux", normalize_entries(other), time_index=10)


def test_a_time_on_heterogeneous_channels_states_the_spread(sample):
    other = _shifted(sample, 3, 2e-5)  # half a 40 us sample: its nearest stored time differs
    model = build_model("flux_loop_spatial_flux", normalize_entries(other), time=0.31)
    assert "channels sampled" in model.suptitle
    low, high = (float(v) for v in model.suptitle.split("channels sampled ")[1].split(" ms")[0].split("-"))
    assert high - low == pytest.approx(0.02, abs=0.005)


def test_omas_and_imas_agree_on_a_time_index(sample):
    from test_imas_omas_plot_equivalence import assert_models_equal
    from vaft.imas.access import IDSEntry
    from vaft.plot.backend.discovery import describe_one

    with vaft.imas.load(vaft.data.sample(39915, representation="imas"), imas_version="3.41.0") as handle:
        entry = IDSEntry(handle)
        for name in ("flux_loop_spatial_flux", "b_field_probe_spatial_field"):
            expected = build_model(name, [("39915", sample)], time_index=600)
            actual = build_model(name, [("39915", entry)], time_index=600)
            assert_models_equal(actual, expected)
            assert describe_one(name, [("39915", entry)]).times == describe_one(name, [("39915", sample)]).times


def test_an_animation_steps_the_magnetics_samples_not_another_time_base(sample):
    movie = vaft.omas.plot_flux_loop_spatial_flux(sample, time_index=range(0, 2500, 500), animation=True)
    axis = np.asarray(sample["magnetics.time"], dtype=float)
    assert movie.driver.values == tuple(float(axis[i]) for i in range(0, 2500, 500))


def test_a_negative_time_index_counts_from_the_end_as_on_the_pf_time_base(entries):
    last = build_model("flux_loop_spatial_flux", entries, time_index=2499)
    assert _probe_values(build_model("flux_loop_spatial_flux", entries, time_index=-1)) == _probe_values(last)


@pytest.mark.parametrize("pinned", [{"time": 0.3}, {"time_slice": 0}])
def test_an_instant_chosen_with_another_selector_pins_the_interactive_plot(sample, pinned):
    """A caller's time= used to collide with the new time_index slider on the first draw."""
    result = vaft.omas.plot_flux_loop_spatial_flux(sample, interactive=True, interaction_backend="none", **pinned)
    assert "time_index" not in [c.name for c in result.controls]
    result.state.set("coordinate", "theta")  # still redraws


def test_two_shots_of_different_lengths_get_no_shared_slider(sample):
    short = copy.deepcopy(sample)
    keep = 2000
    for path in ["magnetics.time"] + [f"magnetics.flux_loop.{i}.flux.{leaf}" for i in range(11) for leaf in ("time", "data")]:
        short[path] = np.asarray(sample[path])[:keep]
    for leaf in ("time", "data"):
        for i in range(64):
            path = f"magnetics.b_field_pol_probe.{i}.field.{leaf}"
            if path in sample:
                short[path] = np.asarray(sample[path])[:keep]
    record = next(
        r for r in vaft.omas.available_plots([sample, short]) if r.name == "flux_loop_spatial_flux"
    )
    assert record.times["shared"] is False and "option" not in record.times
    assert record.times["count"] == 2500
    assert "time_index" not in record.controls
