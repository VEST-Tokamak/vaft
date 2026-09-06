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
    assert record.controls == ("selection", "channels", "yunit", "coordinate", "validity")
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


def test_the_impa_profile_is_unchanged_by_the_shared_time_snap(entries):
    """The two inline argmin snaps in the IMPA builder now go through resolve_time_sample."""
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
