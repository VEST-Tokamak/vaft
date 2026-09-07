"""Issue #480: controls rebuild and redraw one plot; the state is the contract."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
from vaft.plot.controls import ControlSpec, controls_for
from vaft.plot.navigation import ControlState, SliceNavigator
from vaft.plot.renderers.interactive import Interactive, render_controls


@pytest.fixture(scope="module")
def sample():
    return vaft.omas.load(vaft.data.sample(39915, representation="omas"))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# ControlState
# ---------------------------------------------------------------------------

def _specs():
    return (
        ControlSpec("time_slice", "choice", "Slice", 2, (0, 2, 4), ("a", "b", "c"), group="slice"),
        ControlSpec("layout", "choice", "Layout", "overlay", ("overlay", "subplots")),
        ControlSpec("channels", "multi", "Channels", (), (0, 1, 2), group="selection"),
        ControlSpec("synthetic", "choice", "Overlay", "none", ("none", "equilibrium")),
        ControlSpec("validity", "choice", "Flagged", "show", ("show", "mask"), group="style"),
        ControlSpec("dark", "toggle", "Dark", False),
    )


def test_the_state_starts_at_the_defaults_validates_and_notifies_once_per_change():
    state = ControlState(_specs())
    assert state.values == {"time_slice": 2, "layout": "overlay", "channels": (), "synthetic": "none", "validity": "show", "dark": False}
    seen = []
    unsubscribe = state.subscribe(lambda s: seen.append(s.values["layout"]))
    assert state.set("layout", "subplots") is True and seen == ["subplots"]
    assert state.set("layout", "subplots") is False and seen == ["subplots"]  # no change, no call
    with pytest.raises(ValueError, match="layout must be one of"):
        state.set("layout", "stacked")
    with pytest.raises(KeyError, match="no control named 'gain'"):
        state.set("gain", 1)
    assert state.update(layout="overlay", dark=True) is True and seen == ["subplots", "overlay"]
    unsubscribe()
    state.set("layout", "subplots")
    assert seen == ["subplots", "overlay"]


def test_as_options_leaves_out_none_and_style_and_lets_channels_replace_the_preset():
    state = ControlState(_specs(), {"channels": [2, 0]})
    assert state.as_options() == {"time_slice": 2, "layout": "overlay", "selection": [2, 0], "dark": False}
    assert state.as_style() == {"validity": "show"}
    state.update(channels=(), synthetic="equilibrium")
    assert state.as_options() == {"time_slice": 2, "layout": "overlay", "synthetic": "equilibrium", "dark": False}


def test_a_slice_control_and_a_navigator_follow_each_other():
    state = ControlState(_specs())
    navigator = SliceNavigator([0.1, 0.2, 0.3, 0.4, 0.5], usable=[0, 2, 4], initial=2)
    state.bind_navigator(navigator)
    state.set("time_slice", 4)
    assert navigator.selected == 4
    navigator.select_index(0)
    assert state["time_slice"] == 0


# ---------------------------------------------------------------------------
# render_controls
# ---------------------------------------------------------------------------

def _line_setup(sample):
    from vaft.omas.entries import normalize_entries
    from vaft.plot.backend.recipes import build_model
    from vaft.plot.backends import renderer_for
    from vaft.plot.registry import get_spec

    entries = normalize_entries(sample)
    spec = get_spec("flux_loop_time_flux")
    record = next(r for r in vaft.omas.available_plots(sample) if r.name == "flux_loop_time_flux")
    calls: list[dict] = []

    def build(options):
        calls.append(dict(options))
        return build_model("flux_loop_time_flux", entries, **options)

    def draw(model, **kwargs):
        return renderer_for(spec, model, "matplotlib")(model, **kwargs)

    def draw_plotly(model, **kwargs):
        return renderer_for(spec, model, "plotly")(model, **kwargs)

    return record, build, draw, draw_plotly, calls


def test_a_change_rebuilds_the_model_and_relays_out_the_axes(sample):
    record, build, draw, _, calls = _line_setup(sample)
    result = render_controls(build, ControlState(controls_for(record)), draw=draw, backend="none")
    assert isinstance(result, Interactive)
    figure, axes, state = result
    assert figure is result.figure and axes is result.axes and state is result.state
    assert len(calls) == 1 and "validity" not in calls[-1]  # a style keyword never reaches the builder
    assert not hasattr(axes, "shape")
    state.set("layout", "subplots")
    assert len(calls) == 2 and calls[-1]["layout"] == "subplots"
    assert result.axes.shape == (11,)
    state.set("channels", [0, 3])
    assert calls[-1]["selection"] == [0, 3] and result.axes.shape == (2,)
    assert result.widget is None


def test_style_controls_reach_the_renderer_not_the_builder(sample, monkeypatch):
    record, build, draw, _, calls = _line_setup(sample)
    seen = []

    def spy(model, **kwargs):
        seen.append(dict(kwargs))
        return draw(model, **kwargs)

    result = render_controls(build, ControlState(controls_for(record)), draw=spy, backend="none")
    assert seen[-1]["validity"] == "show"
    result.state.set("validity", "mask")
    assert seen[-1]["validity"] == "mask" and "validity" not in calls[-1]


def test_the_matplotlib_strip_drives_the_state_and_follows_it(sample):
    record, build, draw, _, calls = _line_setup(sample)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Agg cannot update in place
        result = render_controls(build, ControlState(controls_for(record)), draw=draw, backend="matplotlib")
    widgets = result.widget
    names = [c.name for c in result.controls]
    assert [type(w).__name__ for w in widgets] == ["RadioButtons", "CheckButtons"] + ["RadioButtons"] * 5
    layout = widgets[names.index("layout")]
    layout.set_active(1)
    assert result.state["layout"] == "subplots" and calls[-1]["layout"] == "subplots"
    channels = widgets[names.index("channels")]
    channels.set_active(0)
    assert result.state["channels"] == (0,) and calls[-1]["selection"] == [0]
    assert result.figure.get_axes() and len(result.figure.subfigs) == 2


def test_plotly_figures_are_rebuilt_and_a_window_strip_is_refused(sample):
    record, build, _, draw_plotly, calls = _line_setup(sample)
    result = render_controls(build, ControlState(controls_for(record)), draw=draw_plotly, backend="none", render_backend="plotly")
    assert type(result.figure).__name__ == "Figure" and len(result.figure.data) == 11
    result.state.set("selection", "outboard")
    assert calls[-1]["selection"] == "outboard" and len(result.figure.data) == 4
    with pytest.raises(ValueError, match="cannot host Matplotlib window widgets"):
        render_controls(build, ControlState(controls_for(record)), draw=draw_plotly, backend="matplotlib", render_backend="plotly")


def test_the_ipywidgets_box_redraws_matplotlib_and_redisplays_plotly(sample):
    pytest.importorskip("ipywidgets")
    import IPython.display as ipd

    record, build, draw, draw_plotly, calls = _line_setup(sample)
    shown = []
    original = ipd.display

    def spy(*objs, **kwargs):
        shown.extend(type(o).__name__ for o in objs)
    ipd.display = spy
    try:
        result = render_controls(build, ControlState(controls_for(record)), draw=draw, backend="ipywidgets")
        box = result.widget
        assert type(box).__name__ == "VBox" and shown[-1] == "VBox" and "Image" in shown
        dropdown = next(w for w in box.vaft_widgets if w.vaft_control == "layout")
        dropdown.value = "subplots"
        assert result.state["layout"] == "subplots" and calls[-1]["layout"] == "subplots"
        result.state.set("layout", "overlay")
        assert dropdown.value == "overlay"  # the widget follows a change from code
        shown.clear()
        plotly = render_controls(build, ControlState(controls_for(record)), draw=draw_plotly, backend="ipywidgets", render_backend="plotly")
        assert "Figure" in shown and shown[-1] == "VBox"
        plotly.state.set("selection", "all")
        assert len(plotly.figure.data) == 11
    finally:
        ipd.display = original


# ---------------------------------------------------------------------------
# the public entry point
# ---------------------------------------------------------------------------

def test_interactive_true_on_a_plot_adapter_returns_the_controls_of_its_record(sample):
    result = vaft.omas.plot_flux_loop_time_flux(sample, interactive=True, interaction_backend="none", layout="subplots", legend=False)
    assert isinstance(result, Interactive)
    assert [c.name for c in result.controls] == ["selection", "channels", "layout", "yunit", "x", "orientation", "validity"]
    assert result.state["layout"] == "subplots"  # the option given is the starting value
    result.state.set("selection", "outboard")
    assert result.axes.shape == (4,)
    only = vaft.omas.plot_flux_loop_time_flux(sample, interactive=True, interaction_backend="none", controls="layout")
    assert [c.name for c in only.controls] == ["layout"]
    with pytest.raises(ValueError, match="offers no control named 'gain'"):
        vaft.omas.plot_flux_loop_time_flux(sample, interactive=True, controls=["gain"])
    with pytest.raises(TypeError, match="takes no ax="):
        vaft.omas.plot_flux_loop_time_flux(sample, interactive=True, ax=plt.gca())


def test_slice_indexed_plots_offer_the_stored_slices(sample):
    psi = vaft.omas.plot_equilibrium_field_psi(sample, interactive=True, interaction_backend="none")
    assert [c.name for c in psi.controls] == ["time_slice", "units", "overlay", "style"]
    slices = psi.controls[0]
    record = next(r for r in vaft.omas.available_plots(sample) if r.name == "equilibrium_field_psi")
    assert slices.options == tuple(record.slices["usable"]) and slices.default == record.slices["selected"]
    assert slices.labels[4] == "4: 320.0 ms"
    psi.state.update(time_slice=2, units="Wb/rad", style="normalized")
    overview = vaft.omas.plot_equilibrium_overview(sample, interactive=True, interaction_backend="none")
    assert [c.name for c in overview.controls] == ["time_slice", "units", "style"]
    overview.state.set("time_slice", 1)
    assert overview.axes.shape == (7,)


def test_unknown_options_are_refused_by_every_adapter(sample):
    with pytest.raises(ValueError, match="does not take an option named 'selecton'"):
        vaft.omas.plot_flux_loop_time_flux(sample, selecton="active")
    with pytest.raises(ValueError, match="does not take an option named 'convention'"):
        vaft.omas.plot_equilibrium_field_psi(sample, convention="Wb/rad")
    with pytest.raises(ValueError, match="layout must be one of"):
        vaft.omas.plot_flux_loop_time_flux(sample, layout="stacked")


def test_omas_and_imas_offer_the_same_controls(sample):
    from vaft.imas.access import IDSEntry
    from vaft.plot.backend.discovery import describe_one

    with vaft.imas.load(vaft.data.sample(39915, representation="imas"), imas_version="3.41.0") as handle:
        entry = IDSEntry(handle)
        for name in ("flux_loop_time_flux", "equilibrium_field_psi", "equilibrium_profile_q", "mirnov_spectrogram"):
            from_omas = controls_for(describe_one(name, [("shot", sample)]))
            from_imas = controls_for(describe_one(name, [("shot", entry)]))
            assert from_omas == from_imas, name


def test_discovery_names_the_controls_and_the_entry_point(sample):
    record = next(r for r in vaft.omas.available_plots(sample) if r.name == "flux_loop_time_flux")
    assert "controls" in record.interaction
    assert record.interaction_entry_points["controls"] == "plot_flux_loop_time_flux(..., interactive=True)"
    assert record.controls == ("selection", "channels", "layout", "yunit", "x", "orientation", "validity")
    assert "controls: selection, channels" in str(vaft.omas.available_plots(sample, query="flux loop"))


def test_static_plotting_still_imports_no_widget_toolkit():
    import subprocess
    import sys

    # matplotlib.pyplot imports matplotlib.widgets itself, so ipywidgets is
    # the witness (as in test_equilibrium_interactive).
    code = (
        "import sys, vaft.plot.controls, vaft.plot.backend.options, vaft.plot.navigation, "
        "vaft.plot.renderers.interactive; print('ipywidgets' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip() == "False"


def test_a_time_history_offers_no_slice_control(sample):
    """A LineRecipe over every slice is a history: time_slice= means nothing to it."""
    from vaft.plot.backend.discovery import describe_one

    record = describe_one("equilibrium_time_plasma_current", [("shot", sample)])
    assert record.slices == {}
    assert "time_slice" not in [c.name for c in controls_for(record)]
    profile = describe_one("equilibrium_profile_pressure", [("shot", sample)])
    assert [c.name for c in controls_for(profile)][0] == "time_slice"


def test_controls_accepts_a_list_and_the_imas_adapter_offers_the_same(sample):
    from vaft.imas.access import IDSEntry

    subset = vaft.omas.plot_flux_loop_time_flux(sample, interactive=True, interaction_backend="none", controls=["layout", "yunit"])
    assert [c.name for c in subset.controls] == ["layout", "yunit"]
    subset.state.update(layout="subplots", yunit="Wb")
    assert subset.axes.shape == (11,)
    with vaft.imas.load(vaft.data.sample(39915, representation="imas"), imas_version="3.41.0") as handle:
        result = vaft.imas.plot_flux_loop_time_flux(handle, interactive=True, interaction_backend="none")
        assert [c.name for c in result.controls] == ["selection", "channels", "layout", "yunit", "x", "orientation", "validity"]
        result.state.set("selection", "outboard")
        assert result.axes is not None


def test_a_panel_with_a_field_keeps_its_colorbar_in_its_own_cell(sample):
    result = vaft.omas.plot_equilibrium_overview(sample, interactive=True, interaction_backend="none")
    canvas = result.figure.subfigs[0]
    before = len(canvas.axes)
    width = result.axes[0].get_position().width
    result.state.set("time_slice", 1)
    assert len(result.figure.subfigs[0].axes) == before
    assert result.axes[0].get_position().width == pytest.approx(width)
