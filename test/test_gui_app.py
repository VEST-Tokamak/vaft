"""The Panel application over a :class:`BrowserSession` (#1086); never served here."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest

from vaft.gui import app as gui_app
from vaft.gui.state import BrowserSession, Source
from vaft.gui.widgets import panel_controls

pn = pytest.importorskip("panel")

#: name -> the backends its (stub) capability record declares
_PLOTS = {
    "equilibrium_field_psi": ("matplotlib", "plotly"),
    "equilibrium_profile_q": ("matplotlib", "plotly"),
    "plasma_current_time": ("matplotlib",),
    "flux_loop_time_voltage": ("matplotlib", "plotly"),
}


class _Session(BrowserSession):
    """The real session with a short catalog, so no test pays for discovery."""

    def _discover(self, data):
        return [
            SimpleNamespace(name=name, subject=name.split("_")[0], backends=backends)
            for name, backends in _PLOTS.items()
        ]


@pytest.fixture(scope="module")
def ods():
    from vaft.omas import sample_ods

    return sample_ods(39915)


@pytest.fixture(scope="module")
def other_ods():
    from vaft.omas import sample_ods

    return sample_ods(41524)


@pytest.fixture
def app(ods, other_ods, monkeypatch):
    by_shot = {39915: ods, 41524: other_ods}
    monkeypatch.setattr("vaft.gui.state.load_source", lambda source: by_shot[source.value])
    built = gui_app.BrowserApp(_Session())
    assert built.load(Source("sample", 39915))
    yield built
    built.close()


def _label(widget):
    """A widget's label; a laid-out group (the check boxes) carries it as its name."""
    return widget.label if "label" in widget.param else widget.name


def _widget(app, label):
    return next(w for w in app.controls.objects if _label(w) == label)


def test_loading_fills_the_plot_selector_and_draws_the_first(app):
    # {subject: {label: canonical name}}; the stub records carry no view, so a label is the name
    assert {group: list(options.values()) for group, options in app.plot.groups.items()} == {
        "equilibrium": ["equilibrium_field_psi", "equilibrium_profile_q"],
        "plasma": ["plasma_current_time"], "flux": ["flux_loop_time_voltage"],
    }
    assert app.session.plot == "equilibrium_field_psi"
    assert app.session.renderer == "plotly" and app.interactive.visible and not app.static.visible
    import plotly.graph_objects as go

    assert app.figure is app.interactive and isinstance(app.figure.object, go.Figure)
    assert not app.renderer.disabled
    labels = [_label(w) for w in app.controls.objects]
    assert "Equilibrium slice" in labels
    assert not app.alert.visible


def test_choosing_a_plot_rebuilds_the_controls(app):
    app.plot.value = "plasma_current_time"
    assert app.session.plot == "plasma_current_time"
    assert "Equilibrium slice" not in [_label(w) for w in app.controls.objects]
    # no Plotly renderer declared: the static image, and nothing to switch to
    assert app.session.renderer == "matplotlib" and app.static.visible and not app.interactive.visible
    assert app.renderer.disabled and app.static.object is app.session.figure


def test_the_renderer_toggle_keeps_the_reader_s_choices(app):
    state = app.session.state
    chosen = state.spec("time_slice").options[0]
    unit = next(v for v in state.spec("units").options if v != state["units"])
    state.update(time_slice=chosen, units=unit)
    app.renderer.value = "matplotlib"
    assert app.session.renderer == "matplotlib" and app.static.visible
    assert app.session.state["time_slice"] == chosen and app.session.state["units"] == unit
    app.renderer.value = "plotly"
    assert app.session.renderer == "plotly" and app.interactive.object is not None


def test_a_widget_change_reaches_the_state_and_the_pane(app, monkeypatch):
    refreshed = []
    monkeypatch.setattr(app, "_refresh", lambda: refreshed.append(True))
    app.show("equilibrium_field_psi")
    refreshed.clear()
    slider = _widget(app, "Equilibrium slice")
    other = next(v for v in slider.values if v != slider.value)
    slider.value = other
    assert app.session.state["time_slice"] == other
    assert refreshed


def test_a_change_from_code_moves_the_widget(app):
    state = app.session.state
    control = state.spec("time_slice")
    other = next(v for v in control.options if v != state["time_slice"])
    state.set("time_slice", other)
    assert _widget(app, "Equilibrium slice").value == other


def test_a_refused_build_shows_the_reason_and_keeps_the_values(app):
    state = app.session.state
    before = state.values
    errors = []
    widgets = panel_controls(state, on_error=errors.append)
    real_set = state.set

    def refuse(name, value, **kwargs):
        raise ValueError("cannot draw that")

    state.set = refuse
    try:
        units = next(w for w in widgets if _label(w) == state.spec("units").label)
        units.value = next(v for v in units.values if v != units.value)
    finally:
        state.set = real_set
    assert errors and "cannot draw that" in str(errors[0])
    assert state.values == before
    assert units.value == before["units"], "the widget moves back"


def test_a_failed_load_reports_and_keeps_nothing_half_open(monkeypatch):
    def fail(source):
        raise FileNotFoundError("no such file: /nowhere.json")

    monkeypatch.setattr("vaft.gui.state.load_source", fail)
    built = gui_app.BrowserApp(_Session())
    assert not built.load(Source("file", "/nowhere.json"))
    assert built.alert.visible and "FileNotFoundError" in built.alert.object
    assert built.session.ods is None and built.plot.disabled


def test_the_page_is_a_template(app):
    assert isinstance(app.view(), pn.template.FastListTemplate)


def test_build_app_takes_one_source(monkeypatch):
    with pytest.raises(ValueError, match="at most one"):
        gui_app.build_app(sample=39915, shot=39915)
    loaded = []
    monkeypatch.setattr(gui_app.BrowserApp, "load", lambda self, source: loaded.append(source))
    built = gui_app.build_app(shot=[41524, 41672], namespace="main", plot="plasma_current_time")
    assert built.kind.value == "shot" and built.shots.value == "41524, 41672"
    assert built.requested_sources() == [Source("shot", 41524, "main"), Source("shot", 41672, "main")]
    # outside a server, pn.state.onload runs at once: the load was asked for
    assert loaded == [[Source("shot", 41524, "main"), Source("shot", 41672, "main")]]
    built = gui_app.build_app(sample=[39915, 41524])
    assert built.sample.value == [39915, 41524]


def test_shots_are_parsed_from_free_text():
    assert gui_app.parse_shots(" 39915, 41524;41672  45531 ") == [39915, 41524, 41672, 45531]
    assert gui_app.parse_shots("") == []
    with pytest.raises(ValueError, match="whole numbers"):
        gui_app.parse_shots("39915-39920")


def test_several_shots_are_compared_on_one_plot(app):
    assert app.load([Source("sample", 39915), Source("sample", 41524)])
    assert app.session.label == "samples 39915, 41524"
    assert isinstance(app.session.ods, list) and len(app.session.ods) == 2
    assert app.session.plot == "equilibrium_field_psi", "the plot on screen is kept"
    app.plot.value = "plasma_current_time"
    lines = [line for ax in app.session.figure.axes for line in ax.get_lines()]
    assert len(lines) >= 2, "one trace per shot"


def test_a_preset_after_individual_channels_is_applied(app):
    app.plot.value = "flux_loop_time_voltage"
    state = app.session.state
    preset = _widget(app, "Channels")
    boxes = _widget(app, "Individual channels").objects[1]
    assert isinstance(boxes, pn.widgets.CheckBoxGroup), "one click toggles one channel"
    boxes.value = [0, 2]
    assert state["channels"] == (0, 2)
    assert preset.value is not state["selection"] and preset.value is not None, "shows (individual channels)"
    wanted = next(v for v in state.spec("selection").options if v != state["selection"])
    preset.value = wanted
    assert state["selection"] == wanted and state["channels"] == ()
    assert boxes.value == [] and state.as_options()["selection"] == wanted


def _option(app, name, *values):
    """Set the Figure Options form's widgets for field ``name``, as a reader would."""
    for widget, value in zip(app.options_form.widgets[name], values):
        widget.value = value


def test_figure_options_reach_both_renderers(app):
    _option(app, "xlim", 0.2, 0.6)
    assert app.session.figure_options.xlim == (0.2, 0.6)
    assert list(app.interactive.object.layout.xaxis.range) == [0.2, 0.6]
    assert app.interactive.object.layout.uirevision.startswith("equilibrium_field_psi|")
    app.width.value = 640
    assert app.interactive.width == 640 and app.interactive.sizing_mode == "fixed"
    app.plot.value = "plasma_current_time"
    xlim = [ax.get_xlim() for ax in app.session.figure.axes if ax.get_label() != "<colorbar>"]
    assert xlim and all(lim == (0.2, 0.6) for lim in xlim)
    app.options_form.reset_button.clicks += 1
    assert not app.session.figure_options and app.options_form.widgets["xlim"][0].value is None
    assert all(ax.get_xlim() != (0.2, 0.6) for ax in app.session.figure.axes)
    app.width.value = None
    assert app.static.sizing_mode == "stretch_width"


def test_an_impossible_limit_is_reported_not_applied(app):
    _option(app, "xlim", 0.6, 0.6)
    assert app.alert.visible and "low must be below high" in app.alert.object
    assert app.session.figure_options.xlim == (0.6, None)


def test_the_figure_downloads_as_the_matplotlib_rendering(app):
    _option(app, "xlim", None, 0.33)
    assert app.download.filename == "equilibrium_field_psi_sample_39915.png"
    png = app._export().getvalue()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    app.export_format.value = "svg"
    assert app.download.filename.endswith(".svg")
    assert b"<svg" in app._export().getvalue()[:400]


def test_serve_binds_loopback_and_warns_otherwise(monkeypatch):
    calls = []
    monkeypatch.setattr(pn, "serve", lambda panels, **kwargs: calls.append(kwargs))
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    gui_app.serve(port=5123)
    assert calls[-1]["address"] == "127.0.0.1" and calls[-1]["show"] is True
    assert "localhost:5123" in calls[-1]["websocket_origin"]
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 1 10.0.0.2 22")
    with pytest.warns(UserWarning, match="HTTPS"):
        gui_app.serve(address="0.0.0.0", port=5123)
    assert calls[-1]["show"] is False


def test_the_renderer_toggle_keeps_individual_channels_and_their_preset(app):
    app.plot.value = "flux_loop_time_voltage"
    state = app.session.state
    state.set("channels", [0, 2])
    app.renderer.value = "matplotlib"
    state = app.session.state
    assert [c.name for c in state.controls][:2] == ["selection", "channels"], "the preset control survives"
    assert state["channels"] == (0, 2)
    state.set("channels", [])
    assert state.as_options()["selection"] == state["selection"], "unticking returns to the preset"


def test_a_figure_option_keeps_the_controls_of_a_static_plot(app):
    app.plot.value = "plasma_current_time"
    unit = next(v for v in app.session.state.spec("yunit").options if v != app.session.state["yunit"])
    app.session.state.set("yunit", unit)
    _option(app, "xlim", None, 0.33)
    assert app.session.state["yunit"] == unit
    assert all(ax.get_xlim()[1] == 0.33 for ax in app.session.figure.axes)


def test_a_log_axis_refuses_a_non_positive_limit_and_keeps_drawing(app):
    _option(app, "yscale", "log")
    _option(app, "ylim", 0.0, None)
    assert app.alert.visible and "positive ylim" in app.alert.object
    assert app.session.figure_options.ylim is None, "the figure keeps the accepted options"
    assert app.options_form.widgets["ylim"][0].value == 0.0, "the box keeps what was typed, to be corrected"
    state = app.session.state
    other = next(v for v in state.spec("time_slice").options if v != state["time_slice"])
    state.set("time_slice", other)
    assert state["time_slice"] == other, "later control changes still redraw"


def test_a_catalog_failure_keeps_the_previous_sources_on_screen(app, monkeypatch):
    before = (app.session.label, app.session.plot, app.session.figure)

    def fail(_data):
        raise RuntimeError("catalog exploded")

    monkeypatch.setattr(app.session, "_discover", fail)
    assert not app.load([Source("sample", 41524)])
    assert (app.session.label, app.session.plot, app.session.figure) == before
    assert "Still showing **sample 39915**" in app.status.object
    assert not app.download.disabled


def test_sources_with_no_plot_leave_nothing_bound(app, monkeypatch):
    monkeypatch.setattr(app.session, "_discover", lambda _data: [])
    assert app.load([Source("sample", 41524)])
    assert app.session.plot is None and app.controls.objects == []
    assert app.download.disabled and app.plot.disabled and app.renderer.disabled
    assert not app.interactive.visible and not app.static.visible


def test_the_static_choice_is_remembered_for_the_next_plot(app):
    app.renderer.value = "matplotlib"
    app.plot.value = "equilibrium_profile_q"
    assert app.session.renderer == "matplotlib"
    app.renderer.value = "plotly"
    app.plot.value = "equilibrium_field_psi"
    assert app.session.renderer == "plotly"


def test_serve_protects_a_shared_host(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(pn, "serve", lambda panels, **kwargs: calls.append(kwargs))
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("VAFT_GUI_PASSWORD", raising=False)
    gui_app.serve(port=5123)
    assert "basic_auth" not in calls[-1], "a local session needs no password"
    assert {"localhost:5123", "127.0.0.1:5123"} <= set(calls[-1]["websocket_origin"])
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 1 10.0.0.2 22")
    gui_app.serve(port=5123)
    printed = capsys.readouterr().out
    assert calls[-1]["basic_auth"] and calls[-1]["basic_auth"] in printed
    monkeypatch.setenv("VAFT_GUI_PASSWORD", "chosen")
    gui_app.serve(port=5123)
    assert calls[-1]["basic_auth"] == "chosen" and "chosen" not in capsys.readouterr().out
    gui_app.serve(port=5123, auth="none")
    assert "basic_auth" not in calls[-1]
    with pytest.raises(ValueError, match="auth must be one of"):
        gui_app.serve(auth="token")


def test_options_the_figure_refuses_are_rolled_back(app, monkeypatch):
    from vaft.plot import FigureOptions

    kept = app.session.figure_options
    real = FigureOptions.apply_plotly

    def refuse(self, figure):
        if self.xlim == (None, 0.5):
            raise ValueError("refused by the figure")
        return real(self, figure)

    monkeypatch.setattr(FigureOptions, "apply_plotly", refuse)
    assert app.apply_options(FigureOptions(xlim=(None, 0.5))) is False
    assert app.session.figure_options == kept and "refused by the figure" in app.alert.object
    state = app.session.state
    other = next(v for v in state.spec("time_slice").options if v != state["time_slice"])
    state.set("time_slice", other)
    assert state["time_slice"] == other, "the plot is not frozen by the refused options"


def test_file_paths_one_per_line_and_the_server_browser(app, tmp_path, monkeypatch):
    app.kind.value = "file"
    assert app.browse in app._source_inputs.objects and "GEQDSK" in app.formats.object
    app.path.value = " /data/a.json \n\n/data/g012345.00300\n"
    assert app.requested_sources() == [Source("file", "/data/a.json"), Source("file", "/data/g012345.00300")]
    app.path.value = ""
    app.load_button.clicks += 1
    assert app.alert.visible and "file path" in app.alert.object

    monkeypatch.chdir(tmp_path)
    (tmp_path / "eq.json").write_text("{}")
    assert app.browser is None, "nothing is listed before it is asked for"
    app.browse.value = True
    assert isinstance(app.browser, pn.widgets.FileSelector) and app.browser_box.visible
    loaded = []
    monkeypatch.setattr(app, "load", lambda sources: loaded.append(sources))
    app.use_selected.clicks += 1
    assert not loaded and "no file selected" in app.alert.object
    app.browser.value = [str(tmp_path / "eq.json")]
    app.use_selected.clicks += 1
    assert loaded == [[Source("file", str(tmp_path / "eq.json"))]]
    # load is stubbed here; closing after a real load is tested below
    assert app.path.value == str(tmp_path / "eq.json")


def test_uploaded_files_are_stored_per_upload_and_loaded(app):
    from pathlib import Path

    loaded = []
    app.load = lambda sources: loaded.append(sources)
    app.upload.param.update(filename=["../escape.json", "g041524.00300"], value=[b"{}", b"g"])
    first = loaded[-1]
    assert [Path(s.value).name for s in first] == ["escape.json", "g041524.00300"]
    folder = Path(first[0].value).parent
    assert folder.name == "upload-1" and (folder / "escape.json").read_bytes() == b"{}"
    assert not (folder.parent / "escape.json").exists(), "a browser path cannot leave the folder"
    assert first[0].label == "upload-1/escape.json"

    # an IMAS entry uploaded together is one source: its folder
    app.upload.param.update(filename=["master.h5", "equilibrium.h5"], value=[b"m", b"e"])
    (entry,) = loaded[-1]
    assert Path(entry.value).name == "upload-2" and (Path(entry.value) / "master.h5").is_file()

    root = Path(app._uploads.name)
    app.close()
    assert not root.exists(), "uploads go with the browser session"


def test_the_file_browser_closes(app, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    app.kind.value = "file"
    app.browse.value = True
    assert app.browser_box.visible and app.browse.label == "Hide server files"
    app.close_browser.clicks += 1
    assert not app.browser_box.visible and not app.browse.value
    assert app.browse.label == "Browse server files"
    app.browse.value = True
    assert app.load(Source("sample", 39915))
    assert not app.browser_box.visible, "a successful load closes it"


def test_serve_accepts_uploads_larger_than_bokeh_s_default(monkeypatch):
    calls = []
    monkeypatch.setattr(pn, "serve", lambda panels, **kwargs: calls.append(kwargs))
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    gui_app.serve(port=5123)
    assert calls[-1]["websocket_max_message_size"] == gui_app.MAX_UPLOAD_BYTES > 20 * 1024 * 1024


def test_a_database_shot_shows_what_is_in_memory(monkeypatch, ods):
    import omas

    import vaft.database
    from vaft.omas import available_plots

    catalog = available_plots(ods)
    monkeypatch.setattr(vaft.database, "available_plots", lambda shot, source=None, **_: catalog)

    def fake_load(shot, source=None, *, paths=None, **_):
        part = omas.ODS()
        for name in paths:
            if name in ods.keys():
                part[name] = ods[name]
        return part

    monkeypatch.setattr(vaft.database, "load", fake_load)
    monkeypatch.setattr(vaft.database, "stored_ids", lambda shot, source=None: tuple(ods.keys()))
    built = gui_app.BrowserApp(plot="plasma_current_time")
    built.kind.value = "shot"
    built.shots.value = "39915"
    built.load_button.clicks += 1
    assert built.session.database and built.session.plot == "plasma_current_time"
    assert "In memory:" in built.status.object and "magnetics" in built.status.object
    assert not built.ids_button.disabled and "pf_active" in built.ids_choice.options
    built.ids_choice.value = ["pf_active"]
    built.ids_button.clicks += 1
    assert "pf_active" in built.status.object and built.ids_choice.value == []
    built.close()


def _player_of(app, label):
    box = next(w for w in app.controls.objects if _label(w) == f"Play: {label}")
    return box.objects[0], box.objects[1]


def test_equilibrium_slices_play_as_a_movie(app):
    state = app.session.state
    options = list(state.spec("time_slice").options)
    player, interval = _player_of(app, "Equilibrium slice")
    assert isinstance(player, pn.widgets.Player) and (player.start, player.end) == (0, len(options) - 1)
    frames = []
    state.subscribe(lambda s: frames.append(s["time_slice"]))
    player.value = 0
    player.value = 1
    assert frames == [options[0], options[1]] and _widget(app, "Equilibrium slice").value == options[1]
    state.set("time_slice", options[2])
    assert player.value == 2, "the player follows a slice chosen by hand"
    interval.value = 200
    assert player.interval == 200
    player.interval = 400  # the player's own "slower" button
    assert interval.value == 400


def test_camera_frames_play_through_their_time_control():
    from vaft.gui.widgets import panel_controls
    from vaft.omas import render_plot, sample_ods

    drawn = render_plot(
        "camera_visible_image_frame", sample_ods(40600), interactive=True, interaction_backend="none",
    )
    laid_out = panel_controls(drawn.state)
    box = next(w for w in laid_out if _label(w).startswith("Play: "))
    player = box.objects[0]
    low, high, _ = drawn.state.spec("frame_index").options
    assert (player.start, player.end) == (low, high)
    player.value = 10
    assert drawn.state["frame_index"] == 10
    import matplotlib.pyplot as plt

    plt.close(drawn.figure)


def test_a_frame_that_cannot_be_drawn_is_reported_and_the_next_one_clears_it(app):
    state = app.session.state
    options = list(state.spec("time_slice").options)
    player, _ = _player_of(app, "Equilibrium slice")
    player.value = len(options) - 1  # sample slice 8 cannot be drawn as surfaces
    assert app.alert.visible and "style='surfaces'" in app.alert.object
    player.value = 0
    assert not app.alert.visible and state["time_slice"] == options[0]


def test_playback_walks_past_a_frame_that_cannot_be_drawn(app):
    state = app.session.state
    options = list(state.spec("time_slice").options)
    player, _ = _player_of(app, "Equilibrium slice")
    player.value = len(options) - 2
    good = state["time_slice"]
    player.value = len(options) - 1  # refused: sample slice 8 cannot be drawn as surfaces
    assert app.alert.visible and state["time_slice"] == good
    assert player.value == len(options) - 1, "the player keeps its place, so the next tick moves on"
    player.value = 0  # the loop wraps
    assert state["time_slice"] == options[0] and not app.alert.visible


def test_a_refused_first_draw_leaves_no_pyplot_figure(app):
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    for _ in range(3):
        with pytest.raises(ValueError):
            app.session.select("equilibrium_field_psi", renderer="matplotlib", time_slice=8)
    assert set(plt.get_fignums()) == before


def test_unticking_every_overlay_survives_a_redraw(app):
    state = app.session.state
    state.set("overlay", ())
    assert state.as_options()["overlay"] == (), "an empty choice reaches the builder"
    app.renderer.value = "matplotlib"
    assert app.session.state["overlay"] == ()


def test_a_refused_option_stays_typed_but_blocks_no_other_edit(app):
    _option(app, "xlim", 0.5, None)
    _option(app, "xlim", 0.5, 0.2)
    assert app.alert.visible and app.options_form.widgets["xlim"][1].value == 0.2, "kept as typed"
    assert app.session.figure_options.xlim == (0.5, None)
    _option(app, "title", "next")
    assert app.session.figure_options.title == "next", "the next edit is not blocked by the refused value"
    assert app.options_form.widgets["xlim"][1].value is None, "the refused value went back"


def test_a_range_moves_past_its_other_end_low_first(app):
    _option(app, "xlim", 0.0, 0.1)
    _option(app, "xlim", 0.2)  # low past high: refused for now
    assert app.alert.visible and app.session.figure_options.xlim == (0.0, 0.1)
    _option(app, "xlim", 0.2, 0.3)
    assert app.session.figure_options.xlim == (0.2, 0.3) and not app.alert.visible


def test_reproduce_without_a_drawing_reports_instead_of_raising(app):
    app._clear_plot(keep_selector=True)
    app.session.close()
    app.reproduce.text.value = "stale"
    app.reproduce.python.clicks += 1
    assert app.alert.visible and app.reproduce.text.value == "" and app.reproduce.copy.disabled


def test_a_title_replaces_an_interactive_plots_own_title(app):
    app.renderer.value = "matplotlib"
    app.plot.value = "summary_time_voltage_consumption"
    _option(app, "title", "My title")
    figure = app.session.figure
    texts = [t.get_text() for t in (figure._suptitle, *(s._suptitle for s in figure.subfigs)) if t is not None]
    assert texts == ["My title"]


def test_reloading_a_composition_without_its_plots_clears_the_screen(app):
    app.mode.value = "compose"
    app.composer.assign(["plasma_current_time"])
    assert app.draw_composition()
    app.session.grouped_plots = lambda: {}  # the new sources offer none of its plots
    app.load(app.requested_sources())
    assert app.session.composition is None and app.download.disabled
    assert app.static.object is None and app.interactive.object is None


def test_the_upload_widget_lets_go_of_the_bytes(app):
    loaded = []
    app.load = lambda sources: loaded.append(sources) or False
    app.upload.param.update(filename=["a.json"], value=[b"{}"])
    assert loaded and app.upload.value is None
    app.upload.param.update(filename=["a.json"], value=[b"{}"])
    assert len(loaded) == 2, "the same file can be uploaded again"


def test_serve_admits_the_page_on_the_address_it_binds(monkeypatch):
    calls = []
    monkeypatch.setattr(pn, "serve", lambda panels, **kwargs: calls.append(kwargs))
    monkeypatch.setenv("VAFT_GUI_PASSWORD", "x")
    with pytest.warns(UserWarning):
        gui_app.serve(address="10.0.0.5", port=5123)
    assert "10.0.0.5:5123" in calls[-1]["websocket_origin"]



def test_every_origin_serve_lists_is_one_bokeh_accepts(monkeypatch):
    """The server must start: an origin Bokeh cannot parse stops it (``[::1]:5006`` did)."""
    from bokeh.server.util import create_hosts_allowlist

    calls = []
    monkeypatch.setattr(pn, "serve", lambda panels, **kwargs: calls.append(kwargs))
    monkeypatch.setenv("VAFT_GUI_PASSWORD", "x")
    gui_app.serve(port=5123)
    with pytest.warns(UserWarning):
        gui_app.serve(address="0.0.0.0", port=5123)
    with pytest.warns(UserWarning):
        gui_app.serve(address="fe80::1", port=5123)
    for kwargs in calls:
        create_hosts_allowlist(kwargs["websocket_origin"], 5123)


def test_the_page_loads_plotly_before_the_first_figure(monkeypatch):
    """Without the extension the Plotly pane stays blank in the browser."""
    loaded = []
    monkeypatch.setattr(pn, "extension", lambda *names, **kw: loaded.extend(names))
    gui_app.build_app(sample=39915, plot="plasma_current_time")
    assert "plotly" in loaded


# --- Figure Options, reproduction and composition (#1421, #1467) ------------------


def test_the_form_has_a_widget_for_every_option_and_reads_back_intent(app):
    import dataclasses

    from vaft.gui.options_form import SECTIONS
    from vaft.plot import FigureOptions

    names = {f.name for f in dataclasses.fields(FigureOptions)}
    assert set(app.options_form.widgets) == names, "derived from the dataclass, nothing missing"
    assert {name for section in SECTIONS.values() for name in section} <= names
    assert app.options_form.value() == FigureOptions(), "untouched means inherited"
    _option(app, "legend", "off")
    _option(app, "font_family", "Arial, DejaVu Sans")
    _option(app, "ylim", None, 5.0)
    assert app.session.figure_options.to_dict() == {
        "ylim": [None, 5.0], "legend": False, "font_family": ["Arial", "DejaVu Sans"],
    }
    app.options_form.reset_button.clicks += 1
    assert not app.session.figure_options, "reset restores inheritance, not the resolved values"


def test_copy_python_and_cli_reproduce_the_screen(app):
    _option(app, "xlim", 0.30, 0.33)
    app.reproduce.format.value = "single_column"
    code = app.reproduce.write("python")
    assert "format='single_column'" in code and "'xlim': [0.3, 0.33]" in code
    assert "backend" not in code, "a format makes it a Matplotlib figure"
    namespace: dict = {}
    exec(code, namespace)
    assert namespace["axes"].get_xlim() == pytest.approx((0.30, 0.33))
    command = app.reproduce.write("cli")
    assert command.startswith("vaft plot equilibrium_field_psi --sample 39915") and "--format single_column" in command
    assert "pylustrator.start()" in app.reproduce.write("pylustrator") and not app.reproduce.copy.disabled
    app.reproduce.format.value = "(inherited)"
    assert "backend='plotly'" in app.reproduce.write("python"), "the interactive view reproduces as Plotly"


def test_only_changed_controls_are_written(app):
    request = app.session.request()
    assert dict(request.options) == {}, "a control left at its default is the plot's default"
    state = app.session.state
    other = next(v for v in state.spec("time_slice").options if v != state["time_slice"])
    state.set("time_slice", other)
    assert dict(app.session.request().options) == {"time_slice": other}


def test_options_act_on_redraws_a_control_triggers(app):
    app.plot.value = "plasma_current_time"  # static: drawn by Matplotlib
    base = app.session.figure.axes[0].get_lines()[0].get_linewidth()
    _option(app, "line_scale", 2.0)
    assert app.session.figure.axes[0].get_lines()[0].get_linewidth() == pytest.approx(2 * base)
    state = app.session.state
    unit = next(v for v in state.spec("yunit").options if v != state["yunit"])
    widget = _widget(app, "Unit")
    widget.value = unit
    assert app.session.figure.axes[0].get_lines()[0].get_linewidth() == pytest.approx(2 * base), \
        "a control change redraws under the options"


def test_the_composer_draws_exports_and_reproduces_a_composition(app):
    app.mode.value = "compose"
    assert app.compose_box.visible and not app.plot_box.visible
    app.composer.preset.value = "2 × 1"
    app.composer.assign(["plasma_current_time", "equilibrium_profile_q"])
    _option(app, "title", "stack")
    assert app.draw_composition() and app.session.composition is not None
    assert app.session.renderer == "matplotlib", "a cell without a Plotly rendering keeps it static"
    figure = app.session.figure
    assert len([a for a in figure.axes if a.get_label() != "<colorbar>"]) == 2
    assert figure._suptitle.get_text() == "stack"
    assert app.download.filename.startswith("composition_") and app._export().getvalue()[:4] == b"\x89PNG"
    code = app.reproduce.write("python")
    assert "FigureComposition.from_dict(" in code and "vaft.omas.compose(composition, data" in code
    namespace: dict = {}
    exec(code, namespace)
    assert len(namespace["axes"]) == 2
    app.composer.assign(["plasma_current_time", "plasma_current_time"])
    assert app.draw_composition(), "a plot used twice gets distinct cell names"
    app.composer.rows.value = 1
    app.composer.cells[(0, 0)][2].value = 2  # a span over a cell the grid no longer has
    assert not app.draw_composition() and "does not fit" in app.alert.object
    app.mode.value = "plot"
    assert app.session.composition is None and app.session.plot == app.plot.value
