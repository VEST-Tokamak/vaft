"""Video export from the GUI's Export card (#1400) through ``plot_*(..., animation=True)``.

The GUI does not encode anything itself: it asks the public animation path for
the on-screen plot over the sequence its player walks, with the reader's other
options, and streams the file.  Formats are offered only for a plot that has
that sequence; ``gif`` needs no extra, ``mp4``/``webm`` need PyAV.
"""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest

pn = pytest.importorskip("panel")

from vaft.gui import app as gui_app  # noqa: E402
from vaft.gui.figure import EXPORT_FORMATS, VIDEO_FORMATS  # noqa: E402
from vaft.gui.state import BrowserSession, Source  # noqa: E402

_PLOTS = {
    "equilibrium_profile_q": ("matplotlib", "plotly"),
    "plasma_current_time": ("matplotlib",),
    "vacuum_field": ("matplotlib",),
}


class _Session(BrowserSession):
    """The real session with a two-plot catalog, so no test pays for discovery."""

    def _discover(self, data):
        return [
            SimpleNamespace(name=name, subject=name.split("_")[0], backends=backends)
            for name, backends in _PLOTS.items()
        ]


@pytest.fixture(scope="module")
def ods():
    from vaft.omas import sample_ods

    return sample_ods(39915)


@pytest.fixture
def app(ods, monkeypatch):
    monkeypatch.setattr("vaft.gui.state.load_source", lambda source: ods)
    built = gui_app.BrowserApp(_Session())
    assert built.load(Source("sample", 39915))
    built.plot.value = "equilibrium_profile_q"
    yield built
    built.close()


def _gif_frames(data: bytes) -> int:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return image.n_frames


# -- the session --------------------------------------------------------------------------


def test_the_sequence_is_the_slice_control_the_player_walks(app):
    session = app.session
    control = session.sequence_control()
    assert control is not None and control.name == "time_slice"
    assert session.sequence_states() == tuple(control.options)


def test_a_gif_has_one_frame_per_chosen_state(app):
    states = app.session.sequence_states()
    data = app.session.export_video("gif", fps=5, step=2, dpi=40)
    assert data[:6] in (b"GIF87a", b"GIF89a")
    assert _gif_frames(data) == len(states[::2])


def test_the_frame_times_describe_the_video_just_exported(app, ods):
    with pytest.raises(RuntimeError, match="no video has been exported"):
        app.session.video_metadata()
    states = app.session.sequence_states()
    app.session.export_video("gif", fps=5, step=3, dpi=40)
    metadata = json.loads(app.session.video_metadata())
    assert metadata["driver"]["name"] == "time_slice"
    assert metadata["driver"]["indices"] == list(states[::3])
    stored = [float(t) for t in ods["equilibrium.time"]]
    assert metadata["driver"]["values"] == pytest.approx([stored[i] for i in states[::3]])
    assert metadata["presentation"]["fps"] == 5
    assert metadata["encoder"]["container"] == "gif" and metadata["encoder"]["frame_delay_ms"] == 200


def test_the_readers_other_choices_travel_into_the_video(app):
    app.session.state.set("coordinate", "psi_norm")
    app.session.export_video("gif", fps=5, step=4, dpi=40)
    metadata = json.loads(app.session.video_metadata())
    assert metadata["options"].get("coordinate") == "psi_norm"


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda app: app.session.select("plasma_current_time", renderer="matplotlib"), "no sequence"),
        (lambda app: None, "step must be a positive"),
    ],
)
def test_a_video_is_refused_with_a_reason(app, change, match):
    change(app)
    step = 0 if match.startswith("step") else 1
    with pytest.raises(ValueError, match=match):
        app.session.export_video("gif", step=step)


def test_figure_options_are_not_silently_dropped(app, monkeypatch):
    monkeypatch.setattr(app.session, "figure_options", {"xlim": (0.0, 1.0)})
    with pytest.raises(ValueError, match="do not apply figure options"):
        app.session.export_video("gif")


def test_an_unknown_video_format_is_refused(app):
    with pytest.raises(ValueError, match="video format must be one of"):
        app.session.export_video("avi")


def test_mp4_encodes_through_pyav(app):
    av = pytest.importorskip("av", reason="the video extra (PyAV) is optional")
    data = app.session.export_video("mp4", fps=4, step=2, dpi=40)
    with av.open(io.BytesIO(data)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == len(app.session.sequence_states()[::2])


# -- the Export card ------------------------------------------------------------------------


def test_video_formats_are_offered_only_for_a_plot_with_a_sequence(app):
    assert list(app.export_format.options) == list(EXPORT_FORMATS) + list(VIDEO_FORMATS)
    app.plot.value = "plasma_current_time"
    assert list(app.export_format.options) == list(EXPORT_FORMATS)


def test_choosing_a_video_format_shows_its_fields_and_names_the_file(app):
    assert not app.export_fps.visible and not app.download_times.visible
    app.export_format.value = "gif"
    assert app.export_fps.visible and app.export_step.visible and app.download_times.visible
    assert app.download.filename.endswith(".gif") and app.download.label == "Download video"
    states = app.session.sequence_states()
    assert app.export_step.name == f"Every Nth state (of {len(states)})"
    app.export_format.value = "png"
    assert not app.export_fps.visible and app.download.label == "Download figure"


def test_the_card_downloads_a_gif_and_then_its_frame_times(app):
    app.export_format.value = "gif"
    assert app.download_times.disabled, "no video yet: nothing to describe"
    app.export_fps.value, app.export_step.value, app.export_dpi.value = 5, 4, 50
    data = app._export()
    assert data is not None and _gif_frames(data.getvalue()) == len(app.session.sequence_states()[::4])
    assert not app.download_times.disabled
    times = json.loads(app._export_times().getvalue())
    assert times["driver"]["indices"] == list(app.session.sequence_states()[::4])


def test_moving_to_another_sequence_resets_the_stride_and_its_count(app):
    app.export_format.value = "gif"
    assert app.export_step.value == 1 and app.export_step.name.endswith("(of 9)")
    app.plot.value = "vacuum_field"
    samples = len(app.session.sequence_states())
    assert samples == 2500
    assert app.export_step.value == 13 and app.export_step.name == "Every Nth state (of 2500)"
    assert app.export_format.value == "gif" and app.download_times.disabled


def test_a_movie_longer_than_the_browser_export_draws_is_refused(app):
    app.plot.value = "vacuum_field"
    with pytest.raises(ValueError, match="more than the browser export draws"):
        app.session.export_video("gif", step=1)


def test_video_frames_default_to_screen_resolution(app):
    assert app.export_dpi.value == 150
    app.export_format.value = "gif"
    assert app.export_dpi.value == 100
    app.export_format.value = "png"
    assert app.export_dpi.value == 150


def test_a_missing_pyav_is_reported_in_the_alert_not_raised(app, monkeypatch):
    monkeypatch.setitem(sys.modules, "av", None)
    app.export_format.value = "mp4"
    assert app._export() is None
    assert app.alert.visible and "vaft[video]" in app.alert.object
