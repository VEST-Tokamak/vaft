"""The ``animation=True`` contract agreed on #1049, against the packaged samples (#1050).

None of these needs PyAV: frames are checked through ``frames()`` and a
``.gif``, and the ``.mp4`` path is checked for the error it raises when PyAV
is absent.  ``test_plot_video_pyav.py`` encodes and decodes the real thing.
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft.omas as vomas
from _sample_fixtures import sample_ods
from vaft.omas.plotting import normalize_entries
from vaft.plot.backend.discovery import describe_one
from vaft.plot.backend.recipes import build_model
from vaft.plot.controls import controls_for

#: Frames small enough that drawing a handful stays cheap.
DPI = 30


@pytest.fixture(scope="module")
def camera():
    return sample_ods(40600)


@pytest.fixture(scope="module")
def equilibrium():
    return sample_ods(39915)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _frame_times(ods):
    base = "camera_visible.channel.0.detector.0.frame"
    return [float(ods[f"{base}.{i}.time"]) for i in range(len(ods[base]))]


def _slice_control(name, ods):
    record = describe_one(name, normalize_entries(ods))
    (control,) = [c for c in controls_for(record) if c.group == "slice"]
    return control


# -- the sequence coordinate is the slice control ---------------------------------


def test_camera_frames_follow_the_given_order_with_their_physical_times(camera):
    animation = vomas.plot_camera_visible_image(camera, frame_index=[7, 3, 5], animation=True)
    times = _frame_times(camera)
    assert animation.driver.name == "frame_index"
    assert animation.driver.indices == (7, 3, 5)
    assert animation.driver.values == (times[7], times[3], times[5])
    assert (animation.driver.coordinate, animation.driver.unit) == ("time", "s")
    assert len(animation) == 3


def test_omitting_the_driver_animates_every_offered_state(camera, equilibrium):
    control = _slice_control("camera_visible_image", camera)
    low, high, step = control.options
    assert len(vomas.plot_camera_visible_image(camera, animation=True)) == len(range(low, high + 1, step))

    control = _slice_control("equilibrium_field_2d", equilibrium)
    animation = vomas.plot_equilibrium_field_2d(equilibrium, animation=True)
    assert animation.driver.name == "time_slice"
    assert animation.driver.indices == tuple(control.options)
    stored = [float(t) for t in describe_one("equilibrium_field_2d", normalize_entries(equilibrium)).slices["times"]]
    assert animation.driver.values == tuple(stored[i] for i in control.options)


def test_time_range_selects_the_states_inside_it_in_order(camera):
    times = np.asarray(_frame_times(camera))
    window = (times[3] - 1e-7, times[9] + 1e-7)
    animation = vomas.plot_camera_visible_image(camera, time_range=window, animation=True)
    assert animation.driver.indices == tuple(range(3, 10))
    assert animation.metadata["selection"] == {"kind": "time_range", "value": list(window)}


def test_frames_are_one_per_state_uint8_rgb_and_one_size(camera, equilibrium):
    animation = vomas.plot_camera_visible_image(camera, frame_index=[0, 10, 20], animation=True, dpi=DPI)
    frames = list(animation.frames())
    assert len(frames) == 3
    assert {frame.shape for frame in frames} == {frames[0].shape}
    assert frames[0].dtype == np.uint8 and frames[0].shape[2] == 3

    # The R-Z renderer sizes its canvas per slice; the frames still share one.
    animation = vomas.plot_equilibrium_field_2d(equilibrium, time_slice=[0, 4, 8], animation=True, dpi=DPI)
    frames = list(animation.frames())
    assert len(frames) == 3
    assert {frame.shape for frame in frames} == {frames[0].shape}


# -- one scale over the sequence -----------------------------------------------------


def test_camera_frames_share_the_range_of_the_selected_states(camera):
    chosen = [0, 100, 300]
    animation = vomas.plot_camera_visible_image(camera, frame_index=chosen, animation=True)
    entries = normalize_entries(camera)
    values = [build_model("camera_visible_image", entries, frame_index=i).values for i in chosen]
    scale = animation.normalization()
    assert scale["vmin"] == pytest.approx(min(float(np.nanmin(v)) for v in values))
    assert scale["vmax"] == pytest.approx(max(float(np.nanmax(v)) for v in values))
    assert scale["source"] == "sequence range"

    given = vomas.plot_camera_visible_image(camera, frame_index=chosen, animation=True, vmin=0, vmax=255)
    scale = given.normalization()
    assert (scale["vmin"], scale["vmax"], scale["source"]) == (0.0, 255.0, "caller")


def test_filled_equilibrium_maps_share_one_set_of_levels(equilibrium):
    animation = vomas.plot_equilibrium_field_2d(
        equilibrium, time_slice=[0, 4, 7], style="filled", animation=True,
    )
    scale = animation.normalization()
    assert scale["policy"].startswith("one set of contour levels")
    assert scale["levels"][0] <= scale["vmin"] and scale["levels"][-1] >= scale["vmax"]


# -- refusals ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"frame_index": 3}, ValueError, "one state is a static plot"),
        ({"time": 0.31}, ValueError, "one state is a static plot"),
        ({"frame_index": [4]}, ValueError, "one state is a static plot"),
        ({"frame_index": [3, 3]}, ValueError, "one state is a static plot"),
        ({"frame_index": [3, 3, 4]}, ValueError, r"repeats frame_index \[3\]"),
        ({"controls": ("frame_index",)}, TypeError, "offers no controls"),
        ({"frame_index": []}, ValueError, "holds no frame_index"),
        ({"frame_index": [1, 9999]}, ValueError, r"\[9999\] not offered"),
        ({"frame_index": [1, 2], "time_range": (0.3, 0.4)}, ValueError, "not both"),
        ({"time_range": (0.0, 0.1)}, ValueError, "holds no frame_index"),
        ({"fps": 10, "duration": 2}, ValueError, "fps= or duration=, not both"),
        ({"fps": 0}, ValueError, "fps must be a positive"),
        ({"duration": -1.0}, ValueError, "duration must be a positive"),
        ({"timing": "physical"}, NotImplementedError, "reserved"),
        ({"timing": "wall"}, ValueError, "timing must be 'uniform'"),
        ({"vmin": 5, "vmax": 1}, ValueError, "vmin must be below vmax"),
        ({"interactive": True}, TypeError, "choose one"),
        ({"ax": "an axes"}, TypeError, "takes no ax="),
        ({"backend": "plotly"}, ValueError, "Matplotlib"),
        ({"selecton": "all"}, ValueError, "does not take an option named 'selecton'"),
    ],
)
def test_camera_animation_refuses(camera, options, error, match):
    with pytest.raises(error, match=match):
        vomas.plot_camera_visible_image(camera, animation=True, **options)


def test_frames_are_those_of_the_camera_drawn_not_channel_0(camera):
    """The record counts channel 0's frames; another channel has its own count and stamps."""
    import copy

    base = "camera_visible.channel.0.detector.0.frame"
    other = copy.deepcopy(camera)
    for i in range(3):
        other[f"camera_visible.channel.1.detector.0.frame.{i}.image_raw"] = other[f"{base}.{i}.image_raw"]
        other[f"camera_visible.channel.1.detector.0.frame.{i}.time"] = 1.0 + 0.001 * i
    animation = vomas.plot_camera_visible_image(other, channel=1, animation=True)
    assert animation.driver.indices == (0, 1, 2)
    assert animation.driver.values == (1.0, 1.001, 1.002)
    with pytest.raises(ValueError, match=r"\[5\] not offered"):
        vomas.plot_camera_visible_image(other, channel=1, frame_index=[1, 5], animation=True)


@pytest.mark.parametrize("shot", [39915, 40600])
def test_every_slice_control_has_per_state_values(shot):
    """Tripwire (#1380): a plot that gains a slice control must say which time axis it pages.

    ``sequence_values`` dispatches on the plot, not the selector, so a new
    ``time_index`` plot is refused here until its own axis is declared rather
    than being labelled with another plot's stamps.
    """
    from vaft.plot.backend.discovery import describe_entries, sequence_values

    entries = normalize_entries(sample_ods(shot))
    checked = 0
    for record in describe_entries(entries):
        if not record.available:
            continue
        for control in controls_for(record):
            if control.group != "slice":
                continue
            _, unit, values = sequence_values(record.name, entries, control.name)
            highest = control.options[1] if control.kind == "range" else max(control.options)
            assert len(values) > highest, (record.name, control.name, len(values), highest)
            assert unit == "s" and np.all(np.isfinite(values)), record.name
            checked += 1
    assert checked >= 5


def test_the_sequence_seam_is_published_beside_its_sibling():
    """The seam the tripwire above points contributors at must be public.

    ``frame_renderers`` from the same change is in ``render.__all__``; the
    API catalog documents a module's ``__all__``, so the star import and the
    generated reference omitted ``sequence_values`` while naming its sibling.
    """
    from vaft import _api_catalog
    from vaft.plot.backend import discovery, render

    assert "sequence_values" in discovery.__all__ and "frame_renderers" in render.__all__
    namespace: dict = {}
    exec("from vaft.plot.backend.discovery import *", namespace)
    assert namespace["sequence_values"] is discovery.sequence_values
    assert _api_catalog.page_for(discovery.__name__, _api_catalog.load_inventory()) == "plot"


def test_vacuum_field_frames_carry_the_pf_programme_times(equilibrium):
    animation = vomas.plot_vacuum_field(equilibrium, time_index=range(0, 2000, 400), animation=True)
    pf_time = np.asarray(equilibrium["pf_active.time"], dtype=float)
    assert animation.driver.values == tuple(float(pf_time[i]) for i in range(0, 2000, 400))


def test_a_plot_without_a_slice_control_has_no_sequence_coordinate(equilibrium):
    with pytest.raises(ValueError, match="has no sequence coordinate"):
        vomas.plot_equilibrium_overview_histories(equilibrium, animation=True)


def test_image_scale_keywords_are_refused_on_other_views(equilibrium):
    with pytest.raises(ValueError, match="vmin=/vmax= fix the colour scale of an image view"):
        vomas.plot_equilibrium_field_2d(equilibrium, animation=True, vmin=0)


def test_a_state_that_cannot_be_drawn_is_named_not_skipped(equilibrium):
    # Lazy: the call succeeds, the failure surfaces when the states are built.
    animation = vomas.plot_equilibrium_field_2d(
        equilibrium, time_slice=[7, 8], style="surfaces", animation=True, dpi=DPI,
    )
    with pytest.raises(RuntimeError, match=r"time_slice=8 \(time 0\.33"):
        list(animation.frames())


# -- presentation and output ---------------------------------------------------------------


def test_duration_sets_fps_from_the_state_count(camera):
    animation = vomas.plot_camera_visible_image(camera, frame_index=range(0, 40, 2), animation=True, duration=4)
    assert animation.fps == pytest.approx(20 / 4)
    assert animation.metadata["presentation"]["duration"] == pytest.approx(4)


def test_static_calls_are_unchanged(camera):
    figure, axes = vomas.plot_camera_visible_image(camera, frame_index=3)
    assert figure is axes.figure


def test_gif_writes_every_frame_and_the_provenance_sidecar(camera, tmp_path):
    from PIL import Image

    animation = vomas.plot_camera_visible_image(camera, frame_index=[2, 4, 6, 8], animation=True, fps=5, dpi=DPI)
    path = animation.save(tmp_path / "camera.gif")
    with Image.open(path) as image:
        assert image.n_frames == 4
    sidecar = json.loads((tmp_path / "camera.gif.json").read_text())
    assert sidecar["driver"]["indices"] == [2, 4, 6, 8]
    assert sidecar["driver"]["values"] == [_frame_times(camera)[i] for i in (2, 4, 6, 8)]
    assert sidecar["presentation"]["fps"] == 5
    assert sidecar["encoder"]["container"] == "gif"
    assert sidecar["encoder"]["frames"] == sidecar["encoder"]["frames_stored"] == 4
    assert sidecar["encoder"]["frame_delay_ms"] == 200
    assert sidecar["plot"] == "camera_visible_image" and sidecar["label"] == "40600"


def test_sidecar_can_be_skipped_and_a_stale_one_is_removed(camera, tmp_path):
    animation = vomas.plot_camera_visible_image(camera, frame_index=[1, 2], animation=True, dpi=DPI)
    animation.save(tmp_path / "camera.gif")
    animation.save(tmp_path / "camera.gif", sidecar=False)
    assert not (tmp_path / "camera.gif.json").exists()


def test_a_failed_save_keeps_the_previous_movie(equilibrium, tmp_path):
    path = tmp_path / "equilibrium.gif"
    vomas.plot_equilibrium_field_2d(equilibrium, time_slice=[6, 7], animation=True, dpi=DPI).save(path)
    before = path.read_bytes()
    broken = vomas.plot_equilibrium_field_2d(
        equilibrium, time_slice=[7, 8], style="surfaces", animation=True, dpi=DPI,
    )
    with pytest.raises(RuntimeError, match="time_slice=8"):
        broken.save(path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["equilibrium.gif", "equilibrium.gif.json"]


def test_an_unknown_suffix_is_refused(camera, tmp_path):
    animation = vomas.plot_camera_visible_image(camera, frame_index=[1, 2], animation=True)
    with pytest.raises(ValueError, match=r"\.mp4, \.webm, \.gif"):
        animation.save(tmp_path / "camera.avi")


def test_without_pyav_video_names_the_extra_and_the_notebook_shows_a_poster(camera, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "av", None)  # import av -> ImportError
    animation = vomas.plot_camera_visible_image(camera, frame_index=[1, 2], animation=True, dpi=DPI)
    with pytest.raises(ImportError, match=r"pip install vaft\[video\]"):
        animation.save(tmp_path / "camera.mp4")
    assert not (tmp_path / "camera.mp4").exists()
    page = animation._repr_html_()
    assert "<img" in page and "<video" not in page


def test_metadata_is_json_and_names_the_driver(camera):
    animation = vomas.plot_camera_visible_image(camera, frame_index=[1, 2], animation=True)
    data = json.loads(json.dumps(animation.metadata))
    assert data["driver"]["name"] == "frame_index"
    assert data["n_states"] == 2
    assert "2 frames of frame_index" in repr(animation)


def test_the_legacy_animation_adapter_is_deprecated(camera):
    with pytest.warns(DeprecationWarning, match="animation=True"):
        figure, _, _ = vomas.plot_camera_visible_animation_frames(camera, frame_indices=[0, 1])
    plt.close(figure)
