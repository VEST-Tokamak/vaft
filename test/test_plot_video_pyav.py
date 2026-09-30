"""The private PyAV backend of ``animation=True`` (#1050): encode, decode, compare.

Lossy codecs are never compared pixel by pixel; what is checked is what the
contract promises -- one decoded frame per scientific state, the frame size,
and a playback length of ``states / fps``.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

av = pytest.importorskip("av", reason="the video extra (PyAV) is optional")

import vaft.omas as vomas  # noqa: E402
from _sample_fixtures import sample_ods  # noqa: E402
from vaft.plot import _animation  # noqa: E402
from vaft.plot._pyav import encode_video  # noqa: E402

DPI = 30


@pytest.fixture(scope="module")
def camera():
    return sample_ods(40600)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _decode(path):
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
        return frames, container.duration / av.time_base


@pytest.mark.parametrize("suffix", [".mp4", ".webm"])
def test_synthetic_frames_round_trip(tmp_path, suffix):
    frames = [np.full((31, 45, 3), 20 * i, dtype=np.uint8) for i in range(6)]
    facts = encode_video(iter(frames), tmp_path / f"ramp{suffix}", fps=3)
    decoded, duration = _decode(tmp_path / f"ramp{suffix}")
    assert facts["frames"] == len(decoded) == 6
    # yuv420p needs even sides: one edge pixel is added, nothing is resized.
    assert (decoded[0].width, decoded[0].height) == (46, 32)
    assert duration == pytest.approx(6 / 3, abs=0.05)
    assert [round(float(f.time), 3) for f in decoded] == [round(i / 3, 3) for i in range(6)]


@pytest.mark.parametrize("suffix", [".mp4", ".webm"])
def test_camera_sequence_encodes_one_frame_per_state(camera, tmp_path, suffix):
    animation = vomas.plot_camera_visible_image(
        camera, frame_index=range(0, 50, 5), animation=True, fps=4, dpi=DPI,
    )
    first = next(animation.frames())
    path = animation.save(tmp_path / f"camera{suffix}")
    decoded, duration = _decode(path)
    assert len(decoded) == len(animation) == 10
    assert (decoded[0].height, decoded[0].width) == tuple(s + s % 2 for s in first.shape[:2])
    assert duration == pytest.approx(10 / 4, abs=0.05)
    sidecar = json.loads(path.with_name(path.name + ".json").read_text())
    assert sidecar["encoder"]["container"] == suffix[1:]
    assert sidecar["encoder"]["frames"] == 10
    assert sidecar["driver"]["indices"] == list(range(0, 50, 5))


def test_equilibrium_slices_encode_through_the_same_path(tmp_path):
    animation = vomas.plot_equilibrium_field_2d(sample_ods(39915), time_slice=[0, 3, 6], animation=True, dpi=DPI)
    decoded, _ = _decode(animation.save(tmp_path / "equilibrium.mp4"))
    assert len(decoded) == 3


def test_notebook_inlines_a_small_video_and_posters_a_long_one(camera, monkeypatch):
    animation = vomas.plot_camera_visible_image(camera, frame_index=[0, 1, 2], animation=True, dpi=DPI)
    assert '<video' in animation._repr_html_()
    monkeypatch.setattr(_animation, "PREVIEW_MAX_FRAMES", 2)
    page = animation._repr_html_()
    assert "<video" not in page and "<img" in page


def test_a_frame_that_fails_to_draw_leaves_no_video(tmp_path):
    animation = vomas.plot_equilibrium_field_2d(
        sample_ods(39915), time_slice=[7, 8], style="surfaces", animation=True, dpi=DPI,
    )
    with pytest.raises(RuntimeError, match="time_slice=8"):
        animation.save(tmp_path / "broken.mp4")
    assert not (tmp_path / "broken.mp4").exists()
