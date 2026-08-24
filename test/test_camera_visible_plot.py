import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from omas import ODS

from vaft.plot.camera_visible import animate_camera_visible, plot_camera_visible_frame


def _build_ods(n_frames: int = 5, shape: tuple[int, int] = (6, 8)) -> ODS:
    ods = ODS(consistency_check=False)
    ods["camera_visible.channel.0.name"] = "Fast Camera"
    ods["camera_visible.channel.0.detector.0.lines_n"] = shape[0]
    ods["camera_visible.channel.0.detector.0.columns_n"] = shape[1]
    ods["camera_visible.channel.0.detector.0.exposure_time"] = 3.0e-6
    for i in range(n_frames):
        image = np.full(shape, i * 10, dtype=int)
        ods[f"camera_visible.channel.0.detector.0.frame.{i}.image_raw"] = image
        ods[f"camera_visible.channel.0.detector.0.frame.{i}.time"] = 0.280 + i * 0.0004
    return ods


def test_plot_camera_visible_frame_by_index():
    ods = _build_ods()
    fig, ax = plot_camera_visible_frame(ods, frame_index=2, show=False)
    assert ax.images[0].get_array().shape == (6, 8)
    np.testing.assert_array_equal(ax.images[0].get_array(), np.full((6, 8), 20))
    assert "frame 2" in ax.get_title()
    plt.close(fig)


def test_plot_camera_visible_frame_by_nearest_time():
    ods = _build_ods()
    # Frame 2 is at t=0.2808; 0.2809 is closest to it.
    fig, ax = plot_camera_visible_frame(ods, time=0.2809, show=False)
    assert "frame 2" in ax.get_title()
    plt.close(fig)


def test_plot_camera_visible_frame_defaults_to_first_frame():
    ods = _build_ods()
    fig, ax = plot_camera_visible_frame(ods, show=False)
    assert "frame 0" in ax.get_title()
    plt.close(fig)


def test_plot_camera_visible_frame_title_includes_channel_name_and_time():
    ods = _build_ods()
    fig, ax = plot_camera_visible_frame(ods, frame_index=3, show=False)
    assert ax.get_title() == "Fast Camera frame 3 @ t=0.2812s"
    plt.close(fig)


def test_plot_camera_visible_frame_rejects_index_and_time_together():
    ods = _build_ods()
    with pytest.raises(ValueError):
        plot_camera_visible_frame(ods, frame_index=0, time=0.28, show=False)


def test_plot_camera_visible_frame_custom_title_and_axes():
    ods = _build_ods()
    _, ax = plt.subplots()
    fig, returned_ax = plot_camera_visible_frame(ods, ax=ax, title="Custom Title", show=False)
    assert returned_ax is ax
    assert ax.get_title() == "Custom Title"
    plt.close(fig)


def test_plot_camera_visible_frame_missing_data_raises():
    # An empty ODS auto-vivifies an empty frame struct array (len 0) rather
    # than raising KeyError, so the frame-index bounds check is what fires.
    with pytest.raises(IndexError):
        plot_camera_visible_frame(ODS(consistency_check=False), show=False)


def test_plot_camera_visible_frame_out_of_range_index_raises():
    ods = _build_ods(n_frames=3)
    with pytest.raises(IndexError):
        plot_camera_visible_frame(ods, frame_index=10, show=False)


def test_animate_camera_visible_returns_animation_over_all_frames():
    ods = _build_ods(n_frames=4)
    fig, ax, anim = animate_camera_visible(ods, show=False)
    assert list(anim.new_frame_seq()) == list(range(4))
    plt.close(fig)


def test_animate_camera_visible_subset_of_frames():
    ods = _build_ods(n_frames=6)
    fig, ax, anim = animate_camera_visible(ods, frame_indices=[1, 3, 5], show=False)
    assert list(anim.new_frame_seq()) == list(range(3))
    plt.close(fig)


def test_animate_camera_visible_saves_to_gif(tmp_path):
    ods = _build_ods(n_frames=3, shape=(4, 4))
    output = tmp_path / "camera.gif"
    fig, ax, anim = animate_camera_visible(ods, save_path=output, show=False)
    assert output.exists()
    plt.close(fig)


def test_animate_camera_visible_rejects_empty_frame_indices():
    ods = _build_ods()
    with pytest.raises(ValueError):
        animate_camera_visible(ods, frame_indices=[], show=False)
