import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from omas import ODS

from vaft.plot.models import GeometryLayer, Image2D, ImageSequence
from vaft.plot.renderers.images import render_image_2d, render_image_sequence
import vaft.omas as vomas


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


# --- view models -------------------------------------------------------


def test_image2d_rejects_non_2d_values():
    with pytest.raises(ValueError):
        Image2D(values=np.zeros(4))


def test_image2d_accepts_geometry_layer_overlays():
    layer = GeometryLayer(r=np.array([1.0, 2.0]), z=np.array([3.0, 4.0]), kind="points")
    model = Image2D(values=np.zeros((3, 3)), overlays=(layer,))
    assert model.overlays == (layer,)


def test_image_sequence_requires_matching_frame_shapes():
    with pytest.raises(ValueError):
        ImageSequence(frames=(np.zeros((3, 3)), np.zeros((4, 4))), time=np.array([0.0, 1.0]))


def test_image_sequence_requires_time_length_match():
    with pytest.raises(ValueError):
        ImageSequence(frames=(np.zeros((3, 3)),), time=np.array([0.0, 1.0]))


def test_image_sequence_default_vmin_vmax_from_frames():
    frames = (np.full((2, 2), 1.0), np.full((2, 2), 9.0))
    model = ImageSequence(frames=frames, time=np.array([0.0, 1.0]))
    assert model.vmin == 1.0
    assert model.vmax == 9.0


# --- renderers -----------------------------------------------------------


def test_render_image_2d_draws_raster_and_overlay():
    layer = GeometryLayer(
        r=np.array([1.0, 2.0, 3.0]), z=np.array([1.0, 2.0, 1.0]),
        kind="points", label="Wall", style={"color": "yellow"},
    )
    model = Image2D(values=np.arange(12).reshape(3, 4), overlays=(layer,), title="Test frame")
    fig, ax = render_image_2d(model, show=False)
    assert ax.images[0].get_array().shape == (3, 4)
    assert ax.get_title() == "Test frame"
    labels = [line.get_label() for line in ax.lines]
    assert "Wall" in labels
    plt.close(fig)


def test_render_image_2d_rejects_wrong_model_type():
    with pytest.raises(TypeError):
        render_image_2d("not a model")


def test_render_image_sequence_returns_animation():
    frames = tuple(np.full((3, 3), i) for i in range(4))
    model = ImageSequence(frames=frames, time=np.array([0.0, 0.1, 0.2, 0.3]))
    fig, ax, anim = render_image_sequence(model, show=False)
    assert list(anim.new_frame_seq()) == list(range(4))
    plt.close(fig)


def test_render_image_sequence_saves_to_gif(tmp_path):
    frames = tuple(np.full((3, 3), i) for i in range(3))
    model = ImageSequence(frames=frames, time=np.array([0.0, 0.1, 0.2]))
    output = tmp_path / "camera.gif"
    fig, ax, anim = render_image_sequence(model, save_path=output, show=False)
    assert output.exists()
    plt.close(fig)


# --- vaft.omas adapters ----------------------------------------------------


def test_plot_camera_visible_image_frame_by_index():
    ods = _build_ods()
    fig, ax = vomas.plot_camera_visible_image_frame(ods, frame_index=2, show=False)
    assert ax.images[0].get_array().shape == (6, 8)
    assert "frame 2" in ax.get_title()
    plt.close(fig)


def test_plot_camera_visible_image_frame_by_nearest_time():
    ods = _build_ods()
    # Frame 2 is at t=0.2808; 0.2809 is closest to it.
    fig, ax = vomas.plot_camera_visible_image_frame(ods, time=0.2809, show=False)
    assert "frame 2" in ax.get_title()
    plt.close(fig)


def test_plot_camera_visible_image_frame_defaults_to_first_frame():
    ods = _build_ods()
    fig, ax = vomas.plot_camera_visible_image_frame(ods, show=False)
    assert "frame 0" in ax.get_title()
    plt.close(fig)


def test_plot_camera_visible_image_frame_title_includes_channel_name_and_time():
    ods = _build_ods()
    fig, ax = vomas.plot_camera_visible_image_frame(ods, frame_index=3, show=False)
    assert ax.get_title() == "Fast Camera frame 3 @ t=0.2812s"
    plt.close(fig)


def test_plot_camera_visible_image_frame_custom_title_and_axes():
    ods = _build_ods()
    _, ax = plt.subplots()
    fig, returned_ax = vomas.plot_camera_visible_image_frame(ods, ax=ax, title="Custom Title", show=False)
    assert returned_ax is ax
    assert ax.get_title() == "Custom Title"
    plt.close(fig)


def test_plot_camera_visible_image_frame_out_of_range_index_raises():
    ods = _build_ods(n_frames=3)
    with pytest.raises(IndexError):
        vomas.plot_camera_visible_image_frame(ods, frame_index=10, show=False)


def test_plot_camera_visible_animation_frames_returns_animation_over_all_frames():
    ods = _build_ods(n_frames=4)
    fig, ax, anim = vomas.plot_camera_visible_animation_frames(ods, show=False)
    assert list(anim.new_frame_seq()) == list(range(4))
    plt.close(fig)


def test_plot_camera_visible_animation_frames_subset_of_frames():
    ods = _build_ods(n_frames=6)
    fig, ax, anim = vomas.plot_camera_visible_animation_frames(ods, frame_indices=[1, 3, 5], show=False)
    assert list(anim.new_frame_seq()) == list(range(3))
    plt.close(fig)


def test_plot_camera_visible_animation_frames_saves_to_gif(tmp_path):
    ods = _build_ods(n_frames=3, shape=(4, 4))
    output = tmp_path / "camera.gif"
    fig, ax, anim = vomas.plot_camera_visible_animation_frames(ods, save_path=output, show=False)
    assert output.exists()
    plt.close(fig)


def test_camera_visible_image_plots_are_discoverable():
    from vaft.omas.plotting import available_plots

    names = {row["name"] for row in available_plots(ODS(consistency_check=False))}
    # An empty ODS auto-vivifies empty containers, so this checks the plot is
    # at least registered and catalog-visible, not necessarily "available".
    from vaft.plot.registry import canonical_names

    all_names = set(canonical_names())
    assert "camera_visible_image_frame" in all_names
    assert "camera_visible_image_efit_overlay" in all_names
    assert "camera_visible_image_field_line" in all_names
    assert "camera_visible_animation_frames" in all_names
