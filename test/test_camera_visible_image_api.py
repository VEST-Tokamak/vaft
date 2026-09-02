"""One camera image API: base frame + overlays + projection (issue #261 §18-23).

39915 has a packaged calibrated pose, so the fixture adds synthetic frames to
that shot and every overlay projects for real.
"""

import contextlib
import io
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
import vaft.omas
from vaft.omas._plot_recipes import CAMERA_OVERLAYS, build_model, normalize_entries
from vaft.omas.process_wrapper import camera_projection_for
from vaft.process.camera_geometry import CameraProjection

ROWS, COLS = 1024, 1280


def _with_frames(ods, n_frames=3):
    ods["camera_visible.channel.0.name"] = "Fast Camera"
    ods["camera_visible.channel.0.detector.0.lines_n"] = ROWS
    ods["camera_visible.channel.0.detector.0.columns_n"] = COLS
    for i in range(n_frames):
        ods[f"camera_visible.channel.0.detector.0.frame.{i}.image_raw"] = np.full((ROWS, COLS), 100.0 + i)
        ods[f"camera_visible.channel.0.detector.0.frame.{i}.time"] = 0.318 + 0.001 * i
    return ods


def _load(rel):
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path(rel)))


@pytest.fixture(scope="module")
def shot():
    return _with_frames(_load("samples/39915/omas.json.gz"))


@pytest.fixture(scope="module")
def unposed():
    ods = _load("samples/41524/imas.nc")  # no packaged pose for this shot
    return _with_frames(ods)


def _labels(model):
    return [layer.label for layer in model.overlays]


# ---------------------------------------------------------------------------
# base + overlay (sections 18, 19)
# ---------------------------------------------------------------------------

def test_the_bare_frame_has_no_overlay(shot):
    model = build_model("camera_visible_image", normalize_entries(shot))
    assert model.values.shape == (ROWS, COLS) and model.overlays == ()


def test_overlays_project_machine_geometry_into_the_frame(shot):
    model = build_model("camera_visible_image", normalize_entries(shot), time=0.319, overlay=("wall", "equilibrium"))
    assert _labels(model)[:4] == ["Wall", "LCFS", "Magnetic axis", "psi surfaces"]
    for layer in model.overlays:
        assert np.isfinite(layer.r).all() and np.isfinite(layer.z).all()
    wall = model.overlays[0]
    # Pixels land in the sensor: some of the wall is in view.
    assert ((wall.r >= 0) & (wall.r < COLS) & (wall.z >= 0) & (wall.z < ROWS)).any()
    assert "frame 1 @ t=0.3190s" in model.title and "shot 39915: wall + equilibrium" in model.title


def test_one_overlay_alone_and_a_string_spelling(shot):
    only_wall = build_model("camera_visible_image", normalize_entries(shot), overlay="wall")
    assert _labels(only_wall) == ["Wall"]
    only_eq = build_model("camera_visible_image", normalize_entries(shot), overlay=("equilibrium",))
    assert "Wall" not in _labels(only_eq) and "LCFS" in _labels(only_eq)


def test_a_field_line_is_traced_by_the_process_layer_and_only_projected_here(shot):
    model = build_model(
        "camera_visible_image", normalize_entries(shot), overlay="field_line", field_line_start=(0.4, 0.0)
    )
    assert _labels(model) == ["Field line", "Start", "End"]
    assert "field line R0=0.400 m, Z0=0.000 m" in model.title
    with pytest.raises(ValueError, match="needs a seed"):
        build_model("camera_visible_image", normalize_entries(shot), overlay="field_line")


def test_unknown_overlays_are_refused_by_name(shot):
    with pytest.raises(ValueError, match="unknown overlay 'lcfs'"):
        build_model("camera_visible_image", normalize_entries(shot), overlay="lcfs")
    assert CAMERA_OVERLAYS == ("wall", "equilibrium", "field_line")


# ---------------------------------------------------------------------------
# projection is separate from overlay (sections 20, 21)
# ---------------------------------------------------------------------------

def test_the_calibrated_projection_is_the_default_and_a_custom_one_is_accepted(shot):
    calibrated = build_model("camera_visible_image", normalize_entries(shot), overlay="wall")
    packaged = camera_projection_for(39915)
    assert packaged.method == "calibrated" and packaged.provenance["shot"] == 39915
    custom = CameraProjection(packaged.camera_matrix, packaged.dist_coeffs, packaged.rvec, packaged.tvec, method="mine")
    same = build_model("camera_visible_image", normalize_entries(shot), overlay="wall", projection=custom)
    assert np.array_equal(calibrated.overlays[0].r, same.overlays[0].r)
    with pytest.raises(ValueError, match="projection must be one of calibrated"):
        build_model("camera_visible_image", normalize_entries(shot), overlay="wall", projection="raw")


def test_a_shot_without_a_packaged_pose_says_so(unposed):
    with pytest.raises(ValueError, match="no calibrated camera projection is available for shot 41524"):
        build_model("camera_visible_image", normalize_entries(unposed), overlay="wall")
    # The bare frame needs no projection at all.
    assert build_model("camera_visible_image", normalize_entries(unposed)).overlays == ()


def test_every_overlay_goes_through_one_projection_object(shot):
    model = build_model(
        "camera_visible_image", normalize_entries(shot), overlay=("wall", "field_line"), field_line_start=(0.4, 0.0)
    )
    assert _labels(model) == ["Wall", "Field line", "Start", "End"]


# ---------------------------------------------------------------------------
# presets (section 19) and discovery (sections 24, 25)
# ---------------------------------------------------------------------------

def test_the_old_functions_are_presets_of_the_image_api(shot):
    general = build_model("camera_visible_image", normalize_entries(shot), overlay=("wall", "equilibrium"))
    preset = build_model("camera_visible_image_efit_overlay", normalize_entries(shot))
    assert _labels(preset) == _labels(general)
    assert np.array_equal(preset.overlays[1].r, general.overlays[1].r)
    frame = build_model("camera_visible_image_frame", normalize_entries(shot))
    assert frame.overlays == ()
    line = build_model("camera_visible_image_field_line", normalize_entries(shot), r0=0.4, z0=0.0)
    assert _labels(line) == ["Field line", "Start", "End"]
    figure, axes = vaft.omas.plot_camera_visible_image(shot, overlay="wall")
    assert [line.get_label() for line in axes.lines] == ["Wall"]
    plt.close(figure)


def test_discovery_states_overlays_and_projection_availability(shot, unposed):
    registry = vaft.omas.available_plots(query="camera_visible")
    image = registry.find("camera_visible_image")
    assert image.overlays == ("wall", "equilibrium", "field_line")
    assert image.projection == {"methods": ("calibrated",)}
    text = str(registry)
    assert "image  plot_camera_visible_image()" in text and "overlays: wall | equilibrium | field_line" in text
    with_pose = vaft.omas.available_plots(shot, query="camera_visible").find("camera_visible_image")
    assert with_pose.projection["available"] is True
    assert "projection: calibrated — available" in str(vaft.omas.available_plots(shot, query="camera_visible"))
    without = vaft.omas.available_plots(unposed, query="camera_visible").find("camera_visible_image")
    assert without.projection["available"] is False and "41524" in without.projection["reason"]
    assert "camera_visible_image" in vaft.plot.canonical_names()
    from vaft.plot.registry import VIEWS
    assert "3d" not in VIEWS


def test_plotting_does_not_grow_the_ods(shot):
    before = len(shot.flat())
    build_model("camera_visible_image", normalize_entries(shot), overlay=("wall", "equilibrium"))
    assert len(shot.flat()) == before
