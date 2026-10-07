"""Same-shot equilibrium section: stored geometry, calibrated pixels and time."""

from __future__ import annotations

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from vaft.data.resources import sample_camera_visible_frame_paths
from vaft.machine_mapping.camera_visible import vfit_camera_visible_dynamic, vfit_camera_visible_static
from vaft.omas import plot_camera_visible_image, plot_machine_geometry_poloidal
from vaft.omas.process_wrapper import camera_projection_for
from vaft.omas.sample import sample_ods
from vaft.plot.equilibrium_section import DEFAULT_SECTION_PHI, _camera_layer, build_equilibrium_section
from vaft.plot.models import GeometryLayer


@pytest.fixture(scope="module")
def same_shot():
    ods = sample_ods(39915)
    frames = [(time, path) for time, path in sample_camera_visible_frame_paths(39915)
              if 0.316 <= time <= 0.331]
    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for _, path in frames]
    assert len(frames) == 38 and all(image is not None for image in images)
    vfit_camera_visible_static(ods, lines_n=images[0].shape[0], columns_n=images[0].shape[1],
                               source="archived 39915 FAST frames")
    vfit_camera_visible_dynamic(ods, images=images, times_s=[time for time, _ in frames])
    return ods, frames


def test_section_preserves_source_coordinates_and_real_probe_phi(same_shot):
    ods, frames = same_shot
    projection = camera_projection_for(39915)
    section = build_equilibrium_section(ods, time=frames[7][0], projection=projection)
    assert section.section_phi == pytest.approx(np.pi)
    assert section.equilibrium_time == pytest.approx(0.319)
    assert sum(layer.kind == "points" and layer.style.get("marker") == "s"
               for layer in section.rz.layers) == 11
    assert sum(layer.kind == "points" and layer.style.get("marker") == "x"
               for layer in section.rz.layers) == 64
    first_probe = section.camera_layers[
        next(i for i, layer in enumerate(section.camera_layers) if layer.label == "EFIT B-pol probes")]
    r = float(ods["magnetics.b_field_pol_probe.0.position.r"])
    z = float(ods["magnetics.b_field_pol_probe.0.position.z"])
    phi = float(ods["magnetics.b_field_pol_probe.0.position.phi"])
    assert phi != DEFAULT_SECTION_PHI
    uv, valid = projection.project(np.array([[r * np.cos(phi), r * np.sin(phi), z]]) * 100)
    assert valid[0]
    np.testing.assert_allclose([first_probe.r[0], first_probe.z[0]], uv[0])
    rectangle = "pf_active.coil.0.element.0.geometry.rectangle"
    first_r = float(ods[f"{rectangle}.r"]) - float(ods[f"{rectangle}.width"]) / 2
    first_z = float(ods[f"{rectangle}.z"]) - float(ods[f"{rectangle}.height"]) / 2
    expected, valid = projection.project(np.array([[-first_r, 0.0, first_z]]) * 100)
    assert valid[0]
    pf_pixels = next(layer for layer in section.camera_layers if layer.label == "PF active")
    np.testing.assert_allclose([pf_pixels.r[0], pf_pixels.z[0]], expected[0])
    assert not any(layer.role == "equilibrium" and layer.r.size > 2
                   and np.nanmax(layer.z) > 1.0 for layer in section.rz.layers)


def test_projection_marks_invalid_vertices_as_gaps():
    class Projection:
        def project(self, xyz):
            return xyz[:, :2], np.array([True, False, True])

    layer = GeometryLayer([0.2, 0.3, 0.4], [-0.1, 0, 0.1], label="path")
    actual = _camera_layer(layer, 0.0, Projection())
    assert np.isfinite(actual.r[[0, 2]]).all()
    assert np.isnan(actual.r[1]) and np.isnan(actual.z[1])


def test_single_frame_and_rz_use_the_same_section(same_shot):
    ods, frames = same_shot
    figure, axes = plot_machine_geometry_poloidal(ods, overlay="equilibrium_section", time_slice=3)
    try:
        assert "phi=180.0" in axes.get_title()
        assert "LCFS" in axes.get_legend_handles_labels()[1]
    finally:
        plt.close(figure)
    figure, axes = plot_camera_visible_image(ods, overlay="equilibrium_section", frame_index=7)
    try:
        assert len(axes.images) == 1
        assert "EFIT t=319.0 ms" in axes.get_title()
        assert "phi=180.0" in axes.get_title()
    finally:
        plt.close(figure)
    with pytest.raises(ValueError, match="outside the valid equilibrium interval"):
        build_equilibrium_section(ods, time=0.3052)


def test_movie_selects_overlap_and_writes_scientific_sidecar(same_shot, tmp_path):
    ods, frames = same_shot
    automatic = plot_camera_visible_image(ods, overlay="equilibrium_section", animation=True)
    assert len(automatic) == 26
    assert automatic.driver.values[-1] == pytest.approx(0.326)
    movie = plot_camera_visible_image(ods, overlay="equilibrium_section", animation=True,
                                      frame_index=[0, 7, 25], fps=3, dpi=35)
    path = movie.save(tmp_path / "section.mp4")
    assert path.stat().st_size > 0
    sidecar = json.loads((tmp_path / "section.mp4.json").read_text())
    context = sidecar["scientific_context"]
    assert context["source_shot"] == 39915
    assert context["pf_geometry_version"] == "1906"
    assert context["passive_wall_geometry_version"] == "1512"
    assert context["camera_projection"]["shot"] == 39915
    assert [pair["frame_index"] for pair in context["frame_equilibrium_pairs"]] == [0, 7, 25]
    assert [pair["camera_time_s"] for pair in context["frame_equilibrium_pairs"]] == pytest.approx(
        [frames[i][0] for i in (0, 7, 25)])
