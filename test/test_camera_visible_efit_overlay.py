import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from omas import ODS

from vaft.omas.process_wrapper import compute_camera_visible_efit_overlay
from vaft.plot.camera_visible import plot_camera_visible_efit_overlay

SHOT = 39915  # one of the three packaged calibrated shots


def _build_ods(n_frames: int = 4, shape: tuple[int, int] = (12, 16)) -> ODS:
    ods = ODS(consistency_check=False)

    # camera_visible: frame times bracketing the equilibrium time slices below.
    ods["camera_visible.channel.0.name"] = "Fast Camera"
    ods["camera_visible.channel.0.detector.0.lines_n"] = shape[0]
    ods["camera_visible.channel.0.detector.0.columns_n"] = shape[1]
    for i in range(n_frames):
        ods[f"camera_visible.channel.0.detector.0.frame.{i}.image_raw"] = np.full(shape, i * 10, dtype=int)
        ods[f"camera_visible.channel.0.detector.0.frame.{i}.time"] = 0.28 + i * 0.001

    # equilibrium: two time slices, simple radially-symmetric psi grid, LCFS a
    # circle of radius 0.2 m around (R0, Z0) = (0.4, 0.0), magnetic axis at
    # the center.
    r0, z0, lcfs_radius = 0.4, 0.0, 0.2
    R = np.linspace(0.0, 1.0, 41)
    Z = np.linspace(-0.5, 0.5, 41)
    Rm, Zm = np.meshgrid(R, Z, indexing="ij")
    psi_grid = (Rm - r0) ** 2 + Zm**2  # shape (len(R), len(Z))
    psi_axis = 0.0
    psi_boundary = lcfs_radius**2

    theta_lcfs = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    lcfs_r = r0 + lcfs_radius * np.cos(theta_lcfs)
    lcfs_z = z0 + lcfs_radius * np.sin(theta_lcfs)

    ods["equilibrium.time"] = np.array([0.28, 0.281])
    for idx in range(2):
        ts = f"equilibrium.time_slice.{idx}"
        ods[f"{ts}.time"] = float(ods["equilibrium.time"][idx])
        ods[f"{ts}.boundary.outline.r"] = lcfs_r
        ods[f"{ts}.boundary.outline.z"] = lcfs_z
        ods[f"{ts}.global_quantities.magnetic_axis.r"] = r0
        ods[f"{ts}.global_quantities.magnetic_axis.z"] = z0
        ods[f"{ts}.global_quantities.psi_axis"] = psi_axis
        ods[f"{ts}.global_quantities.psi_boundary"] = psi_boundary
        ods[f"{ts}.profiles_2d.0.grid.dim1"] = R
        ods[f"{ts}.profiles_2d.0.grid.dim2"] = Z
        ods[f"{ts}.profiles_2d.0.psi"] = psi_grid

    # wall: a rough limiter outline (not physically accurate, just present).
    theta_wall = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = 0.4 + 0.45 * np.cos(theta_wall)
    ods["wall.description_2d.0.limiter.unit.0.outline.z"] = 0.45 * np.sin(theta_wall)

    return ods


def test_compute_overlay_returns_expected_keys_and_shapes():
    ods = _build_ods()
    result = compute_camera_visible_efit_overlay(ods, SHOT, frame_index=0, n_theta=37)

    assert set(result) == {
        "frame_index", "frame_time", "equilibrium_time_index", "equilibrium_time",
        "image_shape", "wall_uv", "lcfs_uv", "magnetic_axis_uv", "flux_surfaces_uv",
    }
    assert result["frame_index"] == 0
    assert result["image_shape"] == (12, 16)
    assert result["wall_uv"].shape[1] == 2
    assert result["lcfs_uv"].shape[1] == 2
    assert result["magnetic_axis_uv"].shape[1] == 2
    assert set(result["flux_surfaces_uv"]) == {0.25, 0.5, 0.75, 0.95}
    for uv in result["flux_surfaces_uv"].values():
        assert uv.shape[1] == 2


def test_compute_overlay_selects_frame_by_nearest_time():
    ods = _build_ods()
    result = compute_camera_visible_efit_overlay(ods, SHOT, frame_time=0.2807, n_theta=37)
    assert result["frame_index"] == 1  # frame 1 is at t=0.281, nearest to 0.2807


def test_compute_overlay_selects_nearest_equilibrium_time_slice():
    ods = _build_ods()
    # Frame 3 is at t=0.283, closer to equilibrium.time[1]=0.281 than [0]=0.28.
    result = compute_camera_visible_efit_overlay(ods, SHOT, frame_index=3, n_theta=37)
    assert result["equilibrium_time_index"] == 1


def test_compute_overlay_rejects_index_and_time_together():
    ods = _build_ods()
    with pytest.raises(ValueError):
        compute_camera_visible_efit_overlay(ods, SHOT, frame_index=0, frame_time=0.28)


def test_compute_overlay_unsupported_shot_raises_with_supported_list():
    ods = _build_ods()
    with pytest.raises(FileNotFoundError, match="34764.*39915.*47518|39915.*47518.*34764|34764"):
        compute_camera_visible_efit_overlay(ods, 99999, frame_index=0)


def test_compute_overlay_disable_flux_surfaces():
    ods = _build_ods()
    result = compute_camera_visible_efit_overlay(ods, SHOT, frame_index=0, flux_surface_levels=())
    assert result["flux_surfaces_uv"] == {}


def test_plot_camera_visible_efit_overlay_adds_scatter_layers():
    ods = _build_ods()
    fig, ax = plot_camera_visible_efit_overlay(ods, SHOT, frame_index=0, show=False)
    # base image + at least wall/lcfs/mag-axis/flux-surface scatter layers
    assert len(ax.images) == 1
    assert len(ax.collections) >= 3
    labels = [c.get_label() for c in ax.collections]
    assert "Wall" in labels
    assert "LCFS" in labels
    assert "Magnetic axis" in labels
    plt.close(fig)


def test_plot_camera_visible_efit_overlay_respects_visibility_flags():
    ods = _build_ods()
    fig, ax = plot_camera_visible_efit_overlay(
        ods, SHOT, frame_index=0, show_wall=False, show_magnetic_axis=False,
        flux_surface_levels=(), show=False,
    )
    labels = [c.get_label() for c in ax.collections]
    assert "Wall" not in labels
    assert "Magnetic axis" not in labels
    assert "LCFS" in labels
    plt.close(fig)


def test_plot_camera_visible_efit_overlay_title_mentions_shot():
    ods = _build_ods()
    fig, ax = plot_camera_visible_efit_overlay(ods, SHOT, frame_index=2, show=False)
    assert str(SHOT) in ax.get_title()
    plt.close(fig)
