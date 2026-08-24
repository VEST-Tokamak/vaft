import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from omas import ODS

from vaft.process.equilibrium import make_equilibrium_field_interpolator, trace_field_line
from vaft.process.camera_geometry import trajectory_world_points
from vaft.omas.process_wrapper import compute_field_line_trace, compute_camera_visible_field_line_overlay
from vaft.omas import plot_camera_visible_image_field_line as plot_camera_visible_field_line

SHOT = 39915  # a packaged calibrated shot

R0, LCFS_RADIUS = 0.4, 0.2
F0 = 0.1


def _build_field_interpolator(n_grid: int = 81):
    R = np.linspace(0.1, 0.9, n_grid)
    Z = np.linspace(-0.4, 0.4, n_grid)
    Rm, Zm = np.meshgrid(R, Z, indexing="ij")
    psi_grid = (Rm - R0) ** 2 + Zm**2  # concentric circular flux surfaces
    psi_1d = np.linspace(0.0, LCFS_RADIUS**2, 50)
    f_1d = np.full_like(psi_1d, F0)
    return make_equilibrium_field_interpolator(R, Z, psi_grid, psi_1d, f_1d), R, Z


def test_trace_field_line_conserves_psi_on_a_flux_surface():
    b_field, R, Z = _build_field_interpolator()
    r0_start, z0_start = R0 + 0.1, 0.0
    trace = trace_field_line(
        r0_start, z0_start, 0.0, b_field,
        dphi=np.deg2rad(0.5), max_length_m=20.0, direction="forward",
        r_bounds=(R.min(), R.max()), z_bounds=(Z.min(), Z.max()),
    )
    assert trace["termination_reason"] == "max_length_m"
    psi_along = (trace["R"] - R0) ** 2 + trace["Z"] ** 2
    psi_start = (r0_start - R0) ** 2 + z0_start**2
    rel_dev = np.abs(psi_along - psi_start) / psi_start
    assert rel_dev.max() < 1e-4


def test_trace_field_line_error_scales_as_rk4_fourth_order():
    b_field, R, Z = _build_field_interpolator()
    r0_start, z0_start = R0 + 0.1, 0.0
    psi_start = (r0_start - R0) ** 2 + z0_start**2

    def max_rel_dev(dphi_deg: float) -> float:
        trace = trace_field_line(
            r0_start, z0_start, 0.0, b_field,
            dphi=np.deg2rad(dphi_deg), max_length_m=10.0, direction="forward",
            r_bounds=(R.min(), R.max()), z_bounds=(Z.min(), Z.max()),
        )
        psi_along = (trace["R"] - R0) ** 2 + trace["Z"] ** 2
        return np.abs(psi_along - psi_start).max() / psi_start

    dev_coarse = max_rel_dev(2.0)
    dev_fine = max_rel_dev(1.0)
    # RK4 global error ~ O(dphi^4): halving the step should shrink the error
    # by at least ~2^3 (loose bound, avoids flaking on a marginal exponent).
    assert dev_fine < dev_coarse / 8.0


def test_trace_field_line_terminates_at_asymmetric_wall():
    b_field, R, Z = _build_field_interpolator()
    r0_start, z0_start = R0 + 0.1, 0.0
    wall_r = np.array([0.1, 0.9, 0.9, 0.1])
    wall_z = np.array([-0.05, -0.05, 0.4, 0.4])  # clips the bottom of the orbit
    trace = trace_field_line(
        r0_start, z0_start, 0.0, b_field,
        dphi=np.deg2rad(0.5), max_length_m=20.0, direction="forward",
        wall_r=wall_r, wall_z=wall_z,
        r_bounds=(R.min(), R.max()), z_bounds=(Z.min(), Z.max()),
    )
    assert trace["termination_reason"] == "wall"
    assert trace["arc_length_m"][-1] < 20.0
    assert trace["Z"][-1] < 0.0


def test_trace_field_line_direction_both_is_symmetric_about_start():
    b_field, R, Z = _build_field_interpolator()
    r0_start, z0_start = R0 + 0.1, 0.0
    forward = trace_field_line(r0_start, z0_start, 0.0, b_field, dphi=np.deg2rad(1.0), max_length_m=2.0, direction="forward")
    backward = trace_field_line(r0_start, z0_start, 0.0, b_field, dphi=np.deg2rad(1.0), max_length_m=2.0, direction="backward")
    both = trace_field_line(r0_start, z0_start, 0.0, b_field, dphi=np.deg2rad(1.0), max_length_m=2.0, direction="both")

    assert both["phi"].size == forward["phi"].size + backward["phi"].size - 1
    np.testing.assert_allclose(both["phi"][-forward["phi"].size:], forward["phi"])
    # both's phi array is monotonically increasing (backward branch reversed then forward appended)
    assert np.all(np.diff(both["phi"]) > 0)


def test_trace_field_line_arc_length_is_monotonic_for_backward():
    b_field, R, Z = _build_field_interpolator()
    r0_start, z0_start = R0 + 0.1, 0.0
    trace = trace_field_line(
        r0_start, z0_start, 0.0, b_field, dphi=np.deg2rad(1.0), max_length_m=2.0, direction="backward"
    )
    arc = trace["arc_length_m"]
    # arc_length_m is documented as cumulative along the returned array's
    # order, not distance from phi0 -- for "backward" the array is reversed
    # into increasing-phi order, so naively reusing the per-branch values
    # (which start at 0 at phi0, the *last* array element here) made this
    # monotonically decrease instead.
    assert arc[0] == pytest.approx(0.0)
    assert np.all(np.diff(arc) >= 0)
    assert arc[-1] > 0.0


def test_trace_field_line_arc_length_is_monotonic_for_both():
    b_field, R, Z = _build_field_interpolator()
    r0_start, z0_start = R0 + 0.1, 0.0
    trace = trace_field_line(
        r0_start, z0_start, 0.0, b_field, dphi=np.deg2rad(1.0), max_length_m=2.0, direction="both"
    )
    arc = trace["arc_length_m"]
    # Previously this decreased toward phi0 then increased again (a "V"
    # measuring distance *from* phi0), which isn't a monotonic running total
    # along the array and made distance-based consumers incorrect.
    assert arc[0] == pytest.approx(0.0)
    assert np.all(np.diff(arc) >= 0)
    assert arc[-1] == pytest.approx(arc.max())


def test_trace_field_line_rejects_invalid_direction():
    b_field, _, _ = _build_field_interpolator()
    with pytest.raises(ValueError):
        trace_field_line(0.5, 0.0, 0.0, b_field, direction="sideways")


def test_trace_field_line_rejects_nonpositive_dphi():
    b_field, _, _ = _build_field_interpolator()
    with pytest.raises(ValueError):
        trace_field_line(0.5, 0.0, 0.0, b_field, dphi=0.0)


def test_trajectory_world_points_matches_element_wise_conversion():
    r = np.array([0.5, 0.6])
    z = np.array([0.1, -0.1])
    phi = np.array([0.0, np.pi / 2])
    xyz = trajectory_world_points(r, z, phi)
    expected = np.array([[50.0, 0.0, 10.0], [0.0, 60.0, -10.0]])
    np.testing.assert_allclose(xyz, expected, atol=1e-9)


def test_trajectory_world_points_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        trajectory_world_points(np.array([0.5, 0.6]), np.array([0.1]), np.array([0.0, 0.0]))


# --- ODS-facing wrapper + plotting, using the same synthetic camera_visible +
# equilibrium + wall ODS shape as test_camera_visible_efit_overlay.py ---

def _build_ods(n_frames: int = 4, shape: tuple[int, int] = (12, 16)) -> ODS:
    ods = ODS(consistency_check=False)
    ods["camera_visible.channel.0.name"] = "Fast Camera"
    ods["camera_visible.channel.0.detector.0.lines_n"] = shape[0]
    ods["camera_visible.channel.0.detector.0.columns_n"] = shape[1]
    for i in range(n_frames):
        ods[f"camera_visible.channel.0.detector.0.frame.{i}.image_raw"] = np.full(shape, i * 10, dtype=int)
        ods[f"camera_visible.channel.0.detector.0.frame.{i}.time"] = 0.28 + i * 0.001

    r0, z0, lcfs_radius = R0, 0.0, LCFS_RADIUS
    R_grid = np.linspace(0.1, 0.9, 81)
    Z_grid = np.linspace(-0.4, 0.4, 81)
    Rm, Zm = np.meshgrid(R_grid, Z_grid, indexing="ij")
    psi_grid = (Rm - r0) ** 2 + Zm**2
    psi_1d = np.linspace(0.0, lcfs_radius**2, 50)
    f_1d = np.full_like(psi_1d, F0)

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
        ods[f"{ts}.global_quantities.psi_axis"] = 0.0
        ods[f"{ts}.global_quantities.psi_boundary"] = lcfs_radius**2
        ods[f"{ts}.profiles_2d.0.grid.dim1"] = R_grid
        ods[f"{ts}.profiles_2d.0.grid.dim2"] = Z_grid
        ods[f"{ts}.profiles_2d.0.psi"] = psi_grid
        ods[f"{ts}.profiles_1d.psi"] = psi_1d
        ods[f"{ts}.profiles_1d.f"] = f_1d

    theta_wall = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = r0 + 0.35 * np.cos(theta_wall)
    ods["wall.description_2d.0.limiter.unit.0.outline.z"] = 0.35 * np.sin(theta_wall)

    return ods


def test_compute_field_line_trace_returns_expected_keys():
    ods = _build_ods()
    result = compute_field_line_trace(ods, r0=R0 + 0.1, z0=0.0, phi0=0.0, dphi_deg=1.0, max_length_m=5.0)
    assert set(result) >= {
        "phi", "R", "Z", "arc_length_m", "termination_reason",
        "equilibrium_time_index", "equilibrium_time", "start_point", "dphi_deg", "max_length_m", "direction",
    }
    assert result["start_point"] == {"r0": R0 + 0.1, "z0": 0.0, "phi0": 0.0}
    assert result["equilibrium_time_index"] == 0  # defaults to first slice when no time given


def test_compute_field_line_trace_time_index_selects_slice():
    ods = _build_ods()
    result = compute_field_line_trace(ods, r0=R0 + 0.1, z0=0.0, time_index=1, dphi_deg=1.0, max_length_m=1.0)
    assert result["equilibrium_time_index"] == 1


def test_compute_field_line_trace_rejects_time_and_time_index_together():
    ods = _build_ods()
    with pytest.raises(ValueError):
        compute_field_line_trace(ods, r0=R0 + 0.1, z0=0.0, time=0.28, time_index=0)


def test_compute_camera_visible_field_line_overlay_returns_projected_points():
    ods = _build_ods()
    result = compute_camera_visible_field_line_overlay(
        ods, SHOT, r0=R0 + 0.1, z0=0.0, frame_index=0, dphi_deg=1.0, max_length_m=5.0,
    )
    assert set(result) == {
        "frame_index", "frame_time", "equilibrium_time_index", "equilibrium_time",
        "image_shape", "field_line_uv", "field_line_valid", "trace",
    }
    assert result["field_line_uv"].shape[1] == 2
    assert result["field_line_uv"].shape[0] >= 1


def test_compute_camera_visible_field_line_overlay_preserves_discontinuities(monkeypatch):
    """A gap of invalid samples in the middle of a trace must survive as NaN,
    not be compacted away -- compacting would let a renderer draw a single
    connected polyline through the gap, fabricating a segment that doesn't
    correspond to any real part of the field line (see PR #128 review).
    """
    import vaft.omas.process_wrapper as pw

    ods = _build_ods()
    captured = {}

    def fake_project_points(world_cm, rvec, tvec, camera_matrix, dist_coeffs):
        n = world_cm.shape[0]
        captured["n"] = n
        uv = np.column_stack([np.arange(n, dtype=float), np.arange(n, dtype=float)])
        valid = np.ones(n, dtype=bool)
        mid = n // 2
        valid[mid : mid + 2] = False  # a deliberate gap, not just a single dropped point
        return uv, valid

    monkeypatch.setattr(pw, "project_points", fake_project_points)

    result = compute_camera_visible_field_line_overlay(
        ods, SHOT, r0=R0 + 0.1, z0=0.0, frame_index=0, dphi_deg=1.0, max_length_m=5.0,
    )
    uv = result["field_line_uv"]
    valid = result["field_line_valid"]
    n = captured["n"]

    # Not compacted: still the full trajectory length, gap included.
    assert uv.shape == (n, 2)
    assert valid.shape == (n,)
    assert (~valid).sum() >= 2

    nan_rows = np.isnan(uv).any(axis=1)
    np.testing.assert_array_equal(nan_rows, ~valid)
    # Valid rows keep their real projected positions, unperturbed.
    np.testing.assert_array_equal(uv[valid], np.column_stack([np.arange(n), np.arange(n)])[valid])


def test_compute_camera_visible_field_line_overlay_unsupported_shot_raises():
    ods = _build_ods()
    with pytest.raises(FileNotFoundError):
        compute_camera_visible_field_line_overlay(ods, 11111, r0=R0 + 0.1, z0=0.0, frame_index=0)


def test_plot_camera_visible_field_line_draws_line_and_endpoints():
    ods = _build_ods()
    fig, ax = plot_camera_visible_field_line(
        ods, shot=SHOT, r0=R0 + 0.1, z0=0.0, frame_index=0, dphi_deg=1.0, max_length_m=5.0, show=False,
    )
    assert len(ax.images) == 1
    labels = [line.get_label() for line in ax.lines]
    assert "Field line" in labels
    assert "Start" in labels
    plt.close(fig)


def test_plot_camera_visible_field_line_can_include_efit_overlay():
    ods = _build_ods()
    fig, ax = plot_camera_visible_field_line(
        ods, shot=SHOT, r0=R0 + 0.1, z0=0.0, frame_index=0, dphi_deg=1.0, max_length_m=5.0,
        show_wall=True, show_lcfs=True, show_magnetic_axis=True, show=False,
    )
    labels = [line.get_label() for line in ax.lines]
    assert "Wall" in labels
    assert "LCFS" in labels
    assert "Magnetic axis" in labels
    plt.close(fig)
