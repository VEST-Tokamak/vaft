import numpy as np
import pytest

from vaft.process.camera_geometry import project_points, sweep_toroidal, toroidal_ring
from vaft.process.equilibrium import extract_flux_surface_contours
from vaft.omas.process_wrapper import _load_camera_intrinsics, _load_camera_pose


def _identity_camera():
    camera_matrix = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    dist_coeffs = np.zeros(5)
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    return camera_matrix, dist_coeffs, rvec, tvec


def test_project_points_maps_point_on_axis_to_principal_point():
    camera_matrix, dist_coeffs, rvec, tvec = _identity_camera()
    # Camera looks down +Z (no rotation); a point straight ahead on the axis
    # projects to the principal point regardless of depth.
    world = np.array([[0.0, 0.0, 500.0]])
    uv, valid = project_points(world, rvec, tvec, camera_matrix, dist_coeffs)
    np.testing.assert_allclose(uv[0], [50.0, 50.0], atol=1e-9)
    assert valid[0]


def test_project_points_flags_points_behind_camera_invalid():
    camera_matrix, dist_coeffs, rvec, tvec = _identity_camera()
    world = np.array([[0.0, 0.0, 500.0], [0.0, 0.0, -500.0]])
    _, valid = project_points(world, rvec, tvec, camera_matrix, dist_coeffs)
    assert valid[0]
    assert not valid[1]


def test_project_points_flags_extreme_viewing_angle_invalid():
    # A point far off-axis (large normalized x/z) is in front of the camera
    # but well outside the distortion model's valid angular range; without
    # excluding it, cv2's distortion polynomial can map it to a wild pixel
    # value rather than simply extrapolating off-frame -- this reproduces a
    # real failure seen sweeping the VEST wall geometry through +/-90 deg.
    camera_matrix, dist_coeffs, rvec, tvec = _identity_camera()
    dist_coeffs = np.array([-0.33, 0.1427, 0.0, 0.0, -0.0326])  # real VEST distortion
    world = np.array([[0.0, 0.0, 500.0], [5000.0, 0.0, 500.0]])  # xn=10 for the second point
    uv, valid = project_points(world, rvec, tvec, camera_matrix, dist_coeffs)
    assert valid[0]
    assert not valid[1]


def test_project_points_max_normalized_radius_is_configurable():
    camera_matrix, dist_coeffs, rvec, tvec = _identity_camera()
    world = np.array([[750.0, 0.0, 500.0]])  # xn = 1.5, outside default 1.3 but inside 2.0
    _, valid_default = project_points(world, rvec, tvec, camera_matrix, dist_coeffs)
    _, valid_wide = project_points(world, rvec, tvec, camera_matrix, dist_coeffs, max_normalized_radius=2.0)
    assert not valid_default[0]
    assert valid_wide[0]


def test_sweep_toroidal_shape_and_single_theta_reduces_to_input_point():
    r = np.array([1.0])
    z = np.array([0.5])
    xyz = sweep_toroidal(r, z, np.array([0.0]))
    assert xyz.shape == (1, 3)
    np.testing.assert_allclose(xyz[0], [100.0, 0.0, 50.0])  # meters -> cm


def test_sweep_toroidal_multiple_theta_and_points():
    r = np.array([1.0, 2.0])
    z = np.array([0.0, 0.0])
    theta = np.array([0.0, np.pi / 2])
    xyz = sweep_toroidal(r, z, theta)
    assert xyz.shape == (4, 3)
    # theta=0: (r,0,0)*100; theta=pi/2: (0,r,0)*100
    np.testing.assert_allclose(xyz[:2], [[100.0, 0.0, 0.0], [200.0, 0.0, 0.0]], atol=1e-9)
    np.testing.assert_allclose(xyz[2:], [[0.0, 100.0, 0.0], [0.0, 200.0, 0.0]], atol=1e-9)


def test_toroidal_ring_matches_sweep_toroidal_single_point():
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ring = toroidal_ring(0.6, 0.1, theta)
    expected = sweep_toroidal(np.array([0.6]), np.array([0.1]), theta)
    np.testing.assert_allclose(ring, expected)


def test_extract_flux_surface_contours_finds_circle_at_known_radius():
    r0 = 0.5
    R = np.linspace(0.0, 1.0, 101)
    Z = np.linspace(-0.5, 0.5, 101)
    Rm, Zm = np.meshgrid(R, Z, indexing="ij")
    # psi_norm = (R-r0)^2 + Z^2 -> level 0.04 is a circle of radius 0.2 around (r0, 0)
    psi_axis = 0.0
    psi_boundary = 1.0
    psi_grid = (Rm - r0) ** 2 + Zm**2
    contours = extract_flux_surface_contours(psi_grid, R, Z, psi_axis, psi_boundary, [0.04])
    segments = contours[0.04]
    assert len(segments) >= 1
    r_pts, z_pts = segments[0]
    radii = np.hypot(r_pts - r0, z_pts)
    np.testing.assert_allclose(radii, 0.2, atol=0.02)


def test_extract_flux_surface_contours_rejects_degenerate_normalization():
    R = np.linspace(0, 1, 5)
    Z = np.linspace(0, 1, 5)
    psi_grid = np.zeros((5, 5))
    with pytest.raises(ValueError):
        extract_flux_surface_contours(psi_grid, R, Z, psi_axis=1.0, psi_boundary=1.0, levels_norm=[0.5])


def test_extract_flux_surface_contours_rejects_wrong_shape():
    R = np.linspace(0, 1, 5)
    Z = np.linspace(0, 1, 7)
    psi_grid = np.zeros((5, 5))
    with pytest.raises(ValueError):
        extract_flux_surface_contours(psi_grid, R, Z, psi_axis=0.0, psi_boundary=1.0, levels_norm=[0.5])


# --- Validation against the real VEST_Fast Camera_Diagnostics calibration data ---
#
# These hardcode literal values extracted directly from
# camera_geometry.ipynb (compute_port_world_points, cell 14) and real clicked
# calibration pixel coordinates (calib_points_39915.json / the v11 MATLAB
# script's hardcoded corners for 34764, cell 45) -- not read from the external
# repo at test time, so these tests have no runtime dependency on it. They
# confirm vaft.process.camera_geometry.project_points reproduces the
# notebook's own self-reported reprojection error using the packaged
# intrinsics/pose.

def _compute_port_world_points() -> np.ndarray:
    Zport_up = 0.57 - 0.2355
    Zport_down = Zport_up - 0.692
    Rport = 0.803
    angle_port_width = np.arccos(1 - 0.24**2 / (2 * Rport**2))

    def port_corners(base_angle: float) -> np.ndarray:
        a0, a1, ac = base_angle, base_angle + angle_port_width, base_angle + angle_port_width / 2
        Xl, Yl = Rport * np.cos(a0), Rport * np.sin(a0)
        Xr, Yr = Rport * np.cos(a1), Rport * np.sin(a1)
        Xc, Yc = Rport * np.cos(ac), Rport * np.sin(ac)
        Zm = (Zport_up + Zport_down) / 2
        return np.array(
            [
                [Xr, Yr, Zport_up], [Xr, Yr, Zport_down], [Xl, Yl, Zport_down], [Xl, Yl, Zport_up],
                [Xr, Yr, Zm], [Xc, Yc, Zport_down], [Xl, Yl, Zm], [Xc, Yc, Zport_up],
            ]
        ) * 100.0

    a1 = np.deg2rad(30.0) - angle_port_width / 2
    a2 = a1 + np.deg2rad(120.0)
    return np.vstack([port_corners(a1), port_corners(a2)])


# calib_points_39915.json, in the exact point-order the notebook pairs with
# world_pts (cell 16): world_pts[0:8] <-> E*_2 (port 1, ~30 deg), world_pts[8:16] <-> E* (port 2, ~150 deg).
_CLICKED_39915 = {
    "E1": [221, 521], "E2": [226, 790], "E3": [296, 776], "E4": [292, 535],
    "E12": [216, 658], "E23": [263, 779], "E34": [289, 651], "E41": [262, 525],
    "E1_2": [733, 529], "E2_2": [741, 765], "E3_2": [807, 776], "E4_2": [802, 511],
    "E12_2": [744, 647], "E23_2": [780, 770], "E34_2": [813, 648], "E41_2": [773, 522],
}


def test_project_points_reproduces_notebook_reprojection_error_for_39915():
    world_pts = _compute_port_world_points()
    order = [
        "E1_2", "E2_2", "E3_2", "E4_2", "E12_2", "E23_2", "E34_2", "E41_2",
        "E1", "E2", "E3", "E4", "E12", "E23", "E34", "E41",
    ]
    clicked = np.array([_CLICKED_39915[key] for key in order], dtype=float)

    intrinsics = _load_camera_intrinsics()
    pose = _load_camera_pose(39915)
    uv, valid = project_points(
        world_pts, pose["rvec"], pose["tvec"], intrinsics["camera_matrix"], intrinsics["dist_coeffs"]
    )
    assert valid.all()
    errors = np.linalg.norm(uv - clicked, axis=1)
    # Notebook's own reported reprojection error for this shot: mean 3.487 px, max 5.511 px.
    assert errors.mean() == pytest.approx(3.487, abs=0.05)
    assert errors.max() == pytest.approx(5.511, abs=0.05)


def test_project_points_reproduces_notebook_self_consistency_for_34764():
    import cv2

    world_pts = _compute_port_world_points()
    # v11 MATLAB script's hardcoded undistorted-frame port-corner pixels (cell 45).
    clicked_undistorted = np.array(
        [
            [1391, 1059], [1394, 1313], [1472, 1334], [1478, 1039],
            [1394, 1187], [1433, 1331], [1475, 1187], [1433, 1049],
            [827, 1043], [827, 1343], [918, 1319], [916, 1065],
            [827, 1194], [876, 1332], [919, 1193], [873, 1056],
        ],
        dtype=float,
    )

    intrinsics = _load_camera_intrinsics()
    pose = _load_camera_pose(34764)
    # newOrigin is packaged in pose_34764.json alongside rvec/tvec but is not
    # exposed by _load_camera_pose (only used to *derive* this pose, not to
    # project with it -- see module docstring). Read it directly here only to
    # reproduce the notebook's own self-consistency check in the
    # UNDISTORTED-frame coordinate system the click points were recorded in.
    from vaft.machine_mapping import resolve_geometry_asset
    import json

    with open(resolve_geometry_asset("camera_visible/pose_34764.json")) as handle:
        raw_pose = json.load(handle)
    new_origin = np.array(raw_pose["newOrigin"], dtype=float)

    proj, _ = cv2.projectPoints(
        world_pts.astype(np.float64),
        pose["rvec"].reshape(3, 1),
        pose["tvec"].reshape(3, 1),
        intrinsics["camera_matrix"],
        np.zeros(5),  # undistorted frame -> zero distortion, matching the notebook's solve
    )
    proj = proj.reshape(-1, 2)
    errors = np.linalg.norm(proj - (clicked_undistorted + new_origin), axis=1)
    # Notebook's own reported reprojection error for this shot's calibration solve.
    assert errors.mean() == pytest.approx(2.928, abs=0.05)
    assert errors.max() == pytest.approx(6.255, abs=0.05)
