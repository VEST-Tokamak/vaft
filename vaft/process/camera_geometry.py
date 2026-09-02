"""Pinhole camera projection math for the VEST FAST camera.

Pure array-in/array-out functions (no ODS, no file I/O) mirroring the style of
``vaft.process.equilibrium``. Reproduces the forward/virtual-camera projection
used by the VEST_Fast Camera_Diagnostics repository's ``camera_geometry.ipynb``
(see that repo's ``CALIBRATION.md``): a standard OpenCV pinhole model with
Brown-Conrady radial/tangential distortion, `cv2.projectPoints`, applied
directly with a shot's calibrated ``(rvec, tvec)`` pose onto the ORIGINAL
(distorted) camera frame. Per that notebook's own documented finding (cell 50),
the recovered pose is the true physical camera attitude regardless of which
click-point convention was used to solve for it, so no undistortion or
``newOrigin`` handling is needed at projection time -- one formula for every
calibrated shot.

World points are in centimeters throughout, matching the packaged calibration
convention: ``(X, Y, Z)_cm = (R_m * cos(theta), R_m * sin(theta), Z_m) * 100``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CameraProjection:
    """How machine coordinates become camera pixels (issue #261 sections 20-21).

    One calibrated pinhole model: intrinsics (``camera_matrix``, Brown-Conrady
    ``dist_coeffs``), the camera pose for a shot (``rvec``, ``tvec``), the
    ``method`` that produced it, and ``provenance`` -- where the numbers came
    from and how well they reproject.  Overlays of any kind (wall, LCFS,
    field line) go through :meth:`project`; nothing else in the plotting
    path knows a focal length.
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    method: str = "calibrated"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_matrix", np.asarray(self.camera_matrix, dtype=float).reshape(3, 3))
        object.__setattr__(self, "dist_coeffs", np.asarray(self.dist_coeffs, dtype=float).reshape(-1))
        object.__setattr__(self, "rvec", np.asarray(self.rvec, dtype=float).reshape(3))
        object.__setattr__(self, "tvec", np.asarray(self.tvec, dtype=float).reshape(3))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def project(self, world_xyz_cm: np.ndarray, *, max_normalized_radius: float = 1.3) -> tuple[np.ndarray, np.ndarray]:
        """``(pixel_uv, valid_mask)`` for 3-D world points in centimetres."""
        return project_points(
            world_xyz_cm, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs,
            max_normalized_radius=max_normalized_radius,
        )

    def project_rz(self, r_m: np.ndarray, z_m: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
        """Valid pixels of a poloidal (R, Z) curve swept toroidally over ``theta_rad``."""
        pixel_uv, valid = self.project(sweep_toroidal(r_m, z_m, theta_rad))
        return pixel_uv[valid]

    def project_ring(self, r_m: float, z_m: float, theta_rad: np.ndarray) -> np.ndarray:
        """Valid pixels of one toroidal ring at (R, Z)."""
        pixel_uv, valid = self.project(toroidal_ring(r_m, z_m, theta_rad))
        return pixel_uv[valid]


def project_points(
    world_xyz_cm: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    max_normalized_radius: float = 1.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D world points (cm) into 2D pixel coordinates.

    Returns ``(pixel_uv, valid_mask)`` where ``pixel_uv`` is ``(N, 2)`` in
    ``(col, row)`` pixel order. ``valid_mask`` excludes two kinds of points
    the OpenCV Brown-Conrady distortion model cannot be trusted for: points
    behind the camera (non-positive depth in the camera frame), and points
    at a large angle from the optical axis (normalized image-plane radius
    ``hypot(x/z, y/z) >= max_normalized_radius``) -- outside the angular
    range the distortion polynomial was fit over, it can turn non-monotonic
    and map a point to a wildly incorrect pixel location far from the actual
    image, rather than simply extrapolating smoothly off-frame. This mirrors
    the ``depth > 0 & hypot(xn, yn) < rmax`` guard the source notebook uses in
    its own ``project34`` (default ``rmax=1.3``, cell 45).
    """
    import cv2

    world = np.asarray(world_xyz_cm, dtype=np.float64).reshape(-1, 3)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)

    pixel_uv, _ = cv2.projectPoints(world, rvec, tvec, camera_matrix, dist_coeffs)
    pixel_uv = pixel_uv.reshape(-1, 2)

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    camera_frame = (rotation_matrix @ world.T + tvec).T
    depth = camera_frame[:, 2]
    in_front = depth > 0

    with np.errstate(divide="ignore", invalid="ignore"):
        xn = np.where(in_front, camera_frame[:, 0] / depth, np.inf)
        yn = np.where(in_front, camera_frame[:, 1] / depth, np.inf)
    within_fov = np.hypot(xn, yn) < max_normalized_radius

    valid_mask = in_front & within_fov

    return pixel_uv, valid_mask


def sweep_toroidal(r_m: np.ndarray, z_m: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
    """Sweep poloidal ``(r_m, z_m)`` points through toroidal angles ``theta_rad``.

    Returns stacked ``(X, Y, Z)`` world points in centimeters, one row per
    ``(theta, point)`` combination -- ``X = r*cos(theta)``, ``Y = r*sin(theta)``,
    ``Z = z``, all *100 (m -> cm). Reproduces the source notebook's
    ``sweep_torus``.
    """
    r = np.asarray(r_m, dtype=np.float64).reshape(-1)
    z = np.asarray(z_m, dtype=np.float64).reshape(-1)
    theta = np.asarray(theta_rad, dtype=np.float64).reshape(-1)
    if r.shape != z.shape:
        raise ValueError("r_m and z_m must have the same shape.")

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x = np.outer(cos_theta, r).reshape(-1)
    y = np.outer(sin_theta, r).reshape(-1)
    z_tiled = np.tile(z, theta.size)

    return np.stack([x, y, z_tiled], axis=1) * 100.0


def toroidal_ring(r_m: float, z_m: float, theta_rad: np.ndarray) -> np.ndarray:
    """Sweep a single ``(r_m, z_m)`` point into a toroidal ring, in centimeters."""
    return sweep_toroidal(np.array([float(r_m)]), np.array([float(z_m)]), theta_rad)


def trajectory_world_points(r_m: np.ndarray, z_m: np.ndarray, phi_rad: np.ndarray) -> np.ndarray:
    """Convert a matched ``(R, Z, phi)`` trajectory to world points, in centimeters.

    Unlike :func:`sweep_toroidal` (which sweeps poloidal points through every
    toroidal angle -- an outer product), this maps one ``phi`` per ``(R, Z)``
    element-wise, e.g. for a field-line trace where ``R``, ``Z``, ``phi`` are
    a single ordered trajectory, not a swept surface.
    """
    r = np.asarray(r_m, dtype=np.float64).reshape(-1)
    z = np.asarray(z_m, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi_rad, dtype=np.float64).reshape(-1)
    if not (r.shape == z.shape == phi.shape):
        raise ValueError("r_m, z_m, and phi_rad must have the same shape.")

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.stack([x, y, z], axis=1) * 100.0


__all__ = [
    "CameraProjection",
    "project_points",
    "sweep_toroidal",
    "toroidal_ring",
    "trajectory_world_points",
]
