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
    """Project world points into camera pixels, flagging the ones the model cannot be trusted for.

    Parameters
    ----------
    world_xyz_cm : array_like
        World points as ``(N, 3)``, in the machine frame [cm].
    rvec : array_like
        Rotation vector of the calibrated camera pose, Rodrigues form [rad].
    tvec : array_like
        Translation vector of that pose [cm].
    camera_matrix : array_like
        Intrinsic matrix, ``3x3`` [-].
    dist_coeffs : array_like
        Brown-Conrady radial and tangential distortion coefficients [-].
    max_normalized_radius : float, optional
        Largest normalized image-plane radius still considered inside the model's
        fitted range [-].

    Returns
    -------
    pixel_uv : np.ndarray
        Pixel positions as ``(N, 2)`` in column-then-row order [-].
    valid_mask : np.ndarray
        ``True`` where the projection can be trusted [-].

    Convention
    ----------
    **Pixels are returned column first, then row**, which is the opposite of the
    row-then-column order an image array is indexed with. **World points are in
    centimetres**, matching the packaged calibration, while every other geometry
    argument in this package is in metres; the sweep helpers here do that
    conversion.

    The pose is applied directly to the original distorted frame. Per the source
    notebook's own finding, the recovered pose is the true physical camera
    attitude whichever click-point convention solved for it, so no undistortion
    step is needed and one formula serves every calibrated shot.

    Defaults
    --------
    ``max_normalized_radius = 1.3`` is a validated-workflow default, the same
    guard the source notebook applies in its own projection routine.

    Applicability
    -------------
    Machine-independent. Standard OpenCV pinhole geometry; the calibration that
    makes it a VEST camera is supplied by the caller.

    Limitations
    -----------
    Two kinds of point are excluded rather than projected. A point behind the
    camera has non-positive depth and no meaningful pixel. A point far off the
    optical axis lies outside the angular range the distortion polynomial was
    fitted over, where that polynomial can turn non-monotonic and place the point
    at a wildly wrong pixel rather than smoothly off-frame; the mask catches that
    case, which is why it is a validity test and not a crop.

    Requires OpenCV, imported at call time.

    Provenance
    ----------
    .. [1] The VEST FAST camera diagnostics repository's ``camera_geometry.ipynb``
       and its ``CALIBRATION.md``; this reproduces that notebook's projection,
       including the depth and field-of-view guard from its own routine.
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
    """Sweep poloidal points through toroidal angles into world points.

    Parameters
    ----------
    r_m : array_like
        Major radius of each poloidal point [m].
    z_m : array_like
        Height of each poloidal point, same shape as *r_m* [m].
    theta_rad : array_like
        Toroidal angles to sweep through [rad].

    Returns
    -------
    np.ndarray
        World points as ``(len(theta) * len(r), 3)``, one row per angle and point
        combination [cm].

    Raises
    ------
    ValueError
        The two poloidal arrays do not have the same shape.

    Convention
    ----------
    ``X = R cos(theta)``, ``Y = R sin(theta)``, ``Z = Z``, then converted from
    metres to centimetres because the packaged calibration is in centimetres.
    Rows vary fastest over the poloidal points and slowest over the angles.

    This is an **outer product**: every point is swept through every angle, giving
    a surface. :func:`trajectory_world_points` pairs them element-wise instead,
    for a single ordered path. Confusing the two produces an array of the wrong
    length rather than an error.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] Reproduces the source notebook's torus sweep; see
       :func:`project_points` for the repository and calibration reference.
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
    """Sweep one poloidal point into a full toroidal ring.

    Parameters
    ----------
    r_m : float
        Major radius of the point [m].
    z_m : float
        Height of the point [m].
    theta_rad : array_like
        Toroidal angles making up the ring [rad].

    Returns
    -------
    np.ndarray
        World points as ``(len(theta), 3)`` [cm].

    Convention
    ----------
    The single-point case of :func:`sweep_toroidal`, with the same axis
    definitions and the same conversion from metres to centimetres.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] :func:`sweep_toroidal`, which this delegates to.
    """
    return sweep_toroidal(np.array([float(r_m)]), np.array([float(z_m)]), theta_rad)


def trajectory_world_points(r_m: np.ndarray, z_m: np.ndarray, phi_rad: np.ndarray) -> np.ndarray:
    """Convert a matched trajectory of cylindrical coordinates into world points.

    Parameters
    ----------
    r_m : array_like
        Major radius along the trajectory [m].
    z_m : array_like
        Height along the trajectory, same shape [m].
    phi_rad : array_like
        Toroidal angle at each point, same shape [rad].

    Returns
    -------
    np.ndarray
        World points as ``(N, 3)``, one row per trajectory point [cm].

    Raises
    ------
    ValueError
        The three arrays do not all have the same shape.

    Convention
    ----------
    **Element-wise pairing**, one angle per point, for an ordered path such as a
    traced field line. :func:`sweep_toroidal` takes the outer product instead and
    gives a surface. Same axis definitions and the same conversion to centimetres.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] The world-frame convention of the source notebook; see
       :func:`project_points`.
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
