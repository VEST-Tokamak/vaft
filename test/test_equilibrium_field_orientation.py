"""Regression coverage for a real bug: a shape-equality transpose heuristic
that silently corrupted a correctly-oriented SQUARE psi grid (EFIT's common
129x129/65x65 default -- verified against a real VEST GEQDSK, see the fix in
vaft/omas/process_wrapper.py::_read_psi_grid). These tests use a square grid
with an intentionally ELLIPTICAL (R != Z scaled) psi field specifically so a
transpose is numerically detectable via psi_N deviating from 0 at the axis
and from 1 along the LCFS -- a symmetric/circular psi field would not have
caught this, since swapping R and Z leaves it unchanged.
"""

import numpy as np
import pytest
from omas import ODS

from vaft.omas.process_wrapper import (
    _equilibrium_field_slice_data,
    _read_psi_grid,
    compute_camera_visible_efit_overlay,
    compute_field_line_trace,
)

R0, Z0 = 0.5, 0.05  # off-midplane center, to also catch a Z-offset/sign error
A, B = 0.25, 0.15  # elliptical semi-axes: a != b, R-scale != Z-scale
N_GRID = 65  # square grid, matching EFIT's common square default


def _build_elliptical_ods(n_frames: int = 2) -> ODS:
    ods = ODS(consistency_check=False)

    R_grid = np.linspace(0.1, 0.9, N_GRID)
    Z_grid = np.linspace(-0.4, 0.4, N_GRID)
    assert R_grid.size == Z_grid.size == N_GRID  # square grid, the ambiguous case
    Rm, Zm = np.meshgrid(R_grid, Z_grid, indexing="ij")  # psi_grid[i, j] = f(R_grid[i], Z_grid[j])
    # PSI_SCALE keeps psi at a realistic EFIT Weber/rad magnitude (compare the
    # real VEST 39915 equilibrium: psi_axis=-0.0022, psi_boundary=0.0014). An
    # order-1 psi here would imply an unphysical ~90 T poloidal field (B_R ~
    # (1/R) dpsi/dZ), which starves the field-line tracer's fixed dphi step of
    # resolution and produces large, fast-growing psi deviation that looks
    # like a tracer bug but is really just a badly scaled test fixture.
    PSI_SCALE = 0.01
    psi_grid = PSI_SCALE * (((Rm - R0) / A) ** 2 + ((Zm - Z0) / B) ** 2)  # elliptical, R/Z NOT interchangeable
    psi_axis, psi_boundary = 0.0, PSI_SCALE  # LCFS at the psi=PSI_SCALE ellipse itself
    psi_1d = np.linspace(psi_axis, psi_boundary, 50)
    f_1d = np.full_like(psi_1d, 0.1)

    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    lcfs_r = R0 + A * np.cos(theta)
    lcfs_z = Z0 + B * np.sin(theta)

    ods["camera_visible.channel.0.detector.0.lines_n"] = 12
    ods["camera_visible.channel.0.detector.0.columns_n"] = 16
    for i in range(n_frames):
        ods[f"camera_visible.channel.0.detector.0.frame.{i}.image_raw"] = np.zeros((12, 16), dtype=int)
        ods[f"camera_visible.channel.0.detector.0.frame.{i}.time"] = 0.28 + i * 0.001

    ods["equilibrium.time"] = np.array([0.28 + i * 0.001 for i in range(n_frames)])
    for idx in range(n_frames):
        ts = f"equilibrium.time_slice.{idx}"
        ods[f"{ts}.time"] = float(ods["equilibrium.time"][idx])
        ods[f"{ts}.boundary.outline.r"] = lcfs_r
        ods[f"{ts}.boundary.outline.z"] = lcfs_z
        ods[f"{ts}.global_quantities.magnetic_axis.r"] = R0
        ods[f"{ts}.global_quantities.magnetic_axis.z"] = Z0
        ods[f"{ts}.global_quantities.psi_axis"] = psi_axis
        ods[f"{ts}.global_quantities.psi_boundary"] = psi_boundary
        ods[f"{ts}.profiles_2d.0.grid.dim1"] = R_grid
        ods[f"{ts}.profiles_2d.0.grid.dim2"] = Z_grid
        ods[f"{ts}.profiles_2d.0.psi"] = psi_grid  # shape (N_GRID, N_GRID) = (R, Z), per to_omas's convention
        ods[f"{ts}.profiles_1d.psi"] = psi_1d
        ods[f"{ts}.profiles_1d.f"] = f_1d

    theta_wall = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = R0 + 0.35 * np.cos(theta_wall)
    ods["wall.description_2d.0.limiter.unit.0.outline.z"] = Z0 + 0.35 * np.sin(theta_wall)

    return ods


def test_read_psi_grid_preserves_square_grid_orientation():
    ods = _build_elliptical_ods()
    ts = ods["equilibrium.time_slice.0"]
    R_grid = np.asarray(ts["profiles_2d.0.grid.dim1"])
    Z_grid = np.asarray(ts["profiles_2d.0.grid.dim2"])
    psi_grid = _read_psi_grid(ts, R_grid, Z_grid)
    assert psi_grid.shape == (N_GRID, N_GRID)
    # spot-check against the analytic elliptical field directly (no transpose ambiguity here).
    i, j = 40, 10
    expected = 0.01 * (((R_grid[i] - R0) / A) ** 2 + ((Z_grid[j] - Z0) / B) ** 2)  # PSI_SCALE from _build_elliptical_ods
    assert psi_grid[i, j] == pytest.approx(expected, rel=1e-9)


def test_read_psi_grid_raises_on_genuinely_wrong_shape():
    ods = _build_elliptical_ods()
    ts = ods["equilibrium.time_slice.0"]
    R_grid = np.asarray(ts["profiles_2d.0.grid.dim1"])
    Z_grid = np.linspace(-0.4, 0.4, N_GRID + 1)  # deliberately mismatched size
    with pytest.raises(ValueError, match="expected"):
        _read_psi_grid(ts, R_grid, Z_grid)


def test_equilibrium_field_slice_data_psi_norm_correct_at_axis_and_lcfs():
    """The quantitative check the user asked for: psi_N ~ 0 at the magnetic
    axis and ~1 along the supplied boundary.outline, computed through the
    actual interpolator used by the field-line tracer -- on a SQUARE grid,
    which is exactly the case that silently broke before the fix.
    """
    from scipy.interpolate import RectBivariateSpline

    ods = _build_elliptical_ods()
    ts = ods["equilibrium.time_slice.0"]
    data = _equilibrium_field_slice_data(ts)

    spline = RectBivariateSpline(data["R_grid"], data["Z_grid"], data["psi_grid"])
    psi_axis = float(ts["global_quantities.psi_axis"])
    psi_boundary = float(ts["global_quantities.psi_boundary"])

    psiN_axis = (float(spline.ev(R0, Z0)) - psi_axis) / (psi_boundary - psi_axis)
    assert psiN_axis == pytest.approx(0.0, abs=1e-3)

    lcfs_r = np.asarray(ts["boundary.outline.r"])
    lcfs_z = np.asarray(ts["boundary.outline.z"])
    psiN_lcfs = (spline.ev(lcfs_r, lcfs_z) - psi_axis) / (psi_boundary - psi_axis)
    assert psiN_lcfs.mean() == pytest.approx(1.0, abs=1e-2)
    assert psiN_lcfs.std() < 0.05


def test_field_line_trace_stays_near_its_starting_psi_on_square_grid():
    """End-to-end: a field line traced on this square-grid elliptical
    equilibrium must stay on its flux surface (as in the circular synthetic
    case), which fails badly if the grid got silently transposed.
    """
    ods = _build_elliptical_ods()
    r0_start, z0_start = R0 + 0.5 * A, Z0
    trace = compute_field_line_trace(
        ods, r0=r0_start, z0=z0_start, phi0=0.0, time_index=0,
        dphi_deg=0.5, max_length_m=5.0, direction="forward", use_wall_boundary=False,
    )
    psi_start = ((r0_start - R0) / A) ** 2 + ((z0_start - Z0) / B) ** 2
    psi_along = ((trace["R"] - R0) / A) ** 2 + ((trace["Z"] - Z0) / B) ** 2
    rel_dev = np.abs(psi_along - psi_start) / psi_start
    assert rel_dev.max() < 1e-2


def test_compute_camera_visible_efit_overlay_flux_surfaces_on_square_grid_match_lcfs():
    """The flux-surface projection at level=1.0 should trace the same curve
    as the (independently supplied) LCFS boundary.outline -- a strong
    self-consistency check that fails if the psi grid is transposed.
    """
    ods = _build_elliptical_ods()
    result = compute_camera_visible_efit_overlay(
        ods, 39915, frame_index=0, flux_surface_levels=(1.0,),
    )
    assert result["flux_surfaces_uv"][1.0].shape[0] > 0
    # both projections should land in a similar pixel region (same physical curve).
    lcfs_center = result["lcfs_uv"].mean(axis=0)
    fs_center = result["flux_surfaces_uv"][1.0].mean(axis=0)
    np.testing.assert_allclose(fs_center, lcfs_center, atol=15.0)
