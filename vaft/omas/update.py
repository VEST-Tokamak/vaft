"""
Update derived quantities for the ods data structure.
"""

import numpy as np
from scipy.interpolate import interp1d
import logging
from vaft.formula import normalize_psi
from vaft.formula.constants import MU0
from vaft.process.equilibrium import psi_to_rz, volume_average
from omas import *
from vaft.compat import trapz_compat

# update_diagnostics_file(ods, filename)

# print_available_ids(ods)
logger = logging.getLogger(__name__)


def _geometric_flux_surface_grid(ts, n_theta: int):
    """Build a dependency-free flux-surface-like grid from axis and boundary."""
    axis_r = float(ts["global_quantities.magnetic_axis.r"])
    axis_z = float(ts["global_quantities.magnetic_axis.z"])
    psi_axis = float(ts["global_quantities.psi_axis"])
    psi_boundary = float(ts["global_quantities.psi_boundary"])
    boundary_r = np.asarray(ts["boundary.outline.r"], dtype=float)
    boundary_z = np.asarray(ts["boundary.outline.z"], dtype=float)
    if boundary_r.size < 3 or boundary_z.size < 3:
        raise ValueError("equilibrium boundary outline is required for standalone SFL coordinates")

    theta_boundary = np.unwrap(np.arctan2(boundary_z - axis_z, boundary_r - axis_r))
    radius_boundary = np.hypot(boundary_r - axis_r, boundary_z - axis_z)
    order = np.argsort(theta_boundary)
    theta_boundary = theta_boundary[order]
    radius_boundary = radius_boundary[order]
    theta_ext = np.concatenate([theta_boundary - 2 * np.pi, theta_boundary, theta_boundary + 2 * np.pi])
    radius_ext = np.concatenate([radius_boundary, radius_boundary, radius_boundary])

    theta = np.linspace(-np.pi, np.pi, int(n_theta), endpoint=False)
    radius_lcfs = np.interp(theta, theta_ext, radius_ext)
    profiles_1d = ts.get("profiles_1d", {})
    psi_1d = np.asarray(profiles_1d.get("psi", np.linspace(psi_axis, psi_boundary, 33)), dtype=float)
    if psi_1d.size < 2:
        psi_1d = np.linspace(psi_axis, psi_boundary, 33)
    psi_norm = (psi_1d - psi_axis) / (psi_boundary - psi_axis) if psi_boundary != psi_axis else np.zeros_like(psi_1d)
    psi_norm = np.clip(psi_norm, 0.0, 1.0)
    rho = np.sqrt(psi_norm)

    r = axis_r + rho[:, None] * radius_lcfs[None, :] * np.cos(theta)[None, :]
    z = axis_z + rho[:, None] * radius_lcfs[None, :] * np.sin(theta)[None, :]
    psi = psi_axis + psi_norm[:, None] * (psi_boundary - psi_axis)
    return psi_norm, theta, r, z, np.broadcast_to(psi, r.shape), np.broadcast_to(theta, r.shape)

def update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=None):
    """
    Update normalized poloidal flux (psi_norm) for all time slices.
    
    Parameters:
        ods (OMAS structure): Input OMAS data structure
        time_slice (int/list/None): Specific time slice(s) to process. None=all
    """
    # Process all time slices if not specified
    time_slices = range(len(ods['equilibrium.time_slice'])) if time_slice is None else (
        [time_slice] if isinstance(time_slice, (int, np.integer)) else time_slice)
    
    for idx in time_slices:
        ts = ods['equilibrium.time_slice'][idx]
        # Get psi values
        psi = ts['profiles_1d.psi']
        psi_axis = ts['global_quantities.psi_axis']
        psi_bdry = ts['global_quantities.psi_boundary']
        
        # Calculate normalized psi
        psi_n = (psi - psi_axis) / (psi_bdry - psi_axis)
        
        # Store in ODS
        ts['profiles_1d.psi_norm'] = psi_n

def update_equilibrium_profiles_1d_radial_coordinates(ods, time_slice=None, plot_opt=0):
    """
    Update radial coordinates for all time slices using 2D psi mapping.
    
    Parameters:
        ods (OMAS structure): Input OMAS data structure
        time_slice (int/list/None): Specific time slice(s) to process. None=all
        plot_opt (int): 0=no plot, 1=plot validation, 2=plot derivatives
    """
    from scipy.interpolate import interp1d
        
    # Process all time slices if not specified
    time_slices = range(len(ods['equilibrium.time_slice'])) if time_slice is None else (
        [time_slice] if isinstance(time_slice, (int, np.integer)) else time_slice)
    
    for idx in time_slices:
        ts = ods['equilibrium.time_slice'][idx]
        # Extract 2D grid data at magnetic axis
        grid_r = ts['profiles_2d.0.grid.dim1']
        grid_z = ts['profiles_2d.0.grid.dim2']
        psi_2d = ts['profiles_2d.0.psi']
        z_axis = ts['global_quantities.magnetic_axis.z']
        z_idx = np.argmin(np.abs(grid_z - z_axis))
        psi_2d_slice = psi_2d[:, z_idx]

        # Boundary processing
        boundary_r = ts['boundary.outline.r']
        r_min, r_max = np.min(boundary_r), np.max(boundary_r)
        r_axis = ts['global_quantities.magnetic_axis.r']
        
        # Split into inboard/outboard regions
        mask_in = (grid_r >= r_min) & (grid_r <= r_axis)
        mask_out = (grid_r >= r_axis) & (grid_r <= r_max)
        psi_in, r_in = psi_2d_slice[mask_in], grid_r[mask_in]
        psi_out, r_out = psi_2d_slice[mask_out], grid_r[mask_out]

        # Create interpolation functions
        f_in = interp1d(psi_in[::-1], r_in[::-1], 
                       kind='cubic', fill_value='extrapolate')
        f_out = interp1d(psi_out, r_out, 
                        kind='cubic', fill_value='extrapolate')

        # Map 1D profiles
        psi_1d = ts['profiles_1d.psi']
        ts['profiles_1d.r_inboard'] = f_in(psi_1d)
        ts['profiles_1d.r_outboard'] = f_out(psi_1d)

        # Generate validation plots.  Rendering is delegated to vaft.plot; this
        # namespace only shapes the data into view models (issue #63).
        if plot_opt >= 1:
            _plot_radial_mapping_validation(ts, idx, r_in, psi_in, r_out, psi_out, psi_1d)

def update_equilibrium_boundary(ods, time_slice=None):
    """
    Update geometric axis for all time slices.
    """
    def _safe_float(value):
        try:
            v = float(value)
        except Exception:
            return np.nan
        return v if np.isfinite(v) else np.nan

    def _fallback_axis(ts):
        """
        Best-effort finite axis fallback when boundary outline is unavailable.
        Preference:
        1) global magnetic axis
        2) existing boundary.geometric_axis
        3) 2D grid center
        4) hard fallback (0,0)
        """
        r0 = np.nan
        z0 = np.nan
        if "global_quantities.magnetic_axis.r" in ts:
            r0 = _safe_float(ts["global_quantities.magnetic_axis.r"])
        if "global_quantities.magnetic_axis.z" in ts:
            z0 = _safe_float(ts["global_quantities.magnetic_axis.z"])

        if not np.isfinite(r0) and "boundary.geometric_axis.r" in ts:
            r0 = _safe_float(ts["boundary.geometric_axis.r"])
        if not np.isfinite(z0) and "boundary.geometric_axis.z" in ts:
            z0 = _safe_float(ts["boundary.geometric_axis.z"])

        if (not np.isfinite(r0) or not np.isfinite(z0)) and (
            "profiles_2d.0.grid.dim1" in ts and "profiles_2d.0.grid.dim2" in ts
        ):
            try:
                r_grid = np.asarray(ts["profiles_2d.0.grid.dim1"], float).reshape(-1)
                z_grid = np.asarray(ts["profiles_2d.0.grid.dim2"], float).reshape(-1)
                if not np.isfinite(r0) and r_grid.size:
                    r0 = _safe_float(0.5 * (np.nanmin(r_grid) + np.nanmax(r_grid)))
                if not np.isfinite(z0) and z_grid.size:
                    z0 = _safe_float(0.5 * (np.nanmin(z_grid) + np.nanmax(z_grid)))
            except Exception:
                pass

        if not np.isfinite(r0):
            r0 = 0.0
        if not np.isfinite(z0):
            z0 = 0.0

        return float(r0), float(z0)

    if 'equilibrium.time_slice' not in ods or len(ods['equilibrium.time_slice']) == 0:
        logger.warning("No equilibrium.time_slice found while updating boundary.")
        return

    if time_slice is None:
        time_slices = range(len(ods['equilibrium.time_slice']))
    elif isinstance(time_slice, (int, np.integer)):
        time_slices = [int(time_slice)]
    else:
        time_slices = [int(i) for i in time_slice]

    for idx in time_slices:
        if idx < 0 or idx >= len(ods['equilibrium.time_slice']):
            logger.warning("time_slice index %s is out of bounds in update_equilibrium_boundary.", idx)
            continue

        ts = ods['equilibrium']['time_slice'][idx]
        if 'boundary.outline.r' not in ts or 'boundary.outline.z' not in ts:
            logger.warning("boundary.outline not found for time slice %s", idx)
            r0, z0 = _fallback_axis(ts)
            ts['boundary.geometric_axis.r'] = r0
            ts['boundary.geometric_axis.z'] = z0
            continue

        r_outline = np.asarray(ts['boundary.outline.r'], float).reshape(-1)
        z_outline = np.asarray(ts['boundary.outline.z'], float).reshape(-1)
        finite = np.isfinite(r_outline) & np.isfinite(z_outline)
        r_outline = r_outline[finite]
        z_outline = z_outline[finite]
        if r_outline.size < 3:
            logger.warning("boundary.outline has fewer than 3 valid points for time slice %s", idx)
            r0, z0 = _fallback_axis(ts)
            ts['boundary.geometric_axis.r'] = r0
            ts['boundary.geometric_axis.z'] = z0
            continue

        r_min = float(np.min(r_outline))
        r_max = float(np.max(r_outline))
        z_min = float(np.min(z_outline))
        z_max = float(np.max(z_outline))

        ts['boundary.geometric_axis.r'] = 0.5 * (r_max + r_min)
        ts['boundary.geometric_axis.z'] = 0.5 * (z_max + z_min)
        ts['boundary.minor_radius'] = 0.5 * (r_max - r_min)

        # Profile-derived geometry fields are optional; preserve robustness if profiles are missing.
        tri_low = np.nan
        tri_up = np.nan
        elong = np.nan
        try:
            tri_low = float(np.asarray(ts['profiles_1d.triangularity_lower'], float).reshape(-1)[-1])
        except Exception:
            pass
        try:
            tri_up = float(np.asarray(ts['profiles_1d.triangularity_upper'], float).reshape(-1)[-1])
        except Exception:
            pass
        try:
            elong = float(np.asarray(ts['profiles_1d.elongation'], float).reshape(-1)[-1])
        except Exception:
            pass

        if np.isfinite(tri_low):
            ts['boundary.triangularity_lower'] = tri_low
        if np.isfinite(tri_up):
            ts['boundary.triangularity_upper'] = tri_up
        if np.isfinite(tri_low) and np.isfinite(tri_up):
            ts['boundary.triangularity'] = 0.5 * (tri_low + tri_up)
        if np.isfinite(elong):
            ts['boundary.elongation'] = elong

def update_equilibrium_coordinates(ods, time_slice=None, plot_opt=0):
    """
    Main entry point for updating all equilibrium coordinates.
    Updates normalized psi and radial coordinates for all time slices.
    
    Parameters:
        ods (OMAS structure): Input OMAS data structure
        time_slice (int/list/None): Specific time slice(s) to process. None=all
        plot_opt (int): 0=no plot, 1=plot validation
    """
    # Update normalized psi
    update_equilibrium_profiles_1d_normalized_psi(ods, time_slice)
    
    # Update radial coordinates
    update_equilibrium_profiles_1d_radial_coordinates(ods, time_slice, plot_opt)

def update_equilibrium_global_quantities_q_min(ods, time_slice=None):
    """
    Update q_min for all time slices using min() of profiles_1d.q
    """
    # Process all time slices if not specified
    time_slices = range(len(ods['equilibrium.time_slice'])) if time_slice is None else (
        [time_slice] if isinstance(time_slice, (int, np.integer)) else time_slice)
    for idx in time_slices:
        ts = ods['equilibrium.time_slice'][idx]
        ts['global_quantities.q_min'] = ts['profiles_1d.q'].min()

def update_equilibrium_global_quantities_volume(ods, time_slice=None):
    """
    Update volume for all time slices using profiles_1d.volume
    """
    # check if profiles_1d.volume exists for each time slice
    if time_slice is None:
        if 'equilibrium.time_slice' in ods and len(ods['equilibrium.time_slice']):
            time_slice = range(len(ods['equilibrium.time_slice']))
        else:
            print("Warning: No time slices found in ODS. Cannot update stored energy.")
            return
    # Convert single integer to list for iteration
    if isinstance(time_slice, (int, np.integer)):
        time_slice = [time_slice]
    for idx in time_slice:
        ts = ods['equilibrium.time_slice'][idx]
        if 'profiles_1d.volume' not in ts:
            print(f"Warning: profiles_1d.volume not found for time slice {idx}")
            continue
        ts['global_quantities.volume'] = ts['profiles_1d.volume'][-1]

def update_equilibrium_profiles_2d_j_tor(ods, time_slice=None):
    r"""Evaluate the **local** toroidal current density on the 2-D grid.

    Issue #316. This used to map ``profiles_1d.j_tor`` onto (R,Z) with
    ``psi_to_rz``, the way pressure is mapped. That is wrong, because
    ``profiles_1d.j_tor`` is not a flux function: the IMAS DD defines it as the
    flux-surface *average* ``<j_tor/R> / <1/R>``, while the local density varies
    across a surface as ``R`` and ``1/R``. Splatting the average back onto the
    grid produced a field that is constant on each surface, which the DD's
    ``profiles_2d.j_tor`` -- "Toroidal plasma current density" -- is not.

    Nothing needs mapping. ``p'`` and ``ff'`` *are* flux functions and ``R`` is
    the grid, so Grad-Shafranov gives the local value directly:

    .. math::
        j_\varphi(R, Z) = -\sigma_{B_p} (2\pi)^{e_{B_p}}
            \left( R\, p'(\psi) + \frac{f f'(\psi)}{\mu_0 R} \right)

    the same expression, and the same resolved psi convention, that
    :func:`update_equilibrium_profiles_1d_j_tor` averages. Points outside the
    LCFS are written as NaN rather than evaluated: ``p'`` and ``ff'`` are only
    defined out to the boundary, and clipping them into the scrape-off layer
    would draw current where there is none. The test is containment in
    ``boundary.outline``, not ``psi_norm <= 1`` -- see :func:`_inside_boundary`.

    The function was inert before issue #290 -- it skipped every slice for want
    of a 1-D ``j_tor`` -- so supplying that profile is what armed the defect.

    Parameters:
        ods (OMAS structure): Input OMAS data structure, updated in place.
        time_slice (int/list/None): Specific time slice(s) to process. None=all
    """
    from vaft.data.cocos import cocos_spec  # noqa: F401  (used via _sigma_bp)

    for idx in _equilibrium_time_slices(ods, time_slice):
        frame = _equilibrium_flux_frame(ods, idx)
        if frame is None:
            continue
        ts = frame["ts"]
        if "profiles_1d.dpressure_dpsi" not in ts or "profiles_1d.f_df_dpsi" not in ts:
            logger.warning(
                "profiles_1d.dpressure_dpsi/f_df_dpsi not found for time slice %s; "
                "skipping 2-D j_tor.",
                idx,
            )
            continue

        psi_norm_1d = frame["psi_norm"]
        pprime = np.asarray(ts["profiles_1d.dpressure_dpsi"], float).reshape(-1)
        ffprime = np.asarray(ts["profiles_1d.f_df_dpsi"], float).reshape(-1)
        if pprime.size != psi_norm_1d.size or ffprime.size != psi_norm_1d.size:
            logger.warning(
                "profiles_1d lengths disagree for time slice %s; skipping 2-D j_tor.", idx
            )
            continue

        psi_axis = frame["psi_axis_radian"]
        psi_boundary = frame["psi_boundary_radian"]
        psi_norm_2d = (frame["psi_2d_radian"] - psi_axis) / (psi_boundary - psi_axis)

        order = np.argsort(psi_norm_1d)
        grid_r = frame["r_grid"][:, None] * np.ones_like(frame["z_grid"])[None, :]
        pprime_2d = np.interp(psi_norm_2d, psi_norm_1d[order], pprime[order])
        ffprime_2d = np.interp(psi_norm_2d, psi_norm_1d[order], ffprime[order])

        sigma_bp = _sigma_bp(frame["convention"])
        prefactor = -sigma_bp * (2.0 * np.pi) ** frame["exp_bp"]
        with np.errstate(divide="ignore", invalid="ignore"):
            j_tor_2d = prefactor * (grid_r * pprime_2d + ffprime_2d / (MU0 * grid_r))

        # p' and ff' stop at the boundary, so the current does too.
        confined = _inside_boundary(frame, psi_norm_2d)
        ts["profiles_2d.0.j_tor"] = np.where(confined, j_tor_2d, np.nan)



def _inside_boundary(frame, psi_norm_2d):
    """Mask of the confined region, from the LCFS outline where there is one.

    ``psi_norm <= 1`` is a flux threshold, not a containment test. Outside the
    plasma psi is not monotonic -- it turns over near the coils and in the
    private-flux region -- so a large part of the exterior passes the threshold.
    On the packaged 39915 sample that put current at 8-11% of the exterior grid
    points, integrating to 265 kA against a 46 kA plasma on one slice.

    The outline is the real boundary, so it is used when the slice has one. The
    flux threshold remains the fallback for a slice that does not, where it is
    the best available answer rather than a correct one.
    """
    boundary = frame.get("boundary")
    if boundary is None:
        return psi_norm_2d <= 1.0

    from matplotlib.path import Path as _MplPath

    outline_r = np.asarray(boundary[0], float).reshape(-1)
    outline_z = np.asarray(boundary[1], float).reshape(-1)
    grid_r, grid_z = np.meshgrid(frame["r_grid"], frame["z_grid"], indexing="ij")
    inside = _MplPath(np.column_stack([outline_r, outline_z])).contains_points(
        np.column_stack([grid_r.ravel(), grid_z.ravel()])
    ).reshape(psi_norm_2d.shape)
    # Both conditions: the outline can enclose a cell whose psi says otherwise on
    # a coarse grid, and p'/ff' are only defined out to psi_norm = 1 either way.
    return inside & (psi_norm_2d <= 1.0)


def _equilibrium_time_slices(ods, time_slice):
    """Normalize the ``time_slice`` argument the ``update_equilibrium_*`` way."""
    if "equilibrium.time_slice" not in ods or not len(ods["equilibrium.time_slice"]):
        return []
    total = len(ods["equilibrium.time_slice"])
    if time_slice is None:
        return list(range(total))
    if isinstance(time_slice, (int, np.integer)):
        indices = [int(time_slice)]
    else:
        indices = [int(index) for index in time_slice]
    return [index for index in indices if 0 <= index < total]


def _equilibrium_flux_frame(ods, idx):
    """Everything the flux-surface derivations need, in weber per radian.

    Returns ``None`` when the slice cannot support a trace -- no 2-D map, or a
    degenerate solution with ``psi_axis == psi_boundary``, which the packaged
    VEST samples do contain.

    The storage convention comes from :func:`vaft.process.equilibrium.as_equilibrium`,
    whose ``convention.psi_per_radian`` is settled by Ampere's law round the LCFS
    and so answers on legacy Wb/rad artifacts that carry no ``profiles_1d.phi``
    (issue #236).
    """
    from vaft.process.equilibrium import as_equilibrium

    ts = ods["equilibrium.time_slice"][idx]
    required = (
        "profiles_2d.0.grid.dim1",
        "profiles_2d.0.grid.dim2",
        "profiles_2d.0.psi",
        "profiles_1d.psi",
        "global_quantities.psi_axis",
        "global_quantities.psi_boundary",
    )
    missing = [path for path in required if path not in ts]
    if missing:
        logger.warning(
            "equilibrium.time_slice.%s is missing %s; skipping flux-surface derivation.",
            idx,
            ", ".join(missing),
        )
        return None

    psi_axis = float(ts["global_quantities.psi_axis"])
    psi_boundary = float(ts["global_quantities.psi_boundary"])
    if not np.isfinite(psi_axis) or not np.isfinite(psi_boundary) or psi_axis == psi_boundary:
        logger.warning(
            "equilibrium.time_slice.%s has a degenerate psi range; skipping.", idx
        )
        return None

    r_grid = np.asarray(ts["profiles_2d.0.grid.dim1"], float).reshape(-1)
    z_grid = np.asarray(ts["profiles_2d.0.grid.dim2"], float).reshape(-1)
    psi_2d = np.asarray(ts["profiles_2d.0.psi"], float)
    if psi_2d.shape == (z_grid.size, r_grid.size) and psi_2d.shape != (r_grid.size, z_grid.size):
        psi_2d = psi_2d.T
    if psi_2d.shape != (r_grid.size, z_grid.size):
        logger.warning(
            "equilibrium.time_slice.%s psi shape %s does not match its grid; skipping.",
            idx,
            psi_2d.shape,
        )
        return None

    try:
        convention = as_equilibrium(ods, time_index=idx).convention
        per_radian = convention.psi_per_radian
    except Exception as exc:
        logger.warning(
            "COCOS identification failed for time slice %s (%s); assuming the DD weber psi.",
            idx,
            exc,
        )
        convention, per_radian = None, False
    exp_bp = 0 if per_radian else 1
    to_radian = 1.0 / (2.0 * np.pi) ** exp_bp

    psi_1d = np.asarray(ts["profiles_1d.psi"], float).reshape(-1)
    psi_norm = (psi_1d - psi_axis) / (psi_boundary - psi_axis)

    axis_rz = None
    if "global_quantities.magnetic_axis.r" in ts and "global_quantities.magnetic_axis.z" in ts:
        axis_r = float(ts["global_quantities.magnetic_axis.r"])
        axis_z = float(ts["global_quantities.magnetic_axis.z"])
        if np.isfinite(axis_r) and np.isfinite(axis_z):
            axis_rz = (axis_r, axis_z)

    boundary = None
    if "boundary.outline.r" in ts and "boundary.outline.z" in ts:
        outline_r = np.asarray(ts["boundary.outline.r"], float).reshape(-1)
        outline_z = np.asarray(ts["boundary.outline.z"], float).reshape(-1)
        if outline_r.size >= 3 and outline_r.size == outline_z.size:
            boundary = (outline_r, outline_z)

    return {
        "ts": ts,
        "r_grid": r_grid,
        "z_grid": z_grid,
        "psi_2d_radian": psi_2d * to_radian,
        "psi_axis_radian": psi_axis * to_radian,
        "psi_boundary_radian": psi_boundary * to_radian,
        "psi_1d_radian": psi_1d * to_radian,
        "psi_norm": np.clip(psi_norm, 0.0, 1.0),
        "axis_rz": axis_rz,
        "boundary": boundary,
        "exp_bp": exp_bp,
        "to_radian": to_radian,
        "convention": convention,
    }


#: Leaves ``update_equilibrium_profiles_1d_geometry`` writes, and the
#: :func:`~vaft.process.equilibrium.flux_surface_quantities` key each comes from.
#: ``r_inboard``/``r_outboard`` are deliberately absent: they belong to
#: :func:`update_equilibrium_profiles_1d_radial_coordinates`, and one leaf must
#: have one writer.
_GEOMETRY_LEAVES = (
    "gm1",
    "gm5",
    "gm8",
    "gm9",
    "volume",
    "area",
    "surface",
    "elongation",
    "triangularity_upper",
    "triangularity_lower",
    "b_field_max",
    "b_field_min",
)

#: Written after dividing out the per-radian contract, because a psi-derivative
#: transforms inversely to psi and must match the ODS's own psi unit -- the same
#: bargain ``profiles_1d.dpressure_dpsi`` already keeps.
_GEOMETRY_PSI_DERIVATIVES = ("dvolume_dpsi", "darea_dpsi")


def update_equilibrium_profiles_1d_geometry(ods, time_slice=None, *, return_surfaces=False):
    """Derive the flux-surface geometry profiles by tracing the 2-D psi map.

    Writes ``profiles_1d`` ``gm1`` (<1/R^2>), ``gm5`` (<B^2>), ``gm8`` (<R>),
    ``gm9`` (<1/R>), ``volume``, ``area``, ``surface``, ``dvolume_dpsi``,
    ``darea_dpsi``, ``elongation``, ``triangularity_upper``,
    ``triangularity_lower``, ``b_field_max`` and ``b_field_min``.

    An EFIT g-file stores none of these, so an ODS built from one carries only
    ``p``, ``q``, ``f``, ``ff'``, ``p'`` and ``psi``; every consumer that wants
    a volume, a plasma cross-section or a flux-surface average -- the stored
    energy, the database summary's shape columns, and
    :func:`update_equilibrium_profiles_1d_j_tor` -- is blocked without them.

    Accuracy against the OMFIT-produced reference in
    ``vaft/data/kineticEfit/ods_48224_300ms.json``, as a fraction of each
    profile's peak away from the innermost couple of surfaces: the ``gm*`` set
    and ``b_field_max`` to 6e-4, ``surface`` to 1e-3, ``elongation`` to 2e-3,
    ``b_field_min`` to 2e-3, ``volume`` and ``area`` to 5e-3 -- the last being the
    systematic difference between tracing a 129x129 map and OMFIT's own surface
    solve. ``b_field_average`` is *not* written: no averaging definition tried
    reproduces the reference to better than 8%, so what it means there is
    unresolved and writing a guess would be worse than leaving the leaf empty.

    Parameters:
        ods (OMAS structure): Input OMAS data structure, updated in place.
        time_slice (int/list/None): Specific time slice(s) to process. None=all
        return_surfaces (bool): also return ``{time_slice_index: surfaces}``.
            The trace is the expensive part of every derivation built on it, and
            two of its results -- ``bp_dl`` and the per-surface ``length_pol`` --
            are not DD quantities and so are not written to the ODS. A caller
            that needs them, such as
            :func:`update_equilibrium_global_quantities_beta_li`, takes them
            here rather than tracing a second time.
    """
    from vaft.process.equilibrium import flux_surface_quantities

    traced: dict[int, dict] = {}
    for idx in _equilibrium_time_slices(ods, time_slice):
        frame = _equilibrium_flux_frame(ods, idx)
        if frame is None:
            continue
        ts = frame["ts"]

        f_profile = None
        if "profiles_1d.f" in ts:
            candidate = np.asarray(ts["profiles_1d.f"], float).reshape(-1)
            if candidate.size == frame["psi_norm"].size:
                f_profile = candidate

        try:
            surfaces = flux_surface_quantities(
                frame["psi_2d_radian"],
                frame["r_grid"],
                frame["z_grid"],
                frame["psi_axis_radian"],
                frame["psi_boundary_radian"],
                frame["psi_norm"],
                f_profile=f_profile,
                axis_rz=frame["axis_rz"],
                boundary=frame["boundary"],
            )
        except Exception as exc:
            logger.warning("Flux-surface trace failed for time slice %s: %s", idx, exc)
            continue

        traced[idx] = surfaces
        for name in _GEOMETRY_LEAVES:
            values = surfaces[name]
            if np.all(np.isfinite(values)):
                ts[f"profiles_1d.{name}"] = values
        for name in _GEOMETRY_PSI_DERIVATIVES:
            values = surfaces[name] * frame["to_radian"]
            if np.all(np.isfinite(values)):
                ts[f"profiles_1d.{name}"] = values

    if return_surfaces:
        return traced
    return None


def update_equilibrium_profiles_1d_toroidal_flux(ods, time_slice=None):
    """Derive ``profiles_1d`` ``phi``, ``rho_tor`` and ``rho_tor_norm`` from ``q``.

    ``q = dPhi/dPsi`` with both fluxes in weber, so ``Phi`` follows from
    integrating ``q`` against psi and ``rho_tor = sqrt(|Phi| / (pi |B0|))``.

    This replaces the ``sqrt(psi_N)`` proxy that VAFT wrote before issue #192 and
    that the packaged samples still hold -- bit-identical to ``sqrt(psi_N)`` and
    up to 0.126 away from the real coordinate. ``rho_tor_norm`` is
    ``ProfileRecipe.default_coordinate``, so the proxy is the abscissa of every
    1-D equilibrium profile plot drawn from those samples.

    The integral inherits whatever ``q`` says on axis, and EFIT's ``q[0]`` is
    often an outlier -- 8.07 against a neighbourhood of 1.9 in the packaged
    kineticEfit reference, which alone moves ``rho_tor_norm`` by 0.037 over the
    innermost surfaces. The same is true of :func:`vaft.data.eqdsk.to_omas`, so
    the two paths stay consistent with each other; a reference whose ``rho_tor``
    came from an independent surface solve will differ there by about that much.

    Parameters:
        ods (OMAS structure): Input OMAS data structure, updated in place.
        time_slice (int/list/None): Specific time slice(s) to process. None=all
    """
    from vaft.data._derived import rho_tor_profile

    for idx in _equilibrium_time_slices(ods, time_slice):
        frame = _equilibrium_flux_frame(ods, idx)
        if frame is None:
            continue
        ts = frame["ts"]
        if "profiles_1d.q" not in ts:
            logger.warning("profiles_1d.q not found for time slice %s; skipping.", idx)
            continue
        q_profile = np.asarray(ts["profiles_1d.q"], float).reshape(-1)

        b0 = None
        if "equilibrium.vacuum_toroidal_field.b0" in ods:
            field = np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"], float).reshape(-1)
            if field.size:
                b0 = float(field[min(idx, field.size - 1)])
        if b0 is None or not np.isfinite(b0) or b0 == 0.0:
            logger.warning(
                "vacuum_toroidal_field.b0 unavailable for time slice %s; skipping.", idx
            )
            continue

        # The shared routine takes psi in weber; the frame carries Wb/rad.
        result = rho_tor_profile(q_profile, frame["psi_1d_radian"] * 2.0 * np.pi, b0)
        if result is None:
            logger.warning("Toroidal flux is degenerate for time slice %s; skipping.", idx)
            continue
        ts["profiles_1d.phi"] = result.phi
        if result.rho_tor is not None:
            ts["profiles_1d.rho_tor"] = result.rho_tor
        ts["profiles_1d.rho_tor_norm"] = result.rho_tor_norm


def update_equilibrium_profiles_1d_j_tor(ods, time_slice=None):
    r"""Derive the flux-surface-averaged toroidal current density (issue #290).

    The IMAS Data Dictionary defines ``profiles_1d.j_tor`` as
    ``<j_tor/R> / <1/R>``. Substituting the Grad-Shafranov current
    ``j_phi = -sigma_Bp (2 pi)^e_Bp (R p' + f f' / (mu0 R))`` gives

        j_tor = -sigma_Bp (2 pi)^e_Bp (p' + gm1 f f' / mu0) / gm9

    with ``p'`` and ``ff'`` as the ODS stores them, ``gm1 = <1/R^2>`` and
    ``gm9 = <1/R>``. Fed the reference's own ``gm1``/``gm9`` this reproduces the
    OMFIT-stored profile to 1.3e-10 relative; with the averages traced here it
    holds to 4e-4 of peak, and 1.3e-3 at the axis point.

    The sign is not inherited. ``sigma_Bp`` comes from the identified COCOS, and
    :meth:`vaft.data.cocos.CocosSpec.expected_sign` requires ``p'`` to carry
    ``-sigma_ip sigma_Bp``; multiplying by ``-sigma_Bp`` therefore leaves
    ``sign(j_tor) = sigma_ip``, which is the sign the same table requires of
    ``j_phi``. COCOS 1 and 2 share both ``sigma_Bp`` and ``e_Bp``, so the
    candidate ambiguity the packaged samples leave open does not reach the answer.

    ``gm1``/``gm9`` are derived by :func:`update_equilibrium_profiles_1d_geometry`
    when the slice does not already carry them.

    Parameters:
        ods (OMAS structure): Input OMAS data structure, updated in place.
        time_slice (int/list/None): Specific time slice(s) to process. None=all
    """
    for idx in _equilibrium_time_slices(ods, time_slice):
        frame = _equilibrium_flux_frame(ods, idx)
        if frame is None:
            continue
        ts = frame["ts"]
        if "profiles_1d.dpressure_dpsi" not in ts or "profiles_1d.f_df_dpsi" not in ts:
            logger.warning(
                "profiles_1d.dpressure_dpsi/f_df_dpsi not found for time slice %s; skipping.",
                idx,
            )
            continue

        if "profiles_1d.gm1" not in ts or "profiles_1d.gm9" not in ts:
            update_equilibrium_profiles_1d_geometry(ods, time_slice=idx)
        if "profiles_1d.gm1" not in ts or "profiles_1d.gm9" not in ts:
            logger.warning(
                "Could not derive gm1/gm9 for time slice %s; skipping j_tor.", idx
            )
            continue

        pprime = np.asarray(ts["profiles_1d.dpressure_dpsi"], float).reshape(-1)
        ffprime = np.asarray(ts["profiles_1d.f_df_dpsi"], float).reshape(-1)
        gm1 = np.asarray(ts["profiles_1d.gm1"], float).reshape(-1)
        gm9 = np.asarray(ts["profiles_1d.gm9"], float).reshape(-1)
        if not (pprime.size == ffprime.size == gm1.size == gm9.size):
            logger.warning(
                "profiles_1d lengths disagree for time slice %s; skipping j_tor.", idx
            )
            continue

        sigma_bp = _sigma_bp(frame["convention"])
        prefactor = -sigma_bp * (2.0 * np.pi) ** frame["exp_bp"]
        with np.errstate(divide="ignore", invalid="ignore"):
            j_tor = prefactor * (pprime + gm1 * ffprime / MU0) / gm9
        if not np.all(np.isfinite(j_tor)):
            logger.warning(
                "Derived j_tor is not finite for time slice %s; skipping.", idx
            )
            continue
        ts["profiles_1d.j_tor"] = j_tor


def _sigma_bp(convention) -> int:
    """``sigma_Bp`` of the identified convention, defaulting to the DD's +1.

    Every candidate must agree: COCOS 1 and 2 do, and so do 11 and 12, which are
    the pairs an unknown ``clockwise_phi`` leaves open. A genuinely split
    candidate set means the orientation is unidentified, and the DD convention is
    the honest default there.
    """
    from vaft.data.cocos import cocos_spec

    candidates = () if convention is None else tuple(
        index for index in (getattr(convention, "candidates", None) or ()) if index
    )
    signs = set()
    for index in candidates:
        try:
            signs.add(cocos_spec(int(index)).sigma_bp)
        except Exception:
            return 1
    if len(signs) == 1:
        return int(signs.pop())
    return 1


def update_equilibrium_global_quantities_area(ods, time_slice=None):
    """Update ``global_quantities.area`` from the edge of ``profiles_1d.area``.

    The scalar counterpart of :func:`update_equilibrium_global_quantities_volume`,
    which the database summary's ``area_m2`` column reads.
    """
    for idx in _equilibrium_time_slices(ods, time_slice):
        ts = ods["equilibrium.time_slice"][idx]
        if "profiles_1d.area" not in ts:
            logger.warning("profiles_1d.area not found for time slice %s", idx)
            continue
        ts["global_quantities.area"] = float(
            np.asarray(ts["profiles_1d.area"], float).reshape(-1)[-1]
        )



def update_equilibrium_global_quantities_beta_li(ods, time_slice=None, *, surfaces=None):
    r"""Derive ``beta_pol``, ``beta_tor``, ``beta_normal``, ``li_3`` and ``length_pol``.

    Issue #238. A g-file stores none of these and OMFIT supplied them from its
    own flux-surface solve, so losing that path left them empty -- four of the
    database summary's columns among them.

    **The definitions are the IMAS DD's**, read from the DD rather than from
    memory:

    ==============  =====================================================
    ``beta_pol``    ``4 int(p dV) / (R_0 mu_0 Ip^2)``
    ``beta_tor``    ``2 mu0 int(p dV) / V / B0^2``
    ``beta_normal`` ``100 beta_tor a[m] B0[T] / Ip[MA]``
    ``li_3``        ``2 int(B_pol^2 dV) / (mu0^2 Ip^2 R_0)``
    ==============  =====================================================

    ``li_3`` is the one the DD gives no formula for -- it documents only
    "Internal inductance". This is the ITER/Jackson third definition, and it is
    written on the strength of reproducing the packaged OMFIT reference to 0.3%,
    not on the strength of the DD text.

    ``beta_pol`` is where the DD and EFIT disagree, and it is not a numerical
    disagreement: EFIT normalizes by the field implied by the LCFS circumference,
    ``2 mu0 <p>_V / (mu0 Ip/L_pol)^2``, which on the packaged reference gives
    0.0287 against the DD's 0.0227 -- **26% apart**, with the EFIT form matching
    the stored value to 0.1%. The DD form is what goes in the DD leaf; the other
    is available to the database summary under its own name, and #318 owns the
    sensitivity study of the two. Do not "fix" the leaf by matching OMFIT.

    ``R_0`` comes from :func:`resolve_reference_major_radius`, not straight from
    ``vacuum_toroidal_field.r0``: on the VEST database that leaf is corrupt and
    would inflate ``beta_pol`` and ``li_3`` by 1.15-2.1x.

    ``B0`` is ``vacuum_toroidal_field.b0``, as the DD says. That does not
    reproduce the reference's ``beta_tor`` (ratio 1.041, and the field that would
    match is 0.1539 T against a stored 0.1509 T); OMFIT's reference field is
    unidentified and is recorded as such on #238 rather than reverse-engineered.

    ``int(B_pol^2 dV)`` needs only the per-surface line integral ``bp_dl``,
    because the volume element cancels one power of ``B_pol``; and the psi
    convention comes from the same resolved frame as the rest of this module,
    not from a further ``ods_psi_to_wb_per_radian_factor`` call site (#294).

    Parameters:
        ods (OMAS structure): Input OMAS data structure, updated in place.
        time_slice (int/list/None): Specific time slice(s) to process. None=all
        surfaces (dict/None): ``{time_slice_index: surfaces}`` from
            :func:`update_equilibrium_profiles_1d_geometry` with
            ``return_surfaces=True``. ``bp_dl`` is not a DD quantity and is not
            written to the ODS, so without this the trace has to be repeated --
            0.95x the geometry updater's own cost on the packaged sample. Called
            standalone it traces, as it must.
    """
    from vaft.formula.equilibrium import (
        beta_normal_from_beta_tor,
        beta_poloidal_from_pressure_integral,
        beta_toroidal_from_p_B0,
        li_3_from_Bp2_volume_integral,
    )

    for idx in _equilibrium_time_slices(ods, time_slice):
        frame = _equilibrium_flux_frame(ods, idx)
        if frame is None:
            continue
        ts = frame["ts"]

        r0 = resolve_reference_major_radius(ods)
        b0 = _time_indexed_scalar(ods, "equilibrium.vacuum_toroidal_field.b0", idx)
        ip = _scalar_or_nan(ts, "global_quantities.ip")
        if not np.isfinite(r0) or r0 <= 0.0 or not np.isfinite(b0) or b0 == 0.0:
            logger.warning(
                "vacuum_toroidal_field.r0/b0 unusable for time slice %s; skipping betas.",
                idx,
            )
            continue
        if not np.isfinite(ip) or ip == 0.0:
            logger.warning("global_quantities.ip unusable for time slice %s; skipping betas.", idx)
            continue

        if "profiles_1d.volume" not in ts or "profiles_1d.pressure" not in ts:
            # Keep what that trace produced: without it this would trace once to
            # get the volume and again below to get bp_dl.
            healed = update_equilibrium_profiles_1d_geometry(
                ods, time_slice=[idx], return_surfaces=True
            )
            if healed and idx in healed:
                surfaces = dict(surfaces or {})
                surfaces[idx] = healed[idx]
        if "profiles_1d.volume" not in ts or "profiles_1d.pressure" not in ts:
            logger.warning(
                "profiles_1d.pressure/volume unavailable for time slice %s; skipping betas.",
                idx,
            )
            continue

        pressure = np.asarray(ts["profiles_1d.pressure"], float).reshape(-1)
        volume = np.asarray(ts["profiles_1d.volume"], float).reshape(-1)
        if pressure.size != volume.size or pressure.size < 2:
            logger.warning("pressure/volume lengths disagree for time slice %s.", idx)
            continue
        plasma_volume = float(volume[-1])
        if not np.isfinite(plasma_volume) or plasma_volume <= 0.0:
            logger.warning("plasma volume is not positive for time slice %s.", idx)
            continue
        pressure_integral = float(trapz_compat(pressure, x=volume))
        p_average = pressure_integral / plasma_volume

        beta_tor = beta_toroidal_from_p_B0(p_average, b0)
        beta_pol = beta_poloidal_from_pressure_integral(pressure_integral, r0, ip)
        if np.isfinite(beta_tor):
            ts["global_quantities.beta_tor"] = float(beta_tor)
        if np.isfinite(beta_pol):
            ts["global_quantities.beta_pol"] = float(beta_pol)

        minor_radius = _minor_radius(ts)
        if np.isfinite(beta_tor) and np.isfinite(minor_radius) and minor_radius > 0.0:
            beta_normal = beta_normal_from_beta_tor(beta_tor, minor_radius, b0, ip)
            if np.isfinite(beta_normal):
                ts["global_quantities.beta_normal"] = float(beta_normal)

        line_integrals = _surface_line_integrals(
            idx, frame, None if surfaces is None else surfaces.get(idx)
        )
        if line_integrals is None:
            continue
        length_pol, bp_dl = line_integrals
        if np.isfinite(length_pol):
            ts["global_quantities.length_pol"] = float(length_pol)
        bp2_volume = abs(
            2.0 * np.pi * float(trapz_compat(bp_dl, x=frame["psi_1d_radian"]))
        )
        li_3 = li_3_from_Bp2_volume_integral(bp2_volume, ip, r0)
        if np.isfinite(li_3):
            ts["global_quantities.li_3"] = float(li_3)


def _scalar_or_nan(node, path) -> float:
    try:
        return float(np.asarray(node[path], float).reshape(-1)[0])
    except Exception:
        return float("nan")


def _time_indexed_scalar(ods, path, idx) -> float:
    """One entry of a time-dependent scalar, tolerating a single stored value."""
    try:
        values = np.asarray(ods[path], float).reshape(-1)
    except Exception:
        return float("nan")
    if not values.size:
        return float("nan")
    return float(values[min(idx, values.size - 1)])


def _minor_radius(ts) -> float:
    """``a`` from ``boundary.minor_radius``, or from the outline it comes from."""
    if "boundary.minor_radius" in ts:
        value = _scalar_or_nan(ts, "boundary.minor_radius")
        if np.isfinite(value) and value > 0.0:
            return value
    if "boundary.outline.r" in ts:
        outline = np.asarray(ts["boundary.outline.r"], float).reshape(-1)
        if outline.size >= 3:
            return 0.5 * (float(np.max(outline)) - float(np.min(outline)))
    return float("nan")


def _surface_line_integrals(idx, frame, surfaces=None):
    """``(length_pol_edge, bp_dl_profile)`` for one slice.

    ``surfaces`` is a trace the caller already paid for; only without one does
    this trace again.
    """
    from vaft.process.equilibrium import flux_surface_quantities

    ts = frame["ts"]
    if surfaces is not None:
        bp_dl = np.asarray(surfaces["bp_dl"], float)
        if np.all(np.isfinite(bp_dl)):
            return float(np.asarray(surfaces["length_pol"], float)[-1]), bp_dl
        logger.warning("bp_dl is not finite for time slice %s; skipping li_3.", idx)
        return None

    f_profile = None
    if "profiles_1d.f" in ts:
        candidate = np.asarray(ts["profiles_1d.f"], float).reshape(-1)
        if candidate.size == frame["psi_norm"].size:
            f_profile = candidate
    try:
        surfaces = flux_surface_quantities(
            frame["psi_2d_radian"],
            frame["r_grid"],
            frame["z_grid"],
            frame["psi_axis_radian"],
            frame["psi_boundary_radian"],
            frame["psi_norm"],
            f_profile=f_profile,
            axis_rz=frame["axis_rz"],
            boundary=frame["boundary"],
        )
    except Exception as exc:
        logger.warning("Flux-surface trace failed for time slice %s: %s", idx, exc)
        return None
    bp_dl = np.asarray(surfaces["bp_dl"], float)
    if not np.all(np.isfinite(bp_dl)):
        logger.warning("bp_dl is not finite for time slice %s; skipping li_3.", idx)
        return None
    return float(np.asarray(surfaces["length_pol"], float)[-1]), bp_dl



def resolve_reference_major_radius(ods, time_slice_node=None) -> float:
    """``R_0`` for the DD global quantities, cross-checked against the TF IDS.

    ``beta_pol`` and ``li_3`` both divide by ``R_0``, which the DD takes from
    ``equilibrium.vacuum_toroidal_field.r0``. On the VEST database that leaf is
    not trustworthy: every shot sampled from HSDS stores an ``r0`` between 0.19
    and 0.35 while ``tf.r0`` is 0.4, and the two disagree about the physics --
    on shot 39915 the equilibrium's ``b0*r0`` is 0.0347 T.m against ``tf``'s
    0.0601 T.m. ``b0`` alone (0.1498 T) matches ``tf``'s field at R = 0.4, so
    ``r0`` is the corrupt half of the pair and the scalars come out 1.15-2.1x
    too large.

    ``b0 * r0`` is the physical invariant, so that product is what is compared:
    when it disagrees with ``tf.b_field_tor_vacuum_r`` by more than a few percent
    the equilibrium's ``r0`` is rejected in favour of ``tf.r0``, loudly. With no
    ``tf`` to check against the equilibrium's own value stands -- this detects a
    known corruption, it does not overrule a machine that genuinely has a
    different reference radius.

    Fixing the reader does not fix the database; the pipeline that writes
    ``vacuum_toroidal_field`` is where that belongs, and issue #325 owns it.
    """
    equilibrium_r0 = _scalar_or_nan(ods, "equilibrium.vacuum_toroidal_field.r0")
    tf_r0 = _scalar_or_nan(ods, "tf.r0")
    if not np.isfinite(tf_r0) or tf_r0 <= 0.0:
        return equilibrium_r0
    if not np.isfinite(equilibrium_r0) or equilibrium_r0 <= 0.0:
        logger.warning(
            "equilibrium.vacuum_toroidal_field.r0 is unusable (%s); using tf.r0 = %.4f m.",
            equilibrium_r0,
            tf_r0,
        )
        return tf_r0

    tf_field_radius = _finite_median(ods, "tf.b_field_tor_vacuum_r.data")
    b0 = _finite_median(ods, "equilibrium.vacuum_toroidal_field.b0")
    if not np.isfinite(tf_field_radius) or not np.isfinite(b0) or tf_field_radius == 0.0:
        return equilibrium_r0

    equilibrium_field_radius = b0 * equilibrium_r0
    if abs(equilibrium_field_radius - tf_field_radius) <= _TF_CONSISTENCY_TOLERANCE * abs(
        tf_field_radius
    ):
        return equilibrium_r0

    logger.warning(
        "equilibrium.vacuum_toroidal_field is inconsistent with tf: b0*r0 = %.5f T.m "
        "(b0 = %.5f T, r0 = %.5f m) against tf's %.5f T.m. Using tf.r0 = %.4f m for R_0; "
        "the stored r0 is the half that disagrees.",
        equilibrium_field_radius,
        b0,
        equilibrium_r0,
        tf_field_radius,
        tf_r0,
    )
    return tf_r0


#: How far ``b0*r0`` may sit from ``tf``'s ``B*R`` before the equilibrium's ``r0``
#: is rejected. The corruption this catches is a factor of 1.15-2.1, so a few
#: percent separates it from ordinary disagreement between two measurements of
#: the same field.
_TF_CONSISTENCY_TOLERANCE = 0.05


def _finite_median(node, path) -> float:
    """Median of the finite entries of a possibly time-dependent leaf."""
    try:
        values = np.asarray(node[path], float).reshape(-1)
    except Exception:
        return float("nan")
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan")
    return float(np.median(finite))


def update_equilibrium_derived_profiles(ods, time_slice=None):
    """Derive every flux-surface quantity an EFIT-sourced ODS omits, in order.

    Geometry first (it supplies ``gm1``/``gm9``), then the toroidal flux
    coordinate, then ``j_tor``, then the 0-D scalars that read the profiles back.
    Each step is independently callable; this is the one-line entry point.

    ``time_slice`` is normalized once here rather than forwarded verbatim: the
    older updaters below differ in what argument forms they accept, and an int
    used to reach the end of this sequence and raise after the profiles had
    already been written.
    """
    indices = _equilibrium_time_slices(ods, time_slice)
    if not indices:
        return
    surfaces = update_equilibrium_profiles_1d_geometry(
        ods, indices, return_surfaces=True
    )
    update_equilibrium_profiles_1d_toroidal_flux(ods, indices)
    update_equilibrium_profiles_1d_j_tor(ods, indices)
    update_equilibrium_global_quantities_volume(ods, indices)
    update_equilibrium_global_quantities_area(ods, indices)
    update_equilibrium_stored_energy(ods, indices)
    update_equilibrium_boundary(ods, indices)
    # Last: beta_normal reads boundary.minor_radius, which the line above writes.
    # The surfaces traced at the top are handed on rather than traced again.
    update_equilibrium_global_quantities_beta_li(ods, indices, surfaces=surfaces)

def update_equilibrium_profiles_2d_sfl_coordinates(ods, time_slice=None, profiles_2d_idx=1, convention='sfl', n_theta=129, plot_opt=0):
    """
    Update Straight Field Line (SFL) coordinates for `profiles_2d` entries.

    This function computes SFL coordinates (like PEST, equal arc-length, etc.)
    for specified time slices in an OMAS data structure. It populates a
    `profiles_2d` entry with the SFL grid (R, Z, ψ, θ_SFL).

    The `profiles_2d.grid.dim1` will store normalized poloidal flux (ψ_norm).
    The `profiles_2d.grid.dim2` will store the SFL poloidal angle (θ_SFL).
    The `profiles_2d.psi` will store the non-normalized poloidal flux (ψ).
    The `profiles_2d.r`, `profiles_2d.z`, `profiles_2d.theta` will store the
    corresponding R, Z, and SFL poloidal angle values on this grid.

    Parameters:
        ods (OMAS ODS): Input OMAS data structure.
        time_slice (int, list of int, None): Specifies which time slice(s) to process.
            If None, all time slices are processed.
            If int or list of int, these are treated as direct indices into
            `ods['equilibrium.time_slice']`.
        profiles_2d_idx (int): Index of the `profiles_2d` entry to update
            (e.g., `ods['equilibrium.time_slice'][t_idx]['profiles_2d'][profiles_2d_idx]`).
            Default is 1.
        convention (str): Poloidal angle convention for SFL coordinates.
            Supported values: 'sfl' (maps to 'straight_line'), 'straight_line' (PEST-like),
            'equal_arc', 'hamada', 'boozer'. Default is 'sfl'.
        n_theta (int): Number of poloidal points for the SFL grid. Default is 129.
        plot_opt (int): Plotting option:
            0: No plots.
            1: Show interactive plots of the SFL grid (ψ-θ_SFL) and the R-Z mesh.
            2: Same as 1, plus save plots to PNG files.
    """
    time_idx_list = []
    if time_slice is None:
        if 'equilibrium.time_slice' in ods and len(ods['equilibrium.time_slice']):
            time_idx_list = range(len(ods['equilibrium.time_slice']))
        else:
            print("Warning: No time slices found in ODS. Cannot update SFL coordinates.")
            return
    elif isinstance(time_slice, (int, np.integer)):
        time_idx_list = [time_slice]
    elif isinstance(time_slice, (list, np.ndarray)):
        time_idx_list = time_slice
    else:
        raise ValueError(f"time_slice must be an int, list of ints, or None. Got {type(time_slice)}")

    method_map = {
        'sfl': 'straight_line',
        'straight_line': 'straight_line',
        'equal_arc': 'equal_arc',
        'hamada': 'hamada',
        'boozer': 'boozer'
    }
    actual_method = method_map.get(convention.lower(), 'straight_line')
    if convention.lower() not in method_map:
        print(f"Warning: Unknown SFL convention {convention!r}; using geometric straight_line approximation.")
    elif actual_method not in {'sfl', 'straight_line'}:
        print(f"Warning: {convention!r} requested; using dependency-free geometric approximation.")

    for idx in time_idx_list:
        try:
            ts = ods['equilibrium.time_slice'][idx]
            time_val = ts.get('time', float(idx)) # Use actual time if available, else index
        except IndexError:
            print(f"Warning: Time slice index {idx} is out of bounds. Skipping.")
            continue

        try:
            dim1_vals, dim2_vals, R_2d, Z_2d, Psi_2d_values, Theta_2d_values = _geometric_flux_surface_grid(ts, n_theta)
        except Exception as e:
            print(f"Error building standalone SFL coordinates for time slice index {idx} (time {time_val}): {e}")
            continue

        prof2d = ts['profiles_2d'][profiles_2d_idx]

        prof2d['grid.dim1'] = dim1_vals
        prof2d['grid.dim2'] = dim2_vals
        prof2d['grid_type.index'] = 11
        # Name from IMAS data dictionary for grid_type 11: 'psi_norm_straight_field_line_theta'
        # The test script used 'inverse_psi_straight_field_line'.
        # Let's use the IMAS standard name if possible, but ensure consistency with project.
        # For now, sticking to what test script provides as it might be specific to vaft's OMAS usage.
        prof2d['grid_type.name'] = 'inverse_psi_straight_field_line'


        nr = len(dim1_vals)
        nt = len(dim2_vals)

        prof2d['r'] = R_2d
        prof2d['z'] = Z_2d
        prof2d['psi'] = Psi_2d_values
        prof2d['theta'] = Theta_2d_values

        if plot_opt >= 1:
            _plot_sfl_grid(
                prof2d, ts, nr, nt, time_val, profiles_2d_idx, convention,
                save=plot_opt >= 2,
            )

def update_equilibrium_stored_energy(ods, time_slice=None):
    """
    Update stored energy for all time slices. [ref. omas.physics_equilibrium_stored_energy]
    """
    if time_slice is None:
        if 'equilibrium.time_slice' in ods and len(ods['equilibrium.time_slice']):
            time_slice = range(len(ods['equilibrium.time_slice']))
        else:
            print("Warning: No time slices found in ODS. Cannot update stored energy.")
            return
    # Convert a single integer to a list for iteration, as the sibling
    # update_equilibrium_global_quantities_volume already did; without it any
    # caller passing one index -- including update_equilibrium_derived_profiles
    # -- got "TypeError: 'int' object is not iterable".
    if isinstance(time_slice, (int, np.integer)):
        time_slice = [time_slice]
    for idx in time_slice:
        ts = ods['equilibrium.time_slice'][idx]
        # check if profiles_1d.pressure and profiles_1d.volume exist
        if 'profiles_1d.pressure' not in ts or 'profiles_1d.volume' not in ts:
            print(f"Warning: profiles_1d.pressure or profiles_1d.volume not found for time slice {idx}")
            continue
        pressure_equil = ts['profiles_1d.pressure']
        volume_equil = ts['profiles_1d.volume']
        ts['global_quantities.energy_mhd'] = 3.0 / 2.0 * trapz_compat(pressure_equil, x=volume_equil)


def update_core_profiles_global_quantities_volume_average(ods, time_slice=None):
    """
    Update core_profiles global quantities with volume-averaged n_e, T_e, and ion n_i, T_i.
    The function matches core_profiles time indices to equilibrium time slices
    by finding the closest matching time values.
    """
    from vaft.process.equilibrium import psi_to_rz, volume_average
    
    # Basic availability checks
    if 'core_profiles.profiles_1d' not in ods:
        print("Warning: core_profiles.profiles_1d not found in ODS.")
        return
    if 'equilibrium.time_slice' not in ods or not len(ods['equilibrium.time_slice']):
        print("Warning: equilibrium.time_slice not found in ODS.")
        return

    n_core_slices = len(ods['core_profiles.profiles_1d'])
    n_equil_slices = len(ods['equilibrium.time_slice'])

    # Extract time arrays for matching
    core_times = []
    for idx in range(n_core_slices):
        cp_ts = ods['core_profiles.profiles_1d'][idx]
        if 'time' in cp_ts:
            core_times.append(float(cp_ts['time']))
        elif 'core_profiles.time' in ods and idx < len(ods['core_profiles.time']):
            core_times.append(float(ods['core_profiles.time'][idx]))
        else:
            print(f"Warning: time not found for core_profiles.profiles_1d[{idx}], using index as time")
            core_times.append(float(idx))
    
    equil_times = []
    for idx in range(n_equil_slices):
        eq_ts = ods['equilibrium.time_slice'][idx]
        if 'time' in eq_ts:
            equil_times.append(float(eq_ts['time']))
        elif 'equilibrium.time' in ods and idx < len(ods['equilibrium.time']):
            equil_times.append(float(ods['equilibrium.time'][idx]))
        else:
            print(f"Warning: time not found for equilibrium.time_slice[{idx}], using index as time")
            equil_times.append(float(idx))
    
    core_times = np.asarray(core_times)
    equil_times = np.asarray(equil_times)

    # Build list of core profile indices to process
    if time_slice is None:
        core_indices = range(n_core_slices)
    elif isinstance(time_slice, (int, np.integer)):
        core_indices = [time_slice] if time_slice < n_core_slices else []
    else:
        core_indices = [idx for idx in time_slice if idx < n_core_slices]

    # Initialize result lists for all time slices
    n_e_vol_list = []
    T_e_vol_list = []
    ion_vol_dict = {}  # {ion_idx: {'n_i': [], 'T_i': []}}

    # Step 1 & 2: Process each core profile time slice
    for cp_idx in core_indices:
        cp_time = core_times[cp_idx]
        
        # Find closest equilibrium time
        equil_idx = np.argmin(np.abs(equil_times - cp_time))
        time_diff = abs(equil_times[equil_idx] - cp_time)
        
        if time_diff > 0.1:  # Warn if time difference is large (> 100ms)
            print(f"Warning: Large time difference ({time_diff:.3f}s) between core_profiles[{cp_idx}] (t={cp_time:.3f}s) and equilibrium[{equil_idx}] (t={equil_times[equil_idx]:.3f}s)")
        
        cp_ts = ods['core_profiles.profiles_1d'][cp_idx]
        eq_ts = ods['equilibrium.time_slice'][equil_idx]

        # Get 1D flux coordinate for core profiles (always rho_tor_norm)
        grid = cp_ts.get('grid', ods['core_profiles'].get('grid', ODS()))
        if 'rho_tor_norm' not in grid:
            print(f"Warning: rho_tor_norm grid missing for core_profiles.profiles_1d[{cp_idx}], skipping")
            n_e_vol_list.append(np.nan)
            T_e_vol_list.append(np.nan)
            # Append NaN for all existing ions
            for ion_idx in ion_vol_dict.keys():
                ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                ion_vol_dict[ion_idx]['T_i'].append(np.nan)
            continue
        
        rho_tor_norm_cp = np.asarray(grid['rho_tor_norm'], float)

        # Get equilibrium profiles_1d for coordinate conversion
        eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
        
        # Ensure equilibrium has psi_norm (create if missing)
        if 'psi_norm' not in eq_profiles_1d:
            update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=equil_idx)
            eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
            if 'psi_norm' not in eq_profiles_1d:
                print(f"Warning: failed to create psi_norm for equilibrium.time_slice[{equil_idx}], skipping")
                n_e_vol_list.append(np.nan)
                T_e_vol_list.append(np.nan)
                for ion_idx in ion_vol_dict.keys():
                    ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                    ion_vol_dict[ion_idx]['T_i'].append(np.nan)
                continue
        
        # Get equilibrium rho_tor_norm and psi_norm for coordinate mapping
        if 'rho_tor_norm' not in eq_profiles_1d:
            print(f"Warning: rho_tor_norm missing in equilibrium.profiles_1d for time_slice[{equil_idx}], skipping")
            n_e_vol_list.append(np.nan)
            T_e_vol_list.append(np.nan)
            for ion_idx in ion_vol_dict.keys():
                ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                ion_vol_dict[ion_idx]['T_i'].append(np.nan)
            continue
        
        rho_tor_norm_eq = np.asarray(eq_profiles_1d['rho_tor_norm'], float)
        psi_norm_eq = np.asarray(eq_profiles_1d['psi_norm'], float)
        
        # Ensure monotonicity for equilibrium rho_tor_norm and psi_norm mapping
        if not np.all(np.diff(rho_tor_norm_eq) > 0):
            sort_idx = np.argsort(rho_tor_norm_eq)
            rho_tor_norm_eq_sorted = rho_tor_norm_eq[sort_idx]
            psi_norm_eq_sorted = psi_norm_eq[sort_idx]
            unique_mask = np.concatenate(([True], np.diff(rho_tor_norm_eq_sorted) > 1e-10))
            rho_tor_norm_eq_sorted = rho_tor_norm_eq_sorted[unique_mask]
            psi_norm_eq_sorted = psi_norm_eq_sorted[unique_mask]
        else:
            rho_tor_norm_eq_sorted = rho_tor_norm_eq
            psi_norm_eq_sorted = psi_norm_eq
        
        # Use equilibrium psi_norm grid as target coordinate system
        psiN_1d = psi_norm_eq_sorted
        
        # Create inverse mapping: psi_norm -> rho_tor_norm
        interp_psi_to_rho = interp1d(psi_norm_eq_sorted, rho_tor_norm_eq_sorted,
                                     kind='linear',
                                     bounds_error=False,
                                     fill_value=(rho_tor_norm_eq_sorted[0], rho_tor_norm_eq_sorted[-1]))
        rho_tor_norm_at_psiN = interp_psi_to_rho(psiN_1d)

        # Equilibrium 2D grid and ψ(R,Z)
        try:
            R_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim1'], float)
            Z_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim2'], float)
            psi_RZ = np.asarray(eq_ts['profiles_2d.0.psi'], float)
            psi_axis = float(eq_ts['global_quantities.psi_axis'])
            psi_lcfs = float(eq_ts['global_quantities.psi_boundary'])
        except KeyError:
            print(f"Warning: missing profiles_2d.0 or global_quantities.psi_* for equilibrium.time_slice[{equil_idx}], skipping")
            n_e_vol_list.append(np.nan)
            T_e_vol_list.append(np.nan)
            for ion_idx in ion_vol_dict.keys():
                ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                ion_vol_dict[ion_idx]['T_i'].append(np.nan)
            continue

        # Helper function: rho_tor_norm -> psi_norm -> 2D RZ
        def convert_to_2d(profile_1d_rho):
            # Interpolate profile from rho_tor_norm_cp to rho_tor_norm_at_psiN
            interp_func = interp1d(rho_tor_norm_cp, profile_1d_rho,
                                  kind='linear',
                                  bounds_error=False,
                                  fill_value=(profile_1d_rho[0], profile_1d_rho[-1]))
            profile_1d = interp_func(rho_tor_norm_at_psiN)
            profile_RZ, psiN_RZ = psi_to_rz(psiN_1d, profile_1d, psi_RZ, psi_axis, psi_lcfs)
            return profile_RZ, psiN_RZ

        # Step 2: Process electron profiles
        psiN_RZ = None
        n_e_vol = np.nan
        T_e_vol = np.nan
        
        if 'electrons.density' in cp_ts and 'electrons.temperature' in cp_ts:
            try:
                # Process n_e
                n_e_1d_rho = np.asarray(cp_ts['electrons.density'], float)
                n_e_RZ, psiN_RZ = convert_to_2d(n_e_1d_rho)
                n_e_vol, _ = volume_average(n_e_RZ, psiN_RZ, R_grid, Z_grid)
                
                # Process T_e
                T_e_1d_rho = np.asarray(cp_ts['electrons.temperature'], float)
                T_e_RZ, _ = convert_to_2d(T_e_1d_rho)
                T_e_vol, _ = volume_average(T_e_RZ, psiN_RZ, R_grid, Z_grid)
            except Exception as e:
                print(f"Warning: Error processing electron profiles for core_profiles[{cp_idx}]: {e}")
        else:
            print(f"Warning: electrons density/temperature missing in core_profiles.profiles_1d[{cp_idx}]")
        
        n_e_vol_list.append(n_e_vol)
        T_e_vol_list.append(T_e_vol)

        # Step 2: Process ion profiles (each ion individually)
        if 'ion' in cp_ts and cp_ts['ion']:
            # Get list of ion indices
            ion_indices = []
            if isinstance(cp_ts['ion'], dict):
                ion_indices = list(cp_ts['ion'].keys())
            elif isinstance(cp_ts['ion'], (list, tuple)):
                ion_indices = list(range(len(cp_ts['ion'])))
            
            for ion_idx in ion_indices:
                # Initialize ion result lists if not exists
                if ion_idx not in ion_vol_dict:
                    ion_vol_dict[ion_idx] = {'n_i': [], 'T_i': []}
                    # Fill with NaN for previous time slices
                    for _ in range(len(n_e_vol_list) - 1):
                        ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                        ion_vol_dict[ion_idx]['T_i'].append(np.nan)
                
                # Get ion data
                if isinstance(cp_ts['ion'], dict):
                    ion_ts = cp_ts['ion'][ion_idx]
                else:
                    ion_ts = cp_ts['ion'][ion_idx]
                
                # Check if ion_ts is valid and has required keys
                if not isinstance(ion_ts, dict) or ion_ts is None:
                    ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                    ion_vol_dict[ion_idx]['T_i'].append(np.nan)
                    continue
                
                if 'density' not in ion_ts or 'temperature' not in ion_ts:
                    ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                    ion_vol_dict[ion_idx]['T_i'].append(np.nan)
                    continue
                
                # Process ion profiles
                try:
                    n_i_1d_rho = np.asarray(ion_ts['density'], float)
                    T_i_1d_rho = np.asarray(ion_ts['temperature'], float)
                    
                    # Check if arrays are valid
                    if n_i_1d_rho.size == 0 or T_i_1d_rho.size == 0:
                        ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                        ion_vol_dict[ion_idx]['T_i'].append(np.nan)
                        continue
                    if np.all(np.isnan(n_i_1d_rho)) or np.all(np.isnan(T_i_1d_rho)):
                        ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                        ion_vol_dict[ion_idx]['T_i'].append(np.nan)
                        continue
                    
                    # Convert to 2D RZ and compute volume average
                    n_i_RZ, _ = convert_to_2d(n_i_1d_rho)
                    T_i_RZ, _ = convert_to_2d(T_i_1d_rho)
                    
                    n_i_vol, _ = volume_average(n_i_RZ, psiN_RZ, R_grid, Z_grid)
                    T_i_vol, _ = volume_average(T_i_RZ, psiN_RZ, R_grid, Z_grid)
                    
                    ion_vol_dict[ion_idx]['n_i'].append(n_i_vol)
                    ion_vol_dict[ion_idx]['T_i'].append(T_i_vol)
                    
                except Exception as e:
                    print(f"Warning: Error processing ion[{ion_idx}] for core_profiles[{cp_idx}]: {e}")
                    ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                    ion_vol_dict[ion_idx]['T_i'].append(np.nan)
        else:
            # No ions for this time slice, append NaN for all existing ions
            for ion_idx in ion_vol_dict.keys():
                ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                ion_vol_dict[ion_idx]['T_i'].append(np.nan)

    # Step 3: Store results in core_profiles.global_quantities
    if 'core_profiles.global_quantities' not in ods:
        ods['core_profiles.global_quantities'] = ODS()
    
    gq = ods['core_profiles.global_quantities']
    gq['n_e_volume_average'] = n_e_vol_list
    gq['t_e_volume_average'] = T_e_vol_list
    
    # Store ion results only if ion data exists
    if ion_vol_dict:
        # Store ion results
        if 'ion' not in gq:
            gq['ion'] = []
        
        # Ensure ion array has enough elements
        max_ion_idx = max(ion_vol_dict.keys())
        while len(gq['ion']) <= max_ion_idx:
            gq['ion'].append(ODS())
        
        for ion_idx, results in ion_vol_dict.items():
            gq['ion'][ion_idx]['n_i_volume_average'] = results['n_i']
            gq['ion'][ion_idx]['t_i_volume_average'] = results['T_i']


def update_equilibrium_constraints_diamagnetic_flux(ods, time_slice=None):
    """
    Update equilibrium constraints with diamagnetic flux (measured and reconstructed).

    - Measured: interpolates magnetics.diamagnetic_flux.0.data at magnetics.time
      onto each equilibrium time and stores in
      equilibrium.time_slice[:].constraints.diamagnetic_flux.measured.
    - Reconstructed: computes from equilibrium field via compute_reconstructed_diamagnetic_flux
      and stores in equilibrium.time_slice[:].constraints.diamagnetic_flux.reconstructed.

    Parameters
    ----------
    ods : OMAS ODS
        Data structure with equilibrium.time_slice and optionally magnetics.
    time_slice : int, list of int, or None
        Time slice index/indices to update. None = all slices.
    """
    if "equilibrium.time_slice" not in ods or not len(ods["equilibrium.time_slice"]):
        return

    from vaft.omas.process_wrapper import compute_reconstructed_diamagnetic_flux

    time_slices = (
        range(len(ods["equilibrium.time_slice"]))
        if time_slice is None
        else ([time_slice] if isinstance(time_slice, (int, np.integer)) else time_slice)
    )

    # Build interpolator for measured diamagnetic flux (magnetics -> equilibrium time)
    interp_measured = None
    if (
        "magnetics.diamagnetic_flux.0.data" in ods
        and "magnetics.time" in ods
        and len(ods.get("magnetics.diamagnetic_flux", [])) > 0
    ):
        t_mag = np.asarray(ods["magnetics.time"], float)
        flux_mag = np.asarray(ods["magnetics.diamagnetic_flux.0.data"], float)
        if t_mag.size >= 1 and flux_mag.size == t_mag.size:
            if t_mag.size >= 2:
                interp_measured = interp1d(
                    t_mag,
                    flux_mag,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(flux_mag[0], flux_mag[-1]),
                )
            else:
                interp_measured = lambda t: float(flux_mag[0])  # noqa: E731

    for idx in time_slices:
        if idx < 0 or idx >= len(ods["equilibrium.time_slice"]):
            continue
        eq_ts = ods["equilibrium.time_slice"][idx]

        # Measured: interp magnetics at this equilibrium time
        if interp_measured is not None:
            if "time" in eq_ts:
                t_eq = float(eq_ts["time"])
            elif "equilibrium.time" in ods and idx < len(ods["equilibrium.time"]):
                t_eq = float(ods["equilibrium.time"][idx])
            else:
                t_eq = np.nan
            if np.isfinite(t_eq):
                if "constraints" not in eq_ts:
                    eq_ts["constraints"] = ODS()
                if "diamagnetic_flux" not in eq_ts["constraints"]:
                    eq_ts["constraints"]["diamagnetic_flux"] = ODS()
                eq_ts["constraints"]["diamagnetic_flux"]["measured"] = float(
                    interp_measured(t_eq)
                )

        # Reconstructed: compute from equilibrium
        try:
            recon = compute_reconstructed_diamagnetic_flux(ods, time_index=idx)
            if "constraints" not in eq_ts:
                eq_ts["constraints"] = ODS()
            if "diamagnetic_flux" not in eq_ts["constraints"]:
                eq_ts["constraints"]["diamagnetic_flux"] = ODS()
            eq_ts["constraints"]["diamagnetic_flux"]["reconstructed"] = recon
        except Exception as e:
            print(f"Warning: Reconstructed diamagnetic flux failed for time_slice {idx}: {e}")


def _plot_radial_mapping_validation(ts, idx, r_in, psi_in, r_out, psi_out, psi_1d):
    """Render the four radial-coordinate validation panels through ``vaft.plot``."""
    from vaft.plot import Profile1D, Panels, Series, render_panels

    axis_r = ts["global_quantities.magnetic_axis.r"]
    axis_style = {"marker": ".", "linestyle": "none", "color": "k"}

    psi_panel = Profile1D(
        series=(
            Series(x=r_in, y=psi_in, label="2D Inboard", style={"color": "r"}),
            Series(x=ts["profiles_1d.r_inboard"], y=psi_1d, label="Inboard",
                   style={"color": "b", "linestyle": "--"}),
            Series(x=[axis_r], y=[ts["global_quantities.psi_axis"]],
                   label="Magnetic Axis", style=axis_style),
            Series(x=r_out, y=psi_out, label="2D Outboard", style={"color": "g"}),
            Series(x=ts["profiles_1d.r_outboard"], y=psi_1d, label="Outboard",
                   style={"color": "m", "linestyle": "--"}),
        ),
        coordinate_label="R [m]", y_label="Psi",
    )

    def _inboard_outboard(quantity, label, unit="", axis_value=None):
        series = [
            Series(x=ts["profiles_1d.r_inboard"], y=ts[f"profiles_1d.{quantity}"],
                   label="Inboard", style={"color": "r"}),
            Series(x=ts["profiles_1d.r_outboard"], y=ts[f"profiles_1d.{quantity}"],
                   label="Outboard", style={"color": "g"}),
        ]
        if axis_value is not None:
            series.insert(
                1,
                Series(x=[axis_r], y=[axis_value], label="Magnetic Axis",
                       style=axis_style),
            )
        return Profile1D(series=tuple(series), coordinate_label="R [m]",
                         y_label=label, y_unit=unit)

    return render_panels(
        Panels(
            models=(
                psi_panel,
                _inboard_outboard("j_tor", "J_tor", "A/m2"),
                _inboard_outboard("pressure", "Pressure", "Pa"),
                _inboard_outboard("q", "safety factor",
                                  axis_value=ts["global_quantities.q_axis"]),
            ),
            ncols=2, share_x=False,
            suptitle=f"Time Slice {idx} Validation",
        ),
        figsize=(15, 10),
    )


def _plot_sfl_grid(prof2d, ts, nr, nt, time_val, profiles_2d_idx, convention,
                   *, save=False):
    """Render the straight-field-line psi-theta and R-Z meshes through ``vaft.plot``."""
    from vaft.plot import (
        GeometryLayer,
        GeometryLayers,
        LineSeries,
        Series,
        render_geometry_layers,
        render_line_series,
        save_figure,
    )

    theta = prof2d["grid.dim2"]
    psi_theta = LineSeries(
        series=tuple(
            Series(x=theta, y=np.full_like(theta, prof2d["psi"][i_surf, 0]),
                   style={"color": "k", "lw": 0.5})
            for i_surf in range(nr)
        ),
        x_label=r"$\theta_{\rm SFL}$", x_unit="rad",
        y_label=r"$\psi$", y_unit="Wb",
        y_limits=(min(prof2d["psi"][:, 0]), max(prof2d["psi"][:, 0])),
        title=f"Time: {time_val:.4f}s - psi-theta SFL Grid "
              f"(prof2d[{profiles_2d_idx}], {convention})",
    )
    psi_figure, _ = render_line_series(psi_theta, figsize=(10, 8), legend=False)

    layers = [
        GeometryLayer(r=prof2d["r"][i_surf, :], z=prof2d["z"][i_surf, :],
                      label="Flux Surface" if i_surf == 0 else "",
                      style={"color": "b", "lw": 0.7})
        for i_surf in range(nr)
    ]
    layers += [
        GeometryLayer(r=prof2d["r"][:, j_theta], z=prof2d["z"][:, j_theta],
                      label="SFL theta line" if j_theta == 0 else "",
                      style={"color": "r", "linestyle": "--", "lw": 0.5})
        for j_theta in range(0, nt, max(1, nt // 16))
    ]
    global_quantities = ts.get("global_quantities", {})
    if "magnetic_axis.r" in global_quantities and "magnetic_axis.z" in global_quantities:
        layers.append(
            GeometryLayer(
                r=[global_quantities["magnetic_axis.r"]],
                z=[global_quantities["magnetic_axis.z"]],
                kind="points", label="Mag. Axis",
                style={"marker": "x", "color": "k", "markersize": 10, "mew": 2},
            )
        )
    mesh_figure, _ = render_geometry_layers(
        GeometryLayers(
            layers=tuple(layers),
            title=f"Time: {time_val:.4f}s - SFL R-Z Mesh "
                  f"(prof2d[{profiles_2d_idx}], {convention})",
        ),
        figsize=(10, 10),
    )

    if save:
        stem = f"t{time_val:.3f}_idx{profiles_2d_idx}_{convention}"
        save_figure(psi_figure, f"sfl_psi_theta_{stem}.png", close=False)
        save_figure(mesh_figure, f"sfl_rz_mesh_{stem}.png", close=False)
    return (psi_figure, mesh_figure)
