"""Axisymmetric equilibrium: from a flux map to the quantities derived from it.

Physics only. Nothing here reads or writes an ODS: every function takes arrays
and returns arrays or plain records, so the same code serves a GEQDSK, an ODS
slice and an analytic model. The adapters that turn those into one record, and
the parametric models fitted to them, live in ``_equilibrium_parametric`` and
are re-exported through this module, which is their stable import location.

What the module does, in the order a flux map is usually consumed::

    psi(R, Z) on a grid
    -> radial coordinates: psi_norm, and the toroidal radius through q
    -> flux surfaces: contours, their shape, their averages
    -> fields: the poloidal field from psi, the toroidal field from F
    -> global scalars: diamagnetism, the Shafranov integrals, force balance

Notation
--------
psi           : poloidal flux                                          [Wb/rad]
psi_norm      : (psi - psi_axis)/(psi_boundary - psi_axis)                  [-]
F             : poloidal current function, R*B_phi                       [T m]
B_p           : poloidal field magnitude                                   [T]
q             : safety factor                                              [-]
k             : Sauter Eq. 20 prefactor carrying sign and 2*pi              [-]

Conventions
-----------
**Flux is per radian unless a function says otherwise.** The poloidal field
from psi carries a factor ``k = sigma_RphiZ * sigma_Bp / (2*pi)**e_Bp`` [1]_,
which folds the orientation sign and the 2*pi normalization together. A
function that forms a field from psi takes a COCOS index, and with none it
uses the historical weber-per-radian form. Getting this wrong scales the field
by 2*pi and the poloidal beta by its square, plausibly and silently. A function
that takes a *precomputed* field instead inherits whatever convention produced
it and cannot check it.

**Three radial coordinates, never interchangeable.** ``rho_tor_norm`` equals
``sqrt(psi_norm)``, which is ``rho_pol_norm``, only for a flat safety factor.
:func:`derive_radial_coordinates` returns all three with an explicit record of
which is unavailable and why.

**Flux maps are indexed (R, Z)**, matching the IMAS two-dimensional profile
layout. **Boundary contours run counter-clockwise**, because the Shafranov
outward normal is only outward then.

Provenance
----------
.. [1] Sauter and Medvedev, *Tokamak Coordinate Conventions: COCOS*, Comput.
   Phys. Commun. 184, 293 (2013): Eq. 20 for the field prefactor and Eq. 23
   for the sign relations :mod:`vaft.process.cocos` checks against.
.. [2] The EFIT workflow this reproduces, whose ``seva2d`` field evaluation,
   weighted volume integrals and cell-weight map are named at the functions
   that stand in for them.
.. [3] The IMAS data dictionary, for the flux-surface-average family, the
   boundary shape definitions, and the parallel current definition.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Optional

from scipy.interpolate import RectBivariateSpline

import numpy as np
from scipy.interpolate import interp1d

from vaft.formula.constants import MU0


#: The parametric API re-exported at the bottom of this module from
#: ``._equilibrium_parametric``, which keeps this module as its stable public
#: import location.  Listed explicitly so ``__all__`` stays readable and so a
#: name cannot join the public surface just by being imported here.
_PARAMETRIC_EXPORTS = (
    "as_equilibrium",
    "check_equilibrium_requirements",
    "convert_cocos",
    "derive_boundary_representation",
    "derive_global_descriptors",
    "derive_radial_coordinates",
    "evaluate_miller",
    "evaluate_solovev",
    "fit_miller_sequence",
    "fit_miller_surface",
    "solovev_to_equilibrium",
    "solve_solovev_constraints",
    "validate_equilibrium",
)

__all__ = [
    "FLUX_SURFACE_QUANTITIES",
    "MIN_ANNULUS_CELLS",
    "MIN_FLUX_SURFACE_POINTS",
    "calculate_average_boundary_poloidal_field",
    "calculate_diamagnetism",
    "calculate_q_profile_from_psi",
    "find_rational_surfaces",
    "StraightFieldLineMap",
    "lab_to_straight_field_line",
    "straight_field_line_angle_on_grid",
    "straight_field_line_map",
    "straight_field_line_tables",
    "resistive_layer_at",
    "resistive_layer_parameters",
    "calculate_reconstructed_diamagnetic_flux",
    "computed_diamagnetism_from_phi",
    "contour_shape_parameters",
    "efit_virial_volume_integrals",
    "extract_flux_surface_contours",
    "ParallelCurrentResult",
    "flux_surface_quantities",
    "fractional_cell_weights_from_boundary",
    "GradShafranovResidual",
    "grad_shafranov_operator",
    "grad_shafranov_residual",
    "equilibrium_field_on_grid",
    "make_equilibrium_field_interpolator",
    "make_vacuum_field_interpolator",
    "parallel_current_from_toroidal",
    "poloidal_field_at_boundary",
    "prepare_boundary_for_shafranov",
    "psi_to_RZ",
    "plasma_cell_weights",
    "psi_to_radial",
    "psi_to_rho",
    "psi_to_rz",
    "r_at_z_extremum",
    "radial_to_psi",
    "rho_to_psi",
    "scale_boundary_conformal",
    "shafranov_integrals",
    "connection_length_map",
    "ejiri_mirror_geometry",
    "trace_field_line",
    "virial_alpha_conformal_annulus",
    "virial_alpha_thin_annulus",
    "volume_average",
    *_PARAMETRIC_EXPORTS,
]


def radial_to_psi(r, psi_R, psi_Z, psi):
    """Poloidal flux at a major radius on the midplane row of the flux map.

    Parameters
    ----------
    r : float
        Major radius at which to evaluate [m].
    psi_R : array_like
        Major-radius grid axis of the flux map, increasing [m].
    psi_Z : array_like
        Height grid axis of the flux map [m].
    psi : array_like
        Poloidal flux on the grid, indexed ``(R, Z)`` [Wb/rad].

    Returns
    -------
    float
        Poloidal flux at *r*, in the same unit the input map was in [Wb/rad].

    Convention
    ----------
    The flux map is indexed major radius first. The row taken is the one whose
    height is nearest zero, not an interpolation between rows, so on a grid with
    no node at the midplane the result is the flux on the nearest row rather than
    on the midplane itself. The unit is whatever the caller's map carries; nothing
    here rescales it, so a weber map returns weber.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Cubic interpolation along one row, so *r* must lie inside the grid's radial
    span and the result is only as smooth as the map. A machine whose midplane is
    not at zero height needs the row selected before calling.

    Provenance
    ----------
    .. [1] The midplane-row convention follows the EFIT-style flux maps this
       package reads, where the grid is centred on the machine midplane.
    """
    # Find the index of Z=0 in psi_Z array
    z0_idx = np.argmin(np.abs(psi_Z))
    
    # Extract the psi values at Z=0
    psi_at_z0 = psi[:, z0_idx]
    
    # Create 1D interpolation function
    psi_interp = interp1d(psi_R, psi_at_z0, kind='cubic')
    
    # Return interpolated value
    return float(psi_interp(r))

def psi_to_rho(psi_val, q_profile, psi_axis, psi_boundary):
    """Normalized toroidal-flux radius at one poloidal flux value, by integrating q.

    Parameters
    ----------
    psi_val : float
        Poloidal flux at which to evaluate [Wb/rad].
    q_profile : callable
        Safety factor as a function of *normalized* poloidal flux, called on
        ``[0, 1]`` [-].
    psi_axis : float
        Poloidal flux on the magnetic axis [Wb/rad].
    psi_boundary : float
        Poloidal flux at the plasma boundary [Wb/rad].

    Returns
    -------
    float
        The normalized toroidal-flux radius, 0 on axis and 1 at the boundary [-].

    Convention
    ----------
    This is **rho_tor_norm**, the toroidal coordinate, not the poloidal one:
    ``sqrt(int q dpsi_n / int_0^1 q dpsi_n)``. It equals ``sqrt(psi_norm)``, which
    is ``rho_pol_norm``, only for a flat safety factor. The parameter name says
    "rho" for historical reasons and the two must not be interchanged; see
    :func:`vaft.process.equilibrium.derive_radial_coordinates` for the record-based
    form that returns all three coordinates at once and says which is unavailable.

    Because both integrals are of the same profile over the same variable, the
    result is independent of the flux storage convention: the sign and any factor
    of ``2*pi`` cancel in the ratio.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Takes the safety factor as a callable on normalized flux, so a caller holding
    it on a grid must interpolate first. Adaptive quadrature is evaluated afresh
    for every call, which makes converting a whole profile point by point far more
    expensive than :func:`derive_radial_coordinates`, and it assumes the profile
    is integrable to the boundary.

    Provenance
    ----------
    .. [1] The toroidal flux definition ``Phi = int q dpsi`` and its normalized
       radius, as the IMAS data dictionary defines ``rho_tor_norm``.
    """
    from scipy.integrate import quad
    
    # First normalize psi
    psi_N = (psi_val - psi_axis) / (psi_boundary - psi_axis)
    
    # Define the integration for numerator and denominator
    def integrand(x):
        return q_profile(x)
    
    # Compute the integrals
    numerator, _ = quad(integrand, 0, psi_N)
    denominator, _ = quad(integrand, 0, 1.0)
    
    # Return normalized radius
    return np.sqrt(numerator / denominator)

def rho_to_psi(rho, q_profile, psi_axis, psi_boundary, tol=1e-6):
    """Poloidal flux at one normalized toroidal-flux radius, by root finding.

    The inverse of :func:`psi_to_rho`, obtained numerically because the forward
    map is an integral with no closed-form inverse.

    Parameters
    ----------
    rho : float
        Normalized toroidal-flux radius, between 0 and 1 [-].
    q_profile : callable
        Safety factor as a function of normalized poloidal flux [-].
    psi_axis : float
        Poloidal flux on the magnetic axis, one end of the bracket [Wb/rad].
    psi_boundary : float
        Poloidal flux at the plasma boundary, the other end [Wb/rad].
    tol : float, optional
        Relative tolerance passed to the root finder [-].

    Returns
    -------
    float
        The poloidal flux whose toroidal radius is *rho* [Wb/rad].

    Convention
    ----------
    *rho* is the toroidal coordinate, matching :func:`psi_to_rho`, not
    ``sqrt(psi_norm)``. The returned flux is in whatever unit *psi_axis* and
    *psi_boundary* were given in, since the bracket sets the scale.

    Defaults
    --------
    ``tol = 1e-6`` is a numerical convenience: the forward map is monotonic and
    smooth, so Brent's method reaches it in few iterations, and a tighter value
    buys accuracy the quadrature underneath does not have.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Brackets on the axis and boundary flux, so it assumes the forward map is
    monotonic between them, which needs a safety factor of one sign. Each
    iteration runs the forward quadrature, making this the most expensive way to
    convert a coordinate; invert an interpolated table instead when converting
    many points.

    Provenance
    ----------
    .. [1] :func:`psi_to_rho`, whose definition this inverts.
    """
    from scipy.optimize import root_scalar
    
    def objective(psi):
        return psi_to_rho(psi, q_profile, psi_axis, psi_boundary) - rho
    
    # Find psi value that gives desired rho
    result = root_scalar(objective, 
                        bracket=[psi_axis, psi_boundary],
                        method='brentq',
                        rtol=tol)
    
    return result.root

def psi_to_rz(
    psiN_1d: np.ndarray,
    f_1d: np.ndarray,
    psi_RZ: np.ndarray,
    psi_axis: float,
    psi_lcfs: float,
    fill_outside: str = "zero",
    ):
    """Map a flux-surface profile onto the two-dimensional grid through psi.

    Parameters
    ----------
    psiN_1d : array_like
        Normalized poloidal flux the profile is given on [-].
    f_1d : array_like
        The profile, in whatever unit it has [any].
    psi_RZ : array_like
        Poloidal flux on the grid, indexed ``(R, Z)`` [Wb/rad].
    psi_axis : float
        Poloidal flux on the magnetic axis [Wb/rad].
    psi_lcfs : float
        Poloidal flux at the last closed flux surface [Wb/rad].
    fill_outside : str, optional
        ``"zero"`` (default) zeroes every cell outside ``0 <= psiN <= 1``;
        ``"edge"`` leaves the clamped profile there, so a cell continues at the
        nearest end value [-].

    Returns
    -------
    f_RZ : np.ndarray
        The profile on the grid, zero outside the boundary unless
        ``fill_outside="edge"`` [any].
    psiN_RZ : np.ndarray
        Normalized poloidal flux on the grid [-].

    Raises
    ------
    ValueError
        The profile and its abscissa are not one-dimensional and of equal length,
        or ``fill_outside`` is neither ``"zero"`` nor ``"edge"``.

    Processing steps
    ----------------
    1. Normalize the flux map to ``(psi - psi_axis)/(psi_lcfs - psi_axis)``.
    2. Sort the profile by its abscissa, clamp the normalized map into the
       profile's own span, and interpolate linearly.
    3. Zero every cell outside the closed region.

    Convention
    ----------
    The flux map is indexed major radius first. Outside the boundary the value is
    zero, not the edge value and not a NaN, so a sum over the grid is already a
    plasma-only integral -- as far as ``0 <= psiN <= 1`` is the plasma, which
    near the coils it is not (see :func:`volume_average`).  Pass
    ``fill_outside="edge"`` when the map is to be weighted by
    :func:`plasma_cell_weights`: an outline-weighted edge cell can sit just past
    ``psiN = 1``, and a zero there biases the average low (0.5-0.8 % for a
    profile whose edge value is 30 % of its core, on the packaged samples). Because only the normalized flux is used, the absolute
    unit of the three flux arguments cancels: weber and weber per radian give the
    same answer as long as all three agree.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Linear interpolation, so a profile sampled coarsely near a steep gradient is
    smoothed. Clamping means a grid cell outside the profile's own span takes the
    nearest profile value before the zeroing mask is applied, which matters only
    for a profile that does not reach the boundary.

    Provenance
    ----------
    .. [1] The sort, clamp and interpolate sequence reproduces the legacy MATLAB
       workflow's mapping, kept so that results match it cell for cell.
    """
    psiN_1d = np.asarray(psiN_1d, float)
    f_1d = np.asarray(f_1d, float)
    psi_RZ = np.asarray(psi_RZ, float)

    if psiN_1d.ndim != 1 or f_1d.ndim != 1:
        raise ValueError("psiN_1d and f_1d must be 1D arrays.")
    if psiN_1d.size != f_1d.size:
        raise ValueError("psiN_1d and f_1d must have the same length.")

    # Normalized flux on R,Z
    psiN_RZ = (psi_RZ - psi_axis) / (psi_lcfs - psi_axis)

    # MATLAB-style: sort + clip + interp
    idx = np.argsort(psiN_1d)
    x = psiN_1d[idx]
    y = f_1d[idx]

    psiN_clip = np.clip(psiN_RZ, x[0], x[-1])
    f_interp = np.interp(
        psiN_clip.ravel(), x, y
    ).reshape(psi_RZ.shape)

    if fill_outside == "edge":
        return f_interp, psiN_RZ
    if fill_outside != "zero":
        raise ValueError(f"fill_outside must be 'zero' or 'edge'; got {fill_outside!r}")
    # Outside LCFS → 0
    f_RZ = np.where((psiN_RZ >= 0.0) & (psiN_RZ <= 1.0), f_interp, 0.0)
    return f_RZ, psiN_RZ


def calculate_reconstructed_diamagnetic_flux(
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    psi_RZ: np.ndarray,
    psi_axis: float,
    psi_lcfs: float,
    psiN_1d: np.ndarray,
    f_1d: np.ndarray,
    f_vac_val: float,
    weights: np.ndarray | None = None,
) -> float:
    """Diamagnetic flux reconstructed from the equilibrium's own toroidal field.

    The surface integral of the difference between the plasma toroidal field and
    the vacuum field it displaces. Physics only: it reads no ODS and writes none.

    Parameters
    ----------
    R_grid : array_like
        Major-radius grid axis [m].
    Z_grid : array_like
        Height grid axis [m].
    psi_RZ : array_like
        Poloidal flux on the grid, indexed ``(R, Z)`` [Wb/rad].
    psi_axis : float
        Poloidal flux on the magnetic axis [Wb/rad].
    psi_lcfs : float
        Poloidal flux at the last closed flux surface [Wb/rad].
    psiN_1d : array_like
        Normalized poloidal flux the poloidal-current profile is given on [-].
    f_1d : array_like
        Poloidal current function ``F = R*B_phi`` on that abscissa [T m].
    f_vac_val : float
        The same function at the boundary, standing in for the vacuum field
        [T m].
    weights : array_like, optional
        Fraction of each cell inside the plasma, from
        :func:`plasma_cell_weights`; replaces the normalized-flux mask when
        given [-].

    Returns
    -------
    float
        The reconstructed diamagnetic flux, negative for a diamagnetic plasma
        [Wb].

    Processing steps
    ----------------
    1. Map the poloidal current profile onto the grid through
       :func:`psi_to_rz`.
    2. Form the plasma and vacuum toroidal fields as that function over the major
       radius.
    3. Integrate their difference over the cross-section inside the boundary.

    Convention
    ----------
    The sign follows the physics, not a storage choice: a diamagnetic plasma
    reduces the toroidal field inside it, so the integral is negative. Only the
    normalized flux enters, so the absolute unit of the three flux arguments
    cancels as long as they agree. The vacuum proxy must be taken at the boundary,
    not on axis, or the difference is offset everywhere.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A cross-sectional integral on the grid, so the boundary is resolved only to
    the cell. The vacuum field is taken as a single boundary value rather than a
    profile, which is exact only where the poloidal current is flat outside the
    plasma.

    Provenance
    ----------
    .. [1] The definition of diamagnetic flux as the toroidal-field deficit
       integrated over the plasma cross-section, in the form the EFIT-style
       workflow this package reproduces uses.
    """
    # With outline weights an edge cell can sit just past psiN = 1, where a
    # zeroed F would make F - F_vac read as -F_vac and swamp the integral.
    f_2d, psiN_RZ = psi_to_rz(
        psiN_1d, f_1d, psi_RZ, psi_axis, psi_lcfs,
        fill_outside="zero" if weights is None else "edge",
    )
    R_mesh, Z_mesh = np.meshgrid(R_grid, Z_grid, indexing="ij")
    mask_plasma = (psiN_RZ >= 0.0) & (psiN_RZ <= 1.0) & (R_mesh > 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        B_phi_plasma = f_2d / R_mesh
        B_phi_vacuum = f_vac_val / R_mesh

    diff_B = B_phi_plasma - B_phi_vacuum
    cell_fraction = mask_plasma.astype(float) if weights is None else np.asarray(weights, float)
    integrand = np.where(cell_fraction > 0.0, diff_B, 0.0)

    dR = np.gradient(R_grid)[:, None]
    dZ = np.gradient(Z_grid)[None, :]
    dA = np.abs(dR * dZ)

    return float(np.nansum(integrand * dA * cell_fraction))


def calculate_diamagnetism(
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    psi_RZ: np.ndarray,
    psi_axis: float,
    psi_lcfs: float,
    psiN_1d: np.ndarray,
    f_1d: np.ndarray,
    f_vac_val: float,
    B_pa: float,
    V_p: float | None = None,
    weights: np.ndarray | None = None,
) -> float:
    """Diamagnetism from its volume-integral definition.

    Parameters
    ----------
    R_grid : array_like
        Major-radius grid axis [m].
    Z_grid : array_like
        Height grid axis [m].
    psi_RZ : array_like
        Poloidal flux on the grid, indexed ``(R, Z)`` [Wb/rad].
    psi_axis : float
        Poloidal flux on the magnetic axis [Wb/rad].
    psi_lcfs : float
        Poloidal flux at the last closed flux surface [Wb/rad].
    psiN_1d : array_like
        Normalized poloidal flux the poloidal-current profile is given on [-].
    f_1d : array_like
        Poloidal current function ``F = R*B_phi`` on that abscissa [T m].
    f_vac_val : float
        The same function at the boundary, standing in for the vacuum field
        [T m].
    B_pa : float
        Boundary-average poloidal field, the normalizing scale, strictly positive
        [T].
    V_p : float, optional
        Plasma volume. Computed from the same grid and mask when not given
        [m^3].
    weights : array_like, optional
        Fraction of each cell inside the plasma, from
        :func:`plasma_cell_weights`; replaces the normalized-flux mask when
        given [-].

    Returns
    -------
    float
        The diamagnetism, positive for a diamagnetic plasma and negative for a
        paramagnetic one [-].

    Raises
    ------
    ValueError
        The boundary-average poloidal field is not positive.

    Convention
    ----------
    ``mu_i = (1/(B_pa^2 * V)) * integral (B_tv^2 - B_t^2) dV``, with the toroidal
    fields taken as the poloidal current function over the major radius and the
    axisymmetric volume element. Positive means the plasma reduced the toroidal
    field. A result of the unexpected sign usually means the vacuum proxy was
    taken on axis rather than at the boundary, or that the poloidal current
    profile's sign convention disagrees with the equilibrium it came from.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The same per-cell boundary mask as :func:`volume_average`, so the boundary is
    resolved only to the grid. The vacuum field is a single boundary value rather
    than a profile.

    Provenance
    ----------
    .. [1] The volume-integral definition of diamagnetism; the surface-integral
       form is :func:`calculate_reconstructed_diamagnetic_flux`, and the two
       should agree for a consistent equilibrium.
    """
    # With outline weights an edge cell can sit just past psiN = 1, where a
    # zeroed F would make F - F_vac read as -F_vac and swamp the integral.
    f_2d, psiN_RZ = psi_to_rz(
        psiN_1d, f_1d, psi_RZ, psi_axis, psi_lcfs,
        fill_outside="zero" if weights is None else "edge",
    )
    R_mesh, Z_mesh = np.meshgrid(R_grid, Z_grid, indexing="ij")
    mask_plasma = (psiN_RZ >= 0.0) & (psiN_RZ <= 1.0) & (R_mesh > 0.0)

    dR = np.gradient(R_grid)[:, None]
    dZ = np.gradient(Z_grid)[None, :]
    dA = np.abs(dR * dZ)
    dV = 2.0 * np.pi * R_mesh * dA

    # (B_tv² - B_t²) = (F_vac² - F²) / R²; integrand * dV = 2π (F_vac² - F²)/R * dA
    with np.errstate(divide="ignore", invalid="ignore"):
        diff_sq = (f_vac_val**2 - f_2d**2) / (R_mesh**2)
    cell_fraction = mask_plasma.astype(float) if weights is None else np.asarray(weights, float)
    dV = dV * cell_fraction
    integrand = np.where(cell_fraction > 0.0, diff_sq, 0.0)

    integral = float(np.nansum(integrand * dV))

    if V_p is not None and V_p > 0:
        Omega = V_p
    else:
        Omega = float(np.sum(dV[cell_fraction > 0.0]))
        if Omega <= 0.0:
            raise ValueError("Plasma volume is zero or negative.")

    if B_pa <= 0.0 or not np.isfinite(B_pa):
        raise ValueError("B_pa must be positive and finite.")

    return float(integral / (B_pa**2 * Omega))


def volume_average(
    f_RZ: np.ndarray,
    psiN_RZ: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    weights: np.ndarray | None = None,
    ):
    """Volume average of a gridded quantity over the confined region.

    Parameters
    ----------
    f_RZ : array_like
        The quantity on the grid, in whatever unit it has [any].
    psiN_RZ : array_like
        Normalized poloidal flux on the same grid, used only as the mask [-].
    R : array_like
        Major-radius axis, or the full mesh [m].
    Z : array_like
        Height axis, or the full mesh [m].
    weights : array_like, optional
        Fraction of each cell inside the plasma, from
        :func:`plasma_cell_weights`; replaces the normalized-flux mask when
        given [-].

    Returns
    -------
    favg : float
        The volume-weighted average of *f_RZ* inside the boundary [any].
    volume : float
        The volume the average was taken over [m^3].

    Convention
    ----------
    The volume element is the axisymmetric ``2*pi*R dR dZ``, so cells at large
    major radius weigh proportionally more; this is a torus average, not a
    cross-sectional one. Accepts either one-dimensional axes or a full mesh.

    **Pass** ``weights`` **whenever the slice has a boundary outline.**  Without
    them the plasma is every cell with normalized flux between zero and one and
    a positive major radius -- a flux threshold, not a containment test.  Outside
    the plasma psi is not monotonic and turns over by the coils, so the
    threshold admits exterior cells: on the packaged VEST samples the flux-mask
    volume is 1.7 to 18 times the plasma's, and a volume-averaged pressure
    comes out 40-50 % low.  With :func:`plasma_cell_weights` from the outline
    both match the contour-traced reference to 0.1 % and 1 %.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Cell areas come from a gradient of the axes, so the outermost cells are
    one-sided and a strongly non-uniform grid is approximated. The flux mask is a
    per-cell test with no sub-cell weighting; ``weights`` carry each boundary
    cell's area fraction instead.

    Provenance
    ----------
    .. [1] The axisymmetric volume element, and the same normalized-flux mask
       :func:`psi_to_rz` applies.
    """
    f_RZ = np.asarray(f_RZ, float)
    psiN_RZ = np.asarray(psiN_RZ, float)

    # Build mesh and cell area
    if R.ndim == 1 and Z.ndim == 1:
        Rm, Zm = np.meshgrid(R, Z, indexing="ij")
        dR = np.gradient(R)[:, None]
        dZ = np.gradient(Z)[None, :]
        dA = dR * dZ
    else:
        Rm, Zm = R, Z
        dA = np.abs(
            np.gradient(Rm, axis=0) * np.gradient(Zm, axis=1)
        )

    if weights is None:
        # Flux-threshold fallback; see the Convention section.
        cell_fraction = ((psiN_RZ >= 0.0) & (psiN_RZ <= 1.0) & (Rm > 0.0)).astype(float)
    else:
        cell_fraction = np.where(Rm > 0.0, np.asarray(weights, float), 0.0)
        if cell_fraction.shape != Rm.shape:
            raise ValueError(
                f"weights have shape {cell_fraction.shape}; the grid has {Rm.shape}"
            )

    dV = 2.0 * np.pi * Rm * dA * cell_fraction
    inside = cell_fraction > 0.0

    V = np.sum(dV[inside])
    if V == 0.0:
        raise ValueError("Total plasma volume is zero.")

    favg = np.sum(f_RZ[inside] * dV[inside]) / V
    return favg, V


def plasma_cell_weights(
    R: np.ndarray,
    Z: np.ndarray,
    psiN_RZ: np.ndarray,
    outline_r: np.ndarray | None = None,
    outline_z: np.ndarray | None = None,
) -> np.ndarray:
    """Fraction of each grid cell that is plasma, from the boundary outline when there is one.

    Parameters
    ----------
    R : array_like
        Major-radius axis, or the full mesh [m].
    Z : array_like
        Height axis, or the full mesh [m].
    psiN_RZ : array_like
        Normalized poloidal flux on the grid, used only when there is no
        outline [-].
    outline_r : array_like, optional
        Major radius of the last closed flux surface outline [m].
    outline_z : array_like, optional
        Height of the same outline [m].

    Returns
    -------
    np.ndarray
        Area fraction in ``[0, 1]`` on the grid's ``(R, Z)`` shape [-].

    Raises
    ------
    ValueError
        The outline coordinates differ in length [-].

    Convention
    ----------
    With at least three finite outline points this is
    :func:`fractional_cell_weights_from_boundary`: containment in the boundary,
    with each crossed cell's area share.  Otherwise it falls back to the
    normalized-flux threshold ``0 <= psiN <= 1`` as zeros and ones, and logs a
    warning, because that threshold also admits exterior cells wherever psi
    turns over near the coils (see :func:`volume_average`).

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The outline is trusted as given: an open or self-intersecting polygon gives
    whatever point-in-polygon makes of it.
    """
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)
    if R.ndim == 1 and Z.ndim == 1:
        Rm, Zm = np.meshgrid(R, Z, indexing="ij")
    else:
        Rm, Zm = R, Z
    if outline_r is not None and outline_z is not None:
        outline_r = np.asarray(outline_r, float).reshape(-1)
        outline_z = np.asarray(outline_z, float).reshape(-1)
        if outline_r.size != outline_z.size:
            raise ValueError("outline_r and outline_z differ in length")
        finite = np.isfinite(outline_r) & np.isfinite(outline_z)
        if finite.sum() >= 3:
            return fractional_cell_weights_from_boundary(
                Rm, Zm, outline_r[finite], outline_z[finite]
            )
    warnings.warn(
        "no usable boundary outline; taking the plasma as 0 <= psiN <= 1, which "
        "also admits cells outside the boundary",
        RuntimeWarning,
        stacklevel=2,
    )
    psiN = np.asarray(psiN_RZ, float)
    return ((psiN >= 0.0) & (psiN <= 1.0)).astype(float)

def psi_to_radial(
    psi_1d: np.ndarray,
    psi_2d_slice: np.ndarray,
    grid_r: np.ndarray,
    boundary_r: np.ndarray,
    r_axis: float,
    ):
    """Inboard and outboard major radius of each flux value, along the axis row.

    Parameters
    ----------
    psi_1d : array_like
        The poloidal flux profile to map [Wb/rad].
    psi_2d_slice : array_like
        Poloidal flux along the grid row at the magnetic axis height [Wb/rad].
    grid_r : array_like
        Major-radius grid points that row is on [m].
    boundary_r : array_like
        Major radius of the boundary, used for the inboard and outboard limits
        [m].
    r_axis : float
        Major radius of the magnetic axis, where the row is split [m].

    Returns
    -------
    r_inboard : np.ndarray
        Major radius on the high-field side for each flux value [m].
    r_outboard : np.ndarray
        Major radius on the low-field side [m].

    Processing steps
    ----------------
    1. Split the axis row at the magnetic axis into an inboard and an outboard
       branch, on each of which the flux is monotonic.
    2. Build an interpolation of major radius against flux on each branch.
    3. Evaluate both at every value of the profile, clipped to the boundary.

    Convention
    ----------
    Two answers per flux value, because a flux surface crosses the axis height
    twice. The split is at the magnetic axis, which is where the flux turns, so
    each branch is single-valued. Everything is in the flux unit the caller
    supplied; nothing is rescaled.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Only the row nearest the axis height is used, so this describes the midplane
    and not the surface as a whole. A flux value outside the row's own span is
    clipped to the boundary rather than extrapolated. Assumes the flux is
    monotonic on each branch, which fails for a non-nested equilibrium.

    Provenance
    ----------
    .. [1] The inboard and outboard midplane radii as the IMAS data dictionary
       defines them for a one-dimensional profile.
    """
    psi_1d = np.asarray(psi_1d, float)
    psi_2d_slice = np.asarray(psi_2d_slice, float)
    grid_r = np.asarray(grid_r, float)
    boundary_r = np.asarray(boundary_r, float)
    
    # Determine boundary limits
    r_min, r_max = np.min(boundary_r), np.max(boundary_r)
    
    # Split into inboard/outboard regions
    mask_in = (grid_r >= r_min) & (grid_r <= r_axis)
    mask_out = (grid_r >= r_axis) & (grid_r <= r_max)
    psi_in, r_in = psi_2d_slice[mask_in], grid_r[mask_in]
    psi_out, r_out = psi_2d_slice[mask_out], grid_r[mask_out]
    
    # Create interpolation functions
    # Inboard: reverse order for monotonic psi (decreasing from boundary to axis)
    f_in = interp1d(psi_in[::-1], r_in[::-1], 
                   kind='cubic', fill_value='extrapolate')
    f_out = interp1d(psi_out, r_out, 
                    kind='cubic', fill_value='extrapolate')
    
    # Map 1D psi profile to radial coordinates
    r_inboard = f_in(psi_1d)
    r_outboard = f_out(psi_1d)
    
    return r_inboard, r_outboard




# ------------------------------------------------------------------
# Shafranov Integral
# ------------------------------------------------------------------

def poloidal_field_at_boundary(
    R_grid_1d, Z_grid_1d, psi_grid, R_bdry, Z_bdry, cocos=None, psi_per_radian=None,
):
    """Poloidal field on a boundary contour, from the flux map by bicubic spline.

    Parameters
    ----------
    R_grid_1d : array_like
        Major-radius grid axis of the flux map [m].
    Z_grid_1d : array_like
        Height grid axis of the flux map [m].
    psi_grid : array_like
        Poloidal flux on the grid, indexed ``(R, Z)`` [Wb/rad].
    R_bdry : array_like
        Major radius of the contour points [m].
    Z_bdry : array_like
        Height of the contour points [m].
    cocos : int, optional
        COCOS index of *psi_grid*. ``None`` keeps the historical
        weber-per-radian form [-].
    psi_per_radian : bool, optional
        Whether *psi_grid* is per radian, when the index alone does not settle it
        [-].

    Returns
    -------
    B_p_bdry : np.ndarray
        Poloidal field magnitude at each contour point [T].
    B_R_bdry : np.ndarray
        Its major-radius component [T].
    B_Z_bdry : np.ndarray
        Its height component [T].

    Convention
    ----------
    ``B_R = k * (1/R) * dpsi/dZ`` and ``B_Z = -k * (1/R) * dpsi/dR``, where
    ``k = sigma_RphiZ * sigma_Bp / (2*pi)**e_Bp`` carries both the orientation
    sign and the ``2*pi`` normalization together, per Sauter Eq. 20.

    With *cocos* unset the historical EFIT weber-per-radian form ``k = -1`` is
    used. Passing *psi_per_radian* alone leaves the sign untouched and applies
    only the ``2*pi``, which is what a weber-stored flux needs. The flux map is
    indexed major radius first.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A bicubic spline over the whole map, so contour points must lie inside the
    grid and the field near a steep edge is only as good as the map's resolution.
    The convention must be supplied; a weber flux passed without either argument
    gives a field too large by ``2*pi``, and a poloidal beta too small by its
    square.

    Provenance
    ----------
    .. [1] Sauter and Medvedev (2013), Eq. 20, for the prefactor that carries the
       orientation sign and the ``2*pi`` together; evaluated by
       :func:`vaft.formula.equilibrium.poloidal_field_factor`.
    .. [2] Plays the role of EFIT's ``seva2d`` subroutine, from which the
       bicubic-spline-on-the-flux-map approach is taken.
    """
    
    # 1. 2차원 스플라인 객체 생성 (Bicubic Spline)
    # RectBivariateSpline은 격자가 균일하지 않아도 되지만, 정렬되어 있어야 합니다.
    # psi_grid의 축 순서는 (x=R, y=Z)를 가정합니다.
    interp_spline = RectBivariateSpline(R_grid_1d, Z_grid_1d, psi_grid)

    # 2. 경계면 좌표에서의 편미분 계산 (Grid -> Boundary Interpolation)
    # ev(x, y, dx, dy) 메서드는 해당 좌표에서의 미분값을 반환합니다.
    # dPsi/dR
    dPsi_dR = interp_spline.ev(R_bdry, Z_bdry, dx=1, dy=0)
    # dPsi/dZ
    dPsi_dZ = interp_spline.ev(R_bdry, Z_bdry, dx=0, dy=1)

    # 3. 자기장 계산 (Cylindrical Coordinates), Sauter Eq. 20
    #    k = sigma_RphiZ * sigma_Bp / (2*pi)**e_Bp 는 2*pi 정규화와 방향 부호를
    #    함께 담습니다. cocos=None 이면 기존 EFIT Weber/rad 관례(k = -1)를
    #    그대로 사용하되, psi_per_radian=False 로 저장 계열만 알려주면 부호는
    #    그대로 두고 2*pi 정규화만 적용합니다 (Wb 저장 psi 에 필요).
    from vaft.formula.equilibrium import poloidal_field_factor

    k = poloidal_field_factor(cocos, psi_per_radian=psi_per_radian)

    # B_R = k * (1/R) * dPsi/dZ
    B_R_bdry = k * (1.0 / R_bdry) * dPsi_dZ
    
    # B_Z = -k * (1/R) * dPsi/dR
    B_Z_bdry = -k * (1.0 / R_bdry) * dPsi_dR
    
    # 4. Poloidal Field 크기 계산
    B_p_bdry = np.sqrt(B_R_bdry**2 + B_Z_bdry**2)

    return B_p_bdry, B_R_bdry, B_Z_bdry



def calculate_average_boundary_poloidal_field(R_bdry, Z_bdry, B_p_bdry):
    """Contour-length average of the poloidal field around a closed boundary.

    Parameters
    ----------
    R_bdry : array_like
        Major radius of the contour points [m].
    Z_bdry : array_like
        Height of the contour points [m].
    B_p_bdry : array_like
        Poloidal field magnitude at those points [T].

    Returns
    -------
    float
        The average poloidal field, the contour integral of the field over the
        contour length [T].

    Convention
    ----------
    Weighted by arc length, not by point count, so an unevenly sampled contour
    still gives the physical average. The midpoint rule is used between
    consecutive points, and the contour is treated as closed. This is the
    normalizing field that the poloidal beta and the diamagnetism are taken
    against, so it must come from the same convention as the field that produced
    it.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Midpoint rule on the given points, so a coarsely sampled contour underweights
    regions of high curvature. Takes the field magnitudes as given and cannot
    check they were computed in a consistent convention.

    Provenance
    ----------
    .. [1] The definition used with :func:`shafranov_integrals` and
       :func:`calculate_diamagnetism` as their reference field.
    """
    # 1. 배열이 닫혀있는지 확인 (마지막 점 != 첫 점이면 닫아줌)
    if (R_bdry[0] != R_bdry[-1]) or (Z_bdry[0] != Z_bdry[-1]):
        R_bdry = np.append(R_bdry, R_bdry[0])
        Z_bdry = np.append(Z_bdry, Z_bdry[0])
        B_p_bdry = np.append(B_p_bdry, B_p_bdry[0])

    # 2. 미소 길이 성분 계산 (dl)
    dR = np.diff(R_bdry)
    dZ = np.diff(Z_bdry)
    dl = np.sqrt(dR**2 + dZ**2)
    
    # 3. 적분 구간의 대푯값 (Midpoint rule or Trapezoidal)
    B_p_mid = 0.5 * (B_p_bdry[:-1] + B_p_bdry[1:])
    
    # 4. 선적분 수행
    L_total = np.sum(dl)             # ∮ dl
    integral_Bp = np.sum(B_p_mid * dl) # ∮ B_p dl
    
    B_pa = integral_Bp / L_total
    
    return B_pa

def _ensure_closed_boundary(
    R_bdry: np.ndarray,
    Z_bdry: np.ndarray,
    *extras: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Return boundary arrays with the first point appended to the end."""
    R_bdry = np.asarray(R_bdry, dtype=float).copy()
    Z_bdry = np.asarray(Z_bdry, dtype=float).copy()
    out = [R_bdry, Z_bdry]
    out.extend(np.asarray(x, dtype=float).copy() for x in extras)
    if out[0].size == 0:
        return tuple(out)
    if (out[0][0] != out[0][-1]) or (out[1][0] != out[1][-1]):
        out = [np.append(arr, arr[0]) for arr in out]
    return tuple(out)


def _signed_area_closed_polygon(R_bdry: np.ndarray, Z_bdry: np.ndarray) -> float:
    """Return signed area (shoelace) for a closed boundary."""
    if R_bdry.size < 2:
        return 0.0
    return 0.5 * float(
        np.sum(R_bdry[:-1] * Z_bdry[1:] - R_bdry[1:] * Z_bdry[:-1])
    )


def _remove_degenerate_segments(
    R_bdry: np.ndarray,
    Z_bdry: np.ndarray,
    *extras: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, ...]:
    """
    Remove consecutive duplicated points / zero-length segments from a closed boundary.
    """
    if R_bdry.size == 0:
        out = [R_bdry, Z_bdry]
        out.extend(extras)
        return tuple(out)

    keep_idx = [0]
    for i in range(1, R_bdry.size):
        j = keep_idx[-1]
        if np.hypot(R_bdry[i] - R_bdry[j], Z_bdry[i] - Z_bdry[j]) > eps:
            keep_idx.append(i)

    R_new = R_bdry[keep_idx]
    Z_new = Z_bdry[keep_idx]
    extras_new = [arr[keep_idx] for arr in extras]

    R_new, Z_new, *extras_new = _ensure_closed_boundary(R_new, Z_new, *extras_new)
    return (R_new, Z_new, *extras_new)


def _resample_closed_boundary_arrays(
    R_bdry: np.ndarray,
    Z_bdry: np.ndarray,
    *extras: np.ndarray,
    n_points: int = 256,
) -> tuple[np.ndarray, ...]:
    """
    Arc-length resample a closed boundary (and co-located extras) to n_points segments.
    """
    if n_points < 4:
        raise ValueError("n_points must be >= 4 for closed boundary resampling.")

    if R_bdry.size < 2:
        out = [R_bdry, Z_bdry]
        out.extend(extras)
        return tuple(out)

    R_loop = R_bdry[:-1]
    Z_loop = Z_bdry[:-1]
    extras_loop = [arr[:-1] for arr in extras]
    if R_loop.size < 3:
        out = [R_bdry, Z_bdry]
        out.extend(extras)
        return tuple(out)

    R_periodic = np.append(R_loop, R_loop[0])
    Z_periodic = np.append(Z_loop, Z_loop[0])
    extras_periodic = [np.append(arr, arr[0]) for arr in extras_loop]

    dl = np.hypot(np.diff(R_periodic), np.diff(Z_periodic))
    s = np.concatenate(([0.0], np.cumsum(dl)))

    s_unique, idx_unique = np.unique(s, return_index=True)
    if s_unique.size < 2 or s_unique[-1] <= 0.0:
        out = [R_bdry, Z_bdry]
        out.extend(extras)
        return tuple(out)

    R_unique = R_periodic[idx_unique]
    Z_unique = Z_periodic[idx_unique]
    extras_unique = [arr[idx_unique] for arr in extras_periodic]

    s_target = np.linspace(0.0, s_unique[-1], n_points + 1)
    R_target = np.interp(s_target, s_unique, R_unique)
    Z_target = np.interp(s_target, s_unique, Z_unique)
    extras_target = [np.interp(s_target, s_unique, arr) for arr in extras_unique]

    R_target[-1] = R_target[0]
    Z_target[-1] = Z_target[0]
    for arr in extras_target:
        arr[-1] = arr[0]

    return (R_target, Z_target, *extras_target)


def prepare_boundary_for_shafranov(
    R_bdry: np.ndarray,
    Z_bdry: np.ndarray,
    n_points: int = 256,
    enforce_ccw: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Condition a boundary contour into the form the Shafranov integrals require.

    Parameters
    ----------
    R_bdry : array_like
        Major radius of the contour points [m].
    Z_bdry : array_like
        Height of the contour points [m].
    n_points : int, optional
        How many points to resample the contour to [-].
    enforce_ccw : bool, optional
        Whether to reverse the contour when it runs clockwise [-].
    eps : float, optional
        Distance below which two consecutive points count as duplicates [m].

    Returns
    -------
    tuple of np.ndarray
        The conditioned ``(R, Z)`` contour: finite, closed, without duplicate
        points, counter-clockwise, and evenly spaced in arc length [m].

    Processing steps
    ----------------
    1. Drop non-finite points.
    2. Close the contour if it is not already closed.
    3. Remove consecutive duplicates within *eps*.
    4. Reverse the contour if its signed area says it runs clockwise.
    5. Resample uniformly in arc length to *n_points*.

    Defaults
    --------
    ``n_points = 256`` and ``eps = 1e-12`` are numerical conveniences: enough
    points to resolve a tokamak boundary's curvature, and a duplicate threshold
    far below any real machine dimension.

    Convention
    ----------
    **Counter-clockwise is not cosmetic.** :func:`shafranov_integrals` takes the
    outward unit normal as ``(dZ/dl, -dR/dl)``, which points outward only for a
    counter-clockwise contour; a clockwise one silently flips the sign of every
    integral that uses it. That is why the orientation is enforced here rather
    than assumed there.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Assumes a single simple closed curve. A self-intersecting contour, or two
    disjoint pieces, gives a signed area that does not mean what the orientation
    test takes it to mean.

    Provenance
    ----------
    .. [1] The outward-normal convention of :func:`shafranov_integrals`, which is
       what this exists to guarantee.
    """
    R_bdry = np.asarray(R_bdry, dtype=float).reshape(-1)
    Z_bdry = np.asarray(Z_bdry, dtype=float).reshape(-1)
    if R_bdry.size != Z_bdry.size:
        raise ValueError("R_bdry and Z_bdry must have the same length.")

    finite = np.isfinite(R_bdry) & np.isfinite(Z_bdry)
    R_bdry = R_bdry[finite]
    Z_bdry = Z_bdry[finite]
    if R_bdry.size < 3:
        return np.asarray([], float), np.asarray([], float)

    R_bdry, Z_bdry = _ensure_closed_boundary(R_bdry, Z_bdry)
    R_bdry, Z_bdry = _remove_degenerate_segments(R_bdry, Z_bdry, eps=eps)
    if R_bdry.size < 4:
        return np.asarray([], float), np.asarray([], float)

    if enforce_ccw and _signed_area_closed_polygon(R_bdry, Z_bdry) < 0.0:
        R_bdry = R_bdry[::-1]
        Z_bdry = Z_bdry[::-1]

    R_bdry, Z_bdry = _resample_closed_boundary_arrays(
        R_bdry, Z_bdry, n_points=n_points
    )
    return R_bdry, Z_bdry


def _prepare_boundary_and_field_for_shafranov(
    R_bdry: np.ndarray,
    Z_bdry: np.ndarray,
    B_p_bdry: np.ndarray,
    n_points: int = 256,
    enforce_ccw: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize boundary and boundary field together to keep (R,Z,Bp) aligned.
    """
    R_bdry = np.asarray(R_bdry, dtype=float).reshape(-1)
    Z_bdry = np.asarray(Z_bdry, dtype=float).reshape(-1)
    B_p_bdry = np.asarray(B_p_bdry, dtype=float).reshape(-1)
    if R_bdry.size != Z_bdry.size or R_bdry.size != B_p_bdry.size:
        raise ValueError("R_bdry, Z_bdry, and B_p_bdry must have the same length.")

    finite = np.isfinite(R_bdry) & np.isfinite(Z_bdry) & np.isfinite(B_p_bdry)
    R_bdry = R_bdry[finite]
    Z_bdry = Z_bdry[finite]
    B_p_bdry = B_p_bdry[finite]
    if R_bdry.size < 3:
        return np.asarray([], float), np.asarray([], float), np.asarray([], float)

    R_bdry, Z_bdry, B_p_bdry = _ensure_closed_boundary(R_bdry, Z_bdry, B_p_bdry)
    R_bdry, Z_bdry, B_p_bdry = _remove_degenerate_segments(
        R_bdry, Z_bdry, B_p_bdry, eps=eps
    )
    if R_bdry.size < 4:
        return np.asarray([], float), np.asarray([], float), np.asarray([], float)

    if enforce_ccw and _signed_area_closed_polygon(R_bdry, Z_bdry) < 0.0:
        R_bdry = R_bdry[::-1]
        Z_bdry = Z_bdry[::-1]
        B_p_bdry = B_p_bdry[::-1]

    R_bdry, Z_bdry, B_p_bdry = _resample_closed_boundary_arrays(
        R_bdry, Z_bdry, B_p_bdry, n_points=n_points
    )
    return R_bdry, Z_bdry, B_p_bdry


def _cell_area_from_mesh(R_grid: np.ndarray, Z_grid: np.ndarray) -> np.ndarray:
    """Return per-cell dA on an (R,Z) mesh."""
    dR = np.gradient(R_grid, axis=0)
    dZ = np.gradient(Z_grid, axis=1)
    return np.abs(dR * dZ)


def _plasma_cell_weights(
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    R_bdry_closed: np.ndarray,
    Z_bdry_closed: np.ndarray,
    cell_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build 2D plasma cell weights.

    If `cell_weights` is provided, it is used directly (EFIT `www` equivalent).
    Otherwise this returns a 0/1 mask from point-in-polygon.
    """
    if cell_weights is not None:
        w = np.asarray(cell_weights, dtype=float)
        if w.shape != R_grid.shape:
            raise ValueError(
                f"cell_weights shape {w.shape} must match grid shape {R_grid.shape}."
            )
        return np.where(np.isfinite(w), w, 0.0)

    import importlib

    poly_verts = np.column_stack((R_bdry_closed, Z_bdry_closed))
    mpl_path = importlib.import_module("matplotlib.path")
    path = mpl_path.Path(poly_verts)
    points = np.column_stack((R_grid.ravel(), Z_grid.ravel()))
    inside = path.contains_points(points, radius=1e-14).reshape(R_grid.shape)
    return inside.astype(float)


def fractional_cell_weights_from_boundary(
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    R_bdry: np.ndarray,
    Z_bdry: np.ndarray,
    samples_per_axis: int = 5,
) -> np.ndarray:
    """Fractional area of each grid cell that lies inside the plasma boundary.

    An internal replacement for the externally supplied weight map EFIT expects,
    computed from the boundary polygon so that no external file is needed.

    Parameters
    ----------
    R_grid : array_like
        Major-radius grid, as an axis or a mesh [m].
    Z_grid : array_like
        Height grid, as an axis or a mesh [m].
    R_bdry : array_like
        Major radius of the boundary contour [m].
    Z_bdry : array_like
        Height of the boundary contour [m].
    samples_per_axis : int, optional
        Sub-samples per axis within each cell, at least one [-].

    Returns
    -------
    np.ndarray
        Area fraction in ``[0, 1]`` for every cell, on the grid's own shape [-].

    Raises
    ------
    ValueError
        *samples_per_axis* is less than one.

    Defaults
    --------
    ``samples_per_axis = 5`` is a numerical convenience: twenty-five sub-samples
    per cell resolve the boundary to about a fifth of a cell, which is finer than
    the flux map that produced the boundary in the first place.

    Convention
    ----------
    A fraction, not a mask: a cell the boundary crosses gets its area share rather
    than a zero or a one. That is what makes an integral weighted by these
    converge with grid resolution instead of stepping.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Sub-sampling, not exact polygon clipping, so a cell is resolved only to the
    sub-sample spacing and a boundary feature smaller than a cell is missed.
    Assumes a single closed polygon.

    Provenance
    ----------
    .. [1] Replaces EFIT's externally provided cell-weight map; consumed by
       :func:`shafranov_integrals` and :func:`efit_virial_volume_integrals`.
    """
    if samples_per_axis < 1:
        raise ValueError("samples_per_axis must be >= 1.")

    if np.ndim(R_grid) == 1 and np.ndim(Z_grid) == 1:
        R_grid, Z_grid = np.meshgrid(
            np.asarray(R_grid, float),
            np.asarray(Z_grid, float),
            indexing="ij",
        )
    else:
        R_grid = np.asarray(R_grid, float)
        Z_grid = np.asarray(Z_grid, float)
    R_bdry, Z_bdry = _ensure_closed_boundary(R_bdry, Z_bdry)

    import importlib

    poly_verts = np.column_stack((R_bdry, Z_bdry))
    mpl_path = importlib.import_module("matplotlib.path")
    path = mpl_path.Path(poly_verts)

    dR = np.abs(np.gradient(R_grid, axis=0))
    dZ = np.abs(np.gradient(Z_grid, axis=1))

    # Midpoint sub-sampling on each local control cell.
    offsets = (np.arange(samples_per_axis, dtype=float) + 0.5) / samples_per_axis - 0.5
    inside_acc = np.zeros(R_grid.shape, dtype=float)
    for oR in offsets:
        for oZ in offsets:
            sample_R = R_grid + oR * dR
            sample_Z = Z_grid + oZ * dZ
            points = np.column_stack((sample_R.ravel(), sample_Z.ravel()))
            inside = path.contains_points(points, radius=1e-14).reshape(R_grid.shape)
            inside_acc += inside.astype(float)

    return inside_acc / float(samples_per_axis * samples_per_axis)


#: An annulus resolving fewer than this many cells describes the grid rather
#: than the field.  Chosen to match :data:`MIN_FLUX_SURFACE_POINTS`, the same
#: judgement applied to a contour instead of a region.
MIN_ANNULUS_CELLS = 16.0

#: An annulus must be at least this many *sub-samples* wide at its narrowest.
#: The weighting in :func:`fractional_cell_weights_from_boundary` resolves down
#: to one cell over ``samples_per_axis``, not one cell -- a sub-cell annulus is
#: still integrated correctly -- so the limit is the sub-sample spacing. Below
#: it the band falls between sample points and the integral reports the grid
#: phase rather than the field.
MIN_ANNULUS_WIDTH_SUBSAMPLES = 1.0

#: Fraction of the boundary arc length that may have a negative support
#: function before the conformal construction is judged not to describe the
#: shape at all.  A star-shaped boundary has none.
MAX_NONCONVEX_ARC_FRACTION = 0.01


def _boundary_normals_and_support(
    R_b: np.ndarray,
    Z_b: np.ndarray,
    center: tuple[float, float] | None = None,
) -> tuple[np.ndarray, ...]:
    """Segment midpoints, outward normals, lengths and support function.

    ``R_b``/``Z_b`` must already be closed and counter-clockwise, so that
    ``n = (dZ, -dR)/dl`` points outward.  The support function
    ``(x - c).n`` is the local width per unit conformal scaling: shrinking the
    contour by a factor ``1 - t`` about ``c`` moves each point by ``-t(x - c)``,
    whose component along the normal is ``t (x - c).n``.

    It is negative wherever the boundary faces away from ``c`` -- which cannot
    happen for a contour star-shaped about ``c``, and means the conformal
    annulus folds over itself where it does.
    """
    dR, dZ = np.diff(R_b), np.diff(Z_b)
    dl = np.hypot(dR, dZ)
    with np.errstate(divide="ignore", invalid="ignore"):
        nR = np.where(dl > 0.0, dZ / dl, 0.0)
        nZ = np.where(dl > 0.0, -dR / dl, 0.0)
    R_mid = 0.5 * (R_b[:-1] + R_b[1:])
    Z_mid = 0.5 * (Z_b[:-1] + Z_b[1:])
    if center is None:
        R_c = 0.5 * (float(np.min(R_b)) + float(np.max(R_b)))
        Z_c = 0.5 * (float(np.min(Z_b)) + float(np.max(Z_b)))
    else:
        R_c, Z_c = float(center[0]), float(center[1])
    support = (R_mid - R_c) * nR + (Z_mid - Z_c) * nZ
    return R_mid, Z_mid, nR, nZ, dl, support


def scale_boundary_conformal(
    R_bdry: np.ndarray,
    Z_bdry: np.ndarray,
    scale: float,
    center: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale a closed boundary about a centre, giving a contour conformal to it.

    ``(R, Z) -> center + scale * ((R, Z) - center)``, so the result has the same
    shape as the input and lies inside it for ``scale < 1``.

    Parameters
    ----------
    R_bdry : array_like
        Major radius of the boundary points [m].
    Z_bdry : array_like
        Height of the boundary points [m].
    scale : float
        Similarity factor; below 1 shrinks, above 1 grows [-].
    center : tuple of float, optional
        Centre to scale about. The boundary's bounding-box centre when not
        given [m].

    Returns
    -------
    tuple of numpy.ndarray
        The scaled ``(R, Z)``, closed, in the input's point order [m].

    Raises
    ------
    ValueError
        ``R_bdry`` and ``Z_bdry`` differ in length, or ``scale`` is not finite
        and positive.

    Convention
    ----------
    The default centre is the bounding-box centre rather than the area
    centroid, matching what the shape code reports as the geometric axis.

    Applicability
    -------------
    Machine-independent. Any closed contour; nothing here is specific to a
    boundary or to an equilibrium.

    Limitations
    -----------
    A purely geometric offset. It does not consult the flux map, which is the
    point when the contour is a reconstructed LCFS -- a psi_N band would carry
    the interior psi's error into the result, and this does not. The scaled
    contour lies strictly inside the original only for a boundary star-shaped
    about ``center``; otherwise it can fold over itself, and the caller has to
    decide what that means.

    Provenance
    ----------
    .. [1] The conformal-annulus construction of M. W. Bongard et al., Phys.
       Plasmas 23 (2016), as used by
       :func:`virial_alpha_conformal_annulus`.
    """
    R_bdry = np.asarray(R_bdry, dtype=float).reshape(-1)
    Z_bdry = np.asarray(Z_bdry, dtype=float).reshape(-1)
    if R_bdry.size != Z_bdry.size:
        raise ValueError("R_bdry and Z_bdry must have the same length.")
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and positive; got {scale}.")

    R_bdry, Z_bdry = _ensure_closed_boundary(R_bdry, Z_bdry)
    if center is None:
        R_c = 0.5 * (float(np.min(R_bdry)) + float(np.max(R_bdry)))
        Z_c = 0.5 * (float(np.min(Z_bdry)) + float(np.max(Z_bdry)))
    else:
        R_c, Z_c = float(center[0]), float(center[1])

    return R_c + scale * (R_bdry - R_c), Z_c + scale * (Z_bdry - Z_c)


def virial_alpha_thin_annulus(
    R_bdry: np.ndarray,
    Z_bdry: np.ndarray,
    B_R_bdry: np.ndarray,
    B_Z_bdry: np.ndarray,
    center: tuple[float, float] | None = None,
    mode: str = "conformal",
    n_points: int = 512,
) -> float:
    """Virial closure coefficient in the thin-annulus limit, from the boundary field.

    The exact ``thickness -> 0`` limit of :func:`virial_alpha_conformal_annulus`,
    evaluating ``2 * closed_integral(R Bz^2 w dl) / closed_integral(R Bp^2 w dl)``.
    Needs only the field on the LCFS, so unlike the volume integral it asks
    nothing of the plasma interior.

    Processing steps
    ----------------
    Drop non-finite points, close the contour, remove zero-length segments,
    orient counter-clockwise so the normal ``(dZ, -dR)/dl`` points outward,
    resample to ``n_points`` by arc length, then take the segment-midpoint
    quadrature through
    :func:`vaft.formula.equilibrium.virial_alpha_from_R_Bz_Bp_dl`.

    Parameters
    ----------
    R_bdry : array_like
        Major radius of the boundary points [m].
    Z_bdry : array_like
        Height of the boundary points [m].
    B_R_bdry : array_like
        Major-radius field component at those points [T].
    B_Z_bdry : array_like
        Vertical field component at those points [T].
    center : tuple of float, optional
        Centre the conformal annulus shrinks towards. The boundary's
        bounding-box centre when not given [m].
    mode : str, optional
        ``"conformal"`` (default) weights each segment by the support function,
        the limit of a conformally scaled annulus; ``"uniform"`` weights every
        segment equally, the limit of a uniform-offset annulus [-].
    n_points : int, optional
        Segments in the arc-length resampling; default 512 [-].

    Returns
    -------
    float
        The closure coefficient, or NaN when the contour is degenerate or folds
        over the centre [-].

    Raises
    ------
    ValueError
        ``mode`` is neither ``"conformal"`` nor ``"uniform"``, or the boundary
        and field arrays differ in length.

    Convention
    ----------
    ``alpha = 2 <R Bz^2> / <R Bp^2>``, which is 1 for a symmetric circular
    cross-section and 2 in the infinitely elongated limit. Only squares of the
    field enter, so the result does not depend on the sign convention the
    components arrive in.

    Assumptions
    -----------
    ``mode="conformal"`` assumes the boundary is star-shaped about ``center``,
    so that shrinking it about that point sweeps a simple annulus. A conformal
    annulus has local width ``t * (x - c).n``, so its area element is
    ``t * (x - c).n * dl`` and the constant ``t`` cancels; that support function
    is the weight. Weighting uniformly instead is a *different* region and on a
    VEST-like boundary a ~5% different answer, so the two modes are not
    interchangeable.

    Applicability
    -------------
    Machine-independent. Any closed boundary with its poloidal field, from a
    reconstruction or an analytic equilibrium.

    Limitations
    -----------
    Returns NaN rather than a number when more than
    :data:`MAX_NONCONVEX_ARC_FRACTION` of the arc length has negative support:
    the conformal annulus folds over itself there, and an unclamped negative
    weight would subtract from both integrals. ``mode="uniform"`` makes no such
    assumption and still answers. How closely the limit matches a finite
    annulus is a property of the equilibrium, not of this quadrature.

    Provenance
    ----------
    .. [1] M. W. Bongard et al., Phys. Plasmas 23 (2016), the low-aspect-ratio
       virial closure and its annulus estimate of alpha.
    .. [2] Measured against analytic Solov'ev equilibria in
       ``test/test_virial_alpha_approximations.py``.
    """
    if mode not in ("conformal", "uniform"):
        raise ValueError(f"mode must be 'conformal' or 'uniform'; got {mode!r}.")

    R_b = np.asarray(R_bdry, float).reshape(-1)
    Z_b = np.asarray(Z_bdry, float).reshape(-1)
    B_R_b = np.asarray(B_R_bdry, float).reshape(-1)
    B_Z_b = np.asarray(B_Z_bdry, float).reshape(-1)
    if not (R_b.size == Z_b.size == B_R_b.size == B_Z_b.size):
        raise ValueError("boundary and field arrays must have the same length.")

    finite = np.isfinite(R_b) & np.isfinite(Z_b) & np.isfinite(B_R_b) & np.isfinite(B_Z_b)
    R_b, Z_b, B_R_b, B_Z_b = R_b[finite], Z_b[finite], B_R_b[finite], B_Z_b[finite]
    if R_b.size < 3:
        return float("nan")

    R_b, Z_b, B_R_b, B_Z_b = _ensure_closed_boundary(R_b, Z_b, B_R_b, B_Z_b)
    R_b, Z_b, B_R_b, B_Z_b = _remove_degenerate_segments(R_b, Z_b, B_R_b, B_Z_b)
    if R_b.size < 4:
        return float("nan")
    if _signed_area_closed_polygon(R_b, Z_b) < 0.0:  # enforce CCW for the normal
        R_b, Z_b, B_R_b, B_Z_b = R_b[::-1], Z_b[::-1], B_R_b[::-1], B_Z_b[::-1]
    R_b, Z_b, B_R_b, B_Z_b = _resample_closed_boundary_arrays(
        R_b, Z_b, B_R_b, B_Z_b, n_points=n_points
    )

    R_mid, Z_mid, _nR, _nZ, dl, support = _boundary_normals_and_support(
        R_b, Z_b, center
    )
    B_R_mid = 0.5 * (B_R_b[:-1] + B_R_b[1:])
    B_Z_mid = 0.5 * (B_Z_b[:-1] + B_Z_b[1:])
    B_p_mid = np.hypot(B_R_mid, B_Z_mid)

    if mode == "uniform":
        weight = np.ones_like(dl)
    else:
        # A negative support means the conformal annulus folds over itself
        # there, and an unclamped negative weight would subtract from both
        # integrals -- so a mildly non-convex contour is clamped, and one that
        # is genuinely not star-shaped about the centre abstains rather than
        # returning an alpha the finite-thickness path would not agree with.
        total = float(np.sum(dl))
        folded = float(np.sum(dl[support < 0.0]))
        if total <= 0.0 or folded > MAX_NONCONVEX_ARC_FRACTION * total:
            return float("nan")
        weight = np.clip(support, 0.0, None)

    from vaft.formula.equilibrium import virial_alpha_from_R_Bz_Bp_dl

    return virial_alpha_from_R_Bz_Bp_dl(R_mid, B_Z_mid, B_p_mid, dl, weight=weight)


def virial_alpha_conformal_annulus(
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    B_R_grid: np.ndarray,
    B_Z_grid: np.ndarray,
    R_bdry: np.ndarray,
    Z_bdry: np.ndarray,
    thickness: float = 0.1,
    samples_per_axis: int = 5,
) -> dict[str, Any]:
    """Virial closure coefficient from a thin annulus conformal to the LCFS.

    ``alpha_2 = 2 * sum(R Bz^2 w dA) / sum(R Bp^2 w dA)`` over an annulus bounded
    by the boundary and a copy of it scaled by ``1 - thickness``. Obtains the
    closure coefficient without the volume integral over the plasma interior,
    which a reconstruction that fixes the boundary but not the internal field
    cannot supply.

    Processing steps
    ----------------
    Orient the boundary counter-clockwise, build the inner contour with
    :func:`scale_boundary_conformal`, take the annulus weight map as the
    difference of two :func:`fractional_cell_weights_from_boundary` maps so that
    cells cut by either contour carry their area fraction, reject an annulus too
    small or too narrow for the grid, then integrate over the cells that survive
    a shared finite-field mask.

    Parameters
    ----------
    R_grid : array_like
        Major-radius grid, as an axis or a mesh [m].
    Z_grid : array_like
        Height grid, as an axis or a mesh [m].
    B_R_grid : array_like
        Major-radius field component on that grid [T].
    B_Z_grid : array_like
        Vertical field component on that grid [T].
    R_bdry : array_like
        Major radius of the boundary points [m].
    Z_bdry : array_like
        Height of the boundary points [m].
    thickness : float, optional
        Annulus width as a fraction of the distance from the centre to the
        boundary, not an absolute length; default 0.1 [-].
    samples_per_axis : int, optional
        Sub-samples per cell axis in the area-fraction weighting; default 5 [-].

    Returns
    -------
    dict
        ``alpha`` the closure coefficient, NaN unless ``valid``; ``valid``
        whether the annulus could be integrated; ``reason`` why not, or None;
        ``n_cells`` the summed annulus weight [-].

    Defaults
    --------
    ``thickness=0.1`` is a validated workflow default: near the accuracy optimum
    on analytic Solov'ev equilibria and still thousands of cells on a
    reconstruction-sized grid. ``samples_per_axis=5`` is a numerical convenience
    matching :func:`fractional_cell_weights_from_boundary`, and it sets how thin
    an annulus can be resolved.

    Convention
    ----------
    ``alpha = 2 <R Bz^2> / <R Bp^2>``, 1 for a symmetric circular cross-section
    and 2 in the infinitely elongated limit. Only squares of the field enter, so
    the sign convention the components arrive in does not matter.

    Assumptions
    -----------
    The annulus is built by geometric conformal scaling, not as a psi_N band:
    a psi_N band would reintroduce the dependence on reconstructed interior flux
    that this estimate exists to avoid.

    Applicability
    -------------
    Machine-independent. Any boundary with a poloidal-field map covering it.

    Limitations
    -----------
    Abstains -- NaN with a ``reason`` -- rather than returning a number when the
    boundary leaves the grid, when the annulus holds fewer than
    :data:`MIN_ANNULUS_CELLS`, or when it is narrower than
    :data:`MIN_ANNULUS_WIDTH_SUBSAMPLES` sub-samples at its narrowest point, in
    which case the integral would report the grid phase rather than the field.
    Cells whose field is only partly finite are dropped from both integrals
    together, since dropping one from the denominator alone would bias alpha up.
    As ``thickness -> 0`` this tends to
    :func:`virial_alpha_thin_annulus`, whose accuracy against the true alpha is
    slightly *worse* -- a finite annulus samples some interior, and that helps.

    Provenance
    ----------
    .. [1] M. W. Bongard et al., Phys. Plasmas 23 (2016), the low-aspect-ratio
       virial closure and its annulus estimate of alpha.
    .. [2] Measured against analytic Solov'ev equilibria in
       ``test/test_virial_alpha_approximations.py``.
    """
    if np.ndim(R_grid) == 1 and np.ndim(Z_grid) == 1:
        R_grid, Z_grid = np.meshgrid(
            np.asarray(R_grid, float), np.asarray(Z_grid, float), indexing="ij"
        )
    else:
        R_grid = np.asarray(R_grid, float)
        Z_grid = np.asarray(Z_grid, float)
    B_R_grid = np.asarray(B_R_grid, float)
    B_Z_grid = np.asarray(B_Z_grid, float)

    def _fail(reason: str, n_cells: float = 0.0) -> dict[str, Any]:
        return {"alpha": np.nan, "valid": False, "reason": reason, "n_cells": float(n_cells)}

    thickness = float(thickness)
    if not np.isfinite(thickness) or not (0.0 < thickness < 1.0):
        return _fail(f"thickness must lie in (0, 1); got {thickness}.")

    R_out, Z_out = _ensure_closed_boundary(R_bdry, Z_bdry)
    if R_out.size < 4:
        return _fail("boundary has too few points to define an annulus.")
    if _signed_area_closed_polygon(R_out, Z_out) < 0.0:
        R_out, Z_out = R_out[::-1], Z_out[::-1]

    # The outer contour must be on the grid, or the annulus is silently clipped.
    if (
        float(np.min(R_out)) < float(np.min(R_grid))
        or float(np.max(R_out)) > float(np.max(R_grid))
        or float(np.min(Z_out)) < float(np.min(Z_grid))
        or float(np.max(Z_out)) > float(np.max(Z_grid))
    ):
        return _fail("boundary extends beyond the field grid; the annulus would be clipped.")

    R_in, Z_in = scale_boundary_conformal(R_out, Z_out, 1.0 - thickness)

    w_out = fractional_cell_weights_from_boundary(
        R_grid, Z_grid, R_out, Z_out, samples_per_axis=samples_per_axis
    )
    w_in = fractional_cell_weights_from_boundary(
        R_grid, Z_grid, R_in, Z_in, samples_per_axis=samples_per_axis
    )
    weights = np.clip(w_out - w_in, 0.0, 1.0)

    n_cells = float(np.sum(weights))
    if n_cells < MIN_ANNULUS_CELLS:
        return _fail(
            f"annulus resolves {n_cells:.2f} cells, below the {MIN_ANNULUS_CELLS:g} "
            "needed for the integral to describe the field rather than the grid.",
            n_cells,
        )

    dA = _cell_area_from_mesh(R_grid, Z_grid)

    # Total area is not resolution: a long annulus can hold hundreds of cells
    # while being narrower than the sampling can see, in which case the integral
    # reports the grid phase. The narrowest point is what has to be resolved.
    _, _, _, _, _, support = _boundary_normals_and_support(R_out, Z_out)
    min_width = thickness * float(np.min(np.clip(support, 0.0, None)))
    cell = float(np.median(np.sqrt(dA[dA > 0.0]))) if np.any(dA > 0.0) else 0.0
    sample_spacing = cell / float(samples_per_axis)
    if sample_spacing > 0.0 and min_width < MIN_ANNULUS_WIDTH_SUBSAMPLES * sample_spacing:
        return _fail(
            f"annulus is {min_width / sample_spacing:.2f} sub-samples wide at its "
            f"narrowest, below the {MIN_ANNULUS_WIDTH_SUBSAMPLES:g} needed to "
            "resolve the field across it.",
            n_cells,
        )

    # One mask for both integrals: dropping a cell from the denominator (via
    # B_p^2) while keeping it in the numerator (which only reads B_Z) would
    # bias alpha high, so a partially non-finite field excludes the whole cell.
    B_p_sq = B_R_grid**2 + B_Z_grid**2
    finite = np.isfinite(B_R_grid) & np.isfinite(B_Z_grid) & np.isfinite(dA)
    weights = np.where(finite, weights, 0.0)
    num = float(np.sum(R_grid * np.where(finite, B_Z_grid, 0.0) ** 2 * weights * dA))
    den = float(np.sum(R_grid * np.where(finite, B_p_sq, 0.0) * weights * dA))
    if not np.isfinite(num) or not np.isfinite(den) or den == 0.0:
        return _fail("annulus integral of R*Bp^2 is zero or non-finite.", n_cells)

    return {
        "alpha": 2.0 * num / den,
        "valid": True,
        "reason": None,
        "n_cells": n_cells,
    }


def shafranov_integrals(
    R_bdry,
    Z_bdry,
    B_p_bdry,
    R_grid,
    Z_grid,
    B_R_grid,
    B_Z_grid,
    R_0=None,
    Z_0=None,
    p_boundary: float = 0.0,
    B_ref: float | None = None,
    cell_weights: np.ndarray | None = None,
    volume: float | None = None,
):
    """The three Shafranov boundary integrals and the virial coefficient.

    Parameters
    ----------
    R_bdry : array_like
        Major radius of the boundary contour [m].
    Z_bdry : array_like
        Height of the boundary contour [m].
    B_p_bdry : array_like
        Poloidal field magnitude on that contour [T].
    R_grid : array_like
        Major-radius grid, as an axis or a mesh [m].
    Z_grid : array_like
        Height grid, as an axis or a mesh [m].
    B_R_grid : array_like
        Major-radius field component on the grid [T].
    B_Z_grid : array_like
        Height field component on the grid [T].
    R_0 : float, optional
        Reference major radius. The boundary's geometric centre when not given
        [m].
    Z_0 : float, optional
        Reference height, likewise [m].
    p_boundary : float, optional
        Pressure at the boundary [Pa].
    B_ref : float, optional
        Field the integrals are normalized by. The boundary-average poloidal
        field when not given [T].
    cell_weights : array_like, optional
        Per-cell plasma area fractions, from
        :func:`fractional_cell_weights_from_boundary` [-].
    volume : float, optional
        Plasma volume, computed from the boundary when not given [m^3].

    Returns
    -------
    tuple of float
        ``(S1, S2, S3, alpha)``, all dimensionless. Zeros when the conditioned
        boundary has too few points or the reference volume or field is not
        positive [-].

    Convention
    ----------
    **The boundary must run counter-clockwise**, because the outward unit normal
    is taken as ``(dZ/dl, -dR/dl)``, which points outward only then. The contour
    is conditioned through :func:`prepare_boundary_for_shafranov` on entry, so a
    clockwise input is corrected rather than silently inverting every integral.

    The COCOS is **inherited, not declared**: the poloidal field arrays are taken
    as given, so whatever convention produced them is the convention these
    integrals are in. Passing a field computed in the wrong one gives plausible
    numbers that are wrong by that factor.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Returns zeros rather than raising when the geometry is degenerate, so a caller
    must distinguish a real zero from a refusal. Without *cell_weights* the plasma
    mask is per-cell and the boundary is resolved only to the grid.

    Provenance
    ----------
    .. [1] The Shafranov boundary integrals as EFIT evaluates them; the virial
       internal inductance built from them is
       :func:`vaft.formula.equilibrium.virial_li_from_S_alpha_rt`.
    """

    R_bdry, Z_bdry, B_p_bdry = _prepare_boundary_and_field_for_shafranov(
        R_bdry,
        Z_bdry,
        B_p_bdry,
        n_points=256,
        enforce_ccw=True,
    )
    if R_bdry.size < 4:
        return 0.0, 0.0, 0.0, 0.0

    if np.ndim(R_grid) == 1 and np.ndim(Z_grid) == 1:
        R_grid, Z_grid = np.meshgrid(np.asarray(R_grid, float), np.asarray(Z_grid, float), indexing="ij")
    else:
        R_grid = np.asarray(R_grid, float)
        Z_grid = np.asarray(Z_grid, float)
    B_R_grid = np.asarray(B_R_grid, float)
    B_Z_grid = np.asarray(B_Z_grid, float)

    # R0, Z0가 없으면 기하학적 중심 계산
    if R_0 is None or not np.isfinite(R_0):
        R_0 = (np.min(R_bdry) + np.max(R_bdry)) / 2.0
    if Z_0 is None or not np.isfinite(Z_0):
        Z_0 = (np.min(Z_bdry) + np.max(Z_bdry)) / 2.0

    B_pa = calculate_average_boundary_poloidal_field(R_bdry, Z_bdry, B_p_bdry)
    if B_ref is None:
        B_ref = B_pa

    # --- 1. 부피(Volume) Omega 계산 ---
    dR_b = np.diff(R_bdry)
    dZ_b = np.diff(Z_bdry)
    R_mid_b = 0.5 * (R_bdry[:-1] + R_bdry[1:])
    Z_mid_b = 0.5 * (Z_bdry[:-1] + Z_bdry[1:])
    B_p_mid = 0.5 * (B_p_bdry[:-1] + B_p_bdry[1:])
    dl = np.hypot(dR_b, dZ_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Boundary is normalized to CCW upstream. For CCW contour, outward unit normal is:
        # n = (dZ/dl, -dR/dl)
        nR = np.where(dl > 0.0, dZ_b / dl, 0.0)
        nZ = np.where(dl > 0.0, -dR_b / dl, 0.0)

    Omega = float(np.abs(-np.sum(np.pi * (R_mid_b**2) * dZ_b))) if volume is None else float(volume)

    # --- 2. Surface Integrals (S1, S2, S3) ---
    if Omega <= 0.0 or B_ref <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    coeff = 2.0 * np.pi / (Omega * (B_ref**2))
    g = R_mid_b * (B_p_mid**2 + 2.0 * MU0 * float(p_boundary))
    S1 = coeff * np.sum(g * (nR * (R_mid_b - R_0) + nZ * (Z_mid_b - Z_0)) * dl)
    S2 = coeff * R_0 * np.sum(g * nR * dl)
    S3 = coeff * np.sum(g * (Z_mid_b - Z_0) * nZ * dl)

    # --- 3. Alpha 계산 ---
    weights = _plasma_cell_weights(R_grid, Z_grid, R_bdry, Z_bdry, cell_weights=cell_weights)
    dA = _cell_area_from_mesh(R_grid, Z_grid)
    B_p_sq = B_R_grid**2 + B_Z_grid**2
    # Mask on the weight, not with nansum. The psi-gradient field fallback marks
    # the R = 0 column NaN on purpose and 0 * nan is nan, so a column *outside*
    # the plasma used to void alpha for the whole slice. Dropping every NaN
    # instead would also swallow one *inside* it, where the honest answer is
    # NaN rather than a plausible-looking biased number.
    _inside = weights > 0.0
    alpha_num = np.sum(np.where(_inside, R_grid * (B_Z_grid**2) * weights * dA, 0.0))
    alpha_den = np.sum(np.where(_inside, R_grid * B_p_sq * weights * dA, 0.0))
    # NaN, not the 0.0 sentinel, when the denominator is undetermined. 0.0 is
    # pair_23's exact singular point and a plausible-looking alpha, so it sails
    # past the wrapper's `isfinite` abstain guard and yields finite closures
    # from an equilibrium nothing could measure. The 0.0 above is for degenerate
    # *geometry*, which is a different statement.
    if not np.isfinite(alpha_den):
        alpha = np.nan
    elif alpha_den == 0.0:
        alpha = 0.0
    else:
        alpha = float(2.0 * alpha_num / alpha_den)

    return S1, S2, S3, alpha


def efit_virial_volume_integrals(
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    R_bdry: np.ndarray,
    Z_bdry: np.ndarray,
    B_R_grid: np.ndarray,
    B_Z_grid: np.ndarray,
    p_tot_grid: np.ndarray | None = None,
    B_phi_grid: np.ndarray | None = None,
    B_phi_vac_grid: np.ndarray | None = None,
    F_grid: np.ndarray | None = None,
    F_boundary: float | None = None,
    cell_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """EFIT-style weighted volume integrals for the virial closures.

    Parameters
    ----------
    R_grid : array_like
        Major-radius grid, as an axis or a mesh [m].
    Z_grid : array_like
        Height grid, as an axis or a mesh [m].
    R_bdry : array_like
        Major radius of the boundary contour [m].
    Z_bdry : array_like
        Height of the boundary contour [m].
    B_R_grid : array_like
        Major-radius field component on the grid [T].
    B_Z_grid : array_like
        Height field component on the grid [T].
    p_tot_grid : array_like, optional
        Total pressure on the grid [Pa].
    B_phi_grid : array_like, optional
        Plasma toroidal field on the grid [T].
    B_phi_vac_grid : array_like, optional
        Vacuum toroidal field on the same grid [T].
    F_grid : array_like, optional
        Poloidal current function on the grid [T m].
    F_boundary : float, optional
        That function at the boundary [T m].
    cell_weights : array_like, optional
        Per-cell plasma area fractions; computed from the boundary when not given
        [-].

    Returns
    -------
    dict of str to float
        ``alpha`` and ``rt_denominator_ratio`` dimensionless, ``phi_dia_comp`` in
        weber, ``dV`` and ``volume`` in cubic metres, ``b_p_squared`` in tesla
        squared, and ``rt`` in metres [-].

    Convention
    ----------
    Weighted by fractional cell area rather than a per-cell mask, which is what
    makes these the EFIT-style forms rather than a plain grid sum. Like
    :func:`shafranov_integrals`, the COCOS is inherited from the field arrays
    passed in and is neither declared nor checked here.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The reference radius is guarded against a near-singular denominator, so a
    degenerate geometry returns a finite but meaningless value rather than
    diverging. Every optional input governs one output: omit the pressure and the
    pressure-dependent terms are simply absent.

    Provenance
    ----------
    .. [1] EFIT's weighted volume integrals, with the fractional cell weights
       standing in for its externally supplied weight map; see
       :func:`fractional_cell_weights_from_boundary`.
    """
    if np.ndim(R_grid) == 1 and np.ndim(Z_grid) == 1:
        R_grid, Z_grid = np.meshgrid(np.asarray(R_grid, float), np.asarray(Z_grid, float), indexing="ij")
    else:
        R_grid = np.asarray(R_grid, float)
        Z_grid = np.asarray(Z_grid, float)
    B_R_grid = np.asarray(B_R_grid, float)
    B_Z_grid = np.asarray(B_Z_grid, float)
    R_bdry, Z_bdry = _ensure_closed_boundary(R_bdry, Z_bdry)
    dA = _cell_area_from_mesh(R_grid, Z_grid)
    weights = _plasma_cell_weights(R_grid, Z_grid, R_bdry, Z_bdry, cell_weights=cell_weights)

    B_p_sq = B_R_grid**2 + B_Z_grid**2
    # Mask on the weight, not with nansum. The psi-gradient field fallback marks
    # the R = 0 column NaN on purpose and 0 * nan is nan, so a column *outside*
    # the plasma used to void alpha for the whole slice. Dropping every NaN
    # instead would also swallow one *inside* it, where the honest answer is
    # NaN rather than a plausible-looking biased number.
    _inside = weights > 0.0
    alpha_num = np.sum(np.where(_inside, R_grid * (B_Z_grid**2) * weights * dA, 0.0))
    alpha_den = np.sum(np.where(_inside, R_grid * B_p_sq * weights * dA, 0.0))
    alpha = np.nan if not np.isfinite(alpha_den) or alpha_den == 0.0 else float(2.0 * alpha_num / alpha_den)

    RT = np.nan
    # |int G dA| / int |G| dA -- how much of the RT denominator survives the
    # cancellation inside G. Near zero, RT is a ratio of two small differences
    # and its value says nothing, so the conditioning number is reported
    # alongside RT rather than only used to gate it (#546 s10).
    rt_denominator_ratio = np.nan
    if p_tot_grid is not None and B_phi_grid is not None and B_phi_vac_grid is not None:
        p_tot_grid = np.asarray(p_tot_grid, float)
        B_phi_grid = np.asarray(B_phi_grid, float)
        B_phi_vac_grid = np.asarray(B_phi_vac_grid, float)
        G = 2.0 * MU0 * p_tot_grid + B_p_sq + B_phi_vac_grid**2 - B_phi_grid**2
        G_weighted = G * weights * dA
        RT_num = float(np.nansum(R_grid * G_weighted))
        RT_den = float(np.nansum(G_weighted))
        # Guard against near-singular denominator to prevent RT/R0 blow-up.
        rt_den_scale = float(np.nansum(np.abs(G_weighted)))
        if rt_den_scale > 0.0 and np.isfinite(RT_den):
            rt_denominator_ratio = abs(RT_den) / rt_den_scale
        if (
            np.isfinite(RT_num)
            and np.isfinite(RT_den)
            and rt_den_scale > 0.0
            and abs(RT_den) > 1e-6 * rt_den_scale
        ):
            RT = RT_num / RT_den

    phi_dia_comp = np.nan
    if F_grid is not None and F_boundary is not None:
        F_grid = np.asarray(F_grid, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            phi_term = -((float(F_boundary) - F_grid) / R_grid) * weights * dA
        phi_dia_comp = float(np.nansum(phi_term))

    # The volume element every quantity here is weighted by. Returned so a
    # caller computing volume-integral beta_p or l_i uses the same cells and the
    # same weights as `volume` and the Shafranov integrals, instead of a second
    # masking convention that would make the two incomparable.
    dV = 2.0 * np.pi * R_grid * weights * dA
    volume = float(np.nansum(dV))
    return {
        "alpha": alpha,
        "rt": RT,
        "rt_denominator_ratio": rt_denominator_ratio,
        "phi_dia_comp": phi_dia_comp,
        "volume": volume,
        "dV": dV,
        "b_p_squared": B_p_sq,
    }


def computed_diamagnetism_from_phi(
    phi_dia_comp: float,
    B_t0: float,
    R_0: float,
    volume: float,
    B_ref: float,
) -> float:
    """EFIT-style diamagnetism from an already-computed diamagnetic flux.

    Parameters
    ----------
    phi_dia_comp : float
        The computed diamagnetic flux [Wb].
    B_t0 : float
        Vacuum toroidal field at the reference radius [T].
    R_0 : float
        Reference major radius [m].
    volume : float
        Plasma volume, strictly positive [m^3].
    B_ref : float
        Reference field the result is normalized by, normally the boundary-average
        poloidal field, strictly positive [T].

    Returns
    -------
    float
        The EFIT-style diamagnetism [-].

    Raises
    ------
    ValueError
        The volume or the reference field is not positive.

    Convention
    ----------
    ``4*pi*B_t0*R_0/(V*B_ref^2)`` times the flux. The arithmetic itself lives in
    :func:`vaft.formula.equilibrium.virial_muihat_from_Bt_R0_dphi`; this wrapper
    adds only the positivity precondition its callers rely on. Keeping one
    definition matters because the validation layer compares a diamagnetism from
    this path against one from the formula path, and a correction applied to only
    one of them would desynchronize them silently.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Takes the flux as given and cannot check that it was computed against the
    same reference radius and field used here.

    Provenance
    ----------
    .. [1] The EFIT ``xmui`` definition, evaluated through
       :func:`vaft.formula.equilibrium.virial_muihat_from_Bt_R0_dphi`.
    """
    if volume <= 0.0 or B_ref <= 0.0:
        raise ValueError("volume and B_ref must be positive.")
    from vaft.formula.equilibrium import virial_muihat_from_Bt_R0_dphi

    return float(
        virial_muihat_from_Bt_R0_dphi(B_t0, R_0, phi_dia_comp, B_ref, volume)
    )


psi_to_RZ = psi_to_rz


def extract_flux_surface_contours(
    psi_grid: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    psi_axis: float,
    psi_boundary: float,
    levels_norm: Any,
) -> dict[float, list[tuple[np.ndarray, np.ndarray]]]:
    """Iso-flux contours at requested normalized levels, by marching squares.

    Parameters
    ----------
    psi_grid : array_like
        Poloidal flux on the grid, indexed ``(R, Z)`` [Wb/rad].
    R : array_like
        Major-radius grid axis [m].
    Z : array_like
        Height grid axis [m].
    psi_axis : float
        Poloidal flux on the magnetic axis, the zero of the normalization
        [Wb/rad].
    psi_boundary : float
        Poloidal flux at the boundary, the one [Wb/rad].
    levels_norm : sequence of float
        Normalized flux levels to trace, 0 on axis and 1 at the boundary [-].

    Returns
    -------
    dict of float to list of tuple of np.ndarray
        Each requested level mapped to its contour segments as ``(R, Z)`` point
        arrays. A level with no contour on this grid maps to an empty list [m].

    Convention
    ----------
    The flux map is indexed major radius first, matching the IMAS data
    dictionary's two-dimensional profile layout with the first and second grid
    dimensions as major radius and height. Levels are normalized, so the absolute
    unit of the three flux arguments cancels as long as they agree.

    **A level may have more than one contour.** Inside a diverted plasma a single
    normalized level can trace disconnected pieces, so the value is always a list
    and a caller wanting "the" surface must choose among them, normally the closed
    one containing the magnetic axis.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Marching squares on the grid, so a contour is resolved only to the cell and a
    surface smaller than a few cells is poorly represented. Contours are returned
    in the order the algorithm finds them, not sorted by size or by containment.

    Provenance
    ----------
    .. [1] The IMAS data dictionary layout for a two-dimensional flux profile,
       which fixes the axis order this assumes.
    """
    from skimage import measure

    psi_grid = np.asarray(psi_grid, dtype=float)
    R = np.asarray(R, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float).reshape(-1)
    if psi_grid.shape != (R.size, Z.size):
        raise ValueError(
            f"psi_grid shape {psi_grid.shape} must equal (len(R), len(Z)) = {(R.size, Z.size)}."
        )
    if psi_boundary == psi_axis:
        raise ValueError("psi_boundary must differ from psi_axis to normalize.")

    psi_norm_grid = (psi_grid - psi_axis) / (psi_boundary - psi_axis)

    def _index_to_rz(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        row_idx, col_idx = contour[:, 0], contour[:, 1]
        r_pts = np.interp(row_idx, np.arange(R.size), R)
        z_pts = np.interp(col_idx, np.arange(Z.size), Z)
        return r_pts, z_pts

    contours: dict[float, list[tuple[np.ndarray, np.ndarray]]] = {}
    for level in levels_norm:
        level = float(level)
        raw_contours = measure.find_contours(psi_norm_grid, level=level)
        contours[level] = [_index_to_rz(contour) for contour in raw_contours]

    return contours


#: Below this many vertices a marching-squares contour describes grid artifact
#: rather than geometry.  16 is deliberately permissive: on a 129x129 VEST map
#: the innermost resolved surface carries ~21 vertices, and dropping it costs
#: more than keeping it -- measured against the OMFIT reference, raising the
#: threshold to 24 moves the worst `elongation` error from 1.8e-3 to 2.3e-2.
MIN_FLUX_SURFACE_POINTS = 16


def contour_shape_parameters(r_seg: np.ndarray, z_seg: np.ndarray) -> dict[str, float]:
    """Shape parameters of one closed flux-surface contour.

    Parameters
    ----------
    r_seg : array_like
        Major radius of the contour points [m].
    z_seg : array_like
        Height of the contour points [m].

    Returns
    -------
    dict of str to float
        ``volume`` in cubic metres; ``area`` and ``surface`` in square metres;
        ``elongation``, ``triangularity_upper`` and ``triangularity_lower``
        dimensionless; ``r_inboard`` and ``r_outboard`` in metres [-].

    Raises
    ------
    ValueError
        The contour is degenerate, having zero minor radius.

    Convention
    ----------
    Volume is the exact revolution ``pi * closed_integral(R^2 dZ)``, not Pappus's
    approximation, so it is right even for a contour whose centroid is far from
    its geometric centre. Area is the shoelace formula and surface is
    ``closed_integral 2*pi*R dl``.

    Elongation and both triangularities are taken against the geometric centre
    ``(R_out + R_in)/2``, which is what the IMAS data dictionary reports for the
    boundary. Positive triangularity means the extremum of height sits inboard of
    that centre. The extremum's major radius comes from
    :func:`r_at_z_extremum` rather than from the nearest vertex.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Assumes one simple closed contour; a self-intersecting or open segment gives
    a signed area that does not mean what the formulas take it to mean. Every
    quantity is only as accurate as the contour's sampling.

    Provenance
    ----------
    .. [1] IMAS data dictionary definitions for the boundary's geometric axis and
       triangularity, which fix the centre these are measured against.
    .. [2] The exact revolution volume is
       :func:`vaft.formula.equilibrium.exact_volume_from_RZ_contour`.
    """
    from vaft.formula.equilibrium import exact_volume_from_RZ_contour

    r_seg = np.asarray(r_seg, dtype=float).reshape(-1)
    z_seg = np.asarray(z_seg, dtype=float).reshape(-1)
    r_min, r_max = float(np.min(r_seg)), float(np.max(r_seg))
    z_min, z_max = float(np.min(z_seg)), float(np.max(z_seg))
    minor = 0.5 * (r_max - r_min)
    r_geo = 0.5 * (r_max + r_min)
    if minor <= 0.0:
        raise ValueError("degenerate contour")
    r_closed = np.r_[r_seg, r_seg[0]]
    z_closed = np.r_[z_seg, z_seg[0]]
    segment_length = np.hypot(np.diff(r_closed), np.diff(z_closed))
    r_mid = 0.5 * (r_closed[1:] + r_closed[:-1])
    return {
        "volume": exact_volume_from_RZ_contour(r_seg, z_seg),
        # Poloidal cross-section area, by the shoelace formula.
        "area": 0.5
        * abs(
            float(np.dot(r_seg, np.roll(z_seg, 1)) - np.dot(z_seg, np.roll(r_seg, 1)))
        ),
        "surface": 2.0 * np.pi * float(np.sum(r_mid * segment_length)),
        "elongation": (z_max - z_min) / (2.0 * minor),
        "triangularity_upper": (r_geo - r_at_z_extremum(r_seg, z_seg, upper=True)) / minor,
        "triangularity_lower": (r_geo - r_at_z_extremum(r_seg, z_seg, upper=False)) / minor,
        "r_inboard": r_min,
        "r_outboard": r_max,
    }


def r_at_z_extremum(r_seg: np.ndarray, z_seg: np.ndarray, *, upper: bool) -> float:
    """Major radius where a contour reaches its highest or lowest point.

    Parameters
    ----------
    r_seg : array_like
        Major radius of the contour points [m].
    z_seg : array_like
        Height of the contour points [m].
    upper : bool
        ``True`` for the highest point, ``False`` for the lowest [-].

    Returns
    -------
    float
        The major radius at that extremum [m].

    Convention
    ----------
    Sub-vertex, not nearest-vertex. Taking the major radius at the sampled vertex
    of extreme height is off by several percent in triangularity, because the true
    extremum falls between vertices. A parabola is fitted to height over the three
    points around the extreme vertex, and the major radius is interpolated there.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Falls back to the extreme vertex itself for a contour of fewer than three
    points. The parabola is local, so a contour sampled too coarsely to resolve
    its own curvature near the extremum is still limited by that sampling.

    Provenance
    ----------
    .. [1] Consumed by :func:`contour_shape_parameters` for both triangularities,
       which is where the several-percent error would otherwise land.
    .. [2] The arithmetic is
       :func:`vaft.formula.equilibrium.r_at_z_extremum_from_RZ_contour`; this is
       the process-layer name for it, so the two layers cannot drift apart.
    """
    from vaft.formula.equilibrium import r_at_z_extremum_from_RZ_contour

    return r_at_z_extremum_from_RZ_contour(r_seg, z_seg, upper=upper)


def _closed_contour(r_seg: np.ndarray, z_seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Append the first vertex when the segment is not already closed."""
    if (r_seg[0] - r_seg[-1]) ** 2 + (z_seg[0] - z_seg[-1]) ** 2 > 1e-18:
        return np.r_[r_seg, r_seg[0]], np.r_[z_seg, z_seg[0]]
    return r_seg, z_seg


def _enclosing_segment(
    segments: list[tuple[np.ndarray, np.ndarray]],
    axis_rz: tuple[float, float] | None,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The contour segment that is actually the flux surface, or ``None``.

    A level can return several disconnected segments -- the confined surface,
    private-flux lobes, scrape-off branches clipped by the grid. Longest-wins
    picks the wrong one often enough to matter (up to every level on a limited
    VEST slice), so the segment enclosing the magnetic axis wins outright and
    length only breaks ties among those.

    ``min_points`` is applied *after* that choice, never before it. Screening on
    size first lets a large scrape-off branch outlive the small contour that is
    actually the flux surface: doing so moved the derived plasma current 4-6% off
    ``global_quantities.ip`` on the later, smaller slices of the packaged VEST
    sample. An enclosing contour too coarse to use is reported as unresolved
    (``None``, so the caller interpolates the gap) rather than replaced.
    """
    if not segments:
        return None
    candidates = segments
    if axis_rz is not None:
        from matplotlib.path import Path as _MplPath

        enclosing = []
        for r_seg, z_seg in segments:
            r_closed, z_closed = _closed_contour(r_seg, z_seg)
            if _MplPath(np.column_stack([r_closed, z_closed])).contains_point(axis_rz):
                enclosing.append((r_seg, z_seg))
        if enclosing:
            candidates = enclosing
    chosen = max(candidates, key=lambda segment: segment[0].size)
    return chosen if chosen[0].size >= min_points else None


#: Every profile :func:`flux_surface_quantities` returns.
FLUX_SURFACE_QUANTITIES = (
    "gm1",
    "gm5",
    "gm8",
    "gm9",
    "dvolume_dpsi",
    "darea_dpsi",
    "volume",
    "area",
    "surface",
    "elongation",
    "triangularity_upper",
    "triangularity_lower",
    "r_inboard",
    "r_outboard",
    "b_field_max",
    "b_field_min",
    "bp_dl",
    "length_pol",
    "q",
)


def flux_surface_quantities(
    psi_grid: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    psi_axis: float,
    psi_boundary: float,
    levels_norm: Any,
    *,
    f_profile: Any = None,
    axis_rz: tuple[float, float] | None = None,
    boundary: tuple[np.ndarray, np.ndarray] | None = None,
    min_points: int = MIN_FLUX_SURFACE_POINTS,
) -> dict[str, np.ndarray]:
    """Flux-surface averages and shape parameters on each normalized flux level.

    Parameters
    ----------
    psi_grid : array_like
        Poloidal flux on the grid, indexed ``(R, Z)``, **per radian** [Wb/rad].
    R : array_like
        Major-radius grid axis [m].
    Z : array_like
        Height grid axis [m].
    psi_axis : float
        Poloidal flux on the magnetic axis [Wb/rad].
    psi_boundary : float
        Poloidal flux at the boundary [Wb/rad].
    levels_norm : sequence of float
        Normalized flux levels to evaluate on, 0 on axis and 1 at the boundary
        [-].
    f_profile : array_like, optional
        Poloidal current function ``F = R B_phi`` on the same levels. Required for
        the mean square field and safety factor [T m].
    axis_rz : tuple of float, optional
        Magnetic axis position, used to pick the confined segment when a level
        traces several [m].
    boundary : tuple of np.ndarray, optional
        Boundary outline to use at the outermost level instead of tracing it [m].
    min_points : int, optional
        Fewest contour vertices a level needs before it is treated as resolved
        [-].

    Returns
    -------
    dict of str to np.ndarray
        One array per name in :data:`FLUX_SURFACE_QUANTITIES`, each as long as
        *levels_norm*. Volume in cubic metres, areas in square metres, radii and
        ``length_pol`` in metres, ``bp_dl`` in tesla-metre, ``gm1`` in inverse
        square metres, ``gm5`` in tesla squared, ``gm8`` in metres, ``gm9`` in
        inverse metres, the shape parameters and safety factor ``q`` dimensionless,
        and the two derivatives per radian [-].

    Convention
    ----------
    **Per radian.** The normalization by axis and boundary flux is scale-free, but
    the flux gradient that weights every average is not, which is why the
    per-radian contract matters. ``dvolume_dpsi`` and ``darea_dpsi`` are likewise
    per radian; a caller storing them against a weber flux divides by ``2*pi``.

    The average weights by the volume element,
    ``<X> = closed_integral(X R dl / |grad psi|) / closed_integral(R dl / |grad psi|)``,
    giving ``gm1 = <1/R^2>``, ``gm8 = <R>``, ``gm9 = <1/R>`` and, with
    *f_profile*, ``gm5 = <B^2>`` from ``|B|^2 = (|grad psi|^2 + F^2)/R^2``.
    ``bp_dl`` is instead the plain contour integral of the poloidal field, which is
    what an integral of that field squared over volume needs, since the volume
    element cancels one power of it. The flux map is indexed major radius first,
    matching :func:`extract_flux_surface_contours`.

    Processing steps
    ----------------
    1. Trace the contour at each level, taking the confined segment when several
       are returned, and substituting the supplied outline at the boundary.
    2. Form the averages and the shape parameters on each resolved contour.
    3. Return NaN for a level whose contour is missing or below *min_points*, then
       fill those by interpolation against the square root of the normalized flux,
       the coordinate in which near-axis geometry is linear.
    4. Set the on-axis level to its exact point limits.

    Defaults
    --------
    ``min_points`` is a numerical convenience tied to what a flux map can resolve;
    see :data:`MIN_FLUX_SURFACE_POINTS` for the measurement behind its value.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The innermost one or two levels are the least accurate everywhere, because the
    flux gradient is small and varies fastest there. That is a property of the map,
    not of this routine: on the packaged OMFIT reference the *stored*
    ``dvolume_dpsi`` runs 8.53, 22.85, 29.59, 29.05, 28.85 from the axis out, a
    ramp that cannot be physical since the derivative approaches a finite limit,
    while the trace here gives a smooth 30.8, 29.3, 29.3. Compare against a
    reference only outside a normalized flux of about 0.05, and check the
    near-axis values against the integral of the derivative recovering the volume
    instead.

    On axis the surface is a point: the volume, area and surface are exactly zero
    and the three geometric averages take their axis limits, but the shape
    parameters and the two derivatives are undefined there and are extrapolated
    from the innermost resolved surface.

    Provenance
    ----------
    .. [1] The flux-surface average as the IMAS data dictionary defines the ``gm``
       family, and the same contour extraction
       :func:`extract_flux_surface_contours` provides.
    """
    psi_grid = np.asarray(psi_grid, dtype=float)
    R = np.asarray(R, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float).reshape(-1)
    levels = np.asarray(levels_norm, dtype=float).reshape(-1)
    if psi_grid.shape != (R.size, Z.size):
        raise ValueError(
            f"psi_grid shape {psi_grid.shape} must equal (len(R), len(Z)) = {(R.size, Z.size)}."
        )
    if psi_boundary == psi_axis:
        raise ValueError("psi_boundary must differ from psi_axis to normalize.")

    f_values = None
    if f_profile is not None:
        f_values = np.asarray(f_profile, dtype=float).reshape(-1)
        if f_values.size != levels.size:
            raise ValueError("f_profile must have one value per level")

    out = {name: np.full(levels.size, np.nan) for name in FLUX_SURFACE_QUANTITIES}
    spline = RectBivariateSpline(R, Z, psi_grid)

    # A supplied boundary replaces the traced edge contour only when it is at
    # least as well resolved as an interior level would have to be.  A coarse
    # EFIT outline -- `update_equilibrium_boundary` passes anything with 3 points
    # -- would otherwise set the edge, and the edge anchors the gap fill inward.
    edge_from_boundary = boundary is not None and (
        np.asarray(boundary[0], dtype=float).reshape(-1).size >= min_points
    )
    # The axis level has no contour and the boundary level may come from the
    # stored outline, so only the rest need tracing.
    traced = [
        float(level)
        for level in levels
        if level != 0.0 and not (edge_from_boundary and level == 1.0)
    ]
    contours = (
        extract_flux_surface_contours(psi_grid, R, Z, psi_axis, psi_boundary, traced)
        if traced
        else {}
    )

    from vaft.formula.equilibrium import q_from_flux_surface_averages

    for index, level in enumerate(levels):
        if level == 0.0:
            if axis_rz is not None:
                r_axis = float(axis_rz[0])
                out["gm1"][index] = 1.0 / r_axis**2
                out["gm8"][index] = r_axis
                out["gm9"][index] = 1.0 / r_axis
                out["r_inboard"][index] = r_axis
                out["r_outboard"][index] = r_axis
            out["volume"][index] = 0.0
            out["area"][index] = 0.0
            out["surface"][index] = 0.0
            out["bp_dl"][index] = 0.0
            out["length_pol"][index] = 0.0
            continue

        if edge_from_boundary and level == 1.0:
            segment = (
                np.asarray(boundary[0], dtype=float).reshape(-1),
                np.asarray(boundary[1], dtype=float).reshape(-1),
            )
        else:
            segment = _enclosing_segment(
                contours.get(float(level), []), axis_rz, min_points
            )
        if segment is None:
            continue

        r_seg, z_seg = segment
        try:
            shape = contour_shape_parameters(r_seg, z_seg)
        except ValueError:
            continue
        for name, value in shape.items():
            out[name][index] = value

        r_closed, z_closed = _closed_contour(r_seg, z_seg)
        r_mid = 0.5 * (r_closed[1:] + r_closed[:-1])
        z_mid = 0.5 * (z_closed[1:] + z_closed[:-1])
        length = np.hypot(np.diff(r_closed), np.diff(z_closed))
        grad = np.hypot(
            spline.ev(r_mid, z_mid, dx=1, dy=0), spline.ev(r_mid, z_mid, dx=0, dy=1)
        )
        finite = np.isfinite(grad) & (grad > 0)
        if np.count_nonzero(finite) < 3:
            continue
        r_mid, length, grad = r_mid[finite], length[finite], grad[finite]
        weight = r_mid * length / grad
        total = float(np.sum(weight))
        if not np.isfinite(total) or total <= 0:
            continue
        # closed_integral(B_p dl), with B_p = |grad psi|/R.  This is all that
        # int(B_p^2 dV) needs: dV = 2*pi*R dl dpsi/|grad psi| cancels one power
        # of B_p exactly, leaving int B_p^2 dV = 2*pi * sum_k (oint B_p dl)_k
        # dpsi_k.  Per radian, like the two derivatives.
        out["bp_dl"][index] = float(np.sum((grad / r_mid) * length))
        out["length_pol"][index] = float(np.sum(length))
        out["gm1"][index] = float(np.sum(weight / r_mid**2) / total)
        out["gm8"][index] = float(np.sum(weight * r_mid) / total)
        out["gm9"][index] = float(np.sum(weight / r_mid) / total)
        out["dvolume_dpsi"][index] = 2.0 * np.pi * total
        out["darea_dpsi"][index] = float(np.sum(length / grad))
        if f_values is not None and np.isfinite(f_values[index]):
            b_mod = np.hypot(grad / r_mid, f_values[index] / r_mid)
            out["gm5"][index] = float(np.sum(weight * b_mod**2) / total)
            out["b_field_max"][index] = float(np.max(b_mod))
            out["b_field_min"][index] = float(np.min(b_mod))
            out["q"][index] = float(
                q_from_flux_surface_averages(
                    out["gm1"][index], out["dvolume_dpsi"][index], f_values[index]
                )
            )

    if f_values is not None:
        # q is quadratic in radius r ~ sqrt(psi_N), making it smooth and linear in
        # psi_N near the magnetic axis. Extrapolate q to axis level (psi_N = 0) in psi_N.
        # Use a low-degree polynomial fit over innermost resolved surfaces (psi_N <= 0.25)
        # to filter out discrete polygon vertex jitter on tiny near-axis contours.
        axis_mask = levels == 0.0
        if axis_mask.any() and not np.isfinite(out["q"][axis_mask]).all():
            finite_idx = np.where(np.isfinite(out["q"]))[0]
            if len(finite_idx) >= 2:
                sorted_finite = finite_idx[np.argsort(levels[finite_idx])]
                fit_idx = [i for i in sorted_finite if levels[i] <= 0.25]
                if len(fit_idx) < 3:
                    fit_idx = sorted_finite[: min(len(sorted_finite), 5)]
                p_fit = levels[fit_idx]
                q_fit = out["q"][fit_idx]
                poly = np.polyfit(p_fit, q_fit, deg=1)
                out["q"][axis_mask] = float(np.polyval(poly, 0.0))
            elif len(finite_idx) == 1:
                out["q"][axis_mask] = out["q"][finite_idx[0]]

    # Gaps are filled against sqrt(psi_N), not psi_N: near the axis a flux
    # surface's linear size goes as sqrt(psi_N), so every quantity that vanishes
    # there is linear in sqrt and badly curved in psi_N.  Interpolating a dropped
    # innermost level in psi_N underestimates `surface` by a third.
    # q is instead smooth and linear in psi_N, so its gaps are filled against levels.
    coordinate = np.sqrt(np.clip(levels, 0.0, None))
    order = np.argsort(coordinate, kind="stable")
    order_linear = np.argsort(levels, kind="stable")
    for name, values in out.items():
        missing = ~np.isfinite(values)
        if missing.any() and not missing.all():
            if name == "q":
                good_sorted = order_linear[np.isfinite(values[order_linear])]
                values[missing] = np.interp(
                    levels[missing], levels[good_sorted], values[good_sorted]
                )
            else:
                good_sorted = order[np.isfinite(values[order])]
                values[missing] = np.interp(
                    coordinate[missing], coordinate[good_sorted], values[good_sorted]
                )
    return out


def calculate_q_profile_from_psi(
    psi_grid: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    f_profile: Any,
    psi_axis: float | None = None,
    psi_boundary: float | None = None,
    levels_norm: Any = None,
    *,
    axis_rz: tuple[float, float] | None = None,
    boundary: tuple[np.ndarray, np.ndarray] | None = None,
    cocos: int = 11,
    sigma_ip: int = 1,
    sigma_b0: int = 1,
    min_points: int = MIN_FLUX_SURFACE_POINTS,
    return_details: bool = False,
) -> np.ndarray | dict[str, Any]:
    """Calculate the safety factor profile q(psi) from a 2D poloidal flux map and F(psi).

    Parameters
    ----------
    psi_grid : array_like
        Poloidal flux on the grid, shaped ``(len(R), len(Z))``, in the convention
        ``cocos`` specifies [Wb or Wb/rad].
    R : array_like
        Major-radius grid axis [m].
    Z : array_like
        Height grid axis [m].
    f_profile : float, array_like, callable, or tuple of array_like
        Poloidal current function ``F = R B_phi``, as a scalar float, 1D array,
        callable ``f(psi)``, or tuple ``(psi_f, f_values)`` [T m].
    psi_axis : float, optional
        Poloidal flux at the magnetic axis, in the convention ``cocos`` specifies [Wb or Wb/rad].
    psi_boundary : float, optional
        Poloidal flux at the plasma boundary/LCFS, in the convention ``cocos`` specifies [Wb or Wb/rad].
    levels_norm : sequence of float, optional
        Normalized flux levels to evaluate on, 0 on axis and 1 at the boundary [-].
    axis_rz : tuple of float, optional
        Magnetic axis coordinates ``(R_axis, Z_axis)`` [m].
    boundary : tuple of np.ndarray, optional
        Boundary outline ``(R_bdry, Z_bdry)`` [m].
    cocos : int, optional
        COCOS coordinate convention index (1-8, 11-18) describing the input
        conventions and target sign for ``q`` [-].
    sigma_ip : int, optional
        Sign of plasma current (+1 or -1) in the equilibrium coordinate system [-].
    sigma_b0 : int, optional
        Sign of toroidal field (+1 or -1) in the equilibrium coordinate system [-].
    min_points : int, optional
        Fewest contour vertices a level needs before it is treated as resolved [-].
    return_details : bool, optional
        Whether to return diagnostic parameters and the full flux-surface
        quantities alongside the safety factor array [-].

    Returns
    -------
    np.ndarray or dict of str to Any
        If ``return_details`` is False, returns a 1D array of safety factor values ``q``
        on ``levels_norm`` [-].
        If ``return_details`` is True, returns a dict with keys ``"q"``, ``"levels_norm"``,
        ``"psi_levels"``, ``"q_axis"``, ``"q_95"``, and ``"surfaces"`` [-].

    Raises
    ------
    ValueError
        The flux map is not shaped to ``(len(R), len(Z))``, ``psi_axis`` or
        ``psi_boundary`` cannot be determined or are equal, or ``f_profile``
        format is invalid.

    Convention
    ----------
    In Sauter and Medvedev (2013), the safety factor is defined as:
    $$q(\\psi) = \\frac{F(\\psi)}{2\\pi} \\oint \\frac{dl_p}{R^2 B_p}
              = \\frac{F(\\psi)}{2\\pi} (2\\pi)^{e_{B_p}} \\oint \\frac{dl_p}{R |\\nabla\\psi|}$$
    Expressed in terms of the geometric flux-surface averages computed by
    :func:`flux_surface_quantities` (which operates in Wb/rad, $e_{B_p} = 0$, $dV/d\\psi$ per radian):
    $$q(\\psi) = \\sigma \\cdot \\frac{F(\\psi)}{(2\\pi)^2} \\left\\langle \\frac{1}{R^2} \\right\\rangle \\frac{dV}{d\\psi}$$
    where $\\langle 1/R^2 \\rangle$ is ``gm1``, $dV/d\\psi$ is ``dvolume_dpsi``, and $\\sigma$ is
    the orientation sign from Sauter Eq. 23: $\\sigma_q = \\sigma_{Ip} \\sigma_{B0} \\sigma_{\\rho\\theta\\varphi}$.
    When ``cocos`` indicates full Weber storage ($e_{B_p} = 1$, COCOS 11-18), the input fluxes are
    divided by $2\\pi$ to enter the contour tracing engine, and the resulting profile carries the
    exact COCOS-mandated sign.

    Processing steps
    ----------------
    1. Validate grid dimensions and identify the flux conversion factor from ``cocos``.
    2. Normalize or deduce ``psi_axis`` and ``psi_boundary``, converting to Wb/rad if necessary.
    3. Evaluate and interpolate ``f_profile`` onto the requested ``levels_norm``.
    4. Call :func:`flux_surface_quantities` to trace contours and compute geometric averages.
    5. Apply on-axis quadratic extrapolation in radius (linear in normalized flux $\\psi_N$) to
       resolve $q_0$.
    6. Attach the COCOS sign factor $\\sigma_q$ and package the output.

    Defaults
    --------
    ``levels_norm`` defaults to 65 points from 0 to 1 (numerical convenience).
    ``cocos=11`` matches the IMAS standard data dictionary convention.

    Applicability
    -------------
    Machine-independent. Works for any 2D tokamak poloidal flux map.

    Provenance
    ----------
    .. [1] O. Sauter and S. Yu. Medvedev, Comput. Phys. Commun. 184 (2013) 293,
           Eq. (17) and Table I for COCOS relations and safety factor definitions.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011), Sec. 3.4.
    """
    from vaft.data.cocos import cocos_spec

    if f_profile is None:
        raise ValueError("f_profile must be provided and cannot be None.")

    R_arr = np.asarray(R, dtype=float).reshape(-1)
    Z_arr = np.asarray(Z, dtype=float).reshape(-1)
    psi_arr = np.asarray(psi_grid, dtype=float)
    if psi_arr.shape != (R_arr.size, Z_arr.size):
        raise ValueError(
            f"psi_grid shape {psi_arr.shape} must equal (len(R), len(Z)) = {(R_arr.size, Z_arr.size)}."
        )

    if levels_norm is None:
        levels = np.linspace(0.0, 1.0, 65)
    else:
        levels = np.asarray(levels_norm, dtype=float).reshape(-1)

    spline = None
    if psi_axis is None:
        if axis_rz is not None:
            spline = RectBivariateSpline(R_arr, Z_arr, psi_arr)
            psi_axis = float(spline.ev(axis_rz[0], axis_rz[1]))
        else:
            edge_val = float(
                np.mean(
                    [
                        psi_arr[0, :],
                        psi_arr[-1, :],
                        psi_arr[:, 0],
                        psi_arr[:, -1],
                    ]
                )
            )
            diff = psi_arr - edge_val
            idx_max = np.unravel_index(np.argmax(np.abs(diff)), psi_arr.shape)
            psi_axis = float(psi_arr[idx_max])
            if axis_rz is None:
                axis_rz = (float(R_arr[idx_max[0]]), float(Z_arr[idx_max[1]]))

    if psi_boundary is None:
        if boundary is not None:
            if spline is None:
                spline = RectBivariateSpline(R_arr, Z_arr, psi_arr)
            r_b = np.asarray(boundary[0], dtype=float).reshape(-1)
            z_b = np.asarray(boundary[1], dtype=float).reshape(-1)
            psi_boundary = float(np.mean(spline.ev(r_b, z_b)))
        else:
            raise ValueError(
                "psi_boundary could not be deduced; provide psi_boundary or boundary outline."
            )

    if psi_axis == psi_boundary:
        raise ValueError("psi_axis and psi_boundary must differ to define normalized flux.")

    psi_levels = psi_axis + levels * (psi_boundary - psi_axis)

    if callable(f_profile):
        f_values = np.asarray([float(f_profile(p)) for p in psi_levels], dtype=float)
    elif isinstance(f_profile, (tuple, list)) and len(f_profile) == 2:
        psi_f = np.asarray(f_profile[0], dtype=float).reshape(-1)
        f_raw = np.asarray(f_profile[1], dtype=float).reshape(-1)
        if psi_f.size != f_raw.size:
            raise ValueError("f_profile tuple (psi, f) must have equal-length arrays.")
        order = np.argsort(psi_f)
        f_values = np.interp(psi_levels, psi_f[order], f_raw[order])
    elif np.ndim(f_profile) == 0:
        f_values = np.full(levels.size, float(f_profile))
    else:
        f_arr = np.asarray(f_profile, dtype=float).reshape(-1)
        if f_arr.size != levels.size:
            raise ValueError(
                f"f_profile length ({f_arr.size}) must match levels_norm length ({levels.size})."
            )
        f_values = f_arr

    spec = cocos_spec(cocos)
    scale_to_wb_per_rad = (1.0 / (2.0 * np.pi)) if spec.exp_bp == 1 else 1.0

    psi_grid_rad = psi_arr * scale_to_wb_per_rad
    psi_axis_rad = float(psi_axis) * scale_to_wb_per_rad
    psi_boundary_rad = float(psi_boundary) * scale_to_wb_per_rad

    surfaces = flux_surface_quantities(
        psi_grid=psi_grid_rad,
        R=R_arr,
        Z=Z_arr,
        psi_axis=psi_axis_rad,
        psi_boundary=psi_boundary_rad,
        levels_norm=levels,
        f_profile=f_values,
        axis_rz=axis_rz,
        boundary=boundary,
        min_points=min_points,
    )

    f_mean = float(np.nanmean(f_values)) if f_values.size else 0.0
    if sigma_b0 == 1 and f_mean < -1e-12:
        effective_sigma_b0 = -1
    else:
        effective_sigma_b0 = sigma_b0

    target_sign = spec.expected_sign("q", sigma_ip=sigma_ip, sigma_b0=effective_sigma_b0)
    q_final = target_sign * np.abs(surfaces["q"])
    surfaces["q"] = q_final

    if return_details:
        order_lvl = np.argsort(levels)
        lvl_sorted = levels[order_lvl]
        q_sorted = q_final[order_lvl]
        if (levels == 0.0).any():
            q_axis = float(q_final[levels == 0.0][0])
        elif len(lvl_sorted) >= 2:
            p1, p2 = lvl_sorted[0], lvl_sorted[1]
            q1, q2 = q_sorted[0], q_sorted[1]
            q_axis = float(q1 - (q2 - q1) / (p2 - p1) * p1) if p2 != p1 else float(q1)
        else:
            q_axis = float(q_sorted[0])
        q_95 = float(np.interp(0.95, lvl_sorted, q_sorted))
        return {
            "q": q_final,
            "levels_norm": levels,
            "psi_levels": psi_levels,
            "q_axis": q_axis,
            "q_95": q_95,
            "surfaces": surfaces,
        }
    return q_final


def equilibrium_field_on_grid(
    R_grid_1d: np.ndarray,
    Z_grid_1d: np.ndarray,
    psi_grid: np.ndarray,
    psi_1d: np.ndarray,
    f_1d: np.ndarray,
    cocos=None,
):
    """``(B_R, B_Z, B_phi)`` on the whole ``(R, Z)`` grid, each ``(nR, nZ)``.

    The vectorised twin of :func:`make_equilibrium_field_interpolator`, for a
    caller that wants the field everywhere rather than at a point: the same
    bicubic psi spline, the same Sauter Eq. 20 prefactor, and the same
    ``F(psi)/R`` with ``F`` clipped to the profile's own range outside the
    confined region.  Evaluating the point interpolator over a 129x129 grid
    would be sixteen thousand Python calls; this is one spline evaluation.

    Parameters
    ----------
    R_grid_1d, Z_grid_1d : array_like
        Grid axes [m].
    psi_grid : array_like
        Poloidal flux on ``(len(R), len(Z))``, in the convention ``cocos``
        describes [Wb or Wb/rad].
    psi_1d, f_1d : array_like
        ``profiles_1d.psi`` and ``profiles_1d.f`` [same psi unit; T m].
    cocos : int or None, optional
        COCOS index of *psi_grid*. ``None`` keeps the historical
        weber-per-radian form [-].

    Returns
    -------
    tuple of numpy.ndarray
        ``(B_R, B_Z, B_phi)``, each shaped ``(len(R), len(Z))`` [T].

    Raises
    ------
    ValueError
        The flux map is not shaped to the two grid axes, or the two profile
        arrays have different lengths.

    Convention
    ----------
    The same prefactor as :func:`make_equilibrium_field_interpolator`, per
    Sauter Eq. 20: ``B_R = k (1/R) dpsi/dZ`` and ``B_Z = -k (1/R) dpsi/dR``
    with ``k = sigma_RphiZ * sigma_Bp / (2*pi)**e_Bp``, so *psi_grid* must be
    stored in the convention *cocos* names and a weber-stored flux can be
    corrected only through that index. The toroidal field is the poloidal
    current function over the major radius and inherits the sign of *f_1d*.
    The flux map is indexed major radius first.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Outside the confined region the poloidal current function clips to its
    nearest edge value, the clip-and-interpolate convention
    :func:`psi_to_rz` uses, so the toroidal field there is that clipped
    function over the major radius rather than the true vacuum field. The
    spline extrapolates beyond the grid, where the field should not be read.

    Provenance
    ----------
    .. [1] Sauter and Medvedev (2013), Eq. 20, for the prefactor.
    .. [2] The point-wise twin :func:`make_equilibrium_field_interpolator` in
       this module, which this routine is pinned to by test.
    """
    from vaft.formula.equilibrium import poloidal_field_factor

    R_grid_1d = np.asarray(R_grid_1d, dtype=float).reshape(-1)
    Z_grid_1d = np.asarray(Z_grid_1d, dtype=float).reshape(-1)
    psi_grid = np.asarray(psi_grid, dtype=float)
    if psi_grid.shape != (R_grid_1d.size, Z_grid_1d.size):
        raise ValueError(
            f"psi_grid shape {psi_grid.shape} must equal "
            f"(len(R_grid_1d), len(Z_grid_1d)) = {(R_grid_1d.size, Z_grid_1d.size)}."
        )
    psi_1d = np.asarray(psi_1d, dtype=float).reshape(-1)
    f_1d = np.asarray(f_1d, dtype=float).reshape(-1)
    if psi_1d.size != f_1d.size:
        raise ValueError("psi_1d and f_1d must have the same length.")

    order = np.argsort(psi_1d)
    psi_sorted, f_sorted = psi_1d[order], f_1d[order]
    spline = RectBivariateSpline(R_grid_1d, Z_grid_1d, psi_grid)
    grid_r = R_grid_1d[:, None] * np.ones_like(Z_grid_1d)[None, :]
    k = poloidal_field_factor(cocos)

    dpsi_dr = spline(R_grid_1d, Z_grid_1d, dx=1, dy=0)
    dpsi_dz = spline(R_grid_1d, Z_grid_1d, dx=0, dy=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        b_r = k * dpsi_dz / grid_r
        b_z = -k * dpsi_dr / grid_r
        psi_here = np.clip(spline(R_grid_1d, Z_grid_1d), psi_sorted[0], psi_sorted[-1])
        b_tor = np.interp(psi_here, psi_sorted, f_sorted) / grid_r
    return b_r, b_z, b_tor


def make_equilibrium_field_interpolator(
    R_grid_1d: np.ndarray,
    Z_grid_1d: np.ndarray,
    psi_grid: np.ndarray,
    psi_1d: np.ndarray,
    f_1d: np.ndarray,
    cocos=None,
):
    """Build a callable giving the full magnetic field anywhere on one time slice.

    Parameters
    ----------
    R_grid_1d : array_like
        Major-radius grid axis [m].
    Z_grid_1d : array_like
        Height grid axis [m].
    psi_grid : array_like
        Poloidal flux on the grid, indexed ``(R, Z)`` [Wb/rad].
    psi_1d : array_like
        Flux abscissa the poloidal current function is given on [Wb/rad].
    f_1d : array_like
        Poloidal current function ``F = R*B_phi`` on that abscissa [T m].
    cocos : int, optional
        COCOS index of *psi_grid*. ``None`` keeps the historical
        weber-per-radian form [-].

    Returns
    -------
    callable
        A function of ``(R, Z)`` returning ``(B_R, B_Z, B_phi)`` [T].

    Raises
    ------
    ValueError
        The flux map is not shaped to the two grid axes.

    Convention
    ----------
    Same prefactor as :func:`poloidal_field_at_boundary`, per Sauter Eq. 20:
    ``B_R = k (1/R) dpsi/dZ`` and ``B_Z = -k (1/R) dpsi/dR``, with
    ``k = sigma_RphiZ * sigma_Bp / (2*pi)**e_Bp``. The toroidal field is the
    poloidal current function over the major radius. The flux map is indexed major
    radius first, matching :func:`extract_flux_surface_contours`.

    Unlike :func:`poloidal_field_at_boundary`, this takes **only** the COCOS index
    and no separate per-radian flag, so a weber-stored flux can be corrected here
    only through the index. Supplying neither leaves the field too large by
    ``2*pi``.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The poloidal current function is defined only from axis to boundary. Points
    outside that range, in the scrape-off layer, clip to the nearest edge value,
    the same clip-and-interpolate convention :func:`psi_to_rz` uses. That is an
    approximation for a field line that leaves the confined region: the toroidal
    field there is this clipped function over the major radius rather than the
    true vacuum field. The spline is built once over the whole map and reused, so
    evaluation is cheap but the map must not change.

    Provenance
    ----------
    .. [1] Sauter and Medvedev (2013), Eq. 20, for the prefactor.
    .. [2] The clip-and-interpolate convention of :func:`psi_to_rz`, kept
       deliberately the same.
    """
    R_grid_1d = np.asarray(R_grid_1d, dtype=float).reshape(-1)
    Z_grid_1d = np.asarray(Z_grid_1d, dtype=float).reshape(-1)
    psi_grid = np.asarray(psi_grid, dtype=float)
    if psi_grid.shape != (R_grid_1d.size, Z_grid_1d.size):
        raise ValueError(
            f"psi_grid shape {psi_grid.shape} must equal "
            f"(len(R_grid_1d), len(Z_grid_1d)) = {(R_grid_1d.size, Z_grid_1d.size)}."
        )

    psi_1d = np.asarray(psi_1d, dtype=float).reshape(-1)
    f_1d = np.asarray(f_1d, dtype=float).reshape(-1)
    if psi_1d.size != f_1d.size:
        raise ValueError("psi_1d and f_1d must have the same length.")
    sort_idx = np.argsort(psi_1d)
    psi_1d_sorted = psi_1d[sort_idx]
    f_1d_sorted = f_1d[sort_idx]

    psi_spline = RectBivariateSpline(R_grid_1d, Z_grid_1d, psi_grid)

    from vaft.formula.equilibrium import poloidal_field_factor

    k = poloidal_field_factor(cocos)

    def b_field(R: float, Z: float) -> tuple[float, float, float]:
        dpsi_dR = float(psi_spline.ev(R, Z, dx=1, dy=0))
        dpsi_dZ = float(psi_spline.ev(R, Z, dx=0, dy=1))
        B_R = k * (1.0 / R) * dpsi_dZ
        B_Z = -k * (1.0 / R) * dpsi_dR

        psi_here = float(psi_spline.ev(R, Z))
        psi_clipped = np.clip(psi_here, psi_1d_sorted[0], psi_1d_sorted[-1])
        F = float(np.interp(psi_clipped, psi_1d_sorted, f_1d_sorted))
        B_phi = F / R

        return B_R, B_Z, B_phi

    return b_field


def trace_field_line(
    R0: float,
    Z0: float,
    phi0: float,
    b_field,
    *,
    dphi: float = np.deg2rad(1.0),
    max_length_m: float = 50.0,
    direction: str = "forward",
    wall_r: np.ndarray | None = None,
    wall_z: np.ndarray | None = None,
    r_bounds: tuple[float, float] | None = None,
    z_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Trace one magnetic field line through the equilibrium field.

    Parameters
    ----------
    R0 : float
        Starting major radius [m].
    Z0 : float
        Starting height [m].
    phi0 : float
        Starting toroidal angle [rad].
    b_field : callable
        A function of ``(R, Z)`` returning ``(B_R, B_Z, B_phi)``, normally from
        :func:`make_equilibrium_field_interpolator` [-].
    dphi : float, optional
        Fixed step in toroidal angle [rad].
    max_length_m : float, optional
        Path length at which to stop [m].
    direction : str, optional
        ``"forward"``, ``"backward"``, or ``"both"`` for the two branches joined
        at the start point [-].
    wall_r : array_like, optional
        Major radius of the limiting polygon [m].
    wall_z : array_like, optional
        Height of the limiting polygon [m].
    r_bounds : tuple of float, optional
        Major-radius range outside which to stop [m].
    z_bounds : tuple of float, optional
        Height range outside which to stop [m].

    Returns
    -------
    dict of str to Any
        ``phi`` in radians, ``R`` and ``Z`` in metres, ``arc_length_m`` in metres,
        and ``termination_reason`` as a string [-].

    Processing steps
    ----------------
    1. Integrate ``dR/dphi = R*B_R/B_phi`` and ``dZ/dphi = R*B_Z/B_phi`` by
       fixed-step fourth-order Runge-Kutta in the toroidal angle.
    2. After each step, test the termination conditions in order: leaving the
       limiting polygon, exceeding the path length, leaving the coordinate
       bounds, or a numerically zero toroidal field.
    3. For ``"both"``, trace each branch and concatenate them anchored at the
       start.

    Convention
    ----------
    The toroidal angle is the integration variable, not path length or time,
    which is why a vanishing toroidal field terminates the trace rather than
    slowing it. Points are always returned in order of increasing angle whatever
    the direction traced. The arc length is cumulative along the returned array
    from zero at its first point, so for ``"both"`` it measures from the backward
    end and not from the start point.

    Defaults
    --------
    The one-degree step and the fifty-metre limit are numerical conveniences:
    fine enough to resolve a tokamak field line's pitch, and long enough to reach
    a wall from anywhere inside a small machine.

    Applicability
    -------------
    Machine-independent. The wall polygon is where a machine's own geometry
    enters, and it is supplied by the caller.

    Limitations
    -----------
    Fixed step, so accuracy is set by the step alone and there is no error
    control; a line near a separatrix, where the pitch changes fastest, is the
    least accurate. Terminating on the coordinate bounds is a safety net against
    extrapolating far outside the equilibrium grid, not a physical condition.

    Provenance
    ----------
    .. [1] The field-line equations in toroidal-angle form; the field itself comes
       from :func:`make_equilibrium_field_interpolator`.
    """
    if direction not in ("forward", "backward", "both"):
        raise ValueError("direction must be 'forward', 'backward', or 'both'.")
    if dphi <= 0:
        raise ValueError("dphi must be positive.")

    from matplotlib.path import Path as MplPath

    wall_polygon = None
    if wall_r is not None and wall_z is not None:
        wall_polygon = MplPath(np.column_stack([np.asarray(wall_r, float), np.asarray(wall_z, float)]))

    def _derivative(R: float, Z: float) -> tuple[float, float]:
        B_R, B_Z, B_phi = b_field(R, Z)
        if B_phi == 0.0:
            raise FloatingPointError("B_phi is zero; cannot parameterize the field line by phi here.")
        return R * B_R / B_phi, R * B_Z / B_phi

    def _rk4_step(R: float, Z: float, step: float) -> tuple[float, float]:
        k1_R, k1_Z = _derivative(R, Z)
        k2_R, k2_Z = _derivative(R + 0.5 * step * k1_R, Z + 0.5 * step * k1_Z)
        k3_R, k3_Z = _derivative(R + 0.5 * step * k2_R, Z + 0.5 * step * k2_Z)
        k4_R, k4_Z = _derivative(R + step * k3_R, Z + step * k3_Z)
        R_next = R + (step / 6.0) * (k1_R + 2.0 * k2_R + 2.0 * k3_R + k4_R)
        Z_next = Z + (step / 6.0) * (k1_Z + 2.0 * k2_Z + 2.0 * k3_Z + k4_Z)
        return R_next, Z_next

    def _in_bounds(R: float, Z: float) -> bool:
        if r_bounds is not None and not (r_bounds[0] <= R <= r_bounds[1]):
            return False
        if z_bounds is not None and not (z_bounds[0] <= Z <= z_bounds[1]):
            return False
        if wall_polygon is not None and not wall_polygon.contains_point((R, Z)):
            return False
        return True

    def _run_branch(step: float) -> tuple[list[float], list[float], list[float], list[float], str]:
        phi_list = [phi0]
        R_list = [R0]
        Z_list = [Z0]
        arc_list = [0.0]
        reason = "max_length_m"
        R, Z, phi, arc = R0, Z0, phi0, 0.0
        while arc < max_length_m:
            try:
                R_next, Z_next = _rk4_step(R, Z, step)
            except FloatingPointError:
                reason = "b_phi_zero"
                break
            if not _in_bounds(R_next, Z_next):
                reason = "wall" if wall_polygon is not None and not wall_polygon.contains_point((R_next, Z_next)) else "out_of_bounds"
                break
            phi_next = phi + step
            d_arc = np.sqrt(
                (R_next - R) ** 2 + (Z_next - Z) ** 2 + ((R + R_next) / 2.0 * step) ** 2
            )
            arc_next = arc + d_arc
            if arc_next > max_length_m:
                reason = "max_length_m"
                break
            R, Z, phi, arc = R_next, Z_next, phi_next, arc_next
            phi_list.append(phi)
            R_list.append(R)
            Z_list.append(Z)
            arc_list.append(arc)
        return phi_list, R_list, Z_list, arc_list, reason

    def _cumulative_arc_length(phi: list[float], R: list[float], Z: list[float]) -> np.ndarray:
        """Recompute a monotonically increasing cumulative arc length for an
        assembled (phi, R, Z) sequence, in its given order.

        Each branch's own ``arc_list`` measures distance *from phi0*, so for
        ``"backward"`` (reversed into increasing-phi order) or ``"both"``
        (two branches concatenated around the shared phi0 point) simply
        reusing those values gives a non-monotonic result -- decreasing
        toward phi0, then increasing again -- rather than a running total
        along the returned array. Recomputing from consecutive-point
        distances (the same formula used while stepping) makes it monotonic
        and consistent with the documented "cumulative" contract regardless
        of ``direction``.
        """
        phi_arr = np.asarray(phi, dtype=float)
        R_arr = np.asarray(R, dtype=float)
        Z_arr = np.asarray(Z, dtype=float)
        if phi_arr.size == 0:
            return np.asarray([], dtype=float)
        d_phi = np.diff(phi_arr)
        d_R = np.diff(R_arr)
        d_Z = np.diff(Z_arr)
        R_avg = (R_arr[:-1] + R_arr[1:]) / 2.0
        segment_lengths = np.sqrt(d_R**2 + d_Z**2 + (R_avg * d_phi) ** 2)
        return np.concatenate([[0.0], np.cumsum(segment_lengths)])

    if direction in ("forward", "both"):
        phi_f, R_f, Z_f, arc_f, reason_f = _run_branch(dphi)
    if direction in ("backward", "both"):
        phi_b, R_b, Z_b, arc_b, reason_b = _run_branch(-dphi)

    if direction == "forward":
        phi_all, R_all, Z_all, reason = phi_f, R_f, Z_f, reason_f
    elif direction == "backward":
        phi_all = list(reversed(phi_b))
        R_all = list(reversed(R_b))
        Z_all = list(reversed(Z_b))
        reason = reason_b
    else:
        phi_all = list(reversed(phi_b[1:])) + phi_f
        R_all = list(reversed(R_b[1:])) + R_f
        Z_all = list(reversed(Z_b[1:])) + Z_f
        reason = f"backward:{reason_b}, forward:{reason_f}"

    return {
        "phi": np.asarray(phi_all, dtype=float),
        "R": np.asarray(R_all, dtype=float),
        "Z": np.asarray(Z_all, dtype=float),
        "arc_length_m": _cumulative_arc_length(phi_all, R_all, Z_all),
        "termination_reason": reason,
    }


# Parametric APIs are implemented separately while retaining this module as
# their stable public import location.  The absolute fallback preserves a
# historical test/tool pattern that loads this source file directly by path.
try:  # pragma: no branch - normal package import takes this path
    from ._equilibrium_parametric import *  # noqa: E402,F401,F403
except ImportError:  # direct ``spec_from_file_location`` loading
    from vaft.process._equilibrium_parametric import *  # noqa: E402,F401,F403


def make_vacuum_field_interpolator(
    R_grid_1d: np.ndarray,
    Z_grid_1d: np.ndarray,
    b_r: np.ndarray,
    b_z: np.ndarray,
    r0_b0: float,
):
    r"""Build a callable giving the vacuum magnetic field anywhere on one grid.

    The twin of :func:`make_equilibrium_field_interpolator` for a machine with no
    plasma in it.  That one reads the poloidal field off a flux map and the
    toroidal field off ``profiles_1d.f``, both of which a reconstruction
    supplies; before breakdown there is no reconstruction, and the field the
    coils and the vessel make is all there is.

    Parameters
    ----------
    R_grid_1d : np.ndarray
        Major radius of the grid columns, increasing [m].
    Z_grid_1d : np.ndarray
        Height of the grid rows, increasing [m].
    b_r : np.ndarray
        Radial field on the ``(len(R), len(Z))`` grid [T].
    b_z : np.ndarray
        Vertical field on the same grid [T].
    r0_b0 : float
        The vacuum toroidal field product $R_0 B_0$, as
        ``tf.b_field_tor_vacuum_r`` stores it [T m].

    Returns
    -------
    callable
        A function of ``(R, Z)`` returning ``(B_R, B_Z, B_phi)``, accepting
        scalars or arrays of matching shape [T].

    Raises
    ------
    ValueError
        Grids and field arrays whose shapes disagree [-].

    Convention
    ----------
    The toroidal field is $B_\varphi = R_0 B_0 / R$ exactly, not interpolated:
    in a vacuum it is a pure $1/R$ and the grid would only add error to it.  The
    poloidal components are interpolated, so they carry the grid's resolution --
    which is the caller's choice, not this function's.

    Assumptions
    -----------
    The grid is the one :func:`vaft.omas.compute_vacuum_field_map` builds, whose
    axes are monotonic and whose fields already include the vessel's eddy
    currents.  Nothing here checks that the field is curl-free.

    Applicability
    -------------
    Machine-independent.  The grid and the toroidal product are the caller's.

    Limitations
    -----------
    Bicubic over the supplied grid, so the field between nodes is as good as the
    grid is fine: poor within a coil's near field, where the true field varies on
    the scale of the conductor, and good near a null, where it varies on the
    scale of the machine.  Outside the grid the spline extrapolates rather than
    raising, so a caller that can leave the grid must terminate on its own bound.

    Provenance
    ----------
    .. [1] The vacuum field decomposition: the poloidal part from the Green's
       function response of the coils and the passive structure, the toroidal
       part from the TF product.
    """
    R_grid_1d = np.asarray(R_grid_1d, dtype=float)
    Z_grid_1d = np.asarray(Z_grid_1d, dtype=float)
    b_r = np.asarray(b_r, dtype=float)
    b_z = np.asarray(b_z, dtype=float)
    expected = (R_grid_1d.size, Z_grid_1d.size)
    if b_r.shape != expected or b_z.shape != expected:
        raise ValueError(
            f"b_r and b_z must both be {expected}; got {b_r.shape} and {b_z.shape}."
        )

    spline_b_r = RectBivariateSpline(R_grid_1d, Z_grid_1d, b_r)
    spline_b_z = RectBivariateSpline(R_grid_1d, Z_grid_1d, b_z)
    product = float(r0_b0)

    def b_field(R, Z):
        return spline_b_r.ev(R, Z), spline_b_z.ev(R, Z), product / np.asarray(R, dtype=float)

    return b_field


def connection_length_map(
    seed_r: np.ndarray,
    seed_z: np.ndarray,
    b_field,
    *,
    wall_r: np.ndarray,
    wall_z: np.ndarray,
    phi0: float = 0.0,
    dphi: float = np.deg2rad(2.0),
    max_length_m: float = 150.0,
) -> dict[str, Any]:
    r"""Connection length of the field line through each of many starting points.

    How far a field line runs in both directions before it strikes the wall,
    which is what decides whether an electron accelerating along it avalanches
    or is lost.  Every line is stepped together rather than one at a time, so a
    whole grid costs one pass instead of one pass per point.

    Parameters
    ----------
    seed_r : np.ndarray
        Major radius of each starting point [m].
    seed_z : np.ndarray
        Height of each starting point, broadcastable against ``seed_r`` [m].
    b_field : callable
        A function of ``(R, Z)`` returning ``(B_R, B_Z, B_phi)`` for arrays,
        normally from :func:`make_vacuum_field_interpolator` [T].
    wall_r : np.ndarray
        Major radius of the limiting polygon [m].
    wall_z : np.ndarray
        Height of the limiting polygon [m].
    phi0 : float, optional
        Starting toroidal angle, the same for every line [rad].
    dphi : float, optional
        Fixed step in toroidal angle [rad].
    max_length_m : float, optional
        Path length at which to stop each direction [m].

    Returns
    -------
    dict of str to np.ndarray
        ``length_m`` the two directions summed, with ``nan`` at points outside
        the wall and wherever a step left the field model -- a callable that
        returned a non-finite field -- since nothing is then known about where
        that line ends; ``forward_m`` and ``backward_m`` the branches; ``saturated``
        true where either branch stopped on ``max_length_m`` rather than on the
        wall; ``outside`` true where the point itself is not inside the wall.
        Every array has the shape of ``seed_r`` [m].

    Raises
    ------
    ValueError
        A non-positive ``dphi`` or ``max_length_m``, or seed arrays whose shapes
        do not match [-].

    Processing steps
    ----------------
    1. Drop the seeds that are already outside the limiting polygon; they have
       no connection length and are reported as ``nan``.
    2. Step every surviving line together by fourth-order Runge-Kutta in the
       toroidal angle, once forward and once backward, retiring a line as it
       leaves the polygon or reaches ``max_length_m``.
    3. Add the two branches. A line retired on length contributes
       ``max_length_m`` and is flagged rather than being read as a wall hit.

    Defaults
    --------
    The two-degree step is a numerical convenience, chosen by convergence
    against a half-degree trace on VEST's vacuum field: the median length moves
    by 7e-5 and the 95th percentile by 4e-3 of itself.  The 150 m limit is an
    assumed value, long enough that a Lloyd threshold evaluated on it has
    already saturated and short enough that a line circulating near a null
    terminates.

    Convention
    ----------
    The toroidal angle is the integration variable, as in
    :func:`trace_field_line`, so a vanishing toroidal field ends a line rather
    than slowing it.  The length is the **sum of both directions** from the
    seed, which is the quantity a Townsend avalanche sees; a caller wanting one
    branch should read ``forward_m`` or ``backward_m``.  A saturated line
    reports ``max_length_m``, not infinity, and ``saturated`` is how a consumer
    tells the difference.

    Applicability
    -------------
    Machine-independent.  The wall polygon and the field are both the caller's.

    Limitations
    -----------
    The wall test is applied once per step, so a line ends at the last point
    inside the polygon rather than at the true crossing -- of order
    $R\,\mathrm{d}\varphi$, millimetres against the tens of metres the length
    itself runs to.  Fixed step, so accuracy is set by the step alone and there
    is no error control.  A field line that closes on itself never strikes the
    wall, and this returns ``max_length_m`` for it, flagged; distinguishing a
    closed line from a merely long one is not attempted.

    Provenance
    ----------
    .. [1] The field-line equations in toroidal-angle form, as in
       :func:`trace_field_line`, stepped over many seeds at once.
    .. [2] B. Lloyd et al., Nucl. Fusion 31 (1991) 2031, Sec. 2, for the role the
       connection length plays in the breakdown threshold this feeds.
    """
    from matplotlib.path import Path as MplPath

    if dphi <= 0:
        raise ValueError("dphi must be positive.")
    if max_length_m <= 0:
        raise ValueError("max_length_m must be positive.")
    seed_r = np.asarray(seed_r, dtype=float)
    seed_z = np.asarray(seed_z, dtype=float)
    if seed_r.shape != seed_z.shape:
        raise ValueError(
            f"seed_r and seed_z must have the same shape; got {seed_r.shape} and {seed_z.shape}."
        )

    shape = seed_r.shape
    flat_r = seed_r.ravel()
    flat_z = seed_z.ravel()
    polygon = MplPath(np.column_stack([np.asarray(wall_r, float), np.asarray(wall_z, float)]))
    inside = polygon.contains_points(np.column_stack([flat_r, flat_z]))

    branches = {}
    saturated = np.zeros(flat_r.size, dtype=bool)
    for name, sign in (("forward_m", 1.0), ("backward_m", -1.0)):
        length, hit_limit = _trace_branch_lengths(
            flat_r, flat_z, inside, b_field, polygon, sign * dphi, max_length_m
        )
        branches[name] = length
        saturated |= hit_limit

    total = branches["forward_m"] + branches["backward_m"]
    total[~inside] = np.nan
    for name in branches:
        branches[name][~inside] = np.nan
    return {
        "length_m": total.reshape(shape),
        "forward_m": branches["forward_m"].reshape(shape),
        "backward_m": branches["backward_m"].reshape(shape),
        "saturated": (saturated & inside).reshape(shape),
        "outside": (~inside).reshape(shape),
    }


def ejiri_mirror_geometry(
    r_start: float,
    b_field,
    *,
    wall_r: np.ndarray,
    wall_z: np.ndarray,
    z_fit: float | None = None,
    dphi: float = np.deg2rad(1.0),
    max_length_m: float = 150.0,
) -> dict[str, Any]:
    r"""Ejiri mirror-confinement proxy for one magnetic snapshot.

    Traces the field line through $(R_S, 0)$ and reads off the four lengths the
    Ejiri low-energy orbit model needs -- the starting radius, the inboard
    limiter, the curvature radius and the vertical extent over which the line
    keeps raising $|B|$ -- then evaluates the boundary slope $\alpha$ and the
    geometry factor $F_3$.

    Parameters
    ----------
    r_start : float
        Major radius the electron starts at, on the midplane [m].
    b_field : callable
        A function of ``(R, Z)`` returning ``(B_R, B_Z, B_phi)``, normally from
        :func:`make_vacuum_field_interpolator` [T].
    wall_r : np.ndarray
        Major radius of the limiting polygon [m].
    wall_z : np.ndarray
        Height of the limiting polygon [m].
    z_fit : float, optional
        Half-height of the window for the diagnostic local parabola fit;
        default a quarter of ``z_max`` [m].
    dphi : float, optional
        Fixed step in toroidal angle for the trace [rad].
    max_length_m : float, optional
        Path length at which to stop each branch [m].

    Returns
    -------
    dict of str to Any
        ``r_start``, ``r_inboard_limiter``, ``curvature_radius`` and ``z_max`` in
        metres; ``alpha`` and ``f3`` dimensionless; ``mirror`` true when the line
        dips inward before it ends; ``saturated`` true when a branch stopped on
        ``max_length_m`` rather than on the wall or a turning point;
        ``binding_branch`` naming the weaker mirror; ``curvature_radius_local``
        the local parabola fit in metres, for comparison only; per-branch
        ``z_max_upper``, ``z_max_lower``, ``r_mirror_upper``, ``r_mirror_lower``
        in metres with ``reason_upper`` and ``reason_lower``; and the two
        :func:`trace_field_line` branches as ``trace_upper`` and
        ``trace_lower`` [-].

    Raises
    ------
    ValueError
        A start point outside the wall, no wall crossing of the midplane inboard
        of it, or a non-positive ``z_fit`` [-].

    Processing steps
    ----------------
    1. Trace the line through $(R_S, 0)$ forward and backward against the wall,
       and label the branch that rises the upper one.
    2. $R_{\mathrm{LIN}}$: the wall polygon's midplane crossing nearest the start
       on its inboard side.
    3. Mirror point of each branch: the smallest $R$ the branch reaches before
       it ends, excluding the seed -- the strongest field an electron escaping
       that way has to pass.  Record $|Z|$ and $R$ there.
    4. Keep the **weaker** mirror: the branch whose mirror point sits at the
       larger $R$, so the smaller field rise.  Its $|Z|$ is $Z_{\max}$ and its
       $R$ is $R_m$.  If $R_m \ge R_S$ the line never dips inward.
    5. $R_C = Z_{\max}^2 / \bigl(2(R_S - R_m)\bigr)$, the parabola through the
       start and the mirror point, then $\alpha$ and $F_3$ from
       :func:`vaft.formula.startup.ejiri_mirror_alpha_from_R_S_R_LIN_R_C_Z_max`
       and :func:`vaft.formula.startup.ejiri_f3_from_alpha`.
    6. For comparison, fit $R - R_S = c_1 Z + c_2 Z^2$ over $|Z| \le$
       ``z_fit`` and report $-1/(2c_2)$ as ``curvature_radius_local``.

    Defaults
    --------
    The 150 m limit is a numerical convenience matching
    :func:`connection_length_map`, and it is where the result converges on VEST:
    at the breakdown onset of the packaged shot, from the electron-cyclotron
    resonance, 50 m stops both branches short and gives $Z_{\max} = 0.13$ m and
    $F_3 = 0.38$, while 150, 500 and 1500 m all give $0.30$ m and $0.51$.  The
    quarter-$Z_{\max}$ window and the one-degree step are numerical
    conveniences: at 2, 1, 0.5 and 0.25 degrees $F_3$ agrees to six figures and
    $R_C$ and $Z_{\max}$ to four.

    Convention
    ----------
    **$R_C$ is the secant through the mirror point, not a local fit.**  Ejiri's
    curvature term needs $R_C$ only through $Z_{\max}^2/2R_C = R_S - R_m$, so
    defining $R_C$ that way makes the term exactly $1/\sqrt{M - 1}$ for the
    line's true mirror ratio $M = R_S/R_m$ -- the inboard term with the limiter
    replaced by the line's own mirror point -- whatever shape the line has.  On
    an exact parabola it equals the parabola's $R_C$.  A local fit, which is
    what the model's derivation suggests, is not stable on a real field: on the
    same VEST case it gives $R_C$ from 0.07 to 0.34 m as the window widens from
    a tenth of $Z_{\max}$ to all of it, and had it fed $\alpha$, $F_3$ would
    have run from 0.99 to 0.62 on that choice alone.  The secant is 0.52 m for
    every window.  ``curvature_radius_local`` still reports the fit, so how far
    the line is from Ejiri's parabola stays visible.

    **The mirror point is the global minimum of $R$ along the branch, not the
    first local one.**  An escaping electron has to pass the largest field on
    its way to the wall, wherever it is.  The distinction is not academic: at
    the breakdown onset of VEST's packaged shot the line through the
    electron-cyclotron resonance is tilted at the midplane, so on one side $R$
    rises for a few steps before dipping to 0.60 m.  A first-local-minimum rule
    gives up on that side, reads the wall end at 0.76 m as the mirror point,
    and reports no confinement at all.

    **The weaker mirror decides.**  An electron bouncing between the two mirror
    points escapes through the one with the smaller field rise, which is the
    one at larger $R$, not necessarily the one at smaller $|Z|$.

    **No inward dip means nothing is confined, not an error.**  When $R_m \ge
    R_S$ the line is straight or bows outward, $|B|$ does not rise away from the
    midplane, and curvature traps nothing: this returns $\alpha = \infty$,
    $F_3 = 0$ and ``mirror`` false without calling the formulas, which reject a
    non-positive curvature radius.

    The mirror ratio is read in $R$, which assumes $|B| \propto 1/R$: true when
    the toroidal field dominates, as it does in a pre-breakdown vacuum field.

    Applicability
    -------------
    Machine-independent.  The wall polygon and the field are both the caller's,
    and so is ``r_start`` -- normally the electron-cyclotron resonance, from
    :func:`vaft.formula.startup.electron_cyclotron_resonance_radius`.

    Limitations
    -----------
    An Ejiri-inspired geometric proxy, not the numerical orbit boundary: it
    says nothing about EC power, collisions or breakdown itself.  A branch whose
    first step already leaves the wall is read as an immediate loss on that
    side, so a start point within one step of the wall reports no confinement
    -- shorten ``dphi`` if that is not the answer wanted.  A branch that
    stops on ``max_length_m`` sets ``saturated`` and raises a
    ``RuntimeWarning``: that result is not converged and is not a bound, since a
    longer trace can move both the mirror point and the branch that binds.

    Provenance
    ----------
    .. [1] A. Ejiri and Y. Takase, Nucl. Fusion 47 (2007) 403, Sec. 3, for the
       orbit-boundary slope and the geometry factor.
    .. [2] The trace is :func:`trace_field_line`, and the two relations are the
       :mod:`vaft.formula.startup` kernels named in the processing steps.
    """
    return _ejiri_mirror_geometry(
        r_start,
        b_field,
        wall_r=wall_r,
        wall_z=wall_z,
        z_fit=z_fit,
        dphi=dphi,
        max_length_m=max_length_m,
    )


def _ejiri_mirror_geometry(r_start, b_field, *, wall_r, wall_z, z_fit, dphi, max_length_m):
    """The body of :func:`ejiri_mirror_geometry`.

    Kept private so both public entry points -- that function and
    :func:`vaft.omas.compute_ejiri_mirror_proxy_ods` -- call it directly and the
    saturation warning, raised two frames down, always blames their caller.
    """
    from matplotlib.path import Path as MplPath

    from vaft.formula.startup import (
        ejiri_f3_from_alpha,
        ejiri_mirror_alpha_from_R_S_R_LIN_R_C_Z_max,
    )

    wall_r = np.asarray(wall_r, dtype=float)
    wall_z = np.asarray(wall_z, dtype=float)
    r_start = float(r_start)
    if z_fit is not None and (not np.isfinite(z_fit) or z_fit <= 0.0):
        raise ValueError(f"z_fit must be finite and positive; got {z_fit} m")
    polygon = MplPath(np.column_stack([wall_r, wall_z]))
    if not polygon.contains_point((r_start, 0.0)):
        raise ValueError(
            f"r_start={r_start} m is not inside the wall on the midplane; there is "
            "no field line to trace"
        )
    r_inboard = _midplane_crossing_inboard(wall_r, wall_z, r_start)

    branches = {}
    for direction in ("forward", "backward"):
        trace = trace_field_line(
            r_start,
            0.0,
            0.0,
            b_field,
            dphi=dphi,
            max_length_m=max_length_m,
            direction=direction,
            wall_r=wall_r,
            wall_z=wall_z,
        )
        R = np.asarray(trace["R"], dtype=float)
        Z = np.asarray(trace["Z"], dtype=float)
        if direction == "backward":
            # trace_field_line returns points in order of increasing angle, so
            # the backward branch ends at the seed; walk it from the seed.
            R, Z = R[::-1], Z[::-1]
        z_end, r_end, reason = _mirror_point(R, Z, trace["termination_reason"])
        branches[direction] = {
            "R": R, "Z": Z, "trace": trace,
            "z_end": z_end, "r_end": r_end, "reason": reason,
            # A one-point branch left the wall on its first step and has no
            # direction of its own; it sorts below any branch that moved.
            "height": float(np.median(Z[1:])) if Z.size > 1 else 0.0,
        }
    upper_key = max(branches, key=lambda key: branches[key]["height"])
    lower_key = "backward" if upper_key == "forward" else "forward"
    branches = {"upper": branches[upper_key], "lower": branches[lower_key]}

    saturated = any(b["reason"] == "max_length_m" for b in branches.values())
    if saturated:
        warnings.warn(
            f"a field line from r_start={r_start} m reached max_length_m="
            f"{max_length_m} m before the wall or a turning point, so the mirror "
            "geometry is not converged; raise max_length_m",
            RuntimeWarning,
            stacklevel=3,
        )

    binding = max(branches, key=lambda name: branches[name]["r_end"])
    z_max = branches[binding]["z_end"]
    r_mirror = branches[binding]["r_end"]

    local = np.nan
    window_half = z_fit if z_fit is not None else 0.25 * z_max
    if window_half > 0.0:
        all_R = np.concatenate([b["R"] for b in branches.values()])
        all_Z = np.concatenate([b["Z"] for b in branches.values()])
        window = np.abs(all_Z) <= window_half
        if np.count_nonzero(window) >= 5:
            design = np.column_stack([all_Z[window], all_Z[window] ** 2])
            (_, c2), *_ = np.linalg.lstsq(design, all_R[window] - r_start, rcond=None)
            local = -1.0 / (2.0 * c2) if c2 < 0.0 else np.inf

    result = {
        "r_start": r_start,
        "r_inboard_limiter": r_inboard,
        "z_max": z_max,
        "saturated": saturated,
        "binding_branch": binding,
        "curvature_radius_local": local,
        "z_max_upper": branches["upper"]["z_end"],
        "z_max_lower": branches["lower"]["z_end"],
        "r_mirror_upper": branches["upper"]["r_end"],
        "r_mirror_lower": branches["lower"]["r_end"],
        "reason_upper": branches["upper"]["reason"],
        "reason_lower": branches["lower"]["reason"],
        "trace_upper": branches["upper"]["trace"],
        "trace_lower": branches["lower"]["trace"],
    }
    drop = r_start - r_mirror
    if drop <= 0.0 or z_max <= 0.0:
        result.update(curvature_radius=np.inf, alpha=np.inf, f3=0.0, mirror=False)
        return result

    curvature = z_max**2 / (2.0 * drop)
    alpha = ejiri_mirror_alpha_from_R_S_R_LIN_R_C_Z_max(
        r_start, r_inboard, curvature, z_max
    )
    result.update(
        curvature_radius=curvature,
        alpha=alpha,
        f3=ejiri_f3_from_alpha(alpha),
        mirror=True,
    )
    return result


def _midplane_crossing_inboard(wall_r, wall_z, r_start):
    """The wall's midplane crossing nearest ``r_start`` on its inboard side."""
    r_closed = np.r_[wall_r, wall_r[:1]]
    z_closed = np.r_[wall_z, wall_z[:1]]
    crossings = []
    for r0, z0, r1, z1 in zip(r_closed[:-1], z_closed[:-1], r_closed[1:], z_closed[1:]):
        if z0 == z1:
            if z0 == 0.0:
                crossings.extend([r0, r1])
            continue
        if (z0 <= 0.0 <= z1) or (z1 <= 0.0 <= z0):
            crossings.append(r0 + (r1 - r0) * (0.0 - z0) / (z1 - z0))
    inboard = [r for r in crossings if r < r_start]
    if not inboard:
        raise ValueError(
            f"the wall has no midplane crossing inboard of r_start={r_start} m"
        )
    return float(max(inboard))


def _mirror_point(R, Z, termination_reason):
    """Where a branch is strongest in |B|: its |Z|, its R, and what ended it.

    An electron escaping along the branch must pass every point between the
    seed and the wall, so the field it has to overcome is the largest one on
    that stretch -- the global minimum of ``R``, not the first local one.  The
    two coincide for a line with a single inward dip; they differ for a line
    tilted at the midplane, whose ``R`` first rises a hair before it dips, and
    for a line that dips more than once.  The seed itself is excluded.
    """
    if R.size < 2:
        # The first step already left the wall: nothing on this side raises
        # |B| above its value at the seed, so an electron heading this way is
        # lost at once.  Report the seed itself, which reads as "no mirror".
        return 0.0, float(R[0]), "wall"
    index = int(np.argmin(R[1:])) + 1
    if index < R.size - 1:
        reason = "turning"
    elif "wall" in str(termination_reason):
        reason = "wall"
    else:
        reason = str(termination_reason)
    return float(abs(Z[index])), float(R[index]), reason


def _trace_branch_lengths(flat_r, flat_z, inside, b_field, polygon, step, max_length_m):
    """One direction of :func:`connection_length_map`, carrying only live lines.

    The live set is compacted every step rather than masked in place: a handful
    of lines circulating near a null would otherwise keep every retired line's
    element in the arithmetic until the last of them finished.
    """
    length = np.zeros(flat_r.size)
    hit_limit = np.zeros(flat_r.size, dtype=bool)
    live = np.flatnonzero(inside)
    position_r = flat_r[live].copy()
    position_z = flat_z[live].copy()
    travelled = np.zeros(live.size)

    def slope(r_values, z_values):
        b_r, b_z, b_phi = b_field(r_values, z_values)
        with np.errstate(divide="ignore", invalid="ignore"):
            return r_values * b_r / b_phi, r_values * b_z / b_phi

    while live.size:
        k1_r, k1_z = slope(position_r, position_z)
        k2_r, k2_z = slope(position_r + 0.5 * step * k1_r, position_z + 0.5 * step * k1_z)
        k3_r, k3_z = slope(position_r + 0.5 * step * k2_r, position_z + 0.5 * step * k2_z)
        k4_r, k4_z = slope(position_r + step * k3_r, position_z + step * k3_z)
        next_r = position_r + (step / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        next_z = position_z + (step / 6.0) * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
        next_travelled = travelled + np.sqrt(
            (next_r - position_r) ** 2
            + (next_z - position_z) ** 2
            + ((position_r + next_r) / 2.0 * step) ** 2
        )

        # A step that produced no finite point has left the field model, not the
        # wall: nothing is known about where that line ends, so it has no length
        # rather than the length it had reached when the field gave out.
        finite = np.isfinite(next_r) & np.isfinite(next_z)
        within = np.zeros(next_r.size, dtype=bool)
        within[finite] = polygon.contains_points(
            np.column_stack([next_r[finite], next_z[finite]])
        )
        over = next_travelled > max_length_m

        left_model = ~finite
        stopped_at_wall = finite & ~within
        stopped_on_length = within & over
        length[live[left_model]] = np.nan
        length[live[stopped_at_wall]] = travelled[stopped_at_wall]
        length[live[stopped_on_length]] = max_length_m
        hit_limit[live[stopped_on_length]] = True

        running = within & ~over
        live = live[running]
        position_r = next_r[running]
        position_z = next_z[running]
        travelled = next_travelled[running]
    return length, hit_limit

@dataclass(frozen=True)
class ParallelCurrentResult:
    """Parallel current density derived from an enclosed toroidal current.

    Carries the intermediates as well as the answer: this conversion is easy to
    get wrong by a factor that looks plausible, so ``lambda_`` and
    ``b_phi_area_integral`` are returned for inspection rather than discarded.
    """

    #: <J.B>/B0 on the shells [A.m^-2].
    j_parallel: np.ndarray
    #: Cumulative integral of j_parallel over cross-sectional area [A]; None
    #: when no shell area was supplied.
    current_parallel_inside: Optional[np.ndarray]
    #: lambda(psi) in J = lambda B, i.e. the field-aligned proportionality
    #: [A.m^-2.T^-1].
    lambda_: np.ndarray
    #: int B_phi dA over each shell [T.m^2].
    b_phi_area_integral: np.ndarray
    #: The shell toroidal current the conversion started from [A].
    shell_current_tor: np.ndarray


def parallel_current_from_toroidal(
    shell_current_tor: Any,
    *,
    f: Any,
    gm1: Any,
    gm5: Any,
    shell_volume: Any,
    b0: float,
    shell_area: Any = None,
) -> ParallelCurrentResult:
    """Convert a shell toroidal driven current to the IMAS parallel current.

    A code that drives current usually reports it as a toroidal current per
    flux shell, or as a profile of current enclosed by each surface. IMAS asks
    for something different: ``j_parallel`` is ``<J.B>/B0``. The two are not
    interchangeable, and in a spherical tokamak they differ by tens of percent
    because ``<B^2>`` is much larger than ``B0^2``.

    Assumptions
    -----------
    The driven current is **field-aligned on each flux surface**,
    ``J = lambda(psi) B``. This is the usual statement for a current driven by
    parallel momentum input -- beams, EC, LH -- and it is what makes the
    conversion possible at all from a toroidal quantity. It is an assumption,
    not an identity: it omits any perpendicular (diamagnetic,
    Pfirsch-Schlueter) part, so a caller converting a *total* plasma current
    with this routine will see that part as a residual.

    Notes
    -----
    With ``J = lambda B`` and ``B_phi = F/R``, the toroidal current through a
    shell is ``dI = lambda * int B_phi dA``. Writing the poloidal area element
    as ``dA = dV / (2 pi R)`` turns that integral into flux-surface averages::

        int B_phi dA = F <R^-2> dV / (2 pi)
        lambda       = 2 pi dI / (F <R^-2> dV)
        j_parallel   = lambda <B^2> / B0

    In the large-aspect-ratio limit -- ``F -> R0 B0``, ``<B^2> -> B0^2``,
    ``<R^-2> -> R0^-2``, ``dV -> 2 pi R0 dA`` -- this reduces to ``dI / dA``,
    the intuitive toroidal current density. That limit is a check on the
    result, never a substitute for it.

    Parameters
    ----------
    shell_current_tor : array_like
        Toroidal current in each flux shell [A].
        Not the enclosed profile: pass ``numpy.diff`` of an enclosed one.
    f : array_like
        ``F = R B_phi`` at the shell centres [T.m].
        A profile, not a constant: a paramagnetic plasma can carry an F well
        above its vacuum value.
    gm1 : array_like
        Flux-surface-averaged ``<R^-2>`` at the shell centres [m^-2].
    gm5 : array_like
        Flux-surface-averaged ``<B^2>`` at the shell centres [T^2].
    shell_volume : array_like
        Volume of each shell [m^3].
    b0 : float
        The vacuum toroidal field IMAS normalizes by [T].
        For a ``core_sources`` consumer this is ``vacuum_toroidal_field.b0`` of
        that IDS, which the data dictionary names explicitly in the definition
        of ``j_parallel``.
    shell_area : array_like, optional
        Cross-sectional area of each shell [m^2].
        Supplying it also returns ``current_parallel_inside``, the cumulative
        surface integral of ``j_parallel``.

    Returns
    -------
    ParallelCurrentResult
        ``j_parallel`` in A/m^2 and, when *shell_area* was given,
        ``current_parallel_inside`` in A, plus the intermediates [-].

    Applicability
    -------------
    Machine-independent. Requires an axisymmetric equilibrium and a driven
    current that is field-aligned. Nothing here is specific to a solver: the
    inputs are physics quantities, so any code reporting a toroidal driven
    current and an equilibrium can use it.

    Convention
    ----------
    Sign is inherited from *shell_current_tor* and *f*, not imposed. The result
    therefore carries the sign convention of whatever produced them, which is
    the caller's to reconcile with the equilibrium it will sit beside.

    Limitations
    -----------
    Shells where ``F <R^-2> dV`` vanishes -- a degenerate or zero-width shell,
    typically at the magnetic axis -- yield zero rather than a division by
    zero. The field-aligned assumption is the dominant error for a total
    current and a much smaller one for a driven current.

    Provenance
    ----------
    .. [1] IMAS definitions read from the data dictionary: ``j_parallel`` is
       ``average(J.B)/B0`` and ``current_parallel_inside`` its cumulative
       surface integral.
    """
    dI = np.asarray(shell_current_tor, dtype=float)
    f_arr = np.asarray(f, dtype=float)
    gm1_arr = np.asarray(gm1, dtype=float)
    gm5_arr = np.asarray(gm5, dtype=float)
    dV = np.asarray(shell_volume, dtype=float)

    shapes = {dI.shape, f_arr.shape, gm1_arr.shape, gm5_arr.shape, dV.shape}
    if len(shapes) != 1:
        raise ValueError(
            "shell_current_tor, f, gm1, gm5 and shell_volume must share a "
            f"shape; got {sorted(str(s) for s in shapes)}"
        )
    if not np.isfinite(b0) or b0 == 0.0:
        raise ValueError(f"b0 must be finite and non-zero, got {b0!r}")

    b_phi_area = f_arr * gm1_arr * dV / (2.0 * np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        lambda_ = np.where(b_phi_area != 0.0, dI / b_phi_area, 0.0)
    lambda_ = np.nan_to_num(lambda_, nan=0.0, posinf=0.0, neginf=0.0)
    j_parallel = lambda_ * gm5_arr / b0

    inside = None
    if shell_area is not None:
        dA = np.asarray(shell_area, dtype=float)
        if dA.shape != dI.shape:
            raise ValueError(
                f"shell_area has shape {dA.shape}, expected {dI.shape}"
            )
        inside = np.cumsum(j_parallel * dA)

    return ParallelCurrentResult(
        j_parallel=j_parallel,
        current_parallel_inside=inside,
        lambda_=lambda_,
        b_phi_area_integral=b_phi_area,
        shell_current_tor=dI,
    )



@dataclass(frozen=True)
class GradShafranovResidual:
    """How far an equilibrium is from satisfying force balance, surface by surface.

    ``residual`` is the root-mean-square of ``delta_star - source`` over the grid
    points falling on each of ``psi_norm``'s surfaces, in the units of the source
    term.  ``relative`` divides it by ``scale``, which is **one number for the
    whole plasma** -- the RMS of the source over every masked point -- so the
    profile can be read across surfaces.  It is not normalised per surface, so
    near the edge, where the source itself is small, it understates the local
    error.  ``mask`` marks the points that were used.
    """

    psi_norm: np.ndarray
    residual: np.ndarray
    relative: np.ndarray
    scale: float
    delta_star: np.ndarray
    source: np.ndarray
    mask: np.ndarray


def grad_shafranov_operator(psi: Any, r: Any, z: Any) -> np.ndarray:
    """The Grad-Shafranov elliptic operator on a rectangular grid.

    Parameters
    ----------
    psi : array_like
        Poloidal flux on the grid, indexed ``(R, Z)`` [Wb/rad].
    r : array_like
        Major-radius grid axis [m].
    z : array_like
        Height grid axis [m].

    Returns
    -------
    np.ndarray
        The operator applied to the flux, on the same grid [Wb/(rad m^2)].

    Raises
    ------
    ValueError
        The flux is not indexed ``(R, Z)`` with the given axes.

    Convention
    ----------
    ``Delta* psi = d2psi/dR2 - (1/R) dpsi/dR + d2psi/dZ2``, the axisymmetric
    elliptic operator whose source is the Grad-Shafranov right-hand side. The flux
    is indexed major radius first, the orientation an EFIT-sourced ODS stores.
    The result scales with the flux unit, so it is per radian for a per-radian
    flux; comparing it against a source term requires both in the same convention.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Second-order central differences, one-sided at the edges, so the outermost two
    cells inherit that lower accuracy. A caller comparing against a source term
    should ignore that frame, as :func:`grad_shafranov_residual` does.

    Provenance
    ----------
    .. [1] The Grad-Shafranov equation in its standard axisymmetric form.
    """
    psi = np.asarray(psi, dtype=float)
    r = np.asarray(r, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    if psi.ndim != 2:
        raise ValueError(f"psi must be a 2-D (R, Z) grid; got shape {psi.shape}")
    if psi.shape != (r.size, z.size):
        raise ValueError(
            f"psi has shape {psi.shape} but the grid is (R={r.size}, Z={z.size}); "
            "psi is indexed (R, Z)"
        )

    dpsi_dr = np.gradient(psi, r, axis=0, edge_order=2)
    d2psi_dr2 = np.gradient(dpsi_dr, r, axis=0, edge_order=2)
    d2psi_dz2 = np.gradient(np.gradient(psi, z, axis=1, edge_order=2), z, axis=1, edge_order=2)
    return d2psi_dr2 - dpsi_dr / r[:, None] + d2psi_dz2


def grad_shafranov_residual(
    psi: Any,
    r: Any,
    z: Any,
    *,
    psi_1d: Any,
    pprime: Any,
    ffprime: Any,
    psi_axis: float | None = None,
    psi_boundary: float | None = None,
    boundary_r: Any = None,
    boundary_z: Any = None,
    n_surfaces: int = 32,
    psi_norm_max: float = 0.99,
    edge_cells: int = 2,
) -> GradShafranovResidual:
    """How far an equilibrium is from satisfying its own force balance, surface by surface.

    The Grad-Shafranov equation says a flux map and its own pressure and poloidal
    current gradients are not independent. Evaluating both sides on the stored grid
    says how well a *particular* reconstruction closed it, which is a different and
    stricter question than whether a solver's iteration converged. The number EFIT
    reports as its Grad-Shafranov error is the latter.

    Parameters
    ----------
    psi : array_like
        Poloidal flux on the grid, indexed ``(R, Z)`` [Wb/rad].
    r : array_like
        Major-radius grid axis [m].
    z : array_like
        Height grid axis [m].
    psi_1d : array_like
        Flux abscissa the two source profiles are given on [Wb/rad].
    pprime : array_like
        Pressure gradient against flux [Pa rad/Wb].
    ffprime : array_like
        Poloidal current term's gradient against flux [T^2 m^2 rad/Wb].
    psi_axis : float, optional
        Poloidal flux on the magnetic axis [Wb/rad].
    psi_boundary : float, optional
        Poloidal flux at the boundary [Wb/rad].
    boundary_r : array_like, optional
        Major radius of the separatrix, used to exclude vacuum points [m].
    boundary_z : array_like, optional
        Height of the separatrix [m].
    n_surfaces : int, optional
        How many flux surfaces to reduce the residual onto [-].
    psi_norm_max : float, optional
        Outermost normalized flux included [-].
    edge_cells : int, optional
        Grid cells framed off the border, where the stencil is one-sided [-].

    Returns
    -------
    GradShafranovResidual
        The residual reduced along the flux coordinate, with the normalization it
        was scaled by and the point count that survived the masks [-].

    Raises
    ------
    ValueError
        The three profiles are not the same length, or no grid point survives the
        boundary, the flux-range trim, and the edge frame together.

    Convention
    ----------
    **Every flux quantity here must be per radian**, COCOS 1 to 8. The textbook
    equation takes this form only in that convention: rescaling to full weber
    multiplies the left side by ``2*pi`` and divides each right-hand derivative by
    it, so the two sides move apart by ``(2*pi)**2``. A caller holding a
    dictionary-conformant ODS converts first; the wrapper
    :func:`vaft.omas.compute_grad_shafranov_residual` does it for them.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The two source profiles are interpolated onto the grid through the flux
    itself, so the residual is meaningful only where they are the equilibrium's
    own.

    **Pass the boundary.** Outside the separatrix the flux is not monotonic; it
    folds back into the normalized range near the poloidal field coils, so a
    flux-value cutoff alone re-admits vacuum points where the two profiles are
    extrapolation and the operator is reading coil current. On a free-boundary
    grid that is a third of the points and two orders of magnitude of spurious
    residual, and it does not affect a fixed-boundary grid at all, so comparing
    the two without it is not comparing like with like.

    Provenance
    ----------
    .. [1] The Grad-Shafranov equation; the operator is
       :func:`grad_shafranov_operator` and the source term is formed from the two
       profiles the equilibrium stores.
    """
    psi = np.asarray(psi, dtype=float)
    r = np.asarray(r, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    psi_1d = np.asarray(psi_1d, dtype=float).ravel()
    pprime = np.asarray(pprime, dtype=float).ravel()
    ffprime = np.asarray(ffprime, dtype=float).ravel()
    if not (psi_1d.size == pprime.size == ffprime.size):
        raise ValueError(
            "psi_1d, pprime and ffprime must be the same length; got "
            f"{psi_1d.size}, {pprime.size}, {ffprime.size}"
        )
    if psi_1d.size < 2:
        raise ValueError("psi_1d needs at least two samples to interpolate")

    axis = float(psi_1d[0]) if psi_axis is None else float(psi_axis)
    boundary = float(psi_1d[-1]) if psi_boundary is None else float(psi_boundary)
    if not np.isfinite(axis) or not np.isfinite(boundary) or axis == boundary:
        raise ValueError(
            f"psi_axis and psi_boundary must be finite and distinct; got {axis}, {boundary}"
        )

    delta_star = grad_shafranov_operator(psi, r, z)
    psi_norm_2d = (psi - axis) / (boundary - axis)

    # The 1-D profiles are tabulated against psi, which may run either way.
    order = np.argsort(psi_1d)
    sampled_psi = psi_1d[order]
    pprime_2d = np.interp(psi, sampled_psi, pprime[order])
    ffprime_2d = np.interp(psi, sampled_psi, ffprime[order])
    source = -MU0 * (r[:, None] ** 2) * pprime_2d - ffprime_2d

    mask = np.isfinite(delta_star) & np.isfinite(source)
    mask &= (psi_norm_2d >= 0.0) & (psi_norm_2d <= float(psi_norm_max))
    if boundary_r is not None and boundary_z is not None:
        outline_r = np.asarray(boundary_r, dtype=float).ravel()
        outline_z = np.asarray(boundary_z, dtype=float).ravel()
        if outline_r.size >= 3:
            weights = fractional_cell_weights_from_boundary(r, z, outline_r, outline_z)
            mask &= weights > 0.5
    if edge_cells > 0:
        frame = np.zeros_like(mask)
        frame[edge_cells:-edge_cells, edge_cells:-edge_cells] = True
        mask &= frame
    if not mask.any():
        raise ValueError(
            "no grid point survives the plasma boundary, the psi_norm range "
            f"and a {edge_cells}-cell border; the grid may be smaller than the "
            "border it is being trimmed by, or psi_axis/psi_boundary may be wrong"
        )

    difference = delta_star - source
    scale = float(np.sqrt(np.mean(source[mask] ** 2)))

    edges = np.linspace(0.0, float(psi_norm_max), int(n_surfaces) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    flat_norm = psi_norm_2d[mask]
    flat_diff = difference[mask]
    residual = np.full(centres.size, np.nan)
    for index in range(centres.size):
        in_bin = (flat_norm >= edges[index]) & (flat_norm < edges[index + 1])
        if in_bin.any():
            residual[index] = float(np.sqrt(np.mean(flat_diff[in_bin] ** 2)))

    relative = residual / scale if scale > 0.0 else np.full_like(residual, np.nan)
    return GradShafranovResidual(
        psi_norm=centres,
        residual=residual,
        relative=relative,
        scale=scale,
        delta_star=delta_star,
        source=source,
        mask=mask,
    )


def resistive_layer_parameters(
    psi_norm,
    q,
    n,
    *,
    t_e,
    n_e,
    psi_norm_kinetic=None,
    ion_mass_amu=1.0,
    z_eff=2.0,
    ln_lambda=17.0,
    m_range=None,
):
    """Resistivity and mass density at each rational surface of a toroidal mode.

    Asymptotic-matching codes -- RDCON's ``rmatch``, STRIDE -- ask for one
    resistivity and one mass density *per rational surface*, because the
    resistive layer width and the reconnection rate are set locally. This
    composes the two things needed to answer that from an equilibrium and its
    kinetic profiles: where the surfaces are, and what the plasma is like
    there.

    Parameters
    ----------
    psi_norm : array_like
        Normalized poloidal flux of the ``q`` profile, increasing [-].
    q : array_like
        Safety factor on ``psi_norm`` [-].
    n : int
        Toroidal mode number; its sign is ignored [-].
    t_e : array_like
        Electron temperature [eV].
    n_e : array_like
        Electron density [m^-3].
    psi_norm_kinetic : array_like, optional
        Normalized poloidal flux of ``t_e``/``n_e``; ``psi_norm`` by default,
        which requires the kinetic profiles to already be on the q grid [-].
    ion_mass_amu : float, optional
        Mass of the bulk ion in atomic mass units; 1 (hydrogen) by default,
        which is what VEST runs [-].
    z_eff : float, optional
        Effective ion charge, passed to the Spitzer resistivity [-].
    ln_lambda : float, optional
        Coulomb logarithm, passed to the Spitzer resistivity [-].
    m_range : tuple of int, optional
        Forwarded to :func:`find_rational_surfaces` [-].

    Returns
    -------
    dict of str to ndarray
        ``m``, ``q_rational`` and ``psi_n_rational`` as
        :func:`find_rational_surfaces` returns them, ordered outward, plus
        ``mass_density`` [kg m^-3], the ``t_e`` [eV] and ``n_e`` [m^-3] each
        was computed from, and the Spitzer resistivity ``eta`` [Ohm m].

    Raises
    ------
    ValueError
        A kinetic profile does not match its coordinate in length, or
        ``ion_mass_amu`` is not positive. Propagates
        :func:`find_rational_surfaces`'s own validation.

    Processing steps
    ----------------
    1. Locate the rational surfaces with :func:`find_rational_surfaces`.
    2. Interpolate ``t_e`` and ``n_e`` linearly onto those surfaces.
    3. Evaluate
       :func:`vaft.formula.equilibrium.spitzer_resistivity_from_T_e_Z_eff_ln_Lambda`
       at each, and take the mass density as ``n_e * ion_mass_amu * m_p``,
       which assumes quasineutrality with a single bulk ion species.

    Limitations
    -----------
    Spitzer resistivity carries no neoclassical trapped-particle correction,
    so it underestimates the parallel resistivity of a spherical tokamak,
    where the trapped fraction is large; the returned ``eta`` is a lower
    bound in that sense.

    It also diverges as ``T_e`` goes to zero. A reconstruction whose
    ``T_e`` reaches zero at the separatrix therefore yields a non-finite
    ``eta`` at any rational surface that sits on that point, and surfaces
    within a few percent of it are dominated by however the profile was
    extrapolated rather than by measurement. Both are reported rather than
    clipped, because the right floor is a property of the discharge and not
    of this function.

    The mass density ignores impurities: with a non-unit ``z_eff`` the bulk
    ion density is below ``n_e``, so this overestimates it by roughly the
    dilution factor.

    Applicability
    -------------
    Machine-independent. The ``ion_mass_amu`` default of 1 is the only VEST
    choice, and it is a keyword.

    Provenance
    ----------
    .. [issue] #716 -- the packaged ``rmatch.in`` supplied a single scalar
       where the code wants one value per rational surface, so ``rmatch``
       stopped before writing its global solution.
    """
    import numpy as _np

    from vaft.formula.constants import MI_P
    from vaft.formula.equilibrium import (
        spitzer_resistivity_from_T_e_Z_eff_ln_Lambda,
    )

    if float(ion_mass_amu) <= 0.0:
        raise ValueError(f"ion_mass_amu must be positive, got {ion_mass_amu!r}")

    surfaces = find_rational_surfaces(psi_norm, q, n, m_range=m_range)
    return {
        **surfaces,
        **resistive_layer_at(
            surfaces["psi_n_rational"],
            psi_norm=psi_norm if psi_norm_kinetic is None else psi_norm_kinetic,
            t_e=t_e,
            n_e=n_e,
            ion_mass_amu=ion_mass_amu,
            z_eff=z_eff,
            ln_lambda=ln_lambda,
        ),
    }


def resistive_layer_at(
    psi_n_surfaces,
    *,
    psi_norm,
    t_e,
    n_e,
    ion_mass_amu=1.0,
    z_eff=2.0,
    ln_lambda=17.0,
):
    """Resistivity and mass density at flux surfaces someone else located.

    The companion to :func:`resistive_layer_parameters`, for when the surfaces
    are already known -- read back from a solver that reported its own, which
    is the only way to be sure the values line up with what that solver will
    index.

    Parameters
    ----------
    psi_n_surfaces : array_like
        Normalized poloidal flux of each surface, ordered as the consumer
        expects them [-].
    psi_norm : array_like
        Normalized poloidal flux of ``t_e``/``n_e`` [-].
    t_e : array_like
        Electron temperature [eV].
    n_e : array_like
        Electron density [m^-3].
    ion_mass_amu : float, optional
        Mass of the bulk ion in atomic mass units; 1 (hydrogen) by default [-].
    z_eff : float, optional
        Effective ion charge [-].
    ln_lambda : float, optional
        Coulomb logarithm [-].

    Returns
    -------
    dict of str to ndarray
        ``mass_density`` [kg m^-3], the ``t_e`` [eV] and ``n_e`` [m^-3] each
        was computed from, and the Spitzer resistivity ``eta`` [Ohm m].

    Raises
    ------
    ValueError
        A kinetic profile does not match its coordinate in length,
        ``psi_norm`` is not strictly increasing, or ``ion_mass_amu`` is not
        positive.

    Processing steps
    ----------------
    1. Interpolate ``t_e`` and ``n_e`` linearly onto ``psi_n_surfaces``.
    2. Evaluate the Spitzer resistivity at each, and take the mass density as
       ``n_e * ion_mass_amu * m_p``.

    Limitations
    -----------
    As :func:`resistive_layer_parameters`: Spitzer carries no neoclassical
    correction and diverges as ``T_e`` goes to zero, and the mass density
    ignores impurity dilution.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [issue] #716.
    """
    import numpy as _np

    from vaft.formula.constants import MI_P
    from vaft.formula.equilibrium import (
        spitzer_resistivity_from_T_e_Z_eff_ln_Lambda,
    )

    if float(ion_mass_amu) <= 0.0:
        raise ValueError(f"ion_mass_amu must be positive, got {ion_mass_amu!r}")

    coordinate = _np.asarray(psi_norm, dtype=float)
    t_e = _np.asarray(t_e, dtype=float)
    n_e = _np.asarray(n_e, dtype=float)
    for name, values in (("t_e", t_e), ("n_e", n_e)):
        if values.shape != coordinate.shape:
            raise ValueError(
                f"{name} has {values.size} points against {coordinate.size} "
                "coordinate points"
            )

    # `np.interp` does not check its abscissa: on a decreasing coordinate (an
    # outboard-in Thomson ordering) it clamps every surface to the first
    # element, so the core resistivity comes back everywhere and looks fine.
    if coordinate.ndim != 1 or coordinate.size < 2 or not _np.all(
        _np.diff(coordinate) > 0.0
    ):
        raise ValueError(
            "psi_norm must be a strictly increasing 1-D coordinate; sort the "
            "kinetic profiles onto it before asking for the layer inputs"
        )

    where = _np.asarray(psi_n_surfaces, dtype=float)
    t_e_at = _np.interp(where, coordinate, t_e)
    n_e_at = _np.interp(where, coordinate, n_e)

    # `_np.float64` rather than `float`: the formula divides by T_e**1.5, and
    # a Python float raises ZeroDivisionError there while IEEE gives `inf`.
    # The divergence at T_e = 0 is a real property of Spitzer and the caller
    # has to see it, so it is returned as a non-finite value rather than as an
    # exception from three frames down.
    with _np.errstate(divide="ignore", invalid="ignore"):
        eta = _np.array(
            [
                spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(
                    _np.float64(value),
                    Z_eff=float(z_eff),
                    ln_Lambda=float(ln_lambda),
                )
                for value in t_e_at
            ],
            dtype=float,
        )

    return {
        "eta": eta,
        "mass_density": n_e_at * float(ion_mass_amu) * MI_P,
        "t_e": t_e_at,
        "n_e": n_e_at,
    }


def find_rational_surfaces(psi_norm, q, n, *, m_range=None):
    """Where a toroidal mode resonates: the surfaces with q = m / n.

    A perturbation with toroidal mode number ``n`` resonates where the safety
    factor equals ``m / n`` for an integer ``m``. This finds those crossings
    by linear interpolation on the ``q`` profile, so the result is the
    equilibrium's own answer and needs no perturbed-equilibrium code.

    Parameters
    ----------
    psi_norm : array_like
        Normalized poloidal flux, increasing [-].
    q : array_like
        Safety factor on ``psi_norm`` [-].
    n : int
        Toroidal mode number; its sign is ignored [-].
    m_range : tuple of int, optional
        ``(m_min, m_max)`` to search; the range ``q`` actually spans by
        default [-].

    Returns
    -------
    dict of str to ndarray
        ``m`` (poloidal mode number), ``q_rational`` (= m / n) and
        ``psi_n_rational`` (where the crossing is), ordered outward [-].

    Raises
    ------
    ValueError
        ``psi_norm`` and ``q`` differ in length, ``psi_norm`` does not
        increase, or ``n`` is zero.

    Processing steps
    ----------------
    1. Drop non-finite samples.
    2. For each integer ``m`` in range, find every bracket where ``q - m/n``
       changes sign and interpolate the crossing linearly.
    3. Sort the crossings outward.

    Limitations
    -----------
    Accuracy is that of linear interpolation, so the error scales with the
    square of the local grid spacing and with the curvature of ``q`` -- which
    means it is worst exactly where ``q`` is steepest, at the edge. On the
    grid GPEC itself writes, which is refined around the rational surfaces
    (a spacing of 3.6e-06 at the outermost surface of the DIII-D example
    against 7.4e-03 in the core), the surfaces come back to 1.4e-09 in
    ``psi_norm``. On a uniform grid of the kind an equilibrium reconstruction
    produces they do not: 6.5e-04 at 129 points and 4.0e-05 at 513, in both
    cases dominated by the outermost surface. Interpolate ``q`` onto a finer
    grid before asking, if the answer has to be better than that.

    A reversed-shear ``q`` resonates twice on the same ``m``, and both
    crossings are returned. A root that sits exactly on a grid point, or a
    ``q`` that is flat at ``m / n`` over an interval, is returned once -- at
    the node and at the start of the flat region respectively -- rather than
    once per sign change, which would double-count it.

    Applicability
    -------------
    Machine-independent. Any monotonic radial coordinate; the name says
    ``psi_norm`` because that is what the perturbed-equilibrium codes report
    their rational surfaces on.

    Provenance
    ----------
    .. [legacy] ``gpec_multimode_metrics.rational_surface_table`` derived the
       same surfaces from GPEC output; this takes them from the equilibrium,
       so they can be known before a run -- at the accuracy the equilibrium's
       own grid supports, which the Limitations quantify.
    """
    psi_norm = np.asarray(psi_norm, dtype=float)
    q = np.asarray(q, dtype=float)
    if psi_norm.ndim != 1 or q.ndim != 1:
        raise ValueError(
            f"psi_norm is {psi_norm.ndim}-D and q is {q.ndim}-D; a profile is one "
            "dimensional, and flattening a (time, radius) array here would "
            "interpolate across the time axis"
        )
    if psi_norm.shape != q.shape:
        raise ValueError(
            f"{psi_norm.size} coordinate points against {q.size} q values"
        )
    if n != int(n):
        raise ValueError(f"n must be a whole toroidal mode number, not {n!r}")
    if int(n) == 0:
        raise ValueError("n must be non-zero; q = m / n is undefined for n = 0")
    finite = np.isfinite(psi_norm) & np.isfinite(q)
    psi_norm, q = psi_norm[finite], q[finite]
    if psi_norm.size < 2:
        raise ValueError("need at least two finite samples to find a crossing")
    if not np.all(np.diff(psi_norm) > 0):
        raise ValueError(
            "psi_norm must increase; a crossing is located by interpolation and a "
            "non-monotonic coordinate would place it ambiguously"
        )

    order = abs(int(n))
    if m_range is None:
        lo = int(np.ceil(np.nanmin(q) * order))
        hi = int(np.floor(np.nanmax(q) * order))
    else:
        lo, hi = int(m_range[0]), int(m_range[1])
        if lo > hi:
            raise ValueError(
                f"m_range is ({lo}, {hi}), which is empty; a reversed range would "
                "return no surfaces and read as a plasma with no resonance"
            )

    modes: list[int] = []
    q_rational: list[float] = []
    positions: list[float] = []
    for m in range(lo, hi + 1):
        target = m / order
        residual = q - target
        # Exact zeros are roots in their own right, and are taken first: a
        # root sitting on a node otherwise shows up as two sign changes,
        # +1 -> 0 and 0 -> -1, that interpolate to the same point, and a q
        # flat at m/n over an interval shows up as one at each end of it.
        # Either way it is one surface, and counting it twice would double its
        # weight in every reduction downstream.
        zero = residual == 0.0
        crossings = [
            float(psi_norm[index])
            for index in np.nonzero(zero)[0]
            if index == 0 or not zero[index - 1]
        ]
        for index in np.nonzero(np.diff(np.sign(residual)) != 0)[0]:
            if zero[index] or zero[index + 1]:
                continue  # already taken, as the zero itself
            span = residual[index + 1] - residual[index]
            weight = -residual[index] / span
            crossings.append(
                float(psi_norm[index] + weight * (psi_norm[index + 1] - psi_norm[index]))
            )
        for crossing in crossings:
            modes.append(m)
            q_rational.append(target)
            positions.append(crossing)

    outward = np.argsort(positions) if positions else np.empty(0, dtype=int)
    return {
        "m": np.asarray(modes, dtype=int)[outward],
        "q_rational": np.asarray(q_rational, dtype=float)[outward],
        "psi_n_rational": np.asarray(positions, dtype=float)[outward],
    }


def straight_field_line_tables(
    theta_turns,
    r,
    z,
    magnetic_axis,
    *,
    jacobian: str,
    minimum_separation: float = 1e-10,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Per-surface tables taking the geometric poloidal angle to the code's own.

    Parameters
    ----------
    theta_turns : array_like
        The code's poloidal angle **in turns**, one value per row of ``r`` and
        ``z``. A closing duplicate is dropped [-].
    r, z : array_like
        Flux-surface geometry, ``(theta, psi)`` [m].
    magnetic_axis : sequence of float
        ``(r, z)`` of the axis the geometric angle is measured about [m].
    jacobian : str
        Which straight-field-line angle these are -- ``hamada``, ``pest``,
        ``boozer`` and so on. Required and recorded, not inferred: the file
        does not carry it, and the mapping is meaningless without knowing
        which angle it lands in [n/a].
    minimum_separation : float, optional
        Geometric-angle spacing below which two samples are treated as one
        [rad].

    Returns
    -------
    tuple of (ndarray, ndarray)
        One ``(lab, sfl)`` pair per flux surface, both strictly increasing and
        spanning exactly one period, ready for
        :func:`lab_to_straight_field_line` [rad].

    Raises
    ------
    ValueError
        ``r`` and ``z`` disagree in shape, ``theta_turns`` does not match their
        first axis, an input is not finite, ``jacobian`` is empty, or a surface
        collapses to fewer than three distinct angles.

    Convention
    ----------
    **The angle arrives in turns, and nothing in the file says so.** GPEC
    writes ``theta_dcon`` from 0 to 1 with no ``units`` attribute, so a caller
    that took it for radians would be wrong by ``2 pi`` and the error would
    look like a mis-shaped plasma rather than a unit mistake. It is converted
    here, once, and the parameter is named for what it holds.

    The geometric angle is ``atan2(z - z_axis, r - r_axis)``, which is the lab
    angle FLARE's Poincare output is in; the straight-field-line angle is what
    GPEC and DCON label their harmonics by. The two agree only on a circular
    concentric equilibrium.

    Applicability
    -------------
    Machine-independent.

    Processing steps
    ----------------
    1. Drop the closing duplicate, which GPEC writes so its grid is periodic.
    2. Per surface, take the geometric angle about the axis.
    3. Roll so the table starts near the outboard midplane, then unwrap both
       angles together.
    4. Reverse when the geometric angle runs backwards, so the table increases
       whichever way the surface was traced.
    5. Drop samples closer than ``minimum_separation``, which a
       near-stagnation point in the geometric angle can produce.
    6. Close the period by repeating the first entry shifted by ``2 pi``, in
       the direction the straight-field-line angle runs along the table.

    Limitations
    -----------
    The relabelling is **one flux surface at a time** and says nothing about
    the radial coordinate: a caller with points between surfaces has to decide
    how to interpolate across them, and doing that in the lab angle is not the
    same as doing it in the straight-field-line one.

    Provenance
    ----------
    .. [legacy] ``library/flare_plotting.py::_build_lab_to_sfl_inverse_tables``
       and ``_interp_periodic_inverse``.
    """
    if not str(jacobian).strip():
        raise ValueError(
            "jacobian must name which straight-field-line angle this lands in "
            "-- hamada, pest, boozer. GPEC's profile output does not carry it, "
            "so it is passed rather than guessed"
        )
    turns = np.atleast_1d(np.asarray(theta_turns, dtype=float))
    radius = np.asarray(r, dtype=float)
    height = np.asarray(z, dtype=float)
    if radius.shape != height.shape or radius.ndim != 2:
        raise ValueError(
            f"r and z must be one 2-D (theta, psi) grid; got {radius.shape} and "
            f"{height.shape}"
        )
    if turns.size != radius.shape[0]:
        raise ValueError(
            f"{turns.size} poloidal angles against {radius.shape[0]} rows of geometry"
        )
    if not (np.all(np.isfinite(turns)) and np.all(np.isfinite(radius))
            and np.all(np.isfinite(height))):
        raise ValueError("the angle or the geometry holds a non-finite value")
    axis = np.asarray(magnetic_axis, dtype=float).ravel()
    if axis.size != 2 or not np.all(np.isfinite(axis)):
        raise ValueError(f"magnetic_axis must be a finite (r, z) pair, not {magnetic_axis!r}")

    sfl = np.mod(turns * 2.0 * np.pi, 2.0 * np.pi)
    # GPEC closes its grid, so the last row repeats the first; keeping it would
    # put a zero-length step in every table.
    if sfl.size > 1 and np.isclose(turns[0] % 1.0, turns[-1] % 1.0):
        sfl, radius, height = sfl[:-1], radius[:-1, :], height[:-1, :]

    tables: list[tuple[np.ndarray, np.ndarray]] = []
    for column in range(radius.shape[1]):
        lab = np.mod(
            np.arctan2(height[:, column] - axis[1], radius[:, column] - axis[0]),
            2.0 * np.pi,
        )
        start = int(np.argmin(np.abs(np.angle(np.exp(1j * lab)))))
        lab_line = np.unwrap(np.roll(lab, -start))
        sfl_line = np.unwrap(np.roll(sfl, -start))
        if lab_line[-1] < lab_line[0]:
            lab_line, sfl_line = lab_line[::-1], sfl_line[::-1]
        if np.any(np.diff(lab_line) <= 0.0):
            order = np.argsort(lab_line)
            lab_line, sfl_line = lab_line[order], sfl_line[order]
        keep = np.r_[True, np.diff(lab_line) > float(minimum_separation)]
        lab_line, sfl_line = lab_line[keep], sfl_line[keep]
        if lab_line.size < 3:
            raise ValueError(
                f"flux surface {column} collapses to {lab_line.size} distinct "
                "geometric angles, so it cannot be relabelled"
            )
        # A surface traced clockwise leaves the straight-field-line angle
        # DEcreasing along the (increasing) geometric one, so its period closes
        # 2 pi below the first entry, not above it.
        direction = 1.0 if sfl_line[-1] >= sfl_line[0] else -1.0
        tables.append((
            np.r_[lab_line, lab_line[0] + 2.0 * np.pi],
            np.r_[sfl_line, sfl_line[0] + direction * 2.0 * np.pi],
        ))
    return tuple(tables)


def lab_to_straight_field_line(theta_lab, table: tuple[np.ndarray, np.ndarray]):
    """Relabel a geometric poloidal angle into the straight-field-line one.

    Parameters
    ----------
    theta_lab : float or array_like
        Geometric angle about the magnetic axis, any branch [rad].
    table : tuple of ndarray
        One ``(lab, sfl)`` pair from :func:`straight_field_line_tables` [rad].

    Returns
    -------
    float or ndarray
        The straight-field-line angle, wrapped to ``[0, 2 pi)`` [rad].

    Raises
    ------
    ValueError
        The table's two arrays disagree in length, or it is too short to
        interpolate.

    Convention
    ----------
    The input is wrapped onto the table's own branch before interpolating.
    The table spans one period starting wherever the surface's outboard
    midplane fell, **not** at zero, so wrapping to ``[0, 2 pi)`` and
    interpolating would read off the end for every angle below that start.

    Applicability
    -------------
    Machine-independent.

    Processing steps
    ----------------
    1. Wrap the angle onto the table's branch.
    2. Interpolate, and wrap the result into ``[0, 2 pi)``.

    Provenance
    ----------
    .. [legacy] ``library/flare_plotting.py::_interp_periodic_inverse``.
    """
    lab_grid, sfl_grid = (np.asarray(part, dtype=float) for part in table)
    if lab_grid.size != sfl_grid.size:
        raise ValueError(
            f"the table pairs {lab_grid.size} geometric angles with "
            f"{sfl_grid.size} straight-field-line ones"
        )
    if lab_grid.size < 3:
        raise ValueError("a relabelling table needs at least three entries")
    angle = np.mod(np.asarray(theta_lab, dtype=float), 2.0 * np.pi)
    start = float(lab_grid[0])
    angle = angle + 2.0 * np.pi * np.ceil((start - angle) / (2.0 * np.pi))
    angle = np.where(angle >= start + 2.0 * np.pi, angle - 2.0 * np.pi, angle)
    result = np.mod(np.interp(angle, lab_grid, sfl_grid), 2.0 * np.pi)
    return float(result) if np.ndim(theta_lab) == 0 else result


class StraightFieldLineMap:
    """The PEST straight-field-line angle of one flux map, evaluable anywhere.

    Built by :func:`straight_field_line_map`; see that function for the
    definition, the construction and its limits.  Holds a table of the
    flux surfaces sampled on rays from the magnetic axis, indexed by
    ``sqrt(psi_norm)`` and the geometric angle, and a bicubic spline of the
    flux itself.  Every method takes points in metres and broadcasts.
    """

    def __init__(self, spline, psi_axis, psi_boundary, magnetic_axis, x_levels,
                 theta, rho, nu, rho_boundary):
        self._spline = spline
        self.psi_axis = float(psi_axis)
        self.psi_boundary = float(psi_boundary)
        self.magnetic_axis = (float(magnetic_axis[0]), float(magnetic_axis[1]))
        self.sqrt_psi_norm = x_levels
        self.theta_geometric = theta
        self.rho = rho
        self.nu = nu
        self.rho_boundary = rho_boundary
        pad = 4
        theta_ext = np.concatenate((theta[-pad:] - 2.0 * np.pi, theta, theta[:pad] + 2.0 * np.pi))
        nu_ext = np.concatenate((nu[:, -pad:], nu, nu[:, :pad]), axis=1)
        rho_b_ext = np.concatenate((rho_boundary[-pad:], rho_boundary, rho_boundary[:pad]))
        self._nu_spline = RectBivariateSpline(x_levels, theta_ext, nu_ext)
        self._theta_ext = theta_ext
        self._rho_b_ext = rho_b_ext
        self._outboard = None

    def outboard_radius(self, psi_norm):
        """Major radius where each normalized flux surface crosses the outboard midplane [m].

        Tabulated once on 2049 points of the outboard ray from the axis to the
        boundary, where the normalized flux is monotonic, and inverted by
        linear interpolation; well below a tenth of a millimetre on a
        machine-sized grid.
        """
        if self._outboard is None:
            ra, za = self.magnetic_axis
            rho = np.linspace(0.0, self.rho_boundary[0], 2049)
            psin = np.maximum.accumulate(self.psi_norm(ra + rho, np.full_like(rho, za)))
            psin[0] = 0.0
            self._outboard = (psin, ra + rho)
        psin, radius = self._outboard
        return np.interp(np.asarray(psi_norm, float), psin, radius)

    def psi(self, r, z):
        """Flux at the points, from the bicubic spline [flux unit of the map]."""
        return self._spline.ev(np.asarray(r, float), np.asarray(z, float))

    def psi_norm(self, r, z):
        """Normalized flux at the points [-]."""
        return (self.psi(r, z) - self.psi_axis) / (self.psi_boundary - self.psi_axis)

    def grad_psi(self, r, z):
        """``|grad psi|`` at the points [flux unit per metre]."""
        r = np.asarray(r, float)
        z = np.asarray(z, float)
        return np.hypot(self._spline.ev(r, z, dx=1), self._spline.ev(r, z, dy=1))

    def geometric_angle(self, r, z):
        """``atan2(z - z_axis, r - r_axis)`` wrapped to ``[0, 2 pi)`` [rad]."""
        ra, za = self.magnetic_axis
        return np.mod(np.arctan2(np.asarray(z, float) - za, np.asarray(r, float) - ra), 2.0 * np.pi)

    def inside(self, r, z):
        """True inside the boundary surface, measured along the axis ray [bool]."""
        ra, za = self.magnetic_axis
        r = np.asarray(r, float)
        z = np.asarray(z, float)
        radius = np.hypot(r - ra, z - za)
        edge = np.interp(self.geometric_angle(r, z), self._theta_ext, self._rho_b_ext)
        return radius <= edge

    def theta_star(self, r, z):
        """Straight-field-line angle in ``[0, 2 pi)``; NaN outside the boundary [rad]."""
        r = np.asarray(r, float)
        z = np.asarray(z, float)
        theta = self.geometric_angle(r, z)
        x = np.sqrt(np.clip(self.psi_norm(r, z), 0.0, 1.0))
        x = np.clip(x, self.sqrt_psi_norm[0], self.sqrt_psi_norm[-1])
        nu = self._nu_spline.ev(x, theta)
        return np.where(self.inside(r, z), _wrap_turn(theta + nu), np.nan)

    def surface(self, psi_norm, n_theta: int | None = None):
        """One flux surface on the geometric-angle grid, solved exactly.

        Returns a dict with ``theta`` (geometric), ``theta_star``, ``r``,
        ``z`` and ``grad_psi`` on that grid, and ``weight``, the integrand
        ``dl / (R |grad psi|)`` per unit geometric angle.
        """
        theta = (self.theta_geometric if n_theta is None
                 else np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False))
        rho, weight = _solve_rays(self._spline, self.psi_axis, self.psi_boundary,
                                  self.magnetic_axis, theta,
                                  np.interp(theta, self._theta_ext, self._rho_b_ext),
                                  np.array([float(psi_norm)]))
        nu = _integrate_periodic_rate(weight[0])
        ra, za = self.magnetic_axis
        r = ra + rho[0] * np.cos(theta)
        z = za + rho[0] * np.sin(theta)
        return {
            "theta": theta,
            "theta_star": _wrap_turn(theta + nu),
            "r": r,
            "z": z,
            "grad_psi": self.grad_psi(r, z),
            "weight": weight[0],
        }


def _wrap_turn(angle):
    """Wrap into ``[0, 2 pi)``; ``np.mod`` returns ``2 pi`` for a tiny negative."""
    wrapped = np.mod(angle, 2.0 * np.pi)
    return np.where(wrapped >= 2.0 * np.pi, wrapped - 2.0 * np.pi, wrapped)


def _refine_o_point(spline, axis, r, z):
    """The flux spline's own O-point nearest a supplied axis.

    A recorded axis is usually the solver's, a fraction of a cell from the
    stationary point of the gridded flux, and rays from it see the flux fall
    before it rises. Newton steps on the spline gradient move it there; a step
    that leaves the two neighbouring cells is refused and the recorded axis
    kept.
    """
    r0, z0 = float(axis[0]), float(axis[1])
    cell = max(float(np.max(np.diff(r))), float(np.max(np.diff(z))))
    rc, zc = r0, z0
    for _ in range(30):
        gr, gz = float(spline.ev(rc, zc, dx=1)), float(spline.ev(rc, zc, dy=1))
        hrr = float(spline.ev(rc, zc, dx=2))
        hzz = float(spline.ev(rc, zc, dy=2))
        hrz = float(spline.ev(rc, zc, dx=1, dy=1))
        det = hrr * hzz - hrz * hrz
        if det <= 0.0:
            return np.array([r0, z0])
        dr = (hzz * gr - hrz * gz) / det
        dz = (hrr * gz - hrz * gr) / det
        rc, zc = rc - dr, zc - dz
        if np.hypot(rc - r0, zc - z0) > 2.0 * cell:
            return np.array([r0, z0])
        if np.hypot(dr, dz) < 1e-12 * cell:
            break
    return np.array([rc, zc])


def _ray_boundary(spline, psi_axis, psi_boundary, axis, theta, rho_max):
    """Distance from the axis to ``psi_norm = 1`` along each ray."""
    ra, za = axis
    samples = np.linspace(0.0, 1.0, 801)[1:]
    out = np.empty(theta.size)
    for k, (angle, limit) in enumerate(zip(theta, rho_max)):
        rho = samples * limit
        psin = (spline.ev(ra + rho * np.cos(angle), za + rho * np.sin(angle)) - psi_axis) / (
            psi_boundary - psi_axis)
        crossed = np.nonzero(psin >= 1.0)[0]
        if crossed.size == 0:
            raise ValueError(
                "the boundary flux is not reached before the grid edge at geometric angle "
                f"{angle:.3f} rad; the flux map must contain the whole boundary surface"
            )
        j = crossed[0]
        if j and np.any(np.diff(psin[: j + 1]) < -1e-9):
            raise ValueError(
                "normalized flux is not monotonic along the ray at geometric angle "
                f"{angle:.3f} rad; the surfaces are not star-shaped about the axis"
            )
        lo = rho[j - 1] if j else 0.0
        hi = rho[j]
        p_lo = psin[j - 1] if j else 0.0
        out[k] = lo + (1.0 - p_lo) * (hi - lo) / (psin[j] - p_lo)
    return out


def _solve_rays(spline, psi_axis, psi_boundary, axis, theta, rho_edge, levels):
    """Radius of each normalized level along each ray, and the PEST weight there.

    Returns ``rho`` and ``weight``, both ``(levels, theta)``; the weight is
    ``rho / (R dpsi_norm/drho)``, which is ``dl / (R |grad psi|)`` per unit
    geometric angle up to the surface's constant flux normalisation.
    """
    ra, za = axis
    span = psi_boundary - psi_axis
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    samples = np.linspace(0.0, 1.0, 401)
    rho_s = samples[:, None] * rho_edge[None, :]
    psin_s = (spline.ev(ra + rho_s * cos_t, za + rho_s * sin_t) - psi_axis) / span
    psin_s[0] = 0.0
    rho = np.empty((levels.size, theta.size))
    for k in range(theta.size):
        rho[:, k] = np.interp(levels, np.maximum.accumulate(psin_s[:, k]), rho_s[:, k])
    for _ in range(3):
        r = ra + rho * cos_t
        z = za + rho * sin_t
        value = (spline.ev(r, z) - psi_axis) / span - levels[:, None]
        slope = (cos_t * spline.ev(r, z, dx=1) + sin_t * spline.ev(r, z, dy=1)) / span
        rho = np.clip(rho - value / slope, 0.0, rho_edge[None, :])
    r = ra + rho * cos_t
    z = za + rho * sin_t
    slope = (cos_t * spline.ev(r, z, dx=1) + sin_t * spline.ev(r, z, dy=1)) / span
    return rho, rho / (r * slope)


def _integrate_periodic_rate(weight):
    """``theta* - theta`` from the PEST rate on a uniform periodic grid.

    ``d theta*/d theta = weight / mean(weight)``, so the difference has a
    zero-mean periodic derivative; it is integrated spectrally and pinned to
    zero at the first sample (the outboard midplane).
    """
    rate = weight / np.mean(weight) - 1.0
    n = rate.size
    coeff = np.fft.rfft(rate)
    k = np.fft.rfftfreq(n, d=1.0 / n)
    integral = np.zeros_like(coeff)
    integral[1:] = coeff[1:] / (1j * k[1:])
    if n % 2 == 0:
        integral[-1] = 0.0
    nu = np.fft.irfft(integral, n)
    return nu - nu[0]


def straight_field_line_map(
    psi,
    r,
    z,
    psi_axis: float,
    psi_boundary: float,
    magnetic_axis,
    *,
    n_theta: int = 256,
    n_surfaces: int = 96,
    sqrt_psi_norm_min: float = 0.02,
) -> StraightFieldLineMap:
    """The PEST straight-field-line poloidal angle of a flux map, built once.

    Parameters
    ----------
    psi : array_like
        Poloidal flux on the grid, indexed ``(R, Z)``, in any storage family:
        only ratios and the geometry enter [Wb or Wb/rad].
    r : array_like
        Major-radius grid axis [m].
    z : array_like
        Height grid axis [m].
    psi_axis : float
        Flux on the magnetic axis, in the unit of *psi* [Wb or Wb/rad].
    psi_boundary : float
        Flux on the boundary surface, in the unit of *psi* [Wb or Wb/rad].
    magnetic_axis : sequence of float
        ``(R, Z)`` of the magnetic axis, the origin of the geometric angle [m].
    n_theta : int, optional
        Geometric-angle samples per surface, uniform on one period [-].
    n_surfaces : int, optional
        Tabulated surfaces, uniform in ``sqrt(psi_norm)`` [-].
    sqrt_psi_norm_min : float, optional
        Innermost tabulated surface, as ``sqrt(psi_norm)``; points closer to
        the axis take its angle offset [-].

    Returns
    -------
    StraightFieldLineMap
        An object evaluating ``theta_star``, ``psi_norm``, ``grad_psi`` and
        the boundary test at arbitrary points, and solving any single surface
        exactly with ``surface(psi_norm)`` [-].

    Raises
    ------
    ValueError
        The flux map is not shaped ``(len(r), len(z))``, the axis and boundary
        flux coincide, the boundary surface leaves the grid, or a surface is
        not star-shaped about the axis.

    Convention
    ----------
    **The PEST angle**, in which field lines are straight:
    ``theta* = 2 pi int dl/(R |grad psi|) / oint dl/(R |grad psi|)`` along a
    flux surface, so ``d phi / d theta* = q`` on every surface. The poloidal
    current function and the flux normalisation are constant on a surface and
    cancel, which is why neither ``F`` nor the COCOS storage family is needed.

    **Origin and direction are geometric, not COCOS.** ``theta* = 0`` on the
    outboard midplane ray (``Z = Z_axis``, ``R > R_axis``) and it increases in
    the same sense as ``atan2(Z - Z_axis, R - R_axis)``: outboard, top,
    inboard, bottom. Whether that is the direction of the poloidal field
    depends on the equilibrium's current direction, which a caller forming a
    helical phase must take from the field, not from this angle.

    Applicability
    -------------
    Machine-independent.

    Processing steps
    ----------------
    1. Fit a bicubic spline to the flux.
    2. Move the axis to the spline's O-point. Along each geometric-angle ray
       from it, find where the normalized flux reaches one, rejecting a ray on
       which it is not monotonic.
    3. On every tabulated surface, solve the ray radius by Newton iteration on
       the spline and form ``rho / (R d psi_norm / d rho)``, which is
       ``dl / (R |grad psi|)`` per unit geometric angle.
    4. Integrate the rate spectrally into ``theta* - theta``, zero at the
       outboard midplane.
    5. Spline that periodic offset over ``(sqrt(psi_norm), theta)``.

    Limitations
    -----------
    The supplied axis is moved to the flux spline's own O-point when that is
    within two cells of it, so the angle's origin is the axis of the map as
    gridded, not the solver's; ``StraightFieldLineMap.magnetic_axis`` records
    where it went. Needs surfaces that every ray from the axis crosses once, which holds for
    the nested surfaces of a tokamak inside its boundary but not beyond an
    X-point, so the boundary surface itself must be closed on the grid. Inside
    the innermost tabulated surface the angle offset is frozen at that
    surface's value; the angle is undefined on the axis itself. Accuracy is
    set by the flux spline, not by the table, because every surface is solved
    on the spline rather than on grid contours.

    Provenance
    ----------
    .. [1] Grimm, Dewar and Manickam, J. Comput. Phys. 49, 94 (1983), the PEST
       coordinate system this angle belongs to.
    .. [2] Sauter and Medvedev, Comput. Phys. Commun. 184, 293 (2013), for the
       straight-field-line relation ``d phi / d theta* = q``.
    """
    psi = np.asarray(psi, dtype=float)
    r = np.asarray(r, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    if psi.shape != (r.size, z.size):
        raise ValueError(f"psi shape {psi.shape} must equal (len(r), len(z)) = {(r.size, z.size)}.")
    if psi_boundary == psi_axis:
        raise ValueError("psi_boundary must differ from psi_axis to normalize.")
    axis = np.asarray(magnetic_axis, dtype=float).reshape(-1)
    if axis.size != 2 or not np.all(np.isfinite(axis)):
        raise ValueError(f"magnetic_axis must be a finite (R, Z) pair, not {magnetic_axis!r}")
    spline = RectBivariateSpline(r, z, psi)
    axis = _refine_o_point(spline, axis, r, z)
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Distance to the grid edge along each ray, a hard bound for the search.
    with np.errstate(divide="ignore", invalid="ignore"):
        to_r = np.where(cos_t > 0, (r[-1] - axis[0]) / cos_t,
                        np.where(cos_t < 0, (r[0] - axis[0]) / cos_t, np.inf))
        to_z = np.where(sin_t > 0, (z[-1] - axis[1]) / sin_t,
                        np.where(sin_t < 0, (z[0] - axis[1]) / sin_t, np.inf))
    rho_max = np.minimum(to_r, to_z)
    rho_edge = _ray_boundary(spline, psi_axis, psi_boundary, axis, theta, rho_max)
    x_levels = np.linspace(float(sqrt_psi_norm_min), 1.0, int(n_surfaces))
    rho, weight = _solve_rays(spline, psi_axis, psi_boundary, axis, theta, rho_edge, x_levels**2)
    nu = np.array([_integrate_periodic_rate(row) for row in weight])
    return StraightFieldLineMap(spline, psi_axis, psi_boundary, axis, x_levels, theta, rho, nu, rho_edge)


def straight_field_line_angle_on_grid(
    psi,
    r,
    z,
    psi_axis: float,
    psi_boundary: float,
    magnetic_axis,
    *,
    n_theta: int = 256,
    n_surfaces: int = 96,
    sqrt_psi_norm_min: float = 0.02,
) -> np.ndarray:
    """The PEST straight-field-line angle on every node of a flux map.

    Parameters
    ----------
    psi : array_like
        Poloidal flux on the grid, indexed ``(R, Z)`` [Wb or Wb/rad].
    r : array_like
        Major-radius grid axis [m].
    z : array_like
        Height grid axis [m].
    psi_axis : float
        Flux on the magnetic axis, in the unit of *psi* [Wb or Wb/rad].
    psi_boundary : float
        Flux on the boundary surface, in the unit of *psi* [Wb or Wb/rad].
    magnetic_axis : sequence of float
        ``(R, Z)`` of the magnetic axis [m].
    n_theta : int, optional
        Geometric-angle samples per surface, as in
        :func:`straight_field_line_map` [-].
    n_surfaces : int, optional
        Tabulated surfaces, as in :func:`straight_field_line_map` [-].
    sqrt_psi_norm_min : float, optional
        Innermost tabulated surface, as ``sqrt(psi_norm)`` [-].

    Returns
    -------
    numpy.ndarray
        ``theta*`` shaped ``(len(r), len(z))``, in ``[0, 2 pi)`` inside the
        boundary and NaN outside it [rad].

    Convention
    ----------
    The angle of :func:`straight_field_line_map`: PEST, zero on the outboard
    midplane, increasing in the sense of ``atan2(Z - Z_axis, R - R_axis)``.
    The array is indexed major radius first, like the flux map.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Node values only; a caller that needs the angle between nodes should keep
    the map from :func:`straight_field_line_map` and evaluate it there, since
    interpolating a wrapped angle across its ``2 pi`` seam is wrong.

    Provenance
    ----------
    .. [1] Grimm, Dewar and Manickam, J. Comput. Phys. 49, 94 (1983), through
       :func:`straight_field_line_map`.
    """
    sfl = straight_field_line_map(psi, r, z, psi_axis, psi_boundary, magnetic_axis,
                                  n_theta=n_theta, n_surfaces=n_surfaces,
                                  sqrt_psi_norm_min=sqrt_psi_norm_min)
    rr, zz = np.meshgrid(np.asarray(r, float).reshape(-1), np.asarray(z, float).reshape(-1), indexing="ij")
    return sfl.theta_star(rr, zz)
