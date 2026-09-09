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
    "parallel_current_from_toroidal",
    "poloidal_field_at_boundary",
    "prepare_boundary_for_shafranov",
    "psi_to_RZ",
    "psi_to_radial",
    "psi_to_rho",
    "psi_to_rz",
    "r_at_z_extremum",
    "radial_to_psi",
    "rho_to_psi",
    "scale_boundary_conformal",
    "shafranov_integrals",
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

    Returns
    -------
    f_RZ : np.ndarray
        The profile on the grid, zero outside the boundary [any].
    psiN_RZ : np.ndarray
        Normalized poloidal flux on the grid [-].

    Raises
    ------
    ValueError
        The profile and its abscissa are not one-dimensional and of equal length.

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
    plasma-only integral. Because only the normalized flux is used, the absolute
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
    f_2d, psiN_RZ = psi_to_rz(psiN_1d, f_1d, psi_RZ, psi_axis, psi_lcfs)
    R_mesh, Z_mesh = np.meshgrid(R_grid, Z_grid, indexing="ij")
    mask_plasma = (psiN_RZ >= 0.0) & (psiN_RZ <= 1.0) & (R_mesh > 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        B_phi_plasma = f_2d / R_mesh
        B_phi_vacuum = f_vac_val / R_mesh

    diff_B = B_phi_plasma - B_phi_vacuum
    integrand = np.where(mask_plasma, diff_B, 0.0)

    dR = np.gradient(R_grid)[:, None]
    dZ = np.gradient(Z_grid)[None, :]
    dA = np.abs(dR * dZ)

    return float(np.nansum(integrand * dA))


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
    f_2d, psiN_RZ = psi_to_rz(psiN_1d, f_1d, psi_RZ, psi_axis, psi_lcfs)
    R_mesh, Z_mesh = np.meshgrid(R_grid, Z_grid, indexing="ij")
    mask_plasma = (psiN_RZ >= 0.0) & (psiN_RZ <= 1.0) & (R_mesh > 0.0)

    dR = np.gradient(R_grid)[:, None]
    dZ = np.gradient(Z_grid)[None, :]
    dA = np.abs(dR * dZ)
    dV = 2.0 * np.pi * R_mesh * dA

    # (B_tv² - B_t²) = (F_vac² - F²) / R²; integrand * dV = 2π (F_vac² - F²)/R * dA
    with np.errstate(divide="ignore", invalid="ignore"):
        diff_sq = (f_vac_val**2 - f_2d**2) / (R_mesh**2)
    integrand = np.where(mask_plasma, diff_sq, 0.0)

    integral = float(np.nansum(integrand * dV))

    if V_p is not None and V_p > 0:
        Omega = V_p
    else:
        Omega = float(np.sum(dV[mask_plasma]))
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
    cross-sectional one. Only cells with normalized flux between zero and one and
    a positive major radius contribute, which makes the mask the definition of
    "the plasma" here. Accepts either one-dimensional axes or a full mesh.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Cell areas come from a gradient of the axes, so the outermost cells are
    one-sided and a strongly non-uniform grid is approximated. The mask is a
    per-cell test with no sub-cell weighting, so the boundary is resolved only to
    the grid; :func:`fractional_cell_weights_from_boundary` is the fractional
    alternative where that matters.

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

    # LCFS mask
    inside = (psiN_RZ >= 0.0) & (psiN_RZ <= 1.0) & (Rm > 0.0)

    dV = 2.0 * np.pi * Rm * dA

    V = np.sum(dV[inside])
    if V == 0.0:
        raise ValueError("Total plasma volume is zero.")

    favg = np.sum(f_RZ[inside] * dV[inside]) / V
    return favg, V

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
    alpha_num = np.sum(R_grid * (B_Z_grid**2) * weights * dA)
    alpha_den = np.sum(R_grid * B_p_sq * weights * dA)
    alpha = 0.0 if alpha_den == 0.0 else float(2.0 * alpha_num / alpha_den)

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
    alpha_num = np.sum(R_grid * (B_Z_grid**2) * weights * dA)
    alpha_den = np.sum(R_grid * B_p_sq * weights * dA)
    alpha = np.nan if alpha_den == 0.0 else float(2.0 * alpha_num / alpha_den)

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
    """
    r_seg = np.asarray(r_seg, dtype=float).reshape(-1)
    z_seg = np.asarray(z_seg, dtype=float).reshape(-1)
    index = int(np.argmax(z_seg) if upper else np.argmin(z_seg))
    size = z_seg.size
    if size < 3:
        return float(r_seg[index])
    prev, nxt = (index - 1) % size, (index + 1) % size
    z_prev, z_here, z_next = float(z_seg[prev]), float(z_seg[index]), float(z_seg[nxt])
    denominator = z_prev - 2.0 * z_here + z_next
    if denominator == 0.0:
        return float(r_seg[index])
    # Vertex of the parabola through (-1, z_prev), (0, z_here), (1, z_next).
    shift = 0.5 * (z_prev - z_next) / denominator
    if not np.isfinite(shift) or abs(shift) > 1.0:
        return float(r_seg[index])
    r_here = float(r_seg[index])
    neighbour = float(r_seg[nxt] if shift > 0 else r_seg[prev])
    return r_here + abs(shift) * (neighbour - r_here)


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
