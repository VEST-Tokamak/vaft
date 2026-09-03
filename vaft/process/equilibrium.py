from typing import Any

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
    "MIN_FLUX_SURFACE_POINTS",
    "calculate_average_boundary_poloidal_field",
    "calculate_diamagnetism",
    "calculate_reconstructed_diamagnetic_flux",
    "computed_diamagnetism_from_phi",
    "contour_shape_parameters",
    "efit_virial_volume_integrals",
    "extract_flux_surface_contours",
    "flux_surface_quantities",
    "fractional_cell_weights_from_boundary",
    "make_equilibrium_field_interpolator",
    "poloidal_field_at_boundary",
    "prepare_boundary_for_shafranov",
    "psi_to_RZ",
    "psi_to_radial",
    "psi_to_rho",
    "psi_to_rz",
    "r_at_z_extremum",
    "radial_to_psi",
    "rho_to_psi",
    "shafranov_integrals",
    "trace_field_line",
    "volume_average",
    *_PARAMETRIC_EXPORTS,
]


def radial_to_psi(r, psi_R, psi_Z, psi):
    """Convert radial coordinate R to poloidal flux ψ using interpolation at Z=0.
    
    Args:
        r (float): Radial coordinate R
        psi_R (ndarray): R grid points for psi
        psi_Z (ndarray): Z grid points for psi
        psi (ndarray): Poloidal flux values on the R,Z grid
    
    Returns:
        float: Interpolated poloidal flux value at (r, Z=0)
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
    """Convert poloidal flux ψ to normalized radius ρ using q-profile integration.
    
    Args:
        psi_val (float): Poloidal flux value
        q_profile (callable): Safety factor q(ψ) profile function
        psi_axis (float): Poloidal flux at magnetic axis (ψa)
        psi_boundary (float): Poloidal flux at plasma boundary (ψb)
    
    Returns:
        float: Normalized radius ρN
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
    """Convert normalized radius ρ to poloidal flux ψ using numerical root finding.
    
    Args:
        rho (float): Normalized radius ρN
        q_profile (callable): Safety factor q(ψ) profile function
        psi_axis (float): Poloidal flux at magnetic axis (ψa)
        psi_boundary (float): Poloidal flux at plasma boundary (ψb)
        tol (float): Tolerance for root finding
        
    Returns:
        float: Poloidal flux value ψ
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
    """
    Map a 1D profile f(psi_N) onto a 2D (R,Z) grid using psi(R,Z).

    Outside LCFS (psi_N < 0 or > 1), the mapped value is set to 0.

    Returns
    -------
    f_RZ : (Nr, Nz) array
        Profile mapped onto (R,Z), zero outside LCFS.
    psiN_RZ : (Nr, Nz) array
        Normalized poloidal flux on (R,Z).
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
    """
    Reconstructed diamagnetic flux from equilibrium field (physics-only, no ODS).

    Phi_dia = Integral_surf (B_phi_plasma - B_phi_vacuum) dA [Wb].
    Uses psi_to_RZ to map F(psi_N) onto (R,Z). Only plasma region (0 <= psi_N <= 1)
    is integrated. For diamagnetic plasma the result is negative.

    Parameters
    ----------
    R_grid, Z_grid : 1D arrays
        Grid coordinates [m].
    psi_RZ : 2D array, shape (len(R_grid), len(Z_grid))
        Poloidal flux on grid [Wb/rad].
    psi_axis, psi_lcfs : float
        Flux at axis and at LCFS.
    psiN_1d, f_1d : 1D arrays
        Normalized flux and F = R*B_phi [T·m] on 1D profile.
    f_vac_val : float
        F at LCFS (vacuum toroidal field proxy) [T·m].

    Returns
    -------
    float
        Reconstructed diamagnetic flux [Wb].
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
    """
    Diamagnetism μ_i from the volume integral definition (physics-only, no ODS).

    μ_i = (1 / (B_pa² Ω)) ∫_Ω (B_tv² - B_t²) dV

    with B_t = F(ψ)/R, B_tv = F_vac/R, dV = 2π R dR dZ. Only plasma (0 ≤ ψ_N ≤ 1)
    is integrated. Uses psi_to_RZ for F(R,Z). If V_p is None, plasma volume Ω
    is computed from the same grid and mask.

    Parameters
    ----------
    R_grid, Z_grid : 1D arrays
        Grid coordinates [m].
    psi_RZ : 2D array, shape (len(R_grid), len(Z_grid))
        Poloidal flux on grid [Wb/rad].
    psi_axis, psi_lcfs : float
        Flux at axis and at LCFS.
    psiN_1d, f_1d : 1D arrays
        Normalized flux and F = R*B_t [T·m] on 1D profile.
    f_vac_val : float
        F at LCFS (vacuum toroidal field proxy) [T·m].
    B_pa : float
        Average poloidal field at boundary [T] (e.g. μ₀ I_p / L_p).
    V_p : float, optional
        Plasma volume [m³]. If None, computed from grid (2π R dA over plasma).

    Returns
    -------
    float
        Diamagnetism μ_i (dimensionless). Positive ⇒ diamagnetic (B_t < B_tv),
        negative ⇒ paramagnetic (B_t > B_tv). If you expect diamagnetic but get
        negative, check that F_vac is taken at the LCFS (not axis) and that the
        F profile sign convention (F = R*B_φ) is consistent with the equilibrium.
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
    """
    Compute volume average <f>_V on an (R,Z) grid using
    dV = 2*pi*R*dR*dZ.

    Only cells with 0 <= psi_N <= 1 contribute to the integral.
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
    """
    Convert 1D psi profile to r_inboard and r_outboard using 2D psi mapping.
    
    This function creates interpolation functions from 2D psi data at the magnetic
    axis Z position, splits the data into inboard and outboard regions, and maps
    the 1D psi profile to radial coordinates.
    
    Parameters
    ----------
    psi_1d : ndarray
        1D poloidal flux profile to map
    psi_2d_slice : ndarray
        2D psi values at magnetic axis Z position (from profiles_2d.0.psi[:, z_idx])
    grid_r : ndarray
        R grid points corresponding to psi_2d_slice
    boundary_r : ndarray
        Boundary R coordinates to determine r_min and r_max
    r_axis : float
        Magnetic axis R coordinate
    
    Returns
    -------
    r_inboard : ndarray
        Inboard radial coordinates corresponding to psi_1d
    r_outboard : ndarray
        Outboard radial coordinates corresponding to psi_1d
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
    """
    자속(Psi) 격자 데이터를 이용하여 경계면(Boundary)에서의 
    Poloidal Magnetic Field (Bp) 벡터와 크기를 계산합니다.

    EFIT의 'seva2d' 서브루틴과 유사한 역할을 수행합니다.

    수식:
        B_R = -(1/R) * (dPsi/dZ)
        B_Z =  (1/R) * (dPsi/dR)
        B_p = sqrt(B_R^2 + B_Z^2)

    Args:
        R_grid_1d (np.array): R 격자 좌표 1D 배열 (m)
        Z_grid_1d (np.array): Z 격자 좌표 1D 배열 (m)
        psi_grid (2D array): 격자 위 자속 값 (Weber/rad). Shape은 (len(R), len(Z)) 여야 함.
        R_bdry (np.array): 경계면 R 좌표 배열 (m)
        Z_bdry (np.array): 경계면 Z 좌표 배열 (m)

    Returns:
        tuple: (B_p_bdry, B_R_bdry, B_Z_bdry)
            - B_p_bdry: 경계면에서의 B_p 크기 (Tesla)
            - B_R_bdry: 경계면에서의 B_R 성분 (Tesla)
            - B_Z_bdry: 경계면에서의 B_Z 성분 (Tesla)
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
    """
    플라즈마 경계에서의 평균 Poloidal Magnetic Field (B_pa)를 계산합니다.
    
    Formula:
        B_pa = ∮ dl B_p / ∮ dl
        
    Args:
        R_bdry (np.array): 경계면의 R 좌표 배열 (m)
        Z_bdry (np.array): 경계면의 Z 좌표 배열 (m)
        B_p_bdry (np.array): 경계면에서의 Poloidal Field (T)
        
    Returns:
        float: B_pa (Average boundary poloidal field)
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
    """
    Normalize boundary for Shafranov computations:
    finite-only -> closed -> degenerate segment removal -> CCW orientation -> arc-length resample.
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
    """
    Estimate EFIT-like fractional cell weights from boundary geometry.

    The returned weight map is in [0, 1], where each entry approximates the
    area fraction of the local control cell that lies inside the plasma
    polygon. This is an internal replacement for externally provided `www`.
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
    """
    Shafranov Integrals (S1, S2, S3) 및 Alpha 파라미터를 계산합니다.
    플라즈마 마스크를 경계면 좌표(R_bdry, Z_bdry)로부터 직접 생성합니다.

    Args:
        R_bdry, Z_bdry (np.array): 플라즈마 경계(LCFS) 좌표 1D 배열
        B_p_bdry (np.array): 경계에서의 Poloidal Magnetic Field 1D 배열
        R_grid, Z_grid (2D array): 전체 계산 영역의 격자 좌표 (meshgrid 형태)
        B_R_grid, B_Z_grid (2D array): 전체 영역의 자기장 (Alpha 계산용)
        R_0, Z_0 (float, optional): Major Radius 및 중심 높이. 
                                    None일 경우 경계의 기하학적 중심 사용.

    Returns:
        tuple: (S1, S2, S3, alpha)
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
    """
    EFIT-style weighted volume integrals on the poloidal grid.

    Returns alpha, RT, and diamagnetic-flux components in SI units.
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

    volume = float(np.nansum(2.0 * np.pi * R_grid * weights * dA))
    return {
        "alpha": alpha,
        "rt": RT,
        "phi_dia_comp": phi_dia_comp,
        "volume": volume,
    }


def computed_diamagnetism_from_phi(
    phi_dia_comp: float,
    B_t0: float,
    R_0: float,
    volume: float,
    B_ref: float,
) -> float:
    """
    Compute EFIT-style xmui from computed diamagnetic flux.

    xmui = (4*pi*B_t0*R_0 / (V*B_ref^2)) * Phi_dia_comp
    """
    if volume <= 0.0 or B_ref <= 0.0:
        raise ValueError("volume and B_ref must be positive.")
    return float((4.0 * np.pi * B_t0 * R_0 * phi_dia_comp) / (volume * B_ref**2))


psi_to_RZ = psi_to_rz


def extract_flux_surface_contours(
    psi_grid: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    psi_axis: float,
    psi_boundary: float,
    levels_norm: Any,
) -> dict[float, list[tuple[np.ndarray, np.ndarray]]]:
    """Extract iso-normalized-psi contours from a 2D (R, Z) psi grid.

    ``psi_grid`` must be shaped ``(len(R), len(Z))`` (first axis indexed by
    ``R``, second by ``Z``), matching the standard IMAS
    ``equilibrium.time_slice[:].profiles_2d[:].psi`` layout with
    ``grid.dim1``/``grid.dim2`` as ``R``/``Z``. ``levels_norm`` are normalized
    psi values (0 = magnetic axis, 1 = boundary/LCFS).

    Returns a dict keyed by each requested level, each value a list of
    ``(R_pts, Z_pts)`` contour segments (a level can have more than one
    disconnected contour, e.g. inside a diverted plasma). A level with no
    contour in the grid maps to an empty list.
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

    Returns ``volume``, ``area``, ``surface``, ``elongation``,
    ``triangularity_upper``, ``triangularity_lower``, ``r_inboard`` and
    ``r_outboard``. Volume revolves the contour with the exact
    ``V = pi * closed_integral(R^2 dZ)``; area is the shoelace formula; surface
    is ``closed_integral 2*pi*R dl``.

    Raises ``ValueError`` on a degenerate contour (zero minor radius).
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
    """R where the contour reaches its highest (or lowest) point.

    Taking R at the sampled vertex of extreme Z is off by several percent in
    triangularity, because the true extremum falls between vertices. Fitting a
    parabola to Z over the three points around it locates the extremum to
    sub-vertex resolution, and R is interpolated there.
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
    r"""Flux-surface averages and shape, on each normalized-psi level.

    ``psi_grid`` must be in **weber per radian** and shaped ``(len(R), len(Z))``,
    matching :func:`extract_flux_surface_contours`. ``psi_axis`` and
    ``psi_boundary`` are in the same unit as ``psi_grid`` only insofar as they
    normalize it -- the normalization is scale-free, but the ``|grad psi|`` that
    weights the averages is not, which is why the per-radian contract matters.

    The flux-surface average weights by the volume element,
    ``<X> = closed_integral(X R dl / |grad psi|) / closed_integral(R dl / |grad psi|)``,
    ``bp_dl`` is ``closed_integral(B_p dl)`` and ``length_pol`` the poloidal
    perimeter; the first is what ``int(B_p^2 dV)`` needs, since the volume
    element cancels one power of ``B_p``.

    so ``gm1 = <1/R^2>``, ``gm8 = <R>``, ``gm9 = <1/R>`` and, when ``f_profile``
    supplies ``F = R B_phi`` on the same levels, ``gm5 = <B^2>`` with
    ``|B|^2 = (|grad psi|^2 + F^2) / R^2``. ``dvolume_dpsi`` and ``darea_dpsi``
    are per radian; a caller storing them against a weber psi divides by 2*pi.

    ``axis_rz`` selects the confined segment when a level returns several (see
    :func:`_enclosing_segment`); ``boundary`` replaces the traced contour at
    ``levels_norm == 1``, which is both exact and cheaper when the ODS already
    carries ``boundary.outline``.

    End points. Levels whose contour is missing or below ``min_points`` come back
    as NaN and are then filled by interpolation against ``sqrt(psi_N)``, the
    coordinate in which near-axis geometry is linear. At ``levels_norm == 0`` the
    surface is a point: ``volume``, ``area`` and ``surface`` are exactly 0,
    ``gm1 -> 1/R_axis^2``, ``gm8 -> R_axis``, ``gm9 -> 1/R_axis`` and
    ``r_inboard = r_outboard = R_axis``; the shape parameters and the two
    derivatives are undefined there and are extrapolated from the innermost
    resolved surface.

    The innermost one or two levels are the least accurate everywhere, because
    ``|grad psi|`` is small and varies fastest there. That is a property of the
    map, not of this routine: on the packaged OMFIT reference the *stored*
    ``dvolume_dpsi`` runs 8.53, 22.85, 29.59, 29.05, 28.85 from the axis out --
    a ramp that cannot be physical, since ``dV/dpsi`` approaches a finite limit
    -- while the trace here gives a smooth 30.8, 29.3, 29.3. Compare against a
    reference only outside ``psi_N ~ 0.05``, and check the near-axis values
    against ``int dV/dpsi = V`` instead.

    Returns a dict over :data:`FLUX_SURFACE_QUANTITIES`, each an array as long as
    ``levels_norm``.
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

    # Gaps are filled against sqrt(psi_N), not psi_N: near the axis a flux
    # surface's linear size goes as sqrt(psi_N), so every quantity that vanishes
    # there is linear in sqrt and badly curved in psi_N.  Interpolating a dropped
    # innermost level in psi_N underestimates `surface` by a third.
    #
    # `np.interp` requires an increasing `xp` and returns nonsense rather than
    # raising when it does not get one, so the levels are sorted here instead of
    # assumed: a psi profile stored boundary-first is a real input, and it
    # corrupted only the quantities that happened to need a gap filled.
    coordinate = np.sqrt(np.clip(levels, 0.0, None))
    order = np.argsort(coordinate, kind="stable")
    for name, values in out.items():
        missing = ~np.isfinite(values)
        if missing.any() and not missing.all():
            good_sorted = order[np.isfinite(values[order])]
            values[missing] = np.interp(
                coordinate[missing], coordinate[good_sorted], values[good_sorted]
            )
    return out


def make_equilibrium_field_interpolator(
    R_grid_1d: np.ndarray,
    Z_grid_1d: np.ndarray,
    psi_grid: np.ndarray,
    psi_1d: np.ndarray,
    f_1d: np.ndarray,
    cocos=None,
):
    """Build a callable ``(R, Z) -> (B_R, B_Z, B_phi)`` for one equilibrium time slice.

    Same convention handling as :func:`poloidal_field_at_boundary`, via Sauter
    Eq. 20: ``B_R = k (1/R) dPsi/dZ``, ``B_Z = -k (1/R) dPsi/dR`` with
    ``k = sigma_RphiZ sigma_Bp / (2*pi)**e_Bp`` (from a bicubic spline
    over the 2D psi grid, built once and reused for every evaluation), and
    ``B_phi = F(psi)/R`` with ``F = R*B_phi`` interpolated from
    ``profiles_1d.{psi, f}`` (``psi_1d``, ``f_1d``). ``psi_grid`` must be
    shaped ``(len(R_grid_1d), len(Z_grid_1d))``, matching
    :func:`extract_flux_surface_contours`.

    ``F(psi)`` is only defined over ``psi_1d`` (axis to boundary); points
    outside that range (e.g. beyond the LCFS, in the scrape-off layer) clip to
    the nearest edge value of ``F`` -- the same clip-and-interpolate
    convention already used by :func:`psi_to_rz`. This is an approximation
    for field lines that leave the confined region: the toroidal field there
    is treated as this clipped ``F(psi)/R`` rather than the true vacuum field.
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
    """Trace a magnetic field line from ``(R0, Z0, phi0)`` using the equilibrium field.

    Integrates ``dR/dphi = R*B_R/B_phi``, ``dZ/dphi = R*B_Z/B_phi`` with
    classic fixed-step 4th-order Runge-Kutta (RK4) in the toroidal angle
    ``phi``, using ``dphi`` (radians) as the step size. ``b_field`` is a
    callable ``(R, Z) -> (B_R, B_Z, B_phi)``, e.g. from
    :func:`make_equilibrium_field_interpolator`.

    ``direction`` is ``"forward"`` (increasing phi, default), ``"backward"``
    (decreasing phi), or ``"both"`` (both branches concatenated, anchored at
    the start point).

    Terminates when any of the following is hit first: the traced point
    leaves the ``(wall_r, wall_z)`` limiter polygon (if given), the
    cumulative 3D arc length exceeds ``max_length_m``, the point leaves
    ``r_bounds``/``z_bounds`` (if given -- a safety net against extrapolating
    far outside the equilibrium grid), or ``B_phi`` is (numerically) zero.
    The stopping reason is reported in the returned ``termination_reason``.

    Returns a dict with ``phi``, ``R``, ``Z`` (1D arrays, meters/radians, in
    increasing-phi order regardless of ``direction``), ``arc_length_m``
    (meters, monotonically increasing from ``0.0`` at the first returned
    point to the total path length at the last -- i.e. cumulative distance
    along the array in its returned order, not distance from ``phi0``), and
    ``termination_reason``.
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
