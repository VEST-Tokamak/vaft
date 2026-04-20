from typing import Any

from scipy.interpolate import RectBivariateSpline

import numpy as np
from scipy.interpolate import interp1d

from vaft.formula.constants import MU0


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

def poloidal_field_at_boundary(R_grid_1d, Z_grid_1d, psi_grid, R_bdry, Z_bdry):
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

    # 3. 자기장 계산 (Cylindrical Coordinates)
    # 주의: Psi 단위가 Weber(Total flux)라면 2*pi로 나누어야 하고, 
    #       EFIT처럼 Weber/rad 단위라면 아래 수식이 맞습니다.
    #       여기서는 일반적인 EFIT 관례인 Weber/rad를 따릅니다.
    
    # B_R = -(1/R) * dPsi/dZ
    B_R_bdry = -(1.0 / R_bdry) * dPsi_dZ
    
    # B_Z = (1/R) * dPsi/dR
    B_Z_bdry = (1.0 / R_bdry) * dPsi_dR
    
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
        nR = np.where(dl > 0.0, -dZ_b / dl, 0.0)
        nZ = np.where(dl > 0.0, dR_b / dl, 0.0)

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
        RT_num = np.sum(R_grid * G * weights * dA)
        RT_den = np.sum(G * weights * dA)
        if RT_den != 0.0:
            RT = float(RT_num / RT_den)

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
