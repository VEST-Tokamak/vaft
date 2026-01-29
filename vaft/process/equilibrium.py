from typing import Any


import numpy as np
from scipy.interpolate import interp1d


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

def psi_to_RZ(
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




