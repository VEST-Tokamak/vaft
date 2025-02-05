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

def psi_to_radial(psi_val, psi_R, psi_Z, psi):
    """Find R,Z coordinates for a given poloidal flux value ψ.
    
    Args:
        psi_val (float): Target poloidal flux value
        psi_R (ndarray): R grid points for psi
        psi_Z (ndarray): Z grid points for psi
        psi (ndarray): Poloidal flux values on the R,Z grid
    
    Returns:
        tuple: (R,Z) coordinates where psi = psi_val
    """
    from scipy.optimize import fsolve
    
    def objective(x):
        r, z = x
        return radial_to_psi(r, psi_R, psi_Z, psi) - psi_val
    
    # Use magnetic axis as initial guess
    r0 = psi_R[np.argmin(np.abs(psi))]
    z0 = psi_Z[np.argmin(np.abs(psi))]
    
    # Solve for R,Z coordinates
    solution = fsolve(objective, [r0, z0])
    return tuple(solution)

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



