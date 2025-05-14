import numpy as np

def psi_norm(psi, psi_axis, psi_boundary):
    """Normalize psi to [0, 1] between axis and boundary."""
    return (psi - psi_axis) / (psi_boundary - psi_axis)


def magnetic_shear(r, q):
    """
    s = (r/q) * (dq/dr)
    arguement:
    - r: minor radius (scalar or array)
    - q: safety factor (scalar or array, r과 같은 shape)
    - dqdr: derivative of q with respect to r (same shape as q)
    output:
    - s: magnetic shear (same shape as r)
    """
    dqdr = np.gradient(q, r)
    return (r / q) * dqdr


# def magnetic_shear_2d_profile(dim1, dim2, type='retangular', psi, B_dim1, B_dim2):
#     """
#     Calculate magnetic shear profile in 2D equilibrium
    
#     Args:
#         dim1: 1D array of dimension 1 coordinates
#     """

def ballooning_alpha(V, R, p, psi):
    """
    α = (mu0 / (2π²)) * (dV/dψ) * sqrt(V / (2π²R)) * (dp/dψ)
    input:
    - V: volume (1D array)
    - R: major radius (1D array)
    - p: pressure (1D array)
    - psi: poloidal magnetic flux (1D array)
    return:
    - alpha: normalized pressure gradient parameter (1D array)
    """
    mu0 = 4 * np.pi * 1e-7
    dVdpsi = np.gradient(V, psi)
    dpdpsi = np.gradient(p, psi)
    sqrt_term = np.sqrt(V / (2 * np.pi**2 * R))
    factor = mu0 / (2 * np.pi**2)
    alpha = factor * dVdpsi * sqrt_term * dpdpsi
    return alpha