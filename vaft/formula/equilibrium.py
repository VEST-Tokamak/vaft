"""
Plasma equilibrium mapping formulas and calculations.

This module provides functions for calculating various plasma equilibrium parameters
including poloidal flux, toroidal flux, safety factor, and stability parameters.

Notation
--------
ψ      : poloidal magnetic flux                     [Wb]
ψ_a    : ψ at magnetic axis                         [Wb]
ψ_b    : ψ at plasma boundary                       [Wb]
Φ(ψ)   : toroidal flux through surface C(ψ)         [Wb]
Φ_b    : Φ(ψ_b)                                     [Wb]
ρ_N    : normalised minor-radius (0 at axis, 1 at edge)
q      : safety factor                              [-]

"""

import warnings
from typing import Union, Tuple
import numpy as np

from .constants import MU0
from .utils import (
    gradient,
    trapz_integral,
    normalize_profile,
    calculate_poloidal_flux,
    calculate_toroidal_flux
)

# ------------------------------------------------------------------
# Poloidal Flux Calculations
# ------------------------------------------------------------------

def psi_from_RBtheta(R: np.ndarray,
                     B_theta: np.ndarray,
                     l: np.ndarray,
                     psi_axis: float = 0.0) -> float:
    """
    Calculate poloidal flux from line integral of R*B_theta.

    Parameters
    ----------
    R : np.ndarray
        Major radius values along field line
    B_theta : np.ndarray
        Poloidal magnetic field values
    l : np.ndarray
        Path length values
    psi_axis : float, optional
        Poloidal flux at magnetic axis, by default 0.0

    Returns
    -------
    float
        Poloidal flux value
    """
    return calculate_poloidal_flux(R, B_theta, l, psi_axis)


def psi_normalised(psi: Union[np.ndarray, float],
                   psi_axis: float,
                  psi_boundary: float) -> Union[np.ndarray, float]:
    """
    Calculate normalized poloidal flux.

    Parameters
    ----------
    psi : Union[np.ndarray, float]
        Poloidal flux values
    psi_axis : float
        Poloidal flux at magnetic axis
    psi_boundary : float
        Poloidal flux at plasma boundary

    Returns
    -------
    Union[np.ndarray, float]
        Normalized poloidal flux: (ψ − ψ_a) / (ψ_b − ψ_a)
    """
    return normalize_profile(psi, psi_axis, psi_boundary)


# Backwards compatibility alias
def normalize_psi(*args, **kw):  # noqa: N802
    """
    DEPRECATED: Use psi_normalised instead.
    
    This function is kept for backwards compatibility.
    """
    warnings.warn("`normalize_psi` is deprecated → use `psi_normalised`",
                 DeprecationWarning, stacklevel=2)
    return psi_normalised(*args, **kw)


# ------------------------------------------------------------------
# Toroidal Flux Calculations
# ------------------------------------------------------------------

def phi_from_Bphi(B_phi: np.ndarray,
                  dA: np.ndarray) -> float:
    """
    Calculate toroidal flux through surface C(ψ).

    Parameters
    ----------
    B_phi : np.ndarray
        Toroidal magnetic field values
    dA : np.ndarray
        Area elements corresponding to B_phi grid points

    Returns
    -------
    float
        Toroidal flux value
    """
    return calculate_toroidal_flux(B_phi, dA)


def rhoN_from_phi(phi: Union[np.ndarray, float],
                  phi_boundary: float) -> Union[np.ndarray, float]:
    """
    Calculate normalized minor radius from toroidal flux.

    Parameters
    ----------
    phi : Union[np.ndarray, float]
        Toroidal flux values
    phi_boundary : float
        Toroidal flux at plasma boundary

    Returns
    -------
    Union[np.ndarray, float]
        Normalized minor radius: √(Φ(ψ) / Φ_b)
    """
    return np.sqrt(phi / phi_boundary)


# ------------------------------------------------------------------
# Safety Factor Calculations
# ------------------------------------------------------------------

def q_from_phi(psi: np.ndarray,
               phi: np.ndarray) -> np.ndarray:
    """
    Calculate safety factor from toroidal flux gradient.

    Parameters
    ----------
    psi : np.ndarray
        Poloidal flux values
    phi : np.ndarray
        Toroidal flux values

    Returns
    -------
    np.ndarray
        Safety factor profile: dΦ/dψ
    """
    return gradient(psi, phi)


def q_from_rhoN(psiN: np.ndarray,
                rhoN: np.ndarray,
                C: float = 1.0) -> np.ndarray:
    """
    Calculate safety factor in normalized flux space.

    Parameters
    ----------
    psiN : np.ndarray
        Normalized poloidal flux values
    rhoN : np.ndarray
        Normalized minor radius values
    C : float, optional
        Scaling factor, by default 1.0

    Returns
    -------
    np.ndarray
        Safety factor profile: C · ρ_N · dρ_N/dψ_N
    """
    drhoN_dpsiN = gradient(psiN, rhoN)
    return C * rhoN * drhoN_dpsiN


def rhoN_from_qpsiN(psiN: np.ndarray,
                    qpsiN: np.ndarray) -> np.ndarray:
    """
    Calculate normalized minor radius from safety factor profile.

    Parameters
    ----------
    psiN : np.ndarray
        Normalized poloidal flux values
    qpsiN : np.ndarray
        Safety factor profile in normalized flux space

    Returns
    -------
    np.ndarray
        Normalized minor radius profile: √(∫₀^{ψ_N} q(ψ′_N) dψ′_N / ∫₀¹ q(ψ′_N) dψ′_N)
    """
    # Cumulative integral using trapezoidal rule to preserve quartiles
    num = np.array([trapz_integral(psiN[:i+1], qpsiN[:i+1]) for i in range(len(psiN))])
    den = trapz_integral(psiN, qpsiN)
    return np.sqrt(num / den)


# ------------------------------------------------------------------
# Stability Parameter Calculations
# ------------------------------------------------------------------

def shear_from_r_q(r: np.ndarray,
                   q: np.ndarray) -> np.ndarray:
    """
    Calculate magnetic shear profile.

    Parameters
    ----------
    r : np.ndarray
        Minor radius values
    q : np.ndarray
        Safety factor profile

    Returns
    -------
    np.ndarray
        Magnetic shear profile: (r/q) · dq/dr
    """
    dqdr = gradient(r, q)
    return (r / q) * dqdr


# Alias for backwards compatibility
magnetic_shear = shear_from_r_q  # noqa: E305


def alpha_from_V_R_p_psi(V: np.ndarray,
                         R: np.ndarray,
                         p: np.ndarray,
                         psi: np.ndarray) -> np.ndarray:
    """
    Calculate ballooning stability parameter.

    Parameters
    ----------
    V : np.ndarray
        Plasma volume profile
    R : np.ndarray
        Major radius profile
    p : np.ndarray
        Pressure profile
    psi : np.ndarray
        Poloidal flux profile

    Returns
    -------
    np.ndarray
        Ballooning stability parameter: (μ₀/2π²) · (dV/dψ) · √(V/2π²R) · (dp/dψ)
    """
    dVdpsi = gradient(psi, V)
    dpdpsi = gradient(psi, p)
    sqrt_term = np.sqrt(V / (2 * np.pi**2 * R))
    factor = MU0 / (2 * np.pi**2)
    return factor * dVdpsi * sqrt_term * dpdpsi


# Alias for backwards compatibility
ballooning_alpha = alpha_from_V_R_p_psi


def normalized_plasma_current(Ip: Union[float, np.ndarray],
                            R: Union[float, np.ndarray],
                            a: Union[float, np.ndarray],
                            Bt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate normalized plasma current.

    Parameters
    ----------
    Ip : Union[float, np.ndarray]
        Plasma current [A]
    R : Union[float, np.ndarray]
        Major radius [m]
    a : Union[float, np.ndarray]
        Minor radius [m]
    Bt : Union[float, np.ndarray]
        Toroidal field [T]

    Returns
    -------
    Union[float, np.ndarray]
        Normalized plasma current: I_p / (a * B_t) [MA/(m·T)]

    Notes
    -----
    The plasma current is converted from amperes to megaamperes internally.
    
    Formula:
    \[
    I_p^* = \frac{I_p}{a B_t}
    \]

    Reference
    ---------
    Phys. Plasmas 23, 072508 (2016); https://doi.org/10.1063/1.4959808
    """
    Ip = Ip / 1e6  # Convert A to MA
    return Ip / (a * Bt)


def kink_safety_factor(R: Union[float, np.ndarray],
                      a: Union[float, np.ndarray],
                      kappa: Union[float, np.ndarray],
                      Ip: Union[float, np.ndarray],
                      Bt: Union[float, np.ndarray],
                      type_: str) -> Tuple[Union[float, np.ndarray], ...]:
    """
    Calculate kink safety factor and related stability parameters.

    Parameters
    ----------
    R : Union[float, np.ndarray]
        Major radius [m]
    a : Union[float, np.ndarray]
        Minor radius [m]
    kappa : Union[float, np.ndarray]
        Elongation [-]
    Ip : Union[float, np.ndarray]
        Plasma current [A]
    Bt : Union[float, np.ndarray]
        Toroidal field [T]
    type_ : str
        Tokamak type: 'circular', 'conventional', or 'ST'

    Returns
    -------
    Tuple[Union[float, np.ndarray], ...]
        Tuple containing:
        - q_kink : Kink safety factor [-]
        - q_min : Minimum kink safety factor for stability [-]
        - beta_max : Maximum beta for ballooning-kink instability [-]
        - beta_crit : Troyon limit [-]
        - ip_max : Maximum plasma current for kink stability [A]

    Notes
    -----
    Based on surface current model from Friedberg 2008 Plasma Physics and Fusion Energy.
    Different calculations are used depending on the tokamak type:
    - circular: Assumes elongation = 1
    - conventional: Uses g-factor correction
    - ST: Spherical tokamak specific calculations

    Reference
    ---------
    Friedberg 2008 Plasma Physics and Fusion Energy (eq 13.158, p 405)
    """
    mu0 = 4 * np.pi * 1e-7
    epsilon = a / R

    if type_ == 'circular':
        q_kink = 2 * np.pi * a**2 * Bt / (mu0 * Ip * R)
        beta_max = None
        beta_crit = None
    elif type_ == 'conventional':
        q_kink = 2 * np.pi * a**2 * kappa * Bt / (mu0 * Ip * R)
        g_factor = 1 / kappa * (1 + 4 / np.pi**2 * (kappa**2 - 1))
        q_kink *= g_factor
        beta_max = np.pi**2 / 16 * kappa * epsilon / q_kink**2
        beta_crit = 0.14 * epsilon * kappa / q_kink
    elif type_ == 'ST':
        q_kink = 2 * np.pi * a**2 * Bt / (mu0 * Ip * R) * (1 + kappa**2 / 2)
        beta_max = 0.072 * (1 + kappa**2) / 2 * epsilon
        betaN_braket = 0.03 * (q_kink - 1) / ((3/4)**4 + (q_kink - 1)**4)**(1/4)
        beta_crit = 5 * betaN_braket * (1 + kappa**2) / 2 * epsilon / q_kink
    else:
        raise ValueError("Invalid type specified. Must be 'circular', 'conventional', or 'ST'")

    q_min = 1 + kappa / 2
    ip_max = q_kink * Ip * 2 / (1 + kappa)

    return q_kink, q_min, beta_max, beta_crit, ip_max
