"""
Plasma geometry and shape calculations.

This module provides functions for calculating various plasma geometric parameters
including volume, elongation, triangularity, and other shape factors.

Notation
--------
V      : plasma volume                                [m³]
κ      : elongation                                   [-]
δ      : triangularity                                [-]
eK     : elongation shape factor                      [-]
PF     : peaking factor                               [-]
"""

import numpy as np

from .utils import calculate_volume_weighted_average

# ------------------------------------------------------------------
# Volume and Area Calculations
# ------------------------------------------------------------------

def volume_from_RZ_boundary(R: np.ndarray,
                           Z: np.ndarray) -> float:
    """
    Calculate plasma volume using Green's theorem for axisymmetric geometry.

    Parameters
    ----------
    R : np.ndarray
        Major radius values of boundary points
    Z : np.ndarray
        Vertical position values of boundary points

    Returns
    -------
    float
        Plasma volume [m³]
        
    Notes
    -----
    Uses Green's theorem for axisymmetric geometry:
    V = 2π ∮ (R Z_n) dl where Z_n is the outward normal component.
    Simplified using polygon area: V ≈ 2π A_poly R̄
    """
    # Calculate polygon area
    area = 0.5 * np.abs(np.dot(R, np.roll(Z, 1)) - np.dot(Z, np.roll(R, 1)))
    # R̄: area-weighted mean radius (approximation)
    R_bar = np.mean(R)
    return 2 * np.pi * area * R_bar


# ------------------------------------------------------------------
# Shape Parameters
# ------------------------------------------------------------------

def elongation_from_RZ_boundary(R: np.ndarray,
                               Z: np.ndarray) -> float:
    """
    Calculate plasma elongation κ = Z_max / (2a).

    Parameters
    ----------
    R : np.ndarray
        Major radius values of boundary points
    Z : np.ndarray
        Vertical position values of boundary points

    Returns
    -------
    float
        Elongation κ
    """
    a = (R.max() - R.min()) / 2
    return (Z.max() - Z.min()) / (2 * a)


def triangularity_from_RZ_boundary(R: np.ndarray,
                                  Z: np.ndarray,
                                  R0: float) -> float:
    """
    Calculate plasma triangularity δ = (R0 - R_sep@Z=0) / a.

    Parameters
    ----------
    R : np.ndarray
        Major radius values of boundary points
    Z : np.ndarray
        Vertical position values of boundary points
    R0 : float
        Major radius of magnetic axis

    Returns
    -------
    float
        Triangularity δ
    """
    R_mid = R[np.argmin(np.abs(Z))]  # Boundary intersection at mid-plane
    a = (R.max() - R.min()) / 2
    return (R0 - R_mid) / a


def eK_from_K(K: float) -> float:
    """
    Calculate elongation shape factor eK = (K²-1)/(K²+1).

    Parameters
    ----------
    K : float
        Elongation ratio

    Returns
    -------
    float
        Elongation shape factor eK
    """
    return (K**2 - 1) / (K**2 + 1)


# ------------------------------------------------------------------
# Profile Shape Factors
# ------------------------------------------------------------------

def peaking_factor_from_central_volumeavg(central: float,
                                         volume_avg: float) -> float:
    """
    Calculate peaking factor PF = X(0) / ⟨X⟩.

    Parameters
    ----------
    central : float
        Central value of profile
    volume_avg : float
        Volume-averaged value of profile

    Returns
    -------
    float
        Peaking factor PF

    Notes
    -----
    Commonly used for density, temperature, and pressure profiles.
    """
    return central / volume_avg 