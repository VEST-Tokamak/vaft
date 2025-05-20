"""
Plasma stability calculations.

This module provides functions for calculating various stability parameters
including beta limits, ballooning stability, and MHD stability criteria.

Notation
--------
β_N    : normalized beta                              [-]
β_p    : poloidal beta                                [-]
β_t    : toroidal beta                                [-]
q_95   : safety factor at 95% flux surface            [-]
α      : ballooning parameter                         [-]
s      : magnetic shear                               [-]
"""

import numpy as np
from typing import Union, Tuple

from .constants import MU0
from .utils import gradient

# ------------------------------------------------------------------
# Beta Limits
# ------------------------------------------------------------------

def beta_N_from_beta_a_B0_Ip(beta: float,
                            a: float,
                            B0: float,
                            I_p: float) -> float:
    """
    Calculate normalized beta β_N = β · a · B0 / I_p.

    Parameters
    ----------
    beta : float
        Plasma beta
    a : float
        Minor radius [m]
    B0 : float
        Toroidal magnetic field [T]
    I_p : float
        Plasma current [A]

    Returns
    -------
    float
        Normalized beta β_N
    """
    return beta * a * B0 / I_p


def beta_pol_from_beta_tor(beta_tor: float,
                          q_95: float) -> float:
    """
    Calculate poloidal beta from toroidal beta.

    Parameters
    ----------
    beta_tor : float
        Toroidal beta
    q_95 : float
        Safety factor at 95% flux surface

    Returns
    -------
    float
        Poloidal beta β_p
    """
    return beta_tor * q_95**2


def beta_tor_from_beta_pol(beta_pol: float,
                          q_95: float) -> float:
    """
    Calculate toroidal beta from poloidal beta.

    Parameters
    ----------
    beta_pol : float
        Poloidal beta
    q_95 : float
        Safety factor at 95% flux surface

    Returns
    -------
    float
        Toroidal beta β_t
    """
    return beta_pol / q_95**2


# ------------------------------------------------------------------
# Ballooning Stability
# ------------------------------------------------------------------

def ballooning_alpha_from_p_B(p: Union[float, np.ndarray],
                            B: Union[float, np.ndarray],
                            R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate ballooning parameter α = -2μ₀ R dp/dψ / B².

    Parameters
    ----------
    p : Union[float, np.ndarray]
        Pressure [Pa]
    B : Union[float, np.ndarray]
        Magnetic field strength [T]
    R : Union[float, np.ndarray]
        Major radius [m]

    Returns
    -------
    Union[float, np.ndarray]
        Ballooning parameter α
    """
    return -2 * MU0 * R * gradient(R, p) / B**2


def ballooning_stability_criterion(alpha: Union[float, np.ndarray],
                                 s: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculate ballooning stability criterion.

    Parameters
    ----------
    alpha : Union[float, np.ndarray]
        Ballooning parameter
    s : Union[float, np.ndarray]
        Magnetic shear

    Returns
    -------
    Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]
        Tuple of (stability parameter, critical alpha)
    """
    alpha_crit = 0.6 * s
    return alpha - alpha_crit, alpha_crit


# ------------------------------------------------------------------
# MHD Stability
# ------------------------------------------------------------------

def kink_stability_criterion(q_95: float,
                           beta_N: float) -> Tuple[float, float]:
    """
    Calculate kink stability criterion.

    Parameters
    ----------
    q_95 : float
        Safety factor at 95% flux surface
    beta_N : float
        Normalized beta

    Returns
    -------
    Tuple[float, float]
        Tuple of (stability parameter, critical beta_N)
    """
    beta_N_crit = 0.028 * q_95
    return beta_N - beta_N_crit, beta_N_crit


def sawtooth_stability_criterion(q_0: float,
                               beta_pol: float) -> Tuple[float, float]:
    """
    Calculate sawtooth stability criterion.

    Parameters
    ----------
    q_0 : float
        Safety factor at magnetic axis
    beta_pol : float
        Poloidal beta

    Returns
    -------
    Tuple[float, float]
        Tuple of (stability parameter, critical beta_pol)
    """
    beta_pol_crit = 0.3 * (1 - q_0)
    return beta_pol - beta_pol_crit, beta_pol_crit 