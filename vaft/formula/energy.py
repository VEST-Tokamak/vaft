"""
Plasma energy calculations.

This module provides functions for calculating various energy-related parameters
including stored energy, power balance, and heating power.

Notation
--------
W      : stored energy                                [J]
P_heat : heating power                                [W]
P_rad  : radiated power                               [W]
P_α    : alpha particle heating power                 [W]
P_loss : power loss                                   [W]
"""

import numpy as np
from typing import Union, Tuple

from .constants import (
    MU0, QE, ME, MI_P,
    E_ALPHA, SIGMA_V_COEF,
    SPITZER_RESISTIVITY_COEF
)

from .utils import calculate_volume_weighted_average

# ------------------------------------------------------------------
# Stored Energy
# ------------------------------------------------------------------

def stored_energy_from_p_V(p: Union[float, np.ndarray],
                          V: float) -> Union[float, np.ndarray]:
    """
    Calculate stored energy W = ∫ p dV.

    Parameters
    ----------
    p : Union[float, np.ndarray]
        Pressure [Pa]
    V : float
        Plasma volume [m³]

    Returns
    -------
    Union[float, np.ndarray]
        Stored energy [J]
    """
    return p * V


def stored_energy_from_beta_V(beta: float,
                            B0: float,
                            V: float) -> float:
    """
    Calculate stored energy from beta.

    Parameters
    ----------
    beta : float
        Plasma beta
    B0 : float
        Toroidal magnetic field [T]
    V : float
        Plasma volume [m³]

    Returns
    -------
    float
        Stored energy [J]

    Notes
    -----
    Uses relation W = β B0² V / (2μ₀)
    """
    return beta * B0**2 * V / (2 * MU0)


# ------------------------------------------------------------------
# Power Balance
# ------------------------------------------------------------------

def power_balance(P_heat: float,
                 P_rad: float,
                 P_alpha: float,
                 P_loss: float) -> Tuple[float, float]:
    """
    Calculate power balance.

    Parameters
    ----------
    P_heat : float
        External heating power [W]
    P_rad : float
        Radiated power [W]
    P_alpha : float
        Alpha particle heating power [W]
    P_loss : float
        Power loss [W]

    Returns
    -------
    Tuple[float, float]
        Tuple of (net power, power balance ratio)
    """
    P_net = P_heat + P_alpha - P_rad - P_loss
    P_ratio = P_net / (P_heat + P_alpha)
    return P_net, P_ratio


def alpha_power(n_D: float,
               n_T: float,
               T_keV: float,
               V: float) -> float:
    """
    Calculate alpha particle heating power.

    Parameters
    ----------
    n_D : float
        Deuterium density [10¹⁹ m⁻³]
    n_T : float
        Tritium density [10¹⁹ m⁻³]
    T_keV : float
        Temperature [keV]
    V : float
        Plasma volume [m³]

    Returns
    -------
    float
        Alpha particle heating power [W]

    Notes
    -----
    Uses simplified fusion power formula
    """
    # Convert to m⁻³
    n_D = n_D * 1e19
    n_T = n_T * 1e19
    
    # Fusion cross section (simplified)
    sigma_v = SIGMA_V_COEF * T_keV**2  # m³/s
    
    return n_D * n_T * sigma_v * E_ALPHA * V


# ------------------------------------------------------------------
# Heating Power
# ------------------------------------------------------------------

def ohmic_heating_power(I_p: float,
                       R0: float,
                       a: float,
                       T_e_keV: float,
                       Z_eff: float = 1.0) -> float:
    """
    Calculate ohmic heating power.

    Parameters
    ----------
    I_p : float
        Plasma current [A]
    R0 : float
        Major radius [m]
    a : float
        Minor radius [m]
    T_e_keV : float
        Electron temperature [keV]
    Z_eff : float, optional
        Effective charge number, by default 1.0

    Returns
    -------
    float
        Ohmic heating power [W]

    Notes
    -----
    Uses Spitzer resistivity
    """
    # Spitzer resistivity
    eta = SPITZER_RESISTIVITY_COEF * Z_eff / T_e_keV**1.5  # Ω·m
    
    # Plasma resistance
    R_plasma = eta * 2 * np.pi * R0 / (np.pi * a**2)
    
    return I_p**2 * R_plasma


def auxiliary_heating_power(P_aux: float,
                          eta_CD: float) -> Tuple[float, float]:
    """
    Calculate auxiliary heating power components.

    Parameters
    ----------
    P_aux : float
        Total auxiliary power [W]
    eta_CD : float
        Current drive efficiency [A/W]

    Returns
    -------
    Tuple[float, float]
        Tuple of (heating power, current drive power)
    """
    P_CD = P_aux / (1 + eta_CD)
    P_heat = P_aux - P_CD
    return P_heat, P_CD 