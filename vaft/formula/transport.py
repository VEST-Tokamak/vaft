"""
Plasma transport and confinement calculations.

This module provides functions for calculating various transport parameters
including collisionality, Alfven speed, and energy confinement time.

Notation
--------
ν*     : effective collisionality                     [-]
v_A    : Alfven speed                                [m/s]
c_s    : ion-sound speed                             [m/s]
τ_E    : energy confinement time                     [s]
"""

import numpy as np
from typing import Dict, List, Union

from .constants import (
    MU0, QE, ME, MI_P,
    COLLISIONALITY_COEF,
    _SCALING_COEFS
)

# ------------------------------------------------------------------
# Collisionality and Transport
# ------------------------------------------------------------------

def collisionality_from_n_T_B_R(n_e: float,
                               T_e_keV: float,
                               B_t: float,
                               R0: float) -> float:
    """
    Calculate effective electron collisionality ν* (dimensionless).

    Parameters
    ----------
    n_e : float
        Electron density [10¹⁹ m⁻³]
    T_e_keV : float
        Electron temperature [keV]
    B_t : float
        Toroidal magnetic field [T]
    R0 : float
        Major radius [m]

    Returns
    -------
    float
        Effective collisionality ν*

    Notes
    -----
    Uses ITER IPB-98y2 form:
    ν* = 6.921e-18 n_e R0 / (T_e² B_t)
    """
    return COLLISIONALITY_COEF * n_e * R0 / (T_e_keV**2 * B_t)


# ------------------------------------------------------------------
# Wave Speeds
# ------------------------------------------------------------------

def v_alfven_from_B_n_mi(B: float,
                         n: float,
                         m_i: float = MI_P) -> float:
    """
    Calculate Alfven speed v_A = B / √(μ₀ n m_i).

    Parameters
    ----------
    B : float
        Magnetic field strength [T]
    n : float
        Particle density [m⁻³]
    m_i : float, optional
        Ion mass [kg], by default proton mass

    Returns
    -------
    float
        Alfven speed [m/s]
    """
    return B / np.sqrt(MU0 * n * m_i)


def c_s_from_Te_Ti_mi(T_e_keV: float,
                      T_i_keV: float,
                      m_i: float = MI_P) -> float:
    """
    Calculate ion-sound speed c_s = √((γ_e T_e + γ_i T_i) / m_i).

    Parameters
    ----------
    T_e_keV : float
        Electron temperature [keV]
    T_i_keV : float
        Ion temperature [keV]
    m_i : float, optional
        Ion mass [kg], by default proton mass

    Returns
    -------
    float
        Ion-sound speed [m/s]

    Notes
    -----
    Assumes γ_e = γ_i = 1 (isothermal)
    """
    Te_J = T_e_keV * 1e3 * QE
    Ti_J = T_i_keV * 1e3 * QE
    return np.sqrt((Te_J + Ti_J) / m_i)


# ------------------------------------------------------------------
# Energy Confinement Time
# ------------------------------------------------------------------

def tauE_from_scaling(I_p: float,
                      R0: float,
                      a: float,
                      kappa: float,
                      n_e: float,
                      B0: float,
                      A: float,
                      P_in: float,
                      scaling: str = "H98y2") -> float:
    """
    Calculate energy confinement time using empirical scaling laws.

    Parameters
    ----------
    I_p : float
        Plasma current [A]
    R0 : float
        Major radius [m]
    a : float
        Minor radius [m]
    kappa : float
        Elongation
    n_e : float
        Electron density [m⁻³]
    B0 : float
        Toroidal magnetic field [T]
    A : float
        Atomic mass number
    P_in : float
        Input power [W]
    scaling : str, optional
        Scaling law name, by default "H98y2"

    Returns
    -------
    float
        Energy confinement time [s]

    Notes
    -----
    Generic power-law confinement scaling (ITER-like):
    τ_E = C · I_p^α · R0^β · a^γ · κ^δ · n_e^ε · B0^ζ · A^η / P_in
    
    Unit conversions:
    - I_p [A] → MA
    - n_e [m⁻³] → 10²⁰ m⁻³
    - P_in [W] → MW
    """
    if scaling not in _SCALING_COEFS:
        raise ValueError(f"Unknown scaling '{scaling}'")

    C, α, β, γ, δ, ε, ζ, η = _SCALING_COEFS[scaling]
    I_MA = I_p * 1e-6
    n_20 = n_e * 1e-20
    P_MW = P_in * 1e-6

    return C * I_MA**α * R0**β * a**γ * kappa**δ * n_20**ε * B0**ζ * A**η / P_MW 