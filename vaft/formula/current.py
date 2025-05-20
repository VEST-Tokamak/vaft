"""
Plasma current calculations.

This module provides functions for calculating various current-related parameters
including current density, current drive efficiency, and bootstrap current.

Notation
--------
j      : current density                              [A/m²]
η_CD   : current drive efficiency                     [A/W]
f_BS   : bootstrap current fraction                   [-]
I_p    : plasma current                               [A]
"""

import numpy as np
from typing import Union, Tuple

from .constants import MU0
from .utils import gradient

# Physical constants
QE = 1.602176634e-19        # [C] - Elementary charge
ME = 9.10938356e-31         # [kg] - Electron mass
MI_P = 1.67262192e-27       # [kg] - Proton mass

# ------------------------------------------------------------------
# Current Density
# ------------------------------------------------------------------

def current_density_from_B(B: Union[float, np.ndarray],
                          R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate current density j = (∇ × B) / μ₀.

    Parameters
    ----------
    B : Union[float, np.ndarray]
        Magnetic field [T]
    R : Union[float, np.ndarray]
        Major radius [m]

    Returns
    -------
    Union[float, np.ndarray]
        Current density [A/m²]
    """
    return gradient(R, B) / MU0


def current_density_from_psi(psi: Union[float, np.ndarray],
                           R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate current density from poloidal flux.

    Parameters
    ----------
    psi : Union[float, np.ndarray]
        Poloidal flux [Wb]
    R : Union[float, np.ndarray]
        Major radius [m]

    Returns
    -------
    Union[float, np.ndarray]
        Current density [A/m²]
    """
    return -gradient(R, psi) / (MU0 * R)


# ------------------------------------------------------------------
# Current Drive
# ------------------------------------------------------------------

def current_drive_efficiency(n_e: float,
                           T_e_keV: float,
                           Z_eff: float = 1.0) -> float:
    """
    Calculate current drive efficiency η_CD.

    Parameters
    ----------
    n_e : float
        Electron density [10¹⁹ m⁻³]
    T_e_keV : float
        Electron temperature [keV]
    Z_eff : float, optional
        Effective charge number, by default 1.0

    Returns
    -------
    float
        Current drive efficiency [A/W]

    Notes
    -----
    Uses ITER scaling for lower hybrid current drive
    """
    return 0.3 * (n_e * T_e_keV / Z_eff)**0.5


def bootstrap_current_fraction(n_e: float,
                             T_e_keV: float,
                             R0: float,
                             a: float,
                             q_95: float) -> float:
    """
    Calculate bootstrap current fraction f_BS.

    Parameters
    ----------
    n_e : float
        Electron density [10¹⁹ m⁻³]
    T_e_keV : float
        Electron temperature [keV]
    R0 : float
        Major radius [m]
    a : float
        Minor radius [m]
    q_95 : float
        Safety factor at 95% flux surface

    Returns
    -------
    float
        Bootstrap current fraction

    Notes
    -----
    Uses ITER scaling for bootstrap current
    """
    beta_p = 0.4 * n_e * T_e_keV * a / (R0 * q_95**2)
    return 0.3 * np.sqrt(beta_p)


# ------------------------------------------------------------------
# Current Limits
# ------------------------------------------------------------------

def current_limit_from_q(q_95: float,
                        a: float,
                        B0: float) -> float:
    """
    Calculate current limit from safety factor.

    Parameters
    ----------
    q_95 : float
        Safety factor at 95% flux surface
    a : float
        Minor radius [m]
    B0 : float
        Toroidal magnetic field [T]

    Returns
    -------
    float
        Current limit [A]

    Notes
    -----
    Uses simple relation I_p = 2πa²B0/(μ₀q_95)
    """
    return 2 * np.pi * a**2 * B0 / (MU0 * q_95)


def current_limit_from_beta(beta_N: float,
                          a: float,
                          B0: float) -> float:
    """
    Calculate current limit from beta.

    Parameters
    ----------
    beta_N : float
        Normalized beta
    a : float
        Minor radius [m]
    B0 : float
        Toroidal magnetic field [T]

    Returns
    -------
    float
        Current limit [A]

    Notes
    -----
    Uses relation I_p = 2πa²B0/(μ₀β_N)
    """
    return 2 * np.pi * a**2 * B0 / (MU0 * beta_N) 