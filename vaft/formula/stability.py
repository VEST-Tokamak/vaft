"""
Plasma stability, operational limits, and transport calculations.

This module provides functions for calculating various stability parameters
including beta limits, ballooning stability, MHD stability criteria, operational
limits, and transport parameters.

Notation
--------
β_N    : normalized beta                              [-]
β_p    : poloidal beta                                [-]
β_t    : toroidal beta                                [-]
q_95   : safety factor at 95% flux surface            [-]
α      : ballooning parameter                         [-]
s      : magnetic shear                               [-]
n_G    : Greenwald density limit                      [10¹⁹ m⁻³]
P_L    : power limit                                  [W]
ν*     : effective collisionality                     [-]
v_A    : Alfven speed                                [m/s]
c_s    : ion-sound speed                             [m/s]
τ_E    : energy confinement time                     [s]
"""

import numpy as np
from typing import Union, Tuple

from .constants import (
    MU0, QE, ME, MI_P,
    COLLISIONALITY_COEF,
    _SCALING_COEFS
)
from .utils import gradient

# ------------------------------------------------------------------
# Beta Calculations
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
# Empirical Data
# ------------------------------------------------------------------

def empirical_li_qa():
    """
    Returns empirical data for edge safety factor (qa) and internal inductance (li) based on JET experiments.

    This function provides arrays of qa and corresponding li values as observed in JET (Joint European Torus)
    disruption studies. These empirical values are used to analyze stability boundaries, particularly in the qa–li
    operational space.

    Observed trends:
    - Lower qa corresponds to higher li, indicating a more peaked central current profile.
    - Higher qa corresponds to lower or saturated li (~0.3), indicating a flatter profile.
    - These values are relevant for assessing operational limits and MHD stability (e.g., kink and tearing modes).

    Data sourced from:
    Wesson et al., "Disruptions in JET," Nuclear Fusion, vol. 29, no. 4, 1989.

    Returns
    -------
    qa : np.ndarray
        Edge safety factor values.
    li : np.ndarray
        Corresponding internal inductance values.
    """
    qa = np.array([2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
                   7, 7, 8, 8, 9, 9, 10, 10])
    li = np.array([0.95, 0.68, 0.93, 0.61, 0.86, 0.5, 0.71, 0.435, 0.7, 0.35,
                   0.67, 0.3, 0.67, 0.3, 0.67, 0.3, 0.67, 0.3])
    return qa, li


def li_from_qa_empirical(qa: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear fit through surveyed (q_a, l_i) points – Handy for
    quick sanity checks when only q_a is available.

    Parameters
    ----------
    qa : np.ndarray
        Edge safety factor values

    Returns
    -------
    np.ndarray
        Internal inductance values
    """
    qa_ref, li_ref = empirical_li_qa()
    return np.interp(qa, qa_ref, li_ref)


# ------------------------------------------------------------------
# Ballooning Stability
# ------------------------------------------------------------------

def ballooning_alpha_from_p_B_R(p: Union[float, np.ndarray],
                            B: Union[float, np.ndarray],
                            R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate ballooning parameter α = -2μ₀ R (dp/dpsi) / B² ≈ -2μ₀ R (dp/dR) / B².

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
    beta_N_crit = 2.8 * q_95
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


# ------------------------------------------------------------------
# Density Limits
# ------------------------------------------------------------------

def greenwald_density(I_p: float,
                      a: float) -> float:
    """
    Calculate Greenwald density limit n_G = I_p/(πa²).

    Parameters
    ----------
    I_p : float
        Plasma current [A]
    a : float
        Minor radius [m]

    Returns
    -------
    float
        Greenwald density limit [10¹⁹ m⁻³]
    """
    return I_p / (np.pi * a**2)


def density_limit_factor(n_e: float,
                        n_G: float) -> float:
    """
    Calculate density limit factor f_n = n_e/n_G.

    Parameters
    ----------
    n_e : float
        Electron density [10¹⁹ m⁻³]
    n_G : float
        Greenwald density limit [10¹⁹ m⁻³]

    Returns
    -------
    float
        Density limit factor
    """
    return n_e / n_G


# ------------------------------------------------------------------
# Power Limits
# ------------------------------------------------------------------

def power_limit_from_beta(beta_N: float,
                         B0: float,
                         V: float) -> float:
    """
    Calculate power limit from beta.

    Parameters
    ----------
    beta_N : float
        Normalized beta
    B0 : float
        Toroidal magnetic field [T]
    V : float
        Plasma volume [m³]

    Returns
    -------
    float
        Power limit [W]
    """
    return beta_N * B0**2 * V / (2 * MU0)


def power_limit_from_q(q_95: float,
                      I_p: float,
                      R0: float) -> float:
    """
    Calculate power limit from safety factor.

    Parameters
    ----------
    q_95 : float
        Safety factor at 95% flux surface
    I_p : float
        Plasma current [A]
    R0 : float
        Major radius [m]

    Returns
    -------
    float
        Power limit [W]
    """
    return 2 * np.pi * R0 * I_p / (MU0 * q_95)


# ------------------------------------------------------------------
# Stability Boundaries
# ------------------------------------------------------------------

def beta_stability_boundary(beta_N: float,
                            q_95: float) -> Tuple[float, float]:
    """
    Calculate beta stability boundary and margin.

    Parameters
    ----------
    beta_N : float
        Normalized beta
    q_95 : float
        Safety factor at 95% flux surface

    Returns
    -------
    Tuple[float, float]
        Tuple of (stability margin, critical beta)
    """
    beta_N_crit = 0.028 * q_95
    stab_margin = beta_N - beta_N_crit
    return stab_margin, beta_N_crit


def plasma_stability_margins(beta_N: float,
                             q_95: float,
                             n_e: float,
                             n_G: float) -> Tuple[float, float, float]:
    """
    Calculate plasma stability margins for beta, safety factor, and density.

    Parameters
    ----------
    beta_N : float
        Normalized beta
    q_95 : float
        Safety factor at 95% flux surface
    n_e : float
        Electron density [10¹⁹ m⁻³]
    n_G : float
        Greenwald density limit [10¹⁹ m⁻³]

    Returns
    -------
    Tuple[float, float, float]
        Tuple of (beta margin, q margin, density margin)
    """
    beta_margin, _ = beta_stability_boundary(beta_N, q_95)
    q_margin = q_95 - 2.0  # Minimum q_95 for stability
    density_margin = density_limit_factor(n_e, n_G)
    return beta_margin, q_margin, density_margin


# ------------------------------------------------------------------
# Transport
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
# Operational Parameters
# ------------------------------------------------------------------

def rhostar_from_Te_a_Bt(Te_eV: float,
                         a_minor: float,
                         B_t: float,
                         m_e: float = ME) -> float:
    """
    Calculate normalized gyro-radius ρ* (electron, minor-radius scaled).

    Parameters
    ----------
    Te_eV : float
        Electron temperature [eV]
    a_minor : float
        Minor radius [m]
    B_t : float
        Toroidal magnetic field [T]
    m_e : float, optional
        Electron mass [kg], by default ME

    Returns
    -------
    float
        Normalized gyro-radius ρ*
    """
    return np.sqrt(Te_eV) / B_t * a_minor
