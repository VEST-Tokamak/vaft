# formulas_operational.py
# ------------------------------------------------------------
#  Operational / discharge-level derived quantities
#  Te(Ohmic), gyro-ρ*, τ_E (scaling laws), kink limits, li–qa
# ------------------------------------------------------------

"""
Plasma operational parameter calculations.

This module provides functions for calculating various operational parameters
including density limits, power limits, and operational boundaries.

Notation
--------
n_G    : Greenwald density limit                      [10¹⁹ m⁻³]
P_L    : power limit                                  [W]
q_95   : safety factor at 95% flux surface            [-]
β_N    : normalized beta                              [-]
"""

import numpy as np
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt

from .constants import MU0
from .utils import gradient

MU0 = 4 * np.pi * 1e-7            # [H m⁻¹]
QE  = 1.602176634e-19             # [C]
ME  = 9.10938356e-31              # [kg]
LNΛ = 17.0                        # Coulomb logarithm  (typical)

# ------------------------------------------------------------------
# 1. Ohmic electron temperature   (iterative Spitzer-like formula)
# ------------------------------------------------------------------
def Te_from_Vloop_Ip_Fneo_Zeff_R_a_kappa(v_loop: float,
                                         I_p: float,
                                         F_neo: float,
                                         Z_eff: float,
                                         R_major: float,
                                         a_minor: float,
                                         kappa: float,
                                         ln_lambda: float = LNΛ) -> float:
    r"""
    Ohmic discharge temperature estimate
    ------------------------------------
    ρ_p  = V_loop / I_p                                  (Ω: plasma resistance)
    η_p  = ρ_p * (π a² κ) / (2π R_major) / F_neo         (Ω m: resistivity)
    T_e  = [ 5.265 × 10⁻⁵ ⋅ ln Λ ⋅ Z_eff / η_p ]^{2/3}    (eV)  – Spitzer-like
    """
    R_p  = v_loop / I_p
    eta  = R_p * (np.pi * a_minor**2 * kappa) / (2*np.pi*R_major) / F_neo
    Te_eV = (5.265e-5 * ln_lambda * Z_eff / eta)**(2/3)
    return Te_eV


# ------------------------------------------------------------------
# 2. Normalised gyro-radius ρ*  (electron, minor-radius scaled)
# ------------------------------------------------------------------
def rhostar_from_Te_a_Bt(Te_eV: float,
                         a_minor: float,
                         B_t: float,
                         m_e: float = ME) -> float:
    r"""
    ρ* = (√(2 m_e T_e) / e B_t) / a   ≈ √T_e / B_t ⋅ (ρ_L0 / a)
    여기서는 상수 ∝√m_e/e 를 a 에 흡수하여 간단화한 실용적 정의 사용.
    """
    return np.sqrt(Te_eV) / B_t * a_minor           # (adimensional)




# ------------------------------------------------------------------
# 4. Kink-limit and β limits (Friedberg 2008)
# ------------------------------------------------------------------
def kink_limits_from_R_a_kappa_Ip_Bt(R0: float,
                                     a: float,
                                     kappa: float,
                                     I_p: float,
                                     B_t: float,
                                     xsect: str = "conventional"):
    """
    q_kink, q_min, β_max, β_crit, I_p,max   for circular / conventional / ST.
    Returns tuple of floats.
    """
    μ0 = MU0
    ε  = a / R0

    if xsect == "circular":
        q_kink = 2*np.pi * a**2 * B_t / (μ0 * I_p * R0)
        beta_max = beta_crit = np.nan
    elif xsect == "conventional":
        q_kink = 2*np.pi * a**2 * kappa * B_t / (μ0 * I_p * R0)
        g = 1/kappa * (1 + 4/np.pi**2 * (kappa**2-1))
        q_kink *= g
        beta_max = np.pi**2/16 * kappa * ε / q_kink**2
        beta_crit = 0.14 * ε * kappa / q_kink
    elif xsect == "ST":
        q_kink = 2*np.pi * a**2 * B_t / (μ0 * I_p * R0) * (1 + kappa**2/2)
        beta_max = 0.072 * (1 + kappa**2)/2 * ε
        beta_crit = 5 * 0.03 * (q_kink-1) / ((0.75)**4 + (q_kink-1)**4)**0.25 \
                    * (1+kappa**2)/2 * ε / q_kink
    else:
        raise ValueError("xsect must be 'circular', 'conventional', or 'ST'")

    q_min  = 1 + kappa/2
    Imax   = q_kink * I_p * 2/(1+kappa)        # A
    return q_kink, q_min, beta_max, beta_crit, Imax


# ------------------------------------------------------------------
# 5. Empirical li–qa data helper
# ------------------------------------------------------------------
def li_from_qa_empirical(qa: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear fit through surveyed (q_a, l_i) points – Handy for
    quick sanity checks when only q_a is available.
    """
    qa_ref = np.array([2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10])
    li_ref = np.array([0.95,0.68,0.93,0.61,0.86,0.50,0.71,0.435,
                       0.70,0.35,0.67,0.30,0.67,0.30,0.67,0.30,0.67,0.30])
    return np.interp(qa, qa_ref, li_ref)


# ------------------------------------------------------------------
# Density Limits
# ------------------------------------------------------------------

def greenwald_density(I_p: float,
                              a: float,
                              plot_opt: bool = False) -> float:
    """
    Calculate Greenwald density limit n_G = I_p/(πa²).

    Parameters
    ----------
    I_p : float
        Plasma current [A]
    a : float
        Minor radius [m]
    plot_opt : bool, optional
        If True, plot density limit vs current, by default False

    Returns
    -------
    float
        Greenwald density limit [10¹⁹ m⁻³]
    """
    n_G = I_p / (np.pi * a**2)
    
    if plot_opt:
        I_p_range = np.linspace(0, I_p * 1.2, 100)
        n_G_range = I_p_range / (np.pi * a**2)
        
        plt.figure(figsize=(8, 6))
        plt.plot(I_p_range * 1e-6, n_G_range, 'b-', label='Greenwald limit')
        plt.plot(I_p * 1e-6, n_G, 'ro', label='Current point')
        plt.xlabel('Plasma Current [MA]')
        plt.ylabel('Density Limit [10¹⁹ m⁻³]')
        plt.title('Greenwald Density Limit')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return n_G


def density_limit_factor(n_e: float,
                        n_G: float,
                        plot_opt: bool = False) -> float:
    """
    Calculate density limit factor f_n = n_e/n_G.

    Parameters
    ----------
    n_e : float
        Electron density [10¹⁹ m⁻³]
    n_G : float
        Greenwald density limit [10¹⁹ m⁻³]
    plot_opt : bool, optional
        If True, plot density limit ratio vs density, by default False

    Returns
    -------
    float
        Density limit factor
    """
    f_n = n_e / n_G
    
    if plot_opt:
        n_e_range = np.linspace(0, n_G * 1.2, 100)
        f_n_range = n_e_range / n_G
        
        plt.figure(figsize=(8, 6))
        plt.plot(n_e_range, f_n_range, 'b-', label='Limit factor')
        plt.plot(n_e, f_n, 'ro', label='Current point')
        plt.axvline(x=n_G, color='r', linestyle='--', label='Greenwald limit')
        plt.xlabel('Electron Density [10¹⁹ m⁻³]')
        plt.ylabel('Density Limit Factor')
        plt.title('Density Limit Factor')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return f_n


# ------------------------------------------------------------------
# Power Limits
# ------------------------------------------------------------------

def power_limit_from_beta(beta_N: float,
                         B0: float,
                         V: float,
                         plot_opt: bool = False) -> float:
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
    plot_opt : bool, optional
        If True, plot power limit vs beta, by default False

    Returns
    -------
    float
        Power limit [W]
    """
    P_L = beta_N * B0**2 * V / (2 * MU0)
    
    if plot_opt:
        beta_range = np.linspace(0, beta_N * 1.2, 100)
        P_L_range = beta_range * B0**2 * V / (2 * MU0)
        
        plt.figure(figsize=(8, 6))
        plt.plot(beta_range, P_L_range * 1e-6, 'b-', label='Power limit')
        plt.plot(beta_N, P_L * 1e-6, 'ro', label='Current point')
        plt.xlabel('Normalized Beta')
        plt.ylabel('Power Limit [MW]')
        plt.title('Beta Power Limit')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return P_L


def power_limit_from_q(q_95: float,
                      I_p: float,
                      R0: float,
                      plot_opt: bool = False) -> float:
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
    plot_opt : bool, optional
        If True, plot power limit vs safety factor, by default False

    Returns
    -------
    float
        Power limit [W]
    """
    P_L = 2 * np.pi * R0 * I_p / (MU0 * q_95)
    
    if plot_opt:
        q_range = np.linspace(1, q_95 * 1.2, 100)
        P_L_range = 2 * np.pi * R0 * I_p / (MU0 * q_range)
        
        plt.figure(figsize=(8, 6))
        plt.plot(q_range, P_L_range * 1e-6, 'b-', label='Power limit')
        plt.plot(q_95, P_L * 1e-6, 'ro', label='Current point')
        plt.xlabel('Safety Factor q_95')
        plt.ylabel('Power Limit [MW]')
        plt.title('Power Limit vs Safety Factor')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return P_L


# ------------------------------------------------------------------
# Stability Boundaries
# ------------------------------------------------------------------

def beta_stability_boundary(beta_N: float,
                                    q_95: float,
                                    plot_opt: bool = False) -> Tuple[float, float]:
    """
    Calculate beta stability boundary and margin.

    Parameters
    ----------
    beta_N : float
        Normalized beta
    q_95 : float
        Safety factor at 95% flux surface
    plot_opt : bool, optional
        If True, plot beta stability boundary, by default False

    Returns
    -------
    Tuple[float, float]
        Tuple of (stability margin, critical beta)
    """
    beta_N_crit = 0.028 * q_95
    stab_margin = beta_N - beta_N_crit
    
    if plot_opt:
        q_range = np.linspace(1, q_95 * 1.2, 100)
        beta_crit_range = 0.028 * q_range
        
        plt.figure(figsize=(8, 6))
        plt.plot(q_range, beta_crit_range, 'b-', label='Stability boundary')
        plt.plot(q_95, beta_N, 'ro', label='Current point')
        plt.fill_between(q_range, beta_crit_range, alpha=0.2, label='Stable region')
        plt.xlabel('Safety Factor q_95')
        plt.ylabel('Normalized Beta')
        plt.title('Beta Stability Boundary')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return stab_margin, beta_N_crit


def plasma_stability_margins(beta_N: float,
                                     q_95: float,
                                     n_e: float,
                                     n_G: float,
                                     plot_opt: bool = False) -> Tuple[float, float, float]:
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
    plot_opt : bool, optional
        If True, plot stability margins, by default False

    Returns
    -------
    Tuple[float, float, float]
        Tuple of (beta margin, q margin, density margin)
    """
    beta_margin, _ = beta_stability_boundary(beta_N, q_95)
    q_margin = q_95 - 2.0  # Minimum q_95 for stability
    density_margin = density_limit_ratio(n_e, n_G)
    
    if plot_opt:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Beta stability plot
        q_range = np.linspace(1, q_95 * 1.2, 100)
        beta_crit_range = 0.028 * q_range
        ax1.plot(q_range, beta_crit_range, 'b-', label='Stability boundary')
        ax1.plot(q_95, beta_N, 'ro', label='Current point')
        ax1.fill_between(q_range, beta_crit_range, alpha=0.2, label='Stable region')
        ax1.set_xlabel('Safety Factor q_95')
        ax1.set_ylabel('Normalized Beta')
        ax1.set_title('Beta Stability Margin')
        ax1.grid(True)
        ax1.legend()
        
        # Q stability plot
        ax2.axvline(x=q_margin, color='r', linestyle='--', label='Minimum q_95')
        ax2.plot(q_95, 0, 'ro', label='Current point')
        ax2.set_xlabel('Safety Factor q_95')
        ax2.set_title('Q Stability Margin')
        ax2.grid(True)
        ax2.legend()
        
        # Density stability plot
        n_e_range = np.linspace(0, n_G * 1.2, 100)
        f_n_range = n_e_range / n_G
        ax3.plot(n_e_range, f_n_range, 'b-', label='Limit ratio')
        ax3.plot(n_e, density_margin, 'ro', label='Current point')
        ax3.axvline(x=n_G, color='r', linestyle='--', label='Greenwald limit')
        ax3.set_xlabel('Electron Density [10¹⁹ m⁻³]')
        ax3.set_title('Density Stability Margin')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
    
    return beta_margin, q_margin, density_margin