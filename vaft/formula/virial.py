"""
Virial theorem calculations for plasma physics.

This module provides functions for calculating various virial theorem integrals
used in plasma physics calculations.

Notation
--------
W      : total energy                                 [J]
W_mag  : magnetic energy                              [J]
W_kin  : kinetic energy                               [J]
W_th   : thermal energy                               [J]
"""

import numpy as np
from typing import Union, Tuple

from .constants import MU0
from .utils import trapz_integral

# ------------------------------------------------------------------
# Energy Calculations
# ------------------------------------------------------------------

def magnetic_energy(B: np.ndarray,
                   V: float) -> float:
    """
    Calculate magnetic energy W_mag = ∫ B²/(2μ₀) dV.

    Parameters
    ----------
    B : np.ndarray
        Magnetic field values
    V : float
        Volume element

    Returns
    -------
    float
        Magnetic energy [J]
    """
    return np.sum(B**2) * V / (2 * MU0)


def kinetic_energy(n: np.ndarray,
                  v: np.ndarray,
                  m: float,
                  V: float) -> float:
    """
    Calculate kinetic energy W_kin = ∫ (1/2)nmv² dV.

    Parameters
    ----------
    n : np.ndarray
        Density values
    v : np.ndarray
        Velocity values
    m : float
        Particle mass
    V : float
        Volume element

    Returns
    -------
    float
        Kinetic energy [J]
    """
    return 0.5 * np.sum(n * m * v**2) * V


def thermal_energy(n: np.ndarray,
                  T: np.ndarray,
                  V: float) -> float:
    """
    Calculate thermal energy W_th = ∫ (3/2)nT dV.

    Parameters
    ----------
    n : np.ndarray
        Density values
    T : np.ndarray
        Temperature values
    V : float
        Volume element

    Returns
    -------
    float
        Thermal energy [J]
    """
    return 1.5 * np.sum(n * T) * V


# ------------------------------------------------------------------
# Virial Theorem
# ------------------------------------------------------------------

def virial_theorem(W_mag: float,
                  W_kin: float,
                  W_th: float) -> Tuple[float, float]:
    """
    Calculate virial theorem balance.

    Parameters
    ----------
    W_mag : float
        Magnetic energy
    W_kin : float
        Kinetic energy
    W_th : float
        Thermal energy

    Returns
    -------
    Tuple[float, float]
        Tuple of (total energy, virial ratio)
    """
    W_total = W_mag + W_kin + W_th
    virial_ratio = (W_kin + W_th) / W_mag
    return W_total, virial_ratio


def virial_stability_criterion(W_mag: float,
                             W_kin: float,
                             W_th: float) -> Tuple[float, float]:
    """
    Calculate virial stability criterion.

    Parameters
    ----------
    W_mag : float
        Magnetic energy
    W_kin : float
        Kinetic energy
    W_th : float
        Thermal energy

    Returns
    -------
    Tuple[float, float]
        Tuple of (stability parameter, critical ratio)
    """
    W_total, virial_ratio = virial_theorem(W_mag, W_kin, W_th)
    critical_ratio = 0.5  # Theoretical value for stability
    return virial_ratio - critical_ratio, critical_ratio


"""
formulas_mhd.py
β_p, l_i, μ̂_i  (volume‐integral & virial-analysis forms)
"""

import numpy as np

# ------------------------------------------------------------------
# 1. Volume-integral definitions
# ------------------------------------------------------------------
def beta_p_from_volume(p: np.ndarray,
                       dV: np.ndarray,
                       B_pa: float,
                       Omega: float,
                       mu0: float = 4*np.pi*1e-7) -> float:
    r"""
    β_p = (2 μ₀) / (B_{pa}² Ω) ∫_Ω p · dV
    """
    return (2 * mu0 / (B_pa**2 * Omega)) * np.sum(p * dV)


def li_from_volume(B_p: np.ndarray,
                   dV: np.ndarray,
                   B_pa: float,
                   Omega: float) -> float:
    r"""
    l_i = 1 / (B_{pa}² Ω) ∫_Ω B_p² · dV
    """
    return (1.0 / (B_pa**2 * Omega)) * np.sum(B_p**2 * dV)


# ------------------------------------------------------------------
# 2. Diamagnetic energy term  μ̂_i  (Bongard 2016)
# ------------------------------------------------------------------
def muihat_from_Bt_R0_dphi(B_t: float,
                           R0: float,
                           dphi: float,
                           B_pa: float,
                           Omega: float) -> float:
    r"""
    μ̂_i ≈ (4 π B_t R₀ Δφ) / (B_{pa}² Ω)
    """
    return (4 * np.pi * B_t * R0 * dphi) / (B_pa**2 * Omega)


# ------------------------------------------------------------------
# 3. Virial-analysis approximations (Bongard 2016)
# ------------------------------------------------------------------
def beta_p_from_S_alpha_mu(S1: float,
                           S2: float,
                           S3: float,
                           alpha: float,
                           mui_hat: float) -> float:
    r"""
    β_p = [ (S₁ + S₂)(α − 1) + α μ̂_i + S₃ ] / [ 3(α − 1) + 1 ]
    """
    num = (S1 + S2) * (alpha - 1) + alpha * mui_hat + S3
    den = 3 * (alpha - 1) + 1
    return num / den


def li_from_S_alpha_mu(S1: float,
                       S2: float,
                       S3: float,
                       alpha: float,
                       mui_hat: float) -> float:
    r"""
    l_i = [ S₁ + S₂ − 2 μ̂_i − 3 S₃ ] / [ 3 α − 2 ]
    """
    num = S1 + S2 - 2 * mui_hat - 3 * S3
    den = 3 * alpha - 2
    return num / den


"""
formulas_virial.py
------------------------------------------------------------
Shafranov virial relations and auxiliary analytic estimates
Ref: A.A. Martynov & V.D. Pustovitov,
     Phys. Plasmas 31 (2024) 082501
"""

import numpy as np
from typing import Tuple


# ------------------------------------------------------------------
# 0. Helpers --------------------------------------------------------
def eK_from_K(K: float) -> float:
    """eK ≡ (K²−1)/(K²+1)  — elongation shape factor."""
    return (K**2 - 1) / (K**2 + 1)


# ------------------------------------------------------------------
# 1. Analytic Pustovitov approximations  SP₁ … SP₃  (Eq 20–22)
# ------------------------------------------------------------------
def S1_approx() -> float:
    """SP₁ = 2.0  — valid to O(ε, D₀, δ)."""
    return 2.0                                 #  [oai_citation:0‡A. A. Martynov, & V. D. Pustovitov. 2024. Virial relations for elongated plasmas in tokamaks- Analytical approximations and numerical calculations. Physics of Plasmas.pdf](file-service://file-QzNCjc6MvYbNMqcytkN8UZ)


def S2_approx_from_D0_a_R0(eK: float,
                           D0: float,
                           a_minor: float,
                           R0: float) -> float:
    r"""SP₂(a) = −(2 a / R0)·(D0+1)·(1+eK/2)  (Eq 21)."""
    return -(2 * a_minor / R0) * (D0 + 1) * (1 + eK / 2)  #  [oai_citation:1‡A. A. Martynov, & V. D. Pustovitov. 2024. Virial relations for elongated plasmas in tokamaks- Analytical approximations and numerical calculations. Physics of Plasmas.pdf](file-service://file-QzNCjc6MvYbNMqcytkN8UZ)


def S3_approx_from_eK_d(eK: float,
                        d_param: float) -> float:
    r"""SP₃ = 1 − eK/2 − δ·(1 − eK²/2)  (Eq 22)."""
    return 1 - 0.5 * eK - d_param * (1 - 0.5 * eK**2)     #  [oai_citation:2‡A. A. Martynov, & V. D. Pustovitov. 2024. Virial relations for elongated plasmas in tokamaks- Analytical approximations and numerical calculations. Physics of Plasmas.pdf](file-service://file-QzNCjc6MvYbNMqcytkN8UZ)


# ------------------------------------------------------------------
# 2. Solve full virial system  (Eq 1–3)  for βₚ, ℓᵢ, ℓ̂ᵢ
# ------------------------------------------------------------------
def bp_li_lihat_from_S123(S1: float,
                          S2: float,
                          S3: float,
                          a_param: float,
                          RT_over_R0: float
                          ) -> Tuple[float, float, float]:
    """
    Invert the linear virial system:

        3βₚ + ℓᵢ − ℓ̂ᵢ         = S₁ + S₂
        βₚ  + ℓᵢ + ℓ̂ᵢ         = (RT/R₀)·S₂
        βₚ  − (a−1)ℓᵢ − ℓ̂ᵢ    = S₃                                    [oai_citation:3‡A. A. Martynov, & V. D. Pustovitov. 2024. Virial relations for elongated plasmas in tokamaks- Analytical approximations and numerical calculations. Physics of Plasmas.pdf](file-service://file-QzNCjc6MvYbNMqcytkN8UZ)

    Returns (βₚ, ℓᵢ, ℓ̂ᵢ).
    """
    # Linear system  A·x = b
    A = np.array([[3,  1, -1],
                  [1,  1,  1],
                  [1, -(a_param - 1), -1]], dtype=float)
    b = np.array([S1 + S2,
                  RT_over_R0 * S2,
                  S3], dtype=float)
    βp, li_int, li_hat = np.linalg.solve(A, b)
    return βp, li_int, li_hat


# ------------------------------------------------------------------
# 3. Shafranov-shift gradient  D₀(b)  (Eq 37) -----------------------
# ------------------------------------------------------------------
def D0_boundary_from_bp_li_eK(beta_p: float,
                              li_int: float,
                              eK: float,
                              b_minor: float,
                              R_plasma: float) -> float:
    r"""
    D₀(b) = − (b / 2Rₚₗ) · [ 2βₚ + ℓᵢ + 0.5 eK ] / [ 1 + 0.5 eK ]       [oai_citation:4‡A. A. Martynov, & V. D. Pustovitov. 2024. Virial relations for elongated plasmas in tokamaks- Analytical approximations and numerical calculations. Physics of Plasmas.pdf](file-service://file-QzNCjc6MvYbNMqcytkN8UZ)
    """
    numerator = 2 * beta_p + li_int + 0.5 * eK
    denominator = 1 + 0.5 * eK
    return - (b_minor / (2 * R_plasma)) * numerator / denominator