"""
Plasma equilibrium, current, energy, and geometry calculations.

This module provides functions for calculating various plasma equilibrium parameters
including poloidal flux, toroidal flux, safety factor, current, energy, and geometry.

Notation
--------
ψ      : poloidal magnetic flux                     [Wb]
ψ_a    : ψ at magnetic axis                         [Wb]
ψ_b    : ψ at plasma boundary                       [Wb]
Φ(ψ)   : toroidal flux through surface C(ψ)         [Wb]
Φ_b    : Φ(ψ_b)                                     [Wb]
ρ_N    : normalised minor-radius (0 at axis, 1 at edge)
q      : safety factor                              [-]
j      : current density                            [A/m²]
I_p    : plasma current                             [A]
W      : stored energy                              [J]
V      : plasma volume                              [m³]
κ      : elongation                                 [-]
δ      : triangularity                              [-]
"""

import warnings
from typing import Union, Tuple
import numpy as np

from .constants import (
    MU0, QE, ME, MI_P,
    E_ALPHA, SIGMA_V_COEF,
    SPITZER_RESISTIVITY_COEF,
    _SCALING_COEFS
)
from .utils import (
    gradient,
    trapz_integral,
    normalize_profile,
    calculate_poloidal_flux,
    calculate_toroidal_flux,
    calculate_volume_weighted_average
)

# ------------------------------------------------------------------
# Poloidal Flux Calculations
# ------------------------------------------------------------------

def psi_from_RBtheta(R: np.ndarray,
                     B_theta: np.ndarray,
                     l: np.ndarray,
                     psi_axis: float = 0.0) -> float:
    """
    # $\psi = \int R B_\theta \, dl + \psi_a$
    # ψ = ∫ R B_θ dl + ψ_a
    """
    return calculate_poloidal_flux(R, B_theta, l, psi_axis)


def psi_normalised(psi: Union[np.ndarray, float],
                   psi_axis: float,
                  psi_boundary: float) -> Union[np.ndarray, float]:
    """
    # $\psi_N = \frac{\psi - \psi_a}{\psi_b - \psi_a}$
    # ψ_N = (ψ − ψ_a) / (ψ_b − ψ_a)
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
    # $\Phi = \int B_\phi \, dA$
    # Φ = ∫ B_φ dA
    """
    return calculate_toroidal_flux(B_phi, dA)


def rhoN_from_phi(phi: Union[np.ndarray, float],
                  phi_boundary: float) -> Union[np.ndarray, float]:
    """
    # $\rho_N = \sqrt{\frac{\Phi}{\Phi_b}}$
    # ρ_N = √(Φ / Φ_b)
    """
    return np.sqrt(phi / phi_boundary)


# ------------------------------------------------------------------
# Safety Factor Calculations
# ------------------------------------------------------------------

def q_from_phi(psi: np.ndarray,
               phi: np.ndarray) -> np.ndarray:
    """
    # $q = \frac{d\Phi}{d\psi}$
    # q = dΦ/dψ
    """
    return gradient(psi, phi)


def q_from_rhoN(psiN: np.ndarray,
                rhoN: np.ndarray,
                C: float = 1.0) -> np.ndarray:
    """
    # $q = C \rho_N \frac{d\rho_N}{d\psi_N}$
    # q = C · ρ_N · dρ_N/dψ_N
    """
    drhoN_dpsiN = gradient(psiN, rhoN)
    return C * rhoN * drhoN_dpsiN


def rhoN_from_qpsiN(psiN: np.ndarray,
                    qpsiN: np.ndarray) -> np.ndarray:
    """
    # $\rho_N = \sqrt{\frac{\int_0^{\psi_N} q(\psi'_N) d\psi'_N}{\int_0^1 q(\psi'_N) d\psi'_N}}$
    # ρ_N = √(∫₀^{ψ_N} q(ψ′_N) dψ′_N / ∫₀¹ q(ψ′_N) dψ′_N)
    """
    # Cumulative integral using trapezoidal rule to preserve quartiles
    num = np.array([trapz_integral(psiN[:i+1], qpsiN[:i+1]) for i in range(len(psiN))])
    den = trapz_integral(psiN, qpsiN)
    return np.sqrt(num / den)


# ------------------------------------------------------------------
# Magnetic Shear
# ------------------------------------------------------------------

def shear_from_r_q(r: np.ndarray,
                   q: np.ndarray) -> np.ndarray:
    """
    # $s = \frac{r}{q} \frac{dq}{dr}$
    # s = (r/q) · dq/dr
    """
    dqdr = gradient(r, q)
    return (r / q) * dqdr


# Alias for backwards compatibility
magnetic_shear = shear_from_r_q  # noqa: E305


# ------------------------------------------------------------------
# Current Density
# ------------------------------------------------------------------

def current_density_from_B(B: Union[float, np.ndarray],
                          R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    # $j = \frac{\nabla \times B}{\mu_0}$
    # j = (∇ × B) / μ₀
    """
    return gradient(R, B) / MU0


def current_density_from_psi(psi: Union[float, np.ndarray],
                           R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    # $j = -\frac{1}{\mu_0 R} \frac{d\psi}{dR}$
    # j = -dψ/(μ₀ R dR)
    """
    return -gradient(R, psi) / (MU0 * R)


# ------------------------------------------------------------------
# Current Drive
# ------------------------------------------------------------------

def current_drive_efficiency(n_e: float,
                           T_e_keV: float,
                           Z_eff: float = 1.0) -> float:
    """
    # $\eta_{CD} = 0.3 \sqrt{\frac{n_e T_e}{Z_{eff}}}$
    # η_CD = 0.3 √(n_e T_e / Z_eff)
    # ITER scaling for lower hybrid current drive
    """
    return 0.3 * (n_e * T_e_keV / Z_eff)**0.5


def bootstrap_current_fraction(n_e: float,
                             T_e_keV: float,
                             R0: float,
                             a: float,
                             q_95: float) -> float:
    """
    # $f_{BS} = 0.3 \sqrt{\beta_p}$, where $\beta_p = \frac{0.4 n_e T_e a}{R_0 q_{95}^2}$
    # f_BS = 0.3 √(β_p), where β_p = 0.4 n_e T_e a / (R₀ q_95²)
    # ITER scaling for bootstrap current
    """
    beta_p = 0.4 * n_e * T_e_keV * a / (R0 * q_95**2)
    return 0.3 * np.sqrt(beta_p)

"""
Magnetic Field $B$
"""

def radial_magnetic_field_from_psi(psi: np.ndarray,
                                   R: np.ndarray,
                                   Z: np.ndarray) -> np.ndarray:
    """
    # $B_r = -1/R \frac{\partial \psi}{\partial Z}$
    # B_r = -1/R dψ/dZ
    """

    return -1/R * gradient(Z, psi)

def vertical_magnetic_field_from_psi(psi: np.ndarray,
                                   R: np.ndarray,
                                   Z: np.ndarray) -> np.ndarray:
    """
    # $B_z = 1/R \frac{\partial \psi}{\partial R}$
    # B_z = 1/R dψ/dR
    """
    return 1/R * gradient(R, psi)





# ------------------------------------------------------------------
# Current Limits
# ------------------------------------------------------------------

def current_limit_from_q(q_95: float,
                        a: float,
                        B0: float) -> float:
    """
    # $I_p = \frac{2\pi a^2 B_0}{\mu_0 q_{95}}$
    # I_p = 2πa²B₀/(μ₀q_95)
    """
    return 2 * np.pi * a**2 * B0 / (MU0 * q_95)


def current_limit_from_beta(beta_N: float,
                          a: float,
                          B0: float) -> float:
    """
    # $I_p = \frac{2\pi a^2 B_0}{\mu_0 \beta_N}$
    # I_p = 2πa²B₀/(μ₀β_N)
    """
    return 2 * np.pi * a**2 * B0 / (MU0 * beta_N)


# ------------------------------------------------------------------
# Stored Energy
# ------------------------------------------------------------------

def stored_energy_from_p_V(p: Union[float, np.ndarray],
                          V: float) -> Union[float, np.ndarray]:
    """
    # $W = \int p \, dV$
    # W = ∫ p dV
    """
    return p * V


def stored_energy_from_beta_V(beta: float,
                            B0: float,
                            V: float) -> float:
    """
    # $W = \frac{\beta B_0^2 V}{2\mu_0}$
    # W = β B₀² V / (2μ₀)
    """
    return beta * B0**2 * V / (2 * MU0)

# ------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------

def volume_from_RZ_boundary(R: np.ndarray,
                           Z: np.ndarray) -> float:
    """
    # $V = 2\pi \oint R Z_n \, dl \approx 2\pi A_{poly} \bar{R}$
    # V = 2π ∮ (R Z_n) dl ≈ 2π A_poly R̄
    # Green's theorem for axisymmetric geometry
    """
    # Calculate polygon area
    area = 0.5 * np.abs(np.dot(R, np.roll(Z, 1)) - np.dot(Z, np.roll(R, 1)))
    # R̄: area-weighted mean radius (approximation)
    R_bar = np.mean(R)
    return 2 * np.pi * area * R_bar


def elongation_from_RZ_boundary(R: np.ndarray,
                               Z: np.ndarray) -> float:
    """
    # $\kappa = \frac{Z_{max} - Z_{min}}{2a}$, where $a = \frac{R_{max} - R_{min}}{2}$
    # κ = (Z_max - Z_min) / (2a), where a = (R_max - R_min) / 2
    """
    a = (R.max() - R.min()) / 2
    return (Z.max() - Z.min()) / (2 * a)


def triangularity_from_RZ_boundary(R: np.ndarray,
                                  Z: np.ndarray,
                                  R0: float) -> float:
    """
    # $\delta = \frac{R_0 - R_{sep}|_{Z=0}}{a}$
    # δ = (R₀ - R_sep|Z=0) / a
    """
    R_mid = R[np.argmin(np.abs(Z))]  # Boundary intersection at mid-plane
    a = (R.max() - R.min()) / 2
    return (R0 - R_mid) / a


def eK_from_K(K: float) -> float:
    """
    # $eK = \frac{K^2 - 1}{K^2 + 1}$
    # eK = (K² - 1) / (K² + 1)
    """
    return (K**2 - 1) / (K**2 + 1)


def peaking_factor(central: float,
                   volume_avg: float) -> float:
    """
    # $PF = \frac{X(0)}{\langle X \rangle}$
    # PF = X(0) / ⟨X⟩
    """
    return central / volume_avg

# ------------------------------------------------------------------
# Plasma Resistance
# ------------------------------------------------------------------

def spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(T_e: float,
                                                 Z_eff: float = 2.0,
                                                 ln_Lambda: float = 17.0) -> float:
    """
    # $\eta(R,Z) = 5.2\times 10^{-5}\; \frac{Z_{\text{eff}}\,\ln\Lambda}{\bigl(T_e(R,Z)\,[\mathrm{eV}]\bigr)^{3/2}} \quad [\Omega\cdot \mathrm{m}]$
    # η = 5.2×10⁻⁵ Z_eff ln(Λ) / (T_e^(3/2))
    """
    return SPITZER_RESISTIVITY_COEF * Z_eff * ln_Lambda / T_e**1.5


# ------------------------------------------------------------------
# Normalized Plasma Current
# ------------------------------------------------------------------

def normalized_plasma_current(Ip: Union[float, np.ndarray],
                            R: Union[float, np.ndarray],
                            a: Union[float, np.ndarray],
                            Bt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    # $I_p^* = \frac{I_p}{a B_t}$
    # I_p^* = I_p / (a B_t) [MA/(m·T)]
    # Phys. Plasmas 23, 072508 (2016); https://doi.org/10.1063/1.4959808
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
    # $q_{kink} = \frac{2\pi a^2 B_t}{\mu_0 I_p R}$ (circular), with modifications for conventional and ST
    # q_kink = 2πa²B_t/(μ₀I_pR) (circular), with modifications for conventional and ST
    # Friedberg 2008 Plasma Physics and Fusion Energy (eq 13.158, p 405)
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


# ------------------------------------------------------------------
# Virial Theorem
# ------------------------------------------------------------------

def virial_magnetic_energy(B: np.ndarray,
                          V: float) -> float:
    """
    # $W_{mag} = \int \frac{B^2}{2\mu_0} \, dV$
    # W_mag = ∫ B²/(2μ₀) dV
    """
    return np.sum(B**2) * V / (2 * MU0)


def virial_kinetic_energy(n: np.ndarray,
                         v: np.ndarray,
                         m: float,
                         V: float) -> float:
    """
    # $W_{kin} = \int \frac{1}{2} n m v^2 \, dV$
    # W_kin = ∫ (1/2)nmv² dV
    """
    return 0.5 * np.sum(n * m * v**2) * V


def virial_thermal_energy(n: np.ndarray,
                         T: np.ndarray,
                         V: float) -> float:
    """
    # $W_{th} = \int \frac{3}{2} n T \, dV$
    # W_th = ∫ (3/2)nT dV
    """
    return 1.5 * np.sum(n * T) * V


def virial_theorem(W_mag: float,
                  W_kin: float,
                  W_th: float) -> Tuple[float, float]:
    """
    # $W_{total} = W_{mag} + W_{kin} + W_{th}$, $\text{virial ratio} = \frac{W_{kin} + W_{th}}{W_{mag}}$
    # W_total = W_mag + W_kin + W_th, virial ratio = (W_kin + W_th) / W_mag
    """
    W_total = W_mag + W_kin + W_th
    virial_ratio = (W_kin + W_th) / W_mag
    return W_total, virial_ratio


def virial_stability_criterion(W_mag: float,
                             W_kin: float,
                             W_th: float) -> Tuple[float, float]:
    """
    # $\text{stability parameter} = \text{virial ratio} - 0.5$, where critical ratio = 0.5
    # stability parameter = virial ratio - 0.5, where critical ratio = 0.5
    """
    W_total, virial_ratio = virial_theorem(W_mag, W_kin, W_th)
    critical_ratio = 0.5  # Theoretical value for stability
    return virial_ratio - critical_ratio, critical_ratio


def virial_beta_p_from_volume(p: np.ndarray,
                              dV: np.ndarray,
                              B_pa: float,
                              Omega: float,
                              mu0: float = None) -> float:
    """
    # $\beta_p = \frac{2\mu_0}{B_{pa}^2 \Omega} \int_\Omega p \, dV$
    # β_p = (2μ₀) / (B_{pa}² Ω) ∫_Ω p dV
    """
    if mu0 is None:
        mu0 = MU0
    return (2 * mu0 / (B_pa**2 * Omega)) * np.sum(p * dV)


def virial_li_from_volume(B_p: np.ndarray,
                         dV: np.ndarray,
                         B_pa: float,
                         Omega: float) -> float:
    """
    # $l_i = \frac{1}{B_{pa}^2 \Omega} \int_\Omega B_p^2 \, dV$
    # l_i = 1 / (B_{pa}² Ω) ∫_Ω B_p² dV
    """
    return (1.0 / (B_pa**2 * Omega)) * np.sum(B_p**2 * dV)


def virial_muihat_from_Bt_R0_dphi(B_t: float,
                                 R0: float,
                                 dphi: float,
                                 B_pa: float,
                                 Omega: float) -> float:
    """
    # $\hat{\mu}_i \approx \frac{4\pi B_t R_0 \Delta\phi}{B_{pa}^2 \Omega}$
    # μ̂_i ≈ (4π B_t R₀ Δφ) / (B_{pa}² Ω)
    """
    return (4 * np.pi * B_t * R0 * dphi) / (B_pa**2 * Omega)


def virial_beta_p_from_S_alpha_mu(S1: float,
                                  S2: float,
                                  S3: float,
                                  alpha: float,
                                  mui_hat: float) -> float:
    """
    # $\beta_p = \frac{(S_1 + S_2)(\alpha - 1) + \alpha \hat{\mu}_i + S_3}{3(\alpha - 1) + 1}$
    # β_p = [(S₁ + S₂)(α − 1) + α μ̂_i + S₃] / [3(α − 1) + 1]
    """
    num = (S1 + S2) * (alpha - 1) + alpha * mui_hat + S3
    den = 3 * (alpha - 1) + 1
    return num / den


def virial_li_from_S_alpha_mu(S1: float,
                             S2: float,
                             S3: float,
                             alpha: float,
                             mui_hat: float) -> float:
    """
    # $l_i = \frac{S_1 + S_2 - 2\hat{\mu}_i - 3S_3}{3\alpha - 2}$
    # l_i = [S₁ + S₂ − 2 μ̂_i − 3 S₃] / [3 α − 2]
    """
    num = S1 + S2 - 2 * mui_hat - 3 * S3
    den = 3 * alpha - 2
    return num / den


def virial_S1_approx() -> float:
    """
    # $S_1 = 2.0$
    # S₁ = 2.0
    # Valid to O(ε, D₀, δ)
    """
    return 2.0


def virial_S2_approx_from_D0_a_R0(eK: float,
                                  D0: float,
                                  a_minor: float,
                                  R0: float) -> float:
    """
    # $S_2 = -\frac{2a}{R_0}(D_0 + 1)\left(1 + \frac{eK}{2}\right)$
    # S₂ = −(2a / R₀)(D₀ + 1)(1 + eK/2)
    # A. A. Martynov, & V. D. Pustovitov. 2024. Virial relations for elongated plasmas in tokamaks- Analytical approximations and numerical calculations. Physics of Plasmas (Eq 21)
    """
    return -(2 * a_minor / R0) * (D0 + 1) * (1 + eK / 2)


def virial_S3_approx_from_eK_d(eK: float,
                              d_param: float) -> float:
    """
    # $S_3 = 1 - \frac{eK}{2} - \delta\left(1 - \frac{eK^2}{2}\right)$
    # S₃ = 1 − eK/2 − δ(1 − eK²/2)
    # A. A. Martynov, & V. D. Pustovitov. 2024. Virial relations for elongated plasmas in tokamaks- Analytical approximations and numerical calculations. Physics of Plasmas (Eq 22)
    """
    return 1 - 0.5 * eK - d_param * (1 - 0.5 * eK**2)


def virial_bp_li_lihat_from_S123(S1: float,
                                 S2: float,
                                 S3: float,
                                 a_param: float,
                                 RT_over_R0: float) -> Tuple[float, float, float]:
    """
    # $3\beta_p + l_i - \hat{l}_i = S_1 + S_2$, $\beta_p + l_i + \hat{l}_i = \frac{R_T}{R_0}S_2$, $\beta_p - (\alpha - 1)l_i - \hat{l}_i = S_3$
    # 3βₚ + ℓᵢ − ℓ̂ᵢ = S₁ + S₂, βₚ + ℓᵢ + ℓ̂ᵢ = (RT/R₀)S₂, βₚ − (α−1)ℓᵢ − ℓ̂ᵢ = S₃
    # A. A. Martynov, & V. D. Pustovitov. 2024. Virial relations for elongated plasmas in tokamaks- Analytical approximations and numerical calculations. Physics of Plasmas (Eq 1-3)
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


def virial_D0_boundary_from_bp_li_eK(beta_p: float,
                                    li_int: float,
                                    eK: float,
                                    b_minor: float,
                                    R_plasma: float) -> float:
    """
    # $D_0(b) = -\frac{b}{2R_{plasma}} \frac{2\beta_p + l_i + 0.5eK}{1 + 0.5eK}$
    # D₀(b) = −(b / 2R_plasma) · [2βₚ + ℓᵢ + 0.5 eK] / [1 + 0.5 eK]
    # A. A. Martynov, & V. D. Pustovitov. 2024. Virial relations for elongated plasmas in tokamaks- Analytical approximations and numerical calculations. Physics of Plasmas (Eq 37)
    """
    numerator = 2 * beta_p + li_int + 0.5 * eK
    denominator = 1 + 0.5 * eK
    return - (b_minor / (2 * R_plasma)) * numerator / denominator

"""
Power Density $S$
"""

# constants
k_B = 1.380649e-23        # J/K
eV_to_J = 1.602176634e-19 # J
K_B_COEF = 0.052          # MW/m^3 (with p in 1e5 Pa, T in keV)

def bremsstrahlung_power_density_from_T_e_p_Z_eff(
    T_e: float,
    p: float,
    Z_eff: float = 2.0
) -> float:
    """
    Bremsstrahlung power density using pressure form:

        S_B = Z_eff * K_B * p^2 / T_k^(3/2)   [MW/m^3]

    Inputs:
      - p: pressure [Pa]
      - T_e: electron temperature [eV]
      - Z_eff: effective charge number


    Internal conversions:
      - T_e (eV) → T_k (keV)
      - p (Pa) → p_bar = p / 1e5

    Output:
      - S_B in W/m^3
    """
    # convert pressure to 1e5 Pa units
    p_bar = p / 1e5
    # convert temperature to keV
    T_k_keV = T_e * 1e-3

    S_B_MW_m3 = K_B_COEF * (p_bar ** 2) / (T_k_keV ** 1.5)

    return Z_eff * S_B_MW_m3 * 1e6  # W/m^3


# physical constants
epsilon_0 = 8.8541878128e-12
c = 299792458.0
h = 6.62607015e-34
m_e = 9.10938356e-31
e = 1.602176634e-19

# prefactor
C_B = (
    np.sqrt(2.0) / (3.0 * np.pi ** 2.5)
    * e**6
    / (epsilon_0**3 * c**3 * h * m_e**1.5)
)

def bremsstrahlung_power_density_from_Z_eff_n_e_T_e(
    n_e_m3: float,
    T_e_eV: float,
    Z_eff: float = 2.0
) -> float:
    """
    Fundamental bremsstrahlung formula.
    
    $S_B = \left(\frac{2^{1/2}}{3\pi^{5/2}}\frac{e^6}{\varepsilon_0^3 c^3 h m_e^{3/2}}\right)Z_{\mathrm{eff}} n_e^2 T_e^{1/2}$ 
    # S_B = (2^(1/2) / (3π^(5/2)) * e^6 / (ε_0^3 c^3 h m_e^(3/2))) * Z_eff * n_e^2 * T_e^(1/2)
    Inputs:
      - n_e_m3 : electron density [m^-3]
      - T_e_eV : electron temperature [eV]

    Internal conversions:
      - T_e (eV) → T_J (J)

    Output:
      - S_B in W/m^3
    """

    T_J = T_e_eV * eV_to_J
    return C_B * Z_eff * n_e_m3**2 * np.sqrt(T_J)

"""
Flux Consumption
"""
def surface_poloidal_flux_from_psi_boundary(psi_boundary: np.ndarray) -> float:
    """
    $ \Phi_{surface} = 2 \pi \psi_{boundary} $
    # \Phi_{surface} = 2π \psi_{boundary}
    """
    return psi_boundary * 2 * np.pi

def loop_voltage_from_total_flux(time_slice: np.ndarray, psi_boundary: np.ndarray) -> float:
    """
    $ $V_{loop} = d/dt \Phi_{surface} $
    # V_loop = d/dt Φ_surface
    """
    return gradient(time_slice, psi_boundary) * 2 * np.pi

def inductive_voltage_from_dW_magdt_I_p(dW_magdt: float, I_p: float) -> float:
    """
    $ $V_{ind} = \frac{dW_{mag}}{dt} / I_p $
    # V_ind = dW_mag/dt / I_p
    """
    return dW_magdt / I_p


"""
------------------------------------------------------------------
Power Balance
------------------------------------------------------------------
"""

def ohmic_heating_power_from_I_p_V_res(I_p: float,
                                        V_res: float) -> float:
    """
    # $P_{ohm} = I_p V_res$
    # if we use V_loop instead of V_res,it assumes no inductive heating (fully resistive heating).
    """
    return I_p * V_res


def alpha_heating_power_from_n_D_n_T_T_keV_V(
    n_D_1e19: float, n_T_1e19: float, T_keV: float, V_m3: float
) -> float:
    n_D = n_D_1e19 * 1e19  # m^-3
    n_T = n_T_1e19 * 1e19  # m^-3
    sigma_v = SIGMA_V_COEF * T_keV**2  # m^3/s (rough fit)
    return n_D * n_T * sigma_v * E_ALPHA * V_m3  # W


alpha_heating_power = alpha_heating_power_from_n_D_n_T_T_keV_V

def nbi_heating_power_from_I_nbi_V_nbi(I_nbi: float,
                                      V_nbi: float) -> float:
    """
    # $P_{nbi} = I_{nbi} V_{nbi}$
    # P_nbi = I_nbi V_nbi
    """
    return I_nbi * V_nbi

def ec_heating_power_from_I_ec_V_ec(I_ec: float,
                                    V_ec: float) -> float:
    """
    # $P_{ec} = I_{ec} V_{ec}$
    # P_ec = I_ec V_ec
    """
    return I_ec * V_ec


def auxiliary_heating_power(P_aux: float,
                          eta_CD: float) -> Tuple[float, float]:
    """
    # $P_{CD} = \frac{P_{aux}}{1 + \eta_{CD}}$, $P_{heat} = P_{aux} - P_{CD}$
    # P_CD = P_aux / (1 + η_CD), P_heat = P_aux - P_CD
    """
    P_CD = P_aux / (1 + eta_CD)
    P_heat = P_aux - P_CD
    return P_heat, P_CD


def heating_power_from_p_ohm_p_aux(P_ohm: float, P_aux: float) -> float:
    """
    # $P_{heat} = P_{ohm} + P_{aux}$
    # P_heat = P_ohm + P_aux
    """
    return P_ohm + P_aux


def cyclotron_radiation_power_from_z_eff_n_e_t_e(Z_eff: float,
                                        n_e: float,
                                        T_e_keV: float) -> float:
    """
    # $p_{\mathrm{cyc}} \approx 1.69\times 10^{-38}\, Z_{\mathrm{eff}}\, n_e^2\, \sqrt{T_e} \quad [\mathrm{W/m^3}]$
    """
    return 1.69e-38 * Z_eff * n_e**2 * np.sqrt(T_e_keV)

def line_radiation_power_from_z_eff_n_e_t_e(Z_eff: float,
                                            n_e: float,
                                            T_e_keV: float) -> float:
    """
    Line radiation power density (placeholder - implementation needed).
    
    Parameters
    ----------
    Z_eff : float
        Effective charge number
    n_e : float
        Electron density [m^-3]
    T_e_keV : float
        Electron temperature [keV]
    
    Returns
    -------
    float
        Line radiation power density [W/m^3]
    """
    # TODO: Implement line radiation power density calculation
    return 0.0

def radiation_power_from_p_brem_p_cyc_p_line(p_brem: float, p_cyc: float, p_line: float) -> float:
    """
    # $p_{\mathrm{rad}} = p_{\mathrm{Br}} + p_{\mathrm{cyc}} + p_{\mathrm{line}}$
    # p_rad = p_brem + p_cyc + p_line
    """
    return p_brem + p_cyc + p_line


def loss_power_from_p_heat_dWdt_p_rad(P_heat: float, dWdt: float, p_rad: float) -> float:
    """
    # $P_{loss} = P_{heat} - \frac{dW}{dt} - p_{rad}$
    # P_loss = P_heat - dW/dt - p_rad
    # lf p_rad = 0, it means toal loss power in plasma, if p_rad > 0, it means loss power from plasma to boundary.
    """
    return P_heat - dWdt - p_rad

"""
Dimensionless Parameters
"""

def inverse_aspect_ratio_from_a_R(a: float, R: float) -> float:
    """
    # $\varepsilon = \frac{a}{R}$
    # \varepsilon = a / R
    # Inverse aspect ratio
    """
    return a / R

def aspect_ratio_from_a_R(a: float, R: float) -> float:
    """
    # $\frac{R}{a} = \frac{1}{\varepsilon}$
    # R / a = 1 / \varepsilon
    # Aspect ratio (not to be confused with κ, which is elongation)
    """
    return R / a



def normalized_larmor_radius_from_M_T_a_Bt(M: float,
                               T: float,
                               a: float,
                               Bt: float) -> float:
    """
    # $\rho_* = \frac{\rho_i}{a}$ with $\rho_i = \frac{\sqrt{2 m_i T_i}}{e B_T}$
    # ρ* = ρ_i / a, ρ_i = √(2 m_i (e T_i[eV])) / (e B_T)
    # Normalized toroidal Larmor radius (ion gyroradius / minor radius)
    # 
    # Parameters
    # ----------
    # M : float
    #     Ion mass [kg]
    # T : float
    #     (Ion) temperature [eV]
    # a : float
    #     Minor radius [m]
    # Bt : float
    #     Toroidal magnetic field [T]
    #
    # Returns
    # -------
    # float
    #     Normalized toroidal Larmor radius [-]
    """
    # ρ* ∝ √(M T) / (a B_T)  (image scaling); constants kept explicitly
    return np.sqrt(2.0 * M * QE * T) / (QE * Bt * a)

def normalized_collisionality_from_nu_ii_T_i_M_i_R_a_q(nu_ii: float,
                                     T_i_eV: float,
                                     M_i: float,
                                     R: float,
                                     a: float,
                                     q: float) -> float:
    """
    # $\nu^* = \nu_{ii} \left(\frac{M_i}{eT_i}\right)^{1/2} \left(\frac{R}{a}\right)^{3/2} qR$
    # ν* = ν_ii * (M_i/(eT_i))^(1/2) * (R/a)^(3/2) * qR
    # Normalized collisionality (connection length / trapped particle mean-free path)
    #
    # Parameters
    # ----------
    # nu_ii : float
    #     Ion-ion collision frequency [s^-1]
    # T_i_eV : float
    #     Ion temperature [eV]
    # M_i : float
    #     Ion mass [kg]
    # R : float
    #     Major radius [m]
    # a : float
    #     Minor radius [m]
    # q : float
    #     Cylindrical safety factor [-]
    #
    # Returns
    # -------
    # float
    #     Normalized collisionality [-]
    """
    # Note: ν* is dimensionless; this form matches the common scaling ν* ~ ν_ii qR / v_th · (R/a)^{3/2}
    return nu_ii * np.sqrt(M_i / (QE * T_i_eV)) * ((R / a)**1.5) * q * R

def normalized_collisionality_from_a_n_q_epsilon_T(a: float,
                                                   n: float,
                                                   q: float,
                                                   epsilon: float,
                                                   T_eV: float,
                                                   C: float = 1.0) -> float:
    """
    # $\nu_* \propto \frac{a\, n\, q}{\varepsilon^{5/2}\, T^2}$
    # ν* ∝ a n q / (ε^(5/2) T^2)
    #
    # Dimensionless collisionality scaling form (as in the provided figure).
    #
    # Parameters
    # ----------
    # a : float
    #     Minor radius [m]
    # n : float
    #     Density [m^-3]
    # q : float
    #     Safety factor [-]
    # epsilon : float
    #     Inverse aspect ratio a/R [-]
    # T_eV : float
    #     Temperature [eV]
    # C : float
    #     Optional proportionality constant (default 1.0). Use to calibrate against a specific ν* definition.
    #
    # Returns
    # -------
    # float
    #     Normalized collisionality (scaled) [-]
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if T_eV <= 0:
        raise ValueError("T_eV must be > 0")
    return C * (a * n * q) / (epsilon**2.5 * T_eV**2)

def cylindrical_safety_factor_from_R_B_epsilon_I_f_kappa_delta(R: float,
                                           B: float,
                                           epsilon: float,
                                           I: float,
                                           f_kappa_delta: float) -> float:
    """
    # $q_{cyl} \approx \frac{2\pi\,\varepsilon^2\, R\, B_T}{\mu_0\, I_p\, f(\kappa,\delta)}$
    # q_cyl ≈ (2π ε² R B_T) / (μ₀ I_p f(κ,δ))
    #
    # (Using ε=a/R) this is consistent with the common scaling q ∝ B_T a² /(R I_p)
    # and the figure’s q ∝ (B_T r ε a)/I_p up to O(1) geometry constants.
    # Cylindrical safety factor
    #
    # Parameters
    # ----------
    # R : float
    #     Major radius [m]
    # B : float
    #     Toroidal magnetic field [T]
    # epsilon : float
    #     Inverse aspect ratio (a/R) [-]
    # I : float
    #     Plasma current [A]
    # f_kappa_delta : float
    #     Function of plasma shape parameters κ (elongation) and δ (triangularity) [-]
    #
    # Returns
    # -------
    # float
    #     Cylindrical safety factor [-]
    """
    return (2.0 * np.pi * epsilon**2 * R * B) / (MU0 * I * f_kappa_delta)


"""
Confinement Time
"""
def confinement_time_from_P_loss_W_th(P_loss: float, W_th: float) -> float:
    """
    # $\tau_{E,\text{th}}^{\text{fit}} = \frac{W_{\text{th}}}{P_{\text{loss}}}$
    # \tau_{E,\text{th}}^{\text{fit}} = W_th / P_loss
    """
    return W_th / P_loss

def confinement_time_from_engineering_parameters(I_p: float, B_t: float, P_loss: float, n_e: float, M: float, R: float, epsilon: float, kappa: float, scaling: str = "IBP98y2") -> float:
    """
    Calculate thermal energy confinement time from engineering parameters using scaling laws.
    
    Formula:
    τ_E,th^fit = C · I_p^α_I · B_T^α_B · P_loss^α_P · n_e^α_n · M^α_M · R^α_R · ε^α_ε · κ^α_κ
    
    Parameters
    ----------
    I_p : float
        Plasma current [A] (converted to MA internally)
    B_t : float
        Toroidal magnetic field [T]
    P_loss : float
        Loss power [W] (converted to MW internally)
    n_e : float
        Line-averaged electron density [m^-3] (converted to 10^20 m^-3 internally)
    M : float
        Average ion mass [amu]
    R : float
        Major radius [m]
    epsilon : float
        Inverse aspect ratio (a/R) [-]
    kappa : float
        Elongation [-]
    scaling : str, optional
        Scaling law name: "IBP98y2", "H98y2", "Petty", "NSTX-MG", or "NSTX", by default "IBP98y2"
    
    Returns
    -------
    float
        Thermal energy confinement time [s]
    
    Notes
    -----
    This function automatically converts SI units to the units required by the scaling laws:
    - I_p: A → MA (× 1e-6)
    - n_e: m^-3 → 10^20 m^-3 (× 1e-20)
    - P_loss: W → MW (× 1e-6)
    """
    if scaling not in _SCALING_COEFS:
        raise ValueError(f"Unknown scaling '{scaling}'. Available: {list(_SCALING_COEFS.keys())}")
    
    # Extract coefficients: [C, α, β, γ, δ, ε, ζ, η]
    # Mapping to: [α_I, α_B, α_P, α_n, α_M, α_R, α_ε, α_κ]
    # Based on: τ_E = C · I_p^α · R^β · a^γ · κ^δ · n_e^ε · B_T^ζ · M^η / P_loss
    # For this function: τ_E = C · I_p^α_I · B_T^α_B · P_loss^α_P · n_e^α_n · M^α_M · R^α_R · ε^α_ε · κ^α_κ
    C, alpha, beta, gamma, delta, epsilon_coef, zeta, eta = _SCALING_COEFS[scaling]
    alpha_I, alpha_B, alpha_P, alpha_n, alpha_M, alpha_R, alpha_epsilon, alpha_kappa = (
        alpha, zeta, -beta, epsilon_coef, eta, beta, gamma, delta
    )
    
    # Unit conversions: SI → scaling law units
    I_p_MA = I_p * 1e-6          # A → MA
    n_e_20 = n_e * 1e-20         # m^-3 → 10^20 m^-3
    P_loss_MW = P_loss * 1e-6    # W → MW
    
    return (C * I_p_MA**alpha_I * B_t**alpha_B * P_loss_MW**alpha_P * 
            n_e_20**alpha_n * M**alpha_M * R**alpha_R * 
            epsilon**alpha_epsilon * kappa**alpha_kappa)


def confinement_factor_from_tau_E_exp_tau_E_IPB89y2(tau_E_exp: float, tau_E_IBP98y2: float) -> float:
    """
    # $\frac{\tau_{E,\text{exp}}}{\tau_{E,\text{IBP98y2}}}$
    # \frac{\tau_{E,\text{exp}}}{\tau_{E,\text{IBP98y2}}}
    """
    return tau_E_exp / tau_E_IBP98y2

def dimensionless_scaling_coeffs_from_engineering_scaling_coeffs(
    a_I, a_B, a_P, a_n, a_M, a_R, a_eps, a_kappa
):
    """
    Converts engineering scaling indices (alpha) to dimensionless scaling indices (mu).
    
    The conversion follows the order: α_I, α_B, α_P, α_n, α_M, α_R, α_ε, α_κ.
    It assumes the safety factor q is constant, using the relation: I_p ∝ (a^2 * B) / R.
    """
    
    # 1. Basis Transformation (Consolidating to fundamental dimensions: L and B)
    # Using the relation I_p ∝ R * eps^2 * B (since a = R * eps)
    a_L = a_R + a_I        # Combined length (L) scaling index
    a_B_star = a_B + a_I   # Combined magnetic field (B) scaling index
    # Note: a_n (density) and a_P (power) remain as primary engineering inputs.

    # 2. Derive Dimensionless Indices (mu) via Power Balance
    # Normalized confinement time follows: Ω_c * τ_E ∝ (ρ*)^μ_ρ * β^μ_β * (ν*)^μ_ν
    # The derivation eliminates Temperature (T) using P = W / τ_E.
    
    denom = 1 + a_P
    if abs(denom) < 1e-9:
        return None

    # Mapping based on Gyro-kinetic transport theory and Kadomtsev's similarity principles
    # mu_rho: Characterizes size scaling (e.g., -3 for Gyro-Bohm, -2 for Bohm)
    mu_rho = (3 * a_L + a_B_star + a_n - 2 * a_P - 5) / denom
    
    # mu_beta: Characterizes plasma pressure scaling
    mu_beta = (-a_L - 2 * a_n - a_B_star + 3 * a_P + 3) / denom
    
    # mu_nu: Characterizes collisionality scaling
    mu_nu = (a_L + 3 * a_n + a_B_star - 2 * a_P - 4) / (2 * denom)

    # round to 3 decimal places
    mu_rho = round(mu_rho, 3)
    mu_beta = round(mu_beta, 3)
    mu_nu = round(mu_nu, 3)
    mu_M = round(a_M, 3)
    mu_kappa = round(a_kappa, 3)

    return mu_rho, mu_beta, mu_nu, mu_M, mu_kappa

def verify_kadomtsev_constraint(mu_rho, mu_beta, mu_nu, a_P):
    """
    Verifies the Kadomtsev constraint using derived dimensionless indices.
    
    In a physically consistent Vlasov-Maxwell system, the engineering indices 
    must satisfy: x = α_L + 2α_n + α_B* - 3α_P = 5.
    This function checks if the dimensionless mapping preserves this identity.
    """
     
    # Reconstructing the constraint value x from the dimensionless indices
    # For a purely physical model, calculated_x should converge to 5.0.
    x_val = 5.0 + (mu_rho * (1 + a_P) - (3 * (mu_rho + 2 * mu_beta - 4 * mu_nu - 2) / 2))
    
    return x_val