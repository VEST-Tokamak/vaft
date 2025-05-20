# formulas_flux_consumption.py
# ------------------------------------------------------------
# Ohmic-flux consumption, Ejima 계수, 내부 인덕턴스 li  (NSTX)
# Based on: J.E. Menard et al., Nucl. Fusion 41 (2001)
# ------------------------------------------------------------

import numpy as np
MU0 = 4*np.pi*1e-7   # [H m⁻¹]

# ------------------------------------------------------------------
# 1. Inductive poloidal-flux consumption  ΔΦ_I  (axial method, Eq 4)
#    ΔΦ_I = h_i μ₀ R₀ I_P / 2
# ------------------------------------------------------------------
def dphiI_from_hi_R_Ip(h_i: float,
                       R0: float,
                       I_p: float) -> float:
    """Inductive flux consumption ΔΦ_I  [Wb]."""
    return 0.5 * h_i * MU0 * R0 * I_p            #  [oai_citation:0‡J. E. Menard, B. P. LeBlanc et al. 2001. Ohmic flux consumption during initial operation of the NSTX spherical torus. Nuclear fusion.pdf](file-service://file-NccMMqFqHftEHrw394QPVo)


# ------------------------------------------------------------------
# 2. Resistive poloidal-flux consumption  ΔΦ_R
#    ΔΦ_R(t) = −∫_0^t V_loop,axis dt′   (Eq 3)
# ------------------------------------------------------------------
def dphiR_from_Vloop_time(V_axis: np.ndarray,
                          t: np.ndarray) -> float:
    """
    Resistive flux consumption ΔΦ_R at time t[-1].
    V_axis, t : 1-D arrays [V], [s]   (monotonic t).
    """
    return -np.trapz(V_axis, t)                  # sign per Eq 3


# ------------------------------------------------------------------
# 3. Ejima (C_E) & Ejima-Wesley (C_EW) coefficients  (text around Eq 9)
# ------------------------------------------------------------------
def CE_from_dphiR_R_Ip(dphi_R: float,
                       R0: float,
                       I_p: float) -> float:
    """C_E = ΔΦ_R / (μ₀ R₀ I_P)"""
    return dphi_R / (MU0 * R0 * I_p)             #  [oai_citation:1‡J. E. Menard, B. P. LeBlanc et al. 2001. Ohmic flux consumption during initial operation of the NSTX spherical torus. Nuclear fusion.pdf](file-service://file-NccMMqFqHftEHrw394QPVo)


def CEW_from_dphiI_dphiR_R_Ip(dphi_I: float,
                              dphi_R: float,
                              R0: float,
                              I_p: float) -> float:
    """C_E-W = (ΔΦ_I + ΔΦ_R) / (μ₀ R₀ I_P)"""
    return (dphi_I + dphi_R) / (MU0 * R0 * I_p)  #  [oai_citation:2‡J. E. Menard, B. P. LeBlanc et al. 2001. Ohmic flux consumption during initial operation of the NSTX spherical torus. Nuclear fusion.pdf](file-service://file-NccMMqFqHftEHrw394QPVo)


# ------------------------------------------------------------------
# 4. Internal inductance l_i   (Eq 13)
#    l_i = 〈B_P²〉_V / 〈B_P〉_l²
# ------------------------------------------------------------------
def li_from_Bp2volavg_Bplineavg(Bp2_vol_avg: float,
                                Bp_line_avg: float) -> float:
    """
    Compute l_i given ⟨B_P²⟩_V and line-average ⟨B_P⟩_l.

    Parameters
    ----------
    Bp2_vol_avg : ⟨B_P²⟩_V   (volume-average of B_P²)   [T²]
    Bp_line_avg : ⟨B_P⟩_l     (line-average of B_P at LCFS) [T]

    Returns
    -------
    l_i : dimensionless internal inductance
    """
    return Bp2_vol_avg / (Bp_line_avg**2)        #  [oai_citation:3‡J. E. Menard, B. P. LeBlanc et al. 2001. Ohmic flux consumption during initial operation of the NSTX spherical torus. Nuclear fusion.pdf](file-service://file-NccMMqFqHftEHrw394QPVo)