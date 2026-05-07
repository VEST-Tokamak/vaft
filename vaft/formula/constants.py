"""
Physical constants used in plasma physics calculations.

This module provides a centralized location for all physical constants
used throughout the formula module.
"""

import numpy as np

# Electromagnetic constants
MU0 = 4 * np.pi * 1e-7      # [H m⁻¹] - Vacuum permeability
EPS0 = 8.8541878128e-12     # [F m⁻¹] - Vacuum permittivity

# Particle properties
QE = 1.602176634e-19        # [C] - Elementary charge
ME = 9.10938356e-31         # [kg] - Electron mass
MI_P = 1.67262192e-27       # [kg] - Proton mass

# Fusion related constants
E_ALPHA = 3.5e6 * QE        # [J] - Alpha particle energy
SIGMA_V_COEF = 1.1e-24      # [m³/s] - Fusion cross section coefficient

# Transport coefficients
SPITZER_RESISTIVITY_COEF = 5.2e-5  # [Ω·m] - Spitzer resistivity coefficient
COLLISIONALITY_COEF = 6.921e-18    # [-] - Collisionality coefficient

# -----------------------------------------------------------------------------
# Confinement-time scaling coefficients
#
# Important unit convention
# -------------------------
# This dictionary is intentionally normalized to a single internal input convention
# so that one evaluator can be used for all scalings:
#
#   Ip_MA : plasma current in MA
#   n_19  : density in 1e19 m^-3
#   P_MW  : power in MW
#   R     : major radius in m
#   Bt    : toroidal magnetic field in T
#
# Why some coefficients were converted
# ------------------------------------
# Not all source papers publish their regressions in the same units.
# - Classical confinement scalings such as ITER89P and H98(y,2) are already
#   written in the usual fusion engineering convention.
# - Some NSTX regressions (e.g. Kaye 2006 table values) are explicitly published
#   in SI-like variables: A, W, m^-3, s.
#
# To avoid mixing incompatible prefactors inside one code path, the SI-published
# regressions were converted into the same engineering-unit convention used by
# the rest of this module. The exponents are unchanged; only the prefactor C is
# transformed.
#
# Practical implication
# ---------------------
# The prefactor C is unit-dependent. It is not universal and must always be used
# together with the exact variable convention assumed by the corresponding entry.
# -----------------------------------------------------------------------------

# Confinement scaling coefficients
# Structure:
# {
#   scaling_name: {
#       "C": float,
#       "exponents": {variable_name: exponent},
#       "density_definition": "line_avg" | "volume_avg",
#       "reference": str,
#       # optional metadata fields
#   }
# }
_SCALING_COEFS = {
    "ITER89P": {
        "C": 0.038,
        "exponents": {
            "Ip_MA": 0.85,    # Plasma current [MA]
            "R": 1.50,        # Major radius [m]
            "epsilon": 0.30,  # Inverse aspect ratio a/R [-]
            "kappa": 0.50,    # Elongation [-]
            "n_19": 0.10,     # Line-averaged electron density [10^19 m^-3]
            "Bt": 0.20,       # Toroidal magnetic field [T]
            "Mi": 0.50,       # Effective ion mass number [amu]
            "P_MW": -0.50,    # Total heating / loss power [MW]
        },
        "density_definition": "line_avg",
        "reference": "ITER Physics Basis 1989 L-mode scaling (ITER89P)",
        "unit_convention_note": (
            "This scaling is conventionally published in fusion engineering units rather than strict SI. "
            "Accordingly, the prefactor C is valid only when Ip is supplied in MA, density in 1e19 m^-3, "
            "and power in MW. If SI inputs (A, m^-3, W) are used directly, the prefactor must be re-derived."
        ),
        "implementation_note": (
            "Kept in its original community-standard unit convention for consistency with the literature and "
            "with common confinement-scaling practice."
        ),
    },

    "H98y2": {
        "C": 0.0562,
        "exponents": {
            "Ip_MA": 0.93,    # Plasma current [MA]
            "R": 1.97,        # Major radius [m]
            "epsilon": 0.58,  # Inverse aspect ratio a/R [-]
            "kappa": 0.78,    # Elongation [-]
            "n_19": 0.41,     # Line-averaged electron density [10^19 m^-3]
            "Bt": 0.15,       # Toroidal magnetic field [T]
            "Mi": 0.19,       # Effective ion mass number [amu]
            "P_MW": -0.69,    # Total heating / loss power [MW]
        },
        "density_definition": "line_avg",
        "reference": "IPB98(y,2) ELMy H-mode scaling",
        "unit_convention_note": (
            "This expression is also defined in the standard confinement-scaling convention: "
            "Ip[MA], n_e[1e19 m^-3], P[MW], R[m], B[T], etc. The coefficient is not dimensionless and "
            "should not be interpreted as SI-universal."
        ),
        "implementation_note": (
            "Stored in the original published unit convention so that comparisons against H98(y,2) values "
            "from papers, databases, and plotting scripts remain direct."
        ),
    },

    "NSTX2006H": {
        "C": 0.0715,
        "exponents": {
            "Ip_MA": 0.57,   # Plasma current [MA]
            "Bt": 1.08,      # Toroidal magnetic field [T]
            "n_19": 0.44,    # Line-averaged electron density [10^19 m^-3]
            "P_MW": -0.73,   # Thermal loss power [MW]
        },
        "density_definition": "line_avg",
        "reference": "S.M. Kaye et al., 2006, Nucl. Fusion 46 848 (NSTX H-mode thermal scaling)",
        "notes": (
            "Implemented using only explicitly published NSTX2006 H-mode dependencies; "
            "unspecified dependencies (R, epsilon, kappa, Mi) are not assumed."
        ),
        "source_unit_note": (
            "The original paper tabulates this regression in SI-like variables: "
            "Ip[A], n_e[m^-3], P[W], tau_E[s]."
        ),
        "conversion_note": (
            "The coefficient here has been converted from the paper's SI form to the codebase's common "
            "engineering-unit convention, i.e. Ip[MA], n_e[1e19 m^-3], and P[MW]. "
            "This was done so all scalings in this dictionary can be evaluated through a single shared interface "
            "without per-model unit branching."
        ),
        "consistency_note": (
            "After conversion, this expression is numerically equivalent to the published NSTX2006 regression "
            "as long as the inputs are supplied in the convention used here."
        ),
    },

    "NSTX2006L": {
        "C": 0.141,
        "exponents": {
            "Ip_MA": 1.01,   # Plasma current [MA]
            "Bt": 0.70,      # Toroidal magnetic field [T]
            "n_19": 0.07,    # Line-averaged electron density [10^19 m^-3]
            "P_MW": -0.37,   # Loss power [MW]
        },
        "density_definition": "line_avg",
        "reference": "S.M. Kaye et al., 2006, Nucl. Fusion 46 848 (NSTX L-mode global scaling)",
        "notes": (
            "Implemented using only explicitly published NSTX2006 L-mode dependencies; "
            "unspecified dependencies (R, epsilon, kappa, Mi) are not assumed."
        ),
        "source_unit_note": (
            "The original publication gives this regression in SI-like variables: "
            "Ip[A], n_e[m^-3], P[W], tau_E[s]."
        ),
        "conversion_note": (
            "The prefactor used here is the SI-published coefficient converted into the internal engineering-unit "
            "convention of this module: Ip[MA], n_e[1e19 m^-3], P[MW]. "
            "This avoids mixing SI-only and convention-only formulas inside one scaling evaluator."
        ),
        "consistency_note": (
            "Do not combine this prefactor with SI inputs directly. The conversion is already absorbed into C."
        ),
    },

    "Kurskiev2022": {
        "C": 0.066,
        "exponents": {
            "Ip_MA": 0.53,   # Plasma current [MA]
            "Bt": 1.05,      # Toroidal magnetic field [T]
            "P_MW": -0.58,   # Absorbed / effective heating power [MW]
            "n_19": 0.65,    # Density in convention units [10^19 m^-3]
            "R": 2.66,       # Major radius [m]
            "kappa": 0.78,   # Elongation [-]
        },
        "density_definition": "line_avg",
        "reference": "G.S. Kurskiev et al., 2022, Nucl. Fusion 62 016011 (ST multi-machine H-mode engineering scaling)",
        "source_interpretation_note": (
            "Although the paper does not restate all units directly in the regression equation, the coefficient "
            "magnitude and surrounding dataset strongly indicate the usual fusion engineering convention rather than "
            "strict SI. In practice, the expression is most consistently read as Ip[MA], n_e[1e19 m^-3], and P[MW]."
        ),
        "conversion_note": (
            "No additional unit conversion was applied here; the stored coefficient is already aligned with the "
            "community-standard convention used by the original scaling and by the rest of this module."
        ),
        "power_note": (
            "The original publication reports absorbed power dependence. "
            "Current implementation maps this onto the existing P_loss / effective-power argument convention. "
            "Use with care if the supplied power is not close to the absorbed-power definition used in the paper."
        ),
        "density_note": (
            "This scaling is treated here as using volume-averaged density. "
            "If only line-averaged density is available, an explicit conversion or approximation should be applied "
            "before evaluation."
        ),
        "scope_note": (
            "This is a multi-machine ST H-mode regression. It should be interpreted as an empirical engineering fit, "
            "not as a dimensionally universal SI law."
        ),
    },
}
