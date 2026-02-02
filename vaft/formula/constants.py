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

# Confinement scaling coefficients
# Structure: {scaling_name: {parameter_name: value}}
# Parameters: C, Ip_MA, R, epsilon, kappa, n_19, Bt, Mi (optional), P_MW
# Ref: https://ukaea.github.io/PROCESS/physics-models/plasma_confinement/
_SCALING_COEFS = {
    "IPB89": {
        "C": 0.038,          # Constant
        "Ip_MA": 0.85,       # Plasma Current [MA]
        "R": 1.50,         # Major Radius [m]
        "epsilon": 0.30,         # Inverse aspect ratio (a/R)
        "kappa": 0.50,       # Elongation
        "n_19": 0.10,       # Electron Density [10^19 m^-3]
        "Bt": 0.20,          # Toroidal Field [T]
        "Mi": 0.50,           # Ion Mass number
        "P_MW": -0.50        # Total Heating Power [MW]
    },
    "H98y2": {
        "C": 0.0562,          # Constant
        "Ip_MA": 0.93,       # Plasma Current [MA]
        "R": 1.97,         # Major Radius [m]
        "epsilon": 0.58,         # Inverse aspect ratio (a/R)
        "kappa": 0.78,       # Elongation
        "n_19": 0.41,       # Electron Density [10^19 m^-3]
        "Bt": 0.15,          # Toroidal Field [T]
        "Mi": 0.19,         # Ion Mass number
        "P_MW": -0.69,         # Total Heating Power [MW]
    },
    "NSTX": { # Ref: Menard 2019. Assume M, R, epsilon, kappa depedency are identical to H98y2
        "C": 0.095,          # Constant
        "Ip_MA": 0.57,       # Plasma Current [MA]
        "R": 1.97,         # Major Radius [m]
        "epsilon": 0.58,         # Inverse aspect ratio (a/R)
        "kappa": 0.78,       # Elongation
        "n_19": 0.44,       # Electron Density [10^19 m^-3]
        "Bt": 1.08,          # Toroidal Field [T]
        "Mi": 0.19,         # Ion Mass number
        "P_MW": -0.73            # Total Heating Power [MW]
    },
} 

