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
_SCALING_COEFS = {
    "IBP98(y,2) L mode": [0.048, 0.85, 1.20, 0.30, 0.50, 0.10, 0.20, 0.50],
    "H98y2":             [0.145, 0.93, 1.39, 0.58, 0.78, 0.41, 0.15, 0.19],
    "Petty":             [0.052, 0.75, 2.09, 0.88, 0.84, 0.32, 0.30,-0.47],
    "NSTX-MG":           [0.056, 0.93, 1.39, 0.58, 0.78, 0.41, 0.15, 0.19],
} 