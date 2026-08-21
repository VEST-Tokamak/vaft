"""Deprecated compatibility import for kinetic-constrained EFIT.

The canonical API lives in :mod:`vaft.code.efit`. This module remains for one
deprecation cycle so existing clients can migrate without an abrupt import
failure.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "vaft.code.kineticEfit is deprecated; use vaft.code.efit for magnetic and "
    "kinetic-constrained EFIT workflows.",
    DeprecationWarning,
    stacklevel=2,
)

from .efit import (  # noqa: E402,F401
    EQE,
    KineticEFITConfig,
    KineticEFITInputs,
    KineticEFITResult,
    PressurePoints,
    build_kinetic_core_profiles,
    inject_pressure_constraint,
    kinetic_pressure_points,
    prepare_kinetic_efit_inputs,
    run_kinetic_chain,
    run_kinetic_efit,
    scale_plasma,
)

__all__ = [
    "EQE",
    "KineticEFITConfig",
    "KineticEFITInputs",
    "KineticEFITResult",
    "PressurePoints",
    "build_kinetic_core_profiles",
    "inject_pressure_constraint",
    "kinetic_pressure_points",
    "prepare_kinetic_efit_inputs",
    "run_kinetic_chain",
    "run_kinetic_efit",
    "scale_plasma",
]
