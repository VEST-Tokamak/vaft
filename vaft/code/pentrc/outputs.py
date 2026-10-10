"""Compatibility path for :mod:`vaft.code.gpec._pentrc_output` (#1883).

The PENTRC reader moved into the GPEC suite that runs PENTRC; this submodule
re-exports the same objects so existing ``vaft.code.pentrc.outputs`` imports
keep working.  New code imports from :mod:`vaft.code.gpec`.
"""

from ..gpec._pentrc_output import (
    PROFILE_VARIABLES,
    TORQUE_GRIDS,
    TORQUE_LONG_NAME,
    TORQUE_METHODS,
    TORQUE_QUANTITIES,
    PentrcFormatError,
    PentrcOutput,
    energy_profile,
    read_pentrc_output,
    torque_profile,
)

__all__ = [
    "PROFILE_VARIABLES",
    "TORQUE_GRIDS",
    "TORQUE_LONG_NAME",
    "TORQUE_METHODS",
    "TORQUE_QUANTITIES",
    "PentrcFormatError",
    "PentrcOutput",
    "energy_profile",
    "read_pentrc_output",
    "torque_profile",
]
