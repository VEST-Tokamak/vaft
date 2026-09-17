"""TRANSP output adapter -- read-only.

TRANSP is run elsewhere; VAFT reads what it produced.  There is no
``inputs.py``, no ``runner.py`` and no ``$TRANSPHOME``, because there is
nothing to configure or launch:

    <runid>.CDF ── read_transp_output ──▶ TranspOutput   (lazy, ~1900 variables)
                ── .slice(time_s) ───────▶ TranspSlice   (one time, both grids)
                ── enclosed_torque ──────▶ (psi_norm, torque [N m])
                ── read_transp_profiles ─▶ KineticProfiles  (the one conversion)

This layer keeps TRANSP's own names, grids and units and converts nothing --
``NE`` is still per cubic centimetre when it reaches you.  The one exception
is :mod:`~vaft.code.transp.profiles`, which converts into VAFT's
kinetic-profile container and is a separate module for that reason: what the
file said and what someone made of it stay distinguishable.

Typical use::

    from vaft.code import transp

    with transp.read_transp_output("45453X01.CDF") as output:
        state = output.slice(0.750)          # nearest sample; state.time_s says which
        density = state.on_x("NE")           # cm^-3, refuses a zone-boundary variable
        psi_norm, torque = transp.enclosed_torque(state)
"""

from .config import TRANSPResult, collect_transp_outputs
from .outputs import (
    EMPTY_PLACEHOLDERS,
    PROFILE_GRIDS,
    TIME_DIMENSIONS,
    VARIABLE_DESCRIPTIONS,
    TranspFormatError,
    TranspOutput,
    TranspSlice,
    TranspVariable,
    read_transp_output,
)
from .profiles import (
    PROFILE_SOURCES,
    TRANSP_UNIT_LADDER,
    exb_frequency,
    kinetic_profiles_from_slice,
    read_transp_profiles,
)
from .torque import enclosed_torque, input_torque_density, zone_volume

__all__ = [
    "EMPTY_PLACEHOLDERS",
    "PROFILE_GRIDS",
    "PROFILE_SOURCES",
    "TIME_DIMENSIONS",
    "TRANSP_UNIT_LADDER",
    "TRANSPResult",
    "TranspFormatError",
    "TranspOutput",
    "TranspSlice",
    "TranspVariable",
    "VARIABLE_DESCRIPTIONS",
    "collect_transp_outputs",
    "enclosed_torque",
    "exb_frequency",
    "input_torque_density",
    "kinetic_profiles_from_slice",
    "read_transp_output",
    "read_transp_profiles",
    "zone_volume",
]
