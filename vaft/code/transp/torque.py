"""Torque from a TRANSP run: the input density, and what it encloses.

Two facts decide everything here, and the code this replaces had both wrong.

``TQTOTNB`` is an empty placeholder -- a dimensionless scalar of value zero in
every run checked -- so the total input torque is ``TQIN``.  It is *not* the
beam sum: at 750 ms of the MAST reference run ``sum(TQIN)`` is -5.607e-7
against ``sum(TQTOT01 + TQTOT02)`` = -5.731e-7, a pointwise difference of the
same order as the signal.  ``TQIN`` is what TRANSP calls the total input
torque; the decomposition is a separate question and is not asserted here.

``TQIN`` is a density in ``N m / cm^3`` and ``DVOL`` a volume in ``cm^3``,
both on the zone-centre grid, so their product is already newton metres.
Converting one of them to SI and not the other is wrong by a factor of a
million, which is why the pairing happens once, here, rather than at each call
site.
"""

from __future__ import annotations

import numpy as np

from .outputs import TranspSlice

__all__ = ["enclosed_torque", "input_torque_density"]


def input_torque_density(slice: TranspSlice) -> np.ndarray:
    """``TQIN`` at this time, in TRANSP's own units [N m / cm^3], on ``X``."""
    return slice.on_x("TQIN")


def zone_volume(slice: TranspSlice) -> np.ndarray:
    """``DVOL`` at this time [cm^3], on ``X`` -- the volume of each zone."""
    return slice.on_x("DVOL")


def enclosed_torque(slice: TranspSlice) -> tuple[np.ndarray, np.ndarray]:
    """Torque enclosed by each zone boundary.

    Returns ``(psi_norm, torque)``: the normalized poloidal flux of the zone
    boundaries and the cumulative torque inside each, in newton metres.

    The result is an ``XB`` quantity even though its ingredients are on ``X``.
    Summing zones one to ``i`` gives the torque inside the *outer* boundary of
    zone ``i``, which is ``XB[i]`` -- the grids interleave, and this is the one
    place that relationship is load-bearing rather than incidental.
    """
    density = np.asarray(input_torque_density(slice), dtype=float)
    volume = np.asarray(zone_volume(slice), dtype=float)
    if density.shape != volume.shape:
        raise ValueError(
            f"TQIN is {density.shape} and DVOL is {volume.shape}; both are zone-centre "
            "quantities and must agree"
        )
    return slice.psi_norm_xb, np.cumsum(density * volume)
