"""PENTRC output adapter -- compatibility namespace (#1883).

PENTRC computes the neoclassical toroidal viscous (NTV) torque from a perturbed
field and a kinetic profile.  It is built with the GPEC suite and runs in a
completed ideal-GPEC cell, so both its runner (:func:`vaft.code.gpec.run_pentrc`)
and its native output reader live in :mod:`vaft.code.gpec`.  This package keeps
the reader's earlier import path: every name here *is* the
:mod:`vaft.code.gpec` object, not a copy.  New code imports from
:mod:`vaft.code.gpec`.

    pentrc_output_n<n>.nc ── read_pentrc_output ─▶ PentrcOutput
                          ── torque_profile ─────▶ (psi_norm, torque [N m])
                          ── energy_profile ─────▶ (psi_norm, dW [J])

The pairing worth knowing about: :func:`vaft.code.transp.enclosed_torque`
gives the torque a beam injects, on its own radial grid, and this gives the NTV
torque.  Comparing them is what an NTV study is for, and neither layer does it
-- the grids differ and interpolating between them is a decision.  The **sign**
relation between the two is not established here either: PENTRC's ``real(T)``
and TRANSP's ``TQIN`` are written by different codes in their own conventions,
and nothing in either file states how they relate.

Typical use::

    from vaft.code import gpec

    with gpec.read_pentrc_output("pentrc_output_n1.nc") as run:
        psi_norm, torque = gpec.torque_profile(run, "fgar")
        total = torque[-1]        # equals the file's own T_total_fgar

Importing this package imports :mod:`vaft.code.gpec`, which owns the reader.
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

from . import outputs  # noqa: F401  (``vaft.code.pentrc.outputs`` stays an attribute)

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
