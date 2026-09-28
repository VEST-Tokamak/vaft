"""PENTRC output adapter -- read-only.

PENTRC computes the neoclassical toroidal viscous (NTV) torque from a perturbed
field and a kinetic profile.  It is run elsewhere; VAFT reads what it produced.
There is no ``inputs.py`` and no runner, for the same reason
:mod:`vaft.code.transp` has none.

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

    from vaft.code import pentrc

    with pentrc.read_pentrc_output("pentrc_output_n1.nc") as run:
        psi_norm, torque = pentrc.torque_profile(run, "fgar")
        total = torque[-1]        # equals the file's own T_total_fgar
"""

from .outputs import (
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
