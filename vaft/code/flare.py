"""FLARE interoperability: the equilibrium handshake.

FLARE traces field lines through a 2-D equilibrium read from a GEQDSK plus a
3-D perturbation, and the two arrive in different conventions. This module is
the part that reconciles them.

**FLARE declares no COCOS.** The word appears nowhere in its source. What it
does instead is take two manual multipliers from its control file --
``scale_Ip`` applied to ``Simag``, ``Sibry`` and ``psirz``, and ``scale_Bt``
applied to ``Bcentr`` and ``fpol`` (``src/fortran/bfield/equi2d.f90:852``) --
and then derive the field directions from whatever comes out
(``:307-309``). Its index is therefore read off its behaviour rather than a
declaration, and :data:`~vaft.data.cocos` registers it as 3 with that
reasoning recorded.

**The multipliers are a convention conversion, and the legacy guessed them.**
``run_flare.py::_resolve_gpec_equilibrium_convention`` inferred ``scale_Ip``
from a g-file's metadata and fell back to ``1.0`` when the parse failed --
silently, so an unreadable header became "no conversion needed". The
difference is not cosmetic: measured on the DIII-D ideal GPEC example, the
same raw ``n = 1`` BRZPHI field traced through the converted and unconverted
equilibrium gives different island topology in the total, plasma and coil
fields alike. This module derives the pair from an identified COCOS and
refuses to proceed when the identification is ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vaft.data.cocos import convention_for
from vaft.process.cocos import (
    cocos_field_scales,
    identify_convention,
    identify_flux_exponent,
)

__all__ = ["FlareEquilibriumScales", "flare_equilibrium_scales"]


@dataclass(frozen=True)
class FlareEquilibriumScales:
    """The two multipliers FLARE's control file needs, and where they came from."""

    scale_ip: int
    scale_bt: int
    source_cocos: int
    target_cocos: int
    psi_per_radian: bool


def flare_equilibrium_scales(
    equilibrium: Any, *, clockwise_phi: bool | None = None, source_cocos: int | None = None
) -> FlareEquilibriumScales:
    """Derive ``scale_Ip`` and ``scale_Bt`` for a GEQDSK FLARE is to trace.

    Parameters
    ----------
    equilibrium : EquilibriumData
        The equilibrium as read, in whatever convention its file used [n/a].
    clockwise_phi : bool, optional
        Whether the machine's toroidal angle runs clockwise seen from above. A
        fact about the machine, not about the file; without it a g-file
        identifies as a pair of indices and this refuses rather than picking
        one [n/a].
    source_cocos : int, optional
        The source index, when it is known from provenance rather than from
        the data. Supplying it skips the identification entirely [-].

    Returns
    -------
    FlareEquilibriumScales
        ``scale_ip`` multiplies the poloidal flux, ``scale_bt`` the toroidal
        field and ``F``; both are ``+1`` or ``-1`` [-].

    Raises
    ------
    ValueError
        The identification returns no candidate, or more than one and no
        ``clockwise_phi`` to separate them, or the flux normalization cannot
        be determined.

    Convention
    ----------
    **Fail closed.** A g-file carries no convention field, and the signs alone
    leave the odd and even index of a pair indistinguishable -- they differ by
    the handedness of the machine's toroidal angle, which is not in the file.
    The legacy resolved that by falling back to no conversion; this refuses,
    because a wrong sign here changes the traced topology rather than shifting
    it, and a silent default is indistinguishable from a correct answer.

    Applicability
    -------------
    Machine-independent. ``clockwise_phi`` is the one machine fact, and it is
    the caller's to supply.

    Processing steps
    ----------------
    1. Identify the source index, unless the caller states it.
    2. Refuse an empty or still-ambiguous candidate set.
    3. When the index came from the data, check the flux normalization is
       determined too -- the 1-8 and 11-18 families differ by ``2 pi`` on
       ``psi``, which these multipliers do not carry. A stated index already
       names its family.
    4. Take the sign multipliers to FLARE's registered index.

    Limitations
    -----------
    Carries signs only. A source in the 11-18 family also needs its flux
    divided by ``2 pi`` before FLARE, and this reports
    :attr:`~FlareEquilibriumScales.psi_per_radian` rather than applying it,
    because FLARE's two multipliers have nowhere to put a magnitude.

    Provenance
    ----------
    .. [flare] ``FLARE/src/fortran/bfield/equi2d.f90:852`` for where the two
       multipliers are applied, and ``:307-309`` for the direction derivation
       that follows.
    .. [measured] The DIII-D ideal GPEC example's ``g147131.02300_DIIID_KEFIT``
       identifies as COCOS 5 in weber per radian and needs ``(-1, +1)``; a
       COCOS 2 CHEASE equilibrium needs ``(+1, -1)``.
    """
    target = convention_for("flare").cocos
    if target is None:  # pragma: no cover - the registry fixes it at 3
        raise ValueError("the flare convention carries no COCOS index")

    if source_cocos is None:
        candidates = identify_convention(equilibrium, clockwise_phi=clockwise_phi)
        if not candidates:
            raise ValueError(
                "the equilibrium's COCOS cannot be identified: bt0, ip, q and "
                "psi_1d are what the signs are read from, and one of them is "
                "missing or inconsistent. FLARE's scale_Ip and scale_Bt are a "
                "convention conversion, so there is no safe default to fall "
                "back to"
            )
        if len(candidates) > 1:
            # Two different ambiguities reach here and they need different
            # facts to resolve, so the message says which one this is rather
            # than naming the commoner one and being wrong half the time.
            families = {index <= 10 for index in candidates}
            handed = {index % 2 for index in candidates}
            missing = []
            if len(handed) > 1:
                missing.append(
                    "the handedness of the machine's toroidal angle (pass "
                    "clockwise_phi)"
                )
            if len(families) > 1:
                missing.append(
                    "whether psi is in weber or weber per radian, which needs "
                    "an LCFS to measure the Ampere ratio against"
                )
            raise ValueError(
                f"the equilibrium identifies as COCOS {list(candidates)}, and "
                f"separating them needs {' and '.join(missing) or 'more than the signs'}"
                ". Pass source_cocos when provenance settles it; picking one "
                "here would change the traced topology on a guess"
            )
        source = int(candidates[0])
    else:
        source = int(source_cocos)

    if source_cocos is None:
        # Only when the index came from the data. A stated index already names
        # its family -- 1-8 is weber per radian, 11-18 weber -- so demanding
        # the residual as well would refuse a file whose provenance settles it.
        exponent, _residual = identify_flux_exponent(equilibrium)
        if exponent is None:
            raise ValueError(
                "the flux normalization is undetermined, so whether psi is in "
                "weber or weber per radian is unknown. The two families differ "
                "by 2*pi on psi, which scale_Ip and scale_Bt cannot carry; pass "
                "source_cocos when provenance settles it"
            )

    scale_ip, scale_bt = cocos_field_scales(source, int(target))
    return FlareEquilibriumScales(
        scale_ip=scale_ip,
        scale_bt=scale_bt,
        source_cocos=source,
        target_cocos=int(target),
        psi_per_radian=source <= 10,
    )
