"""Explicit COCOS coordinate-convention management.

COCOS (Sauter & Medvedev, *Tokamak Equilibrium Coordinate Conventions*, 2013)
identifies, with a single index, the choices a code makes about the toroidal and
poloidal coordinate orientations, the sign of the poloidal flux, and whether that
flux is divided by 2*pi.  Sixteen combinations are possible, and equilibrium
quantities are only comparable between codes once the index is known.

This module is the single source of truth for that model in VAFT.  It holds two
things and no algorithms:

* :class:`CocosSpec` -- the sign/exponent parameters of one index, and the
  ``bp_factor`` coefficient that Sauter Eq. 20 needs to turn ``psi`` into a
  poloidal field.  The parameters themselves come from
  :func:`omas.omas_physics.define_cocos`, so VAFT and OMAS/IMAS cannot drift.
* :class:`CodeConvention` and the registry -- what convention each external code
  and file format expects, so every adapter *declares* its convention instead of
  assuming one.

The operational side (identification, validation, conversion) lives in
:mod:`vaft.process.cocos`; the field expressions live in
:mod:`vaft.formula.equilibrium`.

VAFT works internally in :data:`VAFT_INTERNAL_COCOS` (11), which is what the IMAS
Data Dictionary mandates.  Adapters normalize to it on import and convert away
from it on export.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

__all__ = [
    "COCOS_INDICES",
    "CodeConvention",
    "CocosSpec",
    "VAFT_INTERNAL_COCOS",
    "cocos_spec",
    "convention_for",
    "known_codes",
    "register_convention",
]

#: The convention VAFT normalizes to internally, mandated by the IMAS Data
#: Dictionary ("This version of the IMAS Data Dictionary corresponds to
#: COCOS = 11 coordinate convention").  psi is in weber, not weber/radian.
VAFT_INTERNAL_COCOS = 11

#: The sixteen defined indices.  9 and 10 do not exist: 1-8 carry ``e_Bp = 0``
#: (psi already divided by 2*pi) and 11-18 are the same eight conventions with
#: ``e_Bp = 1`` (full poloidal flux).
COCOS_INDICES = tuple(range(1, 9)) + tuple(range(11, 19))

_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class CocosSpec:
    """The sign and exponent parameters of one COCOS index.

    ``sigma_bp`` and ``sigma_rpz`` fix the orientation of the poloidal flux and
    of the cylindrical system; ``sigma_rhotp`` fixes the poloidal angle and hence
    the sign of ``q``; ``exp_bp`` is Sauter's ``e_Bp``, 0 when psi is stored per
    radian and 1 when it is the full flux in weber.
    """

    index: int
    sigma_bp: int
    sigma_rpz: int
    sigma_rhotp: int
    exp_bp: int
    sign_q_pos: int
    sign_pprime_pos: int

    @property
    def psi_per_radian(self) -> bool:
        """True when psi is in weber/radian (COCOS 1-8)."""
        return self.exp_bp == 0

    @property
    def bp_factor(self) -> float:
        """The Sauter Eq. 20 coefficient ``sigma_RphiZ * sigma_Bp / (2*pi)**e_Bp``.

        ``B_R = bp_factor * (1/R) dpsi/dZ`` and ``B_Z = -bp_factor * (1/R) dpsi/dR``.
        Note this carries *both* the 2*pi normalization and the orientation sign:
        applying only the former leaves the field inverted for half the indices.
        """
        return self.sigma_rpz * self.sigma_bp / _TWO_PI**self.exp_bp

    def expected_sign(self, quantity: str, *, sigma_ip: int, sigma_b0: int) -> int:
        """Sign a quantity must carry under this convention, from Sauter Eq. 23.

        ``sigma_ip`` and ``sigma_b0`` are the signs of the plasma current and the
        vacuum toroidal field *in this convention's own coordinate system*.
        """
        relations = {
            "f": sigma_b0,
            "phi_tor": sigma_b0,
            "dpsi": sigma_ip * self.sigma_bp,
            "pprime": -sigma_ip * self.sigma_bp,
            "j_phi": sigma_ip,
            "q": sigma_ip * sigma_b0 * self.sigma_rhotp,
        }
        try:
            return relations[quantity]
        except KeyError:
            raise ValueError(
                f"unknown quantity {quantity!r}; expected one of {sorted(relations)}"
            ) from None


def cocos_spec(index: int) -> CocosSpec:
    """Return the :class:`CocosSpec` for ``index``.

    The parameters are read from :func:`omas.omas_physics.define_cocos` rather
    than re-tabulated here, so VAFT stays aligned with the OMAS/IMAS model by
    construction.
    """
    if index not in COCOS_INDICES:
        raise ValueError(
            f"COCOS index {index!r} is not defined; expected one of {COCOS_INDICES}"
        )
    cached = _SPEC_CACHE.get(index)
    if cached is not None:
        return cached
    from omas import define_cocos

    raw = define_cocos(index)
    spec = CocosSpec(
        index=index,
        sigma_bp=int(raw["sigma_Bp"]),
        sigma_rpz=int(raw["sigma_RpZ"]),
        sigma_rhotp=int(raw["sigma_rhotp"]),
        exp_bp=int(raw["exp_Bp"]),
        sign_q_pos=int(raw["sign_q_pos"]),
        sign_pprime_pos=int(raw["sign_pprime_pos"]),
    )
    _SPEC_CACHE[index] = spec
    return spec


_SPEC_CACHE: dict[int, CocosSpec] = {}


@dataclass(frozen=True)
class CodeConvention:
    """The COCOS convention one external code or file format works in.

    ``cocos`` is ``None`` when the convention is not fixed by the format and has
    to be identified per file -- GEQDSK is the important case, since a g-file
    carries no convention field and different EFIT builds emit different indices.

    ``confirmed`` is False when the index is VAFT's best inference rather than a
    documented fact; those entries should be treated as assumptions and reported.
    """

    name: str
    cocos: int | None
    psi_unit: str
    reference: str
    notes: str = ""
    confirmed: bool = True

    @property
    def identifies_per_file(self) -> bool:
        return self.cocos is None


_REGISTRY: dict[str, CodeConvention] = {}


def register_convention(convention: CodeConvention) -> CodeConvention:
    """Add ``convention`` to the registry, refusing to silently replace an entry."""
    existing = _REGISTRY.get(convention.name)
    if existing is not None and existing != convention:
        raise ValueError(
            f"convention {convention.name!r} is already registered as "
            f"COCOS {existing.cocos} from {existing.reference}"
        )
    if convention.cocos is not None and convention.cocos not in COCOS_INDICES:
        raise ValueError(
            f"COCOS index {convention.cocos!r} is not defined; "
            f"expected one of {COCOS_INDICES} or None to identify per file"
        )
    _REGISTRY[convention.name] = convention
    return convention


def convention_for(code: str) -> CodeConvention:
    """Return the declared convention for ``code``."""
    try:
        return _REGISTRY[code]
    except KeyError:
        raise KeyError(
            f"no COCOS convention is declared for {code!r}; "
            f"declared codes are: {', '.join(known_codes())}"
        ) from None


def known_codes() -> tuple[str, ...]:
    """Every code with a declared convention, in sorted order."""
    return tuple(sorted(_REGISTRY))


# --- Declared conventions -------------------------------------------------
#
# Sources: Sauter & Medvedev 2013 Sect. IX and Appendix A; the code lists in
# `omas.omas_physics.define_cocos`; and the IMAS Data Dictionary COCOS page.

register_convention(CodeConvention(
    name="imas",
    cocos=11,
    psi_unit="Wb",
    reference="IMAS Data Dictionary, cocos.html",
    notes="Normative for the IMAS DD and therefore for ODS and IDS interchange.",
))

register_convention(CodeConvention(
    name="omas",
    cocos=11,
    psi_unit="Wb",
    reference="IMAS Data Dictionary, cocos.html",
    notes="OMAS ODS follow the IMAS DD convention.",
))

register_convention(CodeConvention(
    name="chease",
    cocos=2,
    psi_unit="Wb/rad",
    reference="Sauter & Medvedev 2013 Sect. IX; omas define_cocos",
    notes=(
        "CHEASE works in normalized units with Ip and B0 positive; Sauter Eq. 22 "
        "requires psi minimum on axis, dp/dpsi negative and q positive in its input."
    ),
))

register_convention(CodeConvention(
    name="geqdsk",
    cocos=None,
    psi_unit="Wb/rad",
    reference="Sauter & Medvedev 2013 Sect. V",
    notes=(
        "A g-file carries no convention field. EFIT builds are commonly COCOS 3, "
        "but the index must be identified per file from the observable signs; the "
        "packaged VEST sample identifies as 1 or 2 depending on clockwise_phi."
    ),
))

register_convention(CodeConvention(
    name="efit",
    cocos=None,
    psi_unit="Wb/rad",
    reference="Sauter & Medvedev 2013 Sect. V",
    notes="Reads and writes GEQDSK; identified per file like any other g-file.",
))

register_convention(CodeConvention(
    name="tes",
    cocos=None,
    psi_unit="Wb/rad",
    reference="vaft/code/tes/outputs.py",
    notes="Emits a g-file, so it inherits the GEQDSK identification path.",
))

register_convention(CodeConvention(
    name="gpec",
    cocos=None,
    psi_unit="Wb/rad",
    reference="vaft/data/gpec/equil.in (eq_type='efit')",
    notes=(
        "DCON/GPEC read a g-file and are told its family through eq_type; the "
        "index follows whatever the supplied g-file identifies as."
    ),
))

register_convention(CodeConvention(
    name="vfit",
    cocos=1,
    psi_unit="Wb/rad",
    reference="implied by vaft/data/vfit.py, which multiplies psi by +2*pi",
    confirmed=False,
    notes=(
        "The importer converts psi to COCOS 11 by multiplying by 2*pi with no "
        "sign change.  That factor is cocos_transform(1, 11)['PSI'] specifically "
        "-- the 2 -> 11 factor is -2*pi -- so the code asserts VFIT is COCOS 1, "
        "not merely that psi is in Wb/rad.  The sign parameters have never been "
        "checked against Sauter Eq. 23, so the index remains an assumption."
    ),
))
