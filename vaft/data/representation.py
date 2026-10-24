"""What a value means, and whether two such meanings may be combined.

A :class:`Representation` declares, for one value, the physical quantity it
is (a canonical name or strict alias from :mod:`vaft.plot.taxonomy`), its unit
tag, and the COCOS index and toroidal Fourier convention it follows when those
apply, together with how well that declaration is supported.
:func:`assess_compatibility` compares two representations and says whether the
values are directly compatible, need an explicit transformation, mean
different things, or cannot be decided from what is declared (#2163).

**An assessment is not a validation verdict.** ``COMPATIBLE`` says the
declarations agree; it does not say the data conforms to them. Whether a
declared COCOS is what the arrays actually hold is
:func:`vaft.validation.cocos.validate_cocos`; whether a stored harmonic belongs
to the declared kernel is
:func:`vaft.validation.perturbation_phase.phase_convention_verdict`.  The
outcomes here never become ``PASS`` or ``FAIL``.

**An assessment never transforms.** It names the provider that would perform
a required transformation (:class:`TransformRef`); executing it is the
caller's separate, explicit call.  No array is read, so the cost does not
depend on data size, and no input is modified.

**Identity comes first.** Matching units or dimensions never make two
quantities the same: a line-averaged and a separatrix density in the same unit
are ``INCOMPATIBLE``, and so are two members of one quantity family.  A unit or
convention difference matters only once identity matches.

Evidence levels (:class:`EvidenceLevel`) record how the conventions of a
representation are known:

``DECLARED``
    stated by a contract, a file header or a code registry, and not checked;
``INFERRED``
    supported by the data's own consistency (sign identification, the
    Ampere-law flux test), which is not independent evidence;
``VERIFIED``
    confirmed by a named check against evidence the value was not built from;
``UNRESOLVED``
    missing or contradicted.

A representation is never promoted from one level to another here.

Identity resolves against :mod:`vaft.plot.taxonomy`, so the first assessment
loads the plotting package once.  Nothing here runs on a numerical path, and no
formula, process or solver module is imported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from vaft.data.fourier import FourierConvention

__all__ = [
    "EvidenceLevel",
    "Compatibility",
    "Representation",
    "TransformRef",
    "CompatibilityAssessment",
    "assess_compatibility",
    "canonical_quantity",
    "representation_from_convention",
]

#: Providers a :class:`TransformRef` names.  They are references to existing
#: functions, not wrappers: nothing here calls them.
UNIT_PROVIDER = "vaft.formula.units.convert_unit"
COCOS_PROVIDER = "vaft.process.equilibrium.convert_cocos"
COCOS_ODS_PROVIDER = "vaft.omas.general.equilibrium_psi_to_weber"
FOURIER_PROVIDER = "vaft.data.fourier.convert_harmonic"


class EvidenceLevel(str, Enum):
    """How the conventions a representation declares are known; ordered weakest first."""

    UNRESOLVED = "unresolved"
    DECLARED = "declared"
    INFERRED = "inferred"
    VERIFIED = "verified"


_EVIDENCE_ORDER = tuple(EvidenceLevel)


class Compatibility(str, Enum):
    """The outcome of :func:`assess_compatibility`; not a validation status."""

    COMPATIBLE = "compatible"
    REQUIRES_TRANSFORM = "requires_transform"
    INCOMPATIBLE = "incompatible"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Representation:
    """The declared meaning of one value: quantity, unit and conventions.

    ``quantity`` is a canonical quantity, a quantity subject or a strict alias
    of either; ``unit`` a unit tag as :func:`vaft.formula.units.parse_unit`
    reads it.  ``cocos`` and ``fourier`` are set only for values whose meaning
    depends on them.  ``None`` always means "not declared", never a default.
    """

    quantity: Optional[str]
    unit: Optional[str]
    cocos: Optional[int] = None
    fourier: Optional[FourierConvention] = None
    evidence: EvidenceLevel = EvidenceLevel.DECLARED
    note: str = ""

    def __post_init__(self):
        object.__setattr__(self, "evidence", EvidenceLevel(self.evidence))
        if self.cocos is not None:
            from vaft.data.cocos import COCOS_INDICES

            if self.cocos not in COCOS_INDICES:
                raise ValueError(f"COCOS index {self.cocos!r} is not defined")


@dataclass(frozen=True)
class TransformRef:
    """A transformation an assessment found necessary, by reference to its provider."""

    kind: str
    provider: str
    parameters: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "provider": self.provider, "parameters": dict(self.parameters)}


@dataclass(frozen=True)
class CompatibilityAssessment:
    """Result of :func:`assess_compatibility`, with every reason and required transformation."""

    status: Compatibility
    reasons: tuple[str, ...]
    source: Representation
    target: Representation
    transformations: tuple[TransformRef, ...] = ()
    unresolved: tuple[str, ...] = ()
    evidence: EvidenceLevel = EvidenceLevel.DECLARED
    """The weaker of the two representations' evidence levels.  ``COMPATIBLE``
    with ``DECLARED`` evidence is structural agreement, nothing more."""

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "reasons": list(self.reasons),
            "transformations": [t.as_dict() for t in self.transformations],
            "unresolved": list(self.unresolved),
            "evidence": self.evidence.value,
        }


def canonical_quantity(term: Optional[str]) -> Optional[str]:
    """The canonical name of ``term`` in :mod:`vaft.plot.taxonomy`, or ``None`` if it names none.

    A quantity subject (``plasma_current``, ``ne``) resolves to its subject,
    a canonical quantity or strict alias (``beta_normal``) to its quantity.
    Families, non-quantity subjects and unknown terms return ``None``; nothing
    is folded or guessed.
    """
    if not term:
        return None
    from vaft.plot import taxonomy

    try:
        subject = taxonomy.resolve_subject(term)
    except KeyError:
        subject = None
    if subject is not None:
        return subject.name if subject.kind == "quantity" else None
    try:
        return taxonomy.resolve_quantity(term)
    except KeyError:
        return None


def _family_of(name: str) -> Optional[str]:
    from vaft.plot import taxonomy

    for family in taxonomy.FAMILIES.values():
        if name in family.members:
            return family.name
    return None


def _weaker(a: EvidenceLevel, b: EvidenceLevel) -> EvidenceLevel:
    return min(a, b, key=_EVIDENCE_ORDER.index)


def assess_compatibility(source: Representation, target: Representation) -> CompatibilityAssessment:
    """Whether a value declared as ``source`` may be used where ``target`` is expected.

    Checks, in order: quantity identity, unit, COCOS, Fourier convention.  A
    different quantity, or units of different dimension, is ``INCOMPATIBLE`` at
    once: either alone proves the values mean different things, whatever else
    is undeclared.  Otherwise a missing or contradicted declaration on either
    side makes the result ``UNRESOLVED``;
    otherwise any difference with a known provider makes it
    ``REQUIRES_TRANSFORM``; otherwise it is ``COMPATIBLE``.  Reasons, required
    transformations and unresolved items are all reported, whatever the status.
    """
    from vaft.formula.units import IncompatibleUnitsError, UnknownUnitError, parse_unit

    reasons: list[str] = []
    unresolved: list[str] = []
    transforms: list[TransformRef] = []
    evidence = _weaker(source.evidence, target.evidence)

    def result(status: Compatibility) -> CompatibilityAssessment:
        return CompatibilityAssessment(status, tuple(reasons), source, target,
                                       tuple(transforms), tuple(unresolved), evidence)

    # --- identity --------------------------------------------------------
    a, b = canonical_quantity(source.quantity), canonical_quantity(target.quantity)
    for side, term, name in (("source", source.quantity, a), ("target", target.quantity, b)):
        if term is None:
            unresolved.append(f"{side} quantity is not declared")
        elif name is None:
            unresolved.append(f"{side} quantity {term!r} is not a quantity of the canonical vocabulary")
    if a is not None and b is not None and a != b:
        family_a, family_b = _family_of(a), _family_of(b)
        if family_a is not None and family_a == family_b:
            reasons.append(f"{a!r} and {b!r} are distinct members of the {family_a!r} family, not one quantity")
        else:
            reasons.append(f"{a!r} and {b!r} are different quantities")
        return result(Compatibility.INCOMPATIBLE)
    if a is not None and a == b:
        reasons.append(f"same quantity {a!r}")

    # --- unit --------------------------------------------------------------
    if source.unit is None or target.unit is None:
        unresolved.append("unit is not declared" if source.unit == target.unit
                          else f"{'source' if source.unit is None else 'target'} unit is not declared")
    else:
        try:
            ua, ub = parse_unit(source.unit), parse_unit(target.unit)
        except UnknownUnitError as error:
            unresolved.append(str(error))
        else:
            if ua.dimensions != ub.dimensions:
                reasons.append(f"units {source.unit!r} and {target.unit!r} differ in dimension")
                return result(Compatibility.INCOMPATIBLE)
            if ua.angle != ub.angle:
                # Only exactly Wb against Wb/rad is a storage-family change, and only a
                # COCOS transform between two declared, different indices carries it.
                flux = {source.unit.strip(), target.unit.strip()} == {"Wb", "Wb/rad"}
                if not flux:
                    unresolved.append(f"{source.unit!r} and {target.unit!r} differ in plane-angle exponent; "
                                      "only Wb against Wb/rad is a convention difference")
                elif source.cocos is None or target.cocos is None:
                    unresolved.append(
                        f"{source.unit!r} against {target.unit!r} is a convention (COCOS) difference, "
                        "and the COCOS of both sides is needed to decide it")
                elif source.cocos == target.cocos:
                    unresolved.append(f"{source.unit!r} against {target.unit!r} under one COCOS "
                                      f"({source.cocos}) is a contradiction")
                else:
                    reasons.append(f"{source.unit!r} against {target.unit!r} is carried by the COCOS transform")
            elif ua.scale != ub.scale:
                try:
                    from vaft.formula.units import unit_conversion_factor

                    factor = unit_conversion_factor(source.unit, target.unit)
                except IncompatibleUnitsError as error:  # pragma: no cover - guarded above
                    unresolved.append(str(error))
                else:
                    reasons.append(f"unit {source.unit!r} -> {target.unit!r} scales by {factor:.12g}")
                    transforms.append(TransformRef("unit", UNIT_PROVIDER,
                                                   {"source": source.unit, "target": target.unit,
                                                    "factor": factor}))
            else:
                reasons.append(f"units {source.unit!r} and {target.unit!r} agree")

    # --- COCOS ---------------------------------------------------------------
    if source.cocos is not None or target.cocos is not None:
        if source.cocos is None or target.cocos is None:
            unresolved.append(f"{'source' if source.cocos is None else 'target'} COCOS is not declared")
        else:
            from vaft.data.cocos import psi_is_per_radian

            flux = parse_unit("Wb")
            for side, rep in (("source", source), ("target", target)):
                if rep.unit is None:
                    continue
                try:
                    spec = parse_unit(rep.unit)
                except UnknownUnitError:
                    continue
                per_radian = psi_is_per_radian(rep.cocos)
                if spec.angle and not per_radian:
                    unresolved.append(f"{side} unit {rep.unit!r} is per radian, but COCOS {rep.cocos} "
                                      "stores the full flux")
                    continue
                if spec.dimensions != flux.dimensions:
                    continue
                if (spec.angle == -1) != per_radian:
                    stored = "per radian" if per_radian else "as the full flux"
                    unresolved.append(f"{side} unit {rep.unit!r} contradicts COCOS {rep.cocos}, "
                                      f"which stores psi {stored}")
            if source.cocos != target.cocos:
                reasons.append(f"COCOS {source.cocos} -> {target.cocos}")
                transforms.append(TransformRef("cocos", COCOS_PROVIDER,
                                               {"source_cocos": source.cocos, "target_cocos": target.cocos}))
            else:
                reasons.append(f"same COCOS {source.cocos}")

    # --- Fourier -------------------------------------------------------------
    if source.fourier is not None or target.fourier is not None:
        if source.fourier is None or target.fourier is None:
            unresolved.append(f"{'source' if source.fourier is None else 'target'} Fourier convention is not declared")
        else:
            from vaft.data.fourier import UnresolvedConventionError, fourier_conversion

            try:
                conjugate, factor = fourier_conversion(source.fourier, target.fourier)
            except UnresolvedConventionError as error:
                unresolved.append(str(error))
            else:
                if conjugate or factor != 1.0:
                    reasons.append(f"Fourier {source.fourier.name} -> {target.fourier.name}: "
                                   f"{'conjugate, ' if conjugate else ''}scale {factor:g}")
                    transforms.append(TransformRef("fourier", FOURIER_PROVIDER,
                                                   {"source": source.fourier.name, "target": target.fourier.name,
                                                    "conjugate": conjugate, "factor": factor}))
                else:
                    reasons.append("Fourier conventions agree")

    # --- evidence ---------------------------------------------------------------
    for side, rep in (("source", source), ("target", target)):
        if rep.evidence is EvidenceLevel.UNRESOLVED:
            unresolved.append(f"{side} conventions are unresolved{': ' + rep.note if rep.note else ''}")

    if unresolved:
        return result(Compatibility.UNRESOLVED)
    if transforms:
        return result(Compatibility.REQUIRES_TRANSFORM)
    return result(Compatibility.COMPATIBLE)


def representation_from_convention(convention, *, quantity: str = "psi", unit: Optional[str] = None) -> Representation:
    """The :class:`Representation` an :class:`~vaft.data.equilibrium.EquilibriumConvention` supports.

    A contradicted or ambiguous convention is ``UNRESOLVED`` and carries no
    COCOS index, and so is one whose ``psi_per_radian`` disagrees with its
    index's storage family.  An index the data's own identification supports
    -- a declaration it confirms, or the single remaining candidate -- is
    ``INFERRED``: self-consistency, not independent evidence.  A declaration
    with no identification behind it is ``DECLARED``.  Nothing reaches
    ``VERIFIED`` from a convention record alone.  ``unit`` defaults to the flux
    unit of the storage family (``Wb`` or ``Wb/rad``).
    """
    if convention.contradicted or convention.ambiguous:
        why = "contradicted by identification" if convention.contradicted else "ambiguous"
        return Representation(quantity, unit, None, None, EvidenceLevel.UNRESOLVED,
                              note=f"COCOS {why} (source: {convention.source})")
    from vaft.data.cocos import psi_is_per_radian

    declared = convention.cocos is not None
    index = convention.cocos if declared else convention.candidates[0]
    family = psi_is_per_radian(index)
    if convention.psi_per_radian is not None and convention.psi_per_radian != family:
        return Representation(quantity, unit, None, None, EvidenceLevel.UNRESOLVED,
                              note=f"psi_per_radian={convention.psi_per_radian} contradicts COCOS {index} "
                                   f"(source: {convention.source})")
    if unit is None:
        unit = "Wb/rad" if family else "Wb"
    supported = index in convention.identified or not declared
    level = EvidenceLevel.INFERRED if supported else EvidenceLevel.DECLARED
    return Representation(quantity, unit, index, None, level, note=f"source: {convention.source}")
