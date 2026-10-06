"""Interpretation metadata of an operational-space projection (#1624).

Axis identities say *what* is plotted; they do not say what the picture is
for. A $\\nu_*$-$\\rho_*$ scatter is a global kinetic-similarity view used for
reactor extrapolation, not a local neoclassical-regime diagram, and a
pedestal-collisionality map is an empirical regime-access map, not a stability
boundary. :class:`ProjectionInterpretation` keeps that meaning next to the
projection, so documentation and UI code can state it instead of leaving the
reader to infer it from axis labels.

Every axis carries an :class:`AxisConvention`: the expression as the source
writes it, the species, radial and averaging definitions, the VAFT function
that evaluates it, and what the source leaves unstated. Two collisionalities
with different conventions are different axis quantities, so they can never be
drawn on one axis.

Nothing here computes physics or reads data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Tuple

from vaft.formula.boundaries import BoundarySource

__all__ = [
    "CATEGORIES",
    "QUANTITY_SCOPES",
    "BOUNDARY_TYPES",
    "AxisConvention",
    "ProjectionInterpretation",
]

#: What kind of space a projection is. ``stability_limit`` is the conventional
#: limit diagram (Hugill, Troyon); the other three are the #1624 taxonomy.
CATEGORIES = (
    "stability_limit",
    "dimensionless_similarity",
    "transport_regime",
    "edge_regime",
    "separatrix_operational_space",
)
QUANTITY_SCOPES = ("global", "local", "pedestal", "edge", "separatrix")
#: ``none``: the space has no boundary of its own (a similarity view);
#: ``regime_access``: an empirical map of where regimes were observed, not a limit.
BOUNDARY_TYPES = ("none", "theoretical", "empirical", "reduced_model", "derived", "regime_access")


def _tuple(value) -> tuple:
    return tuple(value or ())


@dataclass(frozen=True)
class AxisConvention:
    """How one axis quantity is defined, as its source defines it.

    ``quantity`` is the axis quantity's ``name``. ``expression`` is the
    source's formula, written as the source writes it. ``inputs`` maps each
    symbol of the expression to its definition and unit. ``formula`` names the
    VAFT function that evaluates the expression (empty when none does).
    ``unresolved`` lists what the source leaves unstated; it is reported, not
    filled in.
    """

    quantity: str
    expression: str
    species: str
    radial_definition: str
    averaging_definition: str
    source: BoundarySource
    inputs: Mapping[str, str] = field(default_factory=dict, hash=False)
    formula: str = ""
    unresolved: Tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs or {})))
        object.__setattr__(self, "unresolved", _tuple(self.unresolved))
        for name in ("quantity", "expression", "species", "radial_definition", "averaging_definition"):
            if not getattr(self, name):
                raise ValueError(f"AxisConvention needs a non-empty {name!r}")


@dataclass(frozen=True)
class ProjectionInterpretation:
    """What a projection is for, and the conventions it must not mix.

    ``physical_question`` is the question the picture answers;
    ``interpretation`` how to read a position on it; ``not_for`` the readings
    it does not support. ``similarity_role`` says which extrapolation each axis
    represents. ``parameter_conventions`` holds one :class:`AxisConvention`
    per plotted quantity. ``required_profiles`` names the kinetic or profile
    inputs a state must have; a state without them is unassessed, never
    approximated.
    """

    category: str
    physical_question: str
    interpretation: str
    quantity_scope: str
    similarity_role: str
    boundary_type: str
    applicability: str
    species: str
    radial_definition: str
    averaging_definition: str
    parameter_conventions: Tuple[AxisConvention, ...]
    references: Tuple[BoundarySource, ...]
    required_profiles: Tuple[str, ...] = ()
    not_for: Tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("parameter_conventions", "references", "required_profiles", "not_for"):
            object.__setattr__(self, name, _tuple(getattr(self, name)))
        if self.category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}, not {self.category!r}")
        if self.quantity_scope not in QUANTITY_SCOPES:
            raise ValueError(f"quantity_scope must be one of {QUANTITY_SCOPES}, not {self.quantity_scope!r}")
        if self.boundary_type not in BOUNDARY_TYPES:
            raise ValueError(f"boundary_type must be one of {BOUNDARY_TYPES}, not {self.boundary_type!r}")
        for name in ("physical_question", "interpretation", "similarity_role", "applicability", "species",
                     "radial_definition", "averaging_definition"):
            if not getattr(self, name):
                raise ValueError(f"ProjectionInterpretation needs a non-empty {name!r}")
        if not self.references:
            raise ValueError("ProjectionInterpretation needs at least one reference")
        names = [c.quantity for c in self.parameter_conventions]
        if len(set(names)) != len(names):
            raise ValueError(f"one convention per quantity; got {names}")

    def convention(self, quantity: str) -> AxisConvention:
        """The convention of one plotted quantity, by name."""
        for entry in self.parameter_conventions:
            if entry.quantity == quantity:
                return entry
        raise KeyError(f"no convention for {quantity!r}; have {[c.quantity for c in self.parameter_conventions]}")

    @property
    def unresolved(self) -> Tuple[Tuple[str, str], ...]:
        """``(quantity, item)`` for everything a source leaves unstated."""
        return tuple((c.quantity, item) for c in self.parameter_conventions for item in c.unresolved)

    def describe(self) -> str:
        """A plain-text account for documentation and UI: question, reading, scope, conventions, gaps."""
        lines = [
            f"Question: {self.physical_question}",
            f"Reading: {self.interpretation}",
            f"Category: {self.category}; scope: {self.quantity_scope}; boundary type: {self.boundary_type}",
            f"Similarity role: {self.similarity_role}",
            f"Applicability: {self.applicability}",
            f"Species: {self.species}",
            f"Radial definition: {self.radial_definition}",
            f"Averaging: {self.averaging_definition}",
        ]
        if self.required_profiles:
            lines.append("Requires: " + "; ".join(self.required_profiles))
        for c in self.parameter_conventions:
            where = f" ({c.source.citation}" + (f", {c.source.equation}" if c.source.equation else "") + ")"
            lines.append(f"{c.quantity} = {c.expression}{where}")
            for symbol, meaning in c.inputs.items():
                lines.append(f"    {symbol}: {meaning}")
            if c.formula:
                lines.append(f"    evaluated by {c.formula}")
        for item in self.not_for:
            lines.append(f"Not for: {item}")
        for quantity, item in self.unresolved:
            lines.append(f"Unresolved ({quantity}): {item}")
        return "\n".join(lines)
