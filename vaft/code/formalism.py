"""The plasma-formalism provenance record of a run (issue #1727, under #1723).

Which physical formulation did *this run* use?  The solver name does not say:
DCON is ideal or kinetic-MHD depending on ``kin_flag``, CGYRO is electrostatic or
electromagnetic, linear or nonlinear, and ASCOT5 follows full orbits or guiding
centres.  :class:`PlasmaFormalism` records the answer as a few factorized,
controlled fields, derived by each backend from its actual configuration.

The vocabulary is the minimum the Phase A audit justified
(``docs/_guide/Plasma_models.md``, #1725).  Its design rules:

* **Factorized, not a ladder.**  Ideal MHD, drift kinetics and gyrokinetics are
  not rungs of one fidelity scale; they differ along independent axes
  (:data:`AXES`), and nothing here ranks them.
* **Describe the physics, never the inputs.**  There is no field for "kinetic
  profiles were used": a fluid calculation whose resistivity comes from Thomson
  ``T_e`` is still ``fluid`` with no kinetic equation (Phase A §6).
* **Not applicable is not unknown.**  ``None`` means the field has no meaning for
  this formulation (a distribution formulation for ideal MHD); :data:`UNKNOWN`
  means it has one that nobody has documented (a legacy mode).
* **Only generic impossibilities are rejected** -- a fluid run with a kinetic
  equation, a hybrid without a coupling.  Solver-specific detail (collision
  operators, saturation rules, geometry models, integer encodings) goes in
  ``extensions`` untouched rather than being forced into a common enum.
* **Confidence is not physics.**  How sure the classification is lives in the
  Phase A page, never in a value such as ``probably_drift_kinetic``.

The record is immutable and serializes to a stable, versioned dict
(:meth:`PlasmaFormalism.as_dict` / :meth:`PlasmaFormalism.from_dict`).  It is not
a model class, carries no behaviour, and makes no two solvers interchangeable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "AXES",
    "BULK_DESCRIPTIONS",
    "DISTRIBUTION_FORMULATIONS",
    "FIELD_MODELS",
    "FLUID_MODELS",
    "KINETIC_COUPLINGS",
    "KINETIC_EQUATIONS",
    "ORBIT_REPRESENTATIONS",
    "PlasmaFormalism",
    "REGIMES",
    "SCHEMA_VERSION",
    "SCIENTIFIC_OPERATIONS",
    "SPATIAL_DOMAINS",
    "TOPOLOGY_DOMAINS",
    "UNKNOWN",
]

#: Bumped only when a field or a value's meaning changes, never for an added value.
SCHEMA_VERSION = 1

#: A field that has a meaning for this formulation, but nobody has documented it.
UNKNOWN = "unknown"

#: What is being computed.
SCIENTIFIC_OPERATIONS = (
    "equilibrium",
    "ideal_stability",
    "resistive_stability",
    "perturbed_equilibrium",
    "toroidal_torque",
    "neoclassical_transport",
    "microstability",
    "turbulent_transport",
    "orbit_following",
    "fast_ion_sources",
)

#: What represents the bulk plasma.  ``particle`` is a test population in a fixed
#: background; ``reduced`` is a model derived from a theory without solving it
#: (quasilinear, analytic fit, learned surrogate).
BULK_DESCRIPTIONS = ("fluid", "hybrid", "kinetic", "particle", "reduced")

FLUID_MODELS = ("ideal_mhd", "resistive_mhd", "extended_mhd", "reduced_mhd")

#: ``fokker_planck`` is the test-species Fokker-Planck equation that Monte Carlo
#: orbit codes (ASCOT5, NUBEAM) solve; an orbit integrator alone solves none.
KINETIC_EQUATIONS = ("drift_kinetic", "gyrokinetic", "fokker_planck")

#: How a kinetic result re-enters a fluid model.  ``energy``: a delta W_k term in the
#: energy principle (DCON kinetic); ``closure``: a kinetic quantity evaluated on a
#: given fluid displacement, not fed back (PENTRC on a GPEC xi); ``sources``: heating,
#: current, torque returned to a transport code (NUBEAM).
KINETIC_COUPLINGS = ("energy", "pressure", "current", "closure", "sources")

DISTRIBUTION_FORMULATIONS = ("delta_f", "full_f")

#: Separate from the kinetic equation (Phase A §1): following orbits and evolving a
#: distribution are different statements.
ORBIT_REPRESENTATIONS = ("full_orbit", "guiding_center", "gyrocenter", "bounce_averaged")

SPATIAL_DOMAINS = ("local", "radially_global", "whole_volume")

TOPOLOGY_DOMAINS = ("closed_flux_surface", "open_field_line", "across_separatrix")

#: ``static`` is a marginal or perturbed-equilibrium calculation with no time
#: dependence (DCON, GPEC); ``steady_state`` a time-independent kinetic solution (NEO).
REGIMES = ("static", "linear", "nonlinear", "quasilinear", "steady_state", "time_dependent")

#: Coarse, solver-neutral.  CGYRO's ``em-aperp`` / ``em-aperp-bpar`` split stays in
#: its extensions.
FIELD_MODELS = ("mhd_displacement", "electrostatic", "electromagnetic", "prescribed")

#: The axes, in record order, and the vocabulary each is drawn from.
AXES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "scientific_operation": SCIENTIFIC_OPERATIONS,
    "bulk_description": BULK_DESCRIPTIONS,
    "fluid_model": FLUID_MODELS,
    "kinetic_equation": KINETIC_EQUATIONS,
    "kinetic_coupling": KINETIC_COUPLINGS,
    "distribution_formulation": DISTRIBUTION_FORMULATIONS,
    "orbit_representation": ORBIT_REPRESENTATIONS,
    "spatial_domain": SPATIAL_DOMAINS,
    "topology_domain": TOPOLOGY_DOMAINS,
    "regime": REGIMES,
    "field_model": FIELD_MODELS,
})

#: The two fields every record has.
_REQUIRED = ("scientific_operation", "bulk_description")


@dataclass(frozen=True)
class PlasmaFormalism:
    """The physical formulation one run used, as factorized controlled fields.

    Required: ``scientific_operation`` and ``bulk_description``.  Every other field
    is ``None`` when it has no meaning for the formulation and :data:`UNKNOWN` when
    it has one that is undocumented.  ``kinetic_population`` names the populations
    the kinetic equation (or the test-particle model) covers -- free strings such
    as ``"all"``, ``"thermal_ions"``, ``"electrons"``, ``"beam_ions"``.
    ``derived_from`` names the parent theory of a ``reduced`` model (TGLF:
    ``"gyrokinetic"``).  ``solver`` and ``extensions`` carry what is not common:
    the code's own name and its solver-specific physics settings.
    """

    scientific_operation: str
    bulk_description: str
    fluid_model: str | None = None
    kinetic_equation: str | None = None
    kinetic_coupling: str | None = None
    kinetic_population: tuple[str, ...] | None = None
    distribution_formulation: str | None = None
    orbit_representation: str | None = None
    spatial_domain: str | None = None
    topology_domain: str | None = None
    regime: str | None = None
    field_model: str | None = None
    derived_from: str | None = None
    solver: str | None = None
    extensions: Mapping[str, Any] = field(default_factory=dict, hash=False, compare=True)

    def __post_init__(self) -> None:
        for name in _REQUIRED:
            value = getattr(self, name)
            if value is None or value == UNKNOWN:
                raise ValueError(f"{name} is required and cannot be unknown")
        for name, allowed in AXES.items():
            value = getattr(self, name)
            if value is not None and value != UNKNOWN and value not in allowed:
                raise ValueError(f"{name}={value!r} is not one of {allowed} (or None / {UNKNOWN!r})")
        if self.derived_from is not None and self.derived_from not in (*KINETIC_EQUATIONS, *FLUID_MODELS, UNKNOWN):
            raise ValueError(f"derived_from={self.derived_from!r} names no known theory")
        if self.kinetic_population is not None:
            population = self.kinetic_population
            if isinstance(population, str):
                raise TypeError("kinetic_population is a collection of names, not one string")
            object.__setattr__(self, "kinetic_population", tuple(str(item) for item in population))
        object.__setattr__(self, "extensions", MappingProxyType(dict(self.extensions)))
        _check_combination(self)

    def as_dict(self) -> dict[str, Any]:
        """A stable, JSON-ready dict: every field present, ``None`` kept, versioned."""
        out: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "kinetic_population" and value is not None:
                value = list(value)
            elif item.name == "extensions":
                value = dict(value)
            out[item.name] = value
        return out

    def to_json(self) -> str:
        """:meth:`as_dict` as canonical JSON (sorted keys), for hashing and storage."""
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlasmaFormalism":
        """Rebuild a record from :meth:`as_dict` output; a newer schema is refused."""
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"formalism schema_version {version!r}; this VAFT reads {SCHEMA_VERSION}")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known - {"schema_version"})
        if unknown:
            raise ValueError(f"unknown formalism fields {unknown}")
        return cls(**{key: value for key, value in data.items() if key in known})


def _present(value: Any) -> bool:
    """Has a meaning here -- set, or explicitly unknown."""
    return value is not None


def _check_combination(record: PlasmaFormalism) -> None:
    """Reject the combinations that are impossible whatever the solver."""
    bulk = record.bulk_description
    kinetic = _present(record.kinetic_equation)
    fluid = _present(record.fluid_model)
    if bulk == "fluid":
        if kinetic:
            raise ValueError("a fluid formulation solves no kinetic equation; "
                             "a kinetic response inside a fluid model is bulk_description='hybrid'")
        if not fluid:
            raise ValueError("a fluid formulation needs its fluid_model")
    elif bulk == "kinetic":
        if fluid:
            raise ValueError("a kinetic formulation has no fluid_model; with one it is 'hybrid'")
        if not kinetic:
            raise ValueError("a kinetic formulation needs its kinetic_equation")
    elif bulk == "hybrid":
        if not (fluid and kinetic and _present(record.kinetic_coupling)):
            raise ValueError("a hybrid formulation needs fluid_model, kinetic_equation and kinetic_coupling")
    elif bulk == "particle":
        if fluid:
            raise ValueError("a test-particle formulation has no fluid_model")
        if record.kinetic_equation not in (None, "fokker_planck", UNKNOWN):
            raise ValueError("a test-particle formulation solves at most the test-species Fokker-Planck equation")
        if not _present(record.orbit_representation):
            raise ValueError("a test-particle formulation needs its orbit_representation")
    elif bulk == "reduced":
        if kinetic:
            raise ValueError("a reduced model solves no kinetic equation; name its parent theory in derived_from")
    if record.derived_from is not None and bulk != "reduced":
        raise ValueError("derived_from describes a reduced model only")
    if record.kinetic_coupling is not None and bulk not in ("hybrid", "particle"):
        raise ValueError("kinetic_coupling describes how a kinetic result re-enters a fluid or transport model")
    if not kinetic and bulk != "particle":
        for name in ("distribution_formulation", "kinetic_population", "orbit_representation"):
            if _present(getattr(record, name)):
                raise ValueError(f"{name} has no meaning without a kinetic equation or test particles")
