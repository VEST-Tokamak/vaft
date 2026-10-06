"""Canonical species/population state and physics-specific projections of it (issue #1567).

One physical plasma state is not one species representation.  This module
keeps the most explicit composition a product supports -- every component a
nuclide in one charge state and one kinetic population -- and derives reduced
representations only at a physics boundary, each with an explicit record of
what it preserves and what it does not.  The algebra is
:mod:`vaft.formula.impurity`; impurity *composition* (which mixture applies)
is :mod:`vaft.process.impurity` (#1565).  This module adds the axes those two
do not have -- kinetic population, an explicit basis for every dilution, and
the projection contract.

The chain::

    core_profiles ion[] / state[] / density_fast    (species_state_from_core_profiles)
    #1565 resolved or radial composition            (species_state_from_composition)
        -> CanonicalSpeciesState: components = nuclide x charge state x population
        -> composition_moments: n_e(sum Z n), Z_eff, charge fractions by NAMED basis
        -> project_species_state(target_physics) -> SpeciesProjection
               (components for the solver + preserved / not_guaranteed / source ids)

Notation
--------
a          : one component (nuclide, charge state, population)            [-]
Z_a        : charge of component a (a bundle average <Z> where bundled)   [-]
n_a        : density of component a                                       [m^-3]
f_basis    : sum_{a in basis} Z_a n_a / n_e, the charge fraction of a basis [-]

Conventions
-----------
**Component identity is not a solver index.**  ``component_id`` names the
physics (``D-2/Z1/thermal``); a solver's species list is whatever a
projection hands it, and two projections of one state are distinct model
cases even when they share ``state_id``.

**Every dilution names its basis.**  ``thermal_main`` (thermal hydrogen
isotopes), ``fuel`` (hydrogen isotopes in any population), ``impurity``
(Z_n > 1, any population), ``fast_ion`` (every non-thermal population).
There is no unqualified ``dilution``.

**Fast populations stay distinct.**  A thermal and a fast population of one
isotope and charge are two components; they add in charge bookkeeping and
are never merged into one Maxwellian by a projection.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from vaft.spectroscopy import Species

__all__ = [
    "BASES",
    "POPULATIONS",
    "PROJECTION_METHODS",
    "PROJECTION_POLICY",
    "CanonicalSpeciesState",
    "SpeciesComponent",
    "SpeciesProjection",
    "composition_moments",
    "project_species_state",
    "species_state_from_composition",
    "species_state_from_core_profiles",
]

#: Kinetic populations a component may belong to.
POPULATIONS = ("thermal", "nbi_fast", "fusion_born", "rf_minority", "fast_unspecified", "other")

#: Charge-fraction bases :func:`composition_moments` evaluates.
BASES = ("thermal_main", "fuel", "impurity", "fast_ion")

#: Projection methods and what each one guarantees.
PROJECTION_METHODS = {
    "explicit": {
        "preserves": ("every component", "charge density", "Z_eff", "mass density", "populations",
                      "charge states as stored"),
        "not_guaranteed": (),
    },
    "moments_only": {
        "preserves": ("charge density", "Z_eff", "charge fractions of every basis"),
        "not_guaranteed": ("any species list", "collisional coupling", "kinetic response", "atomic rates"),
    },
    "effective_impurity": {
        "preserves": ("charge density", "Z_eff", "impurity mass density", "fuel and fast components",
                      "main-ion dilution"),
        "not_guaranteed": ("gyrokinetic response", "neoclassical friction matrix", "atomic radiation",
                           "charge-state kinetics", "impurity transport", "per-impurity gradients"),
    },
    "charge_state_resolved": {
        "preserves": ("every charge state as a component", "charge density", "Z_eff", "populations"),
        "not_guaranteed": (),
    },
    "fusion": {
        "preserves": ("isotope-resolved fuel components per population", "charge density", "Z_eff",
                      "impurity mass density"),
        "not_guaranteed": ("impurity identity", "impurity charge states", "impurity kinetic response"),
    },
}

#: Default and allowed projection methods per target physics (#1567 Sec. 5).
PROJECTION_POLICY = {
    "quasineutrality": ("moments_only", ("moments_only", "effective_impurity", "explicit")),
    "resistive": ("moments_only", ("moments_only",)),
    "classical": ("explicit", ("explicit", "effective_impurity")),
    "neoclassical": ("explicit", ("explicit",)),
    "turbulence": ("explicit", ("explicit", "effective_impurity")),
    "gyrokinetic": ("explicit", ("explicit",)),
    "atomic": ("charge_state_resolved", ("charge_state_resolved",)),
    "fusion": ("fusion", ("fusion", "explicit")),
}

_HYDROGENIC_MASS = {1: "H", 2: "D", 3: "T"}


def _elements() -> Mapping[str, tuple[int, float]]:
    from vaft.data.synthetic_kinetic_profiles import ION_SPECIES

    return ION_SPECIES


def _profile(value: Any, n: Optional[int], name: str) -> np.ndarray:
    array = np.atleast_1d(np.asarray(value, dtype=float))
    if n is not None and array.size == 1 and n > 1:
        array = np.full(n, float(array[0]))
    if n is not None and array.shape != (n,):
        raise ValueError(f"{name} has shape {array.shape}, the state grid has {n} points")
    return array


@dataclass(frozen=True)
class SpeciesComponent:
    """One physical component: a nuclide, in one charge state, in one kinetic population.

    ``species`` carries the element and mass number (a ``vaft.spectroscopy.Species``);
    ``charge`` is the ionic charge -- a profile ``<Z>(rho)`` when ``bundled`` (several
    charge states of one element stored together), with ``mean_square_charge``
    ``<Z^2>(rho)`` beside it.  ``density`` [m^-3] and ``temperature`` [eV] are on the
    state's grid; ``origin`` says how the density is known (``measured``,
    ``inferred``, ``derived``, ``assumed``, ``modelled`` or ``unlabelled``).
    """

    species: Species
    charge: Any
    population: str
    density: np.ndarray
    mass_amu: float
    temperature: Optional[np.ndarray] = None
    bundled: bool = False
    mean_square_charge: Optional[np.ndarray] = None
    origin: str = "unlabelled"
    source: str = ""
    pseudo_of: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.population not in POPULATIONS:
            raise ValueError(f"population must be one of {POPULATIONS}, got {self.population!r}")
        z_n = self.species.atomic_number
        if z_n is None:
            raise ValueError(f"unknown element {self.species.element!r}")
        charge = np.asarray(self.charge, dtype=float)
        if not np.all(np.isfinite(charge)) or np.any(charge <= 0.0) or np.any(charge > z_n + 1e-9):
            raise ValueError(f"{self.species.element}: charge must lie in (0, Z_n = {z_n}]")
        if not self.bundled and (charge.ndim != 0 or abs(float(charge) - round(float(charge))) > 1e-9):
            raise ValueError("a charge that is not one integer charge state must be marked bundled")
        density = np.atleast_1d(np.asarray(self.density, dtype=float))
        if np.any(density[np.isfinite(density)] < 0.0):
            raise ValueError("density must be non-negative")
        if not (math.isfinite(self.mass_amu) and self.mass_amu > 0.0):
            raise ValueError("mass_amu must be positive")
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "charge", float(charge) if charge.ndim == 0 else charge)

    @property
    def z_n(self) -> int:
        return int(self.species.atomic_number)

    @property
    def hydrogenic(self) -> bool:
        return self.z_n == 1

    @property
    def component_id(self) -> str:
        """``<element>-<A>/Z<charge or bundle>/<population>``, the physics identity.

        A projection's pseudo-ion is ``pseudo(<elements>)/Zeff/<population>``: it
        names what it was reduced from, never a physical nuclide.
        """
        if self.pseudo_of:
            return f"pseudo({'+'.join(self.pseudo_of)})/Zeff/{self.population}"
        a = self.species.mass_number if self.species.mass_number is not None else round(self.mass_amu)
        name = _HYDROGENIC_MASS.get(a, "H") if self.hydrogenic else self.species.element
        z = "bundle" if self.bundled else f"{float(self.charge):g}"
        return f"{name}-{a}/Z{z}/{self.population}"

    def z_moment(self, power: int) -> np.ndarray:
        """``<Z^power>`` on the grid (power 1 or 2)."""
        if power == 1:
            return np.broadcast_to(np.asarray(self.charge, dtype=float), self.density.shape)
        if power == 2:
            if self.mean_square_charge is not None:
                return np.broadcast_to(np.asarray(self.mean_square_charge, dtype=float), self.density.shape)
            return np.broadcast_to(np.asarray(self.charge, dtype=float) ** 2, self.density.shape)
        raise ValueError("power must be 1 or 2")


@dataclass(frozen=True)
class CanonicalSpeciesState:
    """The most explicit species/population composition a product supports, on one grid.

    ``rho`` is the radial grid (``rho_tor_norm``) and ``n_e`` the electron density
    when it is known independently; ``state_id`` hashes the components, the grid
    and the time, so two projections of one state share it.
    """

    components: tuple[SpeciesComponent, ...]
    rho: Optional[np.ndarray] = None
    n_e: Optional[np.ndarray] = None
    time: Optional[float] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components:
            raise ValueError("a species state needs at least one component")
        n = components[0].density.size
        if any(c.density.size != n for c in components):
            raise ValueError("every component must be on the same grid")
        ids = [c.component_id for c in components]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate components: {sorted({i for i in ids if ids.count(i) > 1})}")
        object.__setattr__(self, "components", components)
        if self.rho is not None:
            object.__setattr__(self, "rho", _profile(self.rho, n, "rho"))
        if self.n_e is not None:
            object.__setattr__(self, "n_e", _profile(self.n_e, n, "n_e"))

    @property
    def size(self) -> int:
        return int(self.components[0].density.size)

    @property
    def state_id(self) -> str:
        digest = hashlib.sha256()
        for c in self.components:
            digest.update(c.component_id.encode())
            digest.update(np.ascontiguousarray(c.density).tobytes())
            digest.update(np.ascontiguousarray(c.z_moment(1)).tobytes())
        for extra in (self.rho, self.n_e):
            if extra is not None:
                digest.update(np.ascontiguousarray(extra).tobytes())
        digest.update(repr(self.time).encode())
        return digest.hexdigest()[:16]

    def charge_density(self) -> np.ndarray:
        """``sum_a Z_a n_a`` [m^-3]."""
        return np.sum([c.density * c.z_moment(1) for c in self.components], axis=0)

    def electron_density(self) -> np.ndarray:
        """The stored n_e if known, else the quasi-neutral ``sum_a Z_a n_a`` [m^-3]."""
        return self.n_e if self.n_e is not None else self.charge_density()

    def select(self, basis: str) -> list[SpeciesComponent]:
        """The components of one :data:`BASES` basis."""
        if basis == "thermal_main":
            return [c for c in self.components if c.hydrogenic and c.population == "thermal"]
        if basis == "fuel":
            return [c for c in self.components if c.hydrogenic]
        if basis == "impurity":
            return [c for c in self.components if not c.hydrogenic]
        if basis == "fast_ion":
            return [c for c in self.components if c.population != "thermal"]
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}")


# --- constructors ---------------------------------------------------------------------------


def _nuclide(z_n: float, mass: Optional[float]) -> tuple[Species, float]:
    table = _elements()
    z = int(round(float(z_n)))
    if z == 1:
        a = 1 if mass is None else int(round(float(mass)))
        a = a if a in _HYDROGENIC_MASS else 1
        return Species("H", a), float(mass) if mass is not None else float(table[_HYDROGENIC_MASS[a]][1])
    element = next((sym for sym, (zz, _) in table.items() if zz == z and sym not in ("D", "T")), None)
    if element is None:
        raise ValueError(f"no element with Z_n = {z}")
    standard = float(table[element][1])
    return Species(element, int(round(standard if mass is None else float(mass)))), (
        standard if mass is None else float(mass))


def species_state_from_core_profiles(ods: Any, *, time: Optional[float] = None,
                                     tolerance: float = 5e-4) -> CanonicalSpeciesState:
    """The canonical species state of one ``core_profiles`` slice, read without creating paths.

    Parameters
    ----------
    ods : ODS
        Source; read with non-mutating accessors only [any].
    time : float, optional
        Slice time; required when the IDS holds several slices [s].
    tolerance : float, optional
        Largest ``|t_slice - time|`` accepted [s].

    Returns
    -------
    CanonicalSpeciesState
        One component per stored ``ion[].state[]`` with a density, else one per
        ion (bundled when it carries ``z_ion_1d``); a non-zero ``density_fast``
        becomes a separate ``fast_unspecified`` component; ``n_e`` from the
        electrons [any].

    Raises
    ------
    ValueError
        No slice at the time, no electron density, or an ion without a nuclear
        charge or charge state.

    Processing steps
    ----------------
    1. Match the slice by time; read ``rho_tor_norm`` and n_e.
    2. Per ion: nuclide from ``element.0.z_n``/``a`` (hydrogen isotopes by mass);
       origin from ``density_fit.parameters`` (#1565 record grammar).
    3. Stored charge states become components; otherwise the ion is one
       component -- bundled with ``z_ion_1d``/``z_ion_square_1d`` when present.
    4. ``density_thermal`` (or ``density - density_fast``) is the thermal
       component; ``density_fast`` a separate fast one.

    Input semantics
    ---------------
    A ``core_profiles`` slice, measured, fitted or written by a composition model.

    Output semantics
    ----------------
    The same composition as components with explicit identity and population.

    Convention
    ----------
    IMAS ``density_fast`` does not say what made the fast ions, so the
    population is ``fast_unspecified``; a caller that knows (NUBEAM) relabels it.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A fast component takes no temperature (its distribution is not
    Maxwellian); ``distributions`` (#1216) remains authoritative for it.

    Provenance
    ----------
    .. [issue] #1567 Sec. 1-2, #1565 (origin records).
    """
    from vaft.ods_access import path_count, path_value
    from vaft.process.impurity import _slice_index, _slice_time, composition_record_origin

    index = _slice_index(ods, time, tolerance)
    if index is None:
        raise ValueError(f"no core_profiles slice within {tolerance:g} s of t = {time!r}")
    base = f"core_profiles.profiles_1d.{index}"

    def get(path):
        return path_value(ods, f"{base}.{path}", None)

    ne = get("electrons.density_thermal")
    ne = get("electrons.density") if ne is None else ne
    if ne is None:
        raise ValueError(f"{base} has no electron density")
    ne = np.asarray(ne, dtype=float)
    n = ne.size
    rho = get("grid.rho_tor_norm")
    components: list[SpeciesComponent] = []
    for k in range(path_count(ods, f"{base}.ion")):
        ion = f"ion.{k}"
        z_n, mass = get(f"{ion}.element.0.z_n"), get(f"{ion}.element.0.a")
        if z_n is None:
            raise ValueError(f"{base}.{ion} has no element.0.z_n")
        species, mass_amu = _nuclide(z_n, mass)
        origin = composition_record_origin(get(f"{ion}.density_fit.parameters")) or "unlabelled"
        temperature = get(f"{ion}.temperature")
        temperature = None if temperature is None else _profile(temperature, n, "temperature")
        label = get(f"{ion}.label") or species.element
        n_states = path_count(ods, f"{base}.{ion}.state")
        states = []
        for q in range(n_states):
            density = get(f"{ion}.state.{q}.density_thermal")
            density = get(f"{ion}.state.{q}.density") if density is None else density
            z_lo, z_hi = get(f"{ion}.state.{q}.z_min"), get(f"{ion}.state.{q}.z_max")
            if density is not None and z_lo is not None and z_hi is not None:
                states.append((0.5 * (float(z_lo) + float(z_hi)), _profile(density, n, "state density")))
        if states:
            for charge, density in states:
                components.append(SpeciesComponent(Species(species.element, species.mass_number), charge,
                                                   "thermal", density, mass_amu, temperature, origin=origin,
                                                   source=f"{base}.{ion}.state ({label})"))
            continue
        z1, z2 = get(f"{ion}.z_ion_1d"), get(f"{ion}.z_ion_square_1d")
        z_ion = get(f"{ion}.z_ion")
        if z1 is not None:
            charge, bundled = _profile(z1, n, "z_ion_1d"), True
        elif z_ion is not None:
            charge, bundled = float(z_ion), abs(float(z_ion) - round(float(z_ion))) > 1e-9
        else:
            raise ValueError(f"{base}.{ion} has no charge (z_ion or z_ion_1d)")
        total = get(f"{ion}.density")
        thermal = get(f"{ion}.density_thermal")
        fast = get(f"{ion}.density_fast")
        fast = None if fast is None else _profile(fast, n, "density_fast")
        if thermal is None and total is not None:
            thermal = np.asarray(total, dtype=float) - (0.0 if fast is None else fast)
        if thermal is not None:
            components.append(SpeciesComponent(
                species, charge, "thermal", _profile(thermal, n, "density_thermal"), mass_amu, temperature,
                bundled=bundled, mean_square_charge=None if z2 is None else _profile(z2, n, "z_ion_square_1d"),
                origin=origin, source=f"{base}.{ion} ({label})"))
        if fast is not None and np.any(fast > 0.0):
            components.append(SpeciesComponent(
                species, charge, "fast_unspecified", fast, mass_amu, None, bundled=bundled,
                origin=origin, source=f"{base}.{ion}.density_fast ({label})"))
    if not components:
        raise ValueError(f"{base} carries no ion species")
    return CanonicalSpeciesState(tuple(components), rho=None if rho is None else np.asarray(rho, float),
                                 n_e=ne, time=_slice_time(ods, index),
                                 provenance={"source": base, "constructor": "core_profiles"})


def species_state_from_composition(composition: Any, n_e: Any, *, rho: Any = None,
                                   time: Optional[float] = None, main_isotope: int = 1) -> CanonicalSpeciesState:
    """The canonical species state a #1565 composition implies for a given electron density.

    Parameters
    ----------
    composition : ResolvedImpurityComposition or RadialImpurityComposition
        From :mod:`vaft.process.impurity` [any].
    n_e : float or array-like
        Electron density on the composition's grid [m^-3].
    rho : array-like, optional
        The grid; default the composition's own [-].
    time : float, optional
        Slice time [s].
    main_isotope : int, optional
        Mass number of the hydrogenic main ion (1, 2 or 3) [-].

    Returns
    -------
    CanonicalSpeciesState
        The thermal main ion and, for a radial composition, one component per
        ionised charge state of every element (the most explicit form); for a
        resolved composition, one per species [any].

    Raises
    ------
    ValueError
        A main isotope outside 1-3, or a composition of an unknown type.

    Convention
    ----------
    The origin of every component is the composition's kind (``assumed``,
    ``explicit``, ``derived``, ``inferred``); ``explicit`` is recorded as
    ``assumed``, as the #1565 writers do.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [issue] #1567 Sec. 3, #1565.
    """
    from vaft.process.impurity import RadialImpurityComposition, ResolvedImpurityComposition

    if main_isotope not in _HYDROGENIC_MASS:
        raise ValueError("main_isotope must be 1, 2 or 3")
    table = _elements()
    origin = {"explicit": "assumed"}.get(composition.kind, composition.kind)
    if isinstance(composition, RadialImpurityComposition):
        grid = composition.rho if rho is None else rho
        ne = _profile(n_e, np.size(grid), "n_e")
        components = [SpeciesComponent(Species("H", main_isotope), 1.0, "thermal",
                                       ne * np.asarray(composition.main_ion_fraction, float),
                                       float(table[_HYDROGENIC_MASS[main_isotope]][1]), origin=origin,
                                       source="radial composition: main ion")]
        for k, element in enumerate(composition.elements):
            states = np.asarray(composition.charge_state_fractions[k], dtype=float)
            elemental = ne * np.asarray(composition.elemental_fractions[:, k], dtype=float)
            for q in range(1, states.shape[1]):
                components.append(SpeciesComponent(
                    Species(element, round(table[element][1])), float(q), "thermal", elemental * states[:, q],
                    float(table[element][1]), origin=origin, source=f"radial composition: {element}{q}+"))
        return CanonicalSpeciesState(tuple(components), rho=grid, n_e=ne, time=time,
                                     provenance={"constructor": "radial composition",
                                                 "normalization": dict(composition.normalization)})
    if isinstance(composition, ResolvedImpurityComposition):
        n = None if rho is None else np.size(rho)
        ne = _profile(n_e, n if n is not None else np.size(n_e), "n_e")
        fractions = np.atleast_2d(np.asarray(composition.impurity_fractions, dtype=float))
        if fractions.shape[0] == 1 and ne.size > 1:
            fractions = np.repeat(fractions, ne.size, axis=0)
        main = np.broadcast_to(np.asarray(composition.main_ion_fraction, dtype=float), ne.shape)
        components = [SpeciesComponent(Species("H", main_isotope), 1.0, "thermal", ne * main,
                                       float(table[_HYDROGENIC_MASS[main_isotope]][1]), origin=origin,
                                       source=f"{composition.source}: main ion")]
        for j, s in enumerate(composition.species):
            components.append(SpeciesComponent(Species(s.element, round(s.mass)), s.charge_state, "thermal",
                                               ne * fractions[:, j], s.mass, origin=origin,
                                               source=f"{composition.source}: {s.label}"))
        return CanonicalSpeciesState(tuple(components), rho=rho, n_e=ne, time=time,
                                     provenance={"constructor": "resolved composition",
                                                 "composition_kind": composition.kind})
    raise ValueError(f"unknown composition type {type(composition).__name__}")


# --- composition views ---------------------------------------------------------------------


def composition_moments(state: CanonicalSpeciesState) -> dict[str, Any]:
    """Composition moments of a species state, every dilution with its basis named.

    Parameters
    ----------
    state : CanonicalSpeciesState
        The canonical state [any].

    Returns
    -------
    dict
        ``n_e_quasineutral`` [m^-3], ``quasineutrality_residual`` (relative to the
        stored n_e, or ``None``), ``zeff`` [-], and for every basis of
        :data:`BASES` ``fraction_<basis>`` = sum Z n / n_e [-]; plus
        ``dilution_thermal_main`` and ``dilution_fuel`` = 1 - the main
        fractions [-] [any].

    Convention
    ----------
    Fractions are charge fractions over the stored n_e when known (else the
    quasi-neutral sum): ``fraction_thermal_main + fraction_impurity +`` the
    fast hydrogenic share add to one only for a quasi-neutral state.  Z_eff
    uses ``<Z^2>`` for a bundled component, never ``<Z>^2``.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [issue] #1567 Sec. 3, #1565 dilution comment.
    """
    ne = state.electron_density()
    charge = state.charge_density()
    with np.errstate(invalid="ignore", divide="ignore"):
        out: dict[str, Any] = {
            "n_e_quasineutral": charge,
            "quasineutrality_residual": None if state.n_e is None else (charge - state.n_e) / state.n_e,
            "zeff": np.sum([c.density * c.z_moment(2) for c in state.components], axis=0) / ne,
        }
        for basis in BASES:
            chosen = state.select(basis)
            out[f"fraction_{basis}"] = (np.sum([c.density * c.z_moment(1) for c in chosen], axis=0) / ne
                                        if chosen else np.zeros(state.size))
    out["dilution_thermal_main"] = 1.0 - out["fraction_thermal_main"]
    out["dilution_fuel"] = 1.0 - out["fraction_fuel"]
    out["basis_definitions"] = {
        "thermal_main": "thermal hydrogen isotopes", "fuel": "hydrogen isotopes, any population",
        "impurity": "Z_n > 1, any population", "fast_ion": "every non-thermal population",
    }
    return out


# --- projections ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeciesProjection:
    """A species representation for one target physics, and what it does and does not keep."""

    target_physics: str
    method: str
    components: tuple[SpeciesComponent, ...]
    source_components: tuple[str, ...]
    preserves: tuple[str, ...]
    not_guaranteed: tuple[str, ...]
    state_id: str
    moments: Mapping[str, Any] = field(default_factory=dict)
    notes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def projection_id(self) -> str:
        payload = {"state": self.state_id, "target": self.target_physics, "method": self.method,
                   "components": [c.component_id for c in self.components]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def record(self) -> dict[str, Any]:
        """A JSON-ready provenance record for a run that used this projection."""
        return {"state_id": self.state_id, "projection_id": self.projection_id,
                "target_physics": self.target_physics, "method": self.method,
                "components": [c.component_id for c in self.components],
                "source_components": list(self.source_components),
                "preserves": list(self.preserves), "not_guaranteed": list(self.not_guaranteed),
                "notes": dict(self.notes)}


def _merge_impurities(state: CanonicalSpeciesState) -> tuple[list[SpeciesComponent], dict[str, Any]]:
    """Fuel and fast components kept; thermal impurities reduced to one pseudo-impurity."""
    kept = [c for c in state.components if c.hydrogenic or c.population != "thermal"]
    merged = [c for c in state.components if not c.hydrogenic and c.population == "thermal"]
    if not merged:
        return kept, {"merged": []}
    densities = np.stack([c.density for c in merged], axis=-1)
    z1 = np.stack([np.asarray(c.z_moment(1), float) for c in merged], axis=-1)
    z2 = np.stack([np.asarray(c.z_moment(2), float) for c in merged], axis=-1)
    masses = np.array([c.mass_amu for c in merged])
    # Use the Z and Z^2 moments directly (bundled components carry <Z^2> != <Z>^2).
    charge_density = np.sum(densities * z1, axis=-1)
    z2_density = np.sum(densities * z2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_eff = np.where(charge_density > 0, z2_density / charge_density, np.nan)
        n_eff = np.where(z2_density > 0, charge_density**2 / z2_density, 0.0)
        a_eff = np.where(n_eff > 0, np.sum(densities * masses, axis=-1) / n_eff, np.nan)
    if not np.any(np.isfinite(z_eff)):
        return kept, {"merged": [c.component_id for c in merged], "note": "zero impurity density"}
    z_eff = np.where(np.isfinite(z_eff), z_eff, np.nanmean(z_eff))
    a_eff_scalar = float(np.nanmean(a_eff))
    # The algebra of vaft.formula.impurity.reduce_impurity_mixture, written on the Z and Z^2
    # moments so a bundled component's <Z^2> (not <Z>^2) is honoured.  The pseudo-ion carries
    # the heaviest merged element as its nuclide: Z_I,eff never exceeds that element's Z_n.
    host = max(merged, key=lambda c: c.z_n)
    pseudo = SpeciesComponent(Species(host.species.element, None), z_eff, "thermal", n_eff, a_eff_scalar,
                              bundled=True, mean_square_charge=z_eff**2, origin="derived",
                              source="effective impurity of " + ", ".join(c.component_id for c in merged),
                              pseudo_of=tuple(sorted({c.species.element for c in merged})))
    return kept + [pseudo], {"merged": [c.component_id for c in merged],
                             "effective_charge": z_eff, "effective_mass_amu": a_eff_scalar,
                             "nuclide_label": f"pseudo-impurity carried as {host.species.element} "
                                              "(its Z_n bounds Z_I,eff); not a physical nuclide"}


def project_species_state(state: CanonicalSpeciesState, target_physics: str,
                          method: Optional[str] = None) -> SpeciesProjection:
    """The species representation one target physics receives, with its conservation contract.

    Parameters
    ----------
    state : CanonicalSpeciesState
        The canonical state [any].
    target_physics : str
        One of :data:`PROJECTION_POLICY` (``quasineutrality``, ``resistive``,
        ``classical``, ``neoclassical``, ``turbulence``, ``gyrokinetic``,
        ``atomic``, ``fusion``) [-].
    method : str, optional
        A method the target allows; default the target's own [-].

    Returns
    -------
    SpeciesProjection
        Components for the solver, the source component ids, what is preserved
        and what is not guaranteed, the moments, and ``state_id`` /
        ``projection_id`` [any].

    Raises
    ------
    ValueError
        An unknown target, a method the target does not allow (for example an
        effective impurity for NEO), or a charge-state projection of a state
        whose components are bundled.

    Processing steps
    ----------------
    1. Look up the target's default and allowed methods.
    2. ``explicit``: every component; ``moments_only``: no components, the
       moments; ``effective_impurity``: fuel and fast kept, thermal impurities
       reduced on their Z and Z^2 moments; ``charge_state_resolved``: refuse a
       bundled component; ``fusion``: hydrogen isotopes kept per population,
       impurities reduced.
    3. Attach the method's preserves / not_guaranteed and the moments.

    Convention
    ----------
    #1567 Sec. 5: the representation follows the physics operator, never a
    global convenience.  Same Z_eff does not mean same neoclassical or
    gyrokinetic response, so ``neoclassical`` and ``gyrokinetic`` allow only
    ``explicit``.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    No solver-specific readiness here (that stays with each solver, #1428);
    phase-space detail of fast populations lives in ``distributions`` (#1216).

    Provenance
    ----------
    .. [issue] #1567 Sec. 4-7.
    """
    if target_physics not in PROJECTION_POLICY:
        raise ValueError(f"target_physics must be one of {tuple(PROJECTION_POLICY)}, got {target_physics!r}")
    default, allowed = PROJECTION_POLICY[target_physics]
    method = method or default
    if method not in allowed:
        raise ValueError(f"{target_physics!r} does not allow method {method!r}; allowed: {allowed}")
    contract = PROJECTION_METHODS[method]
    notes: dict[str, Any] = {}
    if method == "explicit":
        components = list(state.components)
    elif method == "moments_only":
        components = []
    elif method == "effective_impurity":
        components, notes = _merge_impurities(state)
    elif method == "charge_state_resolved":
        bundled = [c.component_id for c in state.components if c.bundled]
        if bundled:
            raise ValueError(f"components {bundled} are charge-state bundles; resolve their charge states "
                             "(e.g. from atomic data, #1565) before an atomic projection")
        components = list(state.components)
    else:  # fusion
        components, notes = _merge_impurities(state)
    return SpeciesProjection(
        target_physics=target_physics, method=method, components=tuple(components),
        source_components=tuple(c.component_id for c in state.components),
        preserves=contract["preserves"], not_guaranteed=contract["not_guaranteed"],
        state_id=state.state_id, moments=composition_moments(state), notes=notes,
    )
