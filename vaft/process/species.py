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
        "preserves": ("hydrogen isotopes and 3He per population", "charge density", "Z_eff",
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
    charge states of one element stored together, ``charge_range`` = their lowest and
    highest charge), with ``mean_square_charge`` ``<Z^2>(rho)`` beside it.  ``density``
    [m^-3] and ``temperature`` [eV] are on the state's grid; ``mass_amu`` is a number,
    or a profile only for a projection's pseudo-ion.  ``origin`` says how the density
    is known (``measured``, ``inferred``, ``derived``, ``assumed``, ``modelled`` or
    ``unlabelled``); ``pseudo_of`` is set only on a projection's reduced pseudo-ion
    and names the elements it stands for.
    """

    species: Species
    charge: Any
    population: str
    density: np.ndarray
    mass_amu: Any
    temperature: Optional[np.ndarray] = None
    bundled: bool = False
    mean_square_charge: Optional[np.ndarray] = None
    origin: str = "unlabelled"
    source: str = ""
    pseudo_of: tuple[str, ...] = ()
    charge_range: Optional[tuple[float, float]] = None

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
        mass = np.asarray(self.mass_amu, dtype=float)
        if mass.ndim and not self.pseudo_of:
            raise ValueError("only a projection's pseudo-ion may carry a mass profile")
        if np.any(~np.isfinite(mass[np.isfinite(mass)] if mass.ndim else mass)) or np.any(mass[np.isfinite(mass)] <= 0.0):
            raise ValueError("mass_amu must be positive")
        if self.mean_square_charge is not None:
            # <Z>^2 <= <Z^2> <= Z_n <Z>: a variance is non-negative and no charge exceeds Z_n
            z2 = np.broadcast_to(np.asarray(self.mean_square_charge, dtype=float), density.shape)
            z1 = np.broadcast_to(charge, density.shape)
            ok = np.isfinite(z2)
            if np.any(z2[ok] < z1[ok] ** 2 * (1 - 1e-9)) or np.any(z2[ok] > z_n * z1[ok] * (1 + 1e-9)):
                raise ValueError(f"{self.species.element}: mean_square_charge must satisfy <Z>^2 <= <Z^2> <= Z_n <Z>")
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "charge", float(charge) if charge.ndim == 0 else charge)
        object.__setattr__(self, "mass_amu", float(mass) if mass.ndim == 0 else mass)

    @property
    def z_n(self) -> int:
        return int(self.species.atomic_number)

    @property
    def hydrogenic(self) -> bool:
        return self.z_n == 1

    @property
    def mass_number(self) -> int:
        if self.species.mass_number is not None:
            return int(self.species.mass_number)
        return int(round(float(np.nanmean(np.asarray(self.mass_amu, dtype=float)))))

    @property
    def component_id(self) -> str:
        """``<element>-<A>/Z<charge or range>/<population>``, the physics identity.

        A bundle is ``Z<lo>-<hi>`` (``Zbundle`` when its range is unknown); a
        projection's pseudo-ion is ``pseudo(<elements>)/Zeff/<population>``: it names
        what it was reduced from, never a physical nuclide.
        """
        if self.pseudo_of:
            return f"pseudo({'+'.join(self.pseudo_of)})/Zeff/{self.population}"
        a = self.mass_number
        name = _HYDROGENIC_MASS.get(a, "H") if self.hydrogenic else self.species.element
        if self.bundled:
            z = (f"{self.charge_range[0]:g}-{self.charge_range[1]:g}" if self.charge_range else "bundle")
        else:
            z = f"{float(self.charge):g}"
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
    when it is known independently; ``state_id`` hashes everything a projection or a
    solver consumes -- each component's identity, density, <Z>, <Z^2>, mass,
    temperature and origin, the grid and the time -- independent of component order.
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

        def add(value):
            digest.update(b"|" if value is None else np.ascontiguousarray(np.asarray(value, dtype=float)).tobytes())

        for c in sorted(self.components, key=lambda c: c.component_id):
            digest.update(f"{c.component_id}|{c.origin}".encode())
            for value in (c.density, c.z_moment(1), c.z_moment(2), c.mass_amu, c.temperature):
                add(value)
        add(self.rho)
        add(self.n_e)
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

#: Standard masses of the hydrogen isotopes [u]; a stored mass is matched to one of them.
_HYDROGEN_MASSES = {1: 1.00784, 2: 2.01410, 3: 3.01605}


def _nuclide(z_n: float, mass: Optional[float]) -> tuple[Species, float]:
    """The nuclide of a stored ion: hydrogen isotopes by mass, other elements by Z_n."""
    from vaft.spectroscopy import ATOMIC_NUMBERS

    z = int(round(float(z_n)))
    if z == 1:
        if mass is None:
            return Species("H", 1), _HYDROGEN_MASSES[1]
        a = min(_HYDROGEN_MASSES, key=lambda k: abs(_HYDROGEN_MASSES[k] - float(mass)))
        if abs(_HYDROGEN_MASSES[a] - float(mass)) > 0.1:
            raise ValueError(f"a hydrogenic ion of mass {float(mass):g} u is no hydrogen isotope")
        return Species("H", a), float(mass)
    element = next((sym for sym, zz in ATOMIC_NUMBERS.items() if zz == z and sym not in ("D", "T")), None)
    if element is None:
        raise ValueError(f"no element with Z_n = {z}")
    table = _elements()
    standard = float(table[element][1]) if element in table else (None if mass is None else float(mass))
    if standard is None:
        raise ValueError(f"{element}: no stored mass and no standard mass in the table")
    value = standard if mass is None else float(mass)
    return Species(element, int(round(value))), value


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
    3. Stored charge states become components (a state spanning several
       charges is a bundle with its ``z_average_1d``/``z_average_square_1d``
       (or the scalar ``z_average``/``z_square_average``), its own
       temperature when stored); otherwise the ion is one component -- bundled
       with ``z_ion_1d``/``z_ion_square_1d`` when present.  A share of the ion
       density its states do not carry is reported in the provenance notes.
    4. At every level, ``density_thermal`` (or ``density - density_fast``) is
       the thermal component and ``density_fast`` a separate fast one; an
       ion-level ``density_fast`` is kept even when the states carry none.
    5. n_e is the total ``electrons.density`` (falling back to the thermal
       one), since the ion charge counts fast ions too.

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

    # the total electron density: ion charge includes any fast ions, so n_e must too
    ne = get("electrons.density")
    ne = get("electrons.density_thermal") if ne is None else ne
    if ne is None:
        raise ValueError(f"{base} has no electron density")
    ne = np.asarray(ne, dtype=float)
    n = ne.size
    rho = get("grid.rho_tor_norm")
    components: list[SpeciesComponent] = []
    notes: dict[str, Any] = {}

    def split(prefix: str):
        """(thermal, fast) densities at ``prefix``: density_thermal, or density - density_fast."""
        total, thermal, fast = get(f"{prefix}.density"), get(f"{prefix}.density_thermal"), get(f"{prefix}.density_fast")
        fast = None if fast is None else _profile(fast, n, f"{prefix}.density_fast")
        if thermal is None and total is not None:
            thermal = np.asarray(total, dtype=float) - (0.0 if fast is None else fast)
        return (None if thermal is None else _profile(thermal, n, f"{prefix}.density_thermal")), fast

    def add(species, charge, bundled, z2, thermal, fast, mass_amu, temperature, origin, source, charge_range=None):
        if thermal is not None:
            components.append(SpeciesComponent(species, charge, "thermal", thermal, mass_amu, temperature,
                                               bundled=bundled, mean_square_charge=z2, origin=origin,
                                               source=source, charge_range=charge_range))
        if fast is not None and np.any(fast > 0.0):
            components.append(SpeciesComponent(species, charge, "fast_unspecified", fast, mass_amu, None,
                                               bundled=bundled, mean_square_charge=z2, origin=origin,
                                               source=source + ".density_fast", charge_range=charge_range))

    for k in range(path_count(ods, f"{base}.ion")):
        ion = f"ion.{k}"
        z_n, mass = get(f"{ion}.element.0.z_n"), get(f"{ion}.element.0.a")
        if z_n is None:
            raise ValueError(f"{base}.{ion} has no element.0.z_n")
        species, mass_amu = _nuclide(z_n, mass)
        origin = composition_record_origin(get(f"{ion}.density_fit.parameters")) or "unlabelled"
        ion_temperature = get(f"{ion}.temperature")
        ion_temperature = None if ion_temperature is None else _profile(ion_temperature, n, "temperature")
        label = get(f"{ion}.label") or species.element
        states_read = 0
        state_charge = np.zeros(n)
        for q in range(path_count(ods, f"{base}.{ion}.state")):
            prefix = f"{ion}.state.{q}"
            z_lo, z_hi = get(f"{prefix}.z_min"), get(f"{prefix}.z_max")
            thermal, fast = split(prefix)
            if z_lo is None or z_hi is None or (thermal is None and fast is None):
                continue
            z_lo, z_hi = float(z_lo), float(z_hi)
            if abs(z_hi - z_lo) < 1e-9 and abs(z_lo - round(z_lo)) < 1e-9:
                charge, bundled, z2, rng = z_lo, False, None, None
            else:   # a state spanning several charges is a bundle (IMAS z_average, z_square_average)
                # IMAS: z_average_1d / z_average_square_1d are profiles, z_average / z_square_average scalars
                z_avg = get(f"{prefix}.z_average_1d")
                z_avg = get(f"{prefix}.z_average") if z_avg is None else z_avg
                z_sq = get(f"{prefix}.z_average_square_1d")
                z_sq = get(f"{prefix}.z_square_average") if z_sq is None else z_sq
                charge = _profile(z_avg, n, "z_average") if z_avg is not None else 0.5 * (z_lo + z_hi)
                z2 = None if z_sq is None else _profile(z_sq, n, "z_square_average")
                bundled, rng = True, (z_lo, z_hi)
            temperature = get(f"{prefix}.temperature")
            temperature = ion_temperature if temperature is None else _profile(temperature, n, "state temperature")
            add(Species(species.element, species.mass_number), charge, bundled, z2, thermal, fast, mass_amu,
                temperature, origin, f"{base}.{prefix} ({label})", rng)
            states_read += 1
            state_charge = state_charge + np.nan_to_num((0.0 if thermal is None else thermal)
                                                        + (0.0 if fast is None else fast))
        if states_read:
            ion_total = get(f"{ion}.density")
            if ion_total is not None:
                ion_total = _profile(ion_total, n, "density")
                with np.errstate(invalid="ignore", divide="ignore"):
                    missing = np.nanmax(np.where(ion_total > 0, 1.0 - state_charge / ion_total, 0.0))
                if missing > 1e-6:
                    notes[f"{ion} ({label})"] = (f"its states carry {1 - missing:.3g} of the ion density at "
                                                 "the worst point; the rest has no charge state and is not read")
            # the ion-level fast density, when the states carry none of their own
            if not any(c.population != "thermal" and c.source.startswith(f"{base}.{ion}.") for c in components):
                _, fast = split(ion)
                z_ion = get(f"{ion}.z_ion")
                if fast is not None and np.any(fast > 0.0):
                    if z_ion is None:
                        raise ValueError(f"{base}.{ion} has a density_fast but no z_ion to give it a charge")
                    zf = float(z_ion)
                    bundled_fast = abs(zf - round(zf)) > 1e-9
                    add(species, zf, bundled_fast, None, None, fast, mass_amu, None, origin,
                        f"{base}.{ion} ({label})", None)
            continue
        z1, z2 = get(f"{ion}.z_ion_1d"), get(f"{ion}.z_ion_square_1d")
        z_ion = get(f"{ion}.z_ion")
        if z1 is not None:
            charge, bundled = _profile(z1, n, "z_ion_1d"), True
        elif z_ion is not None:
            charge, bundled = float(z_ion), abs(float(z_ion) - round(float(z_ion))) > 1e-9
        else:
            raise ValueError(f"{base}.{ion} has no charge (z_ion or z_ion_1d)")
        thermal, fast = split(ion)
        add(species, charge, bundled, None if z2 is None else _profile(z2, n, "z_ion_square_1d"),
            thermal, fast, mass_amu, ion_temperature, origin, f"{base}.{ion} ({label})")
    if not components:
        raise ValueError(f"{base} carries no ion species")
    return CanonicalSpeciesState(tuple(components), rho=None if rho is None else np.asarray(rho, float),
                                 n_e=ne, time=_slice_time(ods, index),
                                 provenance={"source": base, "constructor": "core_profiles", "notes": notes})


def species_state_from_composition(composition: Any, n_e: Any, *, rho: Any = None,
                                   time: Optional[float] = None,
                                   main_isotope: Optional[int] = None) -> CanonicalSpeciesState:
    """The canonical species state a #1565 composition implies for a given electron density.

    Parameters
    ----------
    composition : ResolvedImpurityComposition or RadialImpurityComposition
        From :mod:`vaft.process.impurity` [any].
    n_e : float or array-like
        Electron density on the composition's grid [m^-3].
    rho : array-like, optional
        The grid; default the composition's own (a radial composition refuses
        any other) [-].
    time : float, optional
        Slice time; default the composition's own [s].
    main_isotope : int, optional
        Mass number (1, 2 or 3) of a main ion the composition names only as
        ``H``; refused when it names ``D``, ``T`` or another element [-].

    Returns
    -------
    CanonicalSpeciesState
        The thermal main ion and, for a radial composition, one component per
        ionised charge state of every element (the most explicit form); for a
        resolved composition, one per species [any].

    Raises
    ------
    ValueError
        A main isotope outside 1-3 or given for a named main ion, an unknown
        main ion, a grid other than a radial composition's own, an n_e other
        than the one its charge states were computed with, or an unknown
        composition type.

    Convention
    ----------
    The origin of every component is the composition's kind (``assumed``,
    ``explicit``, ``derived``, ``inferred``); ``explicit`` is recorded as
    ``assumed``, as the #1565 writers do.  The main ion and its charge are the
    composition's own (``H``/``D``/``T``, or e.g. ``He`` with charge 2).

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [issue] #1567 Sec. 3, #1565.
    """
    from vaft.process.impurity import RadialImpurityComposition, ResolvedImpurityComposition

    table = _elements()
    origin = {"explicit": "assumed"}.get(composition.kind, composition.kind)
    main_ion = str(getattr(composition, "main_ion", "H"))
    if main_isotope is not None and main_ion != "H":
        raise ValueError(f"the composition names its main ion ({main_ion}); main_isotope applies only to 'H'")
    if main_ion in ("H", "D", "T"):
        a = {"H": main_isotope or 1, "D": 2, "T": 3}[main_ion]
        if a not in _HYDROGENIC_MASS:
            raise ValueError("main_isotope must be 1, 2 or 3")
        main_species, main_charge, main_mass = Species("H", a), 1.0, _HYDROGEN_MASSES[a]
    elif main_ion in table:
        z_main, main_mass = table[main_ion]
        main_species, main_charge = Species(main_ion, int(round(main_mass))), float(z_main)
    else:
        raise ValueError(f"unknown main ion {main_ion!r}")
    if isinstance(composition, RadialImpurityComposition):
        grid = composition.rho
        if rho is not None and not (np.shape(rho) == np.shape(grid) and np.allclose(rho, grid)):
            raise ValueError("a radial composition lives on its own rho grid; pass that grid or none")
        ne = _profile(n_e, np.size(grid), "n_e")
        ok = np.isfinite(composition.ne_m3) & np.isfinite(ne)
        if not np.allclose(ne[ok], np.asarray(composition.ne_m3)[ok], rtol=1e-6):
            raise ValueError("n_e differs from the n_e the charge states were computed with")
        components = [SpeciesComponent(main_species, main_charge, "thermal",
                                       ne * np.asarray(composition.main_ion_fraction, float), main_mass,
                                       origin=origin, source="radial composition: main ion")]
        for k, element in enumerate(composition.elements):
            states = np.asarray(composition.charge_state_fractions[k], dtype=float)
            elemental = ne * np.asarray(composition.elemental_fractions[:, k], dtype=float)
            for q in range(1, states.shape[1]):
                components.append(SpeciesComponent(
                    Species(element, round(table[element][1])), float(q), "thermal", elemental * states[:, q],
                    float(table[element][1]), origin=origin, source=f"radial composition: {element}{q}+"))
        return CanonicalSpeciesState(tuple(components), rho=grid, n_e=ne,
                                     time=composition.time if time is None else time,
                                     provenance={"constructor": "radial composition",
                                                 "normalization": dict(composition.normalization)})
    if isinstance(composition, ResolvedImpurityComposition):
        grid = composition.rho if rho is None else rho
        fractions = np.asarray(composition.impurity_fractions, dtype=float)
        n = (np.size(grid) if grid is not None else
             (fractions.shape[0] if fractions.ndim == 2 else np.size(n_e)))
        ne = _profile(n_e, n, "n_e")
        fractions = np.atleast_2d(fractions)
        if fractions.shape[0] == 1 and ne.size > 1:
            fractions = np.repeat(fractions, ne.size, axis=0)
        main = np.broadcast_to(np.asarray(composition.main_ion_fraction, dtype=float), ne.shape)
        components = [SpeciesComponent(main_species, main_charge, "thermal", ne * main, main_mass,
                                       origin=origin, source=f"{composition.source}: main ion")]
        for j, item in enumerate(composition.species):
            components.append(SpeciesComponent(Species(item.element, round(item.mass)), item.charge_state,
                                               "thermal", ne * fractions[:, j], item.mass, origin=origin,
                                               source=f"{composition.source}: {item.label}"))
        return CanonicalSpeciesState(tuple(components), rho=grid, n_e=ne,
                                     time=composition.time if time is None else time,
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


#: Non-hydrogen nuclides a fusion projection keeps explicit (D-3He, 3He-3He channels).
_FUSION_NUCLIDES = {("He", 3)}


def _merge_impurities(state: CanonicalSpeciesState, keep=lambda c: False) -> tuple[list[SpeciesComponent], dict[str, Any]]:
    """Fuel, fast and ``keep``-selected components kept; the other thermal impurities reduced to one pseudo-ion.

    The algebra of :func:`vaft.formula.impurity.reduce_impurity_mixture`, written on
    the Z and Z^2 moments so a bundled component's <Z^2> (not <Z>^2) is honoured,
    and point by point: the pseudo-ion's mass is a profile wherever the mix
    varies with radius, so the impurity mass density is kept everywhere.  A
    point where any merged density is undefined stays undefined.
    """
    kept = [c for c in state.components if c.hydrogenic or c.population != "thermal" or keep(c)]
    merged = [c for c in state.components if c not in kept]
    if not merged:
        return kept, {"merged": []}
    densities = np.stack([c.density for c in merged], axis=-1)
    z1 = np.stack([np.asarray(c.z_moment(1), float) for c in merged], axis=-1)
    z2 = np.stack([np.asarray(c.z_moment(2), float) for c in merged], axis=-1)
    masses = np.stack([np.broadcast_to(np.asarray(c.mass_amu, float), c.density.shape) for c in merged], axis=-1)
    defined = np.all(np.isfinite(densities), axis=-1)
    charge_density = np.sum(densities * z1, axis=-1)
    z2_density = np.sum(densities * z2, axis=-1)
    mass_density = np.sum(densities * masses, axis=-1)
    present = defined & (charge_density > 0.0)
    if not np.any(present):
        return kept, {"merged": [c.component_id for c in merged], "note": "no impurity density anywhere"}
    with np.errstate(invalid="ignore", divide="ignore"):
        z_eff = np.where(present, z2_density / charge_density, np.nan)
        n_eff = np.where(defined, np.where(present, charge_density**2 / z2_density, 0.0), np.nan)
        a_eff = np.where(present, mass_density / n_eff, np.nan)
    # where the impurity density is zero the pseudo-ion's charge and mass are immaterial: take
    # the nearest defined values so they stay physical (n_eff = 0 there keeps every moment).
    fill = ~present   # zero density (moments kept by n_eff = 0) or undefined (n_eff stays NaN)
    if np.any(fill):
        index = np.arange(z_eff.size)
        # anti-alias: spatial interpolation over rho / r/a, not time -- no sample rate to reduce
        z_eff[fill] = np.interp(index[fill], index[present], z_eff[present])
        # anti-alias: spatial interpolation over rho / r/a, not time -- no sample rate to reduce
        a_eff[fill] = np.interp(index[fill], index[present], a_eff[present])
    host = max(merged, key=lambda c: c.z_n)
    pseudo = SpeciesComponent(Species(host.species.element, None), z_eff, "thermal", n_eff, a_eff,
                              bundled=True, mean_square_charge=z_eff**2, origin="derived",
                              source="effective impurity of " + ", ".join(c.component_id for c in merged),
                              pseudo_of=tuple(sorted({c.species.element for c in merged})))
    return kept + [pseudo], {"merged": [c.component_id for c in merged], "effective_charge": z_eff,
                             "effective_mass_amu": a_eff,
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
       reduced on their Z and Z^2 moments with a mass profile; ``charge_state_resolved``:
       refuse a bundled component; ``fusion``: hydrogen isotopes and 3He kept
       per population, the other impurities reduced.
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
    else:  # fusion: every hydrogen isotope and 3He stay explicit, the rest is reduced
        components, notes = _merge_impurities(
            state, keep=lambda c: (c.species.element, c.mass_number) in _FUSION_NUCLIDES)
    return SpeciesProjection(
        target_physics=target_physics, method=method, components=tuple(components),
        source_components=tuple(c.component_id for c in state.components),
        preserves=contract["preserves"], not_guaranteed=contract["not_guaranteed"],
        state_id=state.state_id, moments=composition_moments(state), notes=notes,
    )
