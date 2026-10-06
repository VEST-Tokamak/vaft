"""Impurity composition: which mixture a plasma is assumed, derived or measured to carry, resolved once.

One resolved impurity composition for every consumer of Z_eff (issue #1565,
Stage B): a species list -- element, charge state, mass and relative
density, kept apart -- with the target effective charge it is closed at, the
absolute fractions ``n_s / n_e`` that closure gives, the reduced
pseudo-impurity and the main-ion dilution, and a ``kind`` saying how the
composition is known.  The algebra is :mod:`vaft.formula.impurity`; this
module decides *which* composition applies and records why.

The chain::

    core_profiles ion[] labelled measured        -> kind "measured"
    explicit composition argument                -> kind "explicit"
    derived composition (an atomic-data model)   -> kind "derived"
    core_profiles ion[] labelled assumed/unlabelled -> kind "assumed"
    machine preset (``"vest"``: vest.yaml impurity_model) -> kind "assumed"
    -> target Z_eff: core_profiles.zeff labelled measured, else the composition's own
    -> n_s/n_e, n_main/n_e, S1, S2, Z_I,eff, f_dil   [ResolvedImpurityComposition]

Notation
--------
w_s       : relative impurity particle density, sum_s w_s = 1          [-]
Z_s       : charge state of impurity species s                         [-]
S_1, S_2  : sum_s w_s Z_s and sum_s w_s Z_s^2                           [-]
Z_I,eff   : charge of the reduced pseudo-impurity, S_2 / S_1           [-]
f_dil     : main-ion dilution 1 - n_main / n_e                         [-]

Conventions
-----------
**Kinds are not interchangeable.**  ``measured`` is a composition a
diagnostic determined; ``explicit`` is what the caller passed; ``derived``
is computed from other data by a stated model; ``assumed`` is a preset or a
stored composition that says it was assumed (or says nothing).  A resistive
Z_eff inferred from the transformer balance (#1214) is none of these: it is
accepted only as ``resistive_zeff=`` and carried beside the composition
under its own provenance, never used as a target and never turned into a
composition (#1566).

**Provenance records** on ``core_profiles`` use one grammar, written by
:func:`composition_record_text` and read by :func:`composition_record_origin`:
``origin=<measured|assumed|derived|inferred>; method=<m>[; key=value...]``.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np

from vaft.formula.impurity import (
    impurity_mixture_moments,
    main_ion_density_from_species,
    reduce_impurity_mixture,
    solve_impurity_mixture_for_target_zeff,
)

__all__ = [
    "COMPOSITION_KINDS",
    "COMPOSITION_ORIGINS",
    "ImpuritySpecies",
    "ImpurityComposition",
    "ResolvedImpurityComposition",
    "RadialImpurityComposition",
    "NORMALIZATIONS",
    "charge_state_moments",
    "composition_from_fractions",
    "composition_from_model",
    "composition_record_origin",
    "composition_record_text",
    "populate_impurity_profiles",
    "populate_radial_impurity_profiles",
    "populate_zeff_profile",
    "resolve_impurity_composition",
    "resolve_radial_composition",
    "surface_composition_profile",
]

#: How a resolved composition is known, highest precedence first.
COMPOSITION_KINDS = ("measured", "explicit", "derived", "assumed")

#: The ``origin=`` values a ``core_profiles`` composition record may carry.
COMPOSITION_ORIGINS = ("measured", "assumed", "derived", "inferred")

#: The machine presets :func:`resolve_impurity_composition` knows by name.
_MACHINE_PRESETS = ("vest",)

_RECORD_FIELD = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*([^;]*)")


def _element_table() -> Mapping[str, tuple[int, float]]:
    from vaft.data.synthetic_kinetic_profiles import ION_SPECIES

    return ION_SPECIES


@dataclass(frozen=True)
class ImpuritySpecies:
    """One impurity species: an element in one charge state, with its relative density.

    ``charge_state`` is the ion's charge, validated against -- never taken
    from -- the element's atomic number; ``mass`` defaults to the standard
    atomic weight; ``relative_density`` is as configured, not normalised.
    """

    element: str
    charge_state: float
    relative_density: float = 1.0
    mass: Optional[float] = None

    def __post_init__(self) -> None:
        table = _element_table()
        if self.element not in table:
            raise ValueError(f"unknown element {self.element!r}")
        z_n, standard = table[self.element]
        charge = float(self.charge_state)
        if not math.isfinite(charge) or not 0.0 < charge <= z_n:
            raise ValueError(
                f"{self.element}: charge_state must lie in (0, Z_n = {z_n}], got {self.charge_state!r}"
            )
        relative = float(self.relative_density)
        if not math.isfinite(relative) or relative < 0.0:
            raise ValueError(f"{self.element}: relative_density must be finite and non-negative")
        mass = standard if self.mass is None else float(self.mass)
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"{self.element}: mass must be positive")
        object.__setattr__(self, "charge_state", charge)
        object.__setattr__(self, "relative_density", relative)
        object.__setattr__(self, "mass", mass)

    @property
    def z_n(self) -> int:
        """The element's atomic number (not its charge)."""
        return int(_element_table()[self.element][0])

    @property
    def label(self) -> str:
        """``C6+``-style label."""
        return f"{self.element}{self.charge_state:g}+"


@dataclass(frozen=True)
class ImpurityComposition:
    """A relative impurity composition and the effective charge it is closed at.

    ``status`` is what the caller knows about it (``explicit``, ``derived``,
    ``assumed``); ``input_record`` keeps the configuration as given --
    unnormalised densities, notes -- beside the normalised weights.
    """

    species: tuple[ImpuritySpecies, ...]
    target_zeff: Optional[float]
    main_ion: str = "H"
    reduction: str = "preserve_charge_z2_mass"
    status: str = "explicit"
    source: str = "caller"
    input_record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        species = tuple(self.species)
        if not species:
            raise ValueError("an impurity composition needs at least one species")
        if not all(isinstance(item, ImpuritySpecies) for item in species):
            raise TypeError("species must be ImpuritySpecies")
        keys = [(item.element, item.charge_state) for item in species]
        if len(set(keys)) != len(keys):
            raise ValueError("a species (element, charge state) appears twice")
        if sum(item.relative_density for item in species) <= 0.0:
            raise ValueError("relative densities sum to zero")
        if self.main_ion not in _element_table():
            raise ValueError(f"unknown main ion {self.main_ion!r}")
        if self.target_zeff is not None:
            target = float(self.target_zeff)
            if not math.isfinite(target) or target < 1.0:
                raise ValueError(f"target_zeff must be finite and at least 1, got {self.target_zeff!r}")
            object.__setattr__(self, "target_zeff", target)
        if self.reduction != "preserve_charge_z2_mass":
            raise ValueError(f"reduction must be 'preserve_charge_z2_mass', got {self.reduction!r}")
        if self.status not in COMPOSITION_KINDS:
            raise ValueError(f"status must be one of {COMPOSITION_KINDS}, got {self.status!r}")
        object.__setattr__(self, "species", species)

    @property
    def weights(self) -> np.ndarray:
        """Relative densities normalised to sum to one."""
        raw = np.array([item.relative_density for item in self.species], dtype=float)
        return raw / raw.sum()

    @property
    def charges(self) -> np.ndarray:
        return np.array([item.charge_state for item in self.species], dtype=float)

    @property
    def masses(self) -> np.ndarray:
        return np.array([item.mass for item in self.species], dtype=float)

    @property
    def main_ion_charge(self) -> float:
        return float(_element_table()[self.main_ion][0])


@dataclass(frozen=True)
class ResolvedImpurityComposition:
    """The composition that applies, its closure, its reduction and why it was chosen.

    Fractions are ``n / n_e``; arrays run over ``rho`` (when the composition
    or its target is a profile) with species on the last axis.  ``kind`` is
    one of :data:`COMPOSITION_KINDS`; ``candidates`` lists every source that
    was looked at, in precedence order, and what became of it.

    ``rho`` is ``rho_tor_norm`` of the ``core_profiles`` grid the composition
    was resolved on.  ``mean_charge`` / ``mean_square_charge`` are <Z>(rho) and
    <Z^2>(rho) per species when the composition was read from bundled ions
    (``z_ion_1d`` / ``z_ion_square_1d``); ``None`` for fixed charge states, where
    ``species[].charge_state`` applies at every point.  For a bundled ion
    ``species[].charge_state`` is the stored scalar ``z_ion`` (a density-weighted
    mean), a label and not the charge used in Z_eff, S1, S2 or the dilution.
    """

    kind: str
    source: str
    species: tuple[ImpuritySpecies, ...]
    main_ion: str
    weights: np.ndarray
    impurity_fractions: np.ndarray
    main_ion_fraction: Union[float, np.ndarray]
    zeff: Union[float, np.ndarray]
    zeff_source: str
    S1: Union[float, np.ndarray]
    S2: Union[float, np.ndarray]
    A_bar: Union[float, np.ndarray]
    effective_charge: Union[float, np.ndarray]
    effective_fraction: Union[float, np.ndarray]
    effective_mass: Union[float, np.ndarray]
    dilution_fraction: Union[float, np.ndarray]
    rho: Optional[np.ndarray] = None
    time: Optional[float] = None
    resistive_zeff: Optional[Mapping[str, Any]] = None
    candidates: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    mean_charge: Optional[np.ndarray] = None
    mean_square_charge: Optional[np.ndarray] = None

    def fraction(self, element: str, charge_state: Optional[float] = None) -> Union[float, np.ndarray]:
        """``n / n_e`` of one species (summed over charge states when none is named)."""
        picks = [
            k for k, item in enumerate(self.species)
            if item.element == element and (charge_state is None or item.charge_state == float(charge_state))
        ]
        if not picks:
            raise KeyError(f"no species {element}{'' if charge_state is None else charge_state}")
        total = np.sum(np.asarray(self.impurity_fractions)[..., picks], axis=-1)
        return float(total) if np.ndim(total) == 0 else total

    def as_record(self) -> dict[str, Any]:
        """A JSON-ready summary (scalars as floats, profiles as lists)."""
        def plain(value):
            if value is None:
                return None
            array = np.asarray(value, dtype=float)
            return float(array) if array.ndim == 0 else array.tolist()

        return {
            "kind": self.kind,
            "source": self.source,
            "main_ion": self.main_ion,
            "species": [
                {"element": s.element, "charge_state": s.charge_state, "mass": s.mass,
                 "relative_density": s.relative_density}
                for s in self.species
            ],
            "weights": plain(self.weights),
            "impurity_fractions": plain(self.impurity_fractions),
            "main_ion_fraction": plain(self.main_ion_fraction),
            "zeff": plain(self.zeff),
            "zeff_source": self.zeff_source,
            "S1": plain(self.S1),
            "S2": plain(self.S2),
            "A_bar": plain(self.A_bar),
            "effective_charge": plain(self.effective_charge),
            "effective_fraction": plain(self.effective_fraction),
            "effective_mass": plain(self.effective_mass),
            "dilution_fraction": plain(self.dilution_fraction),
            "rho": plain(self.rho),
            "mean_charge": plain(self.mean_charge),
            "mean_square_charge": plain(self.mean_square_charge),
            "time": self.time,
            "resistive_zeff": None if self.resistive_zeff is None else dict(self.resistive_zeff),
            "candidates": [dict(item) for item in self.candidates],
            "provenance": dict(self.provenance),
        }


# --- constructors ----------------------------------------------------------------


def composition_from_fractions(
    elements: Sequence[str],
    fractions: Sequence[float],
    charge_states: Sequence[float],
    *,
    target_zeff: Optional[float],
    masses: Optional[Sequence[float]] = None,
    main_ion: str = "H",
    status: str = "explicit",
    source: str = "caller",
) -> ImpurityComposition:
    """An impurity composition from parallel lists of elements, relative fractions and charge states.

    The form issue #1565 Sec. 2 names (``species=["C", "O", "N"]``,
    ``fractions=[0.3, 0.4, 0.3]``) with the one addition it requires: the
    charge state of each species is stated, never read off the atomic number.

    Parameters
    ----------
    elements : sequence of str
        Element symbols [-].
    fractions : sequence of float
        Relative particle densities, non-negative; normalised internally, the
        given values kept in ``input_record`` [-].
    charge_states : sequence of float
        Ionic charge of each species, in ``(0, Z_n]`` [-].
    target_zeff : float or None
        Plasma effective charge to close the composition at [-].
    masses : sequence of float, optional
        Species masses; default the standard atomic weights [u].
    main_ion : str, optional
        Main-ion element, default ``"H"`` [-].
    status : str, optional
        One of :data:`COMPOSITION_KINDS`, default ``"explicit"`` [-].
    source : str, optional
        Where the composition came from, for provenance [-].

    Returns
    -------
    ImpurityComposition
        The validated composition [any].

    Raises
    ------
    ValueError
        Lists of unequal length, an unknown element, a charge state outside
        ``(0, Z_n]``, negative fractions or a bad status.

    Applicability
    -------------
    Machine-independent.
    """
    n = len(elements)
    if len(fractions) != n or len(charge_states) != n or (masses is not None and len(masses) != n):
        raise ValueError("elements, fractions, charge_states (and masses) must have equal lengths")
    species = tuple(
        ImpuritySpecies(
            element=str(elements[k]),
            charge_state=float(charge_states[k]),
            relative_density=float(fractions[k]),
            mass=None if masses is None else float(masses[k]),
        )
        for k in range(n)
    )
    record = {"elements": list(elements), "fractions": [float(v) for v in fractions],
              "charge_states": [float(v) for v in charge_states]}
    return ImpurityComposition(species=species, target_zeff=target_zeff, main_ion=main_ion,
                               status=status, source=source, input_record=record)


def composition_from_model(model: Mapping[str, Any], *, source: Optional[str] = None) -> ImpurityComposition:
    """An impurity composition from a parsed machine ``impurity_model`` record.

    Parameters
    ----------
    model : mapping
        A record as :func:`vaft.machine_mapping.core_profiles.vest_impurity_model`
        returns it: ``status``, ``species`` (element, charge_state, mass,
        relative_density), ``target_zeff``, ``reduction``, ``main_ion`` and
        ``provenance`` [any].
    source : str, optional
        Provenance label; default the record's own ``provenance.source`` [-].

    Returns
    -------
    ImpurityComposition
        The composition, with ``status`` ``"assumed"`` for an ``assumed``
        preset (any other preset status is kept as ``derived``) [any].

    Raises
    ------
    ValueError
        A malformed record (as :class:`ImpuritySpecies` validates it).

    Applicability
    -------------
    Machine-independent.
    """
    species = tuple(
        ImpuritySpecies(
            element=str(item["element"]),
            charge_state=float(item["charge_state"]),
            relative_density=float(item["relative_density"]),
            mass=float(item["mass"]) if item.get("mass") is not None else None,
        )
        for item in model["species"]
    )
    provenance = dict(model.get("provenance") or {})
    status = "assumed" if model.get("status") == "assumed" else "derived"
    return ImpurityComposition(
        species=species,
        target_zeff=model.get("target_zeff"),
        main_ion=str(model.get("main_ion", "H")),
        reduction=str(model.get("reduction", "preserve_charge_z2_mass")),
        status=status,
        source=source or str(provenance.get("source", "machine preset")),
        input_record={"model_status": model.get("status"), **provenance},
    )


# --- the provenance grammar -------------------------------------------------------


def composition_record_text(origin: str, method: str, **fields: Any) -> str:
    """The record a writer stores beside a composition or Z_eff it put into ``core_profiles``.

    Parameters
    ----------
    origin : str
        One of :data:`COMPOSITION_ORIGINS` [-].
    method : str
        How the value was obtained, e.g. ``impurity_model_preset``; further
        keyword arguments are appended as ``key=value`` fields, in order [-].

    Returns
    -------
    str
        ``origin=<origin>; method=<method>[; key=value...]`` [-].

    Raises
    ------
    ValueError
        An unknown origin, or a ``;`` or ``=`` inside a value.

    Applicability
    -------------
    Machine-independent.
    """
    if origin not in COMPOSITION_ORIGINS:
        raise ValueError(f"origin must be one of {COMPOSITION_ORIGINS}, got {origin!r}")
    parts = [f"origin={origin}", f"method={method}"]
    for key, value in fields.items():
        text = f"{value:g}" if isinstance(value, float) else str(value)
        if ";" in text or "=" in text:
            raise ValueError(f"field {key} carries a separator: {text!r}")
        parts.append(f"{key}={text}")
    return "; ".join(parts)


def composition_record_origin(record: Any) -> Optional[str]:
    """The ``origin`` a composition record states, or ``None`` when it states none.

    Parameters
    ----------
    record : str or None
        A ``*_fit.parameters`` record [-].

    Returns
    -------
    str or None
        One of :data:`COMPOSITION_ORIGINS`, ``"unknown"`` for an ``origin``
        outside them, or ``None`` for an absent record or one without an
        ``origin`` field (an unlabelled value) [-].

    Applicability
    -------------
    Machine-independent.
    """
    if record is None:
        return None
    for part in str(record).replace("\n", ";").split(";"):
        match = _RECORD_FIELD.fullmatch(part)
        if match and match.group(1).lower() == "origin":
            origin = match.group(2).strip().lower()
            return origin if origin in COMPOSITION_ORIGINS else "unknown"
    return None


# --- reading core_profiles without materializing it ---------------------------------


def _get(ods: Any, path: str) -> Any:
    from vaft.ods_access import path_value

    return path_value(ods, path, None)


def _slice_index(ods: Any, time: Optional[float], tolerance: float) -> Optional[int]:
    """The ``profiles_1d`` slice nearest ``time``, by each slice's own time.

    ``profiles_1d.k.time`` decides; ``core_profiles.time`` only fills a slice
    that carries none, and only when it has one entry per slice -- a stale or
    longer homogeneous vector must not point the match at the wrong slice or
    past the array.
    """
    from vaft.ods_access import path_count

    count = path_count(ods, "core_profiles.profiles_1d")
    if count == 0:
        return None
    own = [_get(ods, f"core_profiles.profiles_1d.{k}.time") for k in range(count)]
    homogeneous = _get(ods, "core_profiles.time")
    homogeneous = None if homogeneous is None else np.atleast_1d(np.asarray(homogeneous, dtype=float))
    if homogeneous is not None and homogeneous.size != count:
        homogeneous = None
    times = np.array([
        float(t) if t is not None else (homogeneous[k] if homogeneous is not None else np.nan)
        for k, t in enumerate(own)
    ], dtype=float)
    if time is None:
        if count == 1:
            return 0
        raise ValueError("core_profiles holds several slices; pass the time to read")
    if not np.isfinite(times).any():
        return None
    index = int(np.nanargmin(np.abs(times - float(time))))
    return index if abs(times[index] - float(time)) <= tolerance else None


def _slice_time(ods: Any, index: int) -> Optional[float]:
    own = _get(ods, f"core_profiles.profiles_1d.{index}.time")
    if own is not None:
        return float(own)
    times = _get(ods, "core_profiles.time")
    return None if times is None else float(np.atleast_1d(times)[index])


def _hydrogenic(z_n: Any) -> bool:
    return z_n is not None and int(round(float(z_n))) == 1


def _states_mean_square_charge(ods: Any, ion: str, density: np.ndarray) -> Optional[np.ndarray]:
    """<Z^2>(rho) of a bundled ion from its ``state[]`` densities, or None when they cannot give it."""
    from vaft.ods_access import path_count

    n_states = path_count(ods, f"{ion}.state")
    if n_states == 0:
        return None
    total = np.zeros(density.shape)
    for q in range(n_states):
        state = f"{ion}.state.{q}"
        lo, hi = _get(ods, f"{state}.z_min"), _get(ods, f"{state}.z_max")
        n_q = _get(ods, f"{state}.density_thermal")
        if n_q is None:
            n_q = _get(ods, f"{state}.density")
        if lo is None or hi is None or n_q is None or float(lo) != float(hi):
            return None       # a state that is itself a bundle carries no single charge
        total = total + float(lo) ** 2 * np.asarray(n_q, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        # the writer's convention: ``density`` sums every charge state incl. the neutrals
        return np.where(density > 0.0, total / density, np.nan)


def _ion_charge(ods: Any, ion: str, density: np.ndarray):
    """The charge of one stored impurity ion: the scalar ``z_ion``, or the radial moments of a bundled ion.

    Returns ``(z_ion, mean, mean_square, fields)``.  ``mean`` / ``mean_square`` are
    <Z>(rho) / <Z^2>(rho) when the entry is bundled (``z_ion_1d`` present), else
    ``None``; ``fields`` names what was read.  ``z_ion`` is the stored scalar or,
    absent, the density-weighted mean of <Z>(rho).  All ``None`` when the entry
    stores no charge at all.
    """
    z_ion = _get(ods, f"{ion}.z_ion")
    mean = _get(ods, f"{ion}.z_ion_1d")
    if mean is None:
        return (None, None, None, None) if z_ion is None else (float(z_ion), None, None, "z_ion")
    mean = np.asarray(mean, dtype=float)
    fields = ["z_ion_1d"]
    mean2 = _get(ods, f"{ion}.z_ion_square_1d")
    if mean2 is not None:
        mean2 = np.asarray(mean2, dtype=float)
        fields.append("z_ion_square_1d")
    else:
        mean2 = _states_mean_square_charge(ods, ion, density) if mean.shape == density.shape else None
        if mean2 is not None:
            fields.append("state[].density for <Z^2>")
        else:
            mean2 = mean**2
            fields.append("<Z^2> = <Z>^2 (no z_ion_square_1d, no state[])")
    if z_ion is None:
        finite = np.isfinite(mean) & np.isfinite(density) & (density > 0.0)
        if mean.shape != density.shape or not finite.any():
            return None, None, None, None
        z_ion = float(np.sum((density * mean)[finite]) / np.sum(density[finite]))
        fields.append("z_ion = density-weighted <Z>")
    return float(z_ion), mean, mean2, " + ".join(fields)


def _ods_impurities(ods: Any, index: int) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Impurity ions of one ``core_profiles`` slice, their fractions and labels, or why there are none.

    Every ``ion[]`` entry is visited (counted, not scanned until a gap).  The
    main ion is the hydrogen isotopes; a slice whose ions are not hydrogenic
    plus impurities is refused rather than read with a hydrogen main ion it
    does not have.

    A bundled ion (``z_ion_1d`` present, as :func:`populate_radial_impurity_profiles`
    writes it) is read with its radial moments <Z>(rho) and <Z^2>(rho); its scalar
    ``z_ion`` is only the species label.  ``charge_moments`` is ``None`` when every
    ion carries a fixed charge state, else the ``(mean, mean_square)`` arrays over
    ``(rho, species)``; ``charge_fields`` records what each ion was read from.
    """
    from vaft.ods_access import path_count

    base = f"core_profiles.profiles_1d.{index}"
    n_ions = path_count(ods, f"{base}.ion")
    if n_ions == 0:
        return None, None
    ne = _get(ods, f"{base}.electrons.density_thermal")
    if ne is None:
        ne = _get(ods, f"{base}.electrons.density")
    if ne is None:
        return None, "ion densities stored but no electron density to divide by"
    ne = np.asarray(ne, dtype=float)
    rho = _get(ods, f"{base}.grid.rho_tor_norm")
    table = _element_table()
    by_charge = {z: sym for sym, (z, _) in table.items() if sym not in ("D", "T")}
    species, densities, origins, skipped, moments, charge_fields = [], [], [], [], [], {}
    has_main = False
    for k in range(n_ions):
        ion = f"{base}.ion.{k}"
        z_n = _get(ods, f"{ion}.element.0.z_n")
        if _hydrogenic(z_n):
            has_main = True
            continue
        density = _get(ods, f"{ion}.density_thermal")
        if density is None:
            density = _get(ods, f"{ion}.density")
        density = None if density is None else np.asarray(density, dtype=float)
        z_ion = mean = mean2 = fields = None
        if density is not None:
            z_ion, mean, mean2, fields = _ion_charge(ods, ion, density)
        if z_n is None or z_ion is None or density is None:
            skipped.append(f"ion.{k}: missing {'element.0.z_n' if z_n is None else 'density' if density is None else 'z_ion'}")
            continue
        element = by_charge.get(int(round(float(z_n))))
        if element is None:
            skipped.append(f"ion.{k}: unknown element z_n={z_n}")
            continue
        if mean is not None and (mean.shape != ne.shape or mean2.shape != ne.shape):
            skipped.append(f"ion.{k}: z_ion_1d has {mean.size} points, the electron density {ne.size}")
            continue
        mass = _get(ods, f"{ion}.element.0.a")
        species.append(ImpuritySpecies(element=element, charge_state=float(z_ion),
                                       mass=None if mass is None else float(mass)))
        densities.append(density)
        moments.append((mean, mean2))
        charge_fields[f"ion.{k}"] = fields
        origins.append(composition_record_origin(_get(ods, f"{ion}.density_fit.parameters")))
    if not species:
        return None, "; ".join(skipped) or None
    if not has_main:
        return None, "no hydrogenic main ion among the stored ions; a non-hydrogenic main ion is not read"
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_ne = np.where(np.isfinite(ne) & (ne > 0.0), ne, np.nan)
        fractions = np.stack([np.broadcast_to(d, ne.shape) / safe_ne for d in densities], axis=-1)
    charge_moments = None
    if any(mean is not None for mean, _ in moments):
        # a fixed-charge ion beside a bundled one keeps its charge at every point
        charge_moments = (
            np.stack([np.broadcast_to(s.charge_state if m is None else m, ne.shape)
                      for (m, _), s in zip(moments, species)], axis=-1),
            np.stack([np.broadcast_to(s.charge_state**2 if m2 is None else m2, ne.shape)
                      for (_, m2), s in zip(moments, species)], axis=-1),
        )
    distinct = set(origins)
    origin = origins[0] if len(distinct) == 1 else "mixed"
    return ({"species": tuple(species), "fractions": fractions, "skipped": skipped,
             "rho": None if rho is None else np.asarray(rho, dtype=float), "origin": origin,
             "charge_moments": charge_moments, "charge_fields": charge_fields}, None)


def _ods_measured_zeff(ods: Any, index: int) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
    base = f"core_profiles.profiles_1d.{index}"
    zeff = _get(ods, f"{base}.zeff")
    if zeff is None:
        return None
    if composition_record_origin(_get(ods, f"{base}.zeff_fit.parameters")) != "measured":
        return None
    rho = _get(ods, f"{base}.grid.rho_tor_norm")
    return np.asarray(zeff, dtype=float), None if rho is None else np.asarray(rho, dtype=float)


def _resistive_record(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        record = dict(value)
        zeff = record.get("zeff", record.get("value", record.get("estimate")))
    else:
        record, zeff = {}, value
    zeff = float(zeff)
    if not math.isfinite(zeff):
        raise ValueError("resistive_zeff must be finite")
    record.update({"value": zeff, "kind": "resistive_inferred",
                   "note": "carried beside the composition; never a target or a composition (#1566)"})
    return record


# --- the resolver -----------------------------------------------------------------------


def _close(species, main_ion, main_charge, weights, target, *, kind, source, zeff_source, rho, time,
           resistive, candidates, provenance, charge_moments=None) -> ResolvedImpurityComposition:
    """Close relative weights at a target Z_eff by quasi-neutrality.

    A scalar target outside the reachable interval is an error.  A profile
    target (a measured Z_eff) is closed point by point: a point that is not
    finite or lies outside ``[Z_m, S2/S1]`` is left undefined (NaN) and
    counted in the provenance, rather than failing the slice.  With
    ``charge_moments`` (<Z>, <Z^2> per point and species, bundled ions) the
    closure uses the per-point moments ``S1 = sum w <Z>``, ``S2 = sum w <Z^2>``.
    """
    charges = np.array([s.charge_state for s in species], dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    provenance = dict(provenance)
    if charge_moments is None and target.ndim == 0 and weights.ndim == 1:
        solution = solve_impurity_mixture_for_target_zeff(target, weights, charges, main_ion_charge=main_charge)
        fractions = np.asarray(solution.impurity_fractions, dtype=float)
    else:
        if charge_moments is None:
            shape = np.broadcast_shapes(target.shape, weights.shape[:-1])
        else:
            shape = np.broadcast_shapes(target.shape, weights.shape[:-1], np.shape(charge_moments[0])[:-1])
        t = np.broadcast_to(target, shape).reshape(-1)
        w = np.broadcast_to(weights, shape + weights.shape[-1:]).reshape(-1, weights.shape[-1])
        if charge_moments is None:
            z1 = np.broadcast_to(charges, w.shape)
            z2 = z1**2
        else:
            z1 = np.broadcast_to(np.asarray(charge_moments[0], dtype=float), shape + w.shape[-1:]).reshape(w.shape)
            z2 = np.broadcast_to(np.asarray(charge_moments[1], dtype=float), shape + w.shape[-1:]).reshape(w.shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            s1, s2 = np.sum(w * z1, axis=-1), np.sum(w * z2, axis=-1)
            ok = (np.isfinite(t) & np.all(np.isfinite(w), axis=-1) & np.all(w >= 0.0, axis=-1)
                  & np.all(np.isfinite(z1), axis=-1) & np.all(np.isfinite(z2), axis=-1)
                  & (s1 > 0.0) & (s2 - main_charge * s1 > 0.0)
                  & (t >= main_charge - 1e-9) & (t <= s2 / s1 * (1.0 + 1e-9)))
        flat = np.full(w.shape, np.nan)
        if ok.any() and charge_moments is None:
            flat[ok] = solve_impurity_mixture_for_target_zeff(
                t[ok], w[ok], charges, main_ion_charge=main_charge
            ).impurity_fractions
        elif ok.any():
            # the same closure as solve_impurity_mixture_for_target_zeff, with the
            # per-point moments of the bundled ions: n_s/n_e = w_s (Z_eff - Z_m)/(S2 - Z_m S1)
            alpha = (t[ok] - main_charge) / (s2[ok] - main_charge * s1[ok])
            flat[ok] = np.clip(alpha, 0.0, None)[:, None] * w[ok]
        fractions = flat.reshape(shape + weights.shape[-1:])
        target = np.where(ok, t, np.nan).reshape(shape)
        provenance["undefined_points"] = int(np.count_nonzero(~ok))
    return _finish(species, main_ion, main_charge, fractions, target, kind=kind, source=source,
                   zeff_source=zeff_source, rho=rho, time=time, resistive=resistive,
                   candidates=candidates, provenance=provenance, charge_moments=charge_moments)


def _finish(species, main_ion, main_charge, fractions, zeff, *, kind, source, zeff_source, rho, time,
            resistive, candidates, provenance, charge_moments=None) -> ResolvedImpurityComposition:
    """Moments, reduction and dilution of resolved fractions; undefined points stay NaN.

    A point is undefined where ``n_e`` is zero or missing (the edge point of
    a fitted profile is often exactly zero), where a stored fraction is
    negative, or where the impurities carry more charge than there are
    electrons: everything formed there is NaN, rather than an error for the
    whole slice or a number made up for that point.

    ``charge_moments`` -- ``(<Z>, <Z^2>)`` arrays shaped like ``fractions`` --
    are the per-point charges of bundled ions; with them every quantity uses
    <Z> where a fixed charge state uses Z and <Z^2> where it uses Z^2
    (:func:`vaft.formula.impurity.impurity_mixture_moments` Convention), and
    ``species[].charge_state`` is only the label.
    """
    charges = np.array([s.charge_state for s in species], dtype=float)
    masses = np.array([s.mass for s in species], dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    shape = fractions.shape[:-1]
    flat = fractions.reshape(-1, fractions.shape[-1])
    if charge_moments is None:
        z1 = np.broadcast_to(charges, flat.shape)
        z2 = z1**2
    else:
        z1 = np.asarray(charge_moments[0], dtype=float).reshape(-1, flat.shape[-1])
        z2 = np.asarray(charge_moments[1], dtype=float).reshape(-1, flat.shape[-1])
        if z1.shape != flat.shape or z2.shape != flat.shape:
            raise ValueError("the charge moments must be shaped like the fractions (rho, species)")
    with np.errstate(invalid="ignore"):
        defined = (np.all(np.isfinite(flat), axis=-1) & np.all(flat >= 0.0, axis=-1)
                   & np.all(np.isfinite(z1), axis=-1) & np.all(np.isfinite(z2), axis=-1))
        main = np.where(defined, (1.0 - np.sum(flat * z1, axis=-1)) / main_charge, np.nan)
        defined &= main >= -1e-12
    main = np.where(defined, np.clip(main, 0.0, None), np.nan)
    charge_density = np.sum(np.where(defined[:, None], flat * z1, 0.0), axis=-1)
    ok = defined & (charge_density > 0.0)
    no_impurity = defined & ~ok

    def blank():
        return np.full(ok.shape, np.nan)

    weights = np.full(flat.shape, np.nan)
    s1, s2, a_bar = blank(), blank(), blank()
    eff_charge, eff_fraction, eff_mass = blank(), blank(), blank()
    if ok.any() and charge_moments is None:
        sub = flat[ok]
        w = sub / np.sum(sub, axis=-1, keepdims=True)
        weights[ok] = w
        moments = impurity_mixture_moments(w, charges, masses)
        s1[ok], s2[ok], a_bar[ok] = moments.S1, moments.S2, moments.A_bar
        effective = reduce_impurity_mixture(sub, charges, masses)
        eff_charge[ok], eff_fraction[ok], eff_mass[ok] = effective.charge, effective.density, effective.mass
    elif ok.any():
        # impurity_mixture_moments / reduce_impurity_mixture with the per-point
        # <Z>, <Z^2> of the bundled ions in place of Z, Z^2 (an element that is
        # neutral at a point contributes <Z> = 0 there, which those validators refuse)
        sub = flat[ok]
        w = sub / np.sum(sub, axis=-1, keepdims=True)
        weights[ok] = w
        s1[ok], s2[ok], a_bar[ok] = np.sum(w * z1[ok], axis=-1), np.sum(w * z2[ok], axis=-1), w @ masses
        nz, nz2 = np.sum(sub * z1[ok], axis=-1), np.sum(sub * z2[ok], axis=-1)
        eff_charge[ok] = nz2 / nz
        eff_fraction[ok] = nz**2 / nz2
        eff_mass[ok] = (sub @ masses) / eff_fraction[ok]
    if no_impurity.any():  # a target of exactly Z_m: no impurity, the pseudo-impurity is the mixture's own
        uniform = np.full(len(species), 1.0 / len(species))
        weights[no_impurity] = uniform
        with np.errstate(invalid="ignore", divide="ignore"):
            u1 = np.sum(uniform * z1[no_impurity], axis=-1)
            u2 = np.sum(uniform * z2[no_impurity], axis=-1)
            s1[no_impurity], s2[no_impurity], a_bar[no_impurity] = u1, u2, float(uniform @ masses)
            eff_charge[no_impurity] = u2 / u1
            eff_fraction[no_impurity] = 0.0
            eff_mass[no_impurity] = float(uniform @ masses) * u2 / u1**2
    if zeff is None:
        with np.errstate(invalid="ignore"):
            zeff = np.where(defined, main_charge**2 * main + np.sum(flat * z2, axis=-1), np.nan)

    def shaped(value):
        array = np.asarray(value, dtype=float)
        if array.size == ok.size and array.shape != shape:
            array = array.reshape(shape)
        return float(array) if array.ndim == 0 else array

    return ResolvedImpurityComposition(
        kind=kind, source=source, species=tuple(species), main_ion=main_ion,
        weights=weights.reshape(fractions.shape), impurity_fractions=fractions,
        main_ion_fraction=shaped(main), zeff=shaped(zeff), zeff_source=zeff_source,
        S1=shaped(s1), S2=shaped(s2), A_bar=shaped(a_bar),
        effective_charge=shaped(eff_charge), effective_fraction=shaped(eff_fraction),
        effective_mass=shaped(eff_mass), dilution_fraction=shaped(1.0 - main),
        rho=rho, time=time, resistive_zeff=resistive, candidates=tuple(candidates),
        provenance=dict(provenance),
        mean_charge=None if charge_moments is None else np.asarray(charge_moments[0], dtype=float).reshape(fractions.shape),
        mean_square_charge=None if charge_moments is None else np.asarray(charge_moments[1], dtype=float).reshape(fractions.shape),
    )


def resolve_impurity_composition(
    ods: Any = None,
    *,
    time: Optional[float] = None,
    tolerance: float = 5e-4,
    composition: Optional[ImpurityComposition] = None,
    derived: Optional[ImpurityComposition] = None,
    machine_preset: Union[None, str, Mapping[str, Any]] = None,
    shot: Optional[int] = None,
    use_measured_zeff: bool = True,
    resistive_zeff: Any = None,
    info_file: Optional[str] = None,
) -> ResolvedImpurityComposition:
    """Resolve the impurity composition that applies, close it at its Z_eff and reduce it.

    Parameters
    ----------
    ods : ODS, optional
        Source of a stored ``core_profiles`` composition and Z_eff; read
        without creating any path [any].
    time : float, optional
        Time of the ``core_profiles`` slice to read; required when it holds
        more than one slice [s].
    tolerance : float, optional
        Largest ``|t_slice - time|`` accepted [s].
    composition : ImpurityComposition, optional
        An explicit composition from the caller [any].
    derived : ImpurityComposition, optional
        A composition derived by a stated model (atomic data, a resistive
        closure) [any].
    machine_preset : str or mapping, optional
        ``"vest"`` (the ``vest.yaml`` ``impurity_model`` for ``shot``), or a
        parsed ``impurity_model`` record; used only when nothing above applies [any].
    shot : int, optional
        Shot whose machine preset era applies [-].
    use_measured_zeff : bool, optional
        Close a non-measured composition at a ``core_profiles.zeff`` labelled
        measured instead of the composition's own target [-].
    resistive_zeff : float or mapping, optional
        A resistively inferred Z_eff (a Lane Z row) to carry beside the result
        under its own provenance; never used as a target [-].
    info_file : str, optional
        Alternative ``vest.yaml`` for the ``"vest"`` preset [-].

    Returns
    -------
    ResolvedImpurityComposition
        Species, ``n_s/n_e``, ``n_main/n_e``, Z_eff and where it came from,
        the moments ``S1``/``S2``/``A_bar``, the reduced pseudo-impurity,
        the dilution, the kind and the candidates considered [any].

    Raises
    ------
    ValueError
        No source applies, a composition without a target Z_eff and no
        measured one, a target the composition cannot reach, or a
        ``core_profiles`` slice that cannot be matched in time.

    Processing steps
    ----------------
    1. Match the ``core_profiles`` slice by time (never by index); read its
       impurity ions (``element.0.z_n > 1``) and their ``density_fit``
       origin, and its ``zeff`` if labelled ``origin=measured``.  A bundled
       ion is read with its radial charge (``z_ion_1d``, ``z_ion_square_1d``,
       else ``state[]``); ``z_ion`` alone is read as a fixed charge state.
       ``provenance["charge_fields"]`` records which fields each ion gave.
    2. Choose the composition: stored ions labelled measured; else the
       explicit argument; else the derived argument, or stored ions labelled
       derived/inferred; else stored ions labelled assumed or unlabelled;
       else the machine preset.
    3. A stored composition labelled measured is used as stored.  Any other
       is closed by quasi-neutrality at the measured Z_eff profile when there
       is one and ``use_measured_zeff`` (a stored one keeps its relative
       weights per rho), else at its own target (a stored one: as stored).
       A measured-Z_eff point outside ``[Z_m, S2/S1]`` or not finite is left
       undefined and counted in ``provenance["undefined_points"]``.
    4. Reduce to the pseudo-impurity and form the dilution.

    Convention
    ----------
    The precedence measured > explicit > derived > assumed is #1565 Sec. 4.
    A stored composition that does not say how it is known ranks with the
    assumed ones, below an explicit argument: the Lane K products carry an
    H+/C6+ list assumed at Z_eff = 2 with no density label, and reading
    that as a measurement would let an assumption outrank the caller.

    Applicability
    -------------
    Machine-independent.  The ``"vest"`` preset reads ``vest.yaml``
    ``diagnostics.core_profiles.impurity_model`` when, and only when, it is asked for.

    Limitations
    -----------
    An argument composition has fixed charge states per species; a radially
    varying charge-state distribution enters as a ``derived`` composition
    (#1565 Sec. 8) or is read back from the bundled ions
    :func:`populate_radial_impurity_profiles` wrote (``mean_charge`` /
    ``mean_square_charge`` of the result).  A stored
    composition is read only with a hydrogenic main ion; a slice whose ions
    are not hydrogen isotopes plus impurities is skipped with its reason.

    Provenance
    ----------
    .. [issue] #1565 Sec. 3-4 (presets, precedence) and its main-ion
               dilution comment; #1566 (resistive Z_eff kept apart).
    """
    resistive = _resistive_record(resistive_zeff)
    candidates: list[dict[str, Any]] = []
    stored = measured_zeff = None
    slice_time = None
    if ods is not None:
        index = _slice_index(ods, time, tolerance)
        if index is None:
            candidates.append({"source": "core_profiles", "outcome": "skipped",
                               "reason": "no slice within tolerance of the requested time"})
        else:
            slice_time = _slice_time(ods, index)
            stored, why = _ods_impurities(ods, index)
            if why is not None:
                candidates.append({"source": "core_profiles.ion", "outcome": "skipped", "reason": why})
            measured_zeff = _ods_measured_zeff(ods, index) if use_measured_zeff else None
    common = {"time": slice_time, "resistive": resistive}

    arguments = [("explicit", composition), ("derived", derived), ("machine_preset", machine_preset)]

    def note_arguments(chosen_label: Optional[str]) -> None:
        for label, value in arguments:
            if value is not None:
                candidates.append({"source": label,
                                   "outcome": "chosen" if label == chosen_label else "outranked"})

    def from_store(kind: str) -> ResolvedImpurityComposition:
        label = stored["origin"] or "unlabelled"
        candidates.append({"source": "core_profiles.ion", "outcome": "chosen", "origin": label})
        note_arguments(None)
        record = {"method": "stored ion densities", "charge_fields": dict(stored["charge_fields"])}
        if stored["charge_moments"] is not None:
            record["charge"] = "bundled ions: <Z>(rho), <Z^2>(rho) from the radial charge fields"
        if stored["skipped"]:
            record["skipped_ions"] = list(stored["skipped"])
        if kind != "measured" and measured_zeff is not None:
            # measured outranks a stored derived or assumed composition: keep its
            # relative weights per rho, close them at the measured Z_eff
            target, rho = measured_zeff
            if np.shape(target) == stored["fractions"].shape[:-1]:
                with np.errstate(invalid="ignore", divide="ignore"):
                    weights = stored["fractions"] / np.sum(stored["fractions"], axis=-1, keepdims=True)
                record.update(method="stored relative weights closed at the measured Z_eff",
                              stored_origin=label)
                return _close(stored["species"], "H", 1.0, weights, target, kind="derived",
                              source=f"core_profiles.ion ({label}) closed at measured core_profiles.zeff",
                              zeff_source="core_profiles.zeff (origin=measured)", rho=rho,
                              candidates=candidates, provenance=record,
                              charge_moments=stored["charge_moments"], **common)
            candidates.append({"source": "core_profiles.zeff", "outcome": "skipped",
                               "reason": "measured Z_eff is not on the ion-density grid"})
        return _finish(stored["species"], "H", 1.0, stored["fractions"], None, kind=kind,
                       source=f"core_profiles.ion (origin={label})", zeff_source="species",
                       rho=stored["rho"], candidates=candidates, provenance=record,
                       charge_moments=stored["charge_moments"], **common)

    origin = None if stored is None else stored["origin"]
    if origin == "measured":
        return from_store("measured")
    if composition is not None:
        chosen, kind, label = composition, "explicit", "explicit"
    elif derived is not None:
        chosen, kind, label = derived, "derived", "derived"
    elif origin in ("derived", "inferred"):
        return from_store("derived")
    elif stored is not None:
        return from_store("assumed")
    elif machine_preset is not None:
        chosen, kind, label = _preset(machine_preset, shot, info_file), "assumed", "machine_preset"
    else:
        raise ValueError(
            "no impurity composition applies: pass composition=, derived=, or machine_preset= "
            "(a VEST analysis passes machine_preset='vest'); nothing is defaulted"
        )
    if stored is not None:
        candidates.append({"source": "core_profiles.ion", "outcome": "outranked",
                           "origin": origin or "unlabelled"})
    note_arguments(label)
    source = chosen.source

    provenance = {"method": "quasi-neutral closure at the target Z_eff",
                  "composition_status": chosen.status, "input_record": dict(chosen.input_record)}
    closing = (chosen.species, chosen.main_ion, chosen.main_ion_charge, chosen.weights)
    if measured_zeff is not None:
        target, rho = measured_zeff
        provenance["target_zeff_replaced"] = chosen.target_zeff
        return _close(*closing, target, kind="derived", source=f"{source} closed at measured core_profiles.zeff",
                      zeff_source="core_profiles.zeff (origin=measured)", rho=rho,
                      candidates=candidates, provenance=provenance, **common)
    if chosen.target_zeff is None:
        raise ValueError(f"the {kind} composition has no target_zeff and no measured Z_eff is available")
    return _close(*closing, chosen.target_zeff, kind=kind, source=source,
                  zeff_source=f"target_zeff of the {kind} composition", rho=None,
                  candidates=candidates, provenance=provenance, **common)


def _preset(preset: Union[str, Mapping[str, Any]], shot: Optional[int], info_file: Optional[str]) -> ImpurityComposition:
    if isinstance(preset, Mapping):
        return composition_from_model(preset)
    if preset not in _MACHINE_PRESETS:
        raise ValueError(f"unknown machine preset {preset!r}; known: {_MACHINE_PRESETS}")
    from vaft.machine_mapping.core_profiles import vest_impurity_model

    model = vest_impurity_model(shot, info_file=info_file)
    if model is None:
        raise ValueError("vest.yaml configures no impurity_model for this shot era")
    era = "base" if shot is None else f"shot={int(shot)}"
    return composition_from_model(model, source=f"{model['provenance']['source']} ({era})")


# --- writing core_profiles (Stage C) --------------------------------------------------

#: The ``origin`` written for each resolved kind; a composition already measured
#: in ``core_profiles`` is never rewritten.
_WRITE_ORIGIN = {"explicit": "assumed", "assumed": "assumed", "derived": "derived", "inferred": "inferred"}


def _write_slice(ods: Any, resolved: ResolvedImpurityComposition, time: Optional[float],
                 tolerance: float) -> tuple[int, np.ndarray, np.ndarray]:
    if resolved.kind == "measured":
        raise ValueError("this composition is measured in core_profiles already; nothing to write")
    if resolved.kind not in _WRITE_ORIGIN:
        raise ValueError(f"cannot write a composition of kind {resolved.kind!r}")
    if time is not None and resolved.time is not None and abs(float(time) - float(resolved.time)) > tolerance:
        raise ValueError(
            f"the composition was resolved at t = {resolved.time:g} s; writing it into the slice "
            f"at t = {float(time):g} s would pair one slice's composition with another's data"
        )
    target = resolved.time if time is None else time
    index = _slice_index(ods, target, tolerance)
    if index is None:
        raise ValueError(f"no core_profiles slice within {tolerance:g} s of t = {target!r}")
    base = f"core_profiles.profiles_1d.{index}"
    stored, _ = _ods_impurities(ods, index)
    if stored is not None and stored["origin"] == "measured":
        raise ValueError(f"{base} carries an ion composition labelled measured; it is never overwritten")
    ne = _get(ods, f"{base}.electrons.density_thermal")
    if ne is None:
        ne = _get(ods, f"{base}.electrons.density")
    if ne is None:
        raise ValueError(f"{base} has no electron density to scale the composition by")
    ne = np.asarray(ne, dtype=float)
    rho = _get(ods, f"{base}.grid.rho_tor_norm")
    return index, ne, None if rho is None else np.asarray(rho, dtype=float)


def _on_grid(values: Any, resolved: ResolvedImpurityComposition, rho: Optional[np.ndarray], n: int) -> np.ndarray:
    """A resolved quantity on the slice grid: a scalar broadcasts, a profile is interpolated by rho."""
    array = np.asarray(values, dtype=float)
    profile_shape = np.shape(resolved.zeff)
    if array.ndim == 0 or (array.ndim == 1 and not profile_shape):
        return np.broadcast_to(array, (n,) + array.shape).copy()
    if array.shape[0] == n and (resolved.rho is None or rho is None or np.allclose(resolved.rho, rho)):
        return array.copy()
    if resolved.rho is None or rho is None:
        raise ValueError("a resolved profile on another grid needs both rho grids to be interpolated")
    source = np.asarray(resolved.rho, dtype=float)
    if not (np.all(np.isfinite(source)) and np.all(np.diff(source) > 0.0)):
        raise ValueError("the resolved rho grid must be finite and increasing to be interpolated")
    columns = array.reshape(array.shape[0], -1)
    # inside the resolved range only; beyond it the point is undefined and filled below
    # anti-alias: spatial interpolation over rho / r/a, not time -- no sample rate to reduce
    out = np.column_stack([np.interp(rho, source, column, left=np.nan, right=np.nan) for column in columns.T])
    return out.reshape((n,) + array.shape[1:])


def _fill_undefined(values: np.ndarray, rho: Optional[np.ndarray]) -> tuple[np.ndarray, int]:
    """Fill NaN points of a per-rho array (rho on axis 0) from its finite neighbours.

    Linear in rho between defined points and constant beyond the outermost
    one -- a density is never written as NaN over a stored finite value.
    Returns the filled array and how many points were filled.
    """
    out = np.array(values, dtype=float, copy=True)
    flat = out.reshape(out.shape[0], -1)
    undefined = ~np.all(np.isfinite(flat), axis=1)
    if not undefined.any():
        return out, 0
    if undefined.all():
        raise ValueError("the resolved composition is undefined at every point of the slice")
    x = np.arange(flat.shape[0], dtype=float) if rho is None else np.asarray(rho, dtype=float)
    good = ~undefined
    for column in range(flat.shape[1]):
        # anti-alias: spatial interpolation over rho / r/a, not time -- no sample rate to reduce
        flat[undefined, column] = np.interp(x[undefined], x[good], flat[good, column])
    return out, int(np.count_nonzero(undefined))


def _record(resolved: ResolvedImpurityComposition, quantity: str, method: Optional[str]) -> str:
    origin = _WRITE_ORIGIN[resolved.kind]
    return composition_record_text(
        origin, method or f"{resolved.kind}_composition", quantity=quantity,
        zeff_source=resolved.zeff_source.replace("=", ":").replace(";", ","),
        source=resolved.source.replace("=", ":").replace(";", ","),
    )


def _reset_ions(work: Any, base: str, main_ion: str) -> None:
    """Keep the slice's one hydrogenic main ion whole as ``ion.0``, drop every other entry.

    Refuses a non-hydrogenic requested main ion, a slice with two hydrogenic
    ions, or a slice whose ions include no hydrogenic one.  A malformed scalar
    ``ion`` leaf is replaced.  Writes ``H+`` when the slice has no ion at all.
    """
    from vaft.ods_access import path_count

    if not _hydrogenic(_element_table()[main_ion][0]):
        raise ValueError(f"only a hydrogenic main ion is written, not {main_ion!r}")
    hydrogenic, others = [], []
    for k in range(path_count(work, f"{base}.ion")):
        z_n = _get(work, f"{base}.ion.{k}.element.0.z_n")
        (hydrogenic if _hydrogenic(z_n) else others).append(k)
    if len(hydrogenic) > 1:
        raise ValueError(f"{base} stores {len(hydrogenic)} hydrogenic ions; one main ion is written, not a mix")
    if others and not hydrogenic:
        raise ValueError(f"{base} stores ions but no hydrogenic main ion; a non-hydrogenic main ion is refused")
    main_node = copy.deepcopy(work[f"{base}.ion.{hydrogenic[0]}"]) if hydrogenic else None
    if "ion" in work[base].keys():
        # also a malformed non-array ``ion`` leaf: vaft.omas.load gives the campaign
        # FileDB products a bare NaN ``profiles_1d.N.ion``, which cannot take entries
        del work[f"{base}.ion"]
    if main_node is not None:
        work[f"{base}.ion.0"] = main_node
        if "density_fit" in work[f"{base}.ion.0"].keys():
            del work[f"{base}.ion.0.density_fit"]   # its measured/reconstructed arrays describe the old density
    else:
        work[f"{base}.ion.0.label"] = "H+"
        work[f"{base}.ion.0.z_ion"] = 1.0
        work[f"{base}.ion.0.element.0.z_n"] = 1.0
        work[f"{base}.ion.0.element.0.a"] = float(_element_table()["H"][1])
        work[f"{base}.ion.0.element.0.atoms_n"] = 1


def populate_zeff_profile(
    ods: Any,
    resolved: ResolvedImpurityComposition,
    *,
    time: Optional[float] = None,
    tolerance: float = 5e-4,
    method: Optional[str] = None,
) -> Any:
    """A copy of ``ods`` whose ``core_profiles`` slice carries the resolved Z_eff(rho).

    Parameters
    ----------
    ods : ODS
        Source; never modified [any].
    resolved : ResolvedImpurityComposition
        From :func:`resolve_impurity_composition`; not of kind ``measured`` [any].
    time : float, optional
        Slice time; default the resolved composition's own [s].
    tolerance : float, optional
        Largest ``|t_slice - time|`` accepted [s].
    method : str, optional
        ``method=`` field of the record; default ``<kind>_composition`` [-].

    Returns
    -------
    ODS
        A deep copy with ``profiles_1d.zeff`` and ``profiles_1d.zeff_fit.parameters``
        (``origin=assumed|derived; method=...``) written [any].

    Raises
    ------
    ValueError
        A measured composition (already in ``core_profiles``), no slice at the
        time, or no electron density.

    Convention
    ----------
    The record's ``origin`` is ``assumed`` for an explicit or preset
    composition (Z_eff is then the assumed target) and ``derived`` for one
    computed from other data; :func:`resolve_impurity_composition` never reads
    either back as a measured target.  A measured ``zeff`` is never
    overwritten: a composition closed at it is left with the slice's own
    measured profile and record.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [issue] #1565 Sec. 5 (core_profiles.zeff from the resolved composition).
    """
    work = copy.deepcopy(ods)
    index, ne, rho = _write_slice(work, resolved, time, tolerance)
    base = f"core_profiles.profiles_1d.{index}"
    if "measured" in resolved.zeff_source:
        return work  # the slice already carries the measured profile it was closed at
    if composition_record_origin(_get(work, f"{base}.zeff_fit.parameters")) == "measured":
        raise ValueError(f"{base}.zeff is labelled measured; it is never overwritten")
    zeff, filled = _fill_undefined(_on_grid(resolved.zeff, resolved, rho, ne.size), rho)
    work[f"{base}.zeff"] = zeff
    record = _record(resolved, "zeff", method)
    work[f"{base}.zeff_fit.parameters"] = record + (f"; filled_points={filled}" if filled else "")
    return work


def populate_impurity_profiles(
    ods: Any,
    resolved: ResolvedImpurityComposition,
    *,
    time: Optional[float] = None,
    tolerance: float = 5e-4,
    method: Optional[str] = None,
    write_zeff: bool = True,
) -> Any:
    """A copy of ``ods`` whose ``core_profiles`` slice carries the resolved ion composition.

    Parameters
    ----------
    ods : ODS
        Source; never modified [any].
    resolved : ResolvedImpurityComposition
        From :func:`resolve_impurity_composition`; not of kind ``measured`` [any].
    time : float, optional
        Slice time; default the resolved composition's own [s].
    tolerance : float, optional
        Largest ``|t_slice - time|`` accepted [s].
    method : str, optional
        ``method=`` field of the records; default ``<kind>_composition`` [-].
    write_zeff : bool, optional
        Also write ``zeff`` through :func:`populate_zeff_profile` [-].

    Returns
    -------
    ODS
        A deep copy whose slice ``ion[]`` is the hydrogenic main ion (density
        diluted to ``n_e (1 - sum Z_s n_s/n_e) / Z_m``) followed by one entry per
        impurity species (``label``, ``z_ion``, ``element.0.z_n/a``,
        ``density``, ``density_thermal``, ``density_fit.parameters``) [any].

    Raises
    ------
    ValueError
        A measured composition (resolved or stored in the target slice), a
        ``time`` that is not the resolved composition's, no slice at the time,
        no electron density, a stored main ion that is not hydrogenic, or two
        stored hydrogenic ions.

    Processing steps
    ----------------
    1. Match the slice by time and read ``n_e`` and its rho grid.
    2. Keep the existing hydrogenic main ion entry whole (temperature,
       velocity and their records) or create ``H+``; drop every stored
       impurity entry.
    3. Write the main ion's diluted density and each impurity's
       ``n_e * n_s/n_e`` (a profile composition is interpolated in rho).
    4. Give each impurity the main ion's temperature, its
       ``temperature_fit.parameters`` record unchanged (so a Ti reader
       classifies every ion alike) and its toroidal rotation; label each
       density ``origin=...``.  A point where the composition is undefined
       is filled from its neighbours in rho and counted (``filled_points``).
    5. Write ``zeff`` (:func:`populate_zeff_profile`), or, with
       ``write_zeff=False``, drop a non-measured ``zeff`` that the new species
       list would contradict.

    Input semantics
    ---------------
    A ``core_profiles`` slice with an electron density, and a composition
    resolved independently of it (preset, explicit, derived).

    Output semantics
    ----------------
    The same slice with an explicit species list: assumed or derived, never
    measured, as each record says.

    Convention
    ----------
    ``origin=assumed`` for an explicit or preset composition, ``derived`` for
    one computed from other data (atomic data, a measured Z_eff), in the
    grammar :func:`composition_record_origin` reads.  Impurities share the
    main ion's temperature: an assumption, recorded in the density record as
    ``temperature=main_ion``.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A non-hydrogenic main ion is refused.  Only this slice is written.  The
    impurities' copied temperature record reads as whatever the main ion's
    says (measured, assumed or inferred): equal temperatures are an
    assumption noted only in the density record.

    Provenance
    ----------
    .. [issue] #1565 Sec. 5 (core_profiles ion[] and zeff from the resolved
               composition), Lane K #1454 (origin records on core_profiles).
    """
    work = copy.deepcopy(ods)
    index, ne, rho = _write_slice(work, resolved, time, tolerance)
    base = f"core_profiles.profiles_1d.{index}"
    from vaft.ods_access import path_count

    _reset_ions(work, base, resolved.main_ion)
    fractions, filled = _fill_undefined(_on_grid(resolved.impurity_fractions, resolved, rho, ne.size), rho)
    main_fraction, _ = _fill_undefined(_on_grid(resolved.main_ion_fraction, resolved, rho, ne.size), rho)
    fill_note = f"; filled_points={filled}" if filled else ""
    main_density = ne * main_fraction
    work[f"{base}.ion.0.density"] = main_density
    work[f"{base}.ion.0.density_thermal"] = main_density
    work[f"{base}.ion.0.density_fit.parameters"] = _record(resolved, "main_ion_density", method) + fill_note
    temperature = _get(work, f"{base}.ion.0.temperature")
    t_record = _get(work, f"{base}.ion.0.temperature_fit.parameters")
    rotation = _get(work, f"{base}.ion.0.velocity.toroidal")
    for j, item in enumerate(resolved.species, start=1):
        ion = f"{base}.ion.{j}"
        work[f"{ion}.label"] = item.label
        work[f"{ion}.z_ion"] = float(item.charge_state)
        work[f"{ion}.element.0.z_n"] = float(item.z_n)
        work[f"{ion}.element.0.a"] = float(item.mass)
        work[f"{ion}.element.0.atoms_n"] = 1
        density = ne * fractions[:, j - 1]
        work[f"{ion}.density"] = density
        work[f"{ion}.density_thermal"] = density
        work[f"{ion}.density_fit.parameters"] = (
            _record(resolved, "impurity_density", method) + "; temperature=main_ion; rotation=main_ion" + fill_note
        )
        if temperature is not None:
            work[f"{ion}.temperature"] = np.asarray(temperature, dtype=float)
            if t_record is not None:
                work[f"{ion}.temperature_fit.parameters"] = t_record
        if rotation is not None:
            work[f"{ion}.velocity.toroidal"] = np.asarray(rotation, dtype=float)
    if not write_zeff and _get(work, f"{base}.zeff") is not None:
        if composition_record_origin(_get(work, f"{base}.zeff_fit.parameters")) != "measured":
            # a stale zeff beside the new species list would contradict it
            del work[f"{base}.zeff"]
            if "zeff_fit" in work[base].keys():
                del work[f"{base}.zeff_fit"]
    if write_zeff:
        work = populate_zeff_profile(work, resolved, time=time, tolerance=tolerance, method=method)
    return work


# --- radial charge states from atomic data (#1565 Sec. 8) -------------------------------------

#: How the elemental impurity density is scaled once its shape is fixed.
NORMALIZATIONS = ("ne_weighted_mean", "axis", "fixed", "resistive_closure")


@dataclass(frozen=True)
class RadialImpurityComposition:
    """Charge-state-resolved impurity composition on a radial grid.

    Elements keep a fixed relative density and a flat ``n_s/n_e`` shape scaled
    by one factor ``scale`` (``n_s/n_e = scale * w_s``); their charge-state
    distributions follow ``T_e(rho)``, ``n_e(rho)`` from ADF11 data.  Arrays
    run over ``rho``; species-indexed ones carry the element on the last axis;
    ``charge_state_fractions`` is a tuple, one ``(n_rho, Z_n + 1)`` array per
    element with the charge (neutral first) on the last axis.
    """

    kind: str
    rho: np.ndarray
    te_eV: np.ndarray
    ne_m3: np.ndarray
    elements: tuple[str, ...]
    weights: np.ndarray
    scale: float
    elemental_fractions: np.ndarray
    charge_state_fractions: tuple[np.ndarray, ...]
    mean_charge: np.ndarray
    mean_square_charge: np.ndarray
    S1: np.ndarray
    S2: np.ndarray
    effective_charge: np.ndarray
    zeff: np.ndarray
    main_ion_fraction: np.ndarray
    dilution_fraction: np.ndarray
    coronal_mean_charge: np.ndarray
    relaxation_time_s: np.ndarray
    coronal_valid: Optional[np.ndarray]
    main_ion: str = "H"
    time: Optional[float] = None
    normalization: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def zeff_source(self) -> str:
        return f"atomic-data charge states, normalization={self.normalization.get('method')}"

    @property
    def source(self) -> str:
        return f"OpenADAS ADF11 {self.provenance.get('ionization')} charge states"

    def as_rows(self, **key: Any) -> list[dict[str, Any]]:
        """One row per rho, prefixed by ``key`` (e.g. the #1454 state key)."""
        rows = []
        for i, r in enumerate(self.rho):
            row = dict(key)
            row.update(rho=float(r), te_eV=float(self.te_eV[i]), ne_m3=float(self.ne_m3[i]),
                       zeff=float(self.zeff[i]), S1=float(self.S1[i]), S2=float(self.S2[i]),
                       z_i_eff=float(self.effective_charge[i]),
                       main_ion_fraction=float(self.main_ion_fraction[i]),
                       dilution_fraction=float(self.dilution_fraction[i]),
                       coronal_valid=None if self.coronal_valid is None else bool(self.coronal_valid[i]))
            for k, element in enumerate(self.elements):
                row[f"n_{element}_over_ne"] = float(self.elemental_fractions[i, k])
                row[f"mean_z_{element}"] = float(self.mean_charge[i, k])
                row[f"mean_z2_{element}"] = float(self.mean_square_charge[i, k])
                row[f"coronal_mean_z_{element}"] = float(self.coronal_mean_charge[i, k])
                row[f"tau_relax_ms_{element}"] = float(self.relaxation_time_s[i, k] * 1e3)
            rows.append(row)
        return rows


def _tables(element: str, tables: Optional[Mapping[str, Any]], cache_dir) -> tuple[Any, Any, dict[str, str]]:
    from vaft.data.open_adas import default_adf11_files, get_adf11_path, read_adf11

    if tables is not None and element in tables:
        acd, scd = tables[element]
        def label(source):
            return str(getattr(source, "path", source))

        return (read_adf11(acd) if not hasattr(acd, "log_coefficients") else acd,
                read_adf11(scd) if not hasattr(scd, "log_coefficients") else scd,
                {"acd": label(acd), "scd": label(scd)})
    names = default_adf11_files(element)
    acd_path = get_adf11_path(names["acd"], cache_dir=cache_dir)
    scd_path = get_adf11_path(names["scd"], cache_dir=cache_dir)
    return read_adf11(acd_path), read_adf11(scd_path), {"acd": names["acd"], "scd": names["scd"]}


def charge_state_moments(
    element: str,
    te_eV: Any,
    ne_m3: Any,
    *,
    ionization: str = "coronal",
    age_s: Optional[float] = None,
    tables: Optional[Mapping[str, Any]] = None,
    cache_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Charge-state distribution of one element and its first two charge moments.

    Parameters
    ----------
    element : str
        Element symbol [-].
    te_eV : array-like
        Electron temperature, positive [eV].
    ne_m3 : array-like
        Electron density, positive [m^-3].
    ionization : str, optional
        ``"coronal"`` (steady ionisation balance) or ``"transient"`` (a parcel
        ``age_s`` after entering as neutrals) [-].
    age_s : float, optional
        Ionisation age for ``"transient"`` [s].
    tables : mapping, optional
        ``{element: (acd, scd)}`` ADF11 tables or paths overriding the
        OPEN-ADAS defaults [any].
    cache_dir : str, optional
        ADF11 cache directory [-].

    Returns
    -------
    dict
        ``fractions`` (charge on the last axis, neutral first), ``mean_charge``
        and ``mean_square_charge`` [-], ``coronal_mean_charge`` [-],
        ``relaxation_time_s`` [s] and the ``tables`` used [any].

    Raises
    ------
    ValueError
        An unknown ionization model, ``"transient"`` without an age, or
        non-positive profiles.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    No transport and no charge exchange: the transient model is an age, not
    an impurity-transport solution (Aurora, #897).

    Provenance
    ----------
    .. [1] H. P. Summers, *The ADAS User Manual*, version 2.6 (2004), ADF11.
    .. [issue] #1565 Sec. 8.
    """
    from vaft.formula.atomic import (
        coronal_relaxation_time,
        fractional_abundances,
        mean_charge_from_charge_state_densities,
        mean_square_charge_from_charge_state_densities,
        transient_fractional_abundances,
    )

    if ionization not in ("coronal", "transient"):
        raise ValueError(f"ionization must be 'coronal' or 'transient', got {ionization!r}")
    acd, scd, used = _tables(element, tables, cache_dir)
    z_n = _element_table()[element][0]
    if acd.n_charge_states != z_n or scd.n_charge_states != z_n:
        raise ValueError(
            f"{element}: ADF11 tables hold {acd.n_charge_states}/{scd.n_charge_states} charge-state "
            f"blocks, not Z_n = {z_n}; a metastable-resolved or wrong-element file?"
        )
    coronal = fractional_abundances(ne_m3, te_eV, acd, scd)
    if ionization == "transient":
        if age_s is None:
            raise ValueError("the transient model needs the ionisation age age_s")
        fractions = transient_fractional_abundances(ne_m3, te_eV, acd, scd, age_s)
    else:
        fractions = coronal
    return {
        "fractions": fractions,
        "mean_charge": mean_charge_from_charge_state_densities(fractions),
        "mean_square_charge": mean_square_charge_from_charge_state_densities(fractions),
        "coronal_mean_charge": mean_charge_from_charge_state_densities(coronal),
        "transient_mean_charge": None if age_s is None else mean_charge_from_charge_state_densities(
            fractions if ionization == "transient" else transient_fractional_abundances(ne_m3, te_eV, acd, scd, age_s)),
        "relaxation_time_s": coronal_relaxation_time(ne_m3, te_eV, acd, scd),
        "tables": used,
    }


def _radial_weights(rho: np.ndarray, ne: np.ndarray, volume_weights) -> tuple[np.ndarray, str]:
    if volume_weights is not None:
        dv = np.asarray(volume_weights, dtype=float)
        if dv.shape != rho.shape or np.any(dv < 0.0) or not np.all(np.isfinite(dv)):
            raise ValueError("volume_weights must be finite, non-negative and on the rho grid")
        return dv, "caller dV"
    edges = np.concatenate(([0.0], 0.5 * (rho[1:] + rho[:-1]), [rho[-1]]))
    return rho * np.diff(edges), "cylindrical rho drho (no volume given)"


def resolve_radial_composition(
    te_eV: Any,
    ne_m3: Any,
    rho: Any,
    elemental_weights: Mapping[str, float],
    *,
    normalization: str = "ne_weighted_mean",
    target_zeff: float = 2.0,
    ionization: str = "coronal",
    plasma_age_s: Optional[float] = None,
    volume_weights: Any = None,
    resistive_target: Optional[float] = None,
    projection: Optional[Callable[[np.ndarray], float]] = None,
    coronal_tolerance: float = 0.1,
    tables: Optional[Mapping[str, Any]] = None,
    cache_dir: Optional[str] = None,
    time: Optional[float] = None,
) -> RadialImpurityComposition:
    """Radial Z_eff and reduced impurity from an elemental composition and atomic-data charge states.

    Parameters
    ----------
    te_eV : array-like
        Electron temperature on ``rho`` [eV].
    ne_m3 : array-like
        Electron density on ``rho`` [m^-3].
    rho : array-like
        Radial coordinate, increasing [-].
    elemental_weights : mapping
        Relative elemental (all charge states) densities by element, e.g.
        ``{"C": 1, "O": 1}``; normalised internally [-].
    normalization : str, optional
        One of :data:`NORMALIZATIONS`: ``ne_weighted_mean`` (the n_e-weighted
        volume mean of Z_eff equals ``target_zeff``), ``axis`` (Z_eff at the
        innermost point), ``fixed`` (the fully stripped closure at
        ``target_zeff``, i.e. 1/86 each for the VEST preset, then let Z_eff
        fall where the charge states do) or ``resistive_closure``
        (``projection(Z_eff) == resistive_target``) [-].
    target_zeff : float, optional
        Target of the ``ne_weighted_mean``, ``axis`` and ``fixed`` rules [-].
    ionization : str, optional
        ``"coronal"`` or ``"transient"`` (needs ``plasma_age_s``) [-].
    plasma_age_s : float, optional
        Time since breakdown; also the age of the coronal-validity check [s].
    volume_weights : array-like, optional
        ``dV`` per rho point for the volume mean; default ``rho drho``, a
        cylindrical stand-in in rho units [m^3].
    resistive_target : float, optional
        Resistive Z_eff the ``resistive_closure`` rule matches [-].
    projection : callable, optional
        ``Z_eff(rho) -> scalar`` resistive projection (#1566) for
        ``resistive_closure``; it receives the full ``rho`` grid with NaN at
        undefined points and must ignore them [any].
    coronal_tolerance : float, optional
        Largest ``|<Z>_transient(age) - <Z>_coronal|`` still called coronal [-].
    tables : mapping, optional
        ``{element: (acd, scd)}`` overriding the OPEN-ADAS defaults [any].
    cache_dir : str, optional
        ADF11 cache directory [-].
    time : float, optional
        Slice time, carried for the writer [s].

    Returns
    -------
    RadialImpurityComposition
        ``zeff(rho)``, ``<Z>``, ``<Z^2>``, ``S1``, ``S2``, ``Z_I,eff``,
        dilution, elemental and charge-state fractions, the coronal check and
        the normalization record [any].

    Raises
    ------
    ValueError
        An unknown rule or model, a target the composition cannot reach
        (a negative main-ion density), ``resistive_closure`` without its
        projection and target, or no valid profile point.

    Processing steps
    ----------------
    1. ADF11 charge-state fractions of each element at each point (coronal,
       or transient at ``plasma_age_s``); ``<Z>_s``, ``<Z^2>_s``.
    2. ``S1 = sum w_s <Z>_s``, ``S2 = sum w_s <Z^2>_s``.
    3. With ``n_s/n_e = c w_s``, ``Z_eff = Z_m + c (S2 - Z_m S1)`` is linear in
       the one scale ``c``; the normalization rule fixes ``c``.
    4. ``n_main/n_e = (1 - c S1)/Z_m``, ``Z_I,eff = S2/S1``, dilution.
    5. Coronal check: transient ``<Z>`` at ``plasma_age_s`` against coronal.

    Convention
    ----------
    The elemental ratio and a flat ``n_s/n_e`` shape are fixed; only their
    common amplitude is normalised, so the radial structure of Z_eff is the
    charge states' alone.  ``ne_weighted_mean`` is the default the VEST atlas
    uses (#1569); ``fixed`` shows how far Z_eff falls from the stripped value.

    Assumptions
    -----------
    No impurity transport: flat ``n_s/n_e``; equal ages everywhere.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    VEST discharges last tens of ms while carbon relaxes on ~10-80 ms at
    20-200 eV and 1e19 m^-3: the coronal model overstates the charge;
    ``coronal_valid`` says where.  Charge exchange with neutrals is ignored.

    Provenance
    ----------
    .. [1] H. P. Summers, *The ADAS User Manual*, version 2.6 (2004), ADF11.
    .. [issue] #1565 Sec. 8, Lane L log #1569 (normalization decision).
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}, got {normalization!r}")
    te = np.asarray(te_eV, dtype=float)
    ne = np.asarray(ne_m3, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if not (te.shape == ne.shape == rho.shape and rho.ndim == 1):
        raise ValueError("te_eV, ne_m3 and rho must be 1-D arrays of one length")
    valid = np.isfinite(te) & np.isfinite(ne) & (te > 0.0) & (ne > 0.0) & np.isfinite(rho)
    if not valid.any():
        raise ValueError("no point with a finite, positive T_e and n_e")
    elements = tuple(elemental_weights)
    raw = np.array([float(elemental_weights[e]) for e in elements], dtype=float)
    if not elements or np.any(raw < 0.0) or raw.sum() <= 0.0 or not np.all(np.isfinite(raw)):
        raise ValueError("elemental_weights must be non-negative with a positive sum")
    weights = raw / raw.sum()
    table = _element_table()
    for element in elements:
        if element not in table:
            raise ValueError(f"unknown element {element!r}")

    n = rho.size
    nz = len(elements)
    mean = np.full((n, nz), np.nan)
    mean2 = np.full((n, nz), np.nan)
    coronal_mean = np.full((n, nz), np.nan)
    transient_mean = np.full((n, nz), np.nan)
    relax = np.full((n, nz), np.nan)
    states = []
    used = {}
    for k, element in enumerate(elements):
        moments = charge_state_moments(element, te[valid], ne[valid], ionization=ionization,
                                       age_s=plasma_age_s, tables=tables, cache_dir=cache_dir)
        mean[valid, k] = moments["mean_charge"]
        mean2[valid, k] = moments["mean_square_charge"]
        coronal_mean[valid, k] = moments["coronal_mean_charge"]
        relax[valid, k] = moments["relaxation_time_s"]
        if moments["transient_mean_charge"] is not None:
            transient_mean[valid, k] = moments["transient_mean_charge"]
        full = np.full((n, table[element][0] + 1), np.nan)
        full[valid] = moments["fractions"]
        states.append(full)
        used[element] = moments["tables"]

    s1 = mean @ weights
    s2 = mean2 @ weights
    main_charge = float(table["H"][0])
    excess = s2 - main_charge * s1
    record: dict[str, Any] = {"method": normalization}
    if normalization == "ne_weighted_mean":
        dv, how = _radial_weights(rho, ne, volume_weights)
        m = valid & np.isfinite(excess)
        mean_excess = np.sum((ne * dv * excess)[m]) / np.sum((ne * dv)[m])
        scale = (float(target_zeff) - main_charge) / mean_excess
        record.update(target_zeff=float(target_zeff), weighting=f"n_e {how}")
    elif normalization == "axis":
        first = int(np.flatnonzero(valid)[0])
        scale = (float(target_zeff) - main_charge) / excess[first]
        record.update(target_zeff=float(target_zeff), at_rho=float(rho[first]))
    elif normalization == "fixed":
        z_n = np.array([table[e][0] for e in elements], dtype=float)
        stripped = float(weights @ z_n**2 - main_charge * (weights @ z_n))
        scale = (float(target_zeff) - main_charge) / stripped
        record.update(target_zeff_fully_stripped=float(target_zeff))
    else:
        if projection is None or resistive_target is None:
            raise ValueError("resistive_closure needs projection= and resistive_target=")
        from scipy.optimize import brentq

        if not np.nanmax(s1[valid]) > 0.0:
            raise ValueError("every valid point is neutral: no impurity amplitude changes Z_eff")
        upper = 1.0 / np.nanmax(s1[valid])

        def mismatch(c):
            value = float(projection(np.where(valid, main_charge + c * excess, np.nan)))
            if not np.isfinite(value):
                raise ValueError("the projection returned a non-finite value; it must ignore NaN "
                                 "(undefined) points of Z_eff(rho)")
            return value - float(resistive_target)

        lo, hi = mismatch(0.0), mismatch(upper * (1 - 1e-9))
        if lo * hi > 0.0:
            raise ValueError(
                f"no impurity amplitude reproduces resistive Z_eff {resistive_target:g}: the projection "
                f"spans [{lo + resistive_target:g}, {hi + resistive_target:g}]"
            )
        scale = brentq(mismatch, 0.0, upper * (1 - 1e-9), xtol=1e-12)
        record.update(resistive_target=float(resistive_target))
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError(f"the normalization gives an unphysical impurity amplitude {scale!r}")
    main = (1.0 - scale * s1) / main_charge
    if np.nanmin(main[valid]) < -1e-12:
        raise ValueError("the target needs more impurity charge than there are electrons somewhere")
    zeff = main_charge + scale * excess
    record["scale"] = float(scale)
    check = None
    if plasma_age_s is not None:
        lag = np.nanmax(np.abs(transient_mean - coronal_mean), axis=1)
        check = np.where(valid, lag <= coronal_tolerance, False)
        record_check = {"plasma_age_s": float(plasma_age_s), "tolerance": coronal_tolerance,
                        "points_not_coronal": int(np.count_nonzero(valid & ~check))}
    else:
        record_check = {"plasma_age_s": None,
                        "note": "no plasma age given: coronal validity not checked"}
    with np.errstate(invalid="ignore", divide="ignore"):
        effective = s2 / s1
    return RadialImpurityComposition(
        kind="inferred" if normalization == "resistive_closure" else "derived",
        rho=rho, te_eV=te, ne_m3=ne, elements=elements, weights=weights, scale=float(scale),
        elemental_fractions=np.outer(np.where(valid, scale, np.nan), weights),
        charge_state_fractions=tuple(states), mean_charge=mean, mean_square_charge=mean2,
        S1=s1, S2=s2, effective_charge=effective, zeff=zeff, main_ion_fraction=main,
        dilution_fraction=1.0 - main, coronal_mean_charge=coronal_mean, relaxation_time_s=relax,
        coronal_valid=check, time=time, normalization=record,
        provenance={"ionization": ionization, "tables": used, "coronal_check": record_check,
                    "elemental_weights_input": {e: float(elemental_weights[e]) for e in elements}},
    )


def populate_radial_impurity_profiles(
    ods: Any,
    radial: RadialImpurityComposition,
    *,
    time: Optional[float] = None,
    tolerance: float = 5e-4,
    method: Optional[str] = None,
) -> Any:
    """A copy of ``ods`` whose ``core_profiles`` slice carries a charge-state-resolved composition.

    Parameters
    ----------
    ods : ODS
        Source; never modified [any].
    radial : RadialImpurityComposition
        From :func:`resolve_radial_composition` [any].
    time : float, optional
        Slice time; default the composition's own [s].
    tolerance : float, optional
        Largest ``|t_slice - time|`` accepted [s].
    method : str, optional
        ``method=`` field; default ``openadas_<ionization>`` [-].

    Returns
    -------
    ODS
        A deep copy whose slice ``ion[]`` is the diluted hydrogenic main ion
        and one bundled entry per element: ``z_ion`` (the mean charge
        weighted by the element's density at the grid points, no volume
        element), ``z_ion_1d`` = <Z>(rho), ``z_ion_square_1d`` = <Z^2>(rho),
        the elemental ``density`` -- all charge states *including the neutral
        atoms*, the density <Z> and <Z^2> are averaged over -- and
        ``state[q]`` (``z_min = z_max = q``, charge-state density) for every
        ionised state, so ``sum(state.density) = density (1 - f_0)``; plus
        ``zeff`` [any].

    Raises
    ------
    ValueError
        As :func:`populate_impurity_profiles`.

    Processing steps
    ----------------
    1. Match the slice by time and read ``n_e``; refuse a measured target.
    2. Keep the hydrogenic main ion (diluted), drop stored impurity entries.
    3. Write each element bundled, with its charge-state densities, from the
       composition interpolated onto the slice rho grid.
    4. Write ``zeff`` (``origin=derived``, or ``inferred`` for a resistive
       closure).

    Input semantics
    ---------------
    A ``core_profiles`` slice with an electron density.

    Output semantics
    ----------------
    The same slice with bundled impurity ions whose charge varies with rho.

    Convention
    ----------
    IMAS bundled ions: ``z_ion`` is one number, the radial charge lives in
    ``z_ion_1d`` / ``z_ion_square_1d``.  A reader of ``z_ion`` alone loses
    the radial variation and the charge variance -- GACODE does, so the
    transport path lumps per surface instead (Lane L PR 5).

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Neutral atoms are not written (``core_profiles.neutral`` is out of scope).

    Provenance
    ----------
    .. [issue] #1565 Sec. 5 and 8.
    """
    work = copy.deepcopy(ods)
    index, ne, rho = _write_slice(work, radial, time, tolerance)
    base = f"core_profiles.profiles_1d.{index}"
    if composition_record_origin(_get(work, f"{base}.zeff_fit.parameters")) == "measured":
        raise ValueError(f"{base}.zeff is labelled measured; it is never overwritten")
    _reset_ions(work, base, radial.main_ion)
    if rho is None:
        if ne.size != radial.rho.size:
            raise ValueError("the slice has no rho grid and a different size from the composition")
        rho = radial.rho

    def grid(values):
        values = np.asarray(values, dtype=float)
        if values.shape[0] == rho.size and np.allclose(radial.rho, rho):
            out = values.copy()
        else:
            columns = values.reshape(values.shape[0], -1)
            ok = np.isfinite(radial.rho)
            # anti-alias: radial, not time -- the composition onto the slice rho grid.
            out = np.column_stack([np.interp(rho, radial.rho[ok], c[ok], left=np.nan, right=np.nan)
                                   for c in columns.T]).reshape((rho.size,) + values.shape[1:])
        return _fill_undefined(out, rho)

    origin = _WRITE_ORIGIN[radial.kind]
    tag = method or f"openadas_{radial.provenance.get('ionization')}"
    norm = radial.normalization.get("method")
    main, filled = grid(radial.main_ion_fraction)
    fill_note = f"; filled_points={filled}" if filled else ""
    record = composition_record_text(origin, tag, normalization=norm) + fill_note
    work[f"{base}.ion.0.density"] = ne * main
    work[f"{base}.ion.0.density_thermal"] = ne * main
    work[f"{base}.ion.0.density_fit.parameters"] = record
    temperature = _get(work, f"{base}.ion.0.temperature")
    t_record = _get(work, f"{base}.ion.0.temperature_fit.parameters")
    rotation = _get(work, f"{base}.ion.0.velocity.toroidal")
    elemental, _ = grid(radial.elemental_fractions)
    mean, _ = grid(radial.mean_charge)
    mean2, _ = grid(radial.mean_square_charge)
    table = _element_table()
    for k, element in enumerate(radial.elements):
        ion = f"{base}.ion.{k + 1}"
        density = ne * elemental[:, k]
        weight = density / np.sum(density) if np.sum(density) > 0 else np.full(density.shape, 1.0 / density.size)
        work[f"{ion}.label"] = element
        work[f"{ion}.z_ion"] = float(np.sum(weight * mean[:, k]))
        work[f"{ion}.z_ion_1d"] = mean[:, k]
        work[f"{ion}.z_ion_square_1d"] = mean2[:, k]
        work[f"{ion}.element.0.z_n"] = float(table[element][0])
        work[f"{ion}.element.0.a"] = float(table[element][1])
        work[f"{ion}.element.0.atoms_n"] = 1
        work[f"{ion}.density"] = density
        work[f"{ion}.density_thermal"] = density
        work[f"{ion}.density_fit.parameters"] = record + "; temperature=main_ion; rotation=main_ion"
        states, _ = grid(radial.charge_state_fractions[k])
        for q in range(1, states.shape[1]):
            state = f"{ion}.state.{q - 1}"
            work[f"{state}.z_min"] = float(q)
            work[f"{state}.z_max"] = float(q)
            work[f"{state}.label"] = f"{element}{q}+"
            work[f"{state}.density"] = density * states[:, q]
            work[f"{state}.density_thermal"] = density * states[:, q]
        if temperature is not None:
            work[f"{ion}.temperature"] = np.asarray(temperature, dtype=float)
            if t_record is not None:
                work[f"{ion}.temperature_fit.parameters"] = t_record
        if rotation is not None:
            work[f"{ion}.velocity.toroidal"] = np.asarray(rotation, dtype=float)
    zeff, _ = grid(radial.zeff)
    work[f"{base}.zeff"] = zeff
    work[f"{base}.zeff_fit.parameters"] = record
    return work


# --- an explicit composition in a GACODE profile, per surface (Lane L PR 5) ---------------------


def _resolved_on_grid(composition: ResolvedImpurityComposition, rho: np.ndarray):
    """A profile-valued resolved composition on a GACODE profile's rho grid, paired by coordinate.

    Both grids are ``rho_tor_norm``; the composition's ``rho`` is required and
    interpolated onto ``rho`` (same grid: taken as is).  Undefined points and the
    range beyond the composition's grid are filled as :func:`_fill_undefined`
    fills them and counted.  Returns ``(fractions, <Z>, <Z^2>, main_ion_fraction, note)``.
    """
    frac = np.asarray(composition.impurity_fractions, dtype=float)
    source = composition.rho
    if source is None:
        raise ValueError(
            "the composition is profile-valued but carries no rho (rho_tor_norm) to pair it with "
            "profile.rho; resolve it on a core_profiles grid or on profile.rho"
        )
    source = np.asarray(source, dtype=float)
    if source.ndim != 1 or source.size != frac.shape[0]:
        raise ValueError(
            f"the composition's rho has {source.size} points but its fractions {frac.shape[0]}; "
            "they must run over the same grid"
        )
    charges = np.array([s.charge_state for s in composition.species], dtype=float)
    mean = np.broadcast_to(charges, frac.shape) if composition.mean_charge is None else np.asarray(composition.mean_charge, dtype=float)
    mean2 = mean**2 if composition.mean_square_charge is None else np.asarray(composition.mean_square_charge, dtype=float)
    main = np.broadcast_to(np.asarray(composition.main_ion_fraction, dtype=float), (source.size,))
    stacked = np.concatenate([frac, mean, mean2, main[:, None]], axis=1)
    note: dict[str, Any] = {"coordinate": "rho_tor_norm", "profile_valued": True}
    if source.shape == rho.shape and np.allclose(source, rho):
        out, note["interpolated"] = stacked, False
    else:
        if not (np.all(np.isfinite(source)) and np.all(np.diff(source) > 0.0)):
            raise ValueError("the composition's rho grid must be finite and increasing to be interpolated onto profile.rho")
        if source[-1] < rho.min() or source[0] > rho.max():
            raise ValueError(
                f"the composition's rho [{source[0]:g}, {source[-1]:g}] does not overlap profile.rho "
                f"[{rho.min():g}, {rho.max():g}]; resolve the composition on profile.rho"
            )
        # anti-alias: spatial interpolation over rho, not time -- no sample rate to reduce
        out = np.column_stack([np.interp(rho, source, column, left=np.nan, right=np.nan) for column in stacked.T])
        note.update(interpolated=True, source_points=int(source.size))
    out, note["filled_points"] = _fill_undefined(out, rho)
    k = frac.shape[1]
    return out[:, :k], out[:, k:2 * k], out[:, 2 * k:3 * k], out[:, 3 * k], note


def surface_composition_profile(
    profile: Any,
    composition: Union[RadialImpurityComposition, ResolvedImpurityComposition],
    r_over_a: float,
) -> Any:
    """A copy of a GACODE profile whose ions are the main ion plus an explicit impurity composition.

    Parameters
    ----------
    profile : GACODEProfile
        The resolved transport profile (its main ion, temperatures, rotation
        and geometry are kept) [any].
    composition : RadialImpurityComposition or ResolvedImpurityComposition
        A radial composition must be resolved on ``profile.rho``; a resolved
        one may be scalar, or a profile on its own ``rho`` (``rho_tor_norm``),
        which is interpolated onto ``profile.rho`` [any].
    r_over_a : float
        The surface, ``rmin / rmin[-1]``, whose charge states the lumped
        impurities take [-].

    Returns
    -------
    GACODEProfile
        ``z``, ``name``, ``type``, ``mass``, ``ni``, ``ti``, ``vtor``/``vpol``
        and ``z_eff`` replaced; ``provenance`` records the composition and
        the surface [any].

    Raises
    ------
    ValueError
        A profile without a hydrogenic main ion or ``rmin``, a radial
        composition on another grid, or a surface outside the profile.

    Processing steps
    ----------------
    1. Main ion: the profile's first ``z = 1`` species, its density replaced
       by ``n_e n_main/n_e`` of the composition.
    2. At the surface, each element's lumped charge ``Z_s = <Z^2>_s/<Z>_s``.
    3. Its density ``n_s'(rho) = n_e (n_s/n_e) <Z>_s(rho) / Z_s``: the
       element's charge density is kept at every rho (so the gradients stay
       quasi-neutral) and its ``Z^2`` moment exactly at the surface.
    4. Temperatures and rotation of the impurities are the main ion's.

    Input semantics
    ---------------
    A converted GACODE profile (GACODE units: 1e19 m^-3, keV).

    Output semantics
    ----------------
    The same profile with a species list valid at one surface only.

    Convention
    ----------
    GACODE carries one charge per species for the whole profile, so a
    charge that varies with rho is lumped per surface (one profile per TGLF
    or CGYRO surface); with fixed charge states the profile is the same at
    every surface.  Z_eff at the surface equals the composition's.  A
    profile-valued composition is paired with ``profile.rho`` by coordinate
    (both are ``rho_tor_norm``): a radial composition must be resolved on
    ``profile.rho``; a resolved one is interpolated from its own ``rho``,
    which it must carry (``provenance["ni"]["composition_grid"]`` says so).

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The charge-state variance enters only through ``<Z^2>``; the charge's
    own radial gradient away from the surface is not represented.  Equal
    ion temperatures are assumed.

    Provenance
    ----------
    .. [issue] #1565 Sec. 6, Lane L log #1569 (lumped species per surface).
    """
    import dataclasses

    z = np.asarray(profile.z, dtype=float)
    main = np.flatnonzero(np.isclose(z, 1.0))
    if main.size == 0:
        raise ValueError("the profile has no hydrogenic main ion")
    m = int(main[0])
    if profile.rmin is None:
        raise ValueError("the profile carries no rmin to locate the surface")
    rho = np.asarray(profile.rho, dtype=float)
    radius = np.asarray(profile.rmin, dtype=float) / float(profile.rmin[-1])
    if not 0.0 <= float(r_over_a) <= 1.0:
        raise ValueError(f"r_over_a must lie in [0, 1], got {r_over_a!r}")
    # anti-alias: spatial interpolation over rho / r/a, not time -- no sample rate to reduce
    rho_s = float(np.interp(float(r_over_a), radius, rho))
    ne = np.asarray(profile.ne, dtype=float)
    n = rho.size
    table = _element_table()

    if isinstance(composition, RadialImpurityComposition):
        if composition.rho.shape != rho.shape or not np.allclose(composition.rho, rho):
            raise ValueError("resolve the radial composition on profile.rho")
        elements = list(composition.elements)
        fractions, _ = _fill_undefined(composition.elemental_fractions, rho)
        mean, _ = _fill_undefined(composition.mean_charge, rho)
        mean2, _ = _fill_undefined(composition.mean_square_charge, rho)
        main_fraction, _ = _fill_undefined(composition.main_ion_fraction, rho)
        masses = [float(table[e][1]) for e in elements]
        label = f"openadas_{composition.provenance.get('ionization')}_{composition.normalization.get('method')}"
    else:
        elements = [s.element for s in composition.species]
        charges = np.array([s.charge_state for s in composition.species], dtype=float)
        frac = np.asarray(composition.impurity_fractions, dtype=float)
        if frac.ndim == 1:
            fractions = np.broadcast_to(frac, (n, charges.size))
            mean = np.broadcast_to(charges, (n, charges.size))
            mean2 = mean**2
            main_fraction = np.broadcast_to(np.asarray(composition.main_ion_fraction, dtype=float), (n,))
            grid_note = {"coordinate": "rho_tor_norm", "profile_valued": False}
        else:
            fractions, mean, mean2, main_fraction, grid_note = _resolved_on_grid(composition, rho)
        masses = [float(s.mass) for s in composition.species]
        label = f"{composition.kind}_" + ("bundled_charge" if composition.mean_charge is not None else "fixed_charge")
    # anti-alias: spatial interpolation over rho / r/a, not time -- no sample rate to reduce
    lumped = np.array([np.interp(rho_s, rho, mean2[:, k] / mean[:, k]) for k in range(len(elements))])
    densities = [ne * np.asarray(main_fraction, dtype=float)]
    densities += [ne * fractions[:, k] * mean[:, k] / lumped[k] for k in range(len(elements))]
    ti_main = np.atleast_2d(np.asarray(profile.ti, dtype=float))[m]

    def repeat(field):
        if field is None:
            return None
        rows = np.atleast_2d(np.asarray(field, dtype=float))
        return np.vstack([rows[m]] * (1 + len(elements)))

    z_new = np.concatenate(([z[m]], lumped))
    stacked = np.vstack(densities)
    z_eff = np.sum(stacked * z_new[:, None] ** 2, axis=0) / ne
    provenance = dict(profile.provenance)
    provenance["ni"] = {"kind": "derived", "source": f"Lane L composition ({label})",
                        "surface_r_over_a": float(r_over_a), "lumped_charge": dict(zip(elements, lumped.tolist()))}
    if not isinstance(composition, RadialImpurityComposition):
        provenance["ni"]["composition_grid"] = grid_note
    provenance["ti"] = {**dict(profile.provenance.get("ti", {})),
                        "impurities": "equal to the main ion (assumed)"}
    provenance["z_eff"] = {"kind": "derived", "source": "species list (lumped at the surface)",
                           # anti-alias: spatial interpolation over rho / r/a, not time -- no sample rate to reduce
                           "value_at_surface": float(np.interp(rho_s, rho, z_eff))}
    return dataclasses.replace(
        profile,
        z=z_new,
        name=[profile.name[m]] + elements,
        type=[profile.type[m]] + ["[therm]"] * len(elements),
        mass=np.concatenate(([np.asarray(profile.mass, dtype=float)[m]], masses)),
        ni=stacked,
        ti=np.vstack([ti_main] * (1 + len(elements))),
        vtor=repeat(profile.vtor),
        vpol=repeat(profile.vpol),
        z_eff=z_eff,
        provenance=provenance,
    )
