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
from typing import Any, Mapping, Optional, Sequence, Union

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
    "composition_from_fractions",
    "composition_from_model",
    "composition_record_origin",
    "composition_record_text",
    "populate_impurity_profiles",
    "populate_zeff_profile",
    "resolve_impurity_composition",
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


def _ods_impurities(ods: Any, index: int) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Impurity ions of one ``core_profiles`` slice, their fractions and labels, or why there are none.

    Every ``ion[]`` entry is visited (counted, not scanned until a gap).  The
    main ion is the hydrogen isotopes; a slice whose ions are not hydrogenic
    plus impurities is refused rather than read with a hydrogen main ion it
    does not have.
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
    species, densities, origins, skipped = [], [], [], []
    has_main = False
    for k in range(n_ions):
        ion = f"{base}.ion.{k}"
        z_n = _get(ods, f"{ion}.element.0.z_n")
        if _hydrogenic(z_n):
            has_main = True
            continue
        z_ion = _get(ods, f"{ion}.z_ion")
        density = _get(ods, f"{ion}.density_thermal")
        if density is None:
            density = _get(ods, f"{ion}.density")
        if z_n is None or z_ion is None or density is None:
            skipped.append(f"ion.{k}: missing {'element.0.z_n' if z_n is None else 'z_ion' if z_ion is None else 'density'}")
            continue
        element = by_charge.get(int(round(float(z_n))))
        if element is None:
            skipped.append(f"ion.{k}: unknown element z_n={z_n}")
            continue
        mass = _get(ods, f"{ion}.element.0.a")
        species.append(ImpuritySpecies(element=element, charge_state=float(z_ion),
                                       mass=None if mass is None else float(mass)))
        densities.append(np.asarray(density, dtype=float))
        origins.append(composition_record_origin(_get(ods, f"{ion}.density_fit.parameters")))
    if not species:
        return None, "; ".join(skipped) or None
    if not has_main:
        return None, "no hydrogenic main ion among the stored ions; a non-hydrogenic main ion is not read"
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_ne = np.where(np.isfinite(ne) & (ne > 0.0), ne, np.nan)
        fractions = np.stack([np.broadcast_to(d, ne.shape) / safe_ne for d in densities], axis=-1)
    distinct = set(origins)
    origin = origins[0] if len(distinct) == 1 else "mixed"
    return ({"species": tuple(species), "fractions": fractions, "skipped": skipped,
             "rho": None if rho is None else np.asarray(rho, dtype=float), "origin": origin}, None)


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
           resistive, candidates, provenance) -> ResolvedImpurityComposition:
    """Close relative weights at a target Z_eff by quasi-neutrality.

    A scalar target outside the reachable interval is an error.  A profile
    target (a measured Z_eff) is closed point by point: a point that is not
    finite or lies outside ``[Z_m, S2/S1]`` is left undefined (NaN) and
    counted in the provenance, rather than failing the slice.
    """
    charges = np.array([s.charge_state for s in species], dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    provenance = dict(provenance)
    if target.ndim == 0 and weights.ndim == 1:
        solution = solve_impurity_mixture_for_target_zeff(target, weights, charges, main_ion_charge=main_charge)
        fractions = np.asarray(solution.impurity_fractions, dtype=float)
    else:
        shape = np.broadcast_shapes(target.shape, weights.shape[:-1])
        t = np.broadcast_to(target, shape).reshape(-1)
        w = np.broadcast_to(weights, shape + weights.shape[-1:]).reshape(-1, weights.shape[-1])
        with np.errstate(invalid="ignore", divide="ignore"):
            s1, s2 = w @ charges, w @ charges**2
            ok = (np.isfinite(t) & np.all(np.isfinite(w), axis=-1) & np.all(w >= 0.0, axis=-1)
                  & (s1 > 0.0) & (s2 - main_charge * s1 > 0.0)
                  & (t >= main_charge - 1e-9) & (t <= s2 / s1 * (1.0 + 1e-9)))
        flat = np.full(w.shape, np.nan)
        if ok.any():
            flat[ok] = solve_impurity_mixture_for_target_zeff(
                t[ok], w[ok], charges, main_ion_charge=main_charge
            ).impurity_fractions
        fractions = flat.reshape(shape + weights.shape[-1:])
        target = np.where(ok, t, np.nan).reshape(shape)
        provenance["undefined_points"] = int(np.count_nonzero(~ok))
    return _finish(species, main_ion, main_charge, fractions, target, kind=kind, source=source,
                   zeff_source=zeff_source, rho=rho, time=time, resistive=resistive,
                   candidates=candidates, provenance=provenance)


def _finish(species, main_ion, main_charge, fractions, zeff, *, kind, source, zeff_source, rho, time,
            resistive, candidates, provenance) -> ResolvedImpurityComposition:
    """Moments, reduction and dilution of resolved fractions; undefined points stay NaN.

    A point is undefined where ``n_e`` is zero or missing (the edge point of
    a fitted profile is often exactly zero), where a stored fraction is
    negative, or where the impurities carry more charge than there are
    electrons: everything formed there is NaN, rather than an error for the
    whole slice or a number made up for that point.
    """
    charges = np.array([s.charge_state for s in species], dtype=float)
    masses = np.array([s.mass for s in species], dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    shape = fractions.shape[:-1]
    flat = fractions.reshape(-1, fractions.shape[-1])
    with np.errstate(invalid="ignore"):
        defined = np.all(np.isfinite(flat), axis=-1) & np.all(flat >= 0.0, axis=-1)
        main = np.where(defined, (1.0 - flat @ charges) / main_charge, np.nan)
        defined &= main >= -1e-12
    main = np.where(defined, np.clip(main, 0.0, None), np.nan)
    ok = defined & (np.sum(np.where(defined[:, None], flat, 0.0), axis=-1) > 0.0)
    no_impurity = defined & ~ok

    def blank():
        return np.full(ok.shape, np.nan)

    weights = np.full(flat.shape, np.nan)
    s1, s2, a_bar = blank(), blank(), blank()
    eff_charge, eff_fraction, eff_mass = blank(), blank(), blank()
    if ok.any():
        sub = flat[ok]
        w = sub / np.sum(sub, axis=-1, keepdims=True)
        weights[ok] = w
        moments = impurity_mixture_moments(w, charges, masses)
        s1[ok], s2[ok], a_bar[ok] = moments.S1, moments.S2, moments.A_bar
        effective = reduce_impurity_mixture(sub, charges, masses)
        eff_charge[ok], eff_fraction[ok], eff_mass[ok] = effective.charge, effective.density, effective.mass
    if no_impurity.any():  # a target of exactly Z_m: no impurity, the pseudo-impurity is the mixture's own
        uniform = np.full(len(species), 1.0 / len(species))
        weights[no_impurity] = uniform
        moments = impurity_mixture_moments(uniform, charges, masses)
        s1[no_impurity], s2[no_impurity], a_bar[no_impurity] = moments.S1, moments.S2, moments.A_bar
        eff_charge[no_impurity] = moments.S2 / moments.S1
        eff_fraction[no_impurity] = 0.0
        eff_mass[no_impurity] = moments.A_bar * moments.S2 / moments.S1**2
    if zeff is None:
        with np.errstate(invalid="ignore"):
            zeff = np.where(defined, main_charge**2 * main + flat @ charges**2, np.nan)

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
       origin, and its ``zeff`` if labelled ``origin=measured``.
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
    Charge states are fixed per species; a radially varying charge-state
    distribution enters as a ``derived`` composition (#1565 Sec. 8).  A stored
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
        record = {"method": "stored ion densities"}
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
                              candidates=candidates, provenance=record, **common)
            candidates.append({"source": "core_profiles.zeff", "outcome": "skipped",
                               "reason": "measured Z_eff is not on the ion-density grid"})
        return _finish(stored["species"], "H", 1.0, stored["fractions"], None, kind=kind,
                       source=f"core_profiles.ion (origin={label})", zeff_source="species",
                       rho=stored["rho"], candidates=candidates, provenance=record, **common)

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
_WRITE_ORIGIN = {"explicit": "assumed", "assumed": "assumed", "derived": "derived"}


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

    if not _hydrogenic(_element_table()[resolved.main_ion][0]):
        raise ValueError(f"only a hydrogenic main ion is written, not {resolved.main_ion!r}")
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
    fractions, filled = _fill_undefined(_on_grid(resolved.impurity_fractions, resolved, rho, ne.size), rho)
    main_fraction, _ = _fill_undefined(_on_grid(resolved.main_ion_fraction, resolved, rho, ne.size), rho)
    fill_note = f"; filled_points={filled}" if filled else ""
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
