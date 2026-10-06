"""Analytic kinetic closure: dilution-aware thermal pressure and a classical fast-ion estimate (issue #1606).

Connects pieces VAFT already has -- Thomson n_e and T_e, a T_i/T_e policy,
an impurity composition (#1565), the per-species pressure law -- into one
pressure construction that does not assume ``n_i = n_e``, and adds a
lightweight fast-ion baseline (:mod:`vaft.formula.fast_ion`) for when no
NUBEAM run exists.  Thermal and fast pressure stay separate quantities.

The chain::

    core_profiles slice (matched by time): n_e, T_e, T_i or T_i/T_e policy
    + impurity composition (vaft.process.impurity.resolve_impurity_composition)
        -> species densities n_s = n_e (n_s/n_e), main ion diluted   [dilution="species"]
           or n_i = n_e, one hydrogenic ion                           [dilution="none", the old form]
        -> p_e = e n_e T_e,  p_i = e sum_s n_s T_s,  p_thermal = p_e + p_i
    + optional fast-ion source (fast_ion_slowing_down_estimate)
        -> p_fast = 2 W_f / 3
    -> PressureAssembly: p_e, p_i_thermal, p_thermal, p_fast, p_total

Notation
--------
p_e, p_i      : electron and summed ion thermal pressure        [Pa]
p_fast        : scalar fast-ion pressure                         [Pa]
p_total       : p_thermal + p_fast                               [Pa]
S             : fast-ion birth rate per unit volume              [m^-3 s^-1]

Conventions
-----------
**Thermal and fast never share a field.**  ``p_thermal`` holds Maxwellian
species only; ``p_total`` adds ``p_fast``.  Writing fast pressure into a
thermal field is what #1606 Sec. 5 forbids.

**Every assumed number is recorded.**  The T_i source (measured or a ratio
with its record), the composition's kind and source, the dilution mode and
the fast-ion inputs are in ``provenance``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "DILUTION_MODES",
    "FastIonEstimate",
    "KineticClosure",
    "PressureAssembly",
    "assemble_pressure",
    "fast_ion_slowing_down_estimate",
    "infer_kinetic_closure",
]

#: ``species``: ion densities from the composition (main ion diluted);
#: ``none``: the legacy single hydrogenic ion with n_i = n_e, kept for reproducibility.
DILUTION_MODES = ("species", "none")

_QE = 1.602176634e-19


@dataclass(frozen=True)
class PressureAssembly:
    """Pressure contributions on one grid, kept apart [Pa]."""

    p_e: np.ndarray
    p_i_species: Mapping[str, np.ndarray]
    p_i_thermal: np.ndarray
    p_thermal: np.ndarray
    p_fast: Optional[np.ndarray]
    p_total: np.ndarray
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FastIonEstimate:
    """A classical slowing-down estimate of one fast-ion population on a grid."""

    source_rate: np.ndarray
    beam_energy_eV: float
    A_b: float
    Z_b: float
    v_b: float
    v_c: np.ndarray
    E_c_eV: np.ndarray
    tau_s: np.ndarray
    tau_thermalisation: np.ndarray
    n_fast: np.ndarray
    W_fast: np.ndarray
    p_fast: np.ndarray
    provenance: Mapping[str, Any] = field(default_factory=dict)
    time: Optional[float] = None
    rho: Optional[np.ndarray] = None


@dataclass(frozen=True)
class KineticClosure:
    """A resolved kinetic state: composition, densities, temperatures, pressures, provenance."""

    time: Optional[float]
    rho: Optional[np.ndarray]
    n_e: np.ndarray
    T_e: np.ndarray
    ion_densities: Mapping[str, np.ndarray]
    ion_temperatures: Mapping[str, np.ndarray]
    pressure: PressureAssembly
    composition: Any = None
    fast_ion: Optional[FastIonEstimate] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


def _finite_non_negative(value, name, shape=None):
    array = np.asarray(value, dtype=float)
    if shape is not None:
        array = np.broadcast_to(array, shape).copy()
    if np.any(array[np.isfinite(array)] < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def assemble_pressure(
    n_e: Any,
    T_e: Any,
    ion_densities: Mapping[str, Any],
    ion_temperatures: Mapping[str, Any],
    *,
    p_fast: Any = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> PressureAssembly:
    """Electron, per-species ion, thermal, fast and total pressure, each kept separate.

    Parameters
    ----------
    n_e : array-like
        Electron density [m^-3].
    T_e : array-like
        Electron temperature [eV].
    ion_densities : mapping
        Thermal density of each ion species by label [m^-3].
    ion_temperatures : mapping
        Temperature of each ion species, same labels [eV].
    p_fast : array-like, optional
        Scalar pressure of non-thermal populations; ``None`` when there is none
        to add [Pa].
    provenance : mapping, optional
        Record carried into the result [any].

    Returns
    -------
    PressureAssembly
        ``p_e = e n_e T_e``, ``p_i_species`` and their sum ``p_i_thermal``,
        ``p_thermal = p_e + p_i_thermal``, ``p_fast`` and
        ``p_total = p_thermal + p_fast`` [Pa].

    Raises
    ------
    ValueError
        Labels that differ between densities and temperatures, or negative
        densities, temperatures or fast pressure.

    Convention
    ----------
    Thermal and fast pressure are never merged into one field; ``p_total`` is
    the only sum of the two.  Uses :func:`vaft.formula.equilibrium.electron_pressure`
    and :func:`~vaft.formula.equilibrium.ion_pressure` per species.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [issue] #1606 Sec. 2 and 5.
    """
    from vaft.formula.equilibrium import electron_pressure, ion_pressure

    if set(ion_densities) != set(ion_temperatures):
        raise ValueError("ion_densities and ion_temperatures must name the same species")
    ne = _finite_non_negative(n_e, "n_e")
    te = _finite_non_negative(T_e, "T_e", ne.shape)
    p_e = np.asarray(electron_pressure(np.nan_to_num(ne), np.nan_to_num(te)), dtype=float)
    p_e = np.where(np.isfinite(ne) & np.isfinite(te), p_e, np.nan)
    p_species = {}
    for label, density in ion_densities.items():
        n = _finite_non_negative(density, f"density of {label}", ne.shape)
        t = _finite_non_negative(ion_temperatures[label], f"temperature of {label}", ne.shape)
        ok = np.isfinite(n) & np.isfinite(t)
        p_species[label] = np.where(ok, np.asarray(ion_pressure(np.where(ok, n, 0.0), np.where(ok, t, 0.0))), np.nan)
    p_i = np.sum(list(p_species.values()), axis=0) if p_species else np.zeros_like(p_e)
    p_thermal = p_e + p_i
    fast = None if p_fast is None else _finite_non_negative(p_fast, "p_fast", ne.shape)
    return PressureAssembly(p_e=p_e, p_i_species=p_species, p_i_thermal=p_i, p_thermal=p_thermal,
                            p_fast=fast, p_total=p_thermal + (0.0 if fast is None else fast),
                            provenance=dict(provenance or {}))


def fast_ion_slowing_down_estimate(
    n_e: Any,
    T_e: Any,
    field_ions: Mapping[str, tuple[Any, float, float]],
    source_rate: Any,
    beam_energy_eV: float,
    *,
    A_b: float,
    Z_b: float = 1.0,
    ln_Lambda: Any = None,
    time: Optional[float] = None,
    rho: Any = None,
) -> FastIonEstimate:
    """Classical slowing-down density, energy and pressure of one fast-ion source on a grid.

    Parameters
    ----------
    n_e : array-like
        Electron density [m^-3].
    T_e : array-like
        Electron temperature [eV].
    field_ions : mapping
        ``label -> (density, Z, A)`` of every thermal ion species [any].
    source_rate : array-like
        Fast-ion birth rate per unit volume on the grid [m^-3 s^-1].
    beam_energy_eV : float
        Birth energy of the fast ions [eV].
    A_b : float
        Mass number of the fast ion [-].
    Z_b : float, optional
        Charge of the fast ion [-].
    ln_Lambda : array-like, optional
        Electron Coulomb logarithm; default the NRL form of n_e, T_e [-].
    time : float, optional
        Time of the profiles; :func:`infer_kinetic_closure` refuses the
        estimate for a slice further away than its tolerance [s].
    rho : array-like, optional
        Radial grid of the profiles; checked against the slice grid [-].

    Returns
    -------
    FastIonEstimate
        ``v_c``, ``E_c``, ``tau_s``, the thermalisation time, ``n_fast``,
        ``W_fast`` and the isotropic ``p_fast = 2 W_fast / 3`` [any].

    Raises
    ------
    ValueError
        Non-positive beam energy, mass or charge, or a negative source rate.

    Processing steps
    ----------------
    1. ``v_c`` from the field ions (:func:`vaft.formula.fast_ion.critical_velocity_from_T_e_n_species`).
    2. ``tau_s`` on electrons with ln Lambda.
    3. Steady slowing-down density and energy density from ``source_rate``.
    4. ``p_fast = 2 W_fast / 3``.

    Assumptions
    -----------
    Steady state, a single birth energy, classical drag, local deposition
    (no orbit width), an isotropic population, no charge-exchange loss.
    Points with a non-positive or non-finite n_e or T_e (the zero edge of a
    fitted profile) have no defined drag and are returned as NaN.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A baseline, not NUBEAM: orbit losses, anisotropy and charge exchange --
    large in a small device like VEST -- are absent, so it bounds the fast
    pressure from above where those losses matter.

    Provenance
    ----------
    .. [1] T. H. Stix, Plasma Phys. 14 (1972) 367.
    .. [issue] #1606 Sec. 3-4.
    """
    from vaft.formula.equilibrium import coulomb_logarithm_from_n_T
    from vaft.formula.fast_ion import (
        critical_velocity_from_T_e_n_species,
        fast_ion_density_from_source,
        fast_ion_energy_density_from_source,
        fast_ion_pressure_from_energy_density,
        slowing_down_time_between_speeds,
        slowing_down_time_from_T_e_n_e_A_b_Z_b,
    )

    if not beam_energy_eV > 0.0 or not A_b > 0.0 or not Z_b > 0.0:
        raise ValueError("beam_energy_eV, A_b and Z_b must be positive")
    ne = np.atleast_1d(np.asarray(n_e, dtype=float))
    te = np.broadcast_to(np.asarray(T_e, dtype=float), ne.shape)
    labels = list(field_ions)
    n_j = np.stack([np.broadcast_to(np.asarray(field_ions[k][0], float), ne.shape) for k in labels], axis=-1)
    z_j = np.array([float(field_ions[k][1]) for k in labels])
    a_j = np.array([float(field_ions[k][2]) for k in labels])
    lnl = (None if ln_Lambda is None
           else np.broadcast_to(np.asarray(ln_Lambda, dtype=float), ne.shape))
    source = np.broadcast_to(np.asarray(source_rate, dtype=float), ne.shape)
    if np.any(source < 0.0):
        raise ValueError("source_rate must be non-negative")
    # Fitted profiles often end in a zero density or a NaN temperature: those
    # points carry no defined drag and get NaN, as the thermal side does.
    ok = (np.isfinite(ne) & (ne > 0.0) & np.isfinite(te) & (te > 0.0) & np.isfinite(source)
          & np.all(np.isfinite(n_j) & (n_j >= 0.0), axis=-1) & (np.sum(n_j, axis=-1) > 0.0))
    if lnl is not None:
        ok &= np.isfinite(lnl) & (lnl > 0.0)
    v_b = float(np.sqrt(2.0 * beam_energy_eV * _QE / (A_b * 1.66053906660e-27)))
    names = ("v_c", "tau_s", "tau_thermalisation", "n_fast", "W_fast")
    out = {name: np.full(ne.shape, np.nan) for name in names}
    if np.any(ok):
        ln_ok = coulomb_logarithm_from_n_T(ne[ok], te[ok]) if lnl is None else lnl[ok]
        v_c = np.asarray(critical_velocity_from_T_e_n_species(te[ok], ne[ok], n_j[ok], z_j, a_j))
        tau_s = np.asarray(slowing_down_time_from_T_e_n_e_A_b_Z_b(te[ok], ne[ok], A_b, Z_b, ln_ok))
        out["v_c"][ok], out["tau_s"][ok] = v_c, tau_s
        out["tau_thermalisation"][ok] = slowing_down_time_between_speeds(tau_s, v_c, v_b)
        out["n_fast"][ok] = fast_ion_density_from_source(source[ok], tau_s, v_b, v_c)
        out["W_fast"][ok] = fast_ion_energy_density_from_source(source[ok], tau_s, A_b, v_b, v_c)
    w_f = out["W_fast"]
    p_f = np.where(ok, np.asarray(fast_ion_pressure_from_energy_density(np.where(ok, w_f, 0.0))), np.nan)
    return FastIonEstimate(
        source_rate=np.asarray(source), beam_energy_eV=float(beam_energy_eV), A_b=float(A_b), Z_b=float(Z_b),
        v_b=v_b, v_c=out["v_c"], E_c_eV=0.5 * A_b * 1.66053906660e-27 * out["v_c"]**2 / _QE,
        tau_s=out["tau_s"], tau_thermalisation=out["tau_thermalisation"],
        n_fast=out["n_fast"], W_fast=w_f, p_fast=p_f,
        provenance={"model": "classical slowing down (Stix 1972), steady, isotropic, local",
                    "ln_Lambda": "NRL electron form" if ln_Lambda is None else "caller",
                    "undefined_points": int(np.count_nonzero(~ok))},
        time=None if time is None else float(time),
        rho=None if rho is None else np.asarray(rho, dtype=float),
    )


def _main_ion_path(ods: Any, base: str) -> Optional[str]:
    """Path of the one hydrogenic ``ion[]`` entry (``element.0.z_n == 1``), or None.

    The entry is found by its nuclear charge, not its position.  Several
    hydrogenic entries (an isotope mix) are refused rather than one of them
    standing in for the main ion.
    """
    from vaft.ods_access import path_count
    from vaft.process.impurity import _get, _hydrogenic

    found = [f"{base}.ion.{k}" for k in range(path_count(ods, f"{base}.ion"))
             if _hydrogenic(_get(ods, f"{base}.ion.{k}.element.0.z_n"))]
    if len(found) > 1:
        raise ValueError(f"{base} stores {len(found)} hydrogenic ions; one main-ion temperature is read, not a mix")
    return found[0] if found else None


def _check_fast_ion_slice(fast_ion: FastIonEstimate, shape: tuple, time: Optional[float],
                          rho: Any, tolerance: float) -> None:
    """Refuse a fast-ion estimate made on another grid or at another time than the slice."""
    if np.shape(fast_ion.p_fast) != shape:
        raise ValueError(f"fast_ion is on a grid of shape {np.shape(fast_ion.p_fast)}, the slice on {shape}")
    if fast_ion.time is not None and time is not None and abs(fast_ion.time - time) > tolerance:
        raise ValueError(f"fast_ion was estimated at t = {fast_ion.time:g} s, the slice is at t = {time:g} s")
    if fast_ion.rho is not None and rho is not None and not np.allclose(fast_ion.rho, np.asarray(rho, float)):
        raise ValueError("fast_ion was estimated on another rho grid than the slice")


def infer_kinetic_closure(
    ods: Any,
    *,
    time: Optional[float] = None,
    tolerance: float = 5e-4,
    dilution: str = "species",
    composition: Any = None,
    machine_preset: Any = None,
    shot: Optional[int] = None,
    ti_te_ratio: Optional[float] = None,
    ti_record: Optional[str] = None,
    fast_ion: Optional[FastIonEstimate] = None,
) -> KineticClosure:
    """Densities, temperatures and separated pressures of one core_profiles slice, dilution-aware.

    Parameters
    ----------
    ods : ODS
        Source of n_e, T_e and (when stored) the main-ion T_i; read without
        creating paths [any].
    time : float, optional
        Slice time; required when the IDS holds several slices [s].
    tolerance : float, optional
        Largest ``|t_slice - time|`` accepted [s].
    dilution : str, optional
        ``species`` (composition-resolved ion densities) or ``none`` (the
        legacy n_i = n_e single hydrogenic ion) [-].
    composition : ImpurityComposition, optional
        Explicit composition for :func:`vaft.process.impurity.resolve_impurity_composition` [any].
    machine_preset : str or mapping, optional
        Machine preset for the same resolver (``"vest"``) [any].
    shot : int, optional
        Shot whose preset era applies [-].
    ti_te_ratio : float, optional
        T_i = ratio T_e for every ion when the slice stores no T_i [-].
    ti_record : str, optional
        Provenance text for that ratio (e.g. the VEST policy record) [-].
    fast_ion : FastIonEstimate, optional
        A fast-ion estimate on the same grid, added as ``p_fast`` [any].

    Returns
    -------
    KineticClosure
        Composition, ion densities and temperatures, the
        :class:`PressureAssembly` and the provenance of every input [any].

    Raises
    ------
    ValueError
        An unknown dilution mode, no slice at the time, no n_e or T_e, no
        stored T_i and no ratio, several hydrogenic ions, a composition that
        cannot be resolved, or a fast-ion estimate from another grid or time.

    Processing steps
    ----------------
    1. Match the slice by time; read n_e, T_e and the main-ion T_i if stored.
    2. ``species``: resolve the composition (stored ions, explicit, preset;
       #1565 precedence) and form n_s = n_e (n_s/n_e) with the diluted main
       ion; ``none``: one hydrogenic ion with n_i = n_e.
    3. Ion temperatures: the stored main-ion T_i for every ion, else the
       ratio times T_e (recorded as assumed).
    4. Assemble p_e, p_i, p_thermal; add ``fast_ion`` as p_fast.

    Input semantics
    ---------------
    A fitted ``core_profiles`` slice (Thomson n_e, T_e; T_i measured or not).

    Output semantics
    ----------------
    Derived kinetic densities and separated pressures; nothing is written.

    Convention
    ----------
    ``dilution="none"`` reproduces ``pressure_thermal = e n_e (T_e + T_i)`` of
    :func:`vaft.process.profile.core_profiles_from_eq_ratio` exactly; the
    ``species`` mode differs from it by ``e n_e T_i (sum_s n_s/n_e - 1)``.

    Applicability
    -------------
    Machine-independent.  A machine preset reaches it only as ``machine_preset``.

    Limitations
    -----------
    Impurities share the main-ion temperature, read from the one hydrogenic
    ``ion[]`` entry (by ``element.0.z_n``, not position).  The fast-ion
    pressure is the caller's estimate; it must share the slice's grid (and
    its time and rho when the estimate records them).  No deposition model
    is run here.

    Provenance
    ----------
    .. [issue] #1606 Sec. 1-2 and 5; #1565 (composition).
    """
    from vaft.process.impurity import _get, _slice_index, _slice_time, resolve_impurity_composition

    if dilution not in DILUTION_MODES:
        raise ValueError(f"dilution must be one of {DILUTION_MODES}, got {dilution!r}")
    index = _slice_index(ods, time, tolerance)
    if index is None:
        raise ValueError(f"no core_profiles slice within {tolerance:g} s of t = {time!r}")
    base = f"core_profiles.profiles_1d.{index}"
    ne = _get(ods, f"{base}.electrons.density_thermal")
    ne = _get(ods, f"{base}.electrons.density") if ne is None else ne
    te = _get(ods, f"{base}.electrons.temperature")
    if ne is None or te is None:
        raise ValueError(f"{base} needs electrons.density and electrons.temperature")
    ne, te = np.asarray(ne, dtype=float), np.asarray(te, dtype=float)
    rho = _get(ods, f"{base}.grid.rho_tor_norm")
    main_ion = _main_ion_path(ods, base)
    stored_ti = None if main_ion is None else _get(ods, f"{main_ion}.temperature")
    provenance: dict[str, Any] = {"slice": base, "dilution": dilution}
    if stored_ti is not None:
        ti = np.broadcast_to(np.asarray(stored_ti, dtype=float), ne.shape)
        provenance["ti"] = {"source": f"{main_ion}.temperature",
                            "record": _get(ods, f"{main_ion}.temperature_fit.parameters")}
    elif ti_te_ratio is not None:
        ti = float(ti_te_ratio) * te
        provenance["ti"] = {"source": "ti_te_ratio", "ratio": float(ti_te_ratio), "status": "assumed",
                            "record": ti_record}
    else:
        raise ValueError(f"{base} stores no ion temperature; pass ti_te_ratio (and its record)")

    resolved = None
    if dilution == "none":
        densities = {"H+": ne.copy()}
        provenance["composition"] = "legacy n_i = n_e, one hydrogenic ion (explicit fallback)"
    else:
        resolved = resolve_impurity_composition(ods, time=time, tolerance=tolerance, composition=composition,
                                                machine_preset=machine_preset, shot=shot)
        main = np.broadcast_to(np.asarray(resolved.main_ion_fraction, dtype=float), ne.shape)
        fractions = np.asarray(resolved.impurity_fractions, dtype=float)
        if fractions.ndim == 1:
            fractions = np.broadcast_to(fractions, ne.shape + fractions.shape)
        densities = {f"{resolved.main_ion}+": ne * main}
        for j, item in enumerate(resolved.species):
            densities[item.label] = ne * fractions[..., j]
        provenance["composition"] = {"kind": resolved.kind, "source": resolved.source,
                                     "zeff_source": resolved.zeff_source}
    temperatures = {label: ti for label in densities}
    if fast_ion is not None:
        _check_fast_ion_slice(fast_ion, ne.shape, _slice_time(ods, index), rho, tolerance)
        provenance["fast_ion"] = dict(fast_ion.provenance) | {"beam_energy_eV": fast_ion.beam_energy_eV,
                                                              "A_b": fast_ion.A_b}
    pressure = assemble_pressure(ne, te, densities, temperatures,
                                 p_fast=None if fast_ion is None else fast_ion.p_fast, provenance=provenance)
    return KineticClosure(time=_slice_time(ods, index), rho=None if rho is None else np.asarray(rho, float),
                          n_e=ne, T_e=te, ion_densities=densities, ion_temperatures=temperatures,
                          pressure=pressure, composition=resolved, fast_ion=fast_ion, provenance=provenance)
