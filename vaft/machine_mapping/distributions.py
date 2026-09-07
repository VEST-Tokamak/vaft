"""`distributions` IDS mapping for NUBEAM results.

Architecture (issue #170, and the precedent in
:mod:`vaft.machine_mapping.core_sources`): this module is the IDS-populating
layer only. It never reopens a NUBEAM output file -- it reads the solver-native
container owned by :mod:`vaft.code.nubeam`.

`core_sources` answers "what does the beam do to the plasma". This IDS answers
the other half of issue #490 section 6: **what the fast-ion population itself
is**. NUBEAM computes both, and only the first had a home.

**Two families of profile, and NUBEAM writes them differently.** The units are
not guesswork here; they are the ``units`` attributes of the Plasma State
NUBEAM writes, read from the reference ``d3d_output_state.cdf`` that ships with
the distribution:

* ``nbeami`` is ``m^-3`` ("beam species density"), and ``eperp_beami`` /
  ``epll_beami`` are ``keV`` ("beam species <Eperp>, lab frame"). These are
  intensive -- a density and two mean energies per particle -- and are copied or
  combined, never divided by a zone measure.
* ``pbe``, ``pbi`` (``W``), ``curbeam`` (``A``), ``tqbe``, ``tqbi``,
  ``tqbjxb`` (``Nt*m``) and ``sbtherm`` (``#/sec``) are per-zone *integrals*.
  A running total of them is exact; a density is that integral divided by the
  zone measure.

That split is why ``global_quantities`` here is the exact half of this mapping
and ``profiles_1d`` the derived half, which is the reverse of nothing else in
the package but follows directly from what NUBEAM writes.

**The toroidal driven current finally has a home.** ``core_sources`` offers only
``j_parallel = <J.B>/B0``, so NUBEAM's native *toroidal* per-zone current has to
be converted there under an explicit field-aligned assumption -- a conversion
worth 53% on the validated VEST case. ``distributions`` asks for
``current_tor``, a toroidal current *density*, and
``global_quantities.current_tor``, a toroidal current in amps. Both are what
NUBEAM already has: the first is a division by the zone area, the second is a
plain sum. Neither needs the field-aligned assumption, so the native quantity
survives into IMAS unmodified for the first time.

The field is ``current_tor`` and not ``current_fast_tor`` because NUBEAM writes
``curbeam`` *shielded* -- its own long name says so -- meaning the electron
back-current is already accounted for. That is exactly the distinction IMAS
draws between the two fields, and ``current_fast_tor``, which excludes the
back-current, is left absent because NUBEAM does not publish it.

**Species resolution decides what may be written.** ``nbeami``,
``eperp_beami``, ``epll_beami`` and ``sbtherm`` are ``(species, zone)``; the
collisional and current profiles are already summed over species. One
``distribution`` entry is written per beam species, so when a run has more than
one the summed profiles cannot be attributed to any single entry, and they are
skipped and reported rather than repeated into each -- which would triple-count
on any consumer that adds the entries up.

Field names follow IMAS 3.41.0, which is what OMAS validates an ODS against
here. Data dictionary 4 renames every toroidal field ``tor`` -> ``phi``; that
rename is milestone 17's to make across the package.

Quantities with no defensible home stay in the native container: the birth
markers and lost-particle records (``distribution_sources`` is the next
increment), the FRANTIC neutral channels, and the shine-through scalars.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from omas import ODS

from vaft.ods_access import path_count

__all__ = [
    "NBI_PROCESS_INDEX",
    "distributions_from_nubeam",
]

#: ``distribution/process/type`` index for neutral beam injection. The data
#: dictionary states it in the field's own documentation: "index=1 for NBI".
NBI_PROCESS_INDEX = 1
NBI_PROCESS_NAME = "NBI"
NBI_PROCESS_DESCRIPTION = "Source from neutral beam injection"

#: Joules per keV. The elementary charge is exact in the SI, so this is too.
_JOULES_PER_KEV = 1.602176634e-16

#: Species-resolved NUBEAM profile -> the intensive IMAS field it fills with no
#: conversion at all. NUBEAM's units are already the ones IMAS documents.
_COPIED = {
    "nbeami": "density_fast",
}

#: Summed-over-species per-zone integral -> (density field, zone measure) and
#: the exact cumulative field in ``global_quantities``.
_INTEGRALS = {
    "pbe": ("collisions.electrons.power_thermal", "volume"),
    "pbi": ("collisions.ion.0.power_thermal", "volume"),
    "tqbe": ("collisions.electrons.torque_thermal_tor", "volume"),
    "tqbi": ("collisions.ion.0.torque_thermal_tor", "volume"),
    "tqbjxb": ("torque_tor_j_radial", "volume"),
    "pbth": ("thermalisation.energy", "volume"),
    # "shielded" in NUBEAM means the electron back-current is already
    # accounted for, so this is the net driven current. IMAS spells that
    # current_tor; current_fast_tor is the unshielded fast-ion current,
    # which NUBEAM does not publish here.
    "curbeam": ("current_tor", "area"),
}

#: The same profiles again, as the ``global_quantities`` totals. Summing a
#: per-zone integral is exact -- no division, nothing derived -- which is why
#: these are listed separately rather than derived from the densities above.
_TOTALS = {
    "pbe": "collisions.electrons.power_thermal",
    "pbi": "collisions.ion.0.power_thermal",
    "tqbe": "collisions.electrons.torque_thermal_tor",
    "tqbi": "collisions.ion.0.torque_thermal_tor",
    "tqbjxb": "torque_tor_j_radial",
    "pbth": "thermalisation.power",
    "curbeam": "current_tor",
}


def _profile(profiles: Any, name: str) -> Optional[np.ndarray]:
    """A NUBEAM profile as an array, or None when the run did not write it."""
    if name not in profiles:
        return None
    values = np.asarray(profiles[name], dtype=float)
    return values if values.ndim else None


def _summed(profiles: Any, name: str) -> Optional[np.ndarray]:
    """A profile flattened to one radial array, adding species if resolved."""
    values = _profile(profiles, name)
    if values is None:
        return None
    if values.ndim > 1:
        values = values.reshape(-1, values.shape[-1]).sum(axis=0)
    return values


def _per_species(profiles: Any, name: str) -> Optional[list[np.ndarray]]:
    """One radial array per beam species.

    The last axis is radial in every Plasma State profile, so a 2-D array is
    ``(species, zone)`` and a 1-D one is a single species.
    """
    values = _profile(profiles, name)
    if values is None:
        return None
    if values.ndim == 1:
        return [values]
    flat = values.reshape(-1, values.shape[-1])
    return [flat[index] for index in range(flat.shape[0])]


def _species_count(profiles: Any) -> int:
    """How many beam species this run resolved, from the profiles that say so."""
    for name in ("nbeami", "eperp_beami", "epll_beami", "sbtherm"):
        rows = _per_species(profiles, name)
        if rows:
            return len(rows)
    return 1


def _ensure(ods: ODS, base: str, index: int) -> None:
    """Grow an array of structures so *index* is addressable.

    An OMAS array of structures auto-vivifies only at its current length, so a
    caller writing slice 2 before slices 0 and 1 exist would otherwise raise.
    """
    for position in range(path_count(ods, base), index + 1):
        ods[base][position]


def _distribution_positions(ods: ODS, count: int) -> list[int]:
    """Indices of this run's NBI distributions, appending any that are new.

    Reusing the existing entries is what makes a second call with the same
    result a no-op rather than a duplicate population.
    """
    existing = [
        index
        for index in range(path_count(ods, "distributions.distribution"))
        if ods.get(f"distributions.distribution.{index}.process.0.type.index", None)
        == NBI_PROCESS_INDEX
    ]
    positions = existing[:count]
    next_free = path_count(ods, "distributions.distribution")
    while len(positions) < count:
        positions.append(next_free)
        next_free += 1
    return positions


def _set_time_array(ods: ODS, path: str, index: int, value: float) -> None:
    """Write one entry of a time-coordinated leaf, growing it in order."""
    try:
        ods.set_time_array(path, index, value)
    except Exception:
        # Older OMAS, or a leaf it declines to treat as a time array.
        existing = list(np.atleast_1d(np.asarray(ods.get(path, []), dtype=float)))
        while len(existing) <= index:
            existing.append(float("nan"))
        existing[index] = value
        ods[path] = np.asarray(existing, dtype=float)


def distributions_from_nubeam(
    ods: ODS,
    result: Any,
    *,
    time: float = 0.0,
    time_index: int = 0,
    rho: Optional[Sequence[float]] = None,
    species: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Write a NUBEAM result into ``distributions`` as the fast-ion population.

    *species* names the beam species, one label per entry, in the order the
    Plasma State resolves them. NUBEAM's own output carries the profiles but
    not the labels, so an unnamed species is left unnamed rather than guessed.

    Returns a report of what was written and what was skipped, so a caller can
    see which channels this particular run supported without diffing the ODS.
    """
    native = getattr(result, "outputs_native", None) or result
    profiles = getattr(native, "profiles", None) or {}
    if not profiles:
        raise ValueError(
            "this NUBEAM result carries no profiles; distributions needs "
            "state_changes.cdf, which the run did not produce"
        )

    grid = getattr(native, "grid", None)
    if rho is not None:
        edges = np.asarray(rho, dtype=float)
        centres = 0.5 * (edges[:-1] + edges[1:]) if edges.size > 1 else edges
    elif grid is not None:
        centres = grid.rho_centres
    else:
        raise ValueError(
            "this NUBEAM result carries no radial grid, and none was supplied. "
            "The grid lives in the Plasma State; pass rho= if the run directory "
            "no longer holds it."
        )
    zone_volume = grid.zone_volume if grid is not None else None
    zone_area = grid.zone_area if grid is not None else None
    measures = {"volume": zone_volume, "area": zone_area}

    # A grid of the wrong length matches no profile, so every channel would be
    # skipped and the caller would get a structurally valid IDS holding no
    # NUBEAM data at all -- plus a code.parameters block asserting how it was
    # derived. Refuse instead, and name the likeliest cause: rho is the zone
    # boundaries, one point more than a profile, not the centres.
    lengths = {
        int(np.asarray(values).shape[-1])
        for values in profiles.values()
        if np.asarray(values).ndim
    }
    if lengths and centres.size not in lengths:
        raise ValueError(
            f"the radial grid gives {centres.size} zone centres, which matches "
            f"no profile in this result (profiles are {sorted(lengths)} points "
            f"across). rho is the zone boundaries, one point more than a "
            f"profile, not the zone centres."
        )

    count = _species_count(profiles)
    positions = _distribution_positions(ods, count)
    written: list[str] = []
    skipped: list[str] = []

    for ordinal, position in enumerate(positions):
        stem = f"distributions.distribution.{position}"
        ods[f"{stem}.process.0.type.index"] = NBI_PROCESS_INDEX
        ods[f"{stem}.process.0.type.name"] = NBI_PROCESS_NAME
        ods[f"{stem}.process.0.type.description"] = NBI_PROCESS_DESCRIPTION
        # NUBEAM sums over every injector and every energy component before it
        # writes a profile, and the data dictionary spells that sum as 0.
        ods[f"{stem}.process.0.nbi_unit"] = 0
        ods[f"{stem}.process.0.nbi_beamlets_group"] = 0
        ods[f"{stem}.process.0.nbi_energy.index"] = 0
        ods[f"{stem}.process.0.nbi_energy.name"] = "sum"

        if species is not None and ordinal < len(species) and species[ordinal]:
            ods[f"{stem}.species.ion.label"] = str(species[ordinal])
            written.append(f"species[{ordinal}] -> species.ion.label")
        elif species is not None:
            skipped.append(f"species[{ordinal}] -> species.ion.label (not named)")

        _ensure(ods, f"{stem}.profiles_1d", time_index)
        _ensure(ods, f"{stem}.global_quantities", time_index)
        base = f"{stem}.profiles_1d.{time_index}"
        totals = f"{stem}.global_quantities.{time_index}"
        ods[f"{base}.grid.rho_tor_norm"] = centres
        ods[f"{base}.time"] = float(time)
        ods[f"{totals}.time"] = float(time)

        _write_species_fields(
            ods, base, totals, profiles, ordinal, centres, zone_volume,
            written, skipped,
        )
        if count == 1:
            _write_summed_fields(
                ods, base, totals, profiles, centres, measures, written, skipped,
            )
        elif ordinal == 0:
            skipped.append(
                f"pbe, pbi, tqbe, tqbi, tqbjxb, curbeam -> profiles_1d and "
                f"global_quantities ({count} beam species, and NUBEAM writes "
                f"these summed over species; attributing them to one entry "
                f"would be wrong and repeating them would double-count)"
            )

    if species is not None and len(species) > count:
        skipped.append(
            f"species[{count}:] -> species.ion.label "
            f"({len(species)} labels given for {count} beam species)"
        )

    _set_time_array(ods, "distributions.time", time_index, float(time))
    _write_provenance(ods, native)
    return {"distributions": positions, "written": written, "skipped": skipped}


def _write_species_fields(
    ods: ODS,
    base: str,
    totals: str,
    profiles: Any,
    ordinal: int,
    centres: np.ndarray,
    zone_volume: Optional[np.ndarray],
    written: list[str],
    skipped: list[str],
) -> None:
    """The species-resolved half: density, the two pressures, thermalisation."""
    for name, field in _COPIED.items():
        rows = _per_species(profiles, name)
        if rows is None or ordinal >= len(rows) or rows[ordinal].size != centres.size:
            skipped.append(f"{name}[{ordinal}] -> {field}")
            continue
        ods[f"{base}.{field}"] = rows[ordinal]
        written.append(f"{name}[{ordinal}] -> {field}")

    density = _per_species(profiles, "nbeami")
    perp = _per_species(profiles, "eperp_beami")
    par = _per_species(profiles, "epll_beami")

    def _row(rows: Optional[list[np.ndarray]]) -> Optional[np.ndarray]:
        if rows is None or ordinal >= len(rows):
            return None
        return rows[ordinal] if rows[ordinal].size == centres.size else None

    n = _row(density)
    e_perp = _row(perp)
    e_par = _row(par)

    # p_parallel = integral of m v_par^2 f, which is twice the parallel energy
    # density, while p_perp = integral of (1/2) m v_perp^2 f is the
    # perpendicular energy density itself. NUBEAM reports mean energies per
    # particle, so both follow from multiplying by the density.
    if n is not None and e_par is not None:
        parallel = 2.0 * n * e_par * _JOULES_PER_KEV
        ods[f"{base}.pressure_fast_parallel"] = parallel
        written.append("nbeami x epll_beami -> pressure_fast_parallel (derived)")
        if zone_volume is not None and zone_volume.size == centres.size:
            ods[f"{totals}.energy_fast_parallel"] = float(
                np.sum(n * e_par * _JOULES_PER_KEV * zone_volume)
            )
            written.append("nbeami x epll_beami -> energy_fast_parallel (derived)")
    else:
        skipped.append("nbeami x epll_beami -> pressure_fast_parallel")

    if n is not None and e_par is not None and e_perp is not None:
        # The scalar pressure, (p_par + 2 p_perp) / 3, which is two thirds of
        # the fast-ion energy density and reduces to it for an isotropic
        # population.
        ods[f"{base}.pressure_fast"] = (
            (2.0 / 3.0) * n * (e_par + e_perp) * _JOULES_PER_KEV
        )
        written.append("nbeami x (epll+eperp) -> pressure_fast (derived)")
        if zone_volume is not None and zone_volume.size == centres.size:
            ods[f"{totals}.energy_fast"] = float(
                np.sum(n * (e_par + e_perp) * _JOULES_PER_KEV * zone_volume)
            )
            written.append("nbeami x (epll+eperp) -> energy_fast (derived)")
    else:
        skipped.append("nbeami x (epll+eperp) -> pressure_fast")

    if n is not None and zone_volume is not None and zone_volume.size == centres.size:
        ods[f"{totals}.particles_fast_n"] = float(np.sum(n * zone_volume))
        written.append("nbeami -> particles_fast_n (derived)")
    else:
        skipped.append("nbeami -> particles_fast_n")

    thermalised = _per_species(profiles, "sbtherm")
    if (
        thermalised is not None
        and ordinal < len(thermalised)
        and thermalised[ordinal].size == centres.size
    ):
        row = thermalised[ordinal]
        ods[f"{totals}.thermalisation.particles"] = float(np.sum(row))
        written.append("sbtherm -> global thermalisation.particles")
        if zone_volume is not None and zone_volume.size == centres.size:
            ods[f"{base}.thermalisation.particles"] = _density(row, zone_volume)
            written.append("sbtherm -> thermalisation.particles (derived)")
    else:
        skipped.append(f"sbtherm[{ordinal}] -> thermalisation.particles")


def _write_summed_fields(
    ods: ODS,
    base: str,
    totals: str,
    profiles: Any,
    centres: np.ndarray,
    measures: dict[str, Optional[np.ndarray]],
    written: list[str],
    skipped: list[str],
) -> None:
    """The half NUBEAM already summed over species: collisions and current."""
    for name, field in _TOTALS.items():
        values = _summed(profiles, name)
        if values is None or values.size != centres.size:
            skipped.append(f"{name} -> global_quantities.{field}")
            continue
        ods[f"{totals}.{field}"] = float(np.sum(values))
        written.append(f"{name} -> global_quantities.{field}")

    for name, (field, measure) in _INTEGRALS.items():
        values = _summed(profiles, name)
        divisor = measures.get(measure)
        if (
            values is None
            or divisor is None
            or values.size != centres.size
            or divisor.size != centres.size
        ):
            skipped.append(f"{name} -> {field}")
            continue
        ods[f"{base}.{field}"] = _density(values, divisor)
        written.append(f"{name} -> {field} (derived)")


def _density(values: np.ndarray, measure: np.ndarray) -> np.ndarray:
    """A per-zone integral as a density, leaving degenerate zones at zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(measure > 0.0, values / measure, 0.0)


def _write_provenance(ods: ODS, native: Any) -> None:
    """Record the code identity and every quantity that is not a copy."""
    ods["distributions.ids_properties.homogeneous_time"] = 1
    ods["distributions.code.name"] = "NUBEAM"
    ods["distributions.code.repository"] = "https://w3.pppl.gov/NTCC/NUBEAM/"

    runid = getattr(native, "runid", "") or ""
    fragment = (
        "<nubeam>"
        f"<runid>{runid}</runid>"
        "<exact>global_quantities holds sums of NUBEAM's own per-zone "
        "integrals, so current_tor, the collisional powers and torques, and "
        "thermalisation.particles are exact -- no division and nothing "
        "derived. density_fast is copied unchanged; NUBEAM writes nbeami in "
        "m^-3 already.</exact>"
        "<densities>Every profiles_1d field other than density_fast is DERIVED "
        "by dividing a per-zone integral (W, A, N.m, #/sec) by the zone volume "
        "or area collected alongside the profiles.</densities>"
        "<pressures>NUBEAM reports mean energies per particle, not energy "
        "densities: eperp_beami and epll_beami are in keV. pressure_fast_"
        "parallel is DERIVED as 2 n &lt;E_par&gt;, and pressure_fast as "
        "(2/3) n (&lt;E_par&gt; + &lt;E_perp&gt;), the scalar pressure "
        "(p_par + 2 p_perp)/3.</pressures>"
        "<lumped_ions>pbi and tqbi are NUBEAM's sums over every thermal ion ""species, and they are written to collisions.ion[0] because IMAS has ""no lumped entry. Read that entry as the total to the thermal ions, ""not as the main ion: summing collisions.ion[:] gives the right ""number, reading ion[0] alone as one species over-counts it.""</lumped_ions>"
        "<toroidal_current>current_tor, in both profiles_1d and "
        "global_quantities, is NUBEAM's own toroidal driven current -- divided "
        "by the zone area and summed respectively. Unlike core_sources."
        "j_parallel it needs no field-aligned assumption, so this is the "
        "native quantity rather than a conversion of it. curbeam is shielded, "
        "so it belongs in current_tor (which includes the electron "
        "back-current) and not in current_fast_tor (which excludes it); "
        "NUBEAM does not publish the unshielded current, so that field is left "
        "absent.</toroidal_current>"
        "</nubeam>"
    )
    path = "distributions.code.parameters"
    existing = ods.get(path, None)
    if not existing:
        ods[path] = f"<parameters>{fragment}</parameters>"
    elif fragment in existing:
        # Mapping a second time slice into the same ODS re-runs this, and the
        # note is about the mapping rather than about any one slice. Appending
        # it again would leave one identical block per slice.
        return
    elif existing.rstrip().endswith("</parameters>"):
        ods[path] = existing.rstrip()[: -len("</parameters>")] + fragment + "</parameters>"
    else:
        ods[path] = existing + fragment
