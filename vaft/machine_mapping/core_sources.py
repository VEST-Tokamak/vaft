"""`core_sources` IDS mapping for NUBEAM results.

Architecture (issue #170, and the precedent in :mod:`vaft.machine_mapping.mhd_linear`):
this module is the IDS-populating layer only. It never reopens a NUBEAM output
file -- it reads the solver-native container owned by :mod:`vaft.code.nubeam`
and copies across only what has a defensible home in IMAS.

**NUBEAM writes per-zone integrals, not densities.** ``pbe`` is the watts
deposited in a zone, ``curbeam`` the amps driven in it, ``tqbe`` the newton
metres. This is not an inference from the units attributes alone: summing each
profile reproduces NUBEAM's own end-of-step totals to four significant figures
on the validated VEST case -- 113.76 kW against the power balance's 113.8 kW
of electron heating, 5713.3 W against 5713 W to the ions, and 191.22 A against
the step log's 191.2 A of driven current. Reading them as densities is wrong by
roughly the zone volume, a factor of about a hundred.

That makes the mapping unusually clean, because IMAS carries both forms:

* the cumulative ``*_inside`` fields are in W, A and N.m, which is what NUBEAM
  already has. A running sum is the exact quantity the schema documents, with
  no division and nothing derived. Its last value is the total, so the mapping
  self-checks against the step log.
* the density fields are what a transport code reads, and are obtained by
  dividing by the zone volume collected alongside the profiles.

Both are written. The cumulative form is exact; the density form is a
derivation, and says so in ``code.parameters``.

Field names follow IMAS 3.41.0, which is what OMAS validates an ODS against
here. Data dictionary 4 renames every toroidal field ``tor`` -> ``phi``
(``momentum_tor`` becomes ``momentum_phi``, and so on); that rename is
milestone 17's to make across the package, not something to anticipate in one
module with a compatibility shim.

**The driven current is the exception, and it is not a copy of anything.**
NUBEAM reports a *toroidal* current per zone -- the Plasma State names the
family for it, ``curt`` being "total enclosed toroidal current". IMAS asks for
``j_parallel = <J.B>/B0``. These are different quantities, and on a spherical
tokamak they differ by a lot: for the validated VEST case the proper value is
1.48-1.59 times a zone-area quotient, and the parallel current inside the
boundary is 293.6 A against 191.2 A of toroidal current, 53% apart. ``<B^2>``
greatly exceeds ``B0^2`` there, from the large poloidal field and from 61%
paramagnetism.

NUBEAM publishes no beam-driven ``<J.B>``. The Plasma State's ``jdotb`` is the
*total* plasma value and an input -- ``state_changes.cdf`` never updates it --
so it cannot stand in. Both IMAS fields are therefore **derived**, by
:func:`vaft.process.equilibrium.parallel_current_from_toroidal`, under the
explicit assumption that the driven current is field-aligned on each flux
surface. When the flux-surface averages that conversion needs are absent,
neither field is written: a proxy that looks like an IMAS quantity is worse
than an absent one.

Quantities with no defensible home stay in the native container: the FRANTIC
halo and recombination channels (which the reference cases put 15-28% away from
this build against a much smaller Monte Carlo noise floor), the flux-surface
moment coefficients, and the shine-through and orbit-loss scalars.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from omas import ODS

from vaft.ods_access import path_count
from vaft.process.equilibrium import parallel_current_from_toroidal

__all__ = [
    "NBI_SOURCE_INDEX",
    "core_sources_from_nubeam",
]

#: ``core_source_identifier`` index for neutral beam injection, read from the
#: data dictionary's own enumeration (``imas.identifiers``), not from memory.
NBI_SOURCE_INDEX = 2
NBI_SOURCE_NAME = "nbi"
NBI_SOURCE_DESCRIPTION = "Source from Neutral Beam Injection"

#: NUBEAM profile -> the exact cumulative IMAS field it fills after a running
#: sum. No unit conversion: NUBEAM's per-zone integrals are already in the
#: units these fields document.
_CUMULATIVE = {
    "pbe": ("electrons.power_inside", "W"),
    "pbi": ("total_ion_power_inside", "W"),
    "sbedep": ("electrons.particles_inside", "s^-1"),
}

#: NUBEAM profile -> the density field it fills after dividing by the zone
#: volume, and the measure to divide by.
_DENSITY = {
    "pbe": ("electrons.energy", "volume"),
    "pbi": ("total_ion_energy", "volume"),
    "tqbjxb": ("momentum_tor_j_cross_b_field", "volume"),
}


def _flat(profiles: Any, name: str) -> Optional[np.ndarray]:
    """A NUBEAM profile as a 1-D array, summed over species if resolved."""
    if name not in profiles:
        return None
    values = np.asarray(profiles[name], dtype=float)
    if values.ndim == 0:
        return None
    if values.ndim > 1:
        # Species-resolved profiles are (species, zone); the core_sources
        # entries here are the summed source, so add the species together.
        values = values.reshape(-1, values.shape[-1]).sum(axis=0)
    return values


def _source_position(ods: ODS) -> int:
    """Index of the NBI entry in ``core_sources.source``, appending if new."""
    count = path_count(ods, "core_sources.source")
    for index in range(count):
        if ods.get(f"core_sources.source.{index}.identifier.index", None) == NBI_SOURCE_INDEX:
            return index
    return count


def _ensure_profiles_1d(ods: ODS, source: int, time_index: int) -> None:
    """Grow the ``profiles_1d`` AOS so ``time_index`` is addressable.

    An OMAS array of structures auto-vivifies only at its current length, so a
    caller writing slice 2 before slices 0 and 1 exist would otherwise raise.
    """
    base = f"core_sources.source.{source}.profiles_1d"
    for index in range(path_count(ods, base), time_index + 1):
        ods[base][index]



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


def core_sources_from_nubeam(
    ods: ODS,
    result: Any,
    *,
    time: float = 0.0,
    time_index: int = 0,
    rho: Optional[Sequence[float]] = None,
    b0: Optional[float] = None,
) -> dict[str, Any]:
    """Write a NUBEAM result into ``core_sources`` as an NBI source term.

    *b0* is the vacuum toroidal field IMAS normalizes ``j_parallel`` by. When it
    is not given it is taken from the ODS -- ``core_sources`` first, then the
    equilibrium sitting beside it. Without it the driven-current fields are
    skipped rather than guessed.

    Returns a report of what was written and what was skipped, so a caller can
    see which channels this particular run actually supported rather than
    having to diff the ODS.
    """
    native = getattr(result, "outputs_native", None) or result
    profiles = getattr(native, "profiles", None) or {}
    if not profiles:
        raise ValueError(
            "this NUBEAM result carries no profiles; core_sources needs "
            "state_changes.cdf, which the run did not produce"
        )

    grid = getattr(native, "grid", None)
    if rho is not None:
        edges = np.asarray(rho, dtype=float)
        centres = 0.5 * (edges[:-1] + edges[1:]) if edges.size > 1 else edges
        zone_volume = grid.zone_volume if grid is not None else None
        zone_area = grid.zone_area if grid is not None else None
    elif grid is not None:
        centres = grid.rho_centres
        zone_volume = grid.zone_volume
        zone_area = grid.zone_area
    else:
        raise ValueError(
            "this NUBEAM result carries no radial grid, and none was supplied. "
            "The grid lives in the Plasma State; pass rho= if the run directory "
            "no longer holds it."
        )

    if b0 is None:
        for path in (
            "core_sources.vacuum_toroidal_field.b0",
            "equilibrium.vacuum_toroidal_field.b0",
        ):
            found = ods.get(path, None)
            if found is None:
                continue
            values = np.atleast_1d(np.asarray(found, dtype=float))
            finite = values[np.isfinite(values)]
            if finite.size:
                b0 = float(finite[0])
                break

    source = _source_position(ods)
    ods[f"core_sources.source.{source}.identifier.index"] = NBI_SOURCE_INDEX
    ods[f"core_sources.source.{source}.identifier.name"] = NBI_SOURCE_NAME
    ods[f"core_sources.source.{source}.identifier.description"] = NBI_SOURCE_DESCRIPTION

    _ensure_profiles_1d(ods, source, time_index)
    base = f"core_sources.source.{source}.profiles_1d.{time_index}"
    ods[f"{base}.grid.rho_tor_norm"] = centres
    ods[f"{base}.time"] = float(time)

    written: list[str] = []
    skipped: list[str] = []
    measures = {"volume": zone_volume, "area": zone_area}

    for name, (field, _unit) in _CUMULATIVE.items():
        values = _flat(profiles, name)
        if values is None or values.size != centres.size:
            skipped.append(f"{name} -> {field}")
            continue
        ods[f"{base}.{field}"] = np.cumsum(values)
        written.append(f"{name} -> {field}")

    for name, (field, measure) in _DENSITY.items():
        values = _flat(profiles, name)
        divisor = measures.get(measure)
        if values is None or divisor is None or values.size != centres.size:
            skipped.append(f"{name} -> {field}")
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            density = np.where(divisor > 0.0, values / divisor, 0.0)
        ods[f"{base}.{field}"] = density
        written.append(f"{name} -> {field}")

    # Driven current: derived, never copied. See the module docstring.
    driven = _flat(profiles, "curbeam")
    flux_surface = getattr(native, "flux_surface", None)
    if driven is None or driven.size != centres.size:
        skipped.append("curbeam -> j_parallel, current_parallel_inside")
    elif flux_surface is None or zone_volume is None:
        skipped.append(
            "curbeam -> j_parallel, current_parallel_inside "
            "(no flux-surface averages; the conversion needs <B^2>, <R^-2> and F)"
        )
    else:
        f_c, gm1_c, gm5_c = flux_surface.at_zone_centres()
        if b0 is None:
            skipped.append(
                "curbeam -> j_parallel, current_parallel_inside (no b0)"
            )
        elif not (f_c.size == gm1_c.size == gm5_c.size == centres.size):
            skipped.append(
                "curbeam -> j_parallel, current_parallel_inside "
                "(flux-surface averages are on a different grid)"
            )
        else:
            converted = parallel_current_from_toroidal(
                driven,
                f=f_c,
                gm1=gm1_c,
                gm5=gm5_c,
                shell_volume=zone_volume,
                b0=b0,
                shell_area=zone_area,
            )
            ods[f"{base}.j_parallel"] = converted.j_parallel
            written.append("curbeam -> j_parallel (derived)")
            if converted.current_parallel_inside is not None:
                ods[f"{base}.current_parallel_inside"] = (
                    converted.current_parallel_inside
                )
                written.append("curbeam -> current_parallel_inside (derived)")
            # b0 is time-coordinated in the DD, so it grows with the slice
            # index rather than being written as a scalar.
            _set_time_array(ods, "core_sources.vacuum_toroidal_field.b0",
                            time_index, float(b0))
            _set_time_array(ods, "core_sources.time", time_index, float(time))

    # The collisional torque IMAS asks for is the total to the plasma, which
    # NUBEAM splits between electrons and ions; the JxB part has its own field
    # above and is deliberately not added in here as well.
    collisional = [_flat(profiles, n) for n in ("tqbe", "tqbi")]
    present = [c for c in collisional if c is not None and c.size == centres.size]
    if present:
        total = np.sum(present, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ods[f"{base}.momentum_tor"] = np.where(
                zone_volume > 0.0, total / zone_volume, 0.0
            )
        inside = np.cumsum(total)
        jxb = _flat(profiles, "tqbjxb")
        if jxb is not None and jxb.size == centres.size:
            inside = inside + np.cumsum(jxb)
        ods[f"{base}.torque_tor_inside"] = inside
        written.append("tqbe+tqbi -> momentum_tor")
        written.append("tqbe+tqbi+tqbjxb -> torque_tor_inside")

    _write_provenance(ods, native)
    return {"source": source, "written": written, "skipped": skipped}


def _write_provenance(ods: ODS, native: Any) -> None:
    """Record the code identity and the two mappings that are not exact."""
    ods["core_sources.ids_properties.homogeneous_time"] = 1
    ods["core_sources.code.name"] = "NUBEAM"
    ods["core_sources.code.repository"] = "https://w3.pppl.gov/NTCC/NUBEAM/"

    runid = getattr(native, "runid", "") or ""
    fragment = (
        "<nubeam>"
        f"<runid>{runid}</runid>"
        "<per_zone_to_density>NUBEAM writes per-zone integrals (W, A, N.m). "
        "The *_inside fields carry them exactly as a running sum; the density "
        "fields are those integrals divided by the collected zone volume or "
        "area, and are therefore derived.</per_zone_to_density>"
        "<driven_current>NUBEAM reports a toroidal driven current per zone and "
        "publishes no beam-driven &lt;J.B&gt;. Both j_parallel and "
        "current_parallel_inside are therefore DERIVED from equilibrium "
        "geometry, not copied: j_parallel = lambda &lt;B^2&gt;/B0 with lambda = "
        "2 pi dI_tor / (F &lt;R^-2&gt; dV), under the explicit assumption that the "
        "driven current is field-aligned on each flux surface. The native "
        "per-zone toroidal current is retained in the NUBEAM result container; "
        "IMAS core_sources has no toroidal-current field to hold it."
        "</driven_current>"
        "</nubeam>"
    )
    path = "core_sources.code.parameters"
    existing = ods.get(path, None)
    if not existing:
        ods[path] = f"<parameters>{fragment}</parameters>"
    elif existing.rstrip().endswith("</parameters>"):
        ods[path] = existing.rstrip()[: -len("</parameters>")] + fragment + "</parameters>"
    else:
        ods[path] = existing + fragment
