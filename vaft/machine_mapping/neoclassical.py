"""Project a NEO drift-kinetic result into ``core_profiles`` and ``core_transport``.

This module is the IDS-populating layer only. It never re-parses NEO output
itself -- it reads the solver-native container owned by
:mod:`vaft.code.gacode.neo`, the same split ``vaft/machine_mapping/core_sources.py``
documents for NUBEAM. What NEO produced in NEO's own units stays in
``NeoOutputs``; what has a defensible IMAS home is derived here, and what does
not is reported as skipped rather than forced into a nearby field.

Mapping audit (issue #550, phase 5). "Status" is the claim being made about the
correspondence, not about the run.

=========================== ============================================= =================== =========
NEO native                  IMAS destination                              units               status
=========================== ============================================= =================== =========
``transport.jparB``         ``core_profiles.../j_bootstrap``              A/m^2               derived
``transport.jparB`` (2nd)   ``core_profiles.../conductivity_parallel``     ohm^-1.m^-1         derived
``transport.particle_flux`` ``core_transport.../particles.flux``          m^-2.s^-1           derived
``transport.energy_flux``   ``core_transport.../energy.flux``             W.m^-2              derived
``coordinates.rho_tor_norm````.../grid_flux.rho_tor_norm``, ``grid.``     -                   exact
``theory.*``                --                                            normalised          unmapped
``transport.*_velocity``    --                                            normalised          unmapped
``equilibrium``, ``grid``   --                                            normalised          unmapped
=========================== ============================================= =================== =========

**Nothing is copied.** Both mapped quantities are derived twice over -- once to
leave NEO's normalisation, once to meet IMAS's definition:

*Dimensionalisation.* NEO normalises by ``n_0``, ``v_t0``, ``T_0`` and a
**signed** ``B_unit``, all of which ``out.neo.expnorm`` carries and
:class:`~vaft.code.gacode.neo.outputs.NeoNormalisation` exposes. The constants
are GACODE's own, from ``vgen/src/vgen_compute_neo.f90:223``:
``e*n_0*v_t0`` is exactly ``1.6022 * dens_norm * v_t0`` when ``dens_norm`` is in
10^19 m^-3.

*Definition.* ``<j_par B>/(e n_0 v_t0 B_unit)`` dimensionalises to
``<J.B>/B_unit``, but IMAS documents ``j_bootstrap`` as ``average(J.B)/B0``
with ``B0`` the **vacuum** toroidal field. The two differ by ``B_unit/B0`` --
a factor of about two on VEST, and a sign, since ``B_unit`` is negative there
under the GACODE COCOS 2 convention. Dropping the sign inverts the answer
while leaving the magnitude right, which is why the tests assert it explicitly.

**``global_quantities.current_bootstrap`` is deliberately not written.** It is a
*toroidal* current in amperes with its own anti-clockwise-from-above
convention, not the area integral of a parallel ``<J.B>/B0``. ``core_sources.py``
records that conflation costing 53 percent on VEST; a converted value is not
available here, so the field stays unset and the reason is reported.

**Conductivity comes from a second run.** NEO does not report one from a
transport solve: ``vgen`` obtains it by running the same case again with a unit
parallel electric field and every gradient switched off, so that ``jpar`` is the
response to the field alone (``vgen_compute_neo.f90:240-260``).
:func:`~vaft.code.gacode.neo.run_neo_conductivity_case` stages that run, and
``core_profiles_from_neo`` writes ``conductivity_parallel`` when it is passed as
``conductivity=``. Unlike the bootstrap current it needs no ``B0`` and carries no
sign correction -- the COCOS mirroring cancels in a current-over-field ratio.

**Which Z_eff a comparison must use.** NEO builds its collision operator from the
species list, not from the ``z_eff`` column of ``input.gacode``, which it ignores.
A file written with ``z_eff = 2`` but carrying only hydrogen is a run at
``Z_eff = 1``, and the conductivity is the quantity that exposes it: comparing
NEO against an analytic model evaluated at 2 rather than 1 is a 53 percent error
on VEST 48224. :attr:`~vaft.code.gacode.neo.outputs.NeoOutputs.effective_charge`
reports what a run actually used, and the provenance records it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from omas import ODS

from vaft.ods_access import path_count

__all__ = [
    "NEOCLASSICAL_MODEL_INDEX",
    "core_profiles_from_neo",
    "core_transport_from_neo",
]

#: ``core_transport_model_identifier`` index for a neoclassical model, from the
#: data dictionary's own enumeration, not from memory.
NEOCLASSICAL_MODEL_INDEX = 5
NEOCLASSICAL_MODEL_NAME = "neoclassical"
NEOCLASSICAL_MODEL_DESCRIPTION = "Neoclassical transport, from the NEO drift-kinetic solver"

#: NEO's energy flux is the *total* one (``neo_run.f90:95`` assigns
#: ``neo_dke_out(is,2) = eflux(is)``, documented ``Q_is/Q_norm total``), so the
#: convected part is already in it and IMAS must not add the particle flux
#: again. That is what a zero multiplier means.
FLUX_MULTIPLIER = 0.0

#: GACODE writes densities in 10^19 m^-3 and temperatures in keV.
_DENSITY_SCALE = 1.0e19
_ELEMENTARY_CHARGE = 1.602176634e-19


def _native(result: Any) -> Any:
    """Accept either the run wrapper or the native container itself."""
    return getattr(result, "outputs_native", None) or result


def _set_time_array(ods: ODS, path: str, index: int, value: float) -> None:
    """Write one entry of a time-coordinated leaf, growing it in order.

    Mirrors ``core_sources._set_time_array``; kept local rather than imported
    across modules because that one is private to its own mapper.
    """
    try:
        ods.set_time_array(path, index, value)
    except Exception:
        existing = list(np.atleast_1d(np.asarray(ods.get(path, []), dtype=float)))
        while len(existing) <= index:
            existing.append(float("nan"))
        existing[index] = value
        ods[path] = np.asarray(existing, dtype=float)


def _ensure_aos(ods: ODS, base: str, index: int) -> None:
    """Grow an array of structures so ``index`` is addressable.

    OMAS auto-vivifies only at the current length, so writing slice 2 before 0
    and 1 exist would otherwise raise.
    """
    for position in range(path_count(ods, base), index + 1):
        ods[base][position]


class _Scales:
    """The SI scales of one NEO run, per radial point."""

    def __init__(self, normalisation: Any) -> None:
        self.density = np.asarray(normalisation.density_norm, dtype=float) * _DENSITY_SCALE
        self.velocity = np.asarray(normalisation.velocity_norm_times_a, dtype=float)
        self.temperature = (
            np.asarray(normalisation.temperature_norm, dtype=float) * 1.0e3 * _ELEMENTARY_CHARGE
        )
        self.b_unit = np.asarray(normalisation.b_unit, dtype=float)
        self.minor_radius = np.asarray(normalisation.a_meters, dtype=float)

    @property
    def electric_field(self) -> np.ndarray:
        """``T_0[eV]/a`` [V m^-1]: what a normalised parallel field multiplies by.

        GACODE's own definition, from ``vgen_compute_neo.f90:256``:
        ``e_norm = temp_norm*1000/rmin(n_exp)``, with ``temp_norm`` in keV, so the
        volt comes from dividing an energy in electronvolts by the elementary
        charge -- which is why this divides ``self.temperature`` (joules) by it
        rather than multiplying.
        """
        return self.temperature / _ELEMENTARY_CHARGE / self.minor_radius

    @property
    def current(self) -> np.ndarray:
        """``e n_0 v_t0`` [A m^-2]: what a normalised current multiplies by."""
        return _ELEMENTARY_CHARGE * self.density * self.velocity

    @property
    def particle_flux(self) -> np.ndarray:
        """``n_0 v_t0`` [m^-2 s^-1]."""
        return self.density * self.velocity

    @property
    def energy_flux(self) -> np.ndarray:
        """``n_0 v_t0 T_0`` [W m^-2]."""
        return self.density * self.velocity * self.temperature


def _preconditions(native: Any) -> tuple[Optional[_Scales], Optional[np.ndarray], list[str]]:
    """The two products a mapping needs, or the reasons it cannot proceed."""
    skipped: list[str] = []
    scales = grid = None
    if getattr(native, "normalisation", None) is None:
        skipped.append(
            "everything (no out.neo.expnorm: NEO writes the SI scales only for "
            "PROFILE_MODEL >= 2, and without them nothing can leave NEO's normalisation)"
        )
    else:
        scales = _Scales(native.normalisation)
    coordinates = getattr(native, "coordinates", None)
    if coordinates is None or "rho_tor_norm" not in coordinates:
        skipped.append(
            "everything (no out.neo.exprhon: NEO writes the r/a -> rho_tor_norm bridge "
            "only for PROFILE_MODEL >= 2, and IMAS profiles need a flux coordinate)"
        )
    else:
        grid = np.asarray(coordinates["rho_tor_norm"], dtype=float)
    return scales, grid, skipped


def _species_order(native: Any) -> tuple[Optional[int], list[int]]:
    """``(electron index, ion indices)`` in NEO's own species order.

    The electron is the species carrying *negative* charge, not merely the
    smallest one: with adiabatic electrons (``AE_FLAG=1``) NEO writes only ions,
    and taking the minimum would label the lowest-Z ion an electron and drop it
    from the ion list at the same time.
    """
    charge = getattr(native, "species_charge", None)
    if charge is None:
        return None, []
    charge = np.atleast_1d(np.asarray(charge, dtype=float))
    negative = np.flatnonzero(charge < 0.0)
    electron = int(negative[0]) if negative.size else None
    return electron, [index for index in range(charge.size) if index != electron]


def _resolve_b0(
    ods: ODS, b0: Optional[float], *, time: float, time_index: int
) -> Optional[float]:
    """The vacuum field IMAS normalises ``j_bootstrap`` by, for this slice.

    VEST's b0 drifts by up to a factor of two within a shot (#325), so which
    entry is read is a physical question, not a rounding one.

    ``core_profiles`` is preferred, because that is the field IMAS defines
    ``j_bootstrap`` against -- but it is indexed by ``time_index``, its own slice
    number, and skipped when it has no entry there. This mapper writes that leaf
    itself, so a partially populated one is usually its own earlier output;
    reading it back by nearest-time would let the first slice answer for every
    later one. ``equilibrium`` keeps its own time base, which need not align, so
    it is matched on *time*.
    """
    if b0 is not None:
        return float(b0)

    def usable(values: np.ndarray, index: int) -> Optional[float]:
        if index < 0 or index >= values.size:
            return None
        if not np.isfinite(values[index]) or values[index] == 0.0:
            return None
        return float(values[index])

    path = "core_profiles.vacuum_toroidal_field.b0"
    if path in ods:
        found = usable(np.atleast_1d(np.asarray(ods[path], dtype=float)), int(time_index))
        if found is not None:
            return found

    path = "equilibrium.vacuum_toroidal_field.b0"
    if path in ods:
        values = np.atleast_1d(np.asarray(ods[path], dtype=float))
        index = 0
        if "equilibrium.time" in ods:
            times = np.atleast_1d(np.asarray(ods["equilibrium.time"], dtype=float))
            if times.size:
                index = int(np.argmin(np.abs(times - float(time))))
        found = usable(values, index)
        if found is not None:
            return found
    return None


def _onto(target: np.ndarray, source: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, int]:
    """Place *values* onto *target*, leaving NaN where *source* does not reach.

    ``np.interp`` clamps outside its range, which would publish the edge value
    across a region NEO never solved. Returns the array and how many points were
    left unevaluated.
    """
    order = np.argsort(source)
    # anti-alias: spatial. Both grids are rho_tor_norm, a flux coordinate, so
    # there is no sampling rate here to reduce and nothing to fold (#425). The
    # target is the equilibrium's own radial grid, which is finer than NEO's
    # handful of solved surfaces, so this is a refinement rather than a
    # decimation in any case.
    placed = np.interp(target, source[order], values[order])
    outside = (target < source.min()) | (target > source.max())
    placed[outside] = np.nan
    return placed, int(np.count_nonzero(outside))


def core_profiles_from_neo(
    ods: ODS,
    result: Any,
    *,
    time: float = 0.0,
    time_index: int = 0,
    b0: Optional[float] = None,
    conductivity: Any = None,
) -> dict[str, Any]:
    """Write NEO's bootstrap current, and its conductivity when given both runs.

    Parameters
    ----------
    ods
        The ODS to populate, in place.
    result
        A :class:`~vaft.code.gacode.neo.NEOResult` or the
        :class:`~vaft.code.gacode.neo.NeoOutputs` it carries.
    time, time_index
        The ``core_profiles`` slice this run describes.
    b0
        Vacuum toroidal field to normalise by. Read from the ODS when omitted;
        the mapping is refused if neither supplies one, because ``j_bootstrap``
        is defined as ``<J.B>/B0`` and a different ``B0`` is a different number.
    conductivity
        The companion gradient-free run from
        :func:`~vaft.code.gacode.neo.run_neo_conductivity_case`, whose ``jpar`` is
        the response to a unit parallel field. Given one, ``conductivity_parallel``
        is written too. A transport run passed here would be read as a conductivity
        and would be wrong by whatever the bootstrap drive contributes, so the run
        is checked for the ``EPAR0=1`` signature rather than trusted.

    Returns
    -------
    dict
        ``{"written": [...], "skipped": [...]}`` -- which channels this
        particular run supported, so a caller need not diff the ODS.
    """
    native = _native(result)
    scales, grid, skipped = _preconditions(native)
    written: list[str] = []

    transport = getattr(native, "transport", None)
    current = None if transport is None else transport.get("bootstrap_current")
    if current is None:
        skipped.append("jparB -> j_bootstrap (the run wrote no out.neo.transport)")
    field = _resolve_b0(ods, b0, time=time, time_index=time_index)
    if field is None:
        skipped.append(
            "jparB -> j_bootstrap (no vacuum_toroidal_field.b0 on the ODS and none "
            "supplied; IMAS defines j_bootstrap as <J.B>/B0, so it cannot be formed)"
        )

    if scales is not None and grid is not None and current is not None and field is not None:
        base = f"core_profiles.profiles_1d.{time_index}"
        _ensure_aos(ods, "core_profiles.profiles_1d", time_index)
        # <j.B>/B_unit, then into IMAS's <J.B>/B0. B_unit is per-radius and
        # signed; both matter.
        values = np.asarray(current, dtype=float) * scales.current * scales.b_unit / field

        existing = ods.get(f"{base}.grid.rho_tor_norm", None)
        existing = None if existing is None else np.atleast_1d(np.asarray(existing, dtype=float))
        usable = True
        if existing is None or existing.size == 0:
            ods[f"{base}.grid.rho_tor_norm"] = grid
            ods[f"{base}.j_bootstrap"] = values
        else:
            # The slice's grid is its contract with every other profile on it --
            # ne, Te, the ion channels. Replacing it with NEO's would orphan all
            # of them, so the current is placed onto the grid that is already
            # there. NEO solves a handful of surfaces, so outside its span there
            # is no value to place: those points are left NaN rather than
            # extrapolated to something plausible.
            placed, unevaluated = _onto(existing, grid, values)
            # A single-surface run cannot interpolate anywhere, so every point
            # comes back unevaluated. Writing an all-NaN profile and calling it
            # written would tell a caller gating on the report that the IDS now
            # carries a bootstrap current.
            usable = unevaluated < existing.size
            if usable:
                ods[f"{base}.j_bootstrap"] = placed
            else:
                skipped.append(
                    f"j_bootstrap entirely (NEO solved rho_tor_norm "
                    f"{grid.min():.3f}-{grid.max():.3f} at {grid.size} "
                    f"{'surface' if grid.size == 1 else 'surfaces'}, which reaches no "
                    f"point of this slice's {existing.size}-point grid)"
                )
            if unevaluated and usable:
                skipped.append(
                    f"j_bootstrap at {unevaluated} of {existing.size} grid points "
                    f"(NEO solved rho_tor_norm {grid.min():.3f}-{grid.max():.3f}; "
                    "outside that span the value is left unevaluated, not extrapolated)"
                )
        _set_time_array(ods, "core_profiles.time", time_index, float(time))
        ods["core_profiles.ids_properties.homogeneous_time"] = 1
        # Per slice, not once: j_bootstrap is <J.B>/B0, so a reader needs the
        # field *this* slice was divided by. Writing it only the first time
        # leaves a short array beside a growing time base, and the next slice
        # then reads the wrong entry back.
        _set_time_array(ods, "core_profiles.vacuum_toroidal_field.b0", time_index, field)
        if usable:
            written.append("j_bootstrap")

    skipped.append(
        "jparB -> global_quantities.current_bootstrap (that field is a toroidal "
        "current in amperes with its own sign convention, not the area integral of a "
        "parallel <J.B>/B0; NEO publishes no such conversion)"
    )
    if conductivity is None:
        skipped.append(
            "conductivity_parallel (NEO gives it only from a second run with EPAR0=1 "
            "and the gradient scales zeroed; pass one as conductivity=, staged by "
            "vaft.code.gacode.neo.run_neo_conductivity_case)"
        )
    else:
        written.extend(
            _write_conductivity(
                ods, conductivity, time=time, time_index=time_index, skipped=skipped
            )
        )

    if written:
        _write_provenance(ods, native, "core_profiles", conductivity=conductivity)
    return {"written": written, "skipped": skipped}


def _write_conductivity(
    ods: ODS,
    result: Any,
    *,
    time: float,
    time_index: int,
    skipped: list[str],
) -> list[str]:
    """Write ``conductivity_parallel`` from the gradient-free companion run.

    ``sigma = jpar * (e n_0 v_t0) / (T_0[eV]/a)`` -- ``vgen``'s own recipe
    (``vgen_compute_neo.f90:256-258``). Unlike the bootstrap current this needs no
    ``B0`` and no sign correction: ``jpar`` and the field that drove it are both in
    NEO's frame, so the COCOS mirroring cancels in the ratio and a conductivity
    that came out negative would mean something is wrong rather than that VEST runs
    the other way round.
    """
    native = _native(result)
    if not getattr(native, "solved", False):
        skipped.append(
            "conductivity_parallel (the companion run did not solve: "
            + "; ".join(getattr(native, "errors", ()) or ("no reason recorded",))
            + ")"
        )
        return []

    # A transport run and a conductivity run differ only in their input, so the
    # output alone cannot tell them apart. Refusing to guess is the point.
    parameters = _run_parameters(result)
    if parameters is not None and not _is_conductivity_run(parameters):
        skipped.append(
            "conductivity_parallel (the run passed as conductivity= was not staged "
            "with EPAR0=1 and zeroed gradient scales, so its jpar is a bootstrap "
            "current and not a response to a parallel field)"
        )
        return []

    scales, grid, reasons = _preconditions(native)
    if scales is None or grid is None:
        skipped.extend(reason.replace("everything", "conductivity_parallel", 1) for reason in reasons)
        return []
    current = native.bootstrap_current
    if current is None:
        skipped.append(
            "conductivity_parallel (the companion run wrote no out.neo.transport)"
        )
        return []

    values = np.asarray(current, dtype=float) * scales.current / scales.electric_field
    finite = np.isfinite(values)
    if not finite.any() or np.any(values[finite] <= 0.0):
        # sigma = jpar/E_par with both in NEO's frame: the COCOS mirroring cancels,
        # so a non-positive value is not a convention but a wrong run -- a transport
        # solve passed as the companion produces exactly this, and would otherwise
        # be written as a conductivity whenever its settings could not be read.
        skipped.append(
            "conductivity_parallel (the companion run's jpar does not dimensionalise "
            "to a positive conductivity, which a response to a parallel field always "
            "is; the run passed as conductivity= is almost certainly a transport one)"
        )
        return []
    base = f"core_profiles.profiles_1d.{time_index}"
    _ensure_aos(ods, "core_profiles.profiles_1d", time_index)
    existing = ods.get(f"{base}.grid.rho_tor_norm", None)
    existing = None if existing is None else np.atleast_1d(np.asarray(existing, dtype=float))
    if existing is None or existing.size == 0:
        ods[f"{base}.grid.rho_tor_norm"] = grid
        ods[f"{base}.conductivity_parallel"] = values
    else:
        placed, unevaluated = _onto(existing, grid, values)
        if unevaluated >= existing.size:
            skipped.append(
                f"conductivity_parallel entirely (the companion run solved "
                f"rho_tor_norm {grid.min():.3f}-{grid.max():.3f}, which reaches no "
                f"point of this slice's {existing.size}-point grid)"
            )
            return []
        ods[f"{base}.conductivity_parallel"] = placed
        if unevaluated:
            skipped.append(
                f"conductivity_parallel at {unevaluated} of {existing.size} grid "
                f"points (NEO solved rho_tor_norm {grid.min():.3f}-{grid.max():.3f}; "
                "outside that span the value is left unevaluated, not extrapolated)"
            )
    _set_time_array(ods, "core_profiles.time", time_index, float(time))
    ods["core_profiles.ids_properties.homogeneous_time"] = 1
    return ["conductivity_parallel"]


def _conductivity_fragment(result: Any) -> str:
    """How the conductivity was obtained, or that it was not.

    Two runs produced this IDS and they are not interchangeable, so the record
    names each one's settings: a reader who finds j_bootstrap and
    conductivity_parallel side by side would otherwise have no way to know the
    second came from a different solve of the same plasma.
    """
    if result is None:
        return (
            "<conductivity_parallel status=\"unmapped\" reason=\"no companion run "
            "was supplied; NEO yields a conductivity only from a second run with "
            "EPAR0=1 and the gradient scales zeroed\"/>"
        )
    native = _native(result)
    version = getattr(native, "version", None) or {}
    revision = str(version.get("revision", "")).replace("<", "").replace(">", "")
    charge = getattr(native, "effective_charge", None)
    z_eff = "" if charge is None else f" z_eff=\"{float(np.nanmean(np.asarray(charge))):.4g}\""
    return (
        "<conductivity_parallel run=\"second, gradient-free\" "
        "settings=\"EPAR0=1 with PROFILE_DLNNDR_*_SCALE and PROFILE_DLNTDR_*_SCALE "
        "zeroed for every species\" "
        "derived_by=\"jpar * e n_0 v_t0 / (T_0[eV]/a)\" "
        "recipe=\"vgen/src/vgen_compute_neo.f90:256-258\" "
        f"revision=\"{revision}\"{z_eff} units=\"ohm^-1.m^-1\"/>"
    )


def _run_parameters(result: Any) -> Optional[Mapping[str, Any]]:
    """The ``input.neo`` settings behind a result, from wherever they survive.

    Three places, because the common one is the last: a :class:`NEOResult` carries
    them in provenance, a staged case carries them directly, and a bare
    :class:`NeoOutputs` -- what ``collect_neo_outputs`` returns, and what most
    callers have -- carries none, but knows the directory it was read from, and
    ``input.neo`` is still sitting in it.

    Returning ``None`` means the settings could not be found at all, which is the
    only case where the caller's word has to be taken.
    """
    provenance = getattr(result, "provenance", None) or {}
    staged = provenance.get("parameters")
    if staged:
        return staged
    # A staged case, passed before it was run.
    parameters = getattr(result, "parameters", None)
    if parameters:
        return parameters
    native = _native(result)
    directory = getattr(native, "directory", None)
    if directory is None:
        return None
    path = Path(directory) / "input.neo"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    settings: dict[str, Any] = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            settings[key.strip().upper()] = value.strip()
    return settings or None


def _is_conductivity_run(parameters: Mapping[str, Any]) -> bool:
    """Whether these settings are the gradient-free unit-field case."""
    try:
        if float(parameters.get("EPAR0", 0.0)) == 0.0:
            return False
    except (TypeError, ValueError):
        return False
    scales = [
        value
        for key, value in parameters.items()
        if str(key).startswith("PROFILE_DLNNDR_") or str(key).startswith("PROFILE_DLNTDR_")
    ]
    return bool(scales) and all(float(value) == 0.0 for value in scales)


def core_transport_from_neo(
    ods: ODS,
    result: Any,
    *,
    time: float = 0.0,
    time_index: int = 0,
) -> dict[str, Any]:
    """Write NEO's particle and energy fluxes into ``core_transport``.

    The model entry is identified as neoclassical (index 5) and is reused on a
    second call rather than duplicated.

    Returns
    -------
    dict
        ``{"model": int, "written": [...], "skipped": [...]}``.
    """
    native = _native(result)
    scales, grid, skipped = _preconditions(native)
    written: list[str] = []

    transport = getattr(native, "transport", None)
    if transport is None:
        skipped.append("fluxes (the run wrote no out.neo.transport)")
    electron, ions = _species_order(native)
    if getattr(native, "species_charge", None) is None:
        skipped.append(
            "fluxes (the run wrote no out.neo.species, so a flux cannot be attributed "
            "to a species)"
        )
    elif electron is None:
        # AE_FLAG=1: NEO treats the electrons adiabatically and writes only ions.
        # The ion fluxes are still real and are mapped; there is simply no
        # electron channel to fill.
        skipped.append(
            "electrons.particles.flux, electrons.energy.flux (this run carries no "
            "electron species, which is what an adiabatic-electron run looks like)"
        )

    model = _model_position(ods)
    if scales is None or grid is None or transport is None or not ions:
        return {"model": model, "written": written, "skipped": skipped}

    _ensure_aos(ods, "core_transport.model", model)
    ods[f"core_transport.model.{model}.identifier.index"] = NEOCLASSICAL_MODEL_INDEX
    ods[f"core_transport.model.{model}.identifier.name"] = NEOCLASSICAL_MODEL_NAME
    ods[f"core_transport.model.{model}.identifier.description"] = NEOCLASSICAL_MODEL_DESCRIPTION
    ods[f"core_transport.model.{model}.flux_multiplier"] = FLUX_MULTIPLIER

    base = f"core_transport.model.{model}.profiles_1d.{time_index}"
    _ensure_aos(ods, f"core_transport.model.{model}.profiles_1d", time_index)
    # Fluxes live on grid_flux; the data dictionary ties `.flux` leaves to it.
    # grid_d and grid_v stay unset because NEO produces fluxes, not a
    # diffusivity/convection split, and inventing one would be a claim.
    ods[f"{base}.grid_flux.rho_tor_norm"] = grid
    ods[f"{base}.time"] = float(time)

    particle = np.atleast_2d(np.asarray(transport["particle_flux"], dtype=float))
    energy = np.atleast_2d(np.asarray(transport["energy_flux"], dtype=float))
    if electron is not None:
        ods[f"{base}.electrons.particles.flux"] = particle[electron] * scales.particle_flux
        ods[f"{base}.electrons.energy.flux"] = energy[electron] * scales.energy_flux
        written.extend(["electrons.particles.flux", "electrons.energy.flux"])

    mass = getattr(native, "species_mass", None)
    charge = np.atleast_1d(np.asarray(native.species_charge, dtype=float))
    for position, species in enumerate(ions):
        ion = f"{base}.ion.{position}"
        ods[f"{ion}.particles.flux"] = particle[species] * scales.particle_flux
        ods[f"{ion}.energy.flux"] = energy[species] * scales.energy_flux
        ods[f"{ion}.z_ion"] = float(charge[species])
        if mass is not None:
            # NEO carries mass relative to deuterium; input.gacode's own unit is
            # amu, which is what IMAS wants.
            ods[f"{ion}.element.0.a"] = float(np.atleast_1d(mass)[species]) * 2.0
            ods[f"{ion}.element.0.z_n"] = float(charge[species])
        written.extend([f"ion.{position}.particles.flux", f"ion.{position}.energy.flux"])

    skipped.append(
        "momentum flux (NEO writes it only with a rotation model, and its frame is "
        "not declared by core_transport)"
    )
    skipped.append(
        "grid_d, grid_v (NEO produces fluxes, not a diffusivity/convection split)"
    )

    _set_time_array(ods, "core_transport.time", time_index, float(time))
    ods["core_transport.ids_properties.homogeneous_time"] = 1
    field = _resolve_b0(ods, None, time=time, time_index=time_index)
    if field is not None:
        _set_time_array(ods, "core_transport.vacuum_toroidal_field.b0", time_index, field)
    if "equilibrium.vacuum_toroidal_field.r0" in ods:
        ods["core_transport.vacuum_toroidal_field.r0"] = float(
            np.ravel(np.asarray(ods["equilibrium.vacuum_toroidal_field.r0"], dtype=float))[0]
        )

    _write_provenance(ods, native, "core_transport")
    return {"model": model, "written": written, "skipped": skipped}


def _model_position(ods: ODS) -> int:
    """Index of the neoclassical entry in ``core_transport.model``, appending if new."""
    count = path_count(ods, "core_transport.model")
    for index in range(count):
        if (
            ods.get(f"core_transport.model.{index}.identifier.index", None)
            == NEOCLASSICAL_MODEL_INDEX
        ):
            return index
    return count


def _write_provenance(ods: ODS, native: Any, ids: str, *, conductivity: Any = None) -> None:
    """Record the solver identity and every correspondence that is not exact.

    The caveats are attributes rather than prose so that a consumer can test
    them, following ``mhd_linear``'s convention. Angle brackets are escaped:
    a literal ``</parameters>`` in a fragment closes the envelope early, and
    the splice does not check (issue #642).
    """
    version = getattr(native, "version", None) or {}
    revision = str(version.get("revision", "")).replace("<", "").replace(">", "")
    ods[f"{ids}.ids_properties.homogeneous_time"] = 1
    ods[f"{ids}.code.name"] = "NEO"
    ods[f"{ids}.code.repository"] = "https://github.com/gafusion/gacode"
    if revision:
        ods[f"{ids}.code.version"] = revision

    fragment = (
        "<neo>"
        f"<revision>{revision}</revision>"
        "<normalisation>NEO writes normalised quantities. They are dimensionalised "
        "here with the scales in out.neo.expnorm, using GACODE's own constants "
        "(vgen_compute_neo.f90): e*n_0*v_t0 for a current, n_0*v_t0 for a particle "
        "flux, n_0*v_t0*T_0 for an energy flux.</normalisation>"
        "<bootstrap_current derived_by=\"jparB * e n_0 v_t0 * B_unit / B0\" "
        "definition=\"IMAS j_bootstrap is average(J.B)/B0 with B0 the vacuum field; "
        "NEO's dimensionalised jparB is average(J.B)/B_unit. B_unit is per-radius "
        "and signed, and differs from B0 by about a factor of two on VEST.\" "
        "quantity=\"drift-kinetic solve, not NEO's Sauter or Redl theory column\" "
        "units=\"A/m^2\"/>"
        "<energy_flux flux_multiplier=\"0\" definition=\"NEO's eflux is the total "
        "energy flux, so the particle flux must not be added again.\" "
        "note=\"With a rotation model this total is a lab-frame quantity in a "
        "rotating plasma, and core_transport declares no frame.\"/>"
        "<unmapped>Flows, poloidal and toroidal velocities, the analytic theory "
        "columns and the Hirshman-Sigmar fluxes have no definitional IMAS home and "
        "stay in the NEO result container. global_quantities.current_bootstrap is a "
        "toroidal current, not the integral of this parallel one.</unmapped>"
        + _conductivity_fragment(conductivity)
        + "</neo>"
    )
    path = f"{ids}.code.parameters"
    existing = ods.get(path, None)
    if not existing:
        ods[path] = f"<parameters>{fragment}</parameters>"
    elif fragment in existing:
        # Mapping a second time slice re-runs this, and the note is about the
        # mapping rather than any one slice; appending would leave one identical
        # block per slice.
        return
    elif existing.rstrip().endswith("</parameters>"):
        ods[path] = existing.rstrip()[: -len("</parameters>")] + fragment + "</parameters>"
    else:
        ods[path] = existing + fragment
