"""Project TGLF's turbulent fluxes into IMAS ``core_transport`` (issue #553 section 14).

The counterpart of :mod:`vaft.machine_mapping.neoclassical`, and it inherits that
module's rule: a quantity reaches an IDS only when the correspondence survives an audit
by physical definition, and everything else is *reported* as a reason rather than left
silently absent.

Three things make this harder than the NEO mapping, and each is settled here against
GACODE's own source rather than by inference.

**TGLF reports gyro-Bohm units, and the conversion is local.** NEO writes its SI scales
into ``out.neo.expnorm``; TGLF is a local code and writes none. The units come from
:class:`~vaft.code.gacode.tglf.inputs.TGLFNormalisation`, which carries what
``prepare_tglf_input`` already computed for ``BETAE``/``XNUE``/``DEBYE`` --
``gamma_gb = ne c_s (rho_s/a)^2``, ``q_gb = ne k Te c_s (rho_s/a)^2``,
``pi_gb = ne k Te a (rho_s/a)^2``, from ``tgyro/src/tgyro_profile_functions.f90:74-85``.

**TGLF runs at ``r/a`` and ``core_transport`` needs ``rho_tor_norm``.** NEO gets that
bridge from ``out.neo.exprhon``. Here it comes from the profile that defined both, which
is why this function takes one: the alternative is to assume the two coordinates agree,
and on VEST 48224 ``r/a = 0.8`` is ``rho_tor_norm = 0.71``.

**The momentum flux carries a sign that cancels.** ``tglf/src/tglf_LS.f90:1009`` already
multiplies the toroidal stress by ``SIGN_IT``, and ``tgyro/src/tgyro_flux.f90:199,206``
multiplies TGLF's answer by ``-SIGN_IT`` again. The product is ``-1`` whatever the sign
convention, so the physical stress is the *negative* of what ``out.tglf.gbflux`` carries
-- and a mapping that "corrected" for ``SIGN_IT`` once would be right only for a plasma
whose current runs one way.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
from omas import ODS

from vaft.ods_access import path_count

#: ``core_transport_model_identifier`` index for turbulent transport, from the data
#: dictionary's own enumeration (``core_transport_identifier.xml``: ``anomalous`` is 6,
#: "Representation of turbulent transport"), not from memory.
ANOMALOUS_MODEL_INDEX = 6
ANOMALOUS_MODEL_NAME = "anomalous"
ANOMALOUS_MODEL_DESCRIPTION = (
    "Turbulent transport, from the TGLF quasilinear model"
)

#: ``core_transport`` adds ``flux_multiplier`` times the particle flux to the energy
#: flux to obtain the total. TGLF's energy flux is already the total: its weight is
#: built from the full pressure moment, ``1.5*p_tot - 0.5*p_par`` scaled by ``1.5``
#: (``tglf/src/tglf_LS.f90:1001,1007``), so it carries the convective part and must not
#: have it added again. NEO's mapping is 0 for the same reason.
FLUX_MULTIPLIER = 0.0

#: What the raw ``out.tglf.gbflux`` momentum column must be multiplied by to become a
#: physical toroidal stress. See the module docstring: TGLF applies ``SIGN_IT`` and
#: TGYRO applies ``-SIGN_IT``, so the two cancel to a plain inversion.
MOMENTUM_SIGN = -1.0

__all__ = [
    "ANOMALOUS_MODEL_DESCRIPTION",
    "ANOMALOUS_MODEL_INDEX",
    "ANOMALOUS_MODEL_NAME",
    "FLUX_MULTIPLIER",
    "MOMENTUM_SIGN",
    "core_transport_from_tglf",
]


def _set_time_array(ods: ODS, path: str, index: int, value: float) -> None:
    """Write one entry of a time-coordinated leaf, growing it in order."""
    try:
        ods.set_time_array(path, index, value)
    except Exception:
        existing = list(np.atleast_1d(np.asarray(ods.get(path, []), dtype=float)))
        while len(existing) <= index:
            existing.append(float("nan"))
        existing[index] = value
        ods[path] = np.asarray(existing, dtype=float)


def _ensure_aos(ods: ODS, base: str, index: int) -> None:
    """Grow an array of structures so ``index`` is addressable."""
    for position in range(path_count(ods, base), index + 1):
        ods[base][position]


def _model_position(ods: ODS) -> int:
    """Index of the turbulent entry in ``core_transport.model``, appending if new.

    Reused on a second call rather than duplicated, so that writing a neoclassical and
    a turbulent model into one ODS leaves two entries and not four.
    """
    count = path_count(ods, "core_transport.model")
    for index in range(count):
        if (
            ods.get(f"core_transport.model.{index}.identifier.index", None)
            == ANOMALOUS_MODEL_INDEX
        ):
            return index
    return count


def _rho_tor_norm(profile: Any, radii: np.ndarray) -> Optional[np.ndarray]:
    """Map ``r/a`` onto ``rho_tor_norm`` using the profile that defines both.

    ``GACODEProfile.rho`` is ``sqrt(Phi/Phi_boundary)`` and ``rmin`` is the minor radius
    in metres, so the bridge is an interpolation of one against the other. Returns None
    when the profile cannot supply it, which the caller reports rather than guesses at.
    """
    rmin = getattr(profile, "rmin", None)
    rho = getattr(profile, "rho", None)
    if rmin is None or rho is None:
        return None
    rmin = np.asarray(rmin, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if rmin.size < 2 or rmin.size != rho.size or not np.isfinite(rmin[-1]) or rmin[-1] <= 0:
        return None
    # np.interp requires an increasing xp and returns silently wrong values rather than
    # raising when it is not, which would put every flux at a plausible wrong radius.
    if not np.all(np.diff(rmin) > 0.0):
        return None
    # anti-alias: not a time series and not a downsample. This evaluates a monotone
    # radial coordinate map -- r/a against rho_tor_norm, both from the same 123-point
    # profile -- at the handful of surfaces TGLF was run on. There is no sample rate to
    # reduce and no band to alias; the only failure mode is extrapolating past the
    # profile's own range, which `prepare_tglf_input` already refuses upstream.
    return np.interp(np.asarray(radii, dtype=float), rmin / rmin[-1], rho)


def _channel(outputs: Any, name: str) -> Optional[np.ndarray]:
    gbflux = getattr(outputs, "gbflux", None)
    if not gbflux or name not in gbflux:
        return None
    return np.atleast_1d(np.asarray(gbflux[name], dtype=float))


def core_transport_from_tglf(
    ods: ODS,
    surfaces: Iterable[tuple[Any, Any]],
    profile: Any,
    *,
    time: float = 0.0,
    time_index: int = 0,
) -> dict[str, Any]:
    """Write TGLF's particle, energy and momentum fluxes into ``core_transport``.

    Parameters
    ----------
    ods
        The ODS to write into. The model entry is identified as anomalous (index 6) and
        is reused on a second call rather than duplicated.
    surfaces
        ``(TGLFInput, TGLFResult)`` pairs, one per flux surface, in any order. Both
        halves are needed and neither carries the other: the input holds the
        normalisation that leaves gyro-Bohm units behind, the result holds the fluxes.
        A pair whose result did not solve is dropped and named.
    profile
        The :class:`~vaft.code.gacode._profiles.GACODEProfile` the inputs were built
        from, for the ``r/a`` to ``rho_tor_norm`` bridge TGLF cannot supply.
    time, time_index
        Which ``profiles_1d`` slice to write.

    Returns
    -------
    dict
        ``{"model": int, "written": [...], "skipped": [...]}``.
    """
    written: list[str] = []
    skipped: list[str] = []

    usable: list[tuple[Any, Any]] = []
    for local, result in surfaces:
        native = getattr(result, "outputs_native", result)
        if native is None or not getattr(native, "solved", False):
            skipped.append(f"r/a = {getattr(local, 'rho', '?')} (the run did not solve)")
            continue
        if getattr(local, "normalisation", None) is None:
            skipped.append(
                f"r/a = {getattr(local, 'rho', '?')} (the local input carries no "
                f"normalisation, so its gyro-Bohm fluxes cannot be dimensionalised)"
            )
            continue
        usable.append((local, native))

    model = _model_position(ods)
    if not usable:
        skipped.append("everything (no surface both solved and carried a normalisation)")
        return {"model": model, "written": written, "skipped": skipped}

    usable.sort(key=lambda pair: float(pair[0].rho))
    radii = np.array([float(local.rho) for local, _ in usable])
    grid = _rho_tor_norm(profile, radii)
    if grid is None:
        skipped.append(
            "everything (the profile carries no rmin/rho pair, so r/a cannot be "
            "expressed as the rho_tor_norm core_transport indexes fluxes by)"
        )
        return {"model": model, "written": written, "skipped": skipped}

    species = list(usable[0][0].names)
    # Both counts, because they are different numbers: `names` is what the input
    # declared and `n_species` is what the run wrote into out.tglf.grid, and it is the
    # second one that sizes the arrays indexed below. Checking only the first would
    # leave an adiabatic-electron run -- fewer output species than input ones -- to
    # raise IndexError out of a mapper whose contract is to report what it cannot write.
    counts = {len(local.names) for local, _ in usable}
    counts |= {int(native.n_species or -1) for _, native in usable}
    if len(counts) != 1:
        skipped.append(
            f"everything (the surfaces do not agree on one species count: the local "
            f"inputs name {sorted({len(local.names) for local, _ in usable})} and the "
            f"runs wrote {sorted({native.n_species for _, native in usable})})"
        )
        return {"model": model, "written": written, "skipped": skipped}

    _ensure_aos(ods, "core_transport.model", model)
    ods[f"core_transport.model.{model}.identifier.index"] = ANOMALOUS_MODEL_INDEX
    ods[f"core_transport.model.{model}.identifier.name"] = ANOMALOUS_MODEL_NAME
    ods[f"core_transport.model.{model}.identifier.description"] = ANOMALOUS_MODEL_DESCRIPTION
    ods[f"core_transport.model.{model}.flux_multiplier"] = FLUX_MULTIPLIER

    base = f"core_transport.model.{model}.profiles_1d.{time_index}"
    _ensure_aos(ods, f"core_transport.model.{model}.profiles_1d", time_index)
    # Fluxes live on grid_flux; the data dictionary ties `.flux` leaves to it. grid_d
    # and grid_v stay unset because TGLF produces fluxes, not a diffusivity/convection
    # split, and inventing one would be a claim.
    ods[f"{base}.grid_flux.rho_tor_norm"] = grid
    ods[f"{base}.time"] = float(time)

    def profile_of(channel: str, index: int, scale: str, sign: float = 1.0) -> np.ndarray:
        values = []
        for local, native in usable:
            column = _channel(native, channel)
            unit = getattr(local.normalisation, scale)
            values.append(np.nan if column is None else sign * column[index] * unit)
        return np.asarray(values, dtype=float)

    absent = [name for name in ("particle", "energy", "momentum")
              if any(_channel(native, name) is None for _, native in usable)]
    for name in absent:
        skipped.append(f"{name} flux (a surface wrote none)")

    if "particle" not in absent:
        ods[f"{base}.electrons.particles.flux"] = profile_of("particle", 0, "particle_flux")
        written.append("electrons.particles.flux")
    if "energy" not in absent:
        ods[f"{base}.electrons.energy.flux"] = profile_of("energy", 0, "energy_flux")
        written.append("electrons.energy.flux")

    charges = np.atleast_1d(np.asarray(usable[0][0].zs, dtype=float))
    masses = np.atleast_1d(np.asarray(usable[0][0].mass, dtype=float))
    for position in range(1, len(species)):
        ion = f"{base}.ion.{position - 1}"
        if "particle" not in absent:
            ods[f"{ion}.particles.flux"] = profile_of("particle", position, "particle_flux")
            written.append(f"ion.{position - 1}.particles.flux")
        if "energy" not in absent:
            ods[f"{ion}.energy.flux"] = profile_of("energy", position, "energy_flux")
            written.append(f"ion.{position - 1}.energy.flux")
        ods[f"{ion}.label"] = str(species[position])
        ods[f"{ion}.z_ion"] = float(charges[position])
        # TGLF carries mass relative to deuterium; input.gacode's own unit is amu,
        # which is what IMAS wants.
        ods[f"{ion}.element.0.a"] = float(masses[position]) * 2.0
        ods[f"{ion}.element.0.z_n"] = float(charges[position])

    # The toroidal stress is only a prediction when the run was given a rotation to
    # predict it from. `prepare_gacode_profile` does not populate `w0` today, so
    # VEXB_SHEAR and VPAR reach input.tglf as TGLF's own zeros and what comes back is
    # the numerical residue of a plasma that was told not to rotate -- sign-alternating
    # and five orders below the energy channel on VEST. Writing that as
    # `momentum_tor.flux` would hand a consumer noise wearing the shape of a result.
    # NEO's mapper refuses the same quantity for the same reason.
    # The value decides, not the provenance: an input built outside
    # `prepare_tglf_input` can carry `vexb_shear=None` with an empty provenance, and
    # reading only the record would let that write a momentum flux from a run that was
    # given no rotation. The record is consulted as well, for an input that carries a
    # number its own converter marked unavailable.
    rotating = all(
        local.vexb_shear is not None and "vexb_shear" not in local.missing()
        for local, _ in usable
    )
    if "momentum" not in absent and not rotating:
        skipped.append(
            "momentum_tor.flux (every surface records vexb_shear as unavailable, so "
            "TGLF was given no rotation and its toroidal stress is the residue of a "
            "plasma told not to rotate, not a prediction of one)"
        )
    elif "momentum" not in absent:
        # Summed over the species, because `momentum_tor` is one channel for the
        # plasma's toroidal angular momentum while TGLF resolves the stress per
        # species. The electrons are included: small, but not zero, and dropping them
        # would be a choice rather than a definition.
        total = sum(
            profile_of("momentum", index, "momentum_flux", MOMENTUM_SIGN)
            for index in range(len(species))
        )
        ods[f"{base}.momentum_tor.flux"] = total
        written.append("momentum_tor.flux")

    skipped.append(
        "the exchange channel (it is a power density, W/m^3, not a flux: turbulent "
        "energy exchange between species belongs in core_sources and core_transport "
        "has no home for it)"
    )
    skipped.append(
        "grid_d, grid_v (TGLF produces fluxes, not a diffusivity/convection split)"
    )
    skipped.append(
        "conductive and convective components (TGLF's energy weight is built from the "
        "full pressure moment and is not split, which is why flux_multiplier is 0)"
    )

    _set_time_array(ods, "core_transport.time", time_index, float(time))
    ods["core_transport.ids_properties.homogeneous_time"] = 1

    return {"model": model, "written": written, "skipped": skipped}
