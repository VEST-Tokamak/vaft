"""Project the classical (Braginskii) heat-flux baseline into IMAS ``core_transport`` (#1654).

The counterpart of :mod:`vaft.machine_mapping.turbulence` and
:mod:`vaft.machine_mapping.neoclassical`, for the per-surface records that
:func:`vaft.process.transport_state.classical_heat_fluxes` produces (#1435).

**What is a flux and what is not.** A record carries three kinds of quantity, and
only the first is projected as a flux:

- *Physical fluxes of a stated reduced model*: ``electron_energy_flux_W_m2`` and the
  main ion's ``ion_energy_flux_W_m2``, the perpendicular conductive heat flux
  ``q = -kappa dT/dr``. These go to ``energy.flux`` on ``grid_flux``.
- *Transport coefficients of the same model*: ``chi_e_m2_s`` and ``chi_i_m2_s``, with
  ``q = n chi (-dT/dr)``. These go to ``energy.d`` on ``grid_d``, never to a flux leaf.
- *Diagnostics*: collision times, Coulomb logarithms, ``gamma1_perp`` and ``|B|``. They
  stay in the native record; none has a definitional ``core_transport`` home.

The order-of-magnitude ``nu rho^2`` reference scale (#780/#1112) is not produced by
#1435 and is never written here. The record's particle flux is ``None``: the model
does not evaluate it. ``particles.flux`` is therefore left unset, which is "not
evaluated", not zero.

**The identifier is private.** The data dictionary's ``core_transport_identifier``
has no classical entry (0-6 and 19-25; ``neoclassical`` is 5 and is a different
physics model). Its own rule for anything else is a negative index. Index ``-1``
alone is not an identity, since any producer may use it, so a classical entry is
recognised by index *and* name.

**One entry per formulation and EFIT lineage.** ``code.parameters`` records the
formulation (its SHA-256 and text), the EFIT lineage and, per time slice, the
resolved ``state_identity``. Two lineages or two formulations never share an entry,
so a summary cannot average across them.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from typing import Any, Iterable, Optional

import numpy as np
from omas import ODS

from vaft.machine_mapping.turbulence import _rho_tor_norm
from vaft.ods_access import path_count

#: Private ``core_transport_identifier`` index. The data dictionary has no classical
#: entry and reserves negative values for private identifiers.
CLASSICAL_MODEL_INDEX = -1
CLASSICAL_MODEL_NAME = "classical"
CLASSICAL_MODEL_DESCRIPTION = (
    "Classical collisional transport: Braginskii perpendicular conductive heat flux "
    "(VAFT private identifier; the data dictionary has no classical entry)"
)

#: ``energy.flux`` is the conductive heat flux alone. The model evaluates no particle
#: flux, so there is no convective part to add, and a consumer that adds
#: ``flux_multiplier`` times an unset particle flux adds nothing.
FLUX_MULTIPLIER = 0.0

#: Two slices closer than this are the same time (the summary layer uses the same 1 us).
_TIME_TOLERANCE_S = 1e-6

__all__ = [
    "CLASSICAL_MODEL_DESCRIPTION",
    "CLASSICAL_MODEL_INDEX",
    "CLASSICAL_MODEL_NAME",
    "FLUX_MULTIPLIER",
    "classical_parameters",
    "core_transport_from_classical",
    "formulation_sha256",
]


def formulation_sha256(model: Any) -> str:
    """SHA-256 of a record's ``model`` mapping, keys sorted: the formulation identity."""
    return hashlib.sha256(json.dumps(model, sort_keys=True).encode("utf-8")).hexdigest()


def classical_parameters(text: Any) -> Optional[dict[str, Any]]:
    """Parse a classical entry's ``code.parameters``.

    Returns ``{"formulation_sha256", "efit_lineage", "formulation", "slices"}``, where
    ``slices`` is a list of ``(time_s, state_identity)``. Returns None when the text is
    absent or is not this module's envelope.
    """
    if not text:
        return None
    try:
        root = ET.fromstring(str(text))
    except ET.ParseError:
        return None
    node = root.find("classical")
    if root.tag != "parameters" or node is None:
        return None
    slices = []
    for entry in node.findall("slice"):
        try:
            slices.append((float(entry.get("time", "nan")), entry.get("state_identity") or None))
        except ValueError:
            slices.append((float("nan"), entry.get("state_identity") or None))
    formulation = node.findtext("formulation")
    return {
        "formulation_sha256": node.get("formulation_sha256") or None,
        "efit_lineage": node.get("efit_lineage") or None,
        "formulation": json.loads(formulation) if formulation else None,
        "slices": slices,
    }


def _model_position(ods: ODS, sha: str, lineage: Optional[str]) -> int:
    """The entry for this formulation and lineage, or the next free index."""
    count = path_count(ods, "core_transport.model")
    for index in range(count):
        prefix = f"core_transport.model.{index}"
        if (ods.get(f"{prefix}.identifier.index", None) != CLASSICAL_MODEL_INDEX
                or ods.get(f"{prefix}.identifier.name", None) != CLASSICAL_MODEL_NAME):
            continue
        parsed = classical_parameters(ods.get(f"{prefix}.code.parameters", None))
        if parsed and parsed["formulation_sha256"] == sha and parsed["efit_lineage"] == lineage:
            return index
    return count


def _envelope(sha: str, lineage: Optional[str], model: Any, slices: list) -> str:
    root = ET.Element("parameters")
    node = ET.SubElement(root, "classical", formulation_sha256=sha, efit_lineage=lineage or "")
    ET.SubElement(node, "formulation").text = json.dumps(model, sort_keys=True)
    ET.SubElement(node, "energy_flux", flux_multiplier="0",
                  definition="conductive heat flux only; the model evaluates no particle "
                             "flux, so the convective part is absent, not zero")
    ET.SubElement(node, "transport_coefficient", leaf="energy.d",
                  definition="chi with q = n chi (-dT/dr); a coefficient of the same "
                             "model, not a reference scale and not a flux")
    for time_s, identity in slices:
        ET.SubElement(node, "slice", time=repr(float(time_s)), state_identity=identity or "")
    return ET.tostring(root, encoding="unicode")


def _ensure_aos(ods: ODS, base: str, index: int) -> None:
    for position in range(path_count(ods, base), index + 1):
        ods[base][position]


def core_transport_from_classical(
    ods: ODS,
    records: Iterable[dict],
    profile: Any,
    *,
    time: float,
    state_identity: str,
    efit_lineage: Optional[str] = None,
) -> dict[str, Any]:
    """Write classical heat fluxes and diffusivities into ``core_transport``.

    Parameters
    ----------
    ods
        The ODS to write into.
    records
        Per-surface results of
        :func:`~vaft.process.transport_state.classical_heat_fluxes`, in any order.
        A surface without a finite flux, or with ``coulomb_log_valid`` False, is
        dropped and named.
    profile
        The :class:`~vaft.code.gacode._profiles.GACODEProfile` the records were built
        from, for the ``r/a`` to ``rho_tor_norm`` bridge.
    time
        The slice time [s]. It is matched against the entry's existing slices, so a
        repeated call for the same state is idempotent. A different state at the same
        time is refused.
    state_identity, efit_lineage
        The resolved state's identity (required; without one nothing is written) and
        EFIT lineage (lane K's State key contract v1, #1454). They are recorded so that
        rows from different states or lineages are never merged.

    Returns
    -------
    dict
        ``{"model": int | None, "profile_index": int | None, "written": [...],
        "skipped": [...]}``. ``model`` is None when nothing was written.
    """
    written: list[str] = []
    skipped: list[str] = []
    usable = []
    if not state_identity:
        # An unknown state cannot be told apart from another unknown state at the same
        # time, so a write without one could silently replace a different plasma.
        skipped.append("everything (no state_identity: rows from different states at one "
                       "time could not be told apart)")
        return {"model": None, "profile_index": None, "written": written, "skipped": skipped}
    for record in records:
        r = record.get("r_over_a")
        q_e = record.get("electron_energy_flux_W_m2")
        ions = record.get("ion_energy_flux_W_m2") or {}
        if not record.get("coulomb_log_valid", False):
            skipped.append(f"r/a = {r} (T_e below 10 eV: the electron Coulomb logarithm "
                           "is not defined, so neither is the flux)")
            continue
        values = [q_e, *ions.values(), record.get("chi_e_m2_s"), record.get("chi_i_m2_s")]
        if len(ions) != 1 or not all(v is not None and math.isfinite(float(v)) for v in values):
            skipped.append(f"r/a = {r} (no finite electron and single main-ion result)")
            continue
        usable.append(record)

    result = {"model": None, "profile_index": None, "written": written, "skipped": skipped}
    if not usable:
        skipped.append("everything (no surface carried a defined classical result)")
        return result

    shas = {formulation_sha256(record.get("model")) for record in usable}
    keys = {next(iter(record["ion_energy_flux_W_m2"])) for record in usable}
    if len(shas) != 1 or len(keys) != 1:
        skipped.append(f"everything (the surfaces disagree on the formulation ({len(shas)}) "
                       f"or the main ion ({sorted(keys)}); one entry holds one of each)")
        return result
    sha, key = shas.pop(), keys.pop()
    z_ion = float(key.split("=", 1)[1])

    usable.sort(key=lambda record: float(record["r_over_a"]))
    radii = np.array([float(record["r_over_a"]) for record in usable])
    if np.any(np.diff(radii) <= 0.0):
        skipped.append("everything (two records share one surface)")
        return result
    grid = _rho_tor_norm(profile, radii)
    if grid is None:
        skipped.append("everything (the profile carries no rmin/rho pair, so r/a cannot be "
                       "expressed as rho_tor_norm)")
        return result

    model = _model_position(ods, sha, efit_lineage)
    prefix = f"core_transport.model.{model}"
    exists = model < path_count(ods, "core_transport.model")
    parsed = classical_parameters(ods.get(f"{prefix}.code.parameters", None)) if exists else None
    slices = list(parsed["slices"]) if parsed else []
    matches = [i for i, (t, _) in enumerate(slices) if abs(t - float(time)) <= _TIME_TOLERANCE_S]
    if matches:
        profile_index = matches[0]
        if slices[profile_index][1] != state_identity:
            skipped.append(f"everything (time {time} s already holds state "
                           f"{slices[profile_index][1]!r}; a second state is not overwritten)")
            return result
    else:
        profile_index = len(slices)
        slices.append((float(time), state_identity))

    _ensure_aos(ods, "core_transport.model", model)
    ods[f"{prefix}.identifier.index"] = CLASSICAL_MODEL_INDEX
    ods[f"{prefix}.identifier.name"] = CLASSICAL_MODEL_NAME
    ods[f"{prefix}.identifier.description"] = CLASSICAL_MODEL_DESCRIPTION
    ods[f"{prefix}.flux_multiplier"] = FLUX_MULTIPLIER
    ods[f"{prefix}.code.name"] = "vaft.process.transport_state.classical_heat_fluxes"
    ods[f"{prefix}.code.repository"] = "https://github.com/VEST-Tokamak/vaft"
    try:
        from vaft import __version__

        ods[f"{prefix}.code.version"] = str(__version__)
    except ImportError:  # pragma: no cover - the package always defines it
        pass
    ods[f"{prefix}.code.parameters"] = _envelope(sha, efit_lineage, usable[0].get("model"), slices)

    base = f"{prefix}.profiles_1d.{profile_index}"
    _ensure_aos(ods, f"{prefix}.profiles_1d", profile_index)
    ods[f"{base}.time"] = float(time)
    ods[f"{base}.grid_flux.rho_tor_norm"] = grid
    ods[f"{base}.grid_d.rho_tor_norm"] = grid
    ods[f"{base}.electrons.energy.flux"] = np.array(
        [float(record["electron_energy_flux_W_m2"]) for record in usable])
    ods[f"{base}.electrons.energy.d"] = np.array([float(record["chi_e_m2_s"]) for record in usable])
    ods[f"{base}.ion.0.energy.flux"] = np.array(
        [float(record["ion_energy_flux_W_m2"][key]) for record in usable])
    ods[f"{base}.ion.0.energy.d"] = np.array([float(record["chi_i_m2_s"]) for record in usable])
    ods[f"{base}.ion.0.z_ion"] = z_ion
    ods[f"{base}.ion.0.element.0.z_n"] = z_ion
    written += ["electrons.energy.flux", "electrons.energy.d", "ion.0.energy.flux", "ion.0.energy.d"]
    skipped += [
        "particles.flux (the model evaluates no particle flux: not evaluated, not zero)",
        "impurity ions (the model covers electrons and the main ion only)",
        "collision times, Coulomb logarithms, gamma_1', |B| (diagnostics with no "
        "core_transport home; they stay in the native record)",
    ]
    # A classical slice index is this entry's own, not a position on a shared
    # core_transport.time, so the IDS is heterogeneous. That stays true for the NEO,
    # TGLF and gyrokinetics entries too: each of their slices also carries its own
    # profiles_1d time, and their mappers leave an explicit 0 in place.
    ods["core_transport.ids_properties.homogeneous_time"] = 0
    result.update(model=model, profile_index=profile_index)
    return result
