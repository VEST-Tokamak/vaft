"""`nbi` IDS mapping for the VEST neutral beam.

Issue #490 section 5. The geometry here is **NUBEAM-derived, not as-built**: it
comes from ``mdescr_VEST_190307.dat``, the machine description the validated
VEST NUBEAM case runs with, and is stored natively in ``vest.yaml`` so the
provenance survives. Reconciling it against the real hardware is issue #265,
and until that happens the diagnostic registry keeps this mapping ``partial``.

Four conversions are needed, and each is a place the obvious reading is wrong:

* **Tangency radius.** NUBEAM signs it (``srtcen = -0.22129``). IMAS documents
  ``tangency_radius`` as a *major radius* -- a magnitude -- and carries the
  injection sense separately in ``direction`` (-1 clockwise seen from above).
  So the magnitude is taken and the direction written explicitly. The direction
  is not derived from the sign by a rule inferred from one case; it is recorded
  in ``vest.yaml`` from three agreeing observations of the run itself.
* **Widths.** ``mdescr`` gives half-widths. IMAS ``width_horizontal`` is the
  full width of "the smallest rectangle that surrounds the outer dimensions of
  the beamlets", so both are doubled.
* **Angles.** ``mdescr`` is in degrees, IMAS in radians. The two divergences
  are one Gaussian component carrying both axes, not one component per axis:
  IMAS components are populations with a particle fraction each, so a split
  would assert two populations of 100% of the beam.
* **Focal length.** NUBEAM spells "unfocused" as 1.2e11 m. Writing that into
  ``focus`` would be a fictitious hundred-million-kilometre focal length, so
  those fields are left absent instead.

Energy and launched power are deliberately *not* part of this mapping. They
enter a NUBEAM case through its ``profiles`` file, which makes them per-case
modelling inputs rather than machine description; :func:`nbi_run_conditions`
writes them from an actual result, so a reader can never mistake a modelled
beam condition for as-built machine metadata.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from omas import ODS

from vaft.ods_access import path_count

from .utils import load_yaml, package_data_path

__all__ = [
    "nbi",
    "nbi_run_conditions",
]

#: Above this, NUBEAM's focal length means "unfocused" rather than a distance.
_UNFOCUSED_M = 1.0e9


def _unit_position(ods: ODS, name: str) -> int:
    """Index of the named unit in ``nbi.unit``, appending if it is new."""
    count = path_count(ods, "nbi.unit")
    for index in range(count):
        if ods.get(f"nbi.unit.{index}.name", None) == name:
            return index
    return count


def nbi(ods: ODS, shot: int = 0, options: Optional[dict] = None) -> dict[str, Any]:
    """Populate the static ``nbi`` IDS for VEST.

    Returns a report naming the unit written and the fields deliberately left
    absent, so a caller can see the shape of what is *not* known rather than
    having to infer it from missing paths.
    """
    document = load_yaml(package_data_path("vest.yaml"))
    # The beam has no shot-resolved revisions yet: one machine description, from
    # one NUBEAM case. When as-built data arrive (#265) this is where a
    # resolve_shot_revisions call belongs.
    config = (document.get(0) or {}).get("nbi")
    if not config:
        raise ValueError(
            f"no static NBI configuration for shot {shot} in vest.yaml; the "
            "VEST beam is described under the shot-0 defaults"
        )

    absent: list[str] = []
    written: list[str] = []

    for entry in config.get("unit", []):
        index = _unit_position(ods, entry["name"])
        base = f"nbi.unit.{index}"
        ods[f"{base}.name"] = entry["name"]
        ods[f"{base}.identifier"] = entry["name"]

        species = entry.get("species") or {}
        if species:
            ods[f"{base}.species.label"] = str(species.get("label", ""))
            if "a" in species:
                ods[f"{base}.species.a"] = float(species["a"])
            if "z_n" in species:
                ods[f"{base}.species.z_n"] = float(species["z_n"])
            written.append("species")

        source = entry.get("source") or {}
        grid = entry.get("grid") or {}
        group = f"{base}.beamlets_group.0"

        signed = source.get("tangency_radius_signed")
        if signed is not None:
            # A magnitude, per the schema; the sense goes in `direction`.
            ods[f"{group}.tangency_radius"] = abs(float(signed))
            written.append("tangency_radius")
        if source.get("direction") is not None:
            ods[f"{group}.direction"] = int(source["direction"])
            written.append("direction")
        if source.get("elevation") is not None:
            ods[f"{group}.position.z"] = float(source["elevation"])
            written.append("position.z")
        if source.get("toroidal_angle") is not None:
            ods[f"{group}.position.phi"] = math.radians(
                float(source["toroidal_angle"])
            )
            written.append("position.phi")

        if grid.get("half_width") is not None:
            # Full width of the enclosing rectangle, not the half-width stored.
            ods[f"{group}.width_horizontal"] = 2.0 * float(grid["half_width"])
            written.append("width_horizontal")
        if grid.get("half_height") is not None:
            ods[f"{group}.width_vertical"] = 2.0 * float(grid["half_height"])
            written.append("width_vertical")

        # One Gaussian component carrying both axes, not one component per
        # axis. IMAS describes divergence as a superposition of Gaussians, each
        # with its own horizontal *and* vertical divergence and a share of the
        # particles -- and notes that a positive-ion NBI is well described by a
        # single Gaussian. Splitting the axes across two components would claim
        # two populations each holding all the particles.
        horizontal = grid.get("horizontal_divergence")
        vertical = grid.get("vertical_divergence")
        if horizontal is not None or vertical is not None:
            if horizontal is not None:
                ods[f"{group}.divergence_component.0.horizontal"] = math.radians(
                    float(horizontal)
                )
            if vertical is not None:
                ods[f"{group}.divergence_component.0.vertical"] = math.radians(
                    float(vertical)
                )
            ods[f"{group}.divergence_component.0.particles_fraction"] = 1.0
            written.append("divergence_component")

        for key in ("horizontal_focal_length", "vertical_focal_length"):
            value = grid.get(key)
            if value is not None and float(value) >= _UNFOCUSED_M:
                absent.append(f"focus.{key} (NUBEAM says unfocused: {value:g} m)")

        aperture = entry.get("aperture") or {}
        if aperture.get("half_width") is not None:
            ods[f"{base}.aperture.0.geometry_type"] = 3  # rectangle
            ods[f"{base}.aperture.0.x1_width"] = 2.0 * float(aperture["half_width"])
            ods[f"{base}.aperture.0.x2_width"] = 2.0 * float(aperture["half_height"])
            written.append("aperture")

    _write_provenance(ods, config)
    absent.append("energy and power_launched (per-case modelling inputs)")
    return {"written": written, "absent": absent}


def nbi_run_conditions(ods: ODS, result: Any, *, unit: int = 0) -> dict[str, Any]:
    """Write the beam conditions a NUBEAM case actually ran with.

    Separate from :func:`nbi` on purpose. These are modelling inputs chosen for
    one case, not machine description, and #490 section 5 is explicit that a
    modelled NUBEAM input is not equivalent to a measured machine parameter.
    """
    native = getattr(result, "outputs_native", None) or result
    conditions = getattr(native, "beam_conditions", None) or {}
    if not conditions:
        raise ValueError(
            "this NUBEAM result carries no beam conditions; they are read from "
            "the Plasma State, which the run directory no longer holds"
        )

    base = f"nbi.unit.{unit}"
    written = []
    if "energy_keV" in conditions:
        # IMAS wants eV; NUBEAM's Plasma State states keV.
        ods[f"{base}.energy.data"] = [float(conditions["energy_keV"]) * 1.0e3]
        ods[f"{base}.energy.time"] = [float(conditions.get("time", 0.0))]
        written.append("energy")
    if "power_W" in conditions:
        ods[f"{base}.power_launched.data"] = [float(conditions["power_W"])]
        ods[f"{base}.power_launched.time"] = [float(conditions.get("time", 0.0))]
        written.append("power_launched")
    fractions = conditions.get("power_fractions")
    if fractions:
        ods[f"{base}.beam_power_fraction.data"] = [[float(f)] for f in fractions]
        ods[f"{base}.beam_power_fraction.time"] = [float(conditions.get("time", 0.0))]
        written.append("beam_power_fraction")
    return {"written": written}


def _write_provenance(ods: ODS, config: dict) -> None:
    ods["nbi.ids_properties.homogeneous_time"] = 1
    ods["nbi.ids_properties.comment"] = (
        "VEST NBI geometry derived from the NUBEAM machine description "
        f"{config.get('provenance', 'mdescr')}. Not as-built; pending "
        "reconciliation against the NBI hardware package (issue #265)."
    )
