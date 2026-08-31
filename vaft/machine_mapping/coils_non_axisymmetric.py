"""VEST non-axisymmetric 3D coil mapping into ``coils_non_axisymmetric``.

Static geometry and run-specific excitation are separate layers:

- :func:`coils_non_axisymmetric` writes the canonical coil geometry (one IMAS
  coil per toroidal sector filament) and nothing time-dependent, so the IDS is
  static (``homogeneous_time = 2``).
- :func:`apply_coil_excitation` overlays per-sector currents for one run,
  addressed by stable coil identifiers, and switches the IDS to
  ``homogeneous_time = 1``.

The data-dictionary ``coils_non_axisymmetric`` IDS has no coil-group
structure, so set membership is carried by the identifier prefix
(``VEST_3D_<SET>_<sector>``) and mirrored, with the geometry provenance, in a
``code.parameters`` XML fragment per set.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence
from xml.sax.saxutils import escape, quoteattr

import numpy as np
from omas import ODS

from .coil_geometry_3d import (
    CoilExcitation,
    CoilSet3D,
    Vest3DCoilConfig,
    load_vest_3d_coil_config,
)
from .utils import VestConfigurationError, set_path

__all__ = ["coils_non_axisymmetric", "apply_coil_excitation", "sector_identifier"]

_IDS = "coils_non_axisymmetric"


def sector_identifier(coil_set: CoilSet3D, sector: int) -> str:
    """The stable identifier of one sector coil, e.g. ``VEST_3D_MID_01``."""
    return f"{coil_set.identifier}_{sector + 1:02d}"


def _filament_rphiz(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = np.hypot(points_xyz[:, 0], points_xyz[:, 1])
    phi = np.unwrap(np.arctan2(points_xyz[:, 1], points_xyz[:, 0]))
    return radius, phi, points_xyz[:, 2]


def _write_sector_coil(ods: ODS, index: int, coil_set: CoilSet3D, sector: int) -> None:
    filament = coil_set.filaments[sector]
    base = f"{_IDS}.coil.{index}"
    set_path(ods, f"{base}.name", f"{coil_set.name} sector {sector + 1}")
    set_path(ods, f"{base}.identifier", sector_identifier(coil_set, sector))
    set_path(ods, f"{base}.turns", float(coil_set.turns))

    radius, phi, height = _filament_rphiz(filament.points_xyz)
    elements = f"{base}.conductor.0.elements"
    # Consecutive filament points become line elements (type 1); a closed
    # loop of N points yields N-1 segments ending where it started.
    set_path(ods, f"{elements}.types", np.ones(radius.size - 1, dtype=int))
    for node, values in (("start_points", slice(None, -1)), ("end_points", slice(1, None))):
        set_path(ods, f"{elements}.{node}.r", radius[values])
        set_path(ods, f"{elements}.{node}.phi", phi[values])
        set_path(ods, f"{elements}.{node}.z", height[values])


def _set_provenance(ods: ODS, config: Vest3DCoilConfig) -> None:
    fragments = "".join(
        f"<coil_set name={quoteattr(coil_set.name)} "
        f"identifier={quoteattr(coil_set.identifier)} "
        f"turns=\"{coil_set.turns:g}\" "
        f"sectors=\"{len(coil_set.filaments)}\" "
        f"dat_file={quoteattr(coil_set.dat_path.name)}>"
        f"{escape(coil_set.provenance)}</coil_set>"
        for coil_set in config.coil_sets.values()
    )
    set_path(ods, f"{_IDS}.code.name", "vaft")
    set_path(
        ods,
        f"{_IDS}.code.repository",
        "https://github.com/VEST-Tokamak/vaft",
    )
    set_path(ods, f"{_IDS}.code.parameters", f"<parameters>{fragments}</parameters>")
    set_path(
        ods,
        f"{_IDS}.ids_properties.comment",
        "VEST non-axisymmetric 3D coil geometry from the canonical packaged "
        "GPEC-format coil data (vaft.machine_mapping.coil_geometry_3d).",
    )


def coils_non_axisymmetric(
    ods: ODS,
    source: str | None = None,
    options: Optional[Mapping[str, Any]] = None,
) -> None:
    """Populate static VEST 3D coil geometry.

    ``source`` optionally overrides the packaged ``vaft/data`` root directory;
    ``options['coil_sets']`` selects a subset of canonical set names.
    """
    options = dict(options or {})
    config = load_vest_3d_coil_config(source, coil_sets=options.get("coil_sets"))

    index = 0
    for coil_set in config.coil_sets.values():
        for sector in range(len(coil_set.filaments)):
            _write_sector_coil(ods, index, coil_set, sector)
            index += 1
    _set_provenance(ods, config)
    # Only constant/static nodes are filled here, so per the DD rule the IDS
    # is time-independent until an excitation overlay adds `current.time`.
    set_path(ods, f"{_IDS}.ids_properties.homogeneous_time", 2)


def _coil_indices_by_identifier(ods: ODS) -> dict[str, int]:
    if _IDS not in ods or "coil" not in ods[_IDS]:
        raise VestConfigurationError(
            "apply_coil_excitation requires coils_non_axisymmetric geometry; "
            "run vaft.machine_mapping.coils_non_axisymmetric first"
        )
    coils = ods[_IDS]["coil"]
    return {coils[index]["identifier"]: index for index in range(len(coils))}


def apply_coil_excitation(
    ods: ODS,
    excitations: Sequence[CoilExcitation],
    time_s: Sequence[float] | float = 0.0,
) -> None:
    """Overlay run-specific per-sector currents onto the mapped geometry.

    Currents are amperes per turn, matched to coils by stable identifier.
    """
    time_array = np.atleast_1d(np.asarray(time_s, dtype=float))
    by_identifier = _coil_indices_by_identifier(ods)
    prefix_by_set = {
        identifier.rsplit("_", 1)[0] for identifier in by_identifier
    }

    for excitation in excitations:
        config = load_vest_3d_coil_config(coil_sets=[excitation.coil_set])
        coil_set = config[excitation.coil_set]
        if coil_set.identifier not in prefix_by_set:
            raise VestConfigurationError(
                f"Coil set {excitation.coil_set!r} ({coil_set.identifier}) is "
                "not present in the mapped coils_non_axisymmetric geometry"
            )
        if len(excitation.currents_a) != len(coil_set.filaments):
            raise VestConfigurationError(
                f"Coil set {excitation.coil_set!r} has "
                f"{len(coil_set.filaments)} sectors but the excitation "
                f"carries {len(excitation.currents_a)} currents"
            )
        for sector, current in enumerate(excitation.currents_a):
            index = by_identifier[sector_identifier(coil_set, sector)]
            base = f"{_IDS}.coil.{index}.current"
            set_path(ods, f"{base}.time", time_array)
            set_path(
                ods,
                f"{base}.data",
                np.full(time_array.size, float(current)),
            )
    set_path(ods, f"{_IDS}.time", time_array)
    set_path(ods, f"{_IDS}.ids_properties.homogeneous_time", 1)
