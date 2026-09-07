"""Project canonical VEST static geometry onto EFUND's input description.

EFUND (the Green-function table generator of the EFIT toolchain) describes a
machine as rectangles: F-coil elements grouped into current groups, vessel
segments grouped into passive current groups, point flux loops and oriented
B-probes.  VAFT's canonical description of the same machine is the static ODS
(:func:`vaft.omas.vest_upstream.build_static_ods`): ``pf_active`` elements,
``pf_passive`` loop outlines, ``magnetics`` positions.  This module is the one
projection between the two, and its invariant is

    canonical static geometry  ->  EFUND input, in the em_coupling order.

Nothing here decides which machine era a shot belongs to.  The caller obtains
the era through :func:`vaft.omas.vest_upstream.machine_era_for_shot` and builds
the static ODS for it; this module only projects what it is handed (issue #191
forbids duplicating shot-boundary logic in EFUND code).

The F-coil grouping is the existing EFIT convention, not a new one: the k-file
writer selects sixteen of the twenty-six legacy coil channels
(:func:`vaft.code.efit.kfile.efit16_group_indices`) -- PF1 as eight axial
segments plus the upper/lower halves of PF5, PF6, PF9 and PF10 -- and the
routine constraint matrix ties every PF1 segment to the same current.  PF2,
PF3, PF4, PF7 and PF8 are not in that set; they are absent from the table
exactly as they are absent from the k-file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "EFIT16_GROUP_NAMES",
    "PF1_SEGMENT_EDGES",
    "EFUNDGeometry",
    "efund_geometry_from_static",
    "efund_probe_angle_deg",
    "equilibrium_probe_count",
    "rectangle_from_outline",
    "vest_acceptance_envelope",
]

#: The sixteen EFIT current groups, in k-file order.  Same names and order as
#: ``vaft.code.efit.legacy.vfit_pf_active_efit26`` restricted by
#: ``vaft.code.efit.kfile.efit16_group_indices``.
EFIT16_GROUP_NAMES: tuple[str, ...] = (
    "PF1-1",
    "PF1-2",
    "PF1-3",
    "PF1-4",
    "PF1-5",
    "PF1-6",
    "PF1-7",
    "PF1-8",
    "PF5U",
    "PF5L",
    "PF6U",
    "PF6L",
    "PF9U",
    "PF9L",
    "PF10U",
    "PF10L",
)

#: |z| boundaries (m) of the eight legacy PF1 segments: PF1-1..PF1-4 cover
#: z in [0, 0.3), [0.3, 0.6), [0.6, 0.9), [0.9, 1.2] and PF1-5..PF1-8 mirror
#: them below the midplane.  Every segment carries the same current in the
#: routine k-file, so this split changes the bookkeeping, not the field.
PF1_SEGMENT_EDGES: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9, 1.2)

_PF1_SEGMENT_TOLERANCE = 1.0e-9


def rectangle_from_outline(
    r: Sequence[float], z: Sequence[float]
) -> tuple[float, float, float, float]:
    """``(centre_r, centre_z, width, height)`` of an axis-aligned 4-point outline.

    EFUND takes rectangles, not polygons, so an outline that is not an
    axis-aligned rectangle cannot be projected and is refused rather than
    approximated.
    """
    r_values = np.asarray(r, dtype=float).reshape(-1)
    z_values = np.asarray(z, dtype=float).reshape(-1)
    if r_values.size != 4 or z_values.size != 4:
        raise ValueError(
            f"EFUND needs a 4-point rectangular outline, got {r_values.size} points"
        )
    r_unique = np.unique(r_values)
    z_unique = np.unique(z_values)
    if r_unique.size != 2 or z_unique.size != 2:
        raise ValueError("outline is not an axis-aligned rectangle")
    corners = {(float(a), float(b)) for a, b in zip(r_values, z_values)}
    expected = {(float(a), float(b)) for a in r_unique for b in z_unique}
    if corners != expected:
        raise ValueError("outline is not an axis-aligned rectangle")
    width = float(r_unique[1] - r_unique[0])
    height = float(z_unique[1] - z_unique[0])
    return (
        float(r_unique.mean()),
        float(z_unique.mean()),
        width,
        height,
    )


def efund_probe_angle_deg(poloidal_angle_rad: float) -> float:
    """EFUND ``amp2`` (degrees) for a DD ``poloidal_angle`` (radians).

    EFUND orients a probe along ``(cos amp2, sin amp2)`` in ``(R, Z)`` and
    integrates ``B_R cos amp2 + B_Z sin amp2`` along it.  The DD defines
    ``poloidal_angle`` clockwise from +R, so the measured axis is
    ``(cos theta, -sin theta)``; the two agree for ``amp2 = -theta``.  The
    result is reduced to ``[0, 360)`` so the vertical VEST probe
    (``theta = 3 pi / 2``) is written as ``90.0``; the bundled legacy table
    spells the same direction ``-270.0``.
    """
    degrees = -math.degrees(float(poloidal_angle_rad))
    reduced = math.fmod(degrees, 360.0)
    if reduced < 0.0:
        reduced += 360.0
    if math.isclose(reduced, 360.0, abs_tol=1.0e-9):
        reduced = 0.0
    return float(reduced)


def equilibrium_probe_count(ods: Any) -> int:
    """The B-pol probes EFIT's geometry represents.

    The magnetics IDS also carries trailing toroidal-Mirnov phase-reference
    channels which are not equilibrium probes.  Same rule as the k-file writer
    (``kfile._efit_bpol_probe_count``) and as
    ``vaft.validation.efit_channels.efit_probe_count`` where that module exists:
    ``min(present, defined)`` against the canonical channel definition.
    """
    from vaft.machine_mapping.magnetics import (
        vest_equilibrium_magnetics_channel_definitions,
    )

    defined = sum(
        1
        for entry in vest_equilibrium_magnetics_channel_definitions()
        if entry.get("kind") == "b_field_pol_probe"
    )
    present = len(ods["magnetics.b_field_pol_probe"])
    return min(present, defined) if defined else present


def _pf1_segment(z: float) -> int:
    """0-based PF1 segment index (PF1-1..PF1-8) for an element centre."""
    magnitude = abs(float(z))
    edges = PF1_SEGMENT_EDGES
    if magnitude > edges[-1] + _PF1_SEGMENT_TOLERANCE:
        raise ValueError(f"PF1 element at |z|={magnitude} lies outside the segment span")
    for index in range(len(edges) - 1):
        if magnitude < edges[index + 1] or index == len(edges) - 2:
            return index if float(z) >= 0.0 else index + 4
    raise AssertionError("unreachable")


def _group_of(coil_name: str, z: float) -> str | None:
    """EFIT group for a canonical coil element, or ``None`` outside the set."""
    if coil_name == "PF1":
        return EFIT16_GROUP_NAMES[_pf1_segment(z)]
    if coil_name in ("PF5", "PF6", "PF9", "PF10"):
        if float(z) == 0.0:
            raise ValueError(f"{coil_name} element on the midplane cannot be assigned U or L")
        return f"{coil_name}{'U' if float(z) > 0.0 else 'L'}"
    return None


@dataclass(frozen=True)
class EFUNDGeometry:
    """EFUND's rectangle description of one machine era.

    Arrays are in EFUND's own order and units (m, degrees, ohm).  ``fcoil_group``
    and ``vessel_group`` are the 1-based ``fcid`` / ``vsid`` values.
    """

    fcoil_r: np.ndarray
    fcoil_z: np.ndarray
    fcoil_w: np.ndarray
    fcoil_h: np.ndarray
    fcoil_a: np.ndarray
    fcoil_a2: np.ndarray
    fcoil_group: np.ndarray
    fcoil_turns: np.ndarray
    group_names: tuple[str, ...]
    group_turns: np.ndarray
    vessel_r: np.ndarray
    vessel_z: np.ndarray
    vessel_w: np.ndarray
    vessel_h: np.ndarray
    vessel_a: np.ndarray
    vessel_a2: np.ndarray
    vessel_group: np.ndarray
    vessel_resistance: np.ndarray
    vessel_names: tuple[str, ...]
    loop_r: np.ndarray
    loop_z: np.ndarray
    loop_names: tuple[str, ...]
    probe_r: np.ndarray
    probe_z: np.ndarray
    probe_angle_deg: np.ndarray
    probe_length: np.ndarray
    probe_names: tuple[str, ...]
    machine: dict[str, Any] = field(default_factory=dict)

    @property
    def nfcoil(self) -> int:
        return int(self.fcoil_r.size)

    @property
    def nfsum(self) -> int:
        return len(self.group_names)

    @property
    def nvesel(self) -> int:
        return int(self.vessel_r.size)

    @property
    def nvsum(self) -> int:
        return int(np.unique(self.vessel_group).size) if self.nvesel else 0

    @property
    def nsilop(self) -> int:
        return int(self.loop_r.size)

    @property
    def magpri(self) -> int:
        return int(self.probe_r.size)

    def counts(self) -> dict[str, int]:
        """The ``&machinein`` counts this geometry implies."""
        return {
            "nfcoil": self.nfcoil,
            "nfsum": self.nfsum,
            "nsilop": self.nsilop,
            "magpri": self.magpri,
            "necoil": 0,
            "nesum": 0,
            "nvesel": self.nvesel,
            "nvsum": self.nvsum,
            "nacoil": 0,
        }

    def group_summary(self) -> list[dict[str, Any]]:
        """Per F-coil group: element count, turns and the turn-weighted centroid."""
        rows = []
        for index, name in enumerate(self.group_names, start=1):
            mask = self.fcoil_group == index
            turns = self.fcoil_turns[mask]
            weight = turns / turns.sum() if turns.sum() else turns
            rows.append(
                {
                    "name": name,
                    "elements": int(mask.sum()),
                    "turns": float(turns.sum()),
                    "r": float(np.dot(weight, self.fcoil_r[mask])),
                    "z": float(np.dot(weight, self.fcoil_z[mask])),
                    "z_min": float(self.fcoil_z[mask].min()),
                    "z_max": float(self.fcoil_z[mask].max()),
                }
            )
        return rows


def _fcoils(ods: Any) -> dict[str, Any]:
    by_group: dict[str, list[tuple[float, float, float, float, float]]] = {
        name: [] for name in EFIT16_GROUP_NAMES
    }
    for coil_index in range(len(ods["pf_active.coil"])):
        coil = ods[f"pf_active.coil.{coil_index}"]
        coil_name = str(coil["name"])
        for element_index in range(len(coil["element"])):
            element = coil[f"element.{element_index}"]
            geometry = element["geometry"]
            if int(geometry["geometry_type"]) != 2:
                raise ValueError(
                    f"{coil_name} element {element_index} is not a rectangle "
                    f"(geometry_type {int(geometry['geometry_type'])})"
                )
            z = float(geometry["rectangle.z"])
            group = _group_of(coil_name, z)
            if group is None:
                continue
            by_group[group].append(
                (
                    float(geometry["rectangle.r"]),
                    z,
                    float(geometry["rectangle.width"]),
                    float(geometry["rectangle.height"]),
                    abs(float(element["turns_with_sign"])),
                )
            )
    empty = [name for name, rows in by_group.items() if not rows]
    if empty:
        raise ValueError(f"no canonical element maps onto EFIT group(s) {', '.join(empty)}")
    rows = [
        (*values, group_index)
        for group_index, name in enumerate(EFIT16_GROUP_NAMES, start=1)
        for values in by_group[name]
    ]
    table = np.asarray(rows, dtype=float)
    return {
        "fcoil_r": table[:, 0],
        "fcoil_z": table[:, 1],
        "fcoil_w": table[:, 2],
        "fcoil_h": table[:, 3],
        "fcoil_a": np.zeros(table.shape[0]),
        "fcoil_a2": np.zeros(table.shape[0]),
        "fcoil_turns": table[:, 4],
        "fcoil_group": table[:, 5].astype(int),
        "group_names": EFIT16_GROUP_NAMES,
        "group_turns": np.ones(len(EFIT16_GROUP_NAMES)),
    }


def _vessel(ods: Any) -> dict[str, Any]:
    count = len(ods["pf_passive.loop"])
    if "em_coupling.passive_loops" in ods:
        uris = [str(item) for item in ods["em_coupling.passive_loops"]]
        expected = [f"#pf_passive/loop({index + 1})" for index in range(count)]
        if uris != expected:
            raise ValueError(
                "em_coupling.passive_loops is not pf_passive.loop in order; "
                "the EFUND vessel order would not match the coupling asset"
            )
    rows = []
    names = []
    for loop_index in range(count):
        loop = ods[f"pf_passive.loop.{loop_index}"]
        if len(loop["element"]) != 1:
            raise ValueError(
                f"pf_passive.loop.{loop_index} has {len(loop['element'])} elements; "
                "EFUND takes one rectangle per vessel segment"
            )
        outline = loop["element.0.geometry.outline"]
        rc, zc, width, height = rectangle_from_outline(outline["r"], outline["z"])
        rows.append((rc, zc, width, height, float(loop["resistance"])))
        names.append(str(loop["name"]) if "name" in loop else f"loop{loop_index + 1}")
    table = np.asarray(rows, dtype=float).reshape(count, 5)
    return {
        "vessel_r": table[:, 0],
        "vessel_z": table[:, 1],
        "vessel_w": table[:, 2],
        "vessel_h": table[:, 3],
        "vessel_a": np.zeros(count),
        "vessel_a2": np.zeros(count),
        "vessel_group": np.arange(1, count + 1, dtype=int),
        "vessel_resistance": table[:, 4],
        "vessel_names": tuple(names),
    }


def _flux_loops(ods: Any) -> dict[str, Any]:
    count = len(ods["magnetics.flux_loop"])
    r_values = np.empty(count)
    z_values = np.empty(count)
    names = []
    for index in range(count):
        loop = ods[f"magnetics.flux_loop.{index}"]
        if len(loop["position"]) != 1:
            raise ValueError(
                f"magnetics.flux_loop.{index} has {len(loop['position'])} positions; "
                "EFUND point loops take one"
            )
        r_values[index] = float(loop["position.0.r"])
        z_values[index] = float(loop["position.0.z"])
        names.append(str(loop["name"]) if "name" in loop else f"FL{index + 1}")
    return {"loop_r": r_values, "loop_z": z_values, "loop_names": tuple(names)}


def _probes(ods: Any) -> dict[str, Any]:
    count = equilibrium_probe_count(ods)
    r_values = np.empty(count)
    z_values = np.empty(count)
    angles = np.empty(count)
    lengths = np.empty(count)
    names = []
    for index in range(count):
        probe = ods[f"magnetics.b_field_pol_probe.{index}"]
        r_values[index] = float(probe["position.r"])
        z_values[index] = float(probe["position.z"])
        angles[index] = efund_probe_angle_deg(float(probe["poloidal_angle"]))
        lengths[index] = float(probe["length"])
        names.append(str(probe["name"]) if "name" in probe else f"MP{index + 1}")
    return {
        "probe_r": r_values,
        "probe_z": z_values,
        "probe_angle_deg": angles,
        "probe_length": lengths,
        "probe_names": tuple(names),
    }


def _machine_block(ods: Any, manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    block: dict[str, Any] = {}
    if manifest:
        era = manifest.get("machine_era") or {}
        block["era"] = era.get("name")
        block["pf_geometry"] = era.get("pf_geometry")
        block["reference_shot"] = era.get("reference_shot")
        block["static_inputs"] = {
            key: dict(value) for key, value in (manifest.get("input") or {}).items()
        }
    for ids_name in ("em_coupling", "pf_active", "wall"):
        path = f"{ids_name}.ids_properties.comment"
        if path in ods:
            block[f"{ids_name}_comment"] = str(ods[path])
    return block


def vest_acceptance_envelope(
    ods: Any,
    *,
    nw: int = 129,
    rleft: float = 0.05,
    rright: float = 1.2,
    resolved_cells: int = 5,
    base: Any | None = None,
    virial_checks: bool = False,
) -> Any:
    """Derive EFIT's ``&incheck`` geometric bounds from the machine itself.

    EFIT's built-in bounds are DIII-D's and the packaged VEST ``mhdin.dat``
    softens them only part-way, so a VEST equilibrium is rejected for being
    small: ``aminor_min`` is 25 cm where VEST's minor radius runs about 21 cm.
    The fix is not to loosen the bounds until a shot passes -- that would test
    nothing -- but to derive the geometric ones from the limiter, which says
    what the machine can physically contain.

    * ``aminor_max`` is half the limiter's radial extent: a plasma cannot be
      wider than the vessel that holds it.
    * ``aminor_min`` is ``resolved_cells`` grid cells in R.  Below that the
      boundary is not resolved, so accepting it would accept a number rather
      than a plasma.  It is a numerical floor and is stated as one.
    * the centre and centroid bounds keep the plasma inside the limiter by at
      least ``aminor_min``.

    ``li``, ``betap``, ``qstar`` and ``elong`` are left at ``base``'s values,
    because those are physics arguments and not geometry.  Lengths returned
    are centimetres, EFIT's convention in this namelist.

    The virial consistency checks are a separate matter and are **disabled**
    unless ``virial_checks`` says otherwise.  At VEST's aspect ratio of about
    1.45 they are ill-conditioned rather than merely strict (issue #649):
    ``sbpp`` is a difference of two terms near 0.8 leaving 0.01 to 0.2 and
    negative on some slices, a median cancellation of twelvefold, and
    ``sbli`` divides by ``alpha - 1``, which VEST measures at 0.22 to 0.51 and
    falling.  A gate that cannot distinguish a good reconstruction from a bad
    one should not decide acceptance.  Every quantity is still computed and
    written to the a-file; only the rejection stops.
    """
    from dataclasses import replace

    from vaft.code.efit.config import IGNORE_CRITERION, EFITAcceptanceEnvelope

    outline = ods["wall.description_2d.0.limiter.unit.0.outline"]
    r = np.asarray(outline["r"], dtype=float).reshape(-1)
    z = np.asarray(outline["z"], dtype=float).reshape(-1)
    if r.size < 3:
        raise ValueError("the limiter outline is too short to bound anything")

    centimetre = 100.0
    r_min, r_max = float(r.min()) * centimetre, float(r.max()) * centimetre
    z_min, z_max = float(z.min()) * centimetre, float(z.max()) * centimetre
    cell = (float(rright) - float(rleft)) / (int(nw) - 1) * centimetre
    aminor_min = float(resolved_cells) * cell
    aminor_max = 0.5 * (r_max - r_min)
    if aminor_min >= aminor_max:
        raise ValueError(
            f"the resolved floor {aminor_min:.1f} cm is not below the limiter's "
            f"half-width {aminor_max:.1f} cm; check the grid or the outline"
        )
    return replace(
        base or EFITAcceptanceEnvelope(),
        aminor_min=aminor_min,
        aminor_max=aminor_max,
        rcntr_min=r_min + aminor_min,
        rcntr_max=r_max - aminor_min,
        zcntr_min=z_min + aminor_min,
        zcntr_max=z_max - aminor_min,
        rcurrt_min=r_min + aminor_min,
        rcurrt_max=r_max - aminor_min,
        zcurrt_min=z_min + aminor_min,
        zcurrt_max=z_max - aminor_min,
        **(
            {}
            if virial_checks
            else {"delbp_diff": IGNORE_CRITERION, "dbpli_diff": IGNORE_CRITERION}
        ),
    )


def efund_geometry_from_static(
    ods: Any, *, manifest: Mapping[str, Any] | None = None
) -> EFUNDGeometry:
    """Project a static VEST ODS onto EFUND's description.

    ``manifest`` is the record :func:`vaft.omas.vest_upstream.build_static_ods`
    returns beside the ODS; when given, the era and input-asset hashes it
    carries are copied into :attr:`EFUNDGeometry.machine` so a table can say
    what it was generated from.
    """
    parts: dict[str, Any] = {}
    parts.update(_fcoils(ods))
    parts.update(_vessel(ods))
    parts.update(_flux_loops(ods))
    parts.update(_probes(ods))
    parts["machine"] = _machine_block(ods, manifest)
    return EFUNDGeometry(**parts)
