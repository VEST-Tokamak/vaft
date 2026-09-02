"""Build a TokaMaker machine-geometry description from an ODS.

The geometry dict follows the convention of the TokaMaker example
``*_geom.json`` files (ITER/CUTE/HBT ...):

    {"limiter": [[R, Z], ...],
     "coils": {"PF1_U": {"rc": ..., "zc": ..., "w": ..., "h": ...,
                          "nturns": ..., "coil_set": "PF1"}, ...}}

Coils are read directly from ``pf_active`` rectangle elements — deliberately
*not* the legacy 36-sub-coil re-discretisation used by the TES input writer.
Each VEST coil is wound as an up/down-mirrored pair of element stacks, so a
coil becomes two mesh rectangles (``<name>_U``/``<name>_L``) sharing one
TokaMaker *coil set* named after the coil; the set is the unit of current
control, and the per-half ``nturns`` is the signed sum of ``turns_with_sign``
(preserving anti-series windings).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .config import TokaMakerConfig


def _limiter_from_ods(ods: Any, config: TokaMakerConfig) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the limiter polygon from config or the ODS wall IDS."""
    if config.limiter is not None:
        r, z = config.limiter
        return np.asarray(r, dtype=float), np.asarray(z, dtype=float)
    try:
        r = np.asarray(ods["wall.description_2d.0.limiter.unit.0.outline.r"], dtype=float)
        z = np.asarray(ods["wall.description_2d.0.limiter.unit.0.outline.z"], dtype=float)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            "No limiter found: ODS lacks wall.description_2d.0.limiter.unit.0.outline "
            "and TokaMakerConfig.limiter was not supplied."
        ) from exc
    if len(r) < 3:
        raise ValueError("Limiter outline has fewer than 3 points")
    if np.any(r <= 0.0):
        raise ValueError("Limiter outline contains R <= 0 points; TokaMaker requires R > 0")
    return r, z


def split_coil_names(config: TokaMakerConfig) -> set[str]:
    """Upper-cased names of coils whose U/L halves are independent coil sets."""
    names = {str(name).upper() for name in config.split_coils}
    if config.vsc_coil is not None:
        names.add(str(config.vsc_coil).upper())
    return names


def _coil_name(ods: Any, index: int) -> str:
    """Coil (set) name: pf_active name when present, else PF<index+1>.

    Upper-cased because gs_Domain upper-cases region names internally and the
    coil-current mapping must use identical keys.
    """
    try:
        name = str(ods[f"pf_active.coil.{index}.name"]).strip()
    except Exception:
        name = ""
    return (name or f"PF{index + 1}").upper()


def _element_arrays(ods: Any, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coil = ods[f"pf_active.coil.{index}"]
    nelem = len(coil["element"])
    r = np.array([coil[f"element.{j}.geometry.rectangle.r"] for j in range(nelem)], dtype=float)
    z = np.array([coil[f"element.{j}.geometry.rectangle.z"] for j in range(nelem)], dtype=float)
    w = np.array([coil[f"element.{j}.geometry.rectangle.width"] for j in range(nelem)], dtype=float)
    h = np.array([coil[f"element.{j}.geometry.rectangle.height"] for j in range(nelem)], dtype=float)
    turns = np.array([coil[f"element.{j}.turns_with_sign"] for j in range(nelem)], dtype=float)
    return r, z, w, h, turns


def _bounding_rectangle(
    r: np.ndarray, z: np.ndarray, w: np.ndarray, h: np.ndarray, turns: np.ndarray
) -> dict[str, float]:
    """Bounding box of a stack of rectangle elements, with signed total turns."""
    r_lo = float(np.min(r - w / 2.0))
    r_hi = float(np.max(r + w / 2.0))
    z_lo = float(np.min(z - h / 2.0))
    z_hi = float(np.max(z + h / 2.0))
    return {
        "rc": 0.5 * (r_lo + r_hi),
        "zc": 0.5 * (z_lo + z_hi),
        "w": r_hi - r_lo,
        "h": z_hi - z_lo,
        "nturns": float(np.sum(turns)),
    }


def _coil_rectangles_from_ods(ods: Any, config: TokaMakerConfig) -> dict[str, dict]:
    """Read pf_active coils into mesh rectangles grouped by coil set.

    Elements are split into upper (z >= 0) and lower (z < 0) halves; a coil
    whose elements all sit on one side becomes a single rectangle named after
    the coil itself.
    """
    excluded = {name.upper() for name in config.exclude_coils}
    split_names = split_coil_names(config)
    coils: dict[str, dict] = {}
    ncoil = len(ods["pf_active.coil"])
    for i in range(ncoil):
        set_name = _coil_name(ods, i)
        if set_name in excluded:
            continue
        r, z, w, h, turns = _element_arrays(ods, i)
        halves = {"U": z >= 0.0, "L": z < 0.0}
        present = {suffix: mask for suffix, mask in halves.items() if np.any(mask)}
        split = set_name in split_names
        if split and set(present) != {"U", "L"}:
            raise ValueError(
                f"Coil {set_name} (split_coils/vsc_coil) must be an up/down-"
                f"mirrored coil (found halves: {sorted(present)}). Pick a coil "
                "with elements on both sides of Z = 0."
            )
        for suffix, mask in present.items():
            name = set_name if len(present) == 1 else f"{set_name}_{suffix}"
            rect = _bounding_rectangle(r[mask], z[mask], w[mask], h[mask], turns[mask])
            if rect["nturns"] == 0.0:
                raise ValueError(
                    f"Coil {name} has zero net turns_with_sign; cannot build a "
                    "TokaMaker coil region (check pf_active or exclude the coil)."
                )
            # A split coil's halves get their OWN coil sets so their currents
            # can be set independently (VSC pairs, up/down-asymmetric scans).
            rect["coil_set"] = name if split else set_name
            coils[name] = rect
    if not coils:
        raise ValueError("No usable pf_active coils found for the TokaMaker mesh")
    sets = {entry["coil_set"] for entry in coils.values()}
    for wanted in sorted(split_names):
        if not {f"{wanted}_U", f"{wanted}_L"} <= sets:
            raise ValueError(
                f"split coil {wanted!r} (split_coils/vsc_coil) matches no "
                "pf_active coil. Available coils: "
                f"{', '.join(sorted({s.rsplit('_', 1)[0] for s in sets}))}"
            )
    return coils


def tokamaker_geometry_from_ods(ods: Any, config: TokaMakerConfig) -> dict:
    """Build the TokaMaker geometry dict from an ODS.

    Keys: ``limiter`` + ``coils`` always; ``vessel`` (conductor regions from
    ``pf_passive``, see :mod:`.vessel`) only when ``config.include_vessel`` —
    so v1-style configs produce byte-identical geometry and mesh-cache hashes.
    """
    limr, limz = _limiter_from_ods(ods, config)
    geometry = {
        "limiter": [[float(r), float(z)] for r, z in zip(limr, limz)],
        "coils": _coil_rectangles_from_ods(ods, config),
    }
    if config.include_vessel:
        from .vessel import vessel_segments_from_ods

        geometry["vessel"] = vessel_segments_from_ods(ods, config)
    return geometry


def geometry_signature(geometry: dict, config: TokaMakerConfig) -> str:
    """Short stable hash of the geometry and the config fields that shape the mesh.

    Used as the mesh-cache key: any change to the limiter, the coil rectangles,
    or a mesh resolution invalidates the cached HDF5 mesh.
    """
    payload = {
        "geometry": geometry,
        "dx_plasma": config.dx_plasma,
        "dx_coil": config.dx_coil,
        "dx_vacuum": config.dx_vacuum,
        "include_vessel": config.include_vessel,
        "dx_conductor": config.dx_conductor,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]
