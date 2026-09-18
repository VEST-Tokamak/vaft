"""Canonical ODS to NICE-native geometry conversion."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _polygon(r, z) -> list[list[float]]:
    points = np.column_stack((np.asarray(r, float), np.asarray(z, float)))
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3 or not np.isfinite(points).all() or np.any(points[:, 0] <= 0):
        raise ValueError(
            "NICE geometry polygons require at least three finite points with R > 0"
        )
    area2 = np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - np.roll(points[:, 0], -1) * points[:, 1]
    )
    if area2 < 0:
        points = points[::-1]
    return points.tolist()


def _element_polygon(ods: Any, base: str) -> list[list[float]]:
    try:
        return _polygon(
            ods[f"{base}.geometry.outline.r"], ods[f"{base}.geometry.outline.z"]
        )
    except Exception:
        pass
    try:
        rect = ods[f"{base}.geometry.rectangle"]
        r, z = float(rect["r"]), float(rect["z"])
        dr, dz = float(rect["width"]) / 2, float(rect["height"]) / 2
        return _polygon(
            [r - dr, r + dr, r + dr, r - dr], [z - dz, z - dz, z + dz, z + dz]
        )
    except Exception as exc:
        raise ValueError(f"Unsupported NICE conductor geometry at {base}") from exc


def _connected_components(entries: list[dict[str, Any]], tolerance: float = 1.0e-4):
    """Group touching winding elements into NICE's physical coil regions."""
    boxes = []
    for entry in entries:
        points = np.asarray(entry["outline"], float)
        boxes.append(
            (
                points[:, 0].min(),
                points[:, 0].max(),
                points[:, 1].min(),
                points[:, 1].max(),
            )
        )
    remaining = set(range(len(entries)))
    groups = []
    while remaining:
        group, frontier = set(), {remaining.pop()}
        while frontier:
            current = frontier.pop()
            group.add(current)
            a = boxes[current]
            touching = {
                other
                for other in remaining
                if a[0] <= boxes[other][1] + tolerance
                and boxes[other][0] <= a[1] + tolerance
                and a[2] <= boxes[other][3] + tolerance
                and boxes[other][2] <= a[3] + tolerance
            }
            remaining.difference_update(touching)
            frontier.update(touching)
        groups.append(sorted(group))
    return groups


def _aggregate_active_coil(name: str, coil_index: int, entries: list[dict[str, Any]]):
    """Collapse a turn-level winding mesh into small rectangular regions.

    NICE integrates each rectangle with a fixed quadrature.  A whole VEST
    solenoid is too slender for that quadrature, so connected winding packs
    are deterministically split into groups of at most six ODS elements.  The
    complete machine remains below NICE's 100-coil compile-time limit while
    retaining the spatial current distribution near the probes.
    """
    result = []
    for region_index, members in enumerate(_connected_components(entries)):
        all_points = np.concatenate(
            [np.asarray(entries[index]["outline"], float) for index in members]
        )
        span = np.ptp(all_points, axis=0)
        primary, secondary = (1, 0) if span[1] >= span[0] else (0, 1)
        ordered = sorted(
            members,
            key=lambda index: (
                np.mean(entries[index]["outline"], axis=0)[primary],
                np.mean(entries[index]["outline"], axis=0)[secondary],
            ),
        )
        for segment_index, segment in enumerate(
            [ordered[start : start + 6] for start in range(0, len(ordered), 6)]
        ):
            points = np.concatenate(
                [np.asarray(entries[index]["outline"], float) for index in segment]
            )
            r0, r1 = float(points[:, 0].min()), float(points[:, 0].max())
            z0, z1 = float(points[:, 1].min()), float(points[:, 1].max())
            turns = float(sum(entries[index]["turns"] for index in segment))
            result.append(
                {
                    "name": f"{name}:{region_index}:{segment_index}",
                    "ods_path": f"pf_active.coil.{coil_index}",
                    "outline": _polygon([r0, r1, r1, r0], [z0, z0, z1, z1]),
                    "turns": turns,
                    "coil_index": coil_index,
                    "source_element_indices": segment,
                }
            )
    return result


def nice_geometry_from_ods(ods: Any) -> dict[str, Any]:
    """Map wall, PF-active and PF-passive directly from their canonical IDSs."""
    limiter = _polygon(
        ods["wall.description_2d.0.limiter.unit.0.outline.r"],
        ods["wall.description_2d.0.limiter.unit.0.outline.z"],
    )
    active = []
    for i in range(len(ods["pf_active.coil"])):
        coil = ods[f"pf_active.coil.{i}"]
        name = str(coil["name"]) if "name" in coil else f"PF{i+1}"
        elements = []
        for j in range(len(coil["element"])):
            base = f"pf_active.coil.{i}.element.{j}"
            elements.append(
                {
                    "name": f"{name}:{j}",
                    "ods_path": base,
                    "outline": _element_polygon(ods, base),
                    "turns": float(ods[f"{base}.turns_with_sign"]),
                    "coil_index": i,
                }
            )
        active.extend(_aggregate_active_coil(name, i, elements))
    passive = []
    for i in range(len(ods["pf_passive.loop"])):
        loop = ods[f"pf_passive.loop.{i}"]
        name = (
            str(loop["name"])
            if "name" in loop
            else str(loop["identifier"])
            if "identifier" in loop
            else f"passive:{i}"
        )
        for j in range(len(loop["element"])):
            base = f"pf_passive.loop.{i}.element.{j}"
            passive.append(
                {
                    "name": f"{name}:{j}",
                    "ods_path": base,
                    "outline": _element_polygon(ods, base),
                    "turns": 1.0,
                    "loop_index": i,
                }
            )
    return {"limiter": limiter, "pf_active": active, "pf_passive": passive}


def geometry_hash(geometry: dict[str, Any]) -> str:
    payload = json.dumps(
        geometry, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_contour(r, z, geometry):
    """Reject intersections and reversed VacTH integration orientation."""
    contour = np.column_stack((r, z)).astype(float)
    if len(contour) < 3 or not np.isfinite(contour).all() or np.any(contour[:, 0] <= 0):
        raise ValueError("Invalid computational contour")
    area = np.sum(
        contour[:, 0] * np.roll(contour[:, 1], -1)
        - contour[:, 1] * np.roll(contour[:, 0], -1)
    )
    if area <= 0:
        raise ValueError("NICE computational contour must be counter-clockwise")

    def edges(poly):
        return list(zip(poly, np.roll(poly, -1, axis=0)))

    def crosses(a, b, c, d):
        def side(p, q, s):
            u, v = q - p, s - p
            return u[0] * v[1] - u[1] * v[0]

        if np.any(
            np.maximum(np.minimum(a, b), np.minimum(c, d))
            > np.minimum(np.maximum(a, b), np.maximum(c, d)) + 1e-12
        ):
            return False
        return (
            side(a, b, c) * side(a, b, d) <= 1e-24
            and side(c, d, a) * side(c, d, b) <= 1e-24
        )

    ce = edges(contour)
    for i, (a, b) in enumerate(ce):
        for j in range(i + 2, len(ce)):
            if i == 0 and j == len(ce) - 1:
                continue
            if crosses(a, b, *ce[j]):
                raise ValueError("Computational contour self-intersects")
    for label, poly in [("limiter", geometry["limiter"])] + [
        (c["name"], c["outline"])
        for c in geometry["pf_active"] + geometry.get("pf_passive", [])
    ]:
        if any(crosses(a, b, c, d) for a, b in ce for c, d in edges(np.asarray(poly))):
            raise ValueError(f"Computational contour intersects {label}")
    # Once crossings are excluded, a ray-cast of every limiter vertex verifies
    # that this is an enclosing computational boundary, not an interior loop.
    for x, y in geometry["limiter"]:
        inside = False
        for (a, b) in ce:
            if (a[1] > y) != (b[1] > y):
                if x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
                    inside = not inside
        if not inside:
            raise ValueError("Computational contour does not enclose limiter")
