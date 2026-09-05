"""The plasma current as a set of PF-like elements: the ``pf_plasma`` IDS.

IMAS describes an axisymmetric plasma current, for vacuum-field and
coupling calculations, as ``pf_plasma.element[i]`` with a static geometry
(a rectangle or an outline, like a coil turn), an ``area``, and a current
on the IDS time base.  An equilibrium reconstruction that models the
plasma as filaments or as a grid of current elements (VEST-Tokamak/vfit#3,
#4) stores that representation here, so a consumer can rebuild the
plasma's field from the same Green's functions it uses for coils and the
passive wall, and compare reconstruction sources on equal terms.

This module is the reader and writer for that layout; nothing here knows
which code produced the elements.  Elements are rectangles by default
(``geometry_type = 2``), the form VEST already uses for every coil turn
and passive loop, so the same outline helpers draw them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "GEOMETRY_TYPE_RECTANGLE",
    "plasma_current_total",
    "plasma_elements",
    "set_plasma_elements",
]

GEOMETRY_TYPE_RECTANGLE = 2


def set_plasma_elements(
    ods: Any,
    r: np.ndarray,
    z: np.ndarray,
    *,
    width: float | np.ndarray,
    height: float | np.ndarray,
    currents: np.ndarray,
    time: np.ndarray,
    code_name: str | None = None,
    code_parameters: str | None = None,
    comment: str | None = None,
) -> None:
    """Write ``pf_plasma`` as rectangular elements carrying ``currents``.

    ``r``, ``z`` are element centres ``(n,)``; ``width``/``height`` a scalar
    or ``(n,)``; ``currents`` is ``(n, n_times)`` on ``time`` (a ``(n,)``
    vector is a single instant).  The IDS is homogeneous in time and any
    existing ``pf_plasma`` content is replaced: the representation is one
    object, written whole.  ``code_name`` and ``code_parameters`` record
    the producer (``code.parameters`` is free text, XML by IMAS habit).
    """
    r = np.asarray(r, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    time = np.atleast_1d(np.asarray(time, dtype=float))
    currents = np.asarray(currents, dtype=float)
    if currents.ndim == 1:
        currents = currents[:, None]
    if r.shape != z.shape:
        raise ValueError(f"r {r.shape} and z {z.shape} must match")
    if currents.shape != (r.size, time.size):
        raise ValueError(
            f"currents must have shape (n_elements={r.size}, n_times={time.size}), got {currents.shape}"
        )
    width = np.broadcast_to(np.asarray(width, dtype=float), r.shape)
    height = np.broadcast_to(np.asarray(height, dtype=float), r.shape)
    if np.any(width <= 0) or np.any(height <= 0):
        raise ValueError("element width and height must be positive")

    if "pf_plasma" in ods:
        del ods["pf_plasma"]
    ods["pf_plasma.ids_properties.homogeneous_time"] = 1
    if comment is not None:
        ods["pf_plasma.ids_properties.comment"] = str(comment)
    ods["pf_plasma.time"] = time
    if code_name is not None:
        ods["pf_plasma.code.name"] = str(code_name)
    if code_parameters is not None:
        ods["pf_plasma.code.parameters"] = str(code_parameters)
    for i in range(r.size):
        base = f"pf_plasma.element.{i}"
        ods[f"{base}.geometry.geometry_type"] = GEOMETRY_TYPE_RECTANGLE
        ods[f"{base}.geometry.rectangle.r"] = float(r[i])
        ods[f"{base}.geometry.rectangle.z"] = float(z[i])
        ods[f"{base}.geometry.rectangle.width"] = float(width[i])
        ods[f"{base}.geometry.rectangle.height"] = float(height[i])
        ods[f"{base}.area"] = float(width[i] * height[i])
        ods[f"{base}.current"] = currents[i]


def _count(ods: Any, path: str) -> int:
    try:
        return len(ods[path]) if path in ods else 0
    except (KeyError, TypeError, ValueError):
        return 0


def plasma_current_total(ods: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(time, sum of element currents)`` -- the plasma current the elements carry."""
    n = _count(ods, "pf_plasma.element")
    if n == 0:
        raise ValueError("pf_plasma carries no elements")
    time = np.asarray(ods["pf_plasma.time"], dtype=float)
    total = np.zeros(time.size)
    for i in range(n):
        total = total + np.asarray(ods[f"pf_plasma.element.{i}.current"], dtype=float)
    return time, total


def plasma_elements(ods: Any, time: float | None = None) -> dict[str, Any]:
    """The elements' centres, sizes, areas and currents at one instant.

    ``time`` selects the nearest sample of ``pf_plasma.time``; ``None``
    takes the instant of largest total current.  Rectangle elements report
    ``width``/``height``; outline elements report their centroid and the
    bounding box.  Returns a dict of arrays (``r, z, width, height, area,
    current, geometry_type``) plus ``time`` (the sample used), ``index``
    (its position) and ``total`` (the summed current there).
    """
    n = _count(ods, "pf_plasma.element")
    if n == 0:
        raise ValueError("pf_plasma carries no elements")
    axis, total = plasma_current_total(ods)
    index = int(np.argmax(np.abs(total))) if time is None else int(np.argmin(np.abs(axis - float(time))))
    out = {key: np.empty(n) for key in ("r", "z", "width", "height", "area", "current")}
    out["geometry_type"] = np.empty(n, dtype=int)
    for i in range(n):
        base = f"pf_plasma.element.{i}"
        geometry = f"{base}.geometry"
        kind = int(ods[f"{geometry}.geometry_type"]) if f"{geometry}.geometry_type" in ods else GEOMETRY_TYPE_RECTANGLE
        if f"{geometry}.rectangle.r" in ods:
            out["r"][i] = float(ods[f"{geometry}.rectangle.r"])
            out["z"][i] = float(ods[f"{geometry}.rectangle.z"])
            out["width"][i] = float(ods[f"{geometry}.rectangle.width"])
            out["height"][i] = float(ods[f"{geometry}.rectangle.height"])
        elif f"{geometry}.outline.r" in ods:
            r = np.asarray(ods[f"{geometry}.outline.r"], dtype=float)
            z = np.asarray(ods[f"{geometry}.outline.z"], dtype=float)
            out["r"][i], out["z"][i] = float(r.mean()), float(z.mean())
            out["width"][i], out["height"][i] = float(np.ptp(r)), float(np.ptp(z))
        else:
            raise ValueError(f"{base} has neither a rectangle nor an outline geometry")
        out["geometry_type"][i] = kind
        out["area"][i] = float(ods[f"{base}.area"]) if f"{base}.area" in ods else out["width"][i] * out["height"][i]
        current = np.asarray(ods[f"{base}.current"], dtype=float)
        out["current"][i] = float(current[index]) if current.size > index else float(current[-1])
    out["time"] = float(axis[index])
    out["index"] = index
    out["total"] = float(total[index])
    return out
