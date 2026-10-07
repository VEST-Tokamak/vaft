"""One same-shot equilibrium section in R-Z and calibrated camera pixels.

Axisymmetric outlines are shown in the explicitly named toroidal plane. Probe
positions retain their own phi; a missing probe phi never becomes section_phi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vaft.machine_mapping.registry import port_phi
from vaft.plot.backend.access import array, count, get, has
from vaft.plot.models import GeometryLayer, GeometryLayers
from vaft.process.equilibrium import extract_flux_surface_contours

DEFAULT_SECTION_PHI = port_phi("6MR")
DEFAULT_FLUX_LEVELS = (0.25, 0.5, 0.75, 0.95)


def valid_equilibrium_indices(data: Any) -> tuple[int, ...]:
    """Stored slices with an LCFS and a nondegenerate psi normalization."""
    times = array(data, "equilibrium.time")
    if times is None:
        return ()
    valid = []
    for index in range(times.size):
        base = f"equilibrium.time_slice.{index}"
        axis = get(data, f"{base}.global_quantities.psi_axis")
        boundary = get(data, f"{base}.global_quantities.psi_boundary")
        if (np.isfinite(times[index]) and has(data, f"{base}.boundary.outline.r")
                and has(data, f"{base}.boundary.outline.z")
                and axis is not None and boundary is not None
                and np.isfinite(axis) and np.isfinite(boundary) and float(axis) != float(boundary)):
            valid.append(index)
    return tuple(valid)


@dataclass(frozen=True)
class Section:
    """Matched R-Z and camera layers plus the equilibrium time they depict."""

    rz: GeometryLayers
    camera_layers: tuple[GeometryLayer, ...]
    equilibrium_index: int
    equilibrium_time: float
    section_phi: float


def _joined(outlines: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Close independent contours and separate them with NaN for LineCollection."""
    r_parts, z_parts = [], []
    for r, z in outlines:
        r, z = np.asarray(r, dtype=float).ravel(), np.asarray(z, dtype=float).ravel()
        if r.size < 2 or r.shape != z.shape:
            continue
        if r[0] != r[-1] or z[0] != z[-1]:
            r, z = np.r_[r, r[0]], np.r_[z, z[0]]
        r_parts.extend((r, np.array([np.nan])))
        z_parts.extend((z, np.array([np.nan])))
    if not r_parts:
        return np.array([]), np.array([])
    return np.concatenate(r_parts), np.concatenate(z_parts)


def _camera_layer(layer: GeometryLayer, phi: np.ndarray | float, projection: Any) -> GeometryLayer:
    r, z = layer.r, layer.z
    angles = np.full(r.shape, float(phi)) if np.ndim(phi) == 0 else np.asarray(phi, dtype=float)
    if angles.shape != r.shape:
        raise ValueError("phi must match the source geometry shape")
    xyz = np.column_stack((r * np.cos(angles), r * np.sin(angles), z))
    uv = np.full((r.size, 2), np.nan)
    finite = np.isfinite(xyz).all(axis=1)
    if finite.any():
        projected, valid = projection.project(xyz[finite] * 100.0)
        projected = np.asarray(projected, dtype=float).copy()
        projected[~np.asarray(valid, dtype=bool)] = np.nan
        uv[finite] = projected
    return GeometryLayer(uv[:, 0], uv[:, 1], kind="points" if layer.kind == "points" else "polyline",
                         label=layer.label, style=layer.style, role=layer.role)


def _contains_point(r: np.ndarray, z: np.ndarray, x: float, y: float) -> bool:
    """Ray crossing for a closed flux contour, independent of a plot backend."""
    crossing = (z[:-1] > y) != (z[1:] > y)
    intersection = r[:-1] + (r[1:] - r[:-1]) * (y - z[:-1]) / np.where(
        z[1:] == z[:-1], 1.0, z[1:] - z[:-1])
    return bool(np.count_nonzero(crossing & (x < intersection)) % 2)


def build_equilibrium_section(
    data: Any, *, time: float | None = None, time_slice: int | None = None,
    section_phi: float = DEFAULT_SECTION_PHI,
    flux_surface_levels: tuple[float, ...] = DEFAULT_FLUX_LEVELS,
    projection: Any = None,
) -> Section:
    """Build the same source section for an R-Z view and an optional camera view.

    ``time`` is a real camera stamp when present. No equilibrium extrapolation
    is permitted. The nearest stored slice is reported so the approximation is
    visible in both the figure and the movie sidecar.
    """
    from vaft.plot.backend.recipes import _element_outlines

    times = array(data, "equilibrium.time")
    if times is None or not times.size or not np.isfinite(times).all():
        raise ValueError("equilibrium.time must contain finite stored slice times")
    valid_indices = valid_equilibrium_indices(data)
    if not valid_indices:
        raise ValueError("no stored equilibrium slice has a complete section")
    if time is not None and time_slice is not None:
        raise ValueError("give time or time_slice, not both")
    if time is not None:
        if not np.isfinite(time) or not float(times[valid_indices[0]]) <= float(time) <= float(times[valid_indices[-1]]):
            raise ValueError("camera frame time is outside the valid equilibrium interval")
        index = min(valid_indices, key=lambda i: abs(float(times[i]) - float(time)))
    else:
        index = 0 if time_slice is None else int(time_slice)
        if index not in valid_indices:
            raise ValueError(f"equilibrium time_slice {index} has no complete section")
    if not np.isfinite(section_phi):
        raise ValueError("section_phi must be finite radians")
    eq_time = float(times[index])
    prefix = f"equilibrium.time_slice.{index}"
    rz: list[GeometryLayer] = []
    camera: list[GeometryLayer] = []

    def add(layer: GeometryLayer, phi: np.ndarray | float | None = None) -> None:
        rz.append(layer)
        if projection is not None and phi is not None:
            projected = _camera_layer(layer, phi, projection)
            if np.isfinite(projected.r).any():
                camera.append(projected)

    for name, container, color, width in (
        ("PF active", "pf_active.coil", "tab:orange", 0.65),
        ("PF passive", "pf_passive.loop", "tab:green", 0.35),
    ):
        outlines = []
        for i in range(count(data, container)):
            outlines.extend(_element_outlines(data, f"{container}.{i}"))
        r, z = _joined(outlines)
        if r.size:
            add(GeometryLayer(r, z, label=name, style={"color": color, "lw": width}), section_phi)

    wall = []
    for i in range(count(data, "wall.description_2d.0.limiter.unit")):
        base = f"wall.description_2d.0.limiter.unit.{i}.outline"
        r, z = array(data, f"{base}.r"), array(data, f"{base}.z")
        if r is not None and z is not None:
            wall.append((r, z))
    r, z = _joined(wall)
    if r.size:
        add(GeometryLayer(r, z, label="Limiter", style={"color": "yellow", "lw": 1.0}), section_phi)

    for constraint, sensor, position, label, color in (
        ("flux_loop", "magnetics.flux_loop", "position.0", "EFIT flux loops", "tab:red"),
        ("bpol_probe", "magnetics.b_field_pol_probe", "position", "EFIT B-pol probes", "tab:blue"),
    ):
        names = {str(get(data, f"{prefix}.constraints.{constraint}.{i}.source"))
                 for i in range(count(data, f"{prefix}.constraints.{constraint}"))}
        named = set()
        first = True
        for i in range(count(data, sensor)):
            base = f"{sensor}.{i}"
            name = str(get(data, f"{base}.name", ""))
            if not name or name not in names:
                continue
            if name in named:
                raise ValueError(f"duplicate mapped equilibrium sensor name: {name}")
            named.add(name)
            r = array(data, f"{base}.{position}.r")
            z = array(data, f"{base}.{position}.z")
            if r is None or z is None or r.size != 1 or z.size != 1:
                continue
            r, z = np.asarray(r, dtype=float).reshape(1), np.asarray(z, dtype=float).reshape(1)
            layer = GeometryLayer(r, z, kind="points", label=label if first else "",
                                  style={"color": color, "marker": "s" if constraint == "flux_loop" else "x",
                                         "markersize": 3})
            first = False
            rz.append(layer)
            if projection is None:
                continue
            if constraint == "flux_loop":
                # A flux loop encircles the axis. Its IDS has no scalar phi.
                angles = np.linspace(0.0, 2.0 * np.pi, 181)
                ring = GeometryLayer(np.full(angles.shape, float(r[0])),
                                     np.full(angles.shape, float(z[0])),
                                     label=layer.label, style={"color": color, "lw": 0.45})
                projected = _camera_layer(ring, angles, projection)
            else:
                phi = array(data, f"{base}.{position}.phi")
                if phi is None or phi.size != 1 or not np.isfinite(np.asarray(phi).ravel()[0]):
                    continue
                projected = _camera_layer(layer, float(np.asarray(phi).ravel()[0]), projection)
            if np.isfinite(projected.r).any():
                camera.append(projected)

    boundary_r = array(data, f"{prefix}.boundary.outline.r")
    boundary_z = array(data, f"{prefix}.boundary.outline.z")
    if boundary_r is None or boundary_z is None or boundary_r.shape != boundary_z.shape:
        raise ValueError(f"{prefix} has no complete boundary outline")
    r, z = _joined([(boundary_r, boundary_z)])
    add(GeometryLayer(r, z, label="LCFS", style={"color": "magenta", "lw": 1.4},
                      role="equilibrium"), section_phi)
    axis_r = get(data, f"{prefix}.global_quantities.magnetic_axis.r")
    axis_z = get(data, f"{prefix}.global_quantities.magnetic_axis.z")
    if axis_r is not None and axis_z is not None:
        add(GeometryLayer([axis_r], [axis_z], kind="points", label="Magnetic axis",
                          style={"color": "cyan", "marker": "+", "markersize": 7},
                          role="equilibrium"), section_phi)
    if flux_surface_levels:
        grid_r = array(data, f"{prefix}.profiles_2d.0.grid.dim1")
        grid_z = array(data, f"{prefix}.profiles_2d.0.grid.dim2")
        psi = array(data, f"{prefix}.profiles_2d.0.psi")
        psi_axis = get(data, f"{prefix}.global_quantities.psi_axis")
        psi_boundary = get(data, f"{prefix}.global_quantities.psi_boundary")
        if any(value is None for value in (grid_r, grid_z, psi, psi_axis, psi_boundary)):
            raise ValueError(f"{prefix} lacks the psi grid or normalization for flux surfaces")
        if psi.shape != (grid_r.size, grid_z.size):
            raise ValueError("equilibrium psi grid must be indexed (R, Z)")
        surfaces = extract_flux_surface_contours(
            psi, grid_r, grid_z, float(psi_axis), float(psi_boundary), flux_surface_levels)
        first = True
        for level in sorted(surfaces):
            for curve_r, curve_z in surfaces[level]:
                # A psi value can also occur in open branches outside the
                # plasma. Only the closed contour enclosing the magnetic axis
                # is the nested flux surface represented by this level.
                if (curve_r.size < 4 or not np.allclose(
                        [curve_r[0], curve_z[0]], [curve_r[-1], curve_z[-1]], atol=1e-8)
                        or axis_r is None or axis_z is None
                        or not _contains_point(curve_r, curve_z, float(axis_r), float(axis_z))):
                    continue
                layer = GeometryLayer(curve_r, curve_z, label="Flux surfaces" if first else "",
                                      style={"color": "tab:cyan", "lw": 0.55, "ls": ":"},
                                      role="equilibrium")
                add(layer, section_phi)
                first = False

    title = f"VEST equilibrium section at phi={np.rad2deg(section_phi):.1f}°; t_eq={eq_time * 1e3:.1f} ms"
    return Section(GeometryLayers(tuple(rz), title=title), tuple(camera), index, eq_time, float(section_phi))
