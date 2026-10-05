"""Source geometry shared by machine views (metres, IMAS phi in radians).

This extraction layer uses the same accessors and renderer models as canonical
plots. It retains source coordinates and measurement semantics independently of
any view. Missing phi is unknown: only the stored R-Z coordinates are usable.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping

import numpy as np

from .backend.access import array, count, get, has
from .models import Geometry3DLayer, GeometryLayer, as_model_array

__all__ = ["MachineGeometry", "machine_geometry_registry", "project_machine_geometry"]


@dataclass(frozen=True)
class MachineGeometry:
    """One source point, trajectory, LOS, coil path, or directed launch axis.

    ``r``, ``z`` are metres; ``phi`` is radians or absent, never implicitly zero.
    ``direction_xyz`` is an explicit unit vector in machine Cartesian space for
    an axis. Provenance is serialized JSON so it cannot be mutated after the
    geometry is built. A source shot, era, and origin are supplied by the data
    provider's manifest; absent provenance remains unknown, not inferred.
    """
    family: str
    semantic: str
    r: np.ndarray
    z: np.ndarray
    phi: np.ndarray | None = None
    label: str = ""
    source_path: str = ""
    provenance_json: str = "{}"
    direction_xyz: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.semantic not in {"point", "trajectory", "line_of_sight", "coil_path", "directed_axis"}:
            raise ValueError(f"unknown geometry semantic: {self.semantic}")
        for name in ("r", "z", "phi", "direction_xyz"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, as_model_array(value, where=f"MachineGeometry.{name}"))
        if self.r.ndim != 1 or self.z.shape != self.r.shape or not self.r.size:
            raise ValueError("r and z must be nonempty, equally sized 1-D arrays")
        if np.any(self.r[np.isfinite(self.r)] < 0):
            raise ValueError("machine radius cannot be negative")
        if self.phi is not None and self.phi.shape != self.r.shape:
            raise ValueError("phi must match r and z")
        if self.semantic in {"point", "directed_axis"} and self.r.size != 1:
            raise ValueError("a point or directed axis has one source position")
        if self.semantic not in {"point", "directed_axis"} and self.r.size < 2:
            raise ValueError("a path needs at least two positions")
        if self.semantic == "directed_axis":
            if self.phi is None or self.direction_xyz is None or not np.isfinite(np.r_[self.r, self.z, self.phi]).all():
                raise ValueError("an axis requires phi and an explicit Cartesian direction")
            if self.direction_xyz.shape != (3,) or not np.isfinite(self.direction_xyz).all() or not np.isclose(np.linalg.norm(self.direction_xyz), 1):
                raise ValueError("axis direction must be a finite unit vector")
        elif self.direction_xyz is not None:
            raise ValueError("only a directed axis carries a direction")
        if not isinstance(json.loads(self.provenance_json), dict):
            raise ValueError("provenance_json must contain a JSON object")

    @property
    def xyz(self) -> np.ndarray | None:
        """Cartesian source vertices, or None when toroidal location is unknown."""
        if self.phi is None:
            return None
        return as_model_array(np.column_stack((self.r * np.cos(self.phi), self.r * np.sin(self.phi), self.z)), where="MachineGeometry.xyz")


# Family and IDS paths are source truth for every projection. CES contributes
# measurement sites only; no endpoint pair is invented from those sites.
_POINTS = {
    "thomson_scattering": "thomson_scattering.channel",
    "charge_exchange": "charge_exchange.channel",
    "langmuir_probes": "langmuir_probes.embedded",
}
_LOS = {"interferometer": "interferometer.channel", "soft_x_rays": "soft_x_rays.channel"}


def _position(data: Any, path: str) -> tuple[float, float, float | None] | None:
    values = []
    for coordinate in ("r", "z", "phi"):
        value = array(data, f"{path}.{coordinate}")
        if value is None:
            series = array(data, f"{path}.{coordinate}.data")
            # A time-dependent coordinate is static only when every stored
            # sample agrees. No shared fixture time or nearest slice is chosen.
            if series is not None and np.isfinite(series).all() and np.all(series == series.ravel()[0]):
                value = series.ravel()[:1]
        if value is None or value.size != 1 or not np.isfinite(value).all():
            if coordinate == "phi":
                values.append(None)
                continue
            return None
        values.append(float(value.ravel()[0]))
    return tuple(values)


def machine_geometry_registry(data: Any, *, families: tuple[str, ...] | None = None,
                              manifest: Mapping[str, Any] | None = None) -> tuple[MachineGeometry, ...]:
    """Extract stored geometry without importing a renderer or mutating data.

    ``manifest`` carries fixture provenance, not a replacement coordinate source.
    Unknown families raise; mapped families with absent coordinates yield no
    record. Gas injection and CES LOS are intentionally not registered.
    """
    supported = (*_POINTS, *_LOS, "coils_non_axisymmetric")
    selected = supported if families is None else families
    unknown = set(selected) - set(supported)
    if unknown:
        raise ValueError(f"unsupported geometry families: {sorted(unknown)}")
    records = []

    def provenance(family: str, identifier: str) -> str:
        sources = {key: value for key, value in (manifest or {}).get("sources", {}).items()
                   if family in value.get("ids", [])}
        if len(sources) > 1:
            # The fixture has two interferometer artifacts under one IDS. Its
            # channel identifier states the frequency; the source keys repeat
            # it. With no unique match retain candidates, explicitly marked
            # ambiguous, instead of attributing either artifact to the chord.
            frequency = re.search(r"(\d+)\s*ghz", identifier.lower())
            matching = {key: value for key, value in sources.items()
                        if frequency and key.lower().endswith(f"_{frequency[1]}ghz")}
            if len(matching) == 1:
                sources = matching
        return json.dumps({"sources": sources, "geometry_reference": (manifest or {}).get("geometry_reference"),
                           "kind": (manifest or {}).get("kind"), "physical_discharge": (manifest or {}).get("physical_discharge"),
                           "source_ambiguous": len(sources) > 1,
                           "ids_comment": str(get(data, f"{family}.ids_properties.comment", "")),
                           "ids_code_parameters": str(get(data, f"{family}.code.parameters", ""))}, sort_keys=True)

    def add(family: str, path: str, semantic: str, positions: list, label: str,
            identifier: str = "") -> None:
        r, z, phi = zip(*positions)
        records.append(MachineGeometry(family, semantic, r, z,
                       None if any(value is None for value in phi) else phi,
                       label, path, provenance(family, identifier)))

    for family in selected:
        if family in _POINTS:
            container = _POINTS[family]
            for index in range(count(data, container)):
                base = f"{container}.{index}"
                # DD lists charge-exchange and probe positions; TS is scalar.
                position_path = f"{base}.position"
                position = _position(data, position_path)
                if position is None:
                    position_path += ".0"
                    position = _position(data, position_path)
                if position is not None:
                    add(family, position_path, "point", [position], str(get(data, f"{base}.name", f"{family} {index}")))
        elif family in _LOS:
            container = _LOS[family]
            for index in range(count(data, container)):
                base = f"{container}.{index}"
                path = f"{base}.line_of_sight"
                positions = [_position(data, f"{path}.{end}") for end in ("first_point", "second_point")]
                if any(position is None for position in positions):
                    continue
                third_path = f"{path}.third_point"
                third = _position(data, third_path)
                if third is not None and (third[2] is not None or all(point[2] is None for point in positions)):
                    positions.append(third)
                elif has(data, third_path):
                    # An incomplete reflection point is not an absent point.
                    # Drawing only the first leg would silently change the LOS.
                    continue
                add(family, path, "line_of_sight", positions,
                    str(get(data, f"{base}.name", f"{family} {index}")),
                    str(get(data, f"{base}.identifier", "")))
        else:
            for index in range(count(data, "coils_non_axisymmetric.coil")):
                coil = f"coils_non_axisymmetric.coil.{index}"
                for conductor in range(count(data, f"{coil}.conductor")):
                    base = f"{coil}.conductor.{conductor}.elements"
                    starts = [array(data, f"{base}.start_points.{axis}") for axis in ("r", "z", "phi")]
                    ends = [array(data, f"{base}.end_points.{axis}") for axis in ("r", "z", "phi")]
                    if any(value is None for value in (*starts, *ends)):
                        continue
                    if any(value.ndim != 1 or value.shape != starts[0].shape for value in (*starts, *ends)) or not starts[0].size:
                        raise ValueError(f"inconsistent conductor elements at {base}")
                    # Preserve every segment and any discontinuity: do not join
                    # a start to the next start and discard stored endpoints.
                    positions = []
                    previous = None
                    for segment in range(starts[0].size):
                        start = tuple(value[segment] for value in starts)
                        end = tuple(value[segment] for value in ends)
                        if previous is not None:
                            def xyz(position):
                                r, z, phi = position
                                return (r * np.cos(phi), r * np.sin(phi), z)
                            if not np.allclose(xyz(previous), xyz(start), rtol=0, atol=1e-12):
                                positions.append((np.nan, np.nan, np.nan))
                        positions.extend([start, end])
                        previous = end
                    add(family, base, "coil_path", positions, str(get(data, f"{coil}.name", f"coil {index}")))
    return tuple(records)


def project_machine_geometry(record: MachineGeometry, view: str, *, projection: Any = None,
                             axis_length: float = 1.0, samples_per_segment: int = 32) -> GeometryLayer | Geometry3DLayer | None:
    """Adapt source geometry to existing view models.

    CameraProjection consumes cm and supplies its own calibration validity mask.
    Invalid vertices become NaN gaps. LOS and axes are sampled in Cartesian
    space before nonlinear projection. Axis length is presentation only.
    """
    if view not in {"rz", "top", "3d", "camera"}:
        raise ValueError(f"unknown machine view: {view}")
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least two")
    label = f"{record.label or record.family} [{record.semantic}]"
    kind = "points" if record.semantic == "point" else "polyline"
    xyz = record.xyz
    if xyz is None:
        # Without phi, only the stored R-Z vertices are known. A line between
        # them would assert a physical chord whose cylindrical projection is
        # unknown, so mark vertices as points even when their source is a LOS.
        return (GeometryLayer(record.r, record.z, kind="points",
                              label=f"{label} (vertices; phi unknown)")
                if view == "rz" else None)
    if record.semantic == "directed_axis":
        if not np.isfinite(axis_length) or axis_length <= 0:
            raise ValueError("axis_length must be finite and positive")
        xyz = np.stack((xyz[0], xyz[0] + axis_length * record.direction_xyz))
    if kind != "points":
        pieces = [np.linspace(start, end, samples_per_segment) for start, end in zip(xyz[:-1], xyz[1:])]
        xyz = np.concatenate(pieces) if pieces else xyz
    if view == "3d":
        return Geometry3DLayer(*xyz.T, kind=kind, label=label, group=f"{record.family}/{record.semantic}")
    if view == "rz":
        return GeometryLayer(np.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2], kind=kind, label=label)
    if view == "top":
        return GeometryLayer(xyz[:, 0], xyz[:, 1], kind=kind, label=label)
    if projection is None:
        raise ValueError("camera view requires a calibrated CameraProjection")
    uv, valid = projection.project(xyz * 100.0)
    uv = np.array(uv, dtype=float, copy=True)
    uv[~np.asarray(valid, dtype=bool)] = np.nan
    return GeometryLayer(uv[:, 0], uv[:, 1], kind=kind, label=label)
