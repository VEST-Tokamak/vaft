"""Source geometry shared by machine views (metres, IMAS phi in radians).

This extraction layer uses the same accessors and renderer models as canonical
plots. It retains source coordinates and measurement semantics independently of
any view. Missing phi is unknown: only the stored R-Z coordinates are usable.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import json
import re
from typing import Any, Mapping

import numpy as np

from .backend.access import array, count, get, has
from .models import Geometry3DLayer, Geometry3DLayers, GeometryLayer, GeometryLayers, as_model_array

__all__ = ["MachineGeometry", "MACHINE_GEOMETRY_FAMILIES", "CROSS_SHOT_NOTICE",
           "cross_shot_notice", "machine_geometry_registry",
           "machine_geometry_view", "project_machine_geometry"]


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
# Charge exchange stores each coordinate as a time trace (``position.r.data``);
# the other point families store a static ``position.r``.
_DYNAMIC_POSITIONS = {"charge_exchange"}
_LOS = {"interferometer": "interferometer.channel", "soft_x_rays": "soft_x_rays.channel"}
# Only the interferometer line of sight has a reflection point in the DD.
_REFLECTED_LOS = {"interferometer"}
MACHINE_GEOMETRY_FAMILIES = (*_POINTS, *_LOS, "coils_non_axisymmetric", "ec_launchers", "nbi")
_FAMILY_LABELS = {
    "thomson_scattering": "Thomson channel sites",
    "charge_exchange": "CX measurement sites",
    "langmuir_probes": "Langmuir sites",
    "interferometer": "Interferometer LOS",
    "soft_x_rays": "Soft X-ray LOS",
    "coils_non_axisymmetric": "3-D coil filament",
    "ec_launchers": "EC provisional CAD",
    "nbi": "NBI model geometry",
}


CROSS_SHOT_NOTICE = "Cross-shot composite — not a physical VEST discharge"


def cross_shot_notice(reference: Any) -> str:
    """Two-line notice every view of the cross-shot fixture carries in its title.

    The first line says the picture is not one discharge; the second says whose
    machine geometry the other shots' diagnostic coordinates are projected onto.
    The static renderer, the camera view and the shared machine view all take the
    wording from here so the three cannot drift apart.
    """
    return (f"{CROSS_SHOT_NOTICE}\n"
            f"Other-shot diagnostic coordinates projected onto geometry reference shot {reference}")


def _position(data: Any, path: str, *, dynamic: bool = False) -> tuple[float, float, float | None] | None:
    values = []
    for coordinate in ("r", "z", "phi"):
        value = _static_scalar(data, f"{path}.{coordinate}.data" if dynamic else f"{path}.{coordinate}")
        if value is None:
            if coordinate == "phi":
                values.append(None)
                continue
            if coordinate == "z":
                values.append(float("nan"))
                continue
            return None
        values.append(value)
    return tuple(values)


def _static_scalar(data: Any, path: str) -> float | None:
    """A finite scalar or constant time trace; varying traces need a time choice."""
    value = array(data, path)
    if value is None or not np.isfinite(value).all():
        return None
    flat = value.ravel()
    return float(flat[0]) if np.all(flat == flat[0]) else None


def machine_geometry_registry(data: Any, *, families: tuple[str, ...] | None = None,
                              manifest: Mapping[str, Any] | None = None) -> tuple[MachineGeometry, ...]:
    """Extract stored geometry without importing a renderer or mutating data.

    ``manifest`` carries fixture provenance and explicit mapper-derived
    geometry for paths that DD 3.41 cannot represent.
    Unknown families raise; mapped families with absent coordinates yield no
    record. Gas injection and CES LOS are intentionally not registered.
    """
    selected = MACHINE_GEOMETRY_FAMILIES if families is None else families
    unknown = set(selected) - set(MACHINE_GEOMETRY_FAMILIES)
    if unknown:
        raise ValueError(f"unsupported geometry families: {sorted(unknown)}")
    records = []

    def provenance(family: str, identifier: str) -> str:
        source_catalog = {
            **(manifest or {}).get("sources", {}),
            **(manifest or {}).get("geometry_sources", {}),
        }
        sources = {key: value for key, value in source_catalog.items()
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
                position_path = f"{base}.position"
                position = _position(data, position_path, dynamic=family in _DYNAMIC_POSITIONS)
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
                third = _position(data, third_path) if family in _REFLECTED_LOS else None
                phi_complete = third is not None and (third[2] is not None or all(point[2] is None for point in positions))
                z_complete = third is not None and (np.isfinite(third[1]) or all(not np.isfinite(point[1]) for point in positions))
                if phi_complete and z_complete:
                    positions.append(third)
                elif family in _REFLECTED_LOS and has(data, third_path):
                    # An incomplete reflection point is not an absent point.
                    # Drawing only the first leg would silently change the LOS.
                    continue
                add(family, path, "line_of_sight", positions,
                    str(get(data, f"{base}.name", f"{family} {index}")),
                    str(get(data, f"{base}.identifier", "")))
        elif family == "ec_launchers":
            for index in range(count(data, "ec_launchers.beam")):
                beam = f"ec_launchers.beam.{index}"
                position = _position(data, f"{beam}.launching_position")
                if position is None:
                    continue
                label = str(get(data, f"{beam}.name", f"EC beam {index}"))
                add(family, f"{beam}.launching_position", "point", [position], label)
                pol = _static_scalar(data, f"{beam}.steering_angle_pol")
                tor = _static_scalar(data, f"{beam}.steering_angle_tor")
                if position[2] is None or not np.isfinite(position[1]) or pol is None or tor is None:
                    continue
                phi = position[2]
                kr, kphi, kz = (-np.cos(pol) * np.cos(tor), np.sin(tor),
                                 -np.sin(pol) * np.cos(tor))
                direction = [kr * np.cos(phi) - kphi * np.sin(phi),
                             kr * np.sin(phi) + kphi * np.cos(phi), kz]
                records.append(MachineGeometry(
                    family, "directed_axis", [position[0]], [position[1]], [phi],
                    f"{label} launch axis", f"{beam}.steering_angle_pol/tor",
                    provenance(family, ""), direction,
                ))
        elif family == "nbi":
            for unit in range(count(data, "nbi.unit")):
                for group in range(count(data, f"nbi.unit.{unit}.beamlets_group")):
                    position_path = f"nbi.unit.{unit}.beamlets_group.{group}.position"
                    position = _position(data, position_path)
                    if position is not None:
                        add(family, position_path, "point", [position],
                            str(get(data, f"nbi.unit.{unit}.name", f"NBI unit {unit}")))
            # The IDS states tangency magnitude and injection sense but not
            # aperture Z. Without that height there is no 3-D beam direction.
        elif family == "coils_non_axisymmetric":
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
    # The current VEST Thomson mapper writes the port-map phi at each channel
    # but DD 3.41 has no laser-chord leaf. Attach its explicit source chord to
    # an ordinary shot only when two stored channel positions independently
    # agree with that mapper. A generic TS IDS or unmapped/old sample cannot
    # silently acquire VEST toroidal coordinates.
    manifest_has_laser = any(entry.get("family") == "thomson_scattering" and entry.get("semantic") == "trajectory"
                             for entry in (manifest or {}).get("geometry_records", ()))
    if "thomson_scattering" in selected and not manifest_has_laser:
        machine = get(data, "dataset_description.data_entry.machine")
        shot = get(data, "dataset_description.data_entry.pulse")
        try:
            source_shot = int(shot) if shot is not None else None
        except (TypeError, ValueError):
            source_shot = None
        if str(machine).upper() == "VEST" and source_shot is not None and source_shot > 0:
            from vaft.machine_mapping.thomson_scattering import laser_chord_positions, scattering_volume_phi

            sites = [record for record in records if record.family == "thomson_scattering"
                     and record.semantic == "point" and record.phi is not None]
            agrees = len(sites) >= 2 and len({float(site.r[0]) for site in sites}) >= 2
            for site in sites:
                try:
                    mapped_phi = scattering_volume_phi(float(site.r[0]))
                except ValueError:
                    agrees = False
                    break
                agrees &= bool(np.isclose(site.z[0], 0.0, atol=1e-12, rtol=0)
                               and np.isclose(np.angle(np.exp(1j * (site.phi[0] - mapped_phi))),
                                              0.0, atol=1e-10, rtol=0))
            if agrees:
                positions = laser_chord_positions()
                r, z, phi = zip(*positions)
                records.append(MachineGeometry(
                    "thomson_scattering", "trajectory", r, z, phi,
                    "Thomson laser port chord (derived; not as-built)",
                    "vaft.machine_mapping.thomson_scattering.laser_chord_positions",
                    json.dumps({"data_shot": source_shot, "sources": {"thomson_port_map": {
                        "source_artifact": "vaft/machine_mapping/thomson_scattering.py",
                        "geometry_era": "unverified static port-map model",
                        "value_kind": "derived port-map geometry; not surveyed as-built",
                    }}, "derivation": "8MM10 entry to 1MM10 dump; 0.803 m port-flange radius; verified against stored channel phi"},
                               sort_keys=True),
                ))
    # These paths have no suitable standard IDS leaf in DD 3.41. The manifest
    # contains source-mapper outputs with explicit derivation and source hashes,
    # not a renderer-side reconstruction of a plotted shape.
    for index, entry in enumerate((manifest or {}).get("geometry_records", ())):
        family = entry["family"]
        if family not in selected:
            continue
        source_catalog = {
            **(manifest or {}).get("sources", {}),
            **(manifest or {}).get("geometry_sources", {}),
        }
        sources = {key: source_catalog[key] for key in entry["source_keys"]}
        source = {
            "sources": sources,
            "geometry_reference": (manifest or {}).get("geometry_reference"),
            "physical_discharge": (manifest or {}).get("physical_discharge"),
            "kind": (manifest or {}).get("kind"),
            "derivation": entry["derivation"],
            "value_kind": entry["value_kind"],
        }
        records.append(MachineGeometry(
            family=family, semantic=entry["semantic"], r=entry["r_m"],
            z=entry["z_m"], phi=entry["phi_rad"], label=entry["label"],
            source_path=f"manifest.geometry_records.{index}",
            provenance_json=json.dumps(source, sort_keys=True),
            direction_xyz=entry.get("direction_xyz"),
        ))
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
                if view == "rz" and np.isfinite(record.z).all() else None)
    # A gap row (NaN in r, z and phi) marks a stored discontinuity and is kept
    # as a polyline break; only a missing height at a known position makes
    # the record undrawable in a view that needs Z.
    if view in {"rz", "3d", "camera"} and not np.isfinite(record.z[np.isfinite(record.r)]).all():
        return None
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
    uv = np.full((xyz.shape[0], 2), np.nan)
    stored = np.isfinite(xyz).all(axis=1)
    if stored.any():
        projected, valid = projection.project(xyz[stored] * 100.0)
        projected = np.array(projected, dtype=float, copy=True)
        projected[~np.asarray(valid, dtype=bool)] = np.nan
        uv[stored] = projected
    return GeometryLayer(uv[:, 0], uv[:, 1], kind=kind, label=label)


def machine_geometry_view(data: Any, view: str, *, families: tuple[str, ...] | None = None,
                          manifest: Mapping[str, Any] | None = None, projection: Any = None,
                          axis_length: float = 0.25, samples_per_segment: int = 32
                          ) -> GeometryLayers | Geometry3DLayers:
    """One family-selected view over the same immutable machine-space records.

    Camera view requires an existing calibrated ``CameraProjection``. Source
    data and manifest stay independent of renderer and presentation backend.
    """
    if view == "camera" and projection is None:
        raise ValueError("camera view requires a calibrated CameraProjection")
    records = machine_geometry_registry(data, families=families, manifest=manifest)
    layers = []
    labelled: set[tuple[str, str, bool]] = set()
    for index, record in enumerate(records):
        layer = project_machine_geometry(
            record, view, projection=projection, axis_length=axis_length,
            samples_per_segment=samples_per_segment,
        )
        if layer is None or (view == "camera" and not np.isfinite(layer.r).any()):
            continue
        position = MACHINE_GEOMETRY_FAMILIES.index(record.family)
        style = {"color": f"C{position}"}
        if record.semantic == "point":
            style["marker"] = "x" if "derived" in record.label.lower() else "o"
        elif record.semantic in {"trajectory", "directed_axis"}:
            style["linestyle"] = "--"
        derived_thomson = (record.family == "thomson_scattering" and
                           (record.source_path.startswith("manifest.geometry_records") or
                            record.source_path.startswith("vaft.machine_mapping.thomson_scattering.")))
        key = (record.family, record.semantic, derived_thomson)
        if derived_thomson:
            name = ("Thomson derived laser chord" if record.semantic == "trajectory"
                    else "Thomson derived scattering sites")
        elif record.family == "ec_launchers":
            name = ("EC launch axis (provisional CAD)" if record.semantic == "directed_axis"
                    else "EC launch position (provisional CAD)")
        elif record.family == "nbi":
            name = ("NBI model beam axis" if record.semantic == "directed_axis"
                    else "NBI model source")
        elif record.family == "coils_non_axisymmetric" and view == "rz":
            name = "3-D coil filament (R-Z projection)"
        else:
            name = _FAMILY_LABELS[record.family]
        label = name if key not in labelled else ""
        labelled.add(key)
        if view == "3d":
            layer = replace(layer, label=label, style=style,
                            group=f"{record.family}/{record.semantic}/{index}")
        else:
            layer = replace(layer, label=label, style=style)
        layers.append(layer)
    notice = (manifest or {}).get("notice", "")
    if (manifest or {}).get("kind") == "cross-shot-diagnostic-fixture":
        reference = (manifest or {}).get("geometry_reference", {}).get("source_shot")
        notice = cross_shot_notice(reference)
    elif not notice:
        source_comment = get(data, "dataset_description.ids_properties.comment")
        if isinstance(source_comment, str) and "Cross-shot composite fixture" in source_comment:
            notice = source_comment.replace("Cross-shot composite fixture", "Cross-shot composite")
    title = "Machine geometry" + (f"\n{notice}" if notice else "")
    if view == "3d":
        return Geometry3DLayers(tuple(layers), title=title)
    axes = {"rz": ("R [m]", "Z [m]"), "top": ("X [m]", "Y [m]"),
            "camera": ("u [px]", "v [px]")}
    if view not in axes:
        raise ValueError(f"unknown machine view: {view}")
    return GeometryLayers(tuple(layers), x_label=axes[view][0],
                          y_label=axes[view][1], title=title)
