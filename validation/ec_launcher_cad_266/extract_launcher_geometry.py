"""Reduce the EC officer's 6 kW ECH STEP assembly to the launch condition in vest.yaml.

Issue #266. Reproduces every number in ``vest.yaml`` ``0: ec_launchers: beams:
ech_6kw`` from ``MID 6KW ECH ASSY.stp`` (not in the repository; ask on #266).
Needs OpenCascade, which VAFT does not depend on::

    python -m venv occ && occ/bin/pip install cadquery-ocp numpy
    occ/bin/python extract_launcher_geometry.py "MID 6KW ECH ASSY.stp"

Method: the ``MIDDLE SHIELD PART`` is the vessel section, whose r = 800 mm
cylinder gives the machine axis. Its local +Y lies along that axis and is taken
as up; the midpoint of its axial extent as z = 0, checked against the
symmetry of its port holes. The port tube's bore (r = 79.2 mm) gives the launch
axis, and the tube's end surface crossing that axis gives the launch point.
The WR340 inner walls give the TE10 polarization.
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCP.GProp import GProp_GProps
from OCP.OCP.collections import Sequence_TDF_Label
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool

VESSEL_RADIUS_MM = 800.0
BORE_RADIUS_MM = 79.2


def _components(path: str) -> dict[str, object]:
    document = TDocStd_Document(TCollection_ExtendedString("doc"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.ReadFile(path)
    reader.Transfer(document)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    roots = Sequence_TDF_Label()
    tool.GetFreeShapes(roots)
    labels = Sequence_TDF_Label()
    tool.GetComponents_s(roots.Value(1), labels, False)
    out = {}
    for index in range(1, labels.Length() + 1):
        label = labels.Value(index)
        name = TDataStd_Name()
        label.FindAttribute(TDataStd_Name.GetID_s(), name)
        out[name.Get().ToExtString().split(":")[0]] = (tool.GetShape_s(label), tool.GetLocation_s(label))
    return out


def _faces(shape):
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        yield face, BRepAdaptor_Surface(face)
        explorer.Next()


def _vec(xyz) -> np.ndarray:
    return np.array([xyz.X(), xyz.Y(), xyz.Z()])


def _cylinder(shape, radius: float):
    for _, surface in _faces(shape):
        if surface.GetType() == GeomAbs_Cylinder and abs(surface.Cylinder().Radius() - radius) < 0.3:
            axis = surface.Cylinder().Axis()
            return _vec(axis.Location()), _vec(axis.Direction())
    raise LookupError(f"no r = {radius} mm cylinder")


def extract(path: str) -> dict[str, object]:
    parts = _components(path)
    shield, shield_location = parts["MIDDLE SHIELD PART"]
    origin, ez = _cylinder(shield, VESSEL_RADIUS_MM)
    local_y = np.array([shield_location.Transformation().Value(i, 2) for i in (1, 2, 3)])
    ez = ez if ez @ local_y > 0 else -ez  # shield local +Y is up
    # z = 0 is the midpoint of the shield's axial extent, not wherever the
    # cylinder surface happens to be placed; the port holes must sit
    # symmetrically about it for that to be the midplane.
    along = []
    explorer = TopExp_Explorer(shield, TopAbs_VERTEX)
    while explorer.More():
        along.append(float((_vec(BRep_Tool.Pnt_s(TopoDS.Vertex(explorer.Current()))) - origin) @ ez))
        explorer.Next()
    shield_half_height_mm = 0.5 * (max(along) - min(along))
    origin = origin + 0.5 * (max(along) + min(along)) * ez
    port_z = []
    for _, surface in _faces(shield):
        if surface.GetType() == GeomAbs_Cylinder and 20.0 < surface.Cylinder().Radius() < 200.0:
            axis = surface.Cylinder().Axis()
            port_z.append(round(float((_vec(axis.Location()) - origin) @ ez), 2))
    port_z = sorted(set(port_z))
    if not np.allclose(port_z, [-z for z in reversed(port_z)], atol=0.5):
        raise RuntimeError(f"shield port holes are not symmetric about its centre: {port_z}")

    def axial(point):
        v = np.asarray(point) - origin
        z = float(v @ ez)
        return float(np.linalg.norm(v - z * ez)), z

    tube, _ = parts["CF10 FLANGE TUBE"]
    bore_point, bore_dir = _cylinder(tube, BORE_RADIUS_MM)
    if bore_dir @ (bore_point - origin) > 0:
        bore_dir = -bore_dir  # point into the vessel
    # Where the bore axis passes the machine axis, and how far it misses.
    w = bore_point - origin
    b = float(bore_dir @ ez)
    t = (b * float(w @ ez) - float(w @ bore_dir)) / (1.0 - b * b)
    closest = bore_point + t * bore_dir
    miss_mm = float(np.linalg.norm(np.cross(closest - origin, ez)))

    # The tube's plasma-facing end is cut on an r = 800 mm cylinder of its own.
    end_point, end_dir = _cylinder(tube, VESSEL_RADIUS_MM)

    def radial_gap(s):
        v = bore_point + s * bore_dir - end_point
        return float(np.linalg.norm(v - (v @ end_dir) * end_dir)) - VESSEL_RADIUS_MM

    samples = np.linspace(-500.0, 1500.0, 4001)
    values = np.array([radial_gap(s) for s in samples])
    crossing = np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))
    lo, hi = samples[crossing[0]], samples[crossing[0] + 1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.sign(radial_gap(mid)) == np.sign(radial_gap(lo)) else (lo, mid)
    launch_r, launch_z = axial(bore_point + lo * bore_dir)

    # Launch direction in (R, phi, Z) at the launch point.
    e_r = bore_point + lo * bore_dir - origin
    e_r -= (e_r @ ez) * ez
    e_r /= np.linalg.norm(e_r)
    e_phi = np.cross(ez, e_r)
    direction = [float(bore_dir @ e_r), float(bore_dir @ e_phi), float(bore_dir @ ez)]

    # WR340 inner walls: the pair of planes 86.36 mm apart is the broad dimension.
    stub, _ = parts["WR340_3stub"]
    walls: dict[str, list[float]] = {"z": [], "phi": []}
    for face, surface in _faces(stub):
        if surface.GetType() != GeomAbs_Plane:
            continue
        normal = _vec(surface.Plane().Axis().Direction())
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        centre = _vec(props.CentreOfMass()) - origin
        if props.Mass() < 5000.0:  # flange faces, not the guide walls
            continue
        if abs(abs(normal @ ez) - 1.0) < 1e-3:
            walls["z"].append(float(centre @ ez))
        elif abs(abs(normal @ e_phi) - 1.0) < 1e-3:
            walls["phi"].append(float(centre @ e_phi))
    span = {key: max(v) - min(v) for key, v in walls.items() if v}
    # TE10: E is parallel to the narrow dimension.
    polarization = [0.0, 0.0, 1.0] if span["z"] < span["phi"] else [0.0, 1.0, 0.0]

    def planes_on_axis(name, min_area_mm2=1000.0):
        """R of each planar face normal to the launch axis, measured on that axis."""
        shape, _ = parts[name]
        radii = set()
        for face, surface in _faces(shape):
            if surface.GetType() != GeomAbs_Plane:
                continue
            normal = _vec(surface.Plane().Axis().Direction())
            if abs(abs(normal @ bore_dir) - 1.0) > 1e-4:
                continue
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            if props.Mass() > min_area_mm2:
                radii.add(round(axial(_vec(props.CentreOfMass()))[0] / 1000.0, 4))
        return sorted(radii)

    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return {
        "source_sha256": digest,
        "launching_position": {"r": round(launch_r / 1000.0, 4), "z": round(launch_z / 1000.0, 4)},
        "direction": [round(value, 6) + 0.0 for value in direction],
        "polarization": polarization,
        "shield_half_height_mm": round(shield_half_height_mm, 2),
        "shield_port_hole_z_mm": port_z,
        "bore_axis_miss_mm": round(miss_mm, 4),
        "bore_axis_dot_machine_axis": round(b, 7),
        "waveguide_wall_span_mm": {key: round(value, 2) for key, value in span.items()},
        # The WR284 end is the transition's innermost face.
        "stations_r_m": {
            "quartz_window": planes_on_axis("QUARTZ WINDOW"),
            "wr284_wr340_transition": planes_on_axis("WR284_WR340_transition"),
        },
    }


if __name__ == "__main__":
    print(json.dumps(extract(sys.argv[1]), indent=2))
