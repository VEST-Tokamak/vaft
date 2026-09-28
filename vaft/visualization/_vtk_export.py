"""PyVista/VTK conversion and VTK-family export of 3-D scenes (issue #1087).

A :class:`~vaft.plot.models.Geometry3DLayers` becomes a ``pyvista.MultiBlock``:

* each layer is one ``PolyData`` -- a polyline keeps its vertex order as
  line-cell connectivity (a NaN vertex splits it, as Matplotlib draws it),
  a point layer is one vertex cell per point;
* the block tree follows the layer's ``group`` path
  (``machine/pf_active/PF1/0`` is block ``PF1``'s child ``0`` under
  ``pf_active`` under ``machine``), so ParaView lists and toggles subsystems;
  a layer without a group lands under ``layers``;
* every leaf carries ``vaft_label``, ``vaft_group``, ``vaft_kind`` and
  ``vaft_color`` (RGB in ``[0, 1]``) as field data, the root the scene's
  ``vaft_title`` and axis labels.

Coordinates are written as they are: metres on the machine Cartesian axes of
:func:`vaft.machine_mapping.conventions.cylindrical_to_cartesian`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vaft.plot.models import Geometry3DLayer

from ._optional import require_pyvista
from ._scene import as_layers, finite_runs, layer_rgb

__all__ = ["read_vtk_blocks", "to_pyvista", "write_vtk"]

#: The suffixes :func:`write_vtk` writes.
VTK_SUFFIXES = (".vtm", ".vtp")


def _cells(layer: Geometry3DLayer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(points, verts, lines)`` of one layer in VTK's flat cell layout, finite vertices only."""
    runs = finite_runs(layer)
    if layer.kind != "points":
        runs = [run for run in runs if run.size >= 2]  # a lone vertex of a polyline draws nothing
    kept = np.concatenate(runs) if runs else np.zeros(0, dtype=int)
    points = np.column_stack([layer.x, layer.y, layer.z])[kept]
    renumber = {int(old): new for new, old in enumerate(kept)}
    if layer.kind == "points":
        verts = np.column_stack([np.ones(len(kept), dtype=int), np.arange(len(kept))]).ravel()
        return points, verts, np.zeros(0, dtype=int)
    lines: list[int] = []
    for run in runs:
        lines += [run.size, *(renumber[int(index)] for index in run)]
    return points, np.zeros(0, dtype=int), np.asarray(lines, dtype=int)


def _layer_metadata(layer: Geometry3DLayer) -> dict[str, Any]:
    return {
        "vaft_label": layer.label,
        "vaft_group": layer.group,
        "vaft_kind": layer.kind,
        "vaft_color": np.asarray(layer_rgb(layer), dtype=float),
    }


def _polydata(pv: Any, layer: Geometry3DLayer) -> Any:
    points, verts, lines = _cells(layer)
    cells = {}
    if verts.size:
        cells["verts"] = verts
    if lines.size:
        cells["lines"] = lines
    mesh = pv.PolyData(points, **cells)
    for key, value in _layer_metadata(layer).items():
        mesh.field_data[key] = value if not isinstance(value, str) else [value]
    return mesh


def _unique(block: Any, name: str) -> str:
    taken = set(block.keys())
    if name not in taken:
        return name
    index = 2
    while f"{name} #{index}" in taken:
        index += 1
    return f"{name} #{index}"


def _descend(pv: Any, block: Any, name: str) -> Any:
    """The child multiblock ``name`` of ``block``, created when missing."""
    if name in block.keys() and isinstance(block[name], pv.MultiBlock):
        return block[name]
    child = pv.MultiBlock()
    block[_unique(block, name)] = child
    return child


def to_pyvista(model: Any) -> Any:
    """A 3-D scene as a ``pyvista.MultiBlock`` whose tree follows each layer's ``group``.

    ``model`` is a :class:`~vaft.plot.models.Geometry3DLayers` (or one
    :class:`~vaft.plot.models.Geometry3DLayer`), e.g. from
    ``vaft.plot.extract("machine_geometry3d", ods)``.  Needs ``vaft[vtk]``.
    """
    pv = require_pyvista()
    scene = as_layers(model)
    root = pv.MultiBlock()
    for key in ("title", "x_label", "y_label", "z_label"):
        root.field_data[f"vaft_{key}"] = [str(getattr(scene, key))]
    for index, layer in enumerate(scene.layers):
        if not len(_cells(layer)[0]):
            continue  # nothing finite to draw: no block, in either file format
        path = layer.group.split("/") if layer.group else ["layers", f"layer {index}"]
        block = root
        for name in path[:-1]:
            block = _descend(pv, block, name)
        block[_unique(block, path[-1])] = _polydata(pv, layer)
    return root


def _merged_polydata(pv: Any, scene: Any) -> Any:
    """Every layer in one ``PolyData``; the cell array ``vaft_layer`` says which layer a cell is."""
    points, verts, lines, vert_layer, line_layer = [], [], [], [], []
    offset = 0
    drawn = [layer for layer in scene.layers if len(_cells(layer)[0])]
    for index, layer in enumerate(drawn):
        layer_points, layer_verts, layer_lines = _cells(layer)
        for flat, out, owner in ((layer_verts, verts, vert_layer), (layer_lines, lines, line_layer)):
            position = 0
            while position < flat.size:
                size = int(flat[position])
                out += [size, *(flat[position + 1:position + 1 + size] + offset)]
                owner.append(index)
                position += size + 1
        points.append(layer_points)
        offset += len(layer_points)
    stacked = np.concatenate(points) if points else np.zeros((0, 3))
    cells = {}
    if verts:
        cells["verts"] = np.asarray(verts, dtype=int)
    if lines:
        cells["lines"] = np.asarray(lines, dtype=int)
    mesh = pv.PolyData(stacked, **cells) if len(stacked) else pv.PolyData()
    # VTK orders a PolyData's cells verts first, then lines.
    mesh.cell_data["vaft_layer"] = np.asarray(vert_layer + line_layer, dtype=np.int32)
    mesh.field_data["vaft_label"] = [layer.label for layer in drawn]
    mesh.field_data["vaft_group"] = [layer.group for layer in drawn]
    mesh.field_data["vaft_kind"] = [layer.kind for layer in drawn]
    mesh.field_data["vaft_color"] = np.asarray([layer_rgb(layer) for layer in drawn], dtype=float).reshape(-1, 3)
    for key in ("title", "x_label", "y_label", "z_label"):
        mesh.field_data[f"vaft_{key}"] = [str(getattr(scene, key))]
    return mesh


def write_vtk(model: Any, path: str | Path) -> Path:
    """Write a 3-D scene as a VTK-family file ParaView opens; returns the path written.

    ``.vtm``
        the :func:`to_pyvista` multiblock -- subsystems stay separate blocks.
        VTK writes the leaves beside it in a directory of the same stem.
    ``.vtp``
        one merged polydata; the cell array ``vaft_layer`` and the per-layer
        field data (``vaft_label``, ``vaft_group``, ...) keep layer identity.

    A layer with nothing drawable -- no finite point, or a polyline of lone
    vertices only -- is left out of both formats.
    """
    pv = require_pyvista()
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in VTK_SUFFIXES:
        raise ValueError(f"write_vtk writes {', '.join(VTK_SUFFIXES)}; got {target.name!r}")
    scene = as_layers(model)
    data = to_pyvista(scene) if suffix == ".vtm" else _merged_polydata(pv, scene)
    data.save(str(target))
    return target


def _text(field_data: Any, key: str) -> list[str]:
    if key not in field_data.keys():
        return []
    return [str(value) for value in np.atleast_1d(field_data[key])]


def _unpack(flat: np.ndarray) -> list[np.ndarray]:
    cells, position = [], 0
    while position < flat.size:
        size = int(flat[position])
        cells.append(np.asarray(flat[position + 1:position + 1 + size], dtype=int))
        position += size + 1
    return cells


def _leaf_record(mesh: Any) -> dict[str, Any]:
    label, group, kind = (next(iter(_text(mesh.field_data, key)), "") for key in ("vaft_label", "vaft_group", "vaft_kind"))
    return {
        "points": np.asarray(mesh.points, dtype=float).reshape(-1, 3),
        "lines": _unpack(np.asarray(mesh.lines)),
        "verts": _unpack(np.asarray(mesh.verts)),
        "label": label,
        "group": group,
        "kind": kind,
    }


def read_vtk_blocks(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read a file :func:`write_vtk` wrote back into one record per layer.

    Keys are block paths (``machine/pf_active/PF1/0``); each record holds
    ``points`` ``(N, 3)``, the ``lines`` and ``verts`` connectivity as index
    arrays into those points, and ``label``, ``group`` and ``kind``.  A
    round-trip check for tests and examples, not a general VTK reader.
    """
    pv = require_pyvista()
    data = pv.read(str(path))
    records: dict[str, dict[str, Any]] = {}
    if isinstance(data, pv.MultiBlock):
        def walk(block: Any, prefix: str) -> None:
            for name in block.keys():
                child = block[name]
                key = f"{prefix}/{name}" if prefix else str(name)
                if isinstance(child, pv.MultiBlock):
                    walk(child, key)
                elif child is not None:
                    records[key] = _leaf_record(child)
        walk(data, "")
        return records
    labels, groups, kinds = (_text(data.field_data, key) for key in ("vaft_label", "vaft_group", "vaft_kind"))
    owner = np.asarray(data.cell_data["vaft_layer"], dtype=int)
    verts, lines = _unpack(np.asarray(data.verts)), _unpack(np.asarray(data.lines))
    cell_owner = {"verts": owner[:len(verts)], "lines": owner[len(verts):]}
    points = np.asarray(data.points, dtype=float).reshape(-1, 3)
    for index, group in enumerate(groups):
        own = {kind: [cell for cell, who in zip(cells, cell_owner[kind]) if who == index]
               for kind, cells in (("verts", verts), ("lines", lines))}
        used = np.unique(np.concatenate([*own["verts"], *own["lines"]])) if (own["verts"] or own["lines"]) else np.zeros(0, dtype=int)
        renumber = {int(old): new for new, old in enumerate(used)}
        key = group or f"layers/layer {index}"
        if key in records:  # the .vtm block tree suffixes a repeated name the same way
            suffix = 2
            while f"{key} #{suffix}" in records:
                suffix += 1
            key = f"{key} #{suffix}"
        records[key] = {
            "points": points[used],
            "lines": [np.array([renumber[int(i)] for i in cell]) for cell in own["lines"]],
            "verts": [np.array([renumber[int(i)] for i in cell]) for cell in own["verts"]],
            "label": labels[index],
            "group": group,
            "kind": kinds[index],
        }
    return records
