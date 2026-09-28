"""3-D scientific visualization and interoperability (issue #1087).

:mod:`vaft.plot` renders view models with Matplotlib and Plotly; this package
carries the same renderer-independent 3-D models to the tools that need
richer scene semantics, none of which the core depends on:

``to_pyvista`` / ``write_vtk`` / ``read_vtk_blocks``
    a :class:`~vaft.plot.models.Geometry3DLayers` as a PyVista ``MultiBlock``
    whose block tree follows each layer's ``group``, and its VTK-family files
    (``.vtm`` multiblock, ``.vtp`` polydata) for ParaView.  Needs
    ``pip install vaft[vtk]``.
``to_k3d`` / ``coil_phase_explorer``
    the same scene as a K3D plot in Jupyter, and an interactive
    non-axisymmetric coil phasing view whose controls drive existing VAFT
    computation (:func:`vaft.process.coils_non_axisymmetric.phased_sector_currents`,
    :func:`~vaft.process.coils_non_axisymmetric.biot_savart_filaments`).
    Needs ``pip install vaft[jupyter3d]``.

A scene comes from any 3-D recipe, e.g.
``vaft.plot.extract("machine_geometry3d", ods)``; physics and data APIs never
return PyVista or K3D objects.  ParaView is an external application that
reads the exported files, not a dependency.

The layers are points and polylines.  Surface meshes, grids and fields on
them belong to the scientific mesh representation (#909, #1100) and are not
built here.  Importing this package imports neither PyVista nor K3D.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "coil_phase_explorer",
    "read_vtk_blocks",
    "to_k3d",
    "to_pyvista",
    "write_vtk",
]

_LAZY = {
    "to_pyvista": "._vtk_export",
    "write_vtk": "._vtk_export",
    "read_vtk_blocks": "._vtk_export",
    "to_k3d": "._k3d_scene",
    "coil_phase_explorer": "._k3d_scene",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        value = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
