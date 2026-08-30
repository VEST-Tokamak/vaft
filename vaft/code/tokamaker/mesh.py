"""Build and cache the TokaMaker finite-element mesh for a machine geometry.

Meshing is the slow step of a TokaMaker run and depends only on the machine
geometry and per-region resolutions — not on currents, targets, or profiles —
so it is decoupled from the solve and cached as an HDF5 file (see
``resolve_mesh_file`` in ``inputs``). Requires an importable OpenFUSIONToolkit.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ._oft import import_oft
from .config import TokaMakerConfig

_log = logging.getLogger(__name__)


def build_tokamaker_mesh(
    geometry: dict, mesh_file: Path | str, config: TokaMakerConfig
) -> Path:
    """Mesh the geometry with gs_Domain and save it as an HDF5 cache file.

    Regions: an auto-generated rectangular ``boundary`` region ("AIR"), the
    ``plasma`` region bounded by the limiter polygon, one ``coil`` region per
    rectangle (grouped into coil sets via ``coil_set``), and — when the
    geometry carries a ``vessel`` key (``config.include_vessel``) — one
    ``conductor`` region per vessel polygon with its resistivity [Ohm·m].
    Conductors are inert in static solves (TokaMaker gates every eddy term on
    ``dt > 0``), so one mesh serves both static and time-dependent runs.

    Idempotent: an existing ``mesh_file`` is returned unchanged; delete or
    rename it to force a rebuild.
    """
    mesh_file = Path(mesh_file).expanduser()
    if mesh_file.is_file():
        _log.info("Reusing cached TokaMaker mesh %s", mesh_file)
        return mesh_file
    vessel = geometry.get("vessel", {})
    if config.include_vessel and not vessel:
        raise ValueError(
            "include_vessel=True but the geometry has no 'vessel' regions; "
            "rebuild it with tokamaker_geometry_from_ods and the same config."
        )

    oft = import_oft()
    limiter = np.asarray(geometry["limiter"], dtype=float)

    gs_mesh = oft.meshing.gs_Domain()
    gs_mesh.define_region("air", config.dx_vacuum, "boundary")
    gs_mesh.define_region("plasma", config.dx_plasma, "plasma")
    for name, coil in geometry["coils"].items():
        gs_mesh.define_region(
            name,
            config.dx_coil,
            "coil",
            nTurns=coil["nturns"],
            coil_set=coil["coil_set"],
        )
    for name, cond in vessel.items():
        gs_mesh.define_region(
            name,
            cond["dx"],
            "conductor",
            eta=cond["eta"],
            noncontinuous=cond["noncontinuous"] or None,
        )

    gs_mesh.add_polygon(limiter, "plasma", parent_name="air")
    for name, coil in geometry["coils"].items():
        gs_mesh.add_rectangle(
            coil["rc"], coil["zc"], coil["w"], coil["h"], name, parent_name="air"
        )
    for name, cond in vessel.items():
        gs_mesh.add_polygon(
            np.asarray(cond["contour"], dtype=float), name, parent_name="air"
        )

    mesh_pts, mesh_lc, mesh_reg = gs_mesh.build_mesh()
    coil_dict = gs_mesh.get_coils()
    cond_dict = gs_mesh.get_conductors()

    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    oft.meshing.save_gs_mesh(mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict, str(mesh_file))
    _log.info(
        "Built TokaMaker mesh %s: %d points, %d cells, %d regions",
        mesh_file, len(mesh_pts), len(mesh_lc), len(coil_dict) + len(cond_dict) + 1,
    )
    return mesh_file
