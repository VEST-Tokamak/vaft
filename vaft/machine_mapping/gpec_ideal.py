"""Ideal-GPEC output mapping into ``mhd_linear``.

Like :mod:`vaft.machine_mapping.mhd_linear` (issue #170), this module is the
IDS-populating layer only: it never re-parses ``.nc`` files itself, it reads
the native container from :func:`vaft.code.gpec.read_gpec_netcdf` and copies
only quantities with a scientifically correct IMAS home.  Everything else
stays in :class:`~vaft.code.gpec.GpecIdealResult` (persisted as a JSON
sidecar next to the solver output).

Mapping table -- one row per GPEC variable/attribute; the "legacy" column
records how ``GPEC_Research/library/gpec_imas.py`` handled the same quantity
and whether that behavior was kept or revised:

====================== ============ ======== ================================================== ===================
GPEC source            native dims  units    destination                                        status
====================== ============ ======== ================================================== ===================
attr ``n``             scalar       --       ``toroidal_mode[p].n_tor``                         revised-from-legacy
                                             (legacy inferred n from the *filename*)
cyl ``R`` / ``z``      (R,) (z,)    m        ``plasma/vacuum.grid.dim1`` / ``.dim2``            verified
cyl ``b_r_plasma``     (i,z,R)      T        ``plasma.b_field_perturbed.coordinate1.real/imag`` verified (transpose
cyl ``b_z_plasma``                           ``...coordinate2...``                              to (dim1, dim2))
cyl ``b_t_plasma``                           ``...coordinate3...``
cyl ``b_[rzt]`` total  (i,z,R)      T        vacuum = total - plasma (exact linear                verified, tagged
                                             superposition), tagged in ``code.parameters``       in provenance
attr energies          scalars      J        kept as *separate* attributes in                   revised-from-legacy
                                             ``code.parameters``; ``energy_perturbed`` carries  (legacy silently
                                             their documented sum                               summed them)
ctrl ``Phi_coil``,     (i,coil,m)   Wb       none -- native container / JSON sidecar only       verified-unmapped
eigen-decompositions,
``A_*``, ``xi_*``,
``b_n_fun`` etc.
``coil.in`` currents   per sector   A        ``coils_non_axisymmetric`` via                     revised-from-legacy
                                             :func:`vaft.machine_mapping.coils_non_axisymmetric.apply_coil_excitation`
                                             (legacy hard-coded ``turns=1``; canonical mapping
                                             carries turns=20 from the ``.dat`` header)
attr ``shot``/``time`` scalars      -- / s   written by the *caller* (options) -- GPEC records  revised-from-legacy
                                             0/0 when the gfile header has none
====================== ============ ======== ================================================== ===================
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
from omas import ODS

from vaft.code.gpec import GpecIdealResult, read_gpec_netcdf

from .mhd_linear import (
    _append_code_parameters,
    _set_output_flag,
    ensure_toroidal_mode_grid,
)

__all__ = ["gpec_ideal"]

_GRID_DESCRIPTION = "GPEC cylindrical (R, z) grid; dim1=R [m], dim2=z [m]"


def _write_vector_field(entry: ODS, region: str, components) -> None:
    """Write complex ``(z, R)`` components as ``(dim1, dim2)=(R, z)`` fields."""
    for coordinate, values in zip(("coordinate1", "coordinate2", "coordinate3"), components):
        if values is None:
            continue
        node = entry[region]["b_field_perturbed"][coordinate]
        node["real"] = np.ascontiguousarray(values.real.T)
        node["imaginary"] = np.ascontiguousarray(values.imag.T)


def _write_mode_entry(
    ods: ODS,
    time_slice: int,
    position: int,
    result: GpecIdealResult,
    *,
    include_vacuum: bool,
) -> None:
    control = result.control
    entry = ods["mhd_linear"]["time_slice"][time_slice]["toroidal_mode"][position]
    entry["n_tor"] = control.n_tor
    entry["perturbation_type"]["name"] = "coil"
    entry["perturbation_type"]["description"] = (
        "Ideal plasma response to a non-axisymmetric coil field (GPEC)"
    )
    # GPEC's perturbed-energy attributes are dimensional Joules for the total
    # response; the documented total goes into the IMAS field while the
    # decomposition stays structured in code.parameters below.
    entry["energy_perturbed"] = float(control.energy_total)

    cylindrical = result.cylindrical
    if cylindrical is not None:
        regions = ["plasma"]
        if include_vacuum:
            regions.append("vacuum")
        for region in regions:
            grid = entry[region]["grid"]
            grid["dim1"] = np.asarray(cylindrical.R, dtype=float)
            grid["dim2"] = np.asarray(cylindrical.z, dtype=float)
            entry[region]["grid_type"]["name"] = "cylindrical_rz"
            entry[region]["grid_type"]["description"] = _GRID_DESCRIPTION
        _write_vector_field(
            entry,
            "plasma",
            (cylindrical.b_r_plasma, cylindrical.b_z_plasma, cylindrical.b_t_plasma),
        )
        if include_vacuum:
            vacuum_components = tuple(
                None if total is None or plasma is None else total - plasma
                for total, plasma in (
                    (cylindrical.b_r, cylindrical.b_r_plasma),
                    (cylindrical.b_z, cylindrical.b_z_plasma),
                    (cylindrical.b_t, cylindrical.b_t_plasma),
                )
            )
            _write_vector_field(entry, "vacuum", vacuum_components)

    fragment = (
        f'<solver name="gpec" n_tor="{control.n_tor}">'
        f'<jacobian>{control.jacobian}</jacobian>'
        f'<helicity>{control.helicity}</helicity>'
        f'<energy_vacuum units="J">{control.energy_vacuum!r}</energy_vacuum>'
        f'<energy_surface units="J">{control.energy_surface!r}</energy_surface>'
        f'<energy_plasma units="J">{control.energy_plasma!r}</energy_plasma>'
        '<energy_perturbed derivation="energy_vacuum+energy_surface+energy_plasma"'
        ' units="J" source="gpec_control_output global attributes"/>'
        '<vacuum derivation="total_minus_plasma"'
        ' source="gpec_cylindrical_output b_[rzt] - b_[rzt]_plasma"/>'
        "</solver>"
    )
    _append_code_parameters(ods, "mhd_linear", fragment, code_name="GPEC")
    if control.version:
        ods["mhd_linear.code.version"] = control.version
    _set_output_flag(ods, "mhd_linear", time_slice, 0)


def gpec_ideal(ods: ODS, source: str, options: Optional[dict] = None) -> dict[int, dict[str, Any]]:
    """Map one ideal-GPEC run directory into ``mhd_linear``.

    ``source`` is a completed ideal-GPEC run directory (one toroidal mode).
    Options:

    - ``time_slice`` (default 0) and ``modes`` (full requested ``n_tor``
      grid) lay the ``toroidal_mode`` AOS out densely, exactly as
      :func:`vaft.machine_mapping.mhd_linear.mhd_linear` does.
    - ``mode`` selects which ``gpec_*_output_n<mode>.nc`` set to read when
      the directory holds several.
    - ``time_s`` writes the time base.  GPEC's own ``shot``/``time``
      attributes are 0 when the equilibrium header carries no identity (true
      for the VEST reference run), so the caller supplies them; the native
      attributes are never trusted for this.
    - ``include_vacuum`` (default True) also writes the vacuum field as
      total minus plasma -- exact under GPEC's linear superposition, and
      tagged as derived in ``code.parameters``.

    Returns ``{n_tor: {...}}`` with values kept alongside the ODS in the
    caller's manifest.  As a side effect, the lossless native container is
    written to ``<source>/gpec_ideal_native_n<mode>.json``.
    """
    options = dict(options or {})
    time_slice = int(options.get("time_slice", 0))
    include_vacuum = bool(options.get("include_vacuum", True))

    result = read_gpec_netcdf(source, options.get("mode"))
    n_tor = result.n_tor

    grid = [int(n) for n in options.get("modes", [])]
    if n_tor not in grid:
        grid.append(n_tor)
    ensure_toroidal_mode_grid(ods, time_slice, grid)

    time_s = options.get("time_s")
    if time_s is not None:
        ods["mhd_linear.ids_properties.homogeneous_time"] = 1
        times = np.atleast_1d(np.asarray(ods.get("mhd_linear.time", []), dtype=float))
        if time_slice >= times.size:
            times = np.concatenate([times, np.full(time_slice + 1 - times.size, np.nan)])
        times[time_slice] = float(time_s)
        ods["mhd_linear.time"] = times
        ods["mhd_linear.time_slice"][time_slice]["time"] = float(time_s)

    position = grid.index(n_tor)
    _write_mode_entry(ods, time_slice, position, result, include_vacuum=include_vacuum)

    try:
        result.write_json(os.path.join(str(source), f"gpec_ideal_native_n{n_tor}.json"))
    except OSError:
        pass

    return {
        n_tor: {
            "module": "gpec",
            "energy_perturbed": result.control.energy_total,
            "coil_names": list(result.control.coil_names),
            "has_cylindrical": result.cylindrical is not None,
        }
    }
