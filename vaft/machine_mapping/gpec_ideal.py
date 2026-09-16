"""Ideal-GPEC output mapping into ``mhd_linear``.

Like :mod:`vaft.machine_mapping.mhd_linear` (issue #170), this module is the
IDS-populating layer only: it never re-parses ``.nc`` files itself, it reads
the native container from :func:`vaft.code.gpec.read_gpec_netcdf` and copies
only quantities with a scientifically correct IMAS home.  Everything else
stays in :class:`~vaft.code.gpec.GpecIdealResult` (persisted as a JSON
sidecar next to the solver output, which carries the control transcript and,
when the run wrote a profile file, its rational-surface table).  The
provenance the mapper returns says which native outputs the run produced --
``has_cylindrical``, ``has_profile`` -- without reading the bulk arrays.

Mapping table -- one row per GPEC variable/attribute; the "legacy" column
records how ``GPEC_Research/library/gpec_imas.py`` handled the same quantity
and whether that behavior was kept or revised:

====================== ============ ======== ================================================== ===================
GPEC source            native dims  units    destination                                        status
====================== ============ ======== ================================================== ===================
attr ``n``             scalar       --       ``toroidal_mode[p].n_tor``                         revised-from-legacy
                                             (legacy inferred n from the *filename*)
prof ``psi_n``,        (psi,) (m,)  --       ``plasma.grid.dim1`` / ``.dim2`` on a private      revised-from-legacy
``m_out``                                    ``grid_type.index`` (-1), the convention
                                             ``mhd_linear`` already uses for DCON
prof ``Jbgradpsi``     (i,m,psi)    T        ``plasma.b_field_perturbed.coordinate1.real/imag`` revised-from-legacy
                                             (transposed to (dim1, dim2) = (psi, m))
prof rational-surface  (psi_n_r,)   mixed    ``code.parameters`` ``<rational_surfaces>``:        revised-from-legacy
geometry                                     psi, q, dq/dpsi_N, area, and the factor
                                             -n*Phi_res/Delta the derivation needs
cyl ``R``/``z``,       (R,) (z,)    m, T     none -- native container / JSON sidecar only.      revised-from-legacy
``b_[rzt]``,           (i,z,R)               ``plasma`` carries one grid per mode and it        (was mapped; the
``b_[rzt]_plasma``                           carries the spectral field; the slot's only        slot's only renderer
                                             registered renderer reads harmonics                reads harmonics)
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
profile ``Phi_res``,   (i,psi_n_r)  T, A     not mapped -- *derived* from the spectral field    revised-from-legacy
``I_res``, ``Delta``,                        above by :func:`vaft.process.perturbation`         (no IMAS slot exists:
``w_isl``, ``K_isl``                         ``.resonant_delta`` and the geometry recorded in    mhd_linear has no
                                             ``code.parameters``. They have no IMAS slot --      per-surface numeric
                                             ``ntms.mode[].deltaw`` is m^-1 where GPEC's         field, and ntms is a
                                             ``Delta`` is unitless -- and the values stay in     unit mismatch)
                                             the sidecar. Reproduces GPEC's own numbers to
                                             2.8% on ``Delta`` and 1.4% on ``w_isl``
attr ``shot``/``time`` scalars      -- / s   written by the *caller* (options) -- GPEC records  revised-from-legacy
                                             0/0 when the gfile header has none
====================== ============ ======== ================================================== ===================
"""

from __future__ import annotations

import os
from typing import Any, Optional
from xml.sax.saxutils import escape

import numpy as np
from omas import ODS

from vaft.code.gpec import GpecIdealResult, read_gpec_netcdf

from .mhd_linear import (
    _HAMADA_FOURIER_GRID_INDEX,
    claim_ids,
    _append_code_parameters,
    _set_output_flag,
    ensure_toroidal_mode_grid,
)

__all__ = ["gpec_ideal"]

_GRID_DESCRIPTION = "GPEC cylindrical (R, z) grid; dim1=R [m], dim2=z [m]"

#: Which profile-output array carries the perturbed normal flux. ``Jbgradpsi``
#: is the Jacobian-weighted contravariant psi component on the *output*
#: harmonic basis ``m_out``, which is the one GPEC's own resonant chain is
#: built from (``gpec/gpout.f:1675-1711``); ``b_n`` is the unweighted normal
#: field and does not reproduce that chain.
_SPECTRAL_FIELD = "Jbgradpsi"


def _write_spectral_field(entry: ODS, profile, jacobian: str) -> Optional[dict[str, Any]]:
    """Write ``Jbgradpsi`` onto a declared ``(psi_n, m)`` grid.

    Follows the convention :func:`vaft.machine_mapping.mhd_linear` already
    established for DCON's eigenfunction: a private ``grid_type.index``,
    because the IMAS identifier's Fourier grid types (14/24/34/44) name the
    straight-field-line, equal-arc and polar angles only, and this run solved
    in ``jacobian`` coordinates.
    """
    field = profile.extras.get(_SPECTRAL_FIELD)
    if field is None or profile.psi_n is None or profile.m_out is None:
        return None
    psi_n = np.asarray(profile.psi_n, dtype=float)
    m = np.asarray(profile.m_out, dtype=float)
    values = np.asarray(field)
    if values.shape != (m.size, psi_n.size):
        raise ValueError(
            f"{_SPECTRAL_FIELD} has shape {values.shape}, but the file declares "
            f"{m.size} harmonics on {psi_n.size} radial points"
        )
    # Deliberately NOT strided, unlike the DCON eigenfunction path. That path
    # strides because solutions.bin is larger than an entire packaged sample
    # shot; this array is 3.2 MB per mode on the DIII-D reference (129
    # harmonics x 1561 radial points, real and imaginary), and striding costs
    # accuracy exactly where it matters: GPEC clusters its radial grid around
    # the singular surfaces, which is what the resonant derivation fits
    # against. Measured, a stride of 7 moves the derived Delta from 1.0070 to
    # 1.0153 of GPEC's own.

    plasma = entry["plasma"]
    plasma["grid_type"]["index"] = _HAMADA_FOURIER_GRID_INDEX
    plasma["grid_type"]["name"] = "inverse_psi_hamada_fourier"
    plasma["grid_type"]["description"] = (
        f"Normalized poloidal flux as the radial label (dim1) and Fourier modes "
        f"in the {jacobian} poloidal angle (dim2). Private index because the "
        f"IMAS identifier's Fourier grid types (14/24/34/44) name the "
        f"straight-field-line, equal-arc and polar angles only, and GPEC solved "
        f"this case in {jacobian} coordinates."
    )
    plasma["grid"]["dim1"] = psi_n
    plasma["grid"]["dim2"] = m

    # (m, psi) in the container, (dim1, dim2) = (psi, m) in the IDS.
    transposed = np.ascontiguousarray(values.T)
    expected = (psi_n.size, m.size)
    node = plasma["b_field_perturbed"]["coordinate1"]
    for part, array in (("real", transposed.real), ("imaginary", transposed.imag)):
        # OMAS accepts an array that does not match its declared coordinates,
        # so the grid/array agreement this IDS depends on is only guaranteed
        # if it is checked here.
        if array.shape != expected:
            raise ValueError(
                f"{_SPECTRAL_FIELD}.{part} has shape {array.shape} against a "
                f"declared (dim1, dim2) grid of {expected}"
            )
        node[part] = np.ascontiguousarray(array, dtype=float)
    return {
        "radial_points": int(psi_n.size),
        "harmonics": int(m.size),
        "jacobian": jacobian,
    }


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
    include_spectral: bool,
) -> None:
    claim_ids(ods, "mhd_linear", "GPEC")
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

    spectral = None
    geometry = None
    if include_spectral and "profile" in result.source_paths:
        profile = result.profile
        if profile is not None:
            spectral = _write_spectral_field(entry, profile, control.jacobian or "hamada")
            geometry = _rational_surface_geometry(
                profile, control.n_tor, (control.attrs or {}).get("chi1")
            )

    fragment = (
        f'<solver name="gpec" n_tor="{control.n_tor}">'
        f'<jacobian>{control.jacobian}</jacobian>'
        f'<helicity>{control.helicity}</helicity>'
        f'<energy_vacuum units="J">{control.energy_vacuum!r}</energy_vacuum>'
        f'<energy_surface units="J">{control.energy_surface!r}</energy_surface>'
        f'<energy_plasma units="J">{control.energy_plasma!r}</energy_plasma>'
        '<energy_perturbed derivation="energy_vacuum+energy_surface+energy_plasma"'
        ' units="J" source="gpec_control_output global attributes"/>'
    )
    # Everything below is nested inside this mode's <solver>, not appended
    # beside it: `code.parameters` is one IDS-global string that accumulates a
    # fragment per mode, so siblings would leave a reader pairing N <solver>,
    # N <spectral_field> and N <rational_surfaces> by document order alone.
    if spectral is not None:
        fragment += (
            f'<spectral_field variable="{_SPECTRAL_FIELD}"'
            f' jacobian="{escape(str(spectral["jacobian"]))}"'
            f' radial_points="{spectral["radial_points"]}"'
            f' harmonics="{spectral["harmonics"]}"'
            ' units="T" note="Jacobian-weighted contravariant psi component on the'
            ' output harmonic basis; the full-resolution array and the cylindrical'
            ' (R, z) field stay in the gpec_ideal_native_n&lt;mode&gt;.json sidecar"/>'
        )
    if geometry is not None:
        fragment += geometry
    if include_vacuum:
        fragment += (
            '<vacuum derivation="total_minus_plasma" written="false"'
            ' note="the cylindrical decomposition lives in the sidecar; the IDS'
            ' region carries one grid, and it carries the spectral field"/>'
        )
    fragment += "</solver>"
    _append_code_parameters(ods, "mhd_linear", fragment, code_name="GPEC")
    if control.version:
        ods["mhd_linear.code.version"] = control.version
    _set_output_flag(ods, "mhd_linear", time_slice, 0)


def _rational_surface_geometry(profile, n_tor: int, chi1: Optional[float]) -> Optional[str]:
    """The per-surface geometry the resonant derivation needs, as XML.

    ``psi``, ``q``, ``dq/dpsi_N`` and the surface area are equilibrium
    quantities the derivation reads back;
    :func:`~vaft.process.perturbation.resonant_geometric_factor` is measured
    here too, because it absorbs a vacuum surface inductance that
    ``GPEC/gpec/gpvacuum.f`` builds by calling the VACUUM code, and nothing
    downstream can rebuild it. These are a handful of calibration numbers per
    run -- not the resonant table, which stays in the sidecar.
    """
    from vaft.process.perturbation import resonant_geometric_factor

    needed = (profile.psi_n_rational, profile.q_rational,
              profile.dqdpsi_n_rational, profile.area_rational)
    if any(x is None for x in needed) or profile.Delta is None or profile.Phi_res is None:
        return None
    psi, q, dq, area = (np.asarray(x, dtype=float) for x in needed)
    if not (psi.size == q.size == dq.size == area.size) or psi.size == 0:
        return None
    try:
        factor = resonant_geometric_factor(profile.Delta, profile.Phi_res, n_tor)
    except ValueError:
        # A run whose Delta is zero, or whose pair is not matched, simply has
        # no factor to record -- that is not a reason to drop the geometry.
        factor = np.full(psi.size, np.nan)
    if chi1 is None or not np.isfinite(chi1) or float(chi1) == 0.0:
        # Without it the jump cannot be scaled, and a consumer reading this
        # block would have to go back to the control file for one number.
        return None
    rows = "".join(
        f'<surface psi_n="{float(p)!r}" q="{float(qq)!r}" dq_dpsi_n="{float(d)!r}"'
        f' area="{float(a)!r}" geometric_factor="{float(g)!r}"/>'
        for p, qq, d, a, g in zip(psi, q, dq, area, factor)
    )
    return (
        f'<rational_surfaces chi1="{float(chi1)!r}"'
        ' units="psi_n, -, -, m^2, T"'
        ' note="equilibrium geometry for the resonant derivation; chi1 is  d(chi)/d(psi_N), the flux normalisation the jump is scaled by;'
        ' geometric_factor is -n*Phi_res/Delta, which absorbs the vacuum'
        ' surface inductance and cannot be rebuilt downstream">'
        f"{rows}</rational_surfaces>"
    )


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
    - ``include_spectral`` (default True) writes the perturbed resonant flux
      on its ``(psi_n, m)`` grid, which is what makes the resonant response
      derivable from the IDS. It opens the profile file -- 144 MB on the
      DIII-D example -- so a caller that wants only the control-level
      mapping turns it off.
    - ``include_vacuum`` is accepted and ignored. It selected the cylindrical
      vacuum field, and the cylindrical decomposition is no longer written:
      ``plasma`` carries one grid per mode, and it carries the spectral
      field. The full cylindrical arrays remain in the JSON sidecar.

    Returns ``{n_tor: {...}}`` with values kept alongside the ODS in the
    caller's manifest.  As a side effect, the lossless native container is
    written to ``<source>/gpec_ideal_native_n<mode>.json``.
    """
    options = dict(options or {})
    time_slice = int(options.get("time_slice", 0))
    include_vacuum = bool(options.get("include_vacuum", True))
    include_spectral = bool(options.get("include_spectral", True))

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
    _write_mode_entry(ods, time_slice, position, result,
                      include_vacuum=include_vacuum, include_spectral=include_spectral)

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
            # From the recorded path, not from ``result.profile``: that is a
            # lazily-read cached_property, and opening the profile file costs
            # 144 MB on the DIII-D example.  A provenance record should not
            # pay that to answer "was there one?".
            "has_profile": "profile" in result.source_paths,
        }
    }
