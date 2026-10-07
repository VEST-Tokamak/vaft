"""`mhd_linear`/`ntms` IDS mapping helpers for the GPEC solver suite.

Architecture (issue #170): this module is the *IDS-populating* layer only.
It never re-parses DCON/RDCON/STRIDE output files itself -- it reads the
solver-native output containers owned by :mod:`vaft.code.gpec`
(`DconOutput` for DCON, `Pest3MatchingOutput` shared by RDCON/STRIDE), and
copies only the values that have a scientifically correct home in IMAS into
the ODS:

- `n_tor`, `energy_perturbed` (with an explicit normalization caveat --
  DCON's total energy eigenvalue is a dimensionless, normalized quantity
  stored in a field the IMAS schema documents as Joules), and a run-success
  `code.output_flag` land in `mhd_linear`.
- Delta-prime has no field anywhere in `mhd_linear`, but the classical
  (single-surface, diagonal) value *is* a legitimate `ntms.deltaw`
  contribution to the Rutherford equation, so RDCON/STRIDE runs populate
  `ntms` for that value; `ballooning_type="Tearing"` stays a separate,
  correct mode-type tag in `mhd_linear` and is never treated as if it also
  carried the Delta-prime value.
- The Fourier-space eigenfunction from `solutions.bin` reaches
  `plasma.displacement_perpendicular` (and, as the field DCON itself derives
  from it, `plasma.b_field_perturbed.coordinate1`) on an explicitly declared
  `(psi, m)` grid, with `m_pol_dominant` alongside. These are *closest-fit*
  mappings, not exact ones, and each carries a structured caveat in
  `code.parameters` in the same shape as `energy_perturbed`'s: `xi.grad(psi)`
  is a contravariant flux component rather than a perpendicular displacement
  in metres, and its amplitude is an arbitrary eigenvector normalization.
  `grid_type` uses the private (negative) index the IMAS identifier reserves
  for exactly this case -- the schema does enumerate Fourier-in-poloidal-angle
  grids (14/24/34/44), but only for the straight-field-line, equal-arc and
  polar angles, and DCON runs in Hamada coordinates. What the IDS carries is a
  radially strided view; the full-resolution arrays stay in the
  `vaft.code.gpec` container, persisted as JSON next to the solver output.
  `m_pol_dominant` alone needs no caveat: it is dimensionless and invariant
  under the normalization.
- Everything else that has no IMAS home (mode-range provenance, the full
  PEST3 matching matrices, `solutions.bin`'s uninterpreted fourth
  Euler-Lagrange component) goes into `mhd_linear.code.parameters` when it is
  solver-configuration metadata, or stays exclusively in the `vaft.code.gpec`
  output container. A closest-fit mapping is made only where the
  correspondence is meaningful and the mismatch can be stated precisely;
  nothing is written whose meaning cannot be recorded.

Schema reference: https://gafusion.github.io/omas/schema.html
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional, Sequence
import warnings
from xml.sax.saxutils import escape

import numpy as np
from omas import ODS

from vaft.code.gpec import DconOutput, Pest3MatchingOutput, read_dcon_output, read_pest3_matching_output
from vaft.machine_mapping.utils import path_exists
from vaft.ods_access import path_count

#: Version of the ``<solver>`` fragment RDCON/STRIDE write into ``ntms.code.parameters``.
#: 2 added ``time_slice``/``mode_start``/``mode_count`` and one ``<surface>`` per
#: ``ntms.mode[]`` entry (#143 solver attribution, #939 local criteria).
NTMS_FRAGMENT_VERSION = 2

#: Version of the ``<solver name="dcon">`` fragment in ``mhd_linear.code.parameters``.
#: 2 added ``time_slice``/``position`` and the native stability payload (#940):
#: least-stable and full W_p/W_v/W_t spectra, requested vs effective edge
#: treatment with DCON's edge scan, and the D_I/D_R/C_A profiles.
DCON_FRAGMENT_VERSION = 2

#: Version of the RDCON/STRIDE ``<solver>`` fragment in ``mhd_linear.code.parameters``.
#: 2 added ``time_slice``/``position``, the solver's radial local-stability
#: profiles (``psi_n, q, di, dr, h, ca1``) and the full complex Delta-prime
#: matrix (#939), so the ODS alone carries what RDCON analysis reads.
RESISTIVE_FRAGMENT_VERSION = 2

__all__ = [
    "DCON_FRAGMENT_VERSION",
    "MAX_RADIAL_POINTS",
    "NTMS_FRAGMENT_VERSION",
    "RESISTIVE_FRAGMENT_VERSION",
    "claim_ids",
    "ensure_toroidal_mode_grid",
    "extract_dcon_stability",
    "extract_rdcon_stability",
    "initialize_output_flags",
    "mhd_linear",
    "ntms_solver_surfaces",
]

_MODULE_PATTERNS = {
    "dcon": re.compile(r"dcon_output_n(\d+)\.nc"),
    "rdcon": re.compile(r"rdcon_output_n(\d+)\.nc"),
    "stride": re.compile(r"stride_output_n(\d+)\.nc"),
}


def _ensure_time_slice(ods: ODS, ids: str, time_slice: int) -> None:
    """Grow ``<ids>.time_slice`` so index ``time_slice`` is addressable.

    An OMAS array of structures only auto-vivifies at its current length --
    indexing past the end raises ``IndexError`` -- so a solver that succeeds
    for, say, time slice 2 while slices 0 and 1 produced nothing would
    otherwise blow up here and be misreported by the caller as a *failed*
    solver run rather than a successful one. Growing the AOS in order leaves
    the intervening slices as legitimately empty entries.
    """
    for index in range(_time_slice_count(ods, ids), time_slice + 1):
        ods[ids]["time_slice"][index]


def _time_slice_count(ods: ODS, ids: str) -> int:
    """Number of ``<ids>.time_slice`` entries, 0 when the node is absent."""
    return path_count(ods, f"{ids}.time_slice")


def _existing_aos_count(ods: ODS, ids: str, time_slice: int, aos_name: str) -> int:
    """Length of ``<ids>.time_slice.<time_slice>.<aos_name>``, 0 if not present."""
    if time_slice >= _time_slice_count(ods, ids):
        return 0
    return path_count(ods, f"{ids}.time_slice.{time_slice}.{aos_name}")


def _append_code_parameters(ods: ODS, ids: str, fragment_xml: str, *, code_name: str) -> None:
    """Append one ``<solver>`` fragment to ``<ids>.code.parameters``.

    ``code.parameters`` is a single IDS-global string, but `mhd_linear`/`ntms`
    accumulate entries from multiple solver calls (DCON, then RDCON, then
    STRIDE) on the same ``ods`` -- so this appends rather than overwrites,
    keeping every call's provenance rather than only the last one's.
    """
    ods[f"{ids}.code.name"] = code_name
    ods[f"{ids}.code.repository"] = "https://github.com/PrincetonUniversity/GPEC"
    path = f"{ids}.code.parameters"
    existing = ods.get(path, None)
    if not existing:
        ods[path] = f"<parameters>{fragment_xml}</parameters>"
        return
    existing = existing.rstrip()
    if existing.endswith("</parameters>"):
        ods[path] = existing[: -len("</parameters>")] + fragment_xml + "</parameters>"
    else:
        ods[path] = existing + fragment_xml


#: `code.output_flag` value for a time slice this solver did not successfully
#: produce. IMAS documents a negative flag as "the result shall not be used",
#: which is exactly right for a slice we are only padding past to reach a
#: later one -- and it is overwritten with 0 if that slice later succeeds.
_OUTPUT_FLAG_NOT_RUN = -1


def _xml_attrs(**values: Any) -> str:
    """`` key="value"`` pairs, skipping ``None``; floats keep full precision."""
    parts = []
    for key, value in values.items():
        if value is None:
            continue
        text = repr(float(value)) if isinstance(value, (float, np.floating)) else str(value)
        parts.append(f' {key}="{escape(text, {chr(34): "&quot;"})}"')
    return "".join(parts)


def ntms_solver_surfaces(ods: ODS) -> list[dict[str, Any]]:
    """Which solver produced each ``ntms.time_slice[t].mode[i]``, with its native extras.

    Reads the version-2 fragments :func:`mhd_linear` writes into
    ``ntms.code.parameters``. Returns one dict per surface: ``solver``,
    ``n_tor``, ``time_slice``, ``mode`` (the ``ntms.mode[]`` index), ``m``,
    ``psi_n``, ``q``, ``delta_prime_imag`` and, where the solver wrote them,
    ``di``/``dr``/``h``/``ca1``. Version-1 fragments carry no index range and
    yield nothing; their surfaces cannot be attributed. A fragment whose
    surfaces do not cover exactly its declared ``mode_start``/``mode_count``
    range, or that is otherwise malformed, is skipped with a warning. The ODS
    is not modified.
    """
    return [row for _, rows in _ntms_solver_fragments(ods) for row in rows]


def _ntms_solver_fragments(ods: ODS) -> list[tuple[tuple[int, str, int], list[dict[str, Any]]]]:
    """The version-2 ntms fragments in document order, each as ``((slice, solver, n), rows)``."""
    import xml.etree.ElementTree as ET

    if not path_exists(ods, "ntms.code.parameters"):
        return []
    text = ods["ntms.code.parameters"]
    try:
        root = ET.fromstring(text.decode() if isinstance(text, bytes) else str(text))
    except ET.ParseError as exc:
        warnings.warn(f"ntms.code.parameters is not XML ({exc}); no solver attribution", RuntimeWarning, stacklevel=2)
        return []
    fragments: list[tuple[tuple[int, str, int], list[dict[str, Any]]]] = []
    for solver in root.iter("solver"):
        if int(solver.get("version", "1")) < 2:
            continue
        try:
            n_tor, time_slice = int(solver.get("n_tor")), int(solver.get("time_slice"))
            start, count = int(solver.get("mode_start")), int(solver.get("mode_count"))
            entries = [
                {key: int(value) if key in ("mode", "m") else float(value) for key, value in surface.attrib.items()}
                for surface in solver.iter("surface")
            ]
        except (TypeError, ValueError) as exc:
            warnings.warn(f"skipping malformed ntms solver fragment: {exc}", RuntimeWarning, stacklevel=2)
            continue
        if [entry.get("mode") for entry in entries] != list(range(start, start + count)):
            warnings.warn(
                f"skipping ntms fragment for {solver.get('name')} n={n_tor}: its surfaces do not cover "
                f"mode[{start}:{start + count}]",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        name = solver.get("name")
        rows = [{"solver": name, "n_tor": n_tor, "time_slice": time_slice, **entry} for entry in entries]
        fragments.append(((time_slice, name, n_tor), rows))
    return fragments


def ensure_toroidal_mode_grid(ods: ODS, time_slice: int, n_tor_grid: Sequence[int]) -> None:
    """Lay out ``mhd_linear.time_slice[t].toroidal_mode`` as a dense ``n_tor`` grid.

    The analysis model this IDS is written for is a regular ``(time, n_tor)``
    grid: a consumer must be able to read every toroidal mode at a fixed time,
    and a time trace at a fixed ``n_tor``, without reconstructing sparse
    indices or joining on labels. So every *requested* mode gets an entry at
    the same array position in every time slice, whether or not a solver
    produced anything for it.

    Position is layout, never physics: ``n_tor`` is written explicitly on
    every entry (padded ones included) and remains the only thing a consumer
    may read the mode number from. A padded entry carries ``n_tor`` and
    nothing else -- no zeroed or otherwise fabricated payload that could be
    mistaken for a solver result. Whether a cell holds a real result is read
    from the payload's presence, from ``code.output_flag`` for the slice, or
    -- authoritatively, and per ``(time, module, n)`` -- from the stage
    manifest.
    """
    _ensure_time_slice(ods, "mhd_linear", time_slice)
    for position, n_tor in enumerate(n_tor_grid):
        ods["mhd_linear"]["time_slice"][time_slice]["toroidal_mode"][position]["n_tor"] = int(n_tor)


def initialize_output_flags(ods: ODS, ids: str, count: int) -> None:
    """Extend ``<ids>.code.output_flag`` to ``count`` slices, defaulting to "not run".

    Gives the flag array the same dense length as the time base, so a slice no
    solver reached reads as an explicit negative flag rather than as a missing
    array element. Purely additive: flags a solver has already set are left
    alone, so this is safe to call either side of the run loop.
    """
    if count <= 0:
        return
    path = f"{ids}.code.output_flag"
    existing = _output_flags(ods, ids)
    if len(existing) >= count:
        return
    ods[path] = np.asarray(existing + [_OUTPUT_FLAG_NOT_RUN] * (count - len(existing)), dtype=int)


def _output_flags(ods: ODS, ids: str) -> list[int]:
    path = f"{ids}.code.output_flag"
    if path not in ods:
        return []
    values = np.atleast_1d(ods[path])
    return [int(value) for value in values] if values.size else []


def _set_output_flag(ods: ODS, ids: str, time_slice: int, flag: int) -> None:
    """Set ``<ids>.code.output_flag[time_slice]``, padding earlier slices.

    ``output_flag`` is an ``INT_1D`` over ``<ids>.time``, so writing index 2
    of an empty array is an error rather than an append. Pad the gap
    explicitly instead of letting a solver that succeeded only for a later
    time slice fail here.
    """
    values = _output_flags(ods, ids)
    while len(values) <= time_slice:
        values.append(_OUTPUT_FLAG_NOT_RUN)
    values[time_slice] = flag
    ods[f"{ids}.code.output_flag"] = np.asarray(values, dtype=int)


#: Radial samples kept when the Fourier-space eigenfunction is written into the
#: IDS.  DCON integrates on thousands of steps, and at a realistic ``mpert`` the
#: full-resolution arrays make the stage product roughly an order of magnitude
#: larger than an entire packaged sample shot.  The IDS therefore carries a
#: strided *view* -- exact values, every harmonic, fewer radial samples -- while
#: the ``dcon_native_n<mode>.json`` sidecar keeps the full-resolution data.
#: Striding rather than interpolating so that every number in the IDS is one
#: DCON actually computed.
MAX_RADIAL_POINTS = 256

#: `grid_type.index` for DCON's Fourier-space output.  The IMAS identifier does
#: enumerate Fourier-in-poloidal-angle grids (14/24/34/44), but only for the
#: straight-field-line, equal-arc and polar angles, and DCON runs in Hamada
#: coordinates (`vaft/data/gpec/equil.in`'s `jac_type`).  The schema's own
#: escape hatch for exactly this case is a negative index: "Private identifier
#: values must be indicated by a negative index."
_HAMADA_FOURIER_GRID_INDEX = -1


#: ``code.name`` each writer stamps on the IDS it owns. `mhd_linear`'s
#: ``plasma`` region carries one grid per mode, and the GPEC-suite solvers and
#: ideal GPEC fill it with different quantities on different grids -- DCON's
#: eigenfunction against ``(psi, m)`` in Hamada coordinates, ideal GPEC's
#: Jacobian-weighted resonant flux against its own ``(psi, m_out)``. Writing
#: both into one ODS leaves the second writer's grid describing the first
#: writer's array, so each owns an IDS outright.
_IDS_OWNERS = {"GPEC-suite", "GPEC"}


def claim_ids(ods: ODS, ids: str, owner: str) -> None:
    """Refuse to write ``ids`` when a different mapper already owns it.

    ``build_mhd_linear_ods`` and ``build_gpec_ideal_ods`` already build
    separate ODSs, so this is a guard on the invariant rather than a change
    of behaviour: it turns a silent, structurally invalid merge into an
    error at the point of the second write.
    """
    existing = ods.get(f"{ids}.code.name", None)
    if existing and existing != owner and existing in _IDS_OWNERS:
        raise ValueError(
            f"{ids} was written by {existing!r} and this is {owner!r}. Their "
            f"plasma regions declare different grids for the same field, so "
            f"one ODS cannot hold both -- build them separately and keep the "
            f"two products apart."
        )


def _shared_radial_grid(result: DconOutput) -> Optional[np.ndarray]:
    """The one ``psi`` grid every harmonic block shares, or ``None`` if they differ.

    ``solutions.bin`` writes ``psi`` per record, so each harmonic block carries
    its own copy, and :func:`read_solutions_bin` pads short blocks with NaN.  A
    single ``grid.dim1`` is only meaningful if those copies agree; when they do
    not, the IDS has no honest radial axis to offer and the subtree is skipped
    rather than written against whichever block happened to be first.
    """
    eigenfunction = result.eigenfunction
    if eigenfunction is None or eigenfunction.psi.size == 0:
        return None
    psi = np.asarray(eigenfunction.psi, dtype=float)
    finite = np.isfinite(psi)
    if not finite.any() or not (finite == finite[0]).all():
        return None
    grid = psi[0][finite[0]]
    if not np.allclose(psi[:, finite[0]], grid, equal_nan=False):
        return None
    return grid


def _write_eigenfunction(
    ods: ODS, time_slice: int, position: int, result: DconOutput
) -> Optional[dict[str, Any]]:
    """Write the Fourier-space eigenfunction onto a declared ``(psi, m)`` grid.

    Returns the provenance the caller records in ``code.parameters``, or ``None``
    when there is nothing to write.  Both quantities keep DCON's arbitrary
    eigenvector normalization, which is why every write here is accompanied by a
    structured caveat rather than left to be read as the IMAS field's documented
    units.
    """
    eigenfunction = result.eigenfunction
    if eigenfunction is None or eigenfunction.m.size == 0:
        return None
    grid = _shared_radial_grid(result)
    if grid is None:
        warnings.warn(
            f"dcon n={result.n_tor}: solutions.bin's harmonic blocks do not share one "
            "psi grid, so the eigenfunction has no single radial axis to write against; "
            "it stays in the native sidecar only",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    stride = max(1, int(np.ceil(grid.size / MAX_RADIAL_POINTS)))
    columns = np.flatnonzero(np.isfinite(np.asarray(eigenfunction.psi, dtype=float)[0]))[::stride]
    psi_n = grid[::stride]
    m = np.asarray(eigenfunction.m, dtype=float)

    # IMAS stores these as FLT_2D on [grid.dim1, grid.dim2] -- (psi, m) -- while
    # the container is (harmonic, step), so every array is transposed on the way in.
    xi = eigenfunction.xi_psi_real[:, columns].T, eigenfunction.xi_psi_imag[:, columns].T
    b_normal = eigenfunction.b_normal(result.n_tor)[:, columns].T

    plasma = ods["mhd_linear"]["time_slice"][time_slice]["toroidal_mode"][position]["plasma"]
    plasma["grid_type"]["index"] = _HAMADA_FOURIER_GRID_INDEX
    plasma["grid_type"]["name"] = "inverse_psi_hamada_fourier"
    jacobian = (result.coordinates.jacobian if getattr(result, "coordinates", None) else "") or "hamada"
    plasma["grid_type"]["description"] = (
        f"Normalized poloidal flux as the radial label (dim1) and Fourier modes in the "
        f"{jacobian} poloidal angle (dim2). Private index because the IMAS identifier's "
        f"Fourier grid types (14/24/34/44) name the straight-field-line, equal-arc and "
        f"polar angles only, and DCON solved this case in {jacobian} coordinates."
    )
    plasma["grid"]["dim1"] = psi_n
    plasma["grid"]["dim2"] = m

    expected = (psi_n.size, m.size)
    for path, values in (
        ("displacement_perpendicular", xi),
        ("b_field_perturbed.coordinate1", (np.real(b_normal), np.imag(b_normal))),
    ):
        for part, array in zip(("real", "imaginary"), values):
            # OMAS accepts an array that does not match its declared coordinates,
            # so the grid/array agreement this IDS depends on is only guaranteed
            # if it is checked here.
            if array.shape != expected:
                raise ValueError(
                    f"eigenfunction array for {path}.{part} has shape {array.shape}, "
                    f"but the declared (dim1, dim2) grid is {expected}"
                )
            node = plasma
            for key in path.split("."):
                node = node[key]
            node[part] = np.ascontiguousarray(array, dtype=float)

    return {"radial_stride": stride, "radial_points": int(psi_n.size), "harmonics": int(m.size)}


def _write_dcon_entry(ods: ODS, time_slice: int, position: int, result: DconOutput) -> None:
    claim_ids(ods, "mhd_linear", "GPEC-suite")
    # `position` is this mode's slot in the dense n_tor grid, not an append
    # cursor; `n_tor` is (re)written here so the entry is self-describing even
    # if the grid was laid out by a different caller.
    mode_entry = ods["mhd_linear"]["time_slice"][time_slice]["toroidal_mode"][position]
    mode_entry["n_tor"] = result.n_tor
    if result.W_t_eigenvalue is not None and result.W_t_eigenvalue.size:
        # Least-stable total-energy eigenvalue -- normalized/dimensionless,
        # stored despite the field's Joules documentation because it is the
        # closest existing slot; the unit mismatch is recorded explicitly in
        # code.parameters below rather than left for a future reader to guess.
        mode_entry["energy_perturbed"] = float(result.W_t_eigenvalue[0].real)

    # The dominant poloidal harmonic is the one eigenfunction quantity that needs
    # no caveat: it is an exact fit for `m_pol_dominant`, dimensionless, and
    # invariant under the eigenvector's arbitrary normalization (see
    # `DconOutput.m_pol_dominant`). The DD stores it as FLT_0D even though m is
    # integral.
    m_pol_dominant = result.m_pol_dominant
    if m_pol_dominant is not None:
        mode_entry["m_pol_dominant"] = float(m_pol_dominant)

    eigenfunction_provenance = _write_eigenfunction(ods, time_slice, position, result)

    # The units element is structured, not prose: a consumer checking whether
    # `energy_perturbed` is really in the Joules its IMAS documentation
    # promises can test `units != "J"` (or read `normalization`) instead of
    # having to parse an English sentence.  The eigenfunction elements say the
    # same thing about the same kind of mismatch: both arrays are written to the
    # closest appropriate IMAS field, and neither is in that field's documented
    # units, so the discrepancy is recorded where a consumer can test it rather
    # than left to be inferred from the field name.
    eigenfunction_xml = ""
    if eigenfunction_provenance is not None:
        eigenfunction_xml = (
            '<displacement_perpendicular units="1"'
            ' normalization="dcon_eigenvector_arbitrary" imas_documented_units="m"'
            ' quantity="xi.grad(psi)"'
            ' note="contravariant flux component of the displacement, not the'
            ' perpendicular displacement in metres"'
            ' source_file="solutions.bin" source="match/ideal.f:378-389"/>'
            '<b_field_perturbed_coordinate1 units="1"'
            ' normalization="dcon_eigenvector_arbitrary" imas_documented_units="T"'
            ' derived_by="vaft" definition="i*(m - n*q)*xi.grad(psi)"'
            ' source="match/ideal.f:372"/>'
            f'<eigenfunction_grid index="{_HAMADA_FOURIER_GRID_INDEX}"'
            f' radial_stride="{eigenfunction_provenance["radial_stride"]}"'
            f' radial_points="{eigenfunction_provenance["radial_points"]}"'
            f' harmonics="{eigenfunction_provenance["harmonics"]}"'
            ' note="radially strided view of solutions.bin; the full-resolution'
            ' arrays stay in the dcon_native_n&lt;mode&gt;.json sidecar"/>'
        )
    if m_pol_dominant is not None:
        eigenfunction_xml += (
            '<m_pol_dominant source_file="solutions.bin"'
            ' definition="argmax_m of max_psi |xi.grad(psi)|"/>'
        )

    fragment = (
        f'<solver name="dcon" n_tor="{result.n_tor}" version="{DCON_FRAGMENT_VERSION}"'
        f' time_slice="{time_slice}" position="{position}">'
        f"<mlow>{result.mlow}</mlow><mhigh>{result.mhigh}</mhigh>"
        f"<mpert>{result.mpert}</mpert><mband>{result.mband}</mband>"
        '<energy_perturbed units="1" normalization="dcon_normalized"'
        ' imas_documented_units="J" source_variable="W_t_eigenvalue"/>'
        f"{eigenfunction_xml}"
        f"{_dcon_native_xml(result)}"
        "</solver>"
    )
    _append_code_parameters(ods, "mhd_linear", fragment, code_name="DCON")
    _set_output_flag(ods, "mhd_linear", time_slice, 0)


def _xml_numbers(values: Any) -> str:
    """Space-separated full-precision numbers for an XML attribute; NaN stays ``nan``."""
    return " ".join(repr(float(v)) for v in np.asarray(values, dtype=float).ravel())


def _complex_attrs(prefix: str, value: Optional[complex]) -> dict[str, Optional[float]]:
    if value is None:
        return {}
    return {f"{prefix}_re": float(np.real(value)), f"{prefix}_im": float(np.imag(value))}


def _dcon_native_xml(result: DconOutput) -> str:
    """DCON's stability payload with no IMAS slot (#940), as XML elements.

    Everything is a value, not a pointer, and at the reader's full
    resolution: the 1-D profiles are small (mpsi + 1 points), about 16 kB per
    (time slice, n) at mpsi 128. The least-stable entries are selected by
    mode label, as :attr:`DconOutput.total1` is; the spectra carry the labels
    so that can be re-checked. ``energy_perturbed`` keeps its historical
    ``W_t_eigenvalue[0]``, which is the same entry whenever the labels are in
    order (they are in every DCON output seen so far). Evaluation flags are
    written as ``1``/``0`` and omitted when unknown; they describe the run's
    namelist, so they are written whether or not a profile block exists, and
    ``requested_psiedge`` carries DCON's default when the namelist omits the
    key, exactly as the validation layer reads it
    (:attr:`~vaft.code.gpec.DconEvaluation.requested_psiedge`).
    """
    parts = [
        "<least_stable"
        + _xml_attrs(
            selected_by="mode label 1",
            **_complex_attrs("W_t", result.total1),
            **_complex_attrs("W_p", result.plasma1),
            **_complex_attrs("W_v", result.vacuum1),
        )
        + "/>"
    ]
    labels = None if result.mode is None else " ".join(str(int(m)) for m in np.asarray(result.mode).ravel())
    for name in ("W_t", "W_p", "W_v"):
        spectrum = getattr(result, f"{name}_eigenvalue")
        if spectrum is not None and np.asarray(spectrum).size:
            spectrum = np.asarray(spectrum).ravel()
            label_attr = f' mode="{labels}"' if labels is not None and len(labels.split()) == spectrum.size else ""
            parts.append(
                f'<spectrum quantity="{name}"{label_attr} real="{_xml_numbers(spectrum.real)}"'
                f' imag="{_xml_numbers(spectrum.imag)}"/>'
            )
    evaluation = result.evaluation
    parts.append(
        "<edge"
        + _xml_attrs(
            treatment=result.edge_treatment,
            requested_psiedge=None if evaluation is None else evaluation.requested_psiedge,
            psilim=result.psilim,
            qlim=result.qlim,
        )
        + "/>"
    )
    scan = result.edge_scan
    if scan is not None:
        dw = np.asarray(scan.dW)
        parts.append(
            f'<edge_scan psi_n="{_xml_numbers(scan.psi_n)}" q="{_xml_numbers(scan.q)}"'
            f' dW_re="{_xml_numbers(dw.real)}" dW_im="{_xml_numbers(dw.imag)}"/>'
        )
    attrs = []
    if result.psi_n is not None:
        attrs.append(f'psi_n="{_xml_numbers(result.psi_n)}"')
        for name in ("di", "dr", "ca1"):
            values = getattr(result, name)
            if values is not None:
                attrs.append(f'{name}="{_xml_numbers(values)}"')
        if result.ca1_evaluated is not None:
            attrs.append(f'ca1_evaluated="{" ".join("1" if v else "0" for v in np.asarray(result.ca1_evaluated, dtype=bool))}"')
    if evaluation is not None:
        for name in ("mer_flag", "bal_flag"):
            value = getattr(evaluation, name)
            if value is not None:
                attrs.append(f'{name}="{1 if value else 0}"')
    if attrs:
        parts.append("<local_criteria " + " ".join(attrs) + "/>")
    return "".join(parts)


def _numbers(text: Optional[str]) -> Optional[np.ndarray]:
    return None if text is None else np.array([float(v) for v in text.split()], dtype=float)


def _complex_numbers(real: Optional[str], imag: Optional[str]) -> Optional[np.ndarray]:
    re_part, im_part = _numbers(real), _numbers(imag)
    if re_part is None:
        return None
    if im_part is None:
        im_part = np.zeros_like(re_part)
    if re_part.shape != im_part.shape:
        raise ValueError(f"real/imag lengths differ ({re_part.size} vs {im_part.size})")
    return re_part + 1j * im_part


def _flag(text: Optional[str]) -> Optional[bool]:
    """``1``/``0`` as written (legacy ``True``/``False`` too); absent or other text is unknown."""
    return {"1": True, "0": False, "True": True, "False": False}.get(text) if text is not None else None


def extract_dcon_stability(ods: ODS) -> list[dict[str, Any]]:
    """DCON's stability results from ``mhd_linear``, one dict per (time slice, n_tor).

    Reads the version-2 ``<solver name="dcon">`` fragments that :func:`mhd_linear`
    writes into ``mhd_linear.code.parameters`` (#940), returning:

    * ``W_t``, ``W_p``, ``W_v``: least-stable eigenvalues (complex; normalized,
      not Joules) and ``W_t_spectrum`` etc. (complex arrays);
    * ``edge_treatment``, ``requested_psiedge``, ``psilim``, ``qlim`` and, for a
      truncated run, ``edge_scan`` (``psi_n``, ``q``, complex ``dW``);
    * ``psi_n``, ``D_I``, ``D_R``, ``C_A`` (NaN where not evaluated),
      ``mercier_evaluated`` and ``ballooning_evaluated`` (the run's ``mer_flag``
      / ``bal_flag``, None when the namelist is unknown) and
      ``ballooning_points_evaluated``, the number of surfaces where DCON
      integrated the ballooning equation (0 when ``bal_flag`` was on but no
      surface qualified; None without the mask);
    * summaries: ``max_D_I`` / ``psi_n_at_max_D_I``, ``max_D_R`` /
      ``psi_n_at_max_D_R``, ``min_C_A`` / ``psi_n_at_min_C_A`` over evaluated
      points only.

    Values only, no verdicts: the sign of W_t, D_I > 0, D_R > 0 and C_A < 0 are
    the criteria a consumer applies. Every row has the same keys (None where
    the run wrote nothing). If one (time slice, n) was mapped more than once,
    the last fragment wins. Version-1 fragments carry none of this and yield
    nothing. Malformed fragments are skipped with a warning. The ODS is not
    modified.
    """
    import xml.etree.ElementTree as ET

    if not path_exists(ods, "mhd_linear.code.parameters"):
        return []
    text = ods["mhd_linear.code.parameters"]
    try:
        root = ET.fromstring(text.decode() if isinstance(text, bytes) else str(text))
    except ET.ParseError as exc:
        warnings.warn(f"mhd_linear.code.parameters is not XML ({exc}); no DCON payload", RuntimeWarning, stacklevel=2)
        return []
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    for solver in root.iter("solver"):
        if solver.get("name") != "dcon":
            continue
        try:
            if int(solver.get("version", "1")) < 2:
                continue
            row = _parse_dcon_fragment(solver)
        except (TypeError, ValueError) as exc:
            warnings.warn(f"skipping malformed DCON fragment: {exc}", RuntimeWarning, stacklevel=2)
            continue
        latest.pop((row["time_slice"], row["n_tor"]), None)  # re-mapping: the last fragment wins
        latest[(row["time_slice"], row["n_tor"])] = row
    return list(latest.values())


def _parse_dcon_fragment(solver: Any) -> dict[str, Any]:
    def complex_of(element: Any, prefix: str) -> Optional[complex]:
        if element is None or element.get(f"{prefix}_re") is None:
            return None
        return complex(float(element.get(f"{prefix}_re")), float(element.get(f"{prefix}_im", 0.0)))

    row: dict[str, Any] = {
        "n_tor": int(solver.get("n_tor")),
        "time_slice": int(solver.get("time_slice")),
        "position": int(solver.get("position")),
    }
    least = solver.find("least_stable")
    for name in ("W_t", "W_p", "W_v"):
        row[name] = complex_of(least, name)
        row[f"{name}_spectrum"] = None
        row[f"{name}_spectrum_mode"] = None
    for spectrum in solver.findall("spectrum"):
        name = spectrum.get("quantity")
        row[f"{name}_spectrum"] = _complex_numbers(spectrum.get("real"), spectrum.get("imag"))
        mode = spectrum.get("mode")
        row[f"{name}_spectrum_mode"] = None if mode is None else np.array([int(v) for v in mode.split()])
    edge = solver.find("edge")
    row["edge_treatment"] = None if edge is None else edge.get("treatment")
    for key in ("requested_psiedge", "psilim", "qlim"):
        row[key] = None if edge is None or edge.get(key) is None else float(edge.get(key))
    scan = solver.find("edge_scan")
    row["edge_scan"] = (
        None
        if scan is None
        else {
            "psi_n": _numbers(scan.get("psi_n")),
            "q": _numbers(scan.get("q")),
            "dW": _complex_numbers(scan.get("dW_re"), scan.get("dW_im")),
        }
    )
    criteria = solver.find("local_criteria")
    psi = None if criteria is None else _numbers(criteria.get("psi_n"))
    row["psi_n"] = psi
    names = {"di": "D_I", "dr": "D_R", "ca1": "C_A"}
    for native, public in names.items():
        row[public] = None if criteria is None else _numbers(criteria.get(native))
    evaluated = None if criteria is None or criteria.get("ca1_evaluated") is None else (
        np.array([v == "1" for v in criteria.get("ca1_evaluated").split()], dtype=bool)
    )
    mer = None if criteria is None else _flag(criteria.get("mer_flag"))
    bal = None if criteria is None else _flag(criteria.get("bal_flag"))
    row["mercier_evaluated"] = mer
    # The flag is what the namelist asked for, as the validation layer reads it;
    # how many surfaces the scan then reached is a separate number.
    row["ballooning_evaluated"] = bal
    row["ballooning_points_evaluated"] = None if evaluated is None else int(evaluated.sum())

    def extremum(values: Optional[np.ndarray], mask: Optional[np.ndarray], largest: bool):
        if values is None or psi is None or values.shape != psi.shape:
            return None, None
        keep = np.isfinite(values) if mask is None or mask.shape != values.shape else np.isfinite(values) & mask
        if not keep.any():
            return None, None
        index = np.flatnonzero(keep)[int(np.argmax(values[keep]) if largest else np.argmin(values[keep]))]
        return float(values[index]), float(psi[index])

    mercier = row["mercier_evaluated"]
    row["max_D_I"], row["psi_n_at_max_D_I"] = extremum(row["D_I"], None, True) if mercier else (None, None)
    row["max_D_R"], row["psi_n_at_max_D_R"] = extremum(row["D_R"], None, True) if mercier else (None, None)
    ballooning = row["ballooning_evaluated"]
    row["min_C_A"], row["psi_n_at_min_C_A"] = extremum(row["C_A"], evaluated, False) if ballooning else (None, None)
    return row


def _write_resistive_entry(
    ods: ODS, time_slice: int, position: int, result: Pest3MatchingOutput, diagonal: list[dict[str, Any]]
) -> None:
    # `position` is this mode's slot in the dense n_tor grid, not an append
    # cursor; `n_tor` is (re)written here so the entry is self-describing even
    # if the grid was laid out by a different caller.
    claim_ids(ods, "mhd_linear", "GPEC-suite")
    mode_entry = ods["mhd_linear"]["time_slice"][time_slice]["toroidal_mode"][position]
    mode_entry["n_tor"] = result.n_tor
    mode_entry["ballooning_type"]["name"] = "Tearing"

    fragment = (
        f'<solver name="{escape(result.solver)}" n_tor="{result.n_tor}" version="{RESISTIVE_FRAGMENT_VERSION}"'
        f' time_slice="{time_slice}" position="{position}">'
        f"<mlow>{result.mlow}</mlow><mhigh>{result.mhigh}</mhigh>"
        f"<mpert>{result.mpert}</mpert><mband>{result.mband}</mband>"
        f"<msing>{result.msing}</msing>"
        f"{_resistive_native_xml(result)}"
        "</solver>"
    )
    _append_code_parameters(ods, "mhd_linear", fragment, code_name="GPEC-suite")
    _set_output_flag(ods, "mhd_linear", time_slice, 0)

    if not diagonal:
        return
    _ensure_time_slice(ods, "ntms", time_slice)
    start = _existing_aos_count(ods, "ntms", time_slice, "mode")
    for offset, surface in enumerate(diagonal):
        entry = ods["ntms"]["time_slice"][time_slice]["mode"][start + offset]
        entry["n_tor"] = surface["n"]
        entry["m_pol"] = surface["m"]
        contribution = entry["deltaw"][0]
        contribution["name"] = "classical"
        # `deltaw[:].value` is FLT_0D (real-valued): the imaginary part has no
        # slot here and stays only in the native Pest3MatchingOutput.
        contribution["value"] = surface["delta_prime_real"]
    # `ntms.mode[]` has no field naming the code that produced an entry, and
    # RDCON and STRIDE append to the same AOS, so the fragment records the
    # exact index range this call wrote and, per entry, the values with no
    # IMAS slot: the imaginary part of Delta-prime and the local criteria at
    # the surface. Without the range a surface cannot be attributed (#143).
    # Both lists come from the same delta_prime_diagonal(), row for row; pair
    # them by position and say so loudly if a caller ever hands in a diagonal
    # that does not line up, rather than silently dropping the criteria.
    criteria = result.rational_surface_stability()
    if criteria and [row["m"] for row in criteria] != [surface["m"] for surface in diagonal]:
        warnings.warn(
            f"{result.solver} n={result.n_tor}: the surface list does not match the solver's own; "
            "local criteria are left out of ntms.code.parameters",
            RuntimeWarning,
            stacklevel=2,
        )
        criteria = []
    if not criteria:
        criteria = [{} for _ in diagonal]
    surfaces = "".join(
        "<surface"
        + _xml_attrs(
            mode=start + offset,
            m=surface["m"],
            psi_n=surface.get("psi_n"),
            q=surface.get("q"),
            delta_prime_imag=surface.get("delta_prime_imag"),
            **{key: criteria[offset].get(key) for key in ("di", "dr", "h", "ca1")},
        )
        + "/>"
        for offset, surface in enumerate(diagonal)
    )
    ntms_fragment = (
        f'<solver name="{escape(result.solver)}" n_tor="{result.n_tor}" version="{NTMS_FRAGMENT_VERSION}"'
        f' time_slice="{time_slice}" mode_start="{start}" mode_count="{len(diagonal)}">'
        f"<msing>{result.msing}</msing>{surfaces}</solver>"
    )
    _append_code_parameters(ods, "ntms", ntms_fragment, code_name="GPEC-suite")
    _set_output_flag(ods, "ntms", time_slice, 0)


def _resistive_native_xml(result: Pest3MatchingOutput) -> str:
    """RDCON/STRIDE values with no IMAS slot (#939), as XML elements.

    The radial profiles are the solver's own grid at full resolution (mpsi + 1
    points, about 25 kB per (time slice, n) at mpsi 256). Of the PEST3
    matrices only Delta-prime is carried, row-major and complex: it is the one
    the analysis reads (its diagonal is ``ntms.deltaw[0]``, its off-diagonal
    terms couple surfaces). A', B', Gamma' and the Galerkin Delta are the
    intermediate matching matrices it is built from and stay in the native
    :class:`~vaft.code.gpec.Pest3MatchingOutput`, which remains the authority
    for an exact round trip.
    """
    # Read defensively: a result that carries no profiles or matrix (a caller's
    # minimal stand-in, an older container) writes the fragment without them.
    parts = []
    psi_n = getattr(result, "psi_n", None)
    if psi_n is not None:
        attrs = [f'psi_n="{_xml_numbers(psi_n)}"']
        for name in ("q", "di", "dr", "h", "ca1"):
            values = getattr(result, name, None)
            if values is not None and np.asarray(values).shape == np.asarray(psi_n).shape:
                attrs.append(f'{name}="{_xml_numbers(values)}"')
        parts.append("<local_profiles " + " ".join(attrs) + "/>")
    delta_prime = getattr(result, "Delta_prime", None)
    matrix = None if delta_prime is None else np.asarray(delta_prime)
    if matrix is not None and matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        parts.append(
            f'<delta_prime_matrix size="{matrix.shape[0]}" real="{_xml_numbers(matrix.real.ravel())}"'
            f' imag="{_xml_numbers(matrix.imag.ravel())}"/>'
        )
    return "".join(parts)


def extract_rdcon_stability(ods: ODS) -> list[dict[str, Any]]:
    """RDCON/STRIDE stability results from the ODS, one dict per (time slice, solver, n_tor).

    Reads the version-2 resistive fragments :func:`mhd_linear` writes into
    ``mhd_linear.code.parameters`` and joins them with the rational surfaces of
    :func:`ntms_solver_surfaces` and their classical Delta-prime from
    ``ntms.time_slice[t].mode[i].deltaw[0].value``, returning:

    * ``solver`` (``"rdcon"``/``"stride"``), ``n_tor``, ``time_slice``,
      ``position``, ``msing``;
    * ``psi_n``, ``q``, ``D_I``, ``D_R``, ``H``, ``C_A``: the solver's radial
      local-stability profiles (``H`` is RDCON-only; ``None`` where absent);
    * ``delta_prime_matrix``: the full complex ``(msing, msing)`` matrix;
    * ``surfaces``: one dict per rational surface, ``m``, ``psi_n``, ``q``,
      ``delta_prime`` (complex: the real part from ``ntms``, the imaginary part
      from the fragment), and ``di``/``dr``/``h``/``ca1`` at the surface.

    Values only, no verdicts: ``D_I > 0`` (Mercier) and ``D_R > 0`` (GGJ
    resistive interchange) are criteria for a consumer, and Delta-prime with
    ``D_R`` is not a tearing verdict without an inner-layer solution. Every
    row has the same keys. Re-mapping one (time slice, solver, n) keeps the
    last fragment. Version-1 fragments carry none of this and yield nothing;
    malformed ones are skipped with a warning. The ODS is not modified.
    """
    import xml.etree.ElementTree as ET

    if not path_exists(ods, "mhd_linear.code.parameters"):
        return []
    text = ods["mhd_linear.code.parameters"]
    try:
        root = ET.fromstring(text.decode() if isinstance(text, bytes) else str(text))
    except ET.ParseError as exc:
        warnings.warn(f"mhd_linear.code.parameters is not XML ({exc}); no resistive payload", RuntimeWarning, stacklevel=2)
        return []
    latest: dict[tuple[int, str, int], dict[str, Any]] = {}
    for solver in root.iter("solver"):
        name = solver.get("name")
        if name not in ("rdcon", "stride"):
            continue
        try:
            if int(solver.get("version", "1")) < 2:
                continue
            row = _parse_resistive_fragment(solver)
        except (TypeError, ValueError) as exc:
            warnings.warn(f"skipping malformed {name} fragment: {exc}", RuntimeWarning, stacklevel=2)
            continue
        key = (row["time_slice"], row["solver"], row["n_tor"])
        latest.pop(key, None)  # re-mapping: the last fragment wins
        latest[key] = row
    if not latest:
        return []
    # Re-mapping appends a fresh set of ntms.mode[] entries and a fresh fragment;
    # the last fragment per key is the one that pairs with the last mhd_linear
    # fragment, so the earlier run's surfaces are dropped, not appended.
    surfaces: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    for key, rows in _ntms_solver_fragments(ods):
        surfaces[key] = rows
    for key, row in latest.items():
        rows = []
        for surface in surfaces.get(key, []):
            path = f"ntms.time_slice.{surface['time_slice']}.mode.{surface['mode']}.deltaw.0.value"
            real = float(ods[path]) if path_exists(ods, path) else None
            imag = surface.get("delta_prime_imag")
            rows.append({
                "m": surface["m"],
                "psi_n": surface.get("psi_n"),
                "q": surface.get("q"),
                "delta_prime": None if real is None else complex(real, 0.0 if imag is None else imag),
                **{name: surface.get(name) for name in ("di", "dr", "h", "ca1")},
            })
        row["surfaces"] = rows
    return list(latest.values())


def _parse_resistive_fragment(solver: Any) -> dict[str, Any]:
    msing_text = solver.findtext("msing")
    row: dict[str, Any] = {
        "solver": solver.get("name"),
        "n_tor": int(solver.get("n_tor")),
        "time_slice": int(solver.get("time_slice")),
        "position": int(solver.get("position")),
        "msing": None if msing_text is None else int(msing_text),
    }
    profiles = solver.find("local_profiles")
    row["psi_n"] = None if profiles is None else _numbers(profiles.get("psi_n"))
    for native, public in (("q", "q"), ("di", "D_I"), ("dr", "D_R"), ("h", "H"), ("ca1", "C_A")):
        row[public] = None if profiles is None else _numbers(profiles.get(native))
    matrix = solver.find("delta_prime_matrix")
    if matrix is None:
        row["delta_prime_matrix"] = None
    else:
        size = int(matrix.get("size"))
        values = _complex_numbers(matrix.get("real"), matrix.get("imag"))
        if values is None or values.size != size * size:
            raise ValueError(f"delta_prime_matrix holds {0 if values is None else values.size} values, not {size}x{size}")
        row["delta_prime_matrix"] = values.reshape(size, size)
    return row


def mhd_linear(ods: ODS, source: str, options: Optional[dict] = None) -> dict[int, dict[str, Any]]:
    """Parse GPEC-suite output under ``source`` into `mhd_linear`/`ntms`.

    ``options["module"]`` selects which solver's output to read (``"dcon"``
    by default, matching prior behavior); one of ``"dcon"``, ``"rdcon"``,
    ``"stride"``. ``options["modes"]`` is the full requested ``n_tor`` grid:
    supplying it makes the `toroidal_mode` AOS dense and its positions stable
    across time slices and solvers (see :func:`ensure_toroidal_mode_grid`);
    omitting it falls back to whatever ``source`` yielded. Returns a ``{n_tor: {...}}`` dict of values kept alongside
    the ODS in the caller's run manifest (RDCON/STRIDE's per-surface
    Delta-prime, which has no `mhd_linear` slot for the full detail even
    though its diagonal now also reaches `ntms`).

    As a side effect, writes the full lossless native output container
    (:class:`~vaft.code.gpec.DconOutput` or
    :class:`~vaft.code.gpec.Pest3MatchingOutput`) to
    ``<source>/dcon_native_n<mode>.json`` or
    ``<source>/<solver>_matching_n<mode>.json``.
    """
    if options is None:
        options = {}

    time_slice = options.get("time_slice", 0)
    module = str(options.get("module", "dcon")).lower()
    if module not in _MODULE_PATTERNS:
        raise ValueError(f"Unsupported mhd_linear source module: {module!r}")
    pattern = _MODULE_PATTERNS[module]

    modes: list[int] = []
    for filename in sorted(os.listdir(source)):
        match = pattern.fullmatch(filename)
        if match:
            modes.append(int(match.group(1)))
    modes = sorted(set(modes))

    # Parse every matched mode first, dropping ones that fail to read, so the
    # AOS-position loop below only ever enumerates over what will actually be
    # written -- computing `position` from the raw (pre-parse) mode list would
    # skip a position for each parse failure and index straight past the end
    # of the AOS for the next successful entry.
    parsed: list[tuple[int, Any]] = []
    for mode in modes:
        try:
            if module == "dcon":
                parsed.append((mode, read_dcon_output(source, mode=mode)))
            else:
                parsed.append((mode, read_pest3_matching_output(source, solver=module, mode=mode)))
        except Exception as exc:
            # One unreadable output must not abort the other modes (or the
            # other solvers sharing this ODS), but it must not vanish either:
            # without this warning a file the reader rejects is
            # indistinguishable downstream from "this cell was never run".
            warnings.warn(
                f"{module} n={mode}: skipping unreadable output in {source}: "
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

    # The `toroidal_mode` AOS is a dense grid over the *requested* mode set, so
    # every mode keeps the same array position in every time slice and a
    # consumer can slice the ODS as a regular (time, n_tor) grid. Callers that
    # know the grid pass it in; a standalone call without one falls back to
    # whatever this directory actually yielded, which keeps the mapper usable
    # on its own. Either way `n_tor` is written explicitly on every entry --
    # position is layout, never the physical mode number.
    grid: list[int] = [int(n) for n in options.get("modes", [])]
    for _, result in parsed:
        if result.n_tor not in grid:
            if grid:
                # A solver produced a mode the caller did not ask for; keep it
                # (dropping real results is worse than a ragged tail) but say
                # so, since it lands outside the stable part of the grid.
                warnings.warn(
                    f"{module}: n_tor={result.n_tor} is not in the requested mode "
                    f"grid {grid}; appending it after the grid",
                    RuntimeWarning,
                    stacklevel=2,
                )
            grid.append(result.n_tor)
    if grid:
        ensure_toroidal_mode_grid(ods, time_slice, grid)

    extras: dict[int, dict[str, Any]] = {}
    for mode, result in parsed:
        position = grid.index(result.n_tor)
        if module == "dcon":
            _write_dcon_entry(ods, time_slice, position, result)
            extras[result.n_tor] = {
                "module": "dcon",
                "variable": "W_t_eigenvalue",
                "value": None if result.total1 is None else result.total1.real,
            }
            try:
                result.write_json(os.path.join(source, f"dcon_native_n{mode}.json"))
            except OSError:
                pass
        else:
            # Computed once and shared with the IDS writer: it is O(msing) work
            # per (time, mode) cell and both consumers want the same values.
            diagonal = result.delta_prime_diagonal()
            _write_resistive_entry(ods, time_slice, position, result, diagonal)
            extras[result.n_tor] = {
                "module": module,
                "variable": "Delta_prime",
                "value": diagonal,
            }
            try:
                result.write_json(os.path.join(source, f"{module}_matching_n{mode}.json"))
            except OSError:
                pass

    return extras
