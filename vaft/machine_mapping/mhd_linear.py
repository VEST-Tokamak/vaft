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
from vaft.ods_access import path_count

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
        f'<solver name="dcon" n_tor="{result.n_tor}">'
        f"<mlow>{result.mlow}</mlow><mhigh>{result.mhigh}</mhigh>"
        f"<mpert>{result.mpert}</mpert><mband>{result.mband}</mband>"
        '<energy_perturbed units="1" normalization="dcon_normalized"'
        ' imas_documented_units="J" source_variable="W_t_eigenvalue"/>'
        f"{eigenfunction_xml}"
        "</solver>"
    )
    _append_code_parameters(ods, "mhd_linear", fragment, code_name="DCON")
    _set_output_flag(ods, "mhd_linear", time_slice, 0)


def _write_resistive_entry(
    ods: ODS, time_slice: int, position: int, result: Pest3MatchingOutput, diagonal: list[dict[str, Any]]
) -> None:
    # `position` is this mode's slot in the dense n_tor grid, not an append
    # cursor; `n_tor` is (re)written here so the entry is self-describing even
    # if the grid was laid out by a different caller.
    mode_entry = ods["mhd_linear"]["time_slice"][time_slice]["toroidal_mode"][position]
    mode_entry["n_tor"] = result.n_tor
    mode_entry["ballooning_type"]["name"] = "Tearing"

    fragment = (
        f'<solver name="{escape(result.solver)}" n_tor="{result.n_tor}">'
        f"<mlow>{result.mlow}</mlow><mhigh>{result.mhigh}</mhigh>"
        f"<mpert>{result.mpert}</mpert><mband>{result.mband}</mband>"
        f"<msing>{result.msing}</msing>"
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
    ntms_fragment = f'<solver name="{escape(result.solver)}" n_tor="{result.n_tor}"><msing>{result.msing}</msing></solver>'
    _append_code_parameters(ods, "ntms", ntms_fragment, code_name="GPEC-suite")
    _set_output_flag(ods, "ntms", time_slice, 0)


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
