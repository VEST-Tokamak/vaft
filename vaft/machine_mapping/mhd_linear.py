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
- Everything else that has no IMAS home (mode-range provenance, the full
  PEST3 matching matrices, the Fourier-space eigenfunction from
  `solutions.bin`) goes into `mhd_linear.code.parameters` when it is
  solver-configuration metadata, or stays exclusively in the `vaft.code.gpec`
  output container (persisted as JSON next to the solver output) when it is
  a physics result with no IMAS field at all. Nothing is force-fit into a
  structurally-convenient-but-wrong field (e.g. `displacement_perpendicular`
  is deliberately never populated here -- see the module docstrings in
  `vaft.code.gpec._dcon_output` for why).

Schema reference: https://gafusion.github.io/omas/schema.html
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional
import warnings
from xml.sax.saxutils import escape

import numpy as np
from omas import ODS

from vaft.code.gpec import DconOutput, Pest3MatchingOutput, read_dcon_output, read_pest3_matching_output

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
    """Number of ``<ids>.time_slice`` entries, 0 when the node is absent.

    Plain ``in`` checks are non-mutating on an ODS; indexing a path that does
    not exist yet auto-vivifies it.
    """
    if f"{ids}.time_slice" not in ods:
        return 0
    return len(ods[f"{ids}.time_slice"])


def _existing_aos_count(ods: ODS, ids: str, time_slice: int, aos_name: str) -> int:
    """Length of ``<ids>.time_slice.<time_slice>.<aos_name>``, 0 if not present.

    Plain ``in`` checks are non-mutating on an ODS; indexing a path that does
    not exist yet auto-vivifies it, which would corrupt an ``ods`` this
    function was never asked to touch.
    """
    if f"{ids}.time_slice" not in ods:
        return 0
    if time_slice >= len(ods[f"{ids}.time_slice"]):
        return 0
    path = f"{ids}.time_slice.{time_slice}.{aos_name}"
    if path not in ods:
        return 0
    return len(ods[path])


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


def _set_output_flag(ods: ODS, ids: str, time_slice: int, flag: int) -> None:
    """Set ``<ids>.code.output_flag[time_slice]``, padding earlier slices.

    ``output_flag`` is an ``INT_1D`` over ``<ids>.time``, so writing index 2
    of an empty array is an error rather than an append. Pad the gap
    explicitly instead of letting a solver that succeeded only for a later
    time slice fail here.
    """
    path = f"{ids}.code.output_flag"
    existing = ods[path] if path in ods else []
    values = [int(value) for value in np.atleast_1d(existing)] if len(np.atleast_1d(existing)) else []
    while len(values) <= time_slice:
        values.append(_OUTPUT_FLAG_NOT_RUN)
    values[time_slice] = flag
    ods[path] = np.asarray(values, dtype=int)


def _write_dcon_entry(ods: ODS, time_slice: int, position: int, result: DconOutput) -> None:
    _ensure_time_slice(ods, "mhd_linear", time_slice)
    mode_entry = ods["mhd_linear"]["time_slice"][time_slice]["toroidal_mode"][position]
    mode_entry["n_tor"] = result.n_tor
    if result.W_t_eigenvalue is not None and result.W_t_eigenvalue.size:
        # Least-stable total-energy eigenvalue -- normalized/dimensionless,
        # stored despite the field's Joules documentation because it is the
        # closest existing slot; the unit mismatch is recorded explicitly in
        # code.parameters below rather than left for a future reader to guess.
        mode_entry["energy_perturbed"] = float(result.W_t_eigenvalue[0].real)

    # The units element is structured, not prose: a consumer checking whether
    # `energy_perturbed` is really in the Joules its IMAS documentation
    # promises can test `units != "J"` (or read `normalization`) instead of
    # having to parse an English sentence.
    fragment = (
        f'<solver name="dcon" n_tor="{result.n_tor}">'
        f"<mlow>{result.mlow}</mlow><mhigh>{result.mhigh}</mhigh>"
        f"<mpert>{result.mpert}</mpert><mband>{result.mband}</mband>"
        '<energy_perturbed units="1" normalization="dcon_normalized"'
        ' imas_documented_units="J" source_variable="W_t_eigenvalue"/>'
        "</solver>"
    )
    _append_code_parameters(ods, "mhd_linear", fragment, code_name="DCON")
    _set_output_flag(ods, "mhd_linear", time_slice, 0)


def _write_resistive_entry(
    ods: ODS, time_slice: int, position: int, result: Pest3MatchingOutput, diagonal: list[dict[str, Any]]
) -> None:
    _ensure_time_slice(ods, "mhd_linear", time_slice)
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
    ``"stride"``. Returns a ``{n_tor: {...}}`` dict of values kept alongside
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

    # `toroidal_mode` is an IMAS array of structures: entries must be
    # appended sequentially (position != mode number), so the physical
    # toroidal mode number is only ever recovered from `n_tor`, never from
    # array position. `existing` accounts for entries a prior call already
    # wrote for this time slice (e.g. a DCON pass before this RDCON pass on
    # the same `ods`) so repeated calls extend rather than overwrite them.
    existing = _existing_aos_count(ods, "mhd_linear", time_slice, "toroidal_mode") if parsed else 0

    extras: dict[int, dict[str, Any]] = {}
    for offset, (mode, result) in enumerate(parsed):
        position = existing + offset
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
