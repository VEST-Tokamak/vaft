"""Verification of linear MHD stability results: DCON, RDCON and STRIDE (issue #142).

``validate_stability`` asks *was the calculation performed as intended?* of
one DCON run and/or one PEST3 matching run (RDCON or STRIDE), optionally
against a second run of the same case at a different equilibrium resolution.
It answers in the validation layer's one vocabulary (#253, #337):
:class:`~vaft.validation.model.ValidationStatus` per check, aggregated with
:func:`vaft.validation.equilibrium.aggregate_status`.

A check that was **not requested** -- no second-resolution run passed, a local
criterion the run's ``dcon.in`` switched off, or a whole block (``dcon`` or
``matching``) not given -- is ``not_available`` with ``"requested": False`` and
is left out of the aggregate. Evidence that was requested but is missing (no
namelist to say what was evaluated, a matrix absent from the file) stays in it,
so it still turns a pass into ``indeterminate``. Without that split the packaged
``bal_flag=f`` would make every production DCON run indeterminate.

It does **not** say whether a plasma is stable. That is a physical criterion
applied to the solver's numbers (the sign of W_t, Δ′ > 0 at a surface, D_I > 0,
D_R > 0, C_A < 0) and lives with the consumer, for example the Tier A stability
atlas's layer 3 (#1429). Keeping the two apart is the #142 requirement that
numerical health and physical interpretation are reported separately.

Rules, not invented tolerances (#142):

* ``dcon_energies``: the least-stable energies must exist and be finite. With
  ``vac_flag`` off (DCON's default when ``dcon.in`` omits it) DCON computes no
  free-boundary energies, so this and the checks built on W_t are not requested.
* ``dcon_imaginary_part``: W_t is an eigenvalue of a Hermitian matrix and must
  be real. ``warn`` when its imaginary part exceeds its real part, because the
  sign of the real part then no longer decides anything.
* ``dcon_hermiticity``: the total-energy matrix W_t (DCON's W_p + W_v) split
  into Hermitian and anti-Hermitian parts, ``H = (W + W^H)/2`` and
  ``A = (W - W^H)/2``. ``warn`` when ``||A|| >= ||H||``, the matrix analogue of
  the imaginary-part rule; ``fail`` on a non-finite entry.
* ``dcon_local_criteria``: a criterion the run's ``dcon.in`` switched off
  (``mer_flag``/``bal_flag``) is not requested, never read as marginal; an
  unknown namelist is missing evidence (``not_available``); ``fail`` on a
  missing or non-finite value where it was evaluated.
* ``dcon_edge``: DCON's own control flow (``dcon.F:250-262``). The requested
  ``psiedge`` (DCON's default 1.0 when ``dcon.in`` omits it) must agree with
  whether an edge scan exists, and a truncated run must sit at the scan's peak
  of Re dW_edge, searched as ``dcon.F:253`` does: from ``pre_edge`` on
  (:meth:`~vaft.code.gpec.DconEdgeScan.peak_search_start`), over filled
  entries only. ``pre_edge`` counts the nominal scan points below q(psiedge),
  which DCON takes from its cubic ``sq`` spline (``sing.f:224``,
  ``direct.f:194``); the check evaluates the same cubic on the same nodes and
  brackets it by its distance to the linear interpolant plus
  ``Q_AT_PSIEDGE_ATOL``. When the two ends of that bracket start the search
  at different entries and reach different verdicts, the status is
  ``indeterminate``, not ``fail``: the peak was placed, the count was not.
* ``dcon_resolution``: the sign of W_t at two equilibrium resolutions of the
  same n. A disagreement is ``indeterminate`` (the sign is not resolved), not
  a failure of the run; a check run at a different n is ``fail``, as in
  ``matching_resolution``, because it is not a resolution check at all.
* ``matching_matrices``: Δ′ exists, is finite, and is msing × msing.
* ``matching_surfaces``: each rational surface satisfies ``q_s = m/n``; ``fail``
  when ``|n q_s - m| > 1e-6`` (exactly 0 on 398 production RDCON/STRIDE runs),
  on mismatched array lengths, or on a non-positive n or q. A surface outside
  the solver's q profile grid is a ``warn``.
* ``matching_resolution``: per-surface Δ′ at two resolutions of the same solver
  and n. Surfaces pair by m and by ψ_N within 1e-3 (a matching heuristic: the
  rational surface moves with the grid, not by more), one to one; a surface
  found at only one resolution counts as unmatched. The 20 % bound is the
  atlas heuristic, the median change of #141's mid-radius surfaces, and is
  labelled as such. ``pass`` only when every surface is consistent;
  ``indeterminate`` otherwise, and when either run's Δ′ matrix is unusable.

The report is strict JSON: non-finite numbers become ``None``.

These specs live in :data:`STABILITY_CHECKS`, not in
:data:`vaft.validation.registry.CHECKS`, whose contract is the equilibrium
report. They reuse its :class:`~vaft.validation.registry.CheckSpec`.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from vaft.validation.model import ValidationStatus
from vaft.validation.registry import CheckSpec

__all__ = [
    "STABILITY_CHECKS",
    "Q_AT_PSIEDGE_ATOL",
    "SELF_CONSISTENCY_RTOL",
    "SURFACE_IDENTITY_ATOL",
    "SURFACE_MATCH_PSI_ATOL",
    "validate_stability",
]

#: Heuristic (labelled, #142): the atlas's two-resolution consistency bound, the
#: median mpsi 256/512 change of #141's mid-radius RDCON surfaces.
SELF_CONSISTENCY_RTOL = 0.20

_PROVIDER_DCON = "vaft.code.gpec.read_dcon_output"
_PROVIDER_MATCHING = "vaft.code.gpec.read_pest3_matching_output"


def _spec(key: str, unit: str, provider: str, method: str) -> CheckSpec:
    return CheckSpec(key, key.split(".", 1)[0], unit, provider, method)


#: What is stable about each check (unit, provider, rule). Rules only: none of
#: these has a calibrated tolerance yet (#141 follow-up).
STABILITY_CHECKS: dict[str, CheckSpec] = {
    spec.key: spec
    for spec in (
        _spec("verification.dcon_energies", "1", _PROVIDER_DCON,
              "fail unless the least-stable W_t, W_p and W_v exist and are finite"),
        _spec("verification.dcon_imaginary_part", "1", _PROVIDER_DCON,
              "warn when |Im W_t| > |Re W_t| (a Hermitian eigenvalue is real)"),
        _spec("verification.dcon_hermiticity", "1", _PROVIDER_DCON,
              "warn when the anti-Hermitian part of W_t is at least its Hermitian part; fail on non-finite entries"),
        _spec("verification.dcon_local_criteria", "", _PROVIDER_DCON,
              "fail on missing or non-finite evaluated D_I/D_R/C_A; a criterion switched off is not requested"),
        _spec("verification.dcon_edge", "", _PROVIDER_DCON,
              "fail when the requested psiedge disagrees with the edge scan, or a truncated run is off the Re dW_edge peak; "
              "indeterminate when q(psiedge) is too close to a nominal scan point to place the search start"),
        _spec("verification.dcon_resolution", "", _PROVIDER_DCON,
              "indeterminate when the sign of W_t differs between two resolutions of the same n; "
              "fail when the check run is a different n; not_available without a check run"),
        _spec("verification.matching_matrices", "", _PROVIDER_MATCHING,
              "fail unless Delta_prime exists, is finite and msing x msing"),
        _spec("verification.matching_surfaces", "", _PROVIDER_MATCHING,
              "fail when |n*q_s - m| > 1e-6 or the surface arrays disagree; warn for a surface outside the q profile grid"),
        _spec("verification.matching_resolution", "1", _PROVIDER_MATCHING,
              "indeterminate unless every surface's diagonal Δ′ agrees in sign and within "
              "SELF_CONSISTENCY_RTOL (heuristic) between two resolutions; not_available without a check run"),
    )
}


#: |n q_s - m| above this is a misassigned surface. RDCON and STRIDE locate a
#: rational surface by solving q = m/n, so the identity holds to rounding (it is
#: exactly 0 on 398 production runs); 1e-6 only absorbs a text round trip.
SURFACE_IDENTITY_ATOL = 1e-6

#: Heuristic (labelled): two resolutions' rational surfaces are the same surface
#: when their psi_N differ by at most this.
SURFACE_MATCH_PSI_ATOL = 1e-3

#: Floor of the bracket on q(psiedge) in ``dcon_edge``. VAFT's cubic spline
#: and DCON's ``sq`` spline share the nodes but not the boundary condition
#: (not-a-knot vs ``"extrap"``, ``direct.f:194``); on the #792 profile they
#: differ by ~1e-5 at psi_N 0.93, and this is one order above that.
Q_AT_PSIEDGE_ATOL = 1e-4


def _result(status: ValidationStatus, **fields: Any) -> dict[str, Any]:
    return {"status": str(status), **fields}


def _not_requested(reason: str) -> dict[str, Any]:
    return _result(ValidationStatus.NOT_AVAILABLE, requested=False, reason=reason)


def _aggregate(checks: dict[str, dict[str, Any]]) -> ValidationStatus:
    from vaft.validation.equilibrium import aggregate_status

    return aggregate_status(c["status"] for c in checks.values() if c.get("requested", True))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(abs(complex(value)))
    except (TypeError, ValueError):
        return False


def _real(value: Any) -> Optional[float]:
    return None if value is None else float(np.real(value))


# -- DCON ---------------------------------------------------------------------------


def _dcon_energies(out: Any) -> dict[str, Any]:
    values = {"W_t": out.total1, "W_p": out.plasma1, "W_v": out.vacuum1}
    bad = [name for name, value in values.items() if not _finite(value)]
    return _result(
        ValidationStatus.FAIL if bad else ValidationStatus.PASS,
        missing_or_nonfinite=bad,
        **{name: _real(value) for name, value in values.items()},
    )


def _dcon_imaginary_part(out: Any) -> dict[str, Any]:
    w = out.total1
    if not _finite(w):
        return _result(ValidationStatus.NOT_AVAILABLE, reason="no finite W_t")
    if w == 0:
        return _result(ValidationStatus.INDETERMINATE, reason="W_t is exactly zero")
    ratio = abs(w.imag) / abs(w.real) if w.real else math.inf
    return _result(ValidationStatus.WARN if ratio > 1 else ValidationStatus.PASS, imag_over_real=ratio)


def _dcon_hermiticity(out: Any) -> dict[str, Any]:
    matrix = None if out.W_t is None else np.asarray(out.W_t)
    if matrix is None or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        return _result(ValidationStatus.NOT_AVAILABLE, reason="no square total-energy matrix")
    if not np.all(np.isfinite(matrix)):
        return _result(ValidationStatus.FAIL, reason="non-finite entry in the total-energy matrix")
    hermitian = float(np.linalg.norm((matrix + matrix.conj().T) / 2))
    anti = float(np.linalg.norm((matrix - matrix.conj().T) / 2))
    if hermitian == 0 and anti == 0:
        return _result(ValidationStatus.INDETERMINATE, reason="zero matrix")
    ratio = anti / hermitian if hermitian else math.inf
    return _result(ValidationStatus.WARN if ratio >= 1 else ValidationStatus.PASS, anti_over_hermitian=ratio)


def _dcon_local_criteria(out: Any) -> dict[str, Any]:
    evaluation = out.evaluation
    if evaluation is None or evaluation.mer_flag is None or evaluation.bal_flag is None:
        # No dcon.in beside the output: what was evaluated cannot be claimed.
        return _result(ValidationStatus.NOT_AVAILABLE, reason="unknown which criteria the run evaluated")
    mercier, ballooning = evaluation.mercier, evaluation.ballooning
    fields = {"mercier_evaluated": mercier, "ballooning_evaluated": ballooning}
    if not (mercier or ballooning):
        return {**_not_requested("mer_flag and bal_flag are off"), **fields}
    problems = []
    if mercier:
        for name in ("di", "dr"):
            values = getattr(out, name)
            if values is None or not np.all(np.isfinite(np.asarray(values, dtype=float))):
                problems.append(name)
    if ballooning:
        if out.ca1 is None or out.ca1_evaluated is None:
            problems.append("ca1")
        else:
            evaluated = np.asarray(out.ca1_evaluated, dtype=bool)
            if not np.all(np.isfinite(np.asarray(out.ca1, dtype=float)[evaluated])):
                problems.append("ca1")
    return _result(ValidationStatus.FAIL if problems else ValidationStatus.PASS,
                   missing_or_nonfinite=problems, **fields)


#: DCON's own default (dcon_mod.f:124), used when dcon.in exists but omits the
#: key. psiedge's default lives with the reader
#: (:attr:`~vaft.code.gpec.DconEvaluation.requested_psiedge`) so the mhd_linear
#: payload and this check name the same requested edge.
_DCON_DEFAULT_NPERQ_EDGE = 20


def _namelist_known(out: Any) -> bool:
    return out.evaluation is not None and out.evaluation.source == "dcon.in"


def _vacuum_requested(out: Any) -> Optional[bool]:
    """``vac_flag`` from the run's namelist (DCON default off); None when unknown."""
    if not _namelist_known(out):
        return None
    return bool(out.evaluation.vac_flag)


def _dcon_edge(out: Any) -> dict[str, Any]:
    # edge_treatment is derived from the scan's presence, so the check that means
    # something is against what the namelist *asked* for. DCON scans when
    # psiedge < psilim (sing.f:224). The original psilim is overwritten in a
    # truncated run, but it is never below the scan's last point.
    treatment = out.edge_treatment
    scan = out.edge_scan
    requested = None if out.evaluation is None else out.evaluation.requested_psiedge
    fields = {"edge_treatment": treatment, "requested_psiedge": requested}
    if requested is None or out.psilim is None:
        return _result(ValidationStatus.NOT_AVAILABLE, reason="requested psiedge or psilim unknown", **fields)
    if scan is None:
        asked_scan = requested < float(out.psilim)
        return _result(ValidationStatus.FAIL if asked_scan else ValidationStatus.PASS,
                       psilim=float(out.psilim), **fields)
    psi = np.asarray(scan.psi_n, dtype=float)
    dw = np.real(np.asarray(scan.dW))
    if not psi.size or requested >= float(np.nanmax(psi)):
        return _result(ValidationStatus.FAIL, reason="edge scan present although psiedge asked for none", **fields)
    if out.q is None or out.psi_n is None or int(getattr(out, "n_tor", 0) or 0) <= 0:
        return _result(ValidationStatus.NOT_AVAILABLE, reason="no q profile or n to place the peak search", **fields)
    # dcon.F:253 searches dw_edge(pre_edge:i_edge): past the pre-edge entries
    # (sing.f:226-238) and over filled entries only (unfilled keep psi_edge = 0).
    # pre_edge counts nominal points below q(psiedge) from DCON's cubic sq
    # spline; a linear interpolant is ~1e-3 off on the mpsi grid and moves the
    # count by one on a few per cent of psiedge values. The cubic on the same
    # nodes is bracketed by its distance to the linear value (plus the
    # boundary-condition floor) and the search is run from both ends.
    nperq = out.evaluation.nperq_edge or _DCON_DEFAULT_NPERQ_EDGE
    grid, q = np.asarray(out.psi_n, float), np.asarray(out.q, float)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.diff(grid) > 0):
        return _result(ValidationStatus.NOT_AVAILABLE, reason="q profile grid is not strictly increasing", **fields)
    from scipy.interpolate import CubicSpline

    q_at_psiedge = float(CubicSpline(grid, q)(requested))
    q_linear = float(np.interp(requested, grid, q))
    band = max(abs(q_at_psiedge - q_linear), Q_AT_PSIEDGE_ATOL)
    first = scan.peak_search_start(q_at_psiedge, int(out.n_tor), nperq)
    starts = sorted({first, *(scan.peak_search_start(q_at_psiedge + sign * band, int(out.n_tor), nperq) for sign in (-1, 1))})
    fields.update(psilim=float(out.psilim), q_at_psiedge=q_at_psiedge, peak_search_start=first)
    verdicts: dict[int, Optional[float]] = {}
    for start in starts:
        searched = (np.arange(psi.size) >= start) & (psi > 0) & np.isfinite(dw)
        if not searched.any():
            verdicts[start] = None
            continue
        verdicts[start] = float(psi[np.flatnonzero(searched)[int(np.argmax(dw[searched]))]])
    if all(peak is None for peak in verdicts.values()):
        return _result(ValidationStatus.FAIL, reason="truncated run without a usable edge scan", **fields)
    at_peak = {start: peak is not None and math.isclose(peak, float(out.psilim), rel_tol=0, abs_tol=1e-9)
               for start, peak in verdicts.items()}
    if len(set(at_peak.values())) > 1:
        return _result(
            ValidationStatus.INDETERMINATE,
            reason="q(psiedge) within interpolation error of a nominal scan point: the search start is ambiguous",
            peak_search_start_bracket=starts,
            psi_n_at_dW_peak=[verdicts[start] for start in starts],
            **fields,
        )
    return _result(
        ValidationStatus.PASS if at_peak[first] else ValidationStatus.FAIL,
        psi_n_at_dW_peak=verdicts[first],
        **fields,
    )


def _dcon_resolution(out: Any, check: Any) -> dict[str, Any]:
    if check is None:
        return _not_requested("no second-resolution run")
    if int(check.n_tor) != int(out.n_tor):
        return _result(ValidationStatus.FAIL, reason="check run is a different n",
                       n_tor=[int(out.n_tor), int(check.n_tor)])
    a, b = out.total1, check.total1
    if not (_finite(a) and _finite(b)):
        return _result(ValidationStatus.NOT_AVAILABLE, reason="W_t missing in one run")
    agree = bool(np.sign(a.real) == np.sign(b.real) and a.real != 0)
    return _result(
        ValidationStatus.PASS if agree else ValidationStatus.INDETERMINATE,
        W_t=float(a.real),
        W_t_check=float(b.real),
        sign_agreement=agree,
    )


# -- PEST3 matching (RDCON / STRIDE) --------------------------------------------------


def _matching_matrices(out: Any) -> dict[str, Any]:
    matrix = None if out.Delta_prime is None else np.asarray(out.Delta_prime)
    msing = int(out.msing)
    if matrix is None:
        return _result(ValidationStatus.FAIL, reason="no Delta_prime", msing=msing)
    shape_ok = matrix.shape == (msing, msing)
    finite = bool(np.all(np.isfinite(matrix)))
    return _result(
        ValidationStatus.PASS if shape_ok and finite else ValidationStatus.FAIL,
        msing=msing,
        shape=list(matrix.shape),
        finite=finite,
    )


def _matching_surfaces(out: Any) -> dict[str, Any]:
    if out.m is None or out.q_rational is None or out.psi_n_rational is None:
        return _result(ValidationStatus.NOT_AVAILABLE, reason="no rational-surface coordinates")
    n = int(out.n_tor)
    m = np.asarray(out.m, dtype=int)
    q = np.asarray(out.q_rational, dtype=float)
    psi = np.asarray(out.psi_n_rational, dtype=float)
    if n <= 0:
        return _result(ValidationStatus.FAIL, reason=f"non-positive n_tor {n}")
    if not (m.shape == q.shape == psi.shape and m.ndim == 1):
        return _result(ValidationStatus.FAIL, reason="m, q_s and psi_n_s lengths differ",
                       lengths=[int(m.size), int(q.size), int(psi.size)])
    misassigned = [int(mm) for mm, qq in zip(m, q)
                   if not math.isfinite(qq) or qq <= 0 or abs(n * qq - mm) > SURFACE_IDENTITY_ATOL]
    grid = None if getattr(out, "psi_n", None) is None else np.asarray(out.psi_n, dtype=float)
    beyond = []
    if grid is not None and grid.size:
        lo, hi = np.nanmin(grid), np.nanmax(grid)
        beyond = [int(mm) for mm, pp in zip(m, psi) if not lo <= pp <= hi]
    if misassigned:
        status = ValidationStatus.FAIL
    elif beyond:
        status = ValidationStatus.WARN
    else:
        status = ValidationStatus.PASS
    return _result(status, msing=int(m.size), misassigned_m=misassigned, beyond_profile_grid_m=beyond)


def _matching_resolution(out: Any, check: Any) -> dict[str, Any]:
    if check is None:
        return _not_requested("no second-resolution run")
    if (check.solver, int(check.n_tor)) != (out.solver, int(out.n_tor)):
        return _result(ValidationStatus.FAIL, reason="check run is a different solver or n",
                       solver=[out.solver, check.solver], n_tor=[int(out.n_tor), int(check.n_tor)])
    unusable = [name for name, run in (("primary", out), ("check", check))
                if _matching_matrices(run)["status"] != str(ValidationStatus.PASS)]
    if unusable:
        return _result(ValidationStatus.INDETERMINATE, reason="Delta_prime unusable", unusable=unusable)
    primary = out.delta_prime_diagonal()
    other = check.delta_prime_diagonal()
    consistent = inconsistent = unmatched = 0
    claimed: set[int] = set()
    for row in primary:
        partner_index = next(
            (j for j, o in enumerate(other) if j not in claimed and o["m"] == row["m"]
             and o["psi_n"] is not None and row["psi_n"] is not None
             and abs(o["psi_n"] - row["psi_n"]) <= SURFACE_MATCH_PSI_ATOL),
            None,
        )
        if partner_index is None:
            unmatched += 1
            continue
        claimed.add(partner_index)
        partner = other[partner_index]
        a, b = row["delta_prime_real"], partner["delta_prime_real"]
        rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
        if np.sign(a) == np.sign(b) and rel <= SELF_CONSISTENCY_RTOL:
            consistent += 1
        else:
            inconsistent += 1
    # A surface found at only one resolution is as unresolved as one that moved.
    unmatched += len(other) - len(claimed)
    if not primary:
        status = ValidationStatus.NOT_AVAILABLE
    elif inconsistent or unmatched:
        status = ValidationStatus.INDETERMINATE
    else:
        status = ValidationStatus.PASS
    return _result(
        status,
        consistent=consistent,
        inconsistent=inconsistent,
        unmatched=unmatched,
        rtol=SELF_CONSISTENCY_RTOL,
        rtol_basis="heuristic: median mpsi 256/512 change of #141 mid-radius surfaces",
        psi_match_atol=SURFACE_MATCH_PSI_ATOL,
        psi_match_basis="heuristic: a rational surface moves less than this between resolutions",
    )


def validate_stability(
    *,
    dcon: Any = None,
    dcon_check: Any = None,
    matching: Any = None,
    matching_check: Any = None,
) -> dict[str, Any]:
    """Verify one stability case; returns a JSON-serializable report.

    ``dcon``/``dcon_check`` are :class:`~vaft.code.gpec.DconOutput` at the
    reported and a second equilibrium resolution. ``matching``/``matching_check``
    are :class:`~vaft.code.gpec.Pest3MatchingOutput` (RDCON or STRIDE),
    likewise. Any may be None: its checks are then ``not_available`` and not
    requested, and the top-level status aggregates only the blocks given. A
    check run must be the same case at another resolution: a different ``n_tor``
    fails its resolution check (a different solver likewise, for matching). The
    report has the shape of :func:`vaft.validation.validate_equilibrium`'s::

        {"schema_version": 1, "status": ..., "summary": {"dcon": ..., "matching": ...},
         "verification": {"dcon_energies": {"status": ..., ...}, ...}}
    """
    energies_off = dcon is not None and _vacuum_requested(dcon) is False
    off = _not_requested("vac_flag is off: DCON computed no free-boundary energies")
    dcon_checks = (
        {
            "dcon_energies": dict(off) if energies_off else _dcon_energies(dcon),
            "dcon_imaginary_part": dict(off) if energies_off else _dcon_imaginary_part(dcon),
            "dcon_hermiticity": dict(off) if energies_off else _dcon_hermiticity(dcon),
            "dcon_local_criteria": _dcon_local_criteria(dcon),
            "dcon_edge": _dcon_edge(dcon),
            "dcon_resolution": dict(off) if energies_off else _dcon_resolution(dcon, dcon_check),
        }
        if dcon is not None
        else {name: _not_requested("no DCON run given") for name in (
            "dcon_energies", "dcon_imaginary_part", "dcon_hermiticity",
            "dcon_local_criteria", "dcon_edge", "dcon_resolution")}
    )
    matching_checks = (
        {
            "matching_matrices": _matching_matrices(matching),
            "matching_surfaces": _matching_surfaces(matching),
            "matching_resolution": _matching_resolution(matching, matching_check),
        }
        if matching is not None
        else {name: _not_requested("no matching run given") for name in ("matching_matrices", "matching_surfaces", "matching_resolution")}
    )
    summary = {"dcon": _aggregate(dcon_checks), "matching": _aggregate(matching_checks)}
    given = [status for status, run in ((summary["dcon"], dcon), (summary["matching"], matching)) if run is not None]
    from vaft.validation.equilibrium import aggregate_status

    provenance = {}
    if matching is not None:
        provenance["matching_solver"] = matching.solver
    return _json_safe({
        "schema_version": 1,
        "status": str(aggregate_status(given)),
        "summary": {name: str(status) for name, status in summary.items()},
        "provenance": provenance,
        "verification": {**dcon_checks, **matching_checks},
    })
