"""Verification of linear MHD stability results: DCON, RDCON and STRIDE (issue #142).

``validate_stability`` asks *was the calculation performed as intended?* of
one DCON run and/or one PEST3 matching run (RDCON or STRIDE), optionally
against a second run of the same case at a different equilibrium resolution.
It answers in the validation layer's one vocabulary (#253, #337):
:class:`~vaft.validation.model.ValidationStatus` per check, aggregated with
:func:`vaft.validation.equilibrium.aggregate_status`.

It does **not** say whether a plasma is stable. That is a physical criterion
applied to the solver's numbers (the sign of W_t, Δ′ > 0 at a surface, D_I > 0,
D_R > 0, C_A < 0) and lives with the consumer, for example the Tier A stability
atlas's layer 3 (#1429). Keeping the two apart is the #142 requirement that
numerical health and physical interpretation are reported separately.

Rules, not invented tolerances (#142):

* ``dcon_energies``: the least-stable energies must exist and be finite.
* ``dcon_imaginary_part``: W_t is an eigenvalue of a Hermitian matrix and must
  be real. ``warn`` when its imaginary part exceeds its real part, because the
  sign of the real part then no longer decides anything.
* ``dcon_hermiticity``: ``||W - W^H|| / ||W||`` of the total-energy matrix.
  ``warn`` when the non-Hermitian part is as large as the matrix itself.
* ``dcon_local_criteria``: ``indeterminate`` for a criterion the run did not
  evaluate (``mer_flag``/``bal_flag``), never read as marginal; ``fail`` on a
  non-finite value where it was evaluated.
* ``dcon_edge``: DCON's own control flow (``dcon.F:262-279``). A truncated run
  must sit at the peak of Re dW_edge, and a full-edge run carries no edge scan.
* ``dcon_resolution``: the sign of W_t at two equilibrium resolutions. A
  disagreement is ``indeterminate`` (the sign is not resolved), not a failure
  of the run.
* ``matching_matrices``: Δ′ exists, is finite, and is msing × msing.
* ``matching_surfaces``: each rational surface's ``n q_s`` must round to its m
  (``fail`` otherwise). A surface beyond the solver's q profile grid is a
  ``warn``.
* ``matching_resolution``: per-surface Δ′ at two resolutions. The 20 % bound is
  the atlas heuristic, the median change of #141's mid-radius surfaces, and is
  labelled as such. ``pass`` only when every surface is consistent;
  ``indeterminate`` otherwise.

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

__all__ = ["STABILITY_CHECKS", "SELF_CONSISTENCY_RTOL", "validate_stability"]

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
              "warn when ||W_t - W_t^H|| >= ||W_t||; not_available without the matrix"),
        _spec("verification.dcon_local_criteria", "", _PROVIDER_DCON,
              "fail on non-finite evaluated D_I/D_R/C_A; indeterminate when a criterion was not evaluated"),
        _spec("verification.dcon_edge", "", _PROVIDER_DCON,
              "fail unless a truncated run sits at the peak of Re dW_edge and a full-edge run has no edge scan"),
        _spec("verification.dcon_resolution", "", _PROVIDER_DCON,
              "indeterminate when the sign of W_t differs between two resolutions; not_available without a check run"),
        _spec("verification.matching_matrices", "", _PROVIDER_MATCHING,
              "fail unless Delta_prime exists, is finite and msing x msing"),
        _spec("verification.matching_surfaces", "", _PROVIDER_MATCHING,
              "fail when n*q_s does not round to m; warn for a surface beyond the q profile grid"),
        _spec("verification.matching_resolution", "1", _PROVIDER_MATCHING,
              "indeterminate unless every surface's diagonal Δ′ agrees in sign and within "
              "SELF_CONSISTENCY_RTOL (heuristic) between two resolutions; not_available without a check run"),
    )
}


def _result(status: ValidationStatus, **fields: Any) -> dict[str, Any]:
    return {"status": str(status), **fields}


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
    ratio = abs(w.imag) / abs(w.real) if w.real else math.inf
    return _result(ValidationStatus.WARN if ratio > 1 else ValidationStatus.PASS, imag_over_real=ratio)


def _dcon_hermiticity(out: Any) -> dict[str, Any]:
    matrix = None if out.W_t is None else np.asarray(out.W_t)
    if matrix is None or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        return _result(ValidationStatus.NOT_AVAILABLE, reason="no square total-energy matrix")
    norm = float(np.linalg.norm(matrix))
    if not math.isfinite(norm) or norm == 0:
        return _result(ValidationStatus.INDETERMINATE, reason="zero or non-finite matrix norm")
    residual = float(np.linalg.norm(matrix - matrix.conj().T)) / norm
    return _result(ValidationStatus.WARN if residual >= 1 else ValidationStatus.PASS, residual=residual)


def _dcon_local_criteria(out: Any) -> dict[str, Any]:
    evaluation = out.evaluation
    mercier = None if evaluation is None else evaluation.mercier
    ballooning = None if evaluation is None else evaluation.ballooning
    problems = []
    if mercier:
        for name in ("di", "dr"):
            values = getattr(out, name)
            if values is None or not np.all(np.isfinite(np.asarray(values, dtype=float))):
                problems.append(name)
    if ballooning and out.ca1 is not None and out.ca1_evaluated is not None:
        evaluated = np.asarray(out.ca1_evaluated, dtype=bool)
        if not np.all(np.isfinite(np.asarray(out.ca1, dtype=float)[evaluated])):
            problems.append("ca1")
    fields = {"mercier_evaluated": mercier, "ballooning_evaluated": ballooning, "nonfinite": problems}
    if problems:
        return _result(ValidationStatus.FAIL, **fields)
    if not (mercier and ballooning):
        return _result(ValidationStatus.INDETERMINATE, reason="a local criterion was not evaluated", **fields)
    return _result(ValidationStatus.PASS, **fields)


def _dcon_edge(out: Any) -> dict[str, Any]:
    treatment = out.edge_treatment
    scan = out.edge_scan
    if treatment == "full_edge":
        return _result(ValidationStatus.PASS if scan is None else ValidationStatus.FAIL, edge_treatment=treatment)
    if scan is None or out.psilim is None:
        return _result(ValidationStatus.FAIL, edge_treatment=treatment, reason="truncated run without its edge scan")
    peak = float(np.asarray(scan.psi_n)[int(np.argmax(np.real(scan.dW)))])
    at_peak = math.isclose(peak, float(out.psilim), rel_tol=0, abs_tol=1e-9)
    return _result(
        ValidationStatus.PASS if at_peak else ValidationStatus.FAIL,
        edge_treatment=treatment,
        psilim=float(out.psilim),
        psi_n_at_dW_peak=peak,
    )


def _dcon_resolution(out: Any, check: Any) -> dict[str, Any]:
    if check is None:
        return _result(ValidationStatus.NOT_AVAILABLE, reason="no second-resolution run")
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
    misassigned = [int(mm) for mm, qq in zip(m, q) if not math.isfinite(qq) or round(n * qq) != mm]
    grid = None if getattr(out, "psi_n", None) is None else np.asarray(out.psi_n, dtype=float)
    beyond = [] if grid is None or not grid.size else [int(mm) for mm, pp in zip(m, psi) if pp > np.nanmax(grid)]
    if misassigned:
        status = ValidationStatus.FAIL
    elif beyond:
        status = ValidationStatus.WARN
    else:
        status = ValidationStatus.PASS
    return _result(status, msing=int(m.size), misassigned_m=misassigned, beyond_profile_grid_m=beyond)


def _matching_resolution(out: Any, check: Any) -> dict[str, Any]:
    if check is None:
        return _result(ValidationStatus.NOT_AVAILABLE, reason="no second-resolution run")
    primary = out.delta_prime_diagonal()
    other = check.delta_prime_diagonal()
    consistent = inconsistent = unmatched = 0
    for row in primary:
        partner = next(
            (o for o in other if o["m"] == row["m"] and o["psi_n"] is not None and row["psi_n"] is not None
             and abs(o["psi_n"] - row["psi_n"]) <= 1e-3),
            None,
        )
        if partner is None:
            unmatched += 1
            continue
        a, b = row["delta_prime_real"], partner["delta_prime_real"]
        rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
        if np.sign(a) == np.sign(b) and rel <= SELF_CONSISTENCY_RTOL:
            consistent += 1
        else:
            inconsistent += 1
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
    likewise. Any may be None: its checks are then ``not_available``. The
    report has the shape of :func:`vaft.validation.validate_equilibrium`'s::

        {"schema_version": 1, "status": ..., "summary": {"dcon": ..., "matching": ...},
         "verification": {"dcon_energies": {"status": ..., ...}, ...}}
    """
    from vaft.validation.equilibrium import aggregate_status

    unavailable = _result(ValidationStatus.NOT_AVAILABLE, reason="no run given")
    dcon_checks = (
        {
            "dcon_energies": _dcon_energies(dcon),
            "dcon_imaginary_part": _dcon_imaginary_part(dcon),
            "dcon_hermiticity": _dcon_hermiticity(dcon),
            "dcon_local_criteria": _dcon_local_criteria(dcon),
            "dcon_edge": _dcon_edge(dcon),
            "dcon_resolution": _dcon_resolution(dcon, dcon_check),
        }
        if dcon is not None
        else {name: dict(unavailable) for name in (
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
        else {name: dict(unavailable) for name in ("matching_matrices", "matching_surfaces", "matching_resolution")}
    )
    summary = {
        "dcon": str(aggregate_status(c["status"] for c in dcon_checks.values())),
        "matching": str(aggregate_status(c["status"] for c in matching_checks.values())),
    }
    provenance = {}
    if matching is not None:
        provenance["matching_solver"] = matching.solver
    return {
        "schema_version": 1,
        "status": str(aggregate_status(summary.values())),
        "summary": summary,
        "provenance": provenance,
        "verification": {**dcon_checks, **matching_checks},
    }
