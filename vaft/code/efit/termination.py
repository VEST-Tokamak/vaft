"""What EFIT's own log says about how each slice ended (issues #76, #171).

EFIT prints, per slice, the Picard iterations it took, which exit path the
outer loop left by, the numbered criteria ``chkerr`` refused it on, and the
solver routines that gave up.  That is the only place several of those facts
exist: the a-file records *that* a slice was rejected and its ``lflag``, but
not that ``bound`` could not close a contour, and not that ``findax`` placed
the second separatrix off the grid.

The reader lived in ``workflow/efit_numerics/baseline_termination.py`` and was
reachable only by loading that script by path, so three studies imported it
from each other and nothing else could use it at all.  Parsing EFIT's stdout
is a fact about EFIT, not about any machine or any study, so it belongs here.

The record shape is unchanged from the workflow original -- plain dicts, the
same keys -- because three studies and their stored tables are written against
it.  One key is added: ``warnings``.

Why warnings are worth keeping
------------------------------

The original matched ``ERROR in <routine>`` only.  But EFIT's ``findax``
reports *warnings* when it finds a second separatrix it cannot use --
``2nd separatrix point is off grid``, ``not inside vessel`` -- and on shot
46742 those were the dominant signal, 36 of them against 18 errors, on the
slices that failed.  They are evidence about why a boundary could not be
formed, and they were being discarded.

They are kept apart from errors on purpose.  A warning does not mean the slice
failed: EFIT prints them and carries on, and a slice can warn and still be
accepted.  So a warning never opens a slice and never contributes to
``accepted``; it is recorded beside the slice it was printed in.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "EFIT_LOG_PATTERNS",
    "parse_slices",
]

_ITERATION = re.compile(r"\bt=\s*(\d+)\s+it=\s*(\d+)\s+chi2=\s*([0-9.E+-]+).*?err=\s*([0-9.E+-]+)")
_ICONVR = re.compile(r"iconvr=(\d+) satisfied")
_FAILED = re.compile(r"Failed to reach fit/convergence criteria, shot\s+(\d+)\s+([0-9.]+)")
_FAILURE = re.compile(r"Failure #(\d+),\s*([^=]*?)(?:=\s*([0-9.E+-]+))?\s*$")
_SOLVER_ERROR = re.compile(r"ERROR in (\w+) at r=\s*\d+, t=\s*(\d+): (.*)")
_SOLVER_WARNING = re.compile(r"WARNING in (\w+) at r=\s*\d+, t=\s*(\d+): (.*)")

#: The lines this reader understands, named so a caller can say which it
#: relied on and a change here is visible to them.
EFIT_LOG_PATTERNS = {
    "iteration": _ITERATION,
    "iconvr": _ICONVR,
    "failed": _FAILED,
    "failure": _FAILURE,
    "solver_error": _SOLVER_ERROR,
    "solver_warning": _SOLVER_WARNING,
}

#: Below this the fit residual has vanished; above the other, the
#: Grad-Shafranov error has not. Together they are a null solution, not a fit.
_COLLAPSE_CHI2 = 1.0e-5
_COLLAPSE_GS_ERROR = 0.1


def parse_slices(text: str) -> list[dict[str, Any]]:
    """Per-slice termination evidence, in the order EFIT processed them.

    EFIT prints the time in whole milliseconds only, so slices are delimited
    by the iteration counter restarting at 1, not by the time. Everything
    printed after a slice's last iteration and before the next slice's first
    belongs to that slice: its exit path, its solver errors and warnings, and
    the acceptance failures ``chkerr`` reports.

    With one exception, and it is not a small one. A slice that fails in
    ``bound`` before the first Picard iteration prints no iteration line at
    all, only ``ERROR in bound at r=..., t=...``. Delimiting on the iteration
    counter alone hands that error to the *previous* slice and loses the slice
    itself, so a run ending in a run of pre-iteration collapses reports both a
    short universe and a slice carrying failures that are not its own. Solver
    errors therefore also open a slice when they name a time the current slice
    does not have.

    Warnings do not open a slice. EFIT prints them and continues, so opening
    one would invent a slice that never ran; a warning printed before any
    slice has begun is therefore dropped rather than guessed at.

    One warning is not a warning. EFIT announces the chi-square exit as
    ``WARNING in fit ... iconvr=2 satisfied``, which is the line that says how
    the outer loop left, so it is read as the exit path and not as a warning.
    """
    slices: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def start(time_ms: int) -> dict[str, Any]:
        return {
            "time_ms": time_ms,
            "iterations": [],
            "exit_path": None,
            "iconvr": None,
            "accepted": None,
            "failures": [],
            "solver_errors": [],
            "warnings": [],
        }

    for line in text.splitlines():
        found = _ITERATION.search(line)
        if found:
            time_ms, iteration = int(found.group(1)), int(found.group(2))
            if iteration == 1:
                if current is not None:
                    slices.append(current)
                current = start(time_ms)
            if current is None:
                current = start(time_ms)
            current["iterations"].append(
                {"n": iteration, "chi2": float(found.group(3)), "gs_error": float(found.group(4))}
            )
            continue
        found = _SOLVER_ERROR.search(line)
        if found:
            named = int(found.group(2))
            if current is None or current["time_ms"] != named:
                if current is not None:
                    slices.append(current)
                current = start(named)
            current["solver_errors"].append(
                {"routine": found.group(1), "detail": found.group(3).strip()}
            )
            continue
        if current is None:
            continue
        # Before the general warning branch: EFIT announces the chi-square
        # exit as `WARNING in fit ... iconvr=2 satisfied`, and a warning
        # branch that matched first would swallow the one line that says how
        # the outer loop left.
        found = _ICONVR.search(line)
        if found:
            current["iconvr"] = int(found.group(1))
            current["exit_path"] = f"iconvr={found.group(1)}"
            continue
        found = _SOLVER_WARNING.search(line)
        if found:
            current["warnings"].append(
                {"routine": found.group(1), "detail": found.group(3).strip()}
            )
            continue
        if _FAILED.search(line):
            current["accepted"] = False
            continue
        found = _FAILURE.search(line.strip())
        if found:
            current["failures"].append(
                {
                    "code": int(found.group(1)),
                    "criterion": found.group(2).strip().rstrip(",").strip(),
                    "value": float(found.group(3)) if found.group(3) else None,
                }
            )
    if current is not None:
        slices.append(current)

    for record in slices:
        iterations = record["iterations"]
        record["iterations_n"] = max((item["n"] for item in iterations), default=0)
        # The log's chi2 is not the fit's chi-square after the first step: on a
        # slice that collapses to a null solution it falls to ~1e-7 while the
        # a-file reports 200. So both ends are kept and neither is called
        # "the" chi-square -- EFIT's own answer is the a-file's.
        record["chi2_initial"] = iterations[0]["chi2"] if iterations else None
        record["chi2_final"] = iterations[-1]["chi2"] if iterations else None
        record["gs_error"] = iterations[-1]["gs_error"] if iterations else None
        # A null solution: the residual vanishes while the Grad-Shafranov
        # error does not, and the axis never leaves zero. It is not a fit.
        record["collapsed"] = bool(
            iterations
            and record["chi2_final"] is not None
            and record["chi2_final"] < _COLLAPSE_CHI2
            and record["gs_error"] is not None
            and record["gs_error"] > _COLLAPSE_GS_ERROR
        )
        # Warnings deliberately do not enter this: EFIT prints them and carries
        # on, so a slice can warn and still be accepted.
        if record["accepted"] is None:
            record["accepted"] = not record["failures"] and not record["solver_errors"]
        if record["exit_path"] is None:
            record["exit_path"] = "solver_error" if record["solver_errors"] else "iterations_exhausted"
        del record["iterations"]

    return slices
