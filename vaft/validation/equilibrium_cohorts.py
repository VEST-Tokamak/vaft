"""What failed, in which operating regime (issue #76).

Two halves of this question already exist and have never met. #820 put a
stated rule on *what state the plasma was in* -- phase, topology, the current
and rate the labels were derived from. #845 made *how each slice ended*
readable from the library -- the exit path, the criteria ``chkerr`` refused it
on, the solver routines that gave up and the ones that only warned.

Joining them is what lets the issue's own question be asked: do failure modes
sort by operating regime, or is "high-current shots do not converge" a claim
that dissolves when the cohorts are drawn properly?

Joined **by time, never by position**
-------------------------------------

EFIT's log names each slice by whole milliseconds; the reconstruction names
them in seconds. Zipping the two lists is the repeat of this repository's most
frequent defect, and it is invisible whenever the two happen to agree in
length -- which is exactly the case a one-shot fixture exercises. So the join
is a lookup keyed on the rounded millisecond, and the three ways it can be
incomplete are reported rather than absorbed:

* a reconstruction slice the log has nothing for (``termination`` is ``None``),
* a log slice the reconstruction has nothing for (``unmatched_log_times``),
* two reconstruction slices landing on the same millisecond, which EFIT's own
  resolution cannot tell apart -- refused, because silently keeping one of
  them would attribute a failure to the wrong slice.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .equilibrium_regime import (
    RegimeLabels,
    classify_equilibrium_regimes,
)

__all__ = [
    "FAILURE_MODES",
    "SliceOutcome",
    "CohortJoin",
    "join_regime_and_termination",
    "summarize_by_cohort",
]

#: What became of a slice, in one word. ``unknown`` is a real answer: it means
#: the log carried nothing for that instant, not that the slice succeeded.
FAILURE_MODES = (
    "accepted",
    "rejected",
    "collapsed",
    "boundary_failure",
    "axis_failure",
    "solver_failure",
    "unknown",
)

#: Which routine's failure gets its own name. `bound` traces the boundary and
#: `findax` locates the axis and separatrix; between them they are almost all
#: of VEST's solver failures, and lumping them as "solver" would hide the
#: distinction a recovery strategy turns on.
_NAMED_ROUTINES = {"bound": "boundary_failure", "findax": "axis_failure"}


@dataclass(frozen=True)
class SliceOutcome:
    """One slice: the regime it was in, and what became of it."""

    time: float
    regime: RegimeLabels
    #: The log record from :func:`vaft.code.efit.parse_slices`, or ``None``
    #: when the log has no slice at this instant.
    termination: Mapping[str, Any] | None
    mode: str

    @property
    def phase(self) -> str:
        return self.regime.phase

    @property
    def topology(self) -> str:
        return self.regime.topology

    @property
    def solver_routines(self) -> tuple[str, ...]:
        if not self.termination:
            return ()
        return tuple(item["routine"] for item in self.termination.get("solver_errors", ()))

    @property
    def warning_routines(self) -> tuple[str, ...]:
        if not self.termination:
            return ()
        return tuple(item["routine"] for item in self.termination.get("warnings", ()))

    @property
    def failure_codes(self) -> tuple[int, ...]:
        if not self.termination:
            return ()
        return tuple(int(item["code"]) for item in self.termination.get("failures", ()))

    def as_dict(self) -> dict[str, Any]:
        """A JSON-safe record carrying both halves and their inputs."""
        return {
            "time": float(self.time),
            "mode": self.mode,
            "regime": self.regime.as_dict(),
            "exit_path": None if not self.termination else self.termination.get("exit_path"),
            "iterations_n": None if not self.termination else self.termination.get("iterations_n"),
            "solver_routines": list(self.solver_routines),
            "warning_routines": list(self.warning_routines),
            "failure_codes": list(self.failure_codes),
        }


@dataclass(frozen=True)
class CohortJoin:
    """The joined slices, and every way the join was incomplete."""

    outcomes: tuple[SliceOutcome, ...]
    #: Log slices whose millisecond no reconstruction slice claims.
    unmatched_log_times: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.outcomes)

    def __iter__(self):
        return iter(self.outcomes)


def _mode(record: Mapping[str, Any] | None) -> str:
    """One word for what became of a slice, from the log alone.

    Order matters and is not arbitrary. A collapse is reported ahead of the
    solver error it causes: a slice that falls to a null solution then fails
    in `bound` failed *because* it collapsed, and calling that a boundary
    failure would send a reader after the boundary tracer.
    """
    if record is None:
        return "unknown"
    if record.get("collapsed"):
        return "collapsed"
    routines = [item["routine"] for item in record.get("solver_errors", ())]
    for routine in routines:
        if routine in _NAMED_ROUTINES:
            return _NAMED_ROUTINES[routine]
    if routines:
        return "solver_failure"
    if record.get("accepted"):
        return "accepted"
    return "rejected"


def join_regime_and_termination(
    ods: Any,
    slices: Sequence[Mapping[str, Any]],
    *,
    policy=None,
) -> CohortJoin:
    """Put each reconstruction slice's regime beside how EFIT ended it.

    ``slices`` is what :func:`vaft.code.efit.parse_slices` returns for the run
    that produced this reconstruction.
    """
    labels = classify_equilibrium_regimes(ods, policy=policy)

    by_ms: dict[int, int] = {}
    collisions: list[int] = []
    for index, label in enumerate(labels):
        key = int(round(float(label.time) * 1000.0))
        if key in by_ms:
            collisions.append(key)
        by_ms[key] = index
    if collisions:
        raise ValueError(
            "two reconstruction slices fall on the same millisecond "
            f"({', '.join(str(item) for item in sorted(set(collisions)))}), which is all "
            "the resolution EFIT's log has. Joining them would attribute a failure to "
            "the wrong slice; reconstruct at a cadence of 1 ms or coarser, or join "
            "against a-files, which carry the full time."
        )

    log_by_ms = {int(record["time_ms"]): record for record in slices}
    outcomes = []
    for label in labels:
        key = int(round(float(label.time) * 1000.0))
        record = log_by_ms.get(key)
        outcomes.append(
            SliceOutcome(
                time=float(label.time),
                regime=label,
                termination=record,
                mode=_mode(record),
            )
        )
    unmatched = tuple(sorted(set(log_by_ms) - set(by_ms)))
    return CohortJoin(outcomes=tuple(outcomes), unmatched_log_times=unmatched)


def summarize_by_cohort(
    outcomes: Iterable[SliceOutcome], *, group_by: Sequence[str] = ("phase",)
) -> dict[str, Any]:
    """Outcome counts per cohort, plus what each cohort's failures blame.

    ``group_by`` names attributes of :class:`SliceOutcome` -- ``phase`` and
    ``topology`` today -- and cohorts are keyed by their values joined with
    ``|``, in the order given.

    The warning tally is reported separately from the mode because it answers
    a different question. A slice's mode says which routine gave up; the
    warnings say what the solver had been complaining about on the way, and on
    VEST those are mostly ``findax`` reporting a separatrix it could not use.
    A cohort can be dominated by one while its modes name the other.
    """
    fields = tuple(group_by)
    if not fields:
        raise ValueError("group_by must name at least one field")

    items = list(outcomes)
    for field in fields:
        if not hasattr(SliceOutcome, field):
            raise ValueError(f"SliceOutcome has no '{field}' to group by")

    cohorts: dict[str, dict[str, Any]] = {}
    for item in items:
        key = "|".join(str(getattr(item, field)) for field in fields)
        bucket = cohorts.setdefault(
            key,
            {
                "slices": 0,
                "modes": Counter(),
                "solver_routines": Counter(),
                "warning_routines": Counter(),
                "failure_codes": Counter(),
            },
        )
        bucket["slices"] += 1
        bucket["modes"][item.mode] += 1
        bucket["solver_routines"].update(item.solver_routines)
        bucket["warning_routines"].update(item.warning_routines)
        bucket["failure_codes"].update(item.failure_codes)

    return {
        "group_by": list(fields),
        "slices": len(items),
        "cohorts": {
            key: {
                "slices": bucket["slices"],
                "modes": dict(sorted(bucket["modes"].items())),
                "solver_routines": dict(sorted(bucket["solver_routines"].items())),
                "warning_routines": dict(sorted(bucket["warning_routines"].items())),
                "failure_codes": {str(k): v for k, v in sorted(bucket["failure_codes"].items())},
            }
            for key, bucket in sorted(cohorts.items())
        },
    }
