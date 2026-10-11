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

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .equilibrium_regime import (
    RegimeLabels,
    classify_equilibrium_regimes,
)

__all__ = [
    "FAILURE_MODES",
    "SliceOutcome",
    "CohortJoin",
    "cohort_rows",
    "join_regime_and_termination",
    "representative_cases",
    "summarize_by_cohort",
    "write_cohort_table",
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
    #: The product's graded status for this slice (``usable``,
    #: ``output_failed``, ``physical_failed``, ...), or ``unknown``.
    quality: str = "unknown"
    shot: int | None = None
    #: Which run or database the slice came from, for grouping across sources.
    source: str = "unknown"

    @property
    def phase(self) -> str:
        return self.regime.phase

    @property
    def topology(self) -> str:
        return self.regime.topology

    @property
    def current_bin(self) -> str:
        return self.regime.current_bin

    @property
    def rate_bin(self) -> str:
        return self.regime.rate_bin

    @property
    def era(self) -> str:
        return self.regime.era

    @property
    def event(self) -> str:
        return self.regime.event

    @property
    def succeeded(self) -> bool | None:
        """Whether the slice is a usable reconstruction.

        The graded status decides when there is one -- EFIT accepting a slice
        says nothing about whether its pressure is positive -- and the log's
        mode only when there is not. ``None`` when neither knows.
        """
        if self.quality != "unknown":
            return self.quality == "usable"
        if self.mode == "unknown":
            return None
        return self.mode == "accepted"

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
            "shot": self.shot,
            "source": self.source,
            "time": float(self.time),
            "mode": self.mode,
            "quality": self.quality,
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
    slice_statuses: Sequence[Mapping[str, Any]] | None = None,
    shot: int | None = None,
    source: str = "unknown",
    **evidence: Any,
) -> CohortJoin:
    """Put each reconstruction slice's regime beside how EFIT ended it.

    ``slices`` is what :func:`vaft.code.efit.parse_slices` returns for the run
    that produced this reconstruction. ``slice_statuses`` is the product
    manifest's per-slice grading (``{"time", "overall_status", ...}``),
    matched by millisecond like the log. ``evidence`` goes to
    :func:`classify_equilibrium_regimes` (``era``, ``passive_current``,
    ``events``, ``limloc``).

    Pass the run's constraints ODS to cohort every *attempted* slice; a
    reconstruction holds only those that wrote a g-file, so its failures
    would surface only as ``unmatched_log_times``.
    """
    labels = classify_equilibrium_regimes(ods, policy=policy, **evidence)
    status_by_ms: dict[int, str] = {}
    for record in slice_statuses or ():
        status_by_ms[int(round(float(record["time"]) * 1000.0))] = str(
            record.get("overall_status") or "unknown"
        )

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
                quality=status_by_ms.get(key, "unknown"),
                shot=None if shot is None else int(shot),
                source=str(source),
            )
        )
    unmatched = tuple(sorted(set(log_by_ms) - set(by_ms)))
    return CohortJoin(outcomes=tuple(outcomes), unmatched_log_times=unmatched)


def summarize_by_cohort(
    outcomes: Iterable[SliceOutcome], *, group_by: Sequence[str] = ("phase",)
) -> dict[str, Any]:
    """Outcome counts per cohort, plus what each cohort's failures blame.

    ``group_by`` names attributes of :class:`SliceOutcome` -- ``phase``,
    ``topology``, ``current_bin``, ``rate_bin``, ``era``, ``event``,
    ``quality``, ``source`` -- and cohorts are keyed by their values joined with
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
                "quality": Counter(),
            },
        )
        bucket["slices"] += 1
        bucket["modes"][item.mode] += 1
        bucket["solver_routines"].update(item.solver_routines)
        bucket["warning_routines"].update(item.warning_routines)
        bucket["failure_codes"].update(item.failure_codes)
        bucket["quality"][item.quality] += 1

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
                "quality": dict(sorted(bucket["quality"].items())),
            }
            for key, bucket in sorted(cohorts.items())
        },
    }


#: Exported columns, in order. Lists are ``;``-joined so a CSV cell holds one.
_COLUMNS = (
    "shot", "source", "time", "era", "phase", "topology", "limloc",
    "current", "fraction_of_peak", "current_bin", "current_rate",
    "magnitude_rate", "rate_bin", "eddy_current", "eddy_fraction", "event",
    "mode", "quality", "exit_path", "iterations_n", "solver_routines",
    "warning_routines", "failure_codes", "reasons", "rule",
    "classification_version",
)


def cohort_rows(outcomes: Iterable[SliceOutcome]) -> list[dict[str, Any]]:
    """One flat record per slice, every column of :data:`_COLUMNS`."""
    rows = []
    for item in outcomes:
        regime = item.regime
        termination = item.termination or {}
        rows.append(
            {
                "shot": item.shot,
                "source": item.source,
                "time": float(item.time),
                "era": regime.era,
                "phase": regime.phase,
                "topology": regime.topology,
                "limloc": regime.limloc,
                "current": regime.current,
                "fraction_of_peak": regime.fraction_of_peak,
                "current_bin": regime.current_bin,
                "current_rate": regime.current_rate,
                "magnitude_rate": regime.magnitude_rate,
                "rate_bin": regime.rate_bin,
                "eddy_current": regime.eddy_current,
                "eddy_fraction": regime.eddy_fraction,
                "event": regime.event,
                "mode": item.mode,
                "quality": item.quality,
                "exit_path": termination.get("exit_path"),
                "iterations_n": termination.get("iterations_n"),
                "solver_routines": ";".join(item.solver_routines),
                "warning_routines": ";".join(item.warning_routines),
                "failure_codes": ";".join(str(code) for code in item.failure_codes),
                "reasons": ";".join(regime.reasons),
                "rule": regime.rule,
                "classification_version": int(regime.version),
            }
        )
    return rows


def write_cohort_table(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    policy=None,
) -> Path:
    """Write the joined table as CSV or Parquet (by suffix), with its rule.

    The thresholds every label depends on go to ``<path>.meta.json`` beside
    it, whatever the format, and the version is also a column -- a table
    whose rule is not recorded cannot be re-cut or compared. Parquet needs
    ``pandas`` with ``pyarrow``; CSV needs nothing.
    """
    from vaft.machine_mapping.equilibrium_regime import vest_equilibrium_regime_policy

    resolved = policy or vest_equilibrium_regime_policy()
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise ValueError(f"write_cohort_table writes .csv or .parquet, not {target.suffix!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_COLUMNS), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in _COLUMNS})
    else:
        import pandas as pd

        pd.DataFrame([{key: row.get(key) for key in _COLUMNS} for row in rows], columns=list(_COLUMNS)).to_parquet(
            target, index=False
        )
    meta = target.with_name(target.name + ".meta.json")
    meta.write_text(
        json.dumps({"rows": len(rows), "columns": list(_COLUMNS), "thresholds": resolved.thresholds()}, indent=2),
        encoding="utf-8",
    )
    return target


def representative_cases(
    outcomes: Iterable[SliceOutcome],
    *,
    group_by: Sequence[str] = ("phase",),
    per_cohort: int = 1,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Deterministic example slices per cohort: successes and failures.

    Within a cohort, the cases chosen are those whose ``|Ip|`` is nearest the
    cohort's median, ties broken by shot then time -- the same input always
    gives the same cases, and a "typical" case is typical of its cohort
    rather than the first one a loop met. Success follows
    :attr:`SliceOutcome.succeeded`; a slice where it is ``None`` is neither.
    """
    fields = tuple(group_by)
    for field in fields:
        if not hasattr(SliceOutcome, field):
            raise ValueError(f"SliceOutcome has no '{field}' to group by")
    groups: dict[str, list[SliceOutcome]] = {}
    for item in outcomes:
        groups.setdefault("|".join(str(getattr(item, f)) for f in fields), []).append(item)

    def pick(items: list[SliceOutcome]) -> list[dict[str, Any]]:
        currents = [abs(i.regime.current) for i in items if i.regime.current is not None]
        median = float(np.median(currents)) if currents else 0.0
        ranked = sorted(
            items,
            key=lambda i: (
                abs(abs(i.regime.current) - median) if i.regime.current is not None else float("inf"),
                -1 if i.shot is None else int(i.shot),
                float(i.time),
            ),
        )
        return [
            {"shot": i.shot, "time": float(i.time), "current": i.regime.current, "mode": i.mode, "quality": i.quality}
            for i in ranked[: int(per_cohort)]
        ]

    return {
        key: {
            "successful": pick([i for i in items if i.succeeded is True]),
            "failed": pick([i for i in items if i.succeeded is False]),
        }
        for key, items in sorted(groups.items())
    }
