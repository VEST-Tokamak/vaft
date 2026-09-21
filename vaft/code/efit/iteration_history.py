"""EFIT's Picard trajectory per slice, as a reusable object (issue #1038).

A reconstruction is stored as its final state; how EFIT got there is printed
and then thrown away. This module keeps what EFIT already reports without any
instrumentation -- the ``it=`` lines of its terminal log and the per-iteration
arrays of its NetCDF m-file -- as one typed history per slice, so a numerical
study reads a trajectory instead of re-parsing a log.

What the iteration counter is
-----------------------------

EFIT's ``fit`` runs an outer loop over current-profile updates and, inside it,
up to ``NXITER`` equilibrium steps. The counter it prints as ``it=`` and uses
to index the m-file's ``cerror``/``cchisq``/``czmaxi`` arrays is ``nitera``,
incremented inside the *inner* loop: it is cumulative across both, restarting
at 1 on every slice. With ``NXITER=3`` a VEST slice prints ``it=`` up to 15.

The outer and inner counters are never printed, and the inner loop leaves
early once the increment falls below ``ERROR``. So they are recorded only when
``NXITER=1``, where every outer iteration is exactly one inner step; otherwise
they are ``None`` and :attr:`EFITIterationHistory.numbering` says why. They are
not reconstructed from the printed increments, which carry four digits.

Two sources, one trajectory
---------------------------

The log carries every quantity to three or four printed digits; the m-file
carries ``cchisq``, ``cerror`` and ``czmaxi`` in single precision. Where both
exist for a slice and agree on the number of steps, the m-file's values are
used and the log contributes the rest (``dz``, the exit path, the failures).
``czmaxi`` is in centimetres (``update_parameters.F90``); the history is in
metres, as the log's ``zm`` is. Where the counts disagree, the log is kept and
the slice carries a note. The log is the universe of slices whenever it
printed any step at all: a slice EFIT ran no Picard step on still writes an
m-file, with one zero in each array, and that zero is not an iteration.

Picard iteration is a numerical axis, not a physical one, so none of this is
written into ``equilibrium.time_slice``; a run records only a pointer to the
sidecar file (:data:`SIDECAR_NAME`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .termination import _parse_slice_records

__all__ = [
    "ITERATION_HISTORY_LEVELS",
    "ITERATION_HISTORY_SCHEMA",
    "ITERATION_HISTORY_SCHEMA_VERSION",
    "SIDECAR_NAME",
    "EFITIteration",
    "EFITSliceIterationHistory",
    "EFITIterationHistory",
    "parse_iteration_history",
    "iteration_history_from_workdir",
    "read_iteration_history",
    "plot_iteration_convergence",
    "compare_iteration_histories",
]

ITERATION_HISTORY_SCHEMA = "vaft.efit.iteration_history"
ITERATION_HISTORY_SCHEMA_VERSION = 1
#: The levels :class:`vaft.code.efit.EFITConfig` accepts today. ``none`` is
#: routine production: nothing is parsed or written.
ITERATION_HISTORY_LEVELS = ("none", "summary")
#: Levels issue #1038 plans that need EFIT instrumented first: per-iteration
#: diagnostic vectors (phase 2) and psi/axis/boundary snapshots (phase 3).
PLANNED_ITERATION_HISTORY_LEVELS = ("diagnostics", "state", "full")
SIDECAR_NAME = "efit_iteration_history.json"

#: EFIT's defaults for the settings the history's thresholds come from
#: (``set_defaults.f90``, ``data_input.F90``), used only when the run's
#: configuration left them unset and says so.
_EFIT_DEFAULT_NXITER = 1
_EFIT_DEFAULT_ERRMIN = 1.0e-2
#: A log time and an m-file time name the same slice when they agree to the
#: printed millisecond.
_TIME_TOLERANCE_S = 5.0e-4

_CUMULATIVE_MEANING = (
    "EFIT's nitera: the counter printed as it= and the index of the m-file's "
    "cerror/cchisq/czmaxi arrays. It is incremented inside the inner (NXITER) "
    "loop, so it is cumulative across outer and inner iterations, and it "
    "restarts at 1 on every slice."
)


def _nan(value: Any) -> float:
    return float("nan") if value is None else float(value)


def _json_float(value: float) -> float | None:
    return None if value is None or not math.isfinite(float(value)) else float(value)


@dataclass(frozen=True)
class EFITIteration:
    """One Picard step of one slice.

    ``error`` is EFIT's ``errorm``: the largest flux change of the step over
    the flux span, divided by the relaxation factor. It is the increment the
    inner loop stops on, not a Grad-Shafranov residual. ``axis_z`` is the
    magnetic axis height in metres, ``dz`` the vertical-control shift the log
    prints beside it. NaN where the source did not carry a value.
    """

    cumulative: int
    chi2: float
    error: float
    axis_z: float = float("nan")
    dz: float = float("nan")
    outer: int | None = None
    inner: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cumulative": self.cumulative,
            "outer": self.outer,
            "inner": self.inner,
            "chi2": _json_float(self.chi2),
            "error": _json_float(self.error),
            "axis_z": _json_float(self.axis_z),
            "dz": _json_float(self.dz),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EFITIteration":
        return cls(
            cumulative=int(data["cumulative"]),
            outer=None if data.get("outer") is None else int(data["outer"]),
            inner=None if data.get("inner") is None else int(data["inner"]),
            chi2=_nan(data.get("chi2")),
            error=_nan(data.get("error")),
            axis_z=_nan(data.get("axis_z")),
            dz=_nan(data.get("dz")),
        )


@dataclass(frozen=True)
class EFITSliceIterationHistory:
    """The Picard trajectory of one slice and how EFIT said it ended.

    The termination fields are :func:`vaft.code.efit.parse_slices`'s, and are
    ``None`` for a slice known only from its m-file. ``sources`` names what
    the values came from; ``notes`` says what was not used and why.
    """

    time: float
    iterations: tuple[EFITIteration, ...] = ()
    time_ms: int | None = None
    exit_path: str | None = None
    iconvr: int | None = None
    accepted: bool | None = None
    collapsed: bool | None = None
    failures: tuple[Mapping[str, Any], ...] = ()
    warnings: tuple[Mapping[str, Any], ...] = ()
    solver_errors: tuple[Mapping[str, Any], ...] = ()
    sources: tuple[str, ...] = ()
    mfile: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def iterations_n(self) -> int:
        return len(self.iterations)

    def _column(self, name: str) -> np.ndarray:
        return np.asarray([getattr(step, name) for step in self.iterations], dtype=float)

    @property
    def cumulative(self) -> np.ndarray:
        return np.asarray([step.cumulative for step in self.iterations], dtype=int)

    @property
    def chi2(self) -> np.ndarray:
        return self._column("chi2")

    @property
    def error(self) -> np.ndarray:
        return self._column("error")

    @property
    def axis_z(self) -> np.ndarray:
        return self._column("axis_z")

    @property
    def dz(self) -> np.ndarray:
        return self._column("dz")

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "time_ms": self.time_ms,
            "exit_path": self.exit_path,
            "iconvr": self.iconvr,
            "accepted": self.accepted,
            "collapsed": self.collapsed,
            "failures": [_json_mapping(item) for item in self.failures],
            "warnings": [dict(item) for item in self.warnings],
            "solver_errors": [dict(item) for item in self.solver_errors],
            "sources": list(self.sources),
            "mfile": self.mfile,
            "notes": list(self.notes),
            "iterations": [step.to_dict() for step in self.iterations],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EFITSliceIterationHistory":
        return cls(
            time=float(data["time"]),
            time_ms=None if data.get("time_ms") is None else int(data["time_ms"]),
            exit_path=data.get("exit_path"),
            iconvr=data.get("iconvr"),
            accepted=data.get("accepted"),
            collapsed=data.get("collapsed"),
            failures=tuple(dict(item) for item in data.get("failures", ())),
            warnings=tuple(dict(item) for item in data.get("warnings", ())),
            solver_errors=tuple(dict(item) for item in data.get("solver_errors", ())),
            sources=tuple(data.get("sources", ())),
            mfile=data.get("mfile"),
            notes=tuple(data.get("notes", ())),
            iterations=tuple(EFITIteration.from_dict(item) for item in data.get("iterations", ())),
        )


def _json_mapping(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _json_float(value) if isinstance(value, float) else value
        for key, value in item.items()
    }


@dataclass(frozen=True)
class EFITIterationHistory:
    """Every slice's Picard trajectory from one EFIT run.

    ``numbering`` states what the iteration counters mean for this run and
    whether outer/inner were recorded; ``provenance`` is the numerical
    configuration the trajectory belongs to. Slices are in time order; look one
    up by time with :meth:`at_time`, never by position -- a slice EFIT dropped
    shifts every index after it.
    """

    slices: tuple[EFITSliceIterationHistory, ...]
    level: str = "summary"
    numbering: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ITERATION_HISTORY_SCHEMA
    schema_version: int = ITERATION_HISTORY_SCHEMA_VERSION

    def __len__(self) -> int:
        return len(self.slices)

    def __iter__(self) -> Iterator[EFITSliceIterationHistory]:
        return iter(self.slices)

    @property
    def times(self) -> np.ndarray:
        return np.asarray([item.time for item in self.slices], dtype=float)

    def at_time(self, time: float, tolerance: float = _TIME_TOLERANCE_S) -> EFITSliceIterationHistory:
        """The slice nearest ``time`` (seconds); ``KeyError`` if none is within ``tolerance``."""
        if not self.slices:
            raise KeyError(f"no slice at t={time} s: the history is empty")
        distances = np.abs(self.times - float(time))
        index = int(np.argmin(distances))
        if distances[index] > tolerance:
            raise KeyError(
                f"no slice within {tolerance} s of t={time} s; "
                f"slices are at {self.times.tolist()}"
            )
        return self.slices[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "level": self.level,
            "numbering": dict(self.numbering),
            "provenance": _json_tree(self.provenance),
            "slices": [item.to_dict() for item in self.slices],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EFITIterationHistory":
        if data.get("schema") != ITERATION_HISTORY_SCHEMA:
            raise ValueError(f"not an EFIT iteration history: schema={data.get('schema')!r}")
        version = int(data.get("schema_version", 0))
        if version > ITERATION_HISTORY_SCHEMA_VERSION:
            raise ValueError(
                f"iteration history schema version {version} is newer than this "
                f"reader ({ITERATION_HISTORY_SCHEMA_VERSION})"
            )
        return cls(
            slices=tuple(EFITSliceIterationHistory.from_dict(item) for item in data["slices"]),
            level=str(data.get("level", "summary")),
            numbering=dict(data.get("numbering", {})),
            provenance=dict(data.get("provenance", {})),
            schema=str(data["schema"]),
            schema_version=version,
        )

    def write_json(self, path: str | Path) -> Path:
        """Write the history as JSON; NaN is written as ``null``."""
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), indent=1, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return destination


def _json_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_tree(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return _json_float(value)
    return value


def read_iteration_history(path: str | Path) -> EFITIterationHistory:
    """Load a history written by :meth:`EFITIterationHistory.write_json`."""
    return EFITIterationHistory.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _mfile_trajectory(mfile: Any) -> tuple[float | None, dict[str, np.ndarray], str | None]:
    """An m-file's time and per-iteration arrays; ``mfile`` is a path or a parsed MEQDSK."""
    from vaft.data.meqdsk import MEQDSK, read_meqdsk

    parsed = mfile if isinstance(mfile, MEQDSK) else read_meqdsk(mfile)
    source = parsed.source if isinstance(mfile, MEQDSK) else Path(mfile)
    arrays: dict[str, np.ndarray] = {}
    for name in ("cerror", "cchisq", "czmaxi"):
        if name in parsed:
            arrays[name] = np.asarray(parsed._at(name, 0), dtype=float).reshape(-1)
    return parsed.time_seconds(), arrays, (str(source) if source is not None else None)


def _nxiter_from(configuration: Mapping[str, Any] | None) -> tuple[int | None, str]:
    if not configuration:
        return None, "unknown: no run configuration was supplied"
    numerics = (configuration.get("scientific") or {}).get("numerics") or {}
    value = numerics.get("inner_iterations")
    if value is None:
        return _EFIT_DEFAULT_NXITER, "EFIT default: the k-file wrote no NXITER"
    return int(value), "run configuration"


def _numbering(nxiter: int | None, nxiter_source: str) -> dict[str, Any]:
    if nxiter == 1:
        outer_inner = (
            "derived: with NXITER=1 every outer iteration is exactly one inner "
            "step, so outer = cumulative and inner = 1"
        )
    elif nxiter is None:
        outer_inner = "not recorded: NXITER is not known for this run"
    else:
        outer_inner = (
            "not recorded: EFIT prints only the cumulative counter, and its "
            "inner loop can leave before NXITER steps once the increment falls "
            "below ERROR, so the outer/inner boundaries are not in its output"
        )
    return {
        "cumulative": _CUMULATIVE_MEANING,
        "nxiter": nxiter,
        "nxiter_source": nxiter_source,
        "outer_inner": outer_inner,
    }


def _provenance_from(configuration: Mapping[str, Any] | None) -> dict[str, Any]:
    if not configuration:
        return {}
    scientific = configuration.get("scientific") or {}
    execution = configuration.get("execution") or {}
    numerics = dict(scientific.get("numerics") or {})
    return {
        "scientific_sha256": configuration.get("scientific_sha256"),
        "numerics": numerics,
        "initialization": dict(scientific.get("initialization") or {}),
        "uncertainty_mode": (scientific.get("constraints") or {}).get("uncertainty_mode"),
        "args": list(execution.get("args") or ()),
        "shot": execution.get("shot"),
        "executable_sha256": execution.get("executable_sha256"),
        "requested_executable": execution.get("requested_executable"),
    }


def _log_matches(record: Mapping[str, Any], time: float) -> bool:
    # EFIT prints the slice time as whole milliseconds; which rounding it
    # uses is not ours to assume, so either names the slice.
    printed = record["time_ms"]
    milliseconds = time * 1000.0
    return printed in (int(math.floor(milliseconds + 1e-6)), int(round(milliseconds)))


def parse_iteration_history(
    text: str,
    *,
    mfiles: Sequence[Any] = (),
    configuration: Mapping[str, Any] | None = None,
    nxiter: int | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> EFITIterationHistory:
    """The Picard trajectory of every slice in an EFIT run.

    ``text`` is EFIT's terminal log (``run_efit.out``); ``mfiles`` are the
    run's m-files, as paths or parsed :class:`vaft.data.meqdsk.MEQDSK`.
    ``configuration`` is the run's resolved configuration
    (:func:`vaft.code.efit.resolved_efit_configuration`, or ``resolved`` in
    ``efit_configuration.json``); it supplies ``NXITER`` and the numerical
    provenance. ``nxiter`` overrides the configuration's; ``provenance`` is
    merged over what the configuration supplies.
    """
    records = _parse_slice_records(text or "")
    log_has_steps = any(record["iterations"] for record in records)

    if nxiter is not None:
        resolved_nxiter, nxiter_source = int(nxiter), "caller"
    else:
        resolved_nxiter, nxiter_source = _nxiter_from(configuration)
    derive_outer = resolved_nxiter == 1

    def steps_from_log(record: Mapping[str, Any]) -> list[EFITIteration]:
        return [
            EFITIteration(
                cumulative=int(step["n"]),
                chi2=float(step["chi2"]),
                error=float(step["gs_error"]),
                axis_z=float(step["axis_z"]),
                dz=float(step["dz"]),
                outer=int(step["n"]) if derive_outer else None,
                inner=1 if derive_outer else None,
            )
            for step in record["iterations"]
        ]

    # Log slices, in the order EFIT processed them, each waiting for its m-file.
    entries: list[dict[str, Any]] = [
        {
            "time": record["time_ms"] / 1000.0,
            "record": record,
            "steps": steps_from_log(record),
            "sources": ["log"] if record["iterations"] else [],
            "mfile": None,
            "notes": [],
        }
        for record in records
    ]

    trajectories = sorted(
        (_mfile_trajectory(item) for item in mfiles),
        key=lambda item: (item[0] is None, item[0] if item[0] is not None else 0.0),
    )
    for time, arrays, source in trajectories:
        if time is None:
            continue
        entry = next(
            (
                candidate
                for candidate in entries
                if candidate["mfile"] is None
                and candidate["record"] is not None
                and _log_matches(candidate["record"], time)
            ),
            None,
        )
        counts = {name: values.size for name, values in arrays.items()}
        if entry is None:
            entry = {
                "time": time, "record": None, "steps": [], "sources": [],
                "mfile": source, "notes": [],
            }
            entries.append(entry)
            if log_has_steps:
                # The log printed steps for other slices but none for this
                # one: EFIT ran no Picard step here, and the one-entry array
                # it still writes is not an iteration.
                entry["notes"].append(
                    "the log printed no Picard step for this slice; the m-file's "
                    f"arrays ({counts}) were not read as iterations"
                )
                continue
        else:
            entry["time"] = time
            entry["mfile"] = source
        if "cerror" not in arrays:
            entry["notes"].append("the m-file has no cerror; its arrays were not used")
            continue
        count = arrays["cerror"].size
        if entry["steps"] and len(entry["steps"]) != count:
            entry["notes"].append(
                f"the m-file has {count} iterations and the log {len(entry['steps'])}; "
                "the log's values were kept"
            )
            continue
        if not entry["steps"] and count == 1 and all(
            float(values[0]) == 0.0 for values in arrays.values() if values.size
        ):
            entry["notes"].append(
                "the m-file's arrays hold a single zero: EFIT wrote them for a "
                "slice it ran no Picard step on"
            )
            continue

        def column(name: str, scale: float = 1.0) -> np.ndarray:
            values = arrays.get(name)
            if values is None or values.size != count:
                return np.full(count, np.nan)
            return values * scale

        chi2, error = column("cchisq"), column("cerror")
        # czmaxi is zmaxis*100 (update_parameters.F90): centimetres.
        axis_z = column("czmaxi", 0.01)
        base = entry["steps"] or [None] * count
        entry["steps"] = [
            EFITIteration(
                cumulative=(old.cumulative if old is not None else index + 1),
                chi2=float(chi2[index]) if np.isfinite(chi2[index]) or old is None else old.chi2,
                error=float(error[index]),
                axis_z=float(axis_z[index]) if np.isfinite(axis_z[index]) or old is None else old.axis_z,
                dz=old.dz if old is not None else float("nan"),
                outer=(index + 1) if derive_outer else None,
                inner=1 if derive_outer else None,
            )
            for index, old in enumerate(base)
        ]
        entry["sources"].append("mfile")

    entries.sort(key=lambda item: item["time"])
    slices = []
    for entry in entries:
        record = entry["record"] or {}
        slices.append(
            EFITSliceIterationHistory(
                time=float(entry["time"]),
                time_ms=record.get("time_ms"),
                iterations=tuple(entry["steps"]),
                exit_path=record.get("exit_path"),
                iconvr=record.get("iconvr"),
                accepted=record.get("accepted"),
                collapsed=record.get("collapsed"),
                failures=tuple(record.get("failures", ())),
                warnings=tuple(record.get("warnings", ())),
                solver_errors=tuple(record.get("solver_errors", ())),
                sources=tuple(entry["sources"]),
                mfile=entry["mfile"],
                notes=tuple(entry["notes"]),
            )
        )
    merged_provenance = _provenance_from(configuration)
    merged_provenance.update(dict(provenance or {}))
    return EFITIterationHistory(
        slices=tuple(slices),
        numbering=_numbering(resolved_nxiter, nxiter_source),
        provenance=merged_provenance,
    )


def iteration_history_from_workdir(
    workdir: str | Path,
    *,
    shot: int | None = None,
    nxiter: int | None = None,
) -> EFITIterationHistory:
    """Rebuild the history of a finished run from what it left in ``workdir``.

    Reads ``run_efit.out``, the m-files (``m0<shot>.*``) and, when present,
    ``efit_configuration.json`` -- so any past run can be inspected, whether
    or not it asked for a history.
    """
    from .magnetic import _find_outputs

    base = Path(workdir)
    log = base / "run_efit.out"
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    manifest = base / "efit_configuration.json"
    configuration = None
    extra: dict[str, Any] = {"log": str(log) if log.is_file() else None}
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        configuration = payload.get("resolved")
        extra["vaft_version"] = payload.get("vaft_version")
        extra["vaft_revision"] = payload.get("vaft_revision")
        extra["table"] = payload.get("table")
    mfiles = _find_outputs(base, "m", shot)
    return parse_iteration_history(
        text, mfiles=mfiles, configuration=configuration, nxiter=nxiter, provenance=extra
    )


# ---------------------------------------------------------------------------
# Plotting


def _thresholds(history: EFITIterationHistory) -> list[tuple[float, str]]:
    numerics = (history.provenance or {}).get("numerics") or {}
    lines = []
    if numerics.get("error_tolerance") is not None:
        lines.append(
            (float(numerics["error_tolerance"]), f"ERROR = {numerics['error_tolerance']:.0e} (inner-loop exit)")
        )
    if numerics:
        errmin = numerics.get("error_minimum")
        label = "run" if errmin is not None else "EFIT default"
        value = float(errmin) if errmin is not None else _EFIT_DEFAULT_ERRMIN
        lines.append((value, f"ERRMIN = {value:.0e} ({label}; chi-square exit gate)"))
    return lines


def _convergence_panels(
    labelled: Sequence[tuple[str, EFITSliceIterationHistory]],
    thresholds: Sequence[tuple[float, str]],
    title: str,
) -> Any:
    from vaft.plot import LineSeries, Panels, Series
    from vaft.plot.display import DisplaySpec

    drawn = [(label, item) for label, item in labelled if item.iterations]
    if not drawn:
        raise ValueError("no slice in the selection has a recorded Picard step")
    chi2, error, axis = [], [], []
    for label, item in drawn:
        x = item.cumulative.astype(float)
        chi2.append(Series(x=x, y=item.chi2, label=label, style={"marker": "."}))
        error.append(Series(x=x, y=item.error, label=label, style={"marker": "."}))
        axis.append(Series(x=x, y=item.axis_z * 100.0, label=label, style={"marker": "."}))
    last = max(int(item.cumulative[-1]) for _label, item in drawn)
    span = np.asarray([1.0, float(max(last, 2))])
    for value, text in thresholds:
        error.append(
            Series(x=span, y=np.full(2, value), label=text,
                   style={"linestyle": "--", "color": "0.4", "linewidth": 1.0})
        )
    # A legend of many slices gives way to a trace count, which would leave
    # the dashed thresholds unnamed; the title names them either way.
    threshold_title = "dashed: " + ", ".join(text for _value, text in thresholds) if thresholds else ""
    return Panels(
        models=(
            LineSeries(series=tuple(chi2), x_label="Picard iteration (cumulative)",
                       y_label="chi-square (as EFIT prints it)", log_y=True),
            LineSeries(series=tuple(error), x_label="Picard iteration (cumulative)",
                       y_label="flux increment errorm", log_y=True, title=threshold_title),
            # Scientific ticks: an up-down symmetric fit keeps the axis at
            # round-off, which fixed notation prints as a column of zeros.
            LineSeries(series=tuple(axis), x_label="Picard iteration (cumulative)",
                       y_label="magnetic axis Z", y_unit="cm",
                       display=DisplaySpec(quantity="magnetic_axis_z", unit="cm", scale=100.0,
                                           notation="scientific")),
        ),
        ncols=1,
        share_x=True,
        suptitle=title,
    )


def _slice_label(item: EFITSliceIterationHistory) -> str:
    ending = item.exit_path or "no exit line"
    return f"t={item.time * 1e3:.1f} ms ({item.iterations_n} it, {ending})"


def plot_iteration_convergence(
    history: EFITIterationHistory,
    time: float | Sequence[float] | None = None,
    *,
    ax: Any = None,
    show: bool = False,
    **style: Any,
) -> tuple[Any, np.ndarray]:
    """Chi-square, flux increment and axis height against Picard iteration.

    One line per slice -- ``time`` (seconds, one or several) selects them, by
    time; ``None`` draws every slice with a recorded step. The increment panel
    carries the run's ``ERROR`` (where the inner loop stops) and ``ERRMIN``
    (below which the chi-square exit is allowed) when the history knows its
    configuration. The x axis is EFIT's cumulative counter
    (:attr:`EFITIterationHistory.numbering`). Returns ``(figure, axes)``.
    """
    from vaft.plot import render_panels

    if time is None:
        selected = list(history.slices)
    else:
        times = [time] if np.ndim(time) == 0 else list(time)
        selected = [history.at_time(float(value)) for value in times]
    model = _convergence_panels(
        [(_slice_label(item), item) for item in selected],
        _thresholds(history),
        "EFIT Picard convergence",
    )
    return render_panels(model, ax=ax, show=show, **style)


def compare_iteration_histories(
    histories: Mapping[str, EFITIterationHistory],
    time: float,
    *,
    ax: Any = None,
    show: bool = False,
    **style: Any,
) -> tuple[Any, np.ndarray]:
    """The same slice from several runs on one iteration axis.

    ``histories`` maps a label (``"NXITER=1"``) to a history; the slice is
    found in each by ``time``. Thresholds are drawn only when every run agrees
    on them, since a line that belongs to one run would mislabel the others.
    """
    from vaft.plot import render_panels

    labelled = [
        (f"{label}: {history.at_time(time).iterations_n} it, {history.at_time(time).exit_path}",
         history.at_time(time))
        for label, history in histories.items()
    ]
    thresholds = [_thresholds(history) for history in histories.values()]
    shared = thresholds[0] if thresholds and all(item == thresholds[0] for item in thresholds) else []
    model = _convergence_panels(labelled, shared, f"EFIT Picard convergence at t={time * 1e3:.1f} ms")
    return render_panels(model, ax=ax, show=show, **style)
