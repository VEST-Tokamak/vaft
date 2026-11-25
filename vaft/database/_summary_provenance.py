"""Per-row run record for the database summary: how each value was obtained.

``summary(..., provenance=True)`` attaches a :class:`RunRecord` to
``result.attrs["provenance"]``. Its ``table`` holds one row per (table row,
column) with a ``status``:

``computed``
    calculated by VAFT in this run, by its primary method.
``copied``
    the stored value, read as it is (unit conversion aside).
``fallback``
    calculated, but through a substitute branch -- an input taken from another
    leaf, or a default standing in for a missing value. ``inputs`` names it.
``failed``
    a step raised; ``reason`` carries the exception. The value is NaN.
``missing``
    NaN because an input is absent; nothing raised. ``reason`` says which.
``unrecorded``
    the preset's extractor does not record provenance yet; the value is there
    but how it was obtained is not stated.

The record changes no value: with the flag off nothing is recorded and the
extractors run exactly as before. It is collected through a context variable
the extractors consult, so an extractor that knows nothing about it still
works and its columns read ``unrecorded`` / ``missing``.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import logging
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

STATUSES = ("computed", "copied", "fallback", "failed", "missing", "unrecorded")
TABLE_COLUMNS = ("column", "status", "step", "reason", "inputs")

_ACTIVE: contextvars.ContextVar[RunRecorder | None] = contextvars.ContextVar(
    "vaft_summary_run_recorder", default=None
)
#: "time slice 8", "Time slice 8:", "equilibrium.time_slice.8", "time_slice[8]"
_SLICE_IN_MESSAGE = re.compile(r"time[ _]slice[ .\[]?(\d+)", re.IGNORECASE)


class RunRecorder:
    """What one shot's extractor did, keyed by (row position, column)."""

    def __init__(self) -> None:
        self.entries: dict[tuple[int, str], dict] = {}
        self.step_errors: dict[str, str] = {}
        # step -> {row position: [warning text]}; row None = not tied to a slice
        self.step_warnings: dict[str, dict[int | None, list[str]]] = {}
        self.snapshots: list[tuple[str, dict]] = []
        self.facts: dict[str, object] = {}

    def set(self, row: int, column: str, status: str, *, step: str = "", reason: str = "",
            inputs: str = "") -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown provenance status {status!r}")
        self.entries[(int(row), column)] = {
            "status": status, "step": step, "reason": reason, "inputs": inputs,
        }

    def note(self, row: int, columns, reason: str) -> None:
        """A reason found inside a helper, used if the column comes out NaN."""
        for column in columns:
            self.facts.setdefault("notes", {})[(int(row), column)] = reason

    def warnings_for(self, step: str, row: int, *, shot_level: bool = False) -> list[str]:
        """What ``step`` logged about this slice; with ``shot_level``, else about the shot."""
        by_row = self.step_warnings.get(step, {})
        if by_row.get(row):
            return list(by_row[row])
        if shot_level and by_row.get(None):
            return [f"{message} (shot-level)" for message in by_row[None]]
        return []


class _WarningCapture(logging.Handler):
    def __init__(self, recorder: RunRecorder, step: str) -> None:
        super().__init__(level=logging.WARNING)
        self.recorder, self.step = recorder, step

    def emit(self, record: logging.LogRecord) -> None:
        # the handler sits on the shared "vaft" logger; another thread's summary is not ours
        if active() is not self.recorder:
            return
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a malformed log call must not break the run
            return
        match = _SLICE_IN_MESSAGE.search(message)
        row = int(match.group(1)) if match else None
        self.recorder.step_warnings.setdefault(self.step, {}).setdefault(row, []).append(message)


def active() -> RunRecorder | None:
    """The recorder of the shot being extracted, or ``None`` when not recording."""
    return _ACTIVE.get()


@contextlib.contextmanager
def recording(enabled: bool = True):
    """Record what the extractor inside the block does (yields ``None`` if disabled)."""
    if not enabled:
        yield None
        return
    recorder = RunRecorder()
    token = _ACTIVE.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE.reset(token)


@contextlib.contextmanager
def capture_step_warnings(recorder: RunRecorder | None, step: str):
    """Collect the WARNING records ``vaft`` logs while one step runs."""
    if recorder is None:
        yield
        return
    handler = _WarningCapture(recorder, step)
    target = logging.getLogger("vaft")
    target.addHandler(handler)
    try:
        yield
    finally:
        target.removeHandler(handler)


@functools.cache
def code_revision() -> dict:
    """The installed vaft version and, from a git checkout, its commit."""
    import vaft

    root = Path(vaft.__file__).resolve().parents[1]
    commit = None
    dirty = None
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip() or None
        if commit:
            status = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no", "--", "vaft"],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout
            dirty = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"vaft_version": getattr(vaft, "__version__", None), "commit": commit, "dirty": dirty}


class RunRecord:
    """The run record of one summary call, safe to keep in ``DataFrame.attrs``.

    pandas compares ``attrs`` with ``==`` when it concatenates or merges and
    deep-copies them into every derived frame; a bare DataFrame there makes
    ``concat`` raise. This holder compares by identity and is not copied.
    """

    __slots__ = ("table", "code")

    def __init__(self, table: pd.DataFrame, code: dict) -> None:
        self.table = table
        self.code = dict(code)

    def __eq__(self, other) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def counts(self) -> pd.DataFrame:
        """Status counts per column (columns as rows, statuses as columns)."""
        if self.table.empty:
            return pd.DataFrame(columns=list(STATUSES))
        counts = pd.crosstab(self.table["column"], self.table["status"])
        return counts.reindex(columns=[s for s in STATUSES if s in counts.columns])

    def __repr__(self) -> str:
        commit = (self.code.get("commit") or "unknown")[:12]
        return f"RunRecord({len(self.table)} entries, vaft {self.code.get('vaft_version')}, commit {commit})"


def _present(value) -> bool:
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def shot_table(recorder: RunRecorder | None, frame: pd.DataFrame, columns, key_columns) -> pd.DataFrame:
    """The long (row key, column) table for one shot's frame."""
    entries = recorder.entries if recorder is not None else {}
    records = []
    keys = [key for key in key_columns if key in frame.columns]
    for position, (_, row) in enumerate(frame.iterrows()):
        key_values = {key: row[key] for key in keys}
        for column in columns:
            entry = entries.get((position, column))
            if entry is None:
                if column in key_columns:
                    continue
                value = row[column] if column in frame.columns else None
                entry = (
                    {"status": "unrecorded", "step": "",
                     "reason": "this preset's extractor does not record provenance", "inputs": ""}
                    if _present(value)
                    else {"status": "missing", "step": "",
                          "reason": "no value; the extractor recorded no reason", "inputs": ""}
                )
            records.append({**key_values, "column": column, **entry})
    return pd.DataFrame(records, columns=[*keys, *TABLE_COLUMNS])


def run_record(tables: list[pd.DataFrame], key_columns, columns) -> RunRecord:
    """Every shot's table, sorted by row key and the preset's column order."""
    keys = list(key_columns)
    if tables:
        table = pd.concat(tables, ignore_index=True)
    else:
        table = pd.DataFrame(columns=[*keys, *TABLE_COLUMNS])
    order = {name: index for index, name in enumerate(columns)}
    sort_keys = [key for key in keys if key in table.columns]
    table = (
        table.assign(_order=table["column"].map(order))
        .sort_values([*sort_keys, "_order"], kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    return RunRecord(table, code_revision())


# --- equilibrium_global ----------------------------------------------------------

#: column -> the time-slice leaf it is read from.
EQUILIBRIUM_LEAVES = {
    "ip_kA": "global_quantities.ip",
    "psi_axis_Wb": "global_quantities.psi_axis",
    "q_axis": "global_quantities.q_axis",
    "q_95": "global_quantities.q_95",
    "q_min": "global_quantities.q_min.value",
    "beta_pol": "global_quantities.beta_pol",
    "beta_tor": "global_quantities.beta_tor",
    "beta_normal": "global_quantities.beta_normal",
    "li_3": "global_quantities.li_3",
    "energy_mhd_J": "global_quantities.energy_mhd",
    "area_m2": "global_quantities.area",
    "volume_m3": "global_quantities.volume",
    "major_radius_m": "boundary.geometric_axis.r",
    "minor_radius_m": "boundary.minor_radius",
    "elongation": "boundary.elongation",
    "triangularity": "boundary.triangularity",
    "triangularity_upper": "boundary.triangularity_upper",
    "triangularity_lower": "boundary.triangularity_lower",
    "magnetic_axis_r_m": "global_quantities.magnetic_axis.r",
    "magnetic_axis_z_m": "global_quantities.magnetic_axis.z",
    "magnetic_axis_btor_T": "global_quantities.magnetic_axis.b_field_tor",
    "dia_flux_Wb": "constraints.diamagnetic_flux.reconstructed",
}

#: column -> the updater meant to (re)compute its leaf.
EQUILIBRIUM_INTENDED_STEP = {
    "q_min": "update_equilibrium_global_quantities_q_min",
    "beta_pol": "update_equilibrium_global_quantities_beta_li",
    "beta_tor": "update_equilibrium_global_quantities_beta_li",
    "beta_normal": "update_equilibrium_global_quantities_beta_li",
    "li_3": "update_equilibrium_global_quantities_beta_li",
    "energy_mhd_J": "update_equilibrium_stored_energy",
    "area_m2": "update_equilibrium_global_quantities_area",
    "volume_m3": "update_equilibrium_global_quantities_volume",
    "major_radius_m": "update_equilibrium_boundary",
    "minor_radius_m": "update_equilibrium_boundary",
    "elongation": "update_equilibrium_boundary",
    "triangularity": "update_equilibrium_boundary",
    "triangularity_upper": "update_equilibrium_boundary",
    "triangularity_lower": "update_equilibrium_boundary",
    "dia_flux_Wb": "update_equilibrium_constraints_diamagnetic_flux",
}

VIRIAL_STEP = "compute_virial_equilibrium_quantities_ods"

#: extractor-computed column -> the columns (or named inputs) it is formed from.
EQUILIBRIUM_COMPUTED_INPUTS = {
    "aspect_ratio": ("major_radius_m", "minor_radius_m"),
    "normalized_plasma_current": ("ip_kA", "minor_radius_m", "b0"),
    "beta_pol_circumference": ("profiles_1d.pressure", "profiles_1d.volume",
                               "global_quantities.length_pol", "ip_kA"),
    "beta_volume_B2": ("profiles_1d.pressure", "profiles_1d.volume", "ip_kA",
                       "boundary.outline", "magnetic energy"),
    "beta_normal_B2": ("beta_volume_B2", "minor_radius_m", "b0", "ip_kA"),
    "vacuum_b0_T": ("b0", "r0"),
}

#: columns whose value depends on the resolved reference radius / vacuum field.
#: vacuum_b0_T is b0 * R0 / 0.4: with b0 rebuilt as |f|/R0 the radius cancels, so it
#: depends on R0 only when the stored b0 was kept.
R0_DEPENDENT = frozenset({"beta_pol", "li_3"})
B0_DEPENDENT = frozenset({"beta_tor", "beta_normal", "beta_normal_B2",
                          "normalized_plasma_current", "vacuum_b0_T"})


def _leaf(eq_slice, path: str) -> float:
    if path not in eq_slice:  # tested first: item access creates a missing OMAS path
        return float("nan")
    try:
        return float(np.asarray(eq_slice[path], float).reshape(-1)[0])
    except Exception:  # noqa: BLE001 - an unreadable leaf counts as absent
        return float("nan")


def snapshot_equilibrium(recorder: RunRecorder | None, ods, step: str) -> None:
    """The tracked leaves of every slice, after ``step`` ran."""
    if recorder is None:
        return
    slices = ods["equilibrium.time_slice"]
    values = {
        (index, column): _leaf(slices[index], path)
        for index in range(len(slices))
        for column, path in EQUILIBRIUM_LEAVES.items()
    }
    recorder.snapshots.append((step, values))


def _same(a: float, b: float) -> bool:
    return (np.isnan(a) and np.isnan(b)) or a == b


def _writer(recorder: RunRecorder, key) -> str | None:
    """The step that last changed a leaf, if any did."""
    writer = None
    for (_, before), (step, after) in zip(recorder.snapshots, recorder.snapshots[1:]):
        if not _same(before.get(key, np.nan), after.get(key, np.nan)):
            writer = step
    return writer


def _failure(recorder: RunRecorder, step: str, row: int, *, shot_level: bool = False) -> str:
    if step in recorder.step_errors:
        return f"{step} raised {recorder.step_errors[step]}"
    said = recorder.warnings_for(step, row, shot_level=shot_level)
    return f"{step}: {'; '.join(said)}" if said else ""


def classify_equilibrium_rows(recorder: RunRecorder | None, rows: list[dict], time_sources: list[str]) -> None:
    """Fill the recorder with a status for every column of every slice row."""
    if recorder is None:
        return
    r0_source = str(recorder.facts.get("r0_source", "unknown"))
    b0_source = str(recorder.facts.get("b0_source", "unknown"))
    # the tf.r0 branch only substitutes something when the two radii differ
    r0_fallback = r0_source.startswith("tf.r0") and not np.isclose(
        float(recorder.facts.get("r0_stored", np.nan)), float(recorder.facts.get("r0_value", np.nan))
    )
    b0_fallback = bool(recorder.facts.get("b0_fallback", False))
    notes = recorder.facts.get("notes", {})

    def decisive(column: str) -> tuple[str, list[str]]:
        inputs, substitutes = [], []
        if (column in R0_DEPENDENT
                or (column in B0_DEPENDENT and column != "vacuum_b0_T" and not b0_fallback)
                or (column == "vacuum_b0_T" and b0_fallback)):
            inputs.append(f"R0: {r0_source}")
            if r0_fallback:
                substitutes.append("R0 taken from tf.r0")
        if column in B0_DEPENDENT:
            inputs.append(f"B0: {b0_source}")
            if b0_fallback:
                substitutes.append("B0 is the stored value, not rebuilt from profiles_1d.f")
        return "; ".join(inputs), substitutes

    def calculated(index: int, column: str, step: str, reason: str) -> None:
        inputs, substitutes = decisive(column)
        if substitutes:
            recorder.set(index, column, "fallback", step=step, inputs=inputs,
                         reason="; ".join(substitutes))
        else:
            recorder.set(index, column, "computed", step=step, inputs=inputs, reason=reason)

    for index, row in enumerate(rows):
        recorder.set(index, "time_s",
                     "fallback" if time_sources[index].startswith("no ") else "copied",
                     reason=time_sources[index])
        for column, path in EQUILIBRIUM_LEAVES.items():
            key = (index, column)
            first = recorder.snapshots[0][1].get(key, np.nan) if recorder.snapshots else np.nan
            writer = _writer(recorder, key)
            intended = EQUILIBRIUM_INTENDED_STEP.get(column, "")
            value = row.get(column, np.nan)
            if writer is not None and _present(value):
                calculated(index, column, writer, f"{path} written by {writer}")
                continue
            # The updaters skip a slice only when an input is missing, and log it; one
            # that ran clean on this slice and left the leaf as it was recomputed the
            # stored number. That is the normal case for a product processed at ingest.
            if (_present(value) and intended and not np.isnan(first)
                    and intended not in recorder.step_errors
                    and not recorder.warnings_for(intended, index)
                    and intended in {step for step, _ in recorder.snapshots}):
                calculated(index, column, intended, f"recomputed by {intended}; equals the stored {path}")
                continue
            if _present(value):
                why = _failure(recorder, intended, index) if intended else ""
                reason = f"stored {path}" + (
                    f"; kept because {why}" if why
                    else (f"; {intended} left it unchanged" if intended else "")
                )
                if column == "psi_axis_Wb":
                    reason += " (converted to Wb)"
                recorder.set(index, column, "copied", step="", reason=reason,
                             inputs=decisive(column)[0] if intended else "")
                continue
            if intended and intended in recorder.step_errors:
                recorder.set(index, column, "failed", step=intended,
                             reason=_failure(recorder, intended, index, shot_level=True))
                continue
            why = _failure(recorder, intended, index, shot_level=True) if intended else ""
            absent = f"{path} absent" if np.isnan(first) else f"{path} not finite"
            recorder.set(index, column, "missing", step=intended,
                         reason=absent + (f"; {why}" if why else ""))

        for column, needs in EQUILIBRIUM_COMPUTED_INPUTS.items():
            value = row.get(column, np.nan)
            if _present(value):
                calculated(index, column, "summary extractor", "formed from " + ", ".join(needs))
                continue
            gaps = [
                name for name in needs
                if name in row and not _present(row[name])
            ]
            if column in B0_DEPENDENT or column == "vacuum_b0_T":
                if not np.isfinite(_value_at_row(recorder, "b0", index)):
                    gaps.append("b0")
            reason = notes.get((index, column)) or (
                "inputs not finite: " + ", ".join(gaps) if gaps
                else "needs " + ", ".join(needs)
            )
            recorder.set(index, column, "missing", step="summary extractor", reason=reason,
                         inputs=decisive(column)[0])

        recorder.set(index, "vacuum_r0_m", "computed", step="summary extractor",
                     reason="the fixed reference radius vacuum_b0_T is scaled to")

        failed = recorder.step_errors.get(VIRIAL_STEP)
        for column in row:
            if not column.startswith("virial_"):
                continue
            if _present(row[column]):
                recorder.set(index, column, "computed", step=VIRIAL_STEP)
            elif failed is not None:
                recorder.set(index, column, "failed", step=VIRIAL_STEP,
                             reason=f"{VIRIAL_STEP} raised {failed}")
            else:
                said = recorder.warnings_for(VIRIAL_STEP, index, shot_level=True)
                recorder.set(index, column, "missing", step=VIRIAL_STEP,
                             reason="; ".join(said) if said
                             else "the virial calculation returned no value for this slice")


def _value_at_row(recorder: RunRecorder, name: str, index: int) -> float:
    values = recorder.facts.get(name)
    if values is None:
        return float("nan")
    try:
        array = np.asarray(values, float).reshape(-1)
        return float(array[index] if index < array.size else array[0]) if array.size else float("nan")
    except (TypeError, ValueError):
        return float("nan")
