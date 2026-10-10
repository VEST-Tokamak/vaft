"""The Equilibrium workspace (#1352): inspect, constraints, time, quality.

Inspection-first, after Tutorial 03: one selected time slice, shared by every
plot the reader opens, seen four ways --

* **Inspect** -- the 2-D state (flux map, boundary) and 1-D profiles;
* **Constraints & fit** -- what entered the reconstruction and how well it
  was matched (constraints, coverage, weights, residuals, convergence);
* **Time evolution** -- global quantities across the discharge;
* **Quality** -- the fit-quality plots and table, and the verdicts of
  :func:`vaft.validation.equilibrium.validate_equilibrium` for the slice,
  shown as the validation layer states them (#892 will refine the axes).

The plots are the equilibrium records of plot discovery; a mode is only a
presentation rule over a record's ``view`` and ``quantity``
(:func:`mode_of`), and a plot no rule claims is shown under *Inspect*, so a
new equilibrium plot always appears somewhere.  Several open shots compare
on each plot, as in **Plots**.  Nothing here edits or reruns a
reconstruction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._require import require_panel
from .workspaces import PlotWorkspace

__all__ = ["MODES", "EquilibriumWorkspace", "is_equilibrium", "mode_of", "quality_markdown", "quality_rows"]

#: The workspace's modes, in order: key -> label.
MODES = {
    "inspect": "Inspect",
    "constraints": "Constraints & fit",
    "time": "Time evolution",
    "quality": "Quality",
}

#: Quantities that describe the reconstruction's inputs and their match.
_CONSTRAINT_QUANTITIES = frozenset({
    "constraints", "constraint_coverage", "constraint_weights", "residuals", "convergence",
    "pressure_weight_scan",
})
#: Quantities that are quality evidence.
_QUALITY_QUANTITIES = frozenset({"fit_quality", "verification"})


def is_equilibrium(record: Any) -> bool:
    """Whether a discovery record is a plot of the equilibrium."""
    return getattr(record, "domain", None) == "equilibrium" or getattr(record, "subject", None) == "equilibrium"


def mode_of(record: Any) -> str:
    """The mode a plot is shown under (``"inspect"`` when no rule claims it)."""
    view = getattr(record, "view", "") or ""
    quantity = getattr(record, "quantity", "") or ""
    if quantity in _QUALITY_QUANTITIES:
        return "quality"
    if quantity in _CONSTRAINT_QUANTITIES:
        return "constraints"
    if view == "time" or quantity == "histories":
        return "time"
    return "inspect"


def _first_reason(entry: Mapping[str, Any]) -> str:
    if entry.get("reason"):
        return str(entry["reason"])
    for item in entry.get("slices") or ():
        if isinstance(item, Mapping):
            if item.get("reason"):
                return str(item["reason"])
            issues = item.get("issues")
            if issues:
                return "; ".join(str(issue) for issue in issues)
    return ""


def _status(value: Any) -> str:
    return str(getattr(value, "value", value) or "unknown")


def quality_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    """One row per check of a validation report: category, check, status, reason."""
    from vaft.validation.equilibrium import EQUILIBRIUM_CATEGORIES

    rows = []
    for category in EQUILIBRIUM_CATEGORIES:
        checks = report.get(category)
        if not isinstance(checks, Mapping):
            continue
        for name, entry in checks.items():
            if isinstance(entry, Mapping):
                rows.append({
                    "category": category, "check": str(name),
                    "status": _status(entry.get("status")), "reason": _first_reason(entry),
                })
    return rows


def quality_markdown(label: str, report: Mapping[str, Any]) -> str:
    """A validation report as Markdown: the summary per category, then every check."""
    import math

    import numpy as np

    summary = report.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    raw_times = report.get("time")
    raw_times = [] if raw_times is None else np.ravel(np.asarray(raw_times, dtype=float)).tolist()
    times = ", ".join(f"{t * 1e3:.1f} ms" for t in raw_times if math.isfinite(t))
    lines = [f"**{label}** -- overall **{_status(report.get('status'))}**" + (f" at {times}" if times else "")]
    lines.append(" · ".join(f"{category}: **{_status(status)}**" for category, status in summary.items()))
    table = ["| Category | Check | Status | Reason |", "| --- | --- | --- | --- |"]
    for row in quality_rows(report):
        reason = row["reason"].replace("|", "\\|").replace("\n", " ")
        table.append(f"| {row['category']} | {row['check']} | {row['status']} | {reason[:200]} |")
    lines.append("\n".join(table))
    return "\n\n".join(lines)


#: The IDS the validation reads, fetched for database shots before checking.
_VALIDATION_IDS = ("equilibrium", "magnetics", "pf_active", "core_profiles", "thomson_scattering")


class EquilibriumWorkspace(PlotWorkspace):
    """The plot explorer over equilibrium plots, by mode, sharing one time slice."""

    name = "equilibrium"
    title = "Equilibrium"

    def __init__(self, shell: Any, app: Any = None) -> None:
        pn = require_panel()
        super().__init__(shell, app)
        self.mode = pn.widgets.RadioButtonGroup(
            options={label: key for key, label in MODES.items()}, value="inspect",
            orientation="vertical", sizing_mode="stretch_width",
        )
        self.check_slice = pn.widgets.Button(label="Check this slice", color="primary")
        self.check_all = pn.widgets.Button(label="Check all slices")
        self.quality = pn.pane.Markdown("", sizing_mode="stretch_width", styles={"font-size": "0.85em"})
        self.quality_card = pn.Card(
            pn.Row(self.check_slice, self.check_all), self.quality,
            title="Validation verdicts", collapsed=False, visible=False, sizing_mode="stretch_width",
        )
        #: The slice the reader chose, carried to every plot that has one.
        self.time_slice: int | None = None
        #: ((sources, plot), slice) last seen on screen, to tell the reader's
        #: move from a new plot opening on its default.
        self._seen: tuple[Any, Any] = ((), None)
        self.mode.param.watch(lambda event: self.set_mode(event.new), "value")
        self.check_slice.on_click(lambda _event: self.run_checks(all_slices=False))
        self.check_all.on_click(lambda _event: self.run_checks(all_slices=True))
        self.app.on_change.append(self._carry_slice)
        self.set_mode("inspect")

    # -- modes and the shared slice --------------------------------------------------
    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
        if self.mode.value != mode:
            self.mode.value = mode  # the watcher comes back here
            return
        self.quality_card.visible = mode == "quality"
        self.app.set_plot_filter(lambda record: is_equilibrium(record) and mode_of(record) == mode)

    def _carry_slice(self) -> None:
        """Keep one time slice across plots: remember it, and apply it to a new plot.

        The slice is remembered only when the reader moves it on the plot
        already on screen; a new plot -- another one, the same one again, or
        the same one over other shots -- opens on the remembered slice.
        """
        session = self.app.session
        key = (session.sources, session.plot)
        state = session.state
        if state is None or "time_slice" not in state.values:
            self._seen = (key, None)
            return
        current = state["time_slice"]
        seen_key, seen_slice = self._seen
        self._seen = (key, current)
        if seen_key == key and seen_slice is not None:
            if current != seen_slice:
                self.time_slice = current  # the reader moved it
            return
        if self.time_slice is None:
            self.time_slice = current
            return
        options = list(getattr(state.spec("time_slice"), "options", ()) or ())
        if current != self.time_slice and self.time_slice in options:
            state.set("time_slice", self.time_slice)  # redraws; comes back with equal values

    # -- quality evidence ------------------------------------------------------------------
    def run_checks(self, *, all_slices: bool = False) -> list[dict[str, Any]]:
        """Validate the open equilibria (this slice, or every slice); returns the reports."""
        from vaft.validation.equilibrium import validate_equilibrium

        session = self.app.session
        if session.ods is None:
            self.shell.report("open a shot with an equilibrium first", where=self.title)
            return []
        # The slice on screen, else the one remembered.
        state = session.state
        on_screen = state["time_slice"] if state is not None and "time_slice" in state.values else None
        slice_ = None if all_slices else (on_screen if on_screen is not None else self.time_slice)
        self.check_slice.disabled = self.check_all.disabled = True
        self.quality.object = "Checking ..."
        try:
            if session.database:
                try:
                    session.load_ids(_VALIDATION_IDS)  # only what each shot stores is fetched
                finally:
                    self.app._update_status()
            data = session.ods if isinstance(session.ods, list) else [session.ods]
            reports, parts = [], []
            for source, ods in zip(session.sources, data):
                # Each shot on its own: one that cannot be checked does not
                # hide the verdicts of the others.
                try:
                    count = len(ods["equilibrium.time_slice"]) if "equilibrium.time_slice" in ods else 0
                    if slice_ is not None and not 0 <= int(slice_) < count:
                        parts.append(f"**{source.label}** -- not checked: slice {slice_} is not one of its {count}")
                        continue
                    report = validate_equilibrium(ods, time_slice=slice_)
                except Exception as error:  # this shot only
                    parts.append(f"**{source.label}** -- not checked: {str(error).splitlines()[0] if str(error) else type(error).__name__}")
                    continue
                reports.append(report)
                parts.append(quality_markdown(source.label, report))
            self.quality.object = "\n\n---\n\n".join(parts)
        except Exception as error:
            self.quality.object = ""
            self.shell.report(error, where=f"{self.title} checks")
            return []
        finally:
            self.check_slice.disabled = self.check_all.disabled = False
        return reports

    # -- layout ---------------------------------------------------------------------------
    def sidebar(self) -> list[Any]:
        pn = require_panel()
        return [pn.pane.Markdown("### Mode"), self.mode, pn.layout.Divider(), *self.app.sidebar()]

    def main(self) -> list[Any]:
        pn = require_panel()
        return [pn.Column(self.quality_card, *self.app.main(), sizing_mode="stretch_width")]
