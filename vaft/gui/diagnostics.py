"""The Routine Diagnostics workspace (#1348): processed diagnostics, by diagnostic.

The reader picks a *diagnostic* -- plasma current, flux loops, Thomson
scattering -- not a SQL field or an IDS path, and sees the canonical VAFT
plots of its processed data for every open shot.  Nothing here is a list of
diagnostics or plots:

* the diagnostics, their names, categories, IDS paths and status come from
  the diagnostic registry (:func:`vaft.machine_mapping.registry.load_diagnostic_registry`,
  the same source as the documentation's diagnostics table);
* a diagnostic's plots are the discovery records whose required data lies
  under the diagnostic's ``ids_path`` (:func:`diagnostic_plots`), so a new
  plot of a diagnostic appears here without touching the GUI.

The plots are drawn by the plot explorer (:class:`vaft.gui.app.BrowserApp`)
narrowed to the chosen diagnostic, so several shots compare on one plot,
controls and figure options work as in **Plots**, and the shared selection
carries the open shots between workspaces.

Raw-versus-processed comparison needs the raw DAQ source of each registry
diagnostic as an API; until one exists (see the card), the source shown is
what the registry records.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ._require import require_panel
from .workspaces import PlotWorkspace

__all__ = ["DiagnosticsWorkspace", "describe_diagnostic", "diagnostic_plots", "diagnostics_by_category"]


def _path(value: Any) -> str:
    """An IDS path without array markers: ``magnetics.flux_loop[:].flux`` -> ``magnetics.flux_loop.flux``."""
    text = str(value or "")
    for marker in ("[:]", "[]", ".:", ":"):
        text = text.replace(marker, "")
    return text.strip(".")


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + ".")


def diagnostic_plots(record: Mapping[str, Any], plots: Iterable[Any]) -> list[str]:
    """The plots (discovery records) whose required data is the diagnostic's.

    A plot belongs to a diagnostic when one of its required paths lies under
    the diagnostic's registry ``ids_path`` -- the join the data states, not
    a list kept here.
    """
    root = _path(record.get("ids_path"))
    if not root:
        return []
    return [
        plot.name for plot in plots
        if any(_under(_path(path), root) for path in (getattr(plot, "required_paths", ()) or ()))
    ]


def diagnostics_by_category(
    registry: Mapping[str, Mapping[str, Any]], plots: Iterable[Any],
) -> dict[str, dict[str, str]]:
    """``{category: {display name: registry id}}`` for the diagnostics that have plots."""
    plots = list(plots)
    groups: dict[str, dict[str, str]] = {}
    for key, record in registry.items():
        if not diagnostic_plots(record, plots):
            continue
        name = str(record.get("name") or key)
        options = groups.setdefault(str(record.get("category") or "Other"), {})
        options[name if name not in options else f"{name} ({key})"] = key
    return groups


def _listing(values: Any) -> str:
    if isinstance(values, Mapping):
        values = [f"{key}: {value}" for key, value in values.items()]
    values = [str(value) for value in (values or ()) if str(value)]
    return ", ".join(values) if values else "none"


def describe_diagnostic(key: str, record: Mapping[str, Any], names: Iterable[str]) -> str:
    """The registry record as Markdown, with the plots the workspace offers."""
    lines = [f"**{record.get('name') or key}** (`{key}`)"]
    facts = [
        f"Processed data: `{record.get('ids_path') or record.get('ids')}`",
        f"Family: {record.get('family', 'unknown')} -- {record.get('category', '')}",
        f"Availability: {record.get('availability', 'unknown')}; mapping {record.get('mapping_status', 'unknown')}",
    ]
    quantities = record.get("quantities") or {}
    if isinstance(quantities, Mapping):
        for kind in ("measured", "derived", "static"):
            if quantities.get(kind):
                facts.append(f"{kind.capitalize()}: {_listing(quantities[kind])}")
    source = record.get("source") or {}
    if isinstance(source, Mapping) and source:
        facts.append(f"Source: {_listing(source)}")
    facts.append(f"Plots: {', '.join(f'`{name}`' for name in names) or 'none'}")
    lines.append("  \n".join(facts))
    lines.append(
        "Raw source fields and raw-versus-processed comparison are not exposed by an API "
        "yet; this card shows what the diagnostic registry records."
    )
    return "\n\n".join(lines)


class DiagnosticsWorkspace(PlotWorkspace):
    """The plot explorer narrowed to one diagnostic chosen from the registry."""

    name = "diagnostics"
    title = "Diagnostics"

    def __init__(self, shell: Any, app: Any = None, registry: Mapping[str, Any] | None = None) -> None:
        pn = require_panel()
        super().__init__(shell, app)
        if registry is None:
            from vaft.machine_mapping.registry import load_diagnostic_registry

            registry = load_diagnostic_registry()
        self.registry = dict(registry)
        from vaft.plot import available_plots

        # Which diagnostics have plots at all is a registry fact; which of
        # those plots the open shots can draw is the explorer's catalog.
        self._registered_plots = list(available_plots(status=None))
        groups = diagnostics_by_category(self.registry, self._registered_plots)
        first = next((key for options in groups.values() for key in options.values()), None)
        self.diagnostic = pn.widgets.Select(label="Diagnostic", groups=groups or {"": {}}, value=first)
        self.card = pn.pane.Markdown("", sizing_mode="stretch_width", styles={"font-size": "0.9em"})
        self.diagnostic.param.watch(lambda event: self.choose(event.new), "value")
        if first is not None:
            self.choose(first)

    def plots_of(self, key: str) -> list[str]:
        """The registered plots of diagnostic ``key``."""
        return diagnostic_plots(self.registry[key], self._registered_plots)

    def choose(self, key: str | None) -> None:
        """Narrow the explorer to diagnostic ``key``."""
        if not key:
            return
        names = self.plots_of(key)
        self.card.object = describe_diagnostic(key, self.registry[key], names)
        wanted = set(names)
        self.app.set_plot_filter(lambda record: record.name in wanted)

    def sidebar(self) -> list[Any]:
        pn = require_panel()
        return [pn.pane.Markdown("### Diagnostic"), self.diagnostic, pn.layout.Divider(), *self.app.sidebar()]

    def main(self) -> list[Any]:
        pn = require_panel()
        return [pn.Column(pn.Card(self.card, title="About this diagnostic", collapsed=False,
                                  sizing_mode="stretch_width"), *self.app.main(), sizing_mode="stretch_width")]
