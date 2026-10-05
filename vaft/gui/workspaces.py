"""The workspaces ``vaft gui`` ships with (#1174).

* **Plots** -- the plot explorer of #1086: sources, the discovered plot
  catalog, controls, figure options, export.  It publishes what it has open
  and the selected time to the shared selection, and opens what another
  workspace selects.
* **Database** -- the database sources a shot can be read from, the HSDS
  credential configuration h5pyd will use (never its secrets), a connection
  check, and opening database shots in the plot explorer.

Both are thin: the explorer is :class:`vaft.gui.app.BrowserApp`, and the
database workspace calls :mod:`vaft.database.sources`,
:mod:`vaft.database.hscfg` and :func:`vaft.database.utils.is_connect`.
"""

from __future__ import annotations

import os
from typing import Any

from ._require import require_panel
from .shell import WORKSPACES, Shell, WorkspaceSpec
from .state import Source

__all__ = ["DatabaseWorkspace", "PlotWorkspace", "credential_summary"]


class PlotWorkspace:
    """The plot explorer, bound to the shared selection."""

    def __init__(self, shell: Shell, app: Any = None) -> None:
        from .app import BrowserApp

        self.shell = shell
        self.app = app if app is not None else BrowserApp()
        self._pending: tuple[Source, ...] | None = None
        chosen = shell.selection.value.sources
        if chosen and chosen != self.app.session.sources:
            self._pending = chosen  # chosen in another workspace before this one was built
        self.app.on_change.append(self._publish)
        self._unsubscribe = shell.selection.subscribe(self._on_selection)
        if self.app.session.sources and self._pending is None:
            self._publish()  # an explorer handed over with a source already open

    # -- the shared selection ------------------------------------------------------
    def _publish(self) -> None:
        """What the explorer has open and the time it shows, for every workspace."""
        sources = self.app.session.sources
        changes: dict[str, Any] = {"sources": sources, "time": self.app.selected_time()}
        if sources and all(source.kind == "shot" for source in sources):
            changes["namespace"] = sources[0].namespace
        self.shell.selection.update(origin=self, **changes)

    def _on_selection(self, selection: Any, origin: Any) -> None:
        if origin is self or not selection.sources:
            return
        if selection.sources == self.app.session.sources:
            self._pending = None  # back to what is open: nothing left to load
            return
        self.request(selection.sources)

    def request(self, sources: Any) -> None:
        """Open ``sources`` when the reader comes here (now, if this workspace is shown).

        Loading a database shot is slow and the reader may still be choosing,
        so a workspace in the background only remembers the request.
        """
        self._pending = tuple(sources)
        if self.shell.active == "plots":
            self.activate()

    def activate(self) -> None:
        pending, self._pending = self._pending, None
        if pending:
            self.app.choose(pending)
            if not self.app.load(list(pending)):
                self.shell.report(self.app.alert.object or "could not open the selection", where="Plots")
                # The selection says what is open, and the failed request is not.
                self._publish()

    # -- layout --------------------------------------------------------------------
    def sidebar(self) -> list[Any]:
        return self.app.sidebar()

    def main(self) -> list[Any]:
        return self.app.main()

    def close(self) -> None:
        self._unsubscribe()
        self.app.close()


def credential_summary(path: Any = None, environment: dict[str, str] | None = None) -> dict[str, str]:
    """What h5pyd will connect with, secrets reduced to configured/missing.

    Environment variables (``HS_ENDPOINT``, ``HS_USERNAME``, ``HS_PASSWORD``,
    ``HS_API_KEY``) override the file, as they do in h5pyd.  Secret values
    are never returned.
    """
    from pathlib import Path

    from vaft.database import hscfg

    environment = dict(os.environ if environment is None else environment)
    path = Path(path) if path is not None else hscfg.active_path()
    values = hscfg.read_values(path) if path.is_file() else {}
    summary = {"file": str(path) if path.is_file() else f"{path} (missing)"}
    for key in hscfg.HSCFG_KEYS:
        value = environment.get(key.upper()) or values.get(key) or ""
        origin = " (environment)" if environment.get(key.upper()) else ""
        if key in hscfg.SECRET_KEYS:
            summary[key] = ("configured" + origin) if value else "not set"
        else:
            summary[key] = (value + origin) if value else "not set"
    if path.is_file() and hscfg.insecure_permissions(path):
        summary["warning"] = f"{path} is readable by other users; run `chmod 600 {path}`"
    return summary


class DatabaseWorkspace:
    """Database sources, credential configuration, connection, opening shots."""

    def __init__(self, shell: Shell) -> None:
        pn = require_panel()
        from vaft.database import sources as catalog

        self.shell = shell
        self._sources = catalog.known_sources()
        default = catalog.resolve(None)
        current = shell.selection.value.namespace or default
        names = [source.name for source in self._sources]
        self.namespace = pn.widgets.Select(
            label="Namespace", options=names, value=current if current in names else default,
        )
        self.shots = pn.widgets.TextInput(label="Shots", placeholder="39915, 41524")
        self.open_button = pn.widgets.Button(label="Open in Plots", color="primary")
        self.check_button = pn.widgets.Button(label="Test connection")
        self.connection = pn.pane.Markdown("")
        self.credentials = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.table = pn.pane.Markdown(self._source_table(), sizing_mode="stretch_width")
        self.detail = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.namespace.param.watch(self._on_namespace, "value")
        self.open_button.on_click(lambda _event: self.open_shots())
        self.check_button.on_click(lambda _event: self.check())
        self._describe(self.namespace.value)
        self.refresh_credentials()
        self._unsubscribe = shell.selection.subscribe(self._follow)

    def _follow(self, selection: Any, origin: Any) -> None:
        # The namespace of shots opened elsewhere (the explorer's own source
        # picker) is the one shown here.
        if origin is not self and selection.namespace in self.namespace.options:
            self.namespace.value = selection.namespace

    def close(self) -> None:
        self._unsubscribe()

    def _source_table(self) -> str:
        rows = ["| Namespace | Holds | Writable | Coverage |", "| --- | --- | --- | --- |"]
        for source in self._sources:
            purpose = " ".join(str(source.purpose).split())
            rows.append(
                f"| `{source.name}` | {purpose} | {'yes' if source.writable else 'read-only'} "
                f"| {'sparse' if source.sparse else 'every shot'} |"
            )
        return "\n".join(rows)

    def _describe(self, name: str) -> None:
        source = next((source for source in self._sources if source.name == name), None)
        if source is None:
            self.detail.object = ""
            return
        lines = [f"**{source.name}**: {' '.join(str(source.purpose).split())}"]
        if getattr(source, "parent", None):
            lines.append(f"Derived from `{source.parent}`.")
        if getattr(source, "sparse", False):
            lines.append("Sparse: a shot missing here has no product, it is not a missing shot.")
        self.detail.object = "\n\n".join(lines)

    def _on_namespace(self, event: Any) -> None:
        self._describe(event.new)
        self.shell.selection.update(origin=self, namespace=event.new)

    def refresh_credentials(self) -> dict[str, str]:
        summary = credential_summary()
        lines = [
            f"Configuration: `{summary['file']}`",
            f"Endpoint: {summary['hs_endpoint']}",
            f"Username: {summary['hs_username']}",
            f"Password: {summary['hs_password']}",
            f"API key: {summary['hs_api_key']}",
        ]
        if "warning" in summary:
            lines.append(f"**Warning:** {summary['warning']}")
        lines.append("Change it with `vaft hsds configure` in a terminal; the GUI never shows or stores a secret.")
        self.credentials.object = "\n\n".join(lines)
        return summary

    def check(self) -> str:
        state = self.shell.check_connection()
        self.connection.object = f"Database server: **{state}**"
        return state

    def open_shots(self) -> bool:
        """Open the typed shots from the chosen namespace in the plot explorer."""
        from .app import parse_shots

        try:
            shots = parse_shots(self.shots.value)
        except ValueError as error:
            self.shell.report(error, where="Database")
            return False
        if not shots:
            self.shell.report("type one or more shot numbers", where="Database")
            return False
        self.shell.clear()
        namespace = self.namespace.value
        wanted = tuple(Source("shot", shot, namespace) for shot in shots)
        plots = self.shell.workspace("plots")
        self.shell.selection.update(origin=self, namespace=namespace, sources=wanted, time=None)
        # Asked for directly as well: the same shots again (a retry after a
        # failed load) change no selection and so notify nobody.
        if wanted != plots.app.session.sources:
            plots.request(wanted)
        self.shell.show("plots")
        return True

    def sidebar(self) -> list[Any]:
        pn = require_panel()
        return [
            pn.pane.Markdown("### Open database shots"), self.namespace, self.shots, self.open_button,
            pn.layout.Divider(),
            pn.pane.Markdown("### Connection"), self.check_button, self.connection,
        ]

    def main(self) -> list[Any]:
        pn = require_panel()
        return [pn.Column(
            pn.pane.Markdown("## Database sources"), self.detail, self.table,
            pn.pane.Markdown("## Credentials"), self.credentials,
            sizing_mode="stretch_width",
        )]


def _register() -> None:
    for spec in (
        WorkspaceSpec(
            "plots", "Plots", PlotWorkspace,
            "Open samples, files or database shots and explore every plot they support.", order=10,
        ),
        WorkspaceSpec(
            "database", "Database", DatabaseWorkspace,
            "Database sources, the HSDS configuration in use, and opening database shots.", order=20,
        ),
    ):
        if spec.name not in WORKSPACES:
            WORKSPACES.register(spec)


_register()
