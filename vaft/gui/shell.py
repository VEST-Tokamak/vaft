"""The application shell: workspaces, shared selection, status (#1174).

``vaft gui`` serves one :class:`Shell` per browser session.  The shell owns
what every workspace shares -- the :class:`~vaft.gui.selection.SelectionState`,
the navigation between workspaces, the status line and the error surface --
and nothing scientific: a workspace draws and loads through the public VAFT
APIs itself.

Workspaces are registered, not hard-coded, so a later workspace (the domain
workspaces of roadmap #1359) plugs in with one call::

    from vaft.gui.shell import register_workspace

    register_workspace("equilibrium", "Equilibrium", EquilibriumWorkspace, order=30)

A workspace is any object with ``sidebar()`` and ``main()`` returning lists
of Panel objects; ``activate()`` (called each time it is shown) and
``close()`` (when the browser session ends) are optional.  Its factory
receives the shell, through which it reaches the shared selection
(``shell.selection``), reports errors (``shell.report``) and moves the
reader to another workspace (``shell.show``).  A workspace is built the
first time it is shown, so an unused one costs nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from ._require import require_panel
from .selection import Selection, SelectionState

__all__ = [
    "Shell",
    "WORKSPACES",
    "WorkspaceRegistry",
    "WorkspaceSpec",
    "register_workspace",
]


@dataclass(frozen=True)
class WorkspaceSpec:
    """A registered workspace: its key, its title in the navigation, how to build it."""

    name: str
    title: str
    factory: Callable[["Shell"], Any]
    description: str = ""
    #: Position in the navigation, lowest first; ties keep registration order.
    order: int = 100


class WorkspaceRegistry:
    """The workspaces a shell offers, in navigation order."""

    def __init__(self) -> None:
        self._specs: dict[str, WorkspaceSpec] = {}

    def register(self, spec: WorkspaceSpec, *, replace: bool = False) -> WorkspaceSpec:
        if not spec.name or not spec.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"a workspace name is a plain key such as 'plots'; got {spec.name!r}")
        if spec.name in self._specs and not replace:
            raise ValueError(f"a workspace named {spec.name!r} is already registered")
        self._specs[spec.name] = spec
        return spec

    def specs(self) -> list[WorkspaceSpec]:
        order = {name: index for index, name in enumerate(self._specs)}
        return sorted(self._specs.values(), key=lambda spec: (spec.order, order[spec.name]))

    def names(self) -> list[str]:
        return [spec.name for spec in self.specs()]

    def get(self, name: str) -> WorkspaceSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(
                f"no workspace named {name!r}; workspaces: {', '.join(self.names()) or 'none'}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._specs


#: The workspaces ``vaft gui`` offers.  The built-in ones are added by
#: :mod:`vaft.gui.workspaces` when the shell is first built.
WORKSPACES = WorkspaceRegistry()


def register_workspace(
    name: str,
    title: str,
    factory: Callable[["Shell"], Any],
    *,
    description: str = "",
    order: int = 100,
    replace: bool = False,
) -> WorkspaceSpec:
    """Add a workspace to :data:`WORKSPACES`; returns its spec."""
    return WORKSPACES.register(
        WorkspaceSpec(name, title, factory, description, order), replace=replace,
    )


def _first_line(error: BaseException) -> str:
    text = str(error).strip() or type(error).__name__
    return text.splitlines()[0]


#: What the status line says before anyone asked the database server.
NOT_CHECKED = "not checked"
#: Seconds the connection check waits for the database server.
CONNECT_TIMEOUT = 10.0


class Shell:
    """One browser session's page: navigation, the active workspace, status.

    ``initial`` is the workspace shown first (the registry's first when
    ``None``); ``factories`` replace a registered workspace's factory for
    this shell only (``vaft gui`` hands the plot explorer its first source
    this way).  The shell checks the database connection only when asked
    (:meth:`check_connection`): an unreachable server must not hold the
    page's first paint.
    """

    def __init__(
        self,
        registry: WorkspaceRegistry | None = None,
        *,
        initial: str | None = None,
        selection: SelectionState | None = None,
        factories: Mapping[str, Callable[["Shell"], Any]] | None = None,
    ) -> None:
        pn = require_panel()
        if registry is None:
            from . import workspaces  # noqa: F401 - registers the built-in workspaces

            registry = WORKSPACES
        self.registry = registry
        specs = registry.specs()
        if not specs:
            raise ValueError("the shell needs at least one registered workspace")
        initial = initial or specs[0].name
        registry.get(initial)  # an unknown name is refused here, with the choices
        self.selection = selection or SelectionState()
        self.workspaces: dict[str, Any] = {}
        self._factories = dict(factories or {})
        for name in self._factories:
            registry.get(name)
        self.connection = NOT_CHECKED
        self.nav = pn.widgets.RadioButtonGroup(
            options={spec.title: spec.name for spec in specs}, value=initial,
            orientation="vertical", sizing_mode="stretch_width",
        )
        self.hint = pn.pane.Markdown("", margin=(0, 10), styles={"font-size": "0.85em"})
        self.sidebar_box = pn.Column(sizing_mode="stretch_width")
        self.main_box = pn.Column(sizing_mode="stretch_width")
        self.alert = pn.pane.Alert("", alert_type="danger", visible=False, sizing_mode="stretch_width")
        self.status = pn.pane.Markdown("", margin=(0, 10), styles={"font-size": "0.85em"})
        self.active: str | None = None
        self.nav.param.watch(lambda event: self.show(event.new), "value")
        self.selection.subscribe(lambda _selection, _origin: self._update_status())
        self.show(initial)

    # -- workspaces -------------------------------------------------------------
    def workspace(self, name: str) -> Any:
        """The workspace ``name``, built on first use."""
        if name not in self.workspaces:
            factory = self._factories.get(name) or self.registry.get(name).factory
            self.workspaces[name] = factory(self)
        return self.workspaces[name]

    def show(self, name: str) -> Any:
        """Make ``name`` the active workspace; returns it."""
        spec = self.registry.get(name)
        if self.nav.value != name:
            self.nav.value = name  # the watcher comes back here
            return self.workspaces.get(name)
        if self.active == name:
            return self.workspaces.get(name)
        try:
            workspace = self.workspace(name)
            sidebar, main = list(workspace.sidebar()), list(workspace.main())
        except Exception as error:
            # A workspace that cannot be built or laid out leaves the previous
            # one on screen.
            self.report(error, where=spec.title)
            if self.active is not None:
                self.nav.value = self.active
            return None
        self.active = name
        self.hint.object = spec.description
        self.sidebar_box.objects = sidebar
        self.main_box.objects = main
        activate = getattr(workspace, "activate", None)
        if activate is not None:
            try:
                activate()
            except Exception as error:
                self.report(error, where=spec.title)
        self._update_status()
        return workspace

    # -- shared surfaces -----------------------------------------------------------
    def report(self, error: BaseException | str, *, where: str | None = None) -> None:
        """Show an error above the workspace (and as a toast when Panel offers one)."""
        text = error if isinstance(error, str) else _first_line(error)
        if where:
            text = f"{where}: {text}"
        self.alert.object = text
        self.alert.visible = True
        pn = require_panel()
        notifications = getattr(pn.state, "notifications", None)
        if notifications is not None:
            notifications.error(text, duration=6000)

    def clear(self) -> None:
        self.alert.visible = False
        self.alert.object = ""

    def check_connection(self, timeout: float | None = None) -> str:
        """Ask the database server whether it is ready; returns the state shown.

        The question runs in a worker thread and is given up after
        ``timeout`` seconds (:data:`CONNECT_TIMEOUT`): one session waiting on
        an unreachable server must not hold every other session's page.
        """
        import logging
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as Timeout

        from vaft.database import utils

        timeout = CONNECT_TIMEOUT if timeout is None else timeout
        root = logging.getLogger()
        level = root.level  # is_connect lowers the root logger; the server's is kept
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            ready = pool.submit(utils.is_connect).result(timeout=timeout)
            self.connection = "ready" if ready else "not ready"
        except Timeout:
            self.connection = f"no answer within {timeout:g} s"
        except Exception as error:  # an unreachable or misconfigured server
            self.connection = f"unreachable ({_first_line(error)})"
        finally:
            pool.shutdown(wait=False)
            root.setLevel(level)
        self._update_status()
        return self.connection

    def _update_status(self) -> None:
        selection: Selection = self.selection.value
        parts = [f"**Open:** {selection.label}"]
        if selection.time:
            parts.append(f"**Time:** {selection.time}")
        parts.append(f"**Namespace:** {selection.namespace or 'default'}")
        parts.append(f"**Database:** {self.connection}")
        self.status.object = " · ".join(parts)

    # -- layout ---------------------------------------------------------------------
    def sidebar(self) -> list[Any]:
        pn = require_panel()
        return [
            pn.pane.Markdown("### Workspace"), self.nav, self.hint, pn.layout.Divider(), self.sidebar_box,
        ]

    def main(self) -> list[Any]:
        pn = require_panel()
        return [pn.Column(self.status, self.alert, self.main_box, sizing_mode="stretch_width")]

    def view(self) -> Any:
        """The page served by ``vaft gui``."""
        pn = require_panel()
        return pn.template.FastListTemplate(
            title="VAFT", sidebar=self.sidebar(), main=self.main(), sidebar_width=360,
        )

    def close(self) -> None:
        """Close every built workspace, even when one of them fails to."""
        workspaces, self.workspaces = list(self.workspaces.values()), {}
        for workspace in workspaces:
            close = getattr(workspace, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception:  # pragma: no cover - the others still get closed
                import logging

                logging.getLogger(__name__).exception("closing a GUI workspace failed")
