"""Where a figure is being shown, and what interaction that allows.

An interactive plot has one scientific contract (:mod:`vaft.plot.navigation`)
but several places it can be shown, and the control that works in one is
inert in another: a Matplotlib slider needs a live canvas, which a GUI window
or the ``ipympl`` notebook backend gives and the inline notebook backend does
not; an ipywidgets slider needs a kernel and a front end that renders widgets
-- Jupyter Notebook, JupyterLab, or VS Code's notebook editor -- and under
the inline backend must redraw the figure into an output area itself.

:func:`detect_environment` reads those facts from the running process and
:func:`default_interaction_backend` turns them into the backend a renderer
should use when the caller says ``backend="auto"``.  Nothing here draws.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = [
    "Environment",
    "ENVIRONMENT_KINDS",
    "detect_environment",
    "default_interaction_backend",
    "use_non_interactive_backend",
]

#: ``terminal`` -- plain Python; ``ipython`` -- an IPython terminal shell;
#: ``jupyter`` -- a kernel driven by Jupyter Notebook or JupyterLab;
#: ``vscode`` -- a kernel driven by the VS Code Jupyter extension.
ENVIRONMENT_KINDS = ("terminal", "ipython", "jupyter", "vscode")

#: Matplotlib backends whose canvas updates in place after ``draw_idle``.
_LIVE_BACKENDS = {
    "macosx", "qtagg", "qt5agg", "qt6agg", "qtcairo", "tkagg", "tkcairo",
    "gtk3agg", "gtk4agg", "gtk3cairo", "gtk4cairo", "wxagg", "wxcairo",
    "webagg", "nbagg",
}


@dataclass(frozen=True)
class Environment:
    """What the running process can show.

    ``kind`` is one of :data:`ENVIRONMENT_KINDS`; ``backend`` the Matplotlib
    backend in use; ``live_figures`` whether a drawn figure updates in place
    (a GUI window or ``ipympl``); ``widgets`` whether ipywidgets can be
    displayed here (a kernel and the package).
    """

    kind: str
    backend: str
    live_figures: bool
    widgets: bool

    @property
    def in_kernel(self) -> bool:
        return self.kind in ("jupyter", "vscode")


def _shell_class_name() -> str:
    try:
        from IPython import get_ipython
    except ImportError:  # pragma: no cover - IPython is a dependency
        return ""
    shell = get_ipython()
    return type(shell).__name__ if shell is not None else ""


def _under_vscode() -> bool:
    return any(key.startswith("VSCODE_") for key in os.environ)


def detect_environment() -> Environment:
    """Read the execution environment of this process."""
    shell = _shell_class_name()
    if shell == "ZMQInteractiveShell":
        kind = "vscode" if _under_vscode() else "jupyter"
    elif shell:
        kind = "ipython"
    else:
        kind = "terminal"
    # Imported here: resolving the backend can import pyplot, which the
    # namespaces that call this at render time already have.
    import matplotlib

    backend = str(matplotlib.get_backend()).lower()
    live = backend in _LIVE_BACKENDS or backend.startswith("module://ipympl")
    widgets = False
    if kind in ("jupyter", "vscode"):
        try:
            import ipywidgets  # noqa: F401

            widgets = True
        except ImportError:
            widgets = False
    return Environment(kind=kind, backend=backend, live_figures=live, widgets=widgets)


def default_interaction_backend(environment: Environment | None = None) -> str:
    """The navigation backend ``backend="auto"`` resolves to here.

    A live canvas takes the Matplotlib slider, which works alike in a GUI
    window and under ``ipympl``.  A kernel with a static (inline) figure takes
    an ipywidgets slider that redraws the figure into its own output area.
    Anything else -- a script under ``agg``, a kernel without ipywidgets --
    gets a static figure and the navigator to drive from code.
    """
    environment = detect_environment() if environment is None else environment
    if environment.live_figures:
        return "matplotlib"
    if environment.in_kernel and environment.widgets:
        return "ipywidgets"
    return "none"


def use_non_interactive_backend() -> None:
    """Draw to files only: no display, no writable ``$HOME`` needed.

    Selects Matplotlib's ``Agg`` backend (unless one is already in use) and
    points ``MPLCONFIGDIR`` at a temporary directory when nothing set it, the
    way automated workflows and the ``vaft plot --out`` command run.
    """
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="vaft-mpl-"))
    import matplotlib

    matplotlib.use("Agg", force=False)
