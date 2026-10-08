"""Browser GUI over the public VAFT APIs (#1086, roadmap #1359).

The GUI is a presentation layer, not part of the computational API: it loads
data, discovers plots and draws them through the same calls a notebook uses,
and adds Panel widgets on top.  Panel is a VAFT dependency, but this package
imports without it: Panel is imported only when a GUI is actually built, so
``import vaft`` stays light.

The page is an application shell (:mod:`vaft.gui.shell`) hosting
registered workspaces -- the plot explorer and the database view today --
that share one selection (:mod:`vaft.gui.selection`).

Run it with ``vaft gui``.  It binds to 127.0.0.1, so on a remote host or a
cluster node the page is reached through SSH or VS Code port forwarding.
"""

from __future__ import annotations

from typing import Any

from ._require import require_panel
from .selection import Selection, SelectionState
from .shell import WORKSPACES, WorkspaceRegistry, WorkspaceSpec, register_workspace
from .state import BrowserSession, Source
from . import workspaces as _workspaces  # noqa: F401 - registers the built-in workspaces

_LAZY = {
    "BrowserApp": ".app",
    "build_app": ".app",
    "build_shell": ".app",
    "serve": ".app",
    "panel_controls": ".widgets",
    "Shell": ".shell",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BrowserApp",
    "BrowserSession",
    "Selection",
    "SelectionState",
    "Shell",
    "Source",
    "WORKSPACES",
    "WorkspaceRegistry",
    "WorkspaceSpec",
    "build_app",
    "build_shell",
    "panel_controls",
    "register_workspace",
    "require_panel",
    "serve",
]
