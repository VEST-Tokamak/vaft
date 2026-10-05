"""Optional browser GUI over the public VAFT APIs (#1086, roadmap #1359).

The GUI is a presentation layer, not part of the computational API: it loads
data, discovers plots and draws them through the same calls a notebook uses,
and adds Panel widgets on top.  Panel is optional (``pip install
'vaft[gui]'``); this package imports without it and names the extra only
when a GUI is actually built.

Run it with ``vaft gui``.  It binds to 127.0.0.1, so on a remote host or a
cluster node the page is reached through SSH or VS Code port forwarding.
"""

from __future__ import annotations

from typing import Any

from ._require import require_panel
from .state import BrowserSession, Source

_LAZY = {"BrowserApp": ".app", "build_app": ".app", "serve": ".app", "panel_controls": ".widgets"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BrowserApp", "BrowserSession", "Source", "build_app", "panel_controls", "require_panel", "serve"]
