"""A local, read-only Model Context Protocol server over VAFT discovery (#1423).

MCP clients (Claude Code, Codex and others) can ask this server what VAFT can
do -- help topics, the formula and process catalogs, validation checks, the
plot registry and its declared data paths, packaged reference shots and the
operational-boundary registry -- and pull bounded numerical views of the
packaged reference data through VAFT's own extraction layer.

    pip install 'vaft[mcp]'
    python -m vaft.mcp            # or: vaft mcp   (stdio transport)

It is an adapter, not a second API: every answer comes from an existing VAFT
function.  It writes nothing, contacts no server, runs no solver or pipeline
and executes no caller-supplied code.  Importing :mod:`vaft.mcp` does not
import the ``mcp`` SDK; :func:`build_server` does.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["build_server", "main"]


def __getattr__(name: str):
    if name in __all__:
        value = getattr(import_module(".server", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
