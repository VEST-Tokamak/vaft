"""The stdio MCP server: VAFT's read-only tools registered with the ``mcp`` SDK (#1423).

The SDK is imported only inside :func:`build_server`, so this module (and the
API reference that imports it) works on an installation without ``vaft[mcp]``.
"""

from __future__ import annotations

import contextlib
import functools
import sys
from typing import Any

__all__ = ["INSTRUCTIONS", "build_server", "main"]

INSTRUCTIONS = (
    "Read-only access to VAFT, the VEST tokamak analysis toolkit. Start with "
    "get_capabilities. Formulas, processes, validation checks, plots and boundaries "
    "each have a search/list tool and a describe tool; get_plot_requirements gives "
    "the data paths a plot reads; extract_plot_data returns bounded numbers from a "
    "packaged reference shot (list_samples). Nothing here writes data, contacts a "
    "server or runs a solver."
)

_INSTALL_HINT = "the VAFT MCP server needs the MCP SDK: pip install 'vaft[mcp]'"


def _guarded(function, tool_error):
    """``function`` with stdout kept off the protocol stream and VAFT errors made tool errors.

    On stdio, stdout *is* the JSON-RPC channel: a stray ``print`` from a
    library VAFT calls would corrupt it, so the call's stdout goes to stderr.
    """
    from ._tools import _message

    @functools.wraps(function)
    def call(*args: Any, **kwargs: Any):
        with contextlib.redirect_stdout(sys.stderr):
            try:
                return function(*args, **kwargs)
            except (KeyError, ValueError, LookupError, TypeError, FileNotFoundError) as error:
                raise tool_error(_message(error)) from error

    return call


def build_server():
    """A ``FastMCP`` server named ``vaft`` exposing :data:`vaft.mcp._tools.TOOLS`, all read-only."""
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.server.fastmcp.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as error:
        raise ImportError(f"{_INSTALL_HINT} ({error})") from error

    from ._tools import TOOLS

    server = FastMCP("vaft", instructions=INSTRUCTIONS, log_level="WARNING")
    for function in TOOLS:
        server.add_tool(
            _guarded(function, ToolError),
            name=function.__name__,
            title=function.__name__.replace("_", " ").capitalize(),
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        )
    return server


def main(argv: list[str] | None = None) -> int:
    """Serve VAFT over stdio until the client closes the stream."""
    del argv  # no options yet; kept so ``vaft mcp`` and ``python -m vaft.mcp`` share a signature
    try:
        server = build_server()
    except ImportError as error:
        print(f"vaft mcp: {error}", file=sys.stderr)
        return 1
    server.run("stdio")
    return 0
