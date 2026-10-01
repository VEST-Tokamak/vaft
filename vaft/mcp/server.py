"""The stdio MCP server: VAFT's read-only tools registered with the ``mcp`` SDK (#1423).

The SDK is imported only inside :func:`build_server` and :func:`main`, so this
module (and the API reference that imports it) works on an installation
without ``vaft[mcp]``.

Two things keep the stdio transport intact.  Tools run in a worker thread,
one at a time, so a slow extraction never blocks the protocol's event loop.
And :func:`main` moves the JSON-RPC stream onto a private duplicate of file
descriptor 1 and points descriptor 1 at stderr: a ``print`` from Python, C or
Fortran code that VAFT calls lands in stderr instead of corrupting the stream.
"""

from __future__ import annotations

import functools
import inspect
import io
import os
import sys
import threading
import typing
from typing import Any

__all__ = ["INSTRUCTIONS", "build_server", "main"]

INSTRUCTIONS = (
    "Read-only access to VAFT, the VEST tokamak analysis toolkit. Start with "
    "get_capabilities. Formulas, processes, validation checks, plots and boundaries "
    "each have a search/list tool and a describe tool; get_plot_requirements gives "
    "the data paths a plot reads; extract_plot_data returns bounded numbers from a "
    "packaged reference shot (list_samples). Every result reports what it shortened "
    "under 'truncated'. Nothing here writes data, contacts a server or runs a solver."
)

_INSTALL_HINT = "the VAFT MCP server needs the MCP SDK: pip install 'vaft[mcp]'"

#: VAFT, Matplotlib and the loaders are not written for concurrent calls.
_CALL_LOCK = threading.Lock()


def _resolved_signature(function) -> inspect.Signature:
    """``function``'s signature with its string annotations evaluated where it was defined."""
    hints = typing.get_type_hints(function)
    signature = inspect.signature(function)
    return signature.replace(
        parameters=[p.replace(annotation=hints.get(p.name, p.annotation)) for p in signature.parameters.values()],
        return_annotation=hints.get("return", signature.return_annotation),
    )


def _description(function) -> str:
    text = inspect.getdoc(function) or ""
    if function.__name__ == "extract_plot_data":
        from ._tools import extraction_option_names

        try:
            names = extraction_option_names()
        except Exception as error:  # noqa: BLE001 - the description must not stop the server
            names = (f"(unavailable here: {type(error).__name__})",)
        text += "\n\nAllowed options keys: " + ", ".join(names) + "."
    return text


def _threaded(function, tool_error):
    """An async tool running ``function`` in a worker thread, VAFT errors made tool errors."""
    import anyio

    from ._tools import _message

    def guarded(**kwargs: Any):
        with _CALL_LOCK:
            try:
                return function(**kwargs)
            except (LookupError, ValueError, TypeError, FileNotFoundError) as error:
                raise tool_error(_message(error)) from error

    @functools.wraps(function)
    async def call(**kwargs: Any):
        return await anyio.to_thread.run_sync(functools.partial(guarded, **kwargs))

    call.__signature__ = _resolved_signature(function)
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
            _threaded(function, ToolError),
            name=function.__name__,
            title=function.__name__.replace("_", " ").capitalize(),
            description=_description(function),
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        )
    return server


def _protocol_stdout() -> io.TextIOWrapper:
    """Keep the real stdout for the protocol and send everything else on fd 1 to stderr."""
    sys.stdout.flush()
    protocol = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return io.TextIOWrapper(os.fdopen(protocol, "wb"), encoding="utf-8")


async def _serve(server, stdout: io.TextIOWrapper) -> None:
    import anyio
    from mcp.server.stdio import stdio_server

    async with stdio_server(stdout=anyio.wrap_file(stdout)) as (read_stream, write_stream):
        # What FastMCP.run_stdio_async does, with the protected stream.
        await server._mcp_server.run(
            read_stream, write_stream, server._mcp_server.create_initialization_options()
        )


def main(argv: list[str] | None = None) -> int:
    """Serve VAFT over stdio until the client closes the stream."""
    del argv  # no options yet; kept so ``vaft mcp`` and ``python -m vaft.mcp`` share a signature
    os.environ.setdefault("MPLBACKEND", "Agg")  # never open a window from a tool call
    try:
        server = build_server()
    except ImportError as error:
        print(f"vaft mcp: {error}", file=sys.stderr)
        return 1
    import anyio

    anyio.run(_serve, server, _protocol_stdout())
    return 0
