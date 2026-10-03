"""``vaft mcp``: serve VAFT's read-only discovery tools to an MCP client over stdio (#1423).

Equivalent to ``python -m vaft.mcp``.  Register it with a client, for example
``claude mcp add vaft -- vaft mcp``.  Needs ``pip install 'vaft[mcp]'``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="vaft mcp",
        description=(
            "Run the local, read-only VAFT MCP server on stdio (help topics, formula/process "
            "catalogs, validation checks, plots, packaged samples, boundaries). It writes "
            "nothing, contacts no server and runs no solver. Needs: pip install 'vaft[mcp]'."
        ),
    )


def main(argv: Iterable[str] | None = None) -> int:
    _parser().parse_args(list(argv) if argv is not None else None)
    from vaft.mcp.server import main as serve

    return serve()


__all__ = ["main"]
