"""``vaft help [topic] [item]``: what VAFT can do, its defaults, and setup status.

The command-level ``vaft <command> --help`` stays argparse syntax help; this
command prints the same capability pages as :func:`vaft.help`, from the same
registry.  Parsing imports only that registry, so it works before any
scientific dependency is importable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Iterable

from vaft._help._registry import TOPICS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vaft help",
        description=(
            "What VAFT can do, its defaults, and setup status. "
            "For a command's syntax and arguments use: vaft <command> --help"
        ),
        epilog="topics: " + ", ".join(TOPICS),
    )
    parser.add_argument(
        "topic",
        nargs="?",
        type=str.lower,
        choices=tuple(TOPICS),
        metavar="topic",
        help="help topic (default: overview)",
    )
    parser.add_argument("item", nargs="?", help="one entry within the topic, e.g. a formula or source name")
    parser.add_argument("--probe", action="store_true", help="look for the external executables (code topic only)")
    parser.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    return parser


def _jsonable(result):
    if hasattr(result, "as_dict"):
        return result.as_dict()
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return dataclasses.asdict(result)
    if isinstance(result, (dict, list, tuple, str, int, float, bool)) or result is None:
        return result
    return str(result)


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    from vaft._help import help as vaft_help

    try:
        result = vaft_help(arguments.topic, arguments.item, probe=arguments.probe)
    except (KeyError, ValueError) as error:
        message = error.args[0] if error.args else error
        print(f"vaft help: {message}", file=sys.stderr)
        return 2
    except ImportError as error:  # an item view whose subsystem is not importable here
        print(
            f"vaft help: {arguments.topic} {arguments.item} unavailable here: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    if arguments.format == "json":
        print(json.dumps(_jsonable(result), indent=2, default=str))
    elif arguments.format == "markdown" and hasattr(result, "_repr_markdown_"):
        print(result._repr_markdown_())
    else:
        print(result)
    return 0


__all__ = ["main"]
