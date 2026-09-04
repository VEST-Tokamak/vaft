"""``vaft plot``: render a canonical plot for one or more database shots.

The command is a thin front on :mod:`vaft.database.plotting`: it names a
plot and a shot in a source, and the adapter opens exactly the IDS the plot
needs and draws it.  ``--list`` prints what a shot can plot without
downloading it.  Nothing heavier than ``argparse`` is imported before the
arguments are parsed, so ``vaft plot --help`` works in a bare install.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vaft.cli plot",
        description="Render a canonical plot for one or more shots, or list what a shot can plot.",
    )
    parser.add_argument("name", nargs="?", help="canonical plot name, e.g. plasma_current_time")
    parser.add_argument("--shot", action="append", type=int, help="shot number (repeat for several)")
    parser.add_argument("--source", help="HSDS source (default: main)")
    parser.add_argument("--out", help="write the figure here (format by extension) instead of showing it")
    parser.add_argument("--no-lazy", action="store_true", help="stage the declared IDS instead of lazy reads")
    parser.add_argument(
        "--option", action="append", default=[], metavar="KEY=VALUE",
        help="plot option, e.g. selection=all, time_slice=4, style=normalized (repeatable)",
    )
    parser.add_argument("--list", action="store_true", help="list the plots (of the shot, when given)")
    parser.add_argument("--query", help="with --list: narrow the catalogue")
    parser.add_argument("--detail", action="store_true", help="with --list: print every capability")
    return parser


def _parse_option(text: str, parser: argparse.ArgumentParser) -> tuple[str, Any]:
    """``KEY=VALUE`` with a Python literal value where it parses, else a string."""
    key, separator, raw = text.partition("=")
    if not separator or not key.strip():
        parser.error(f"--option expects KEY=VALUE; got {text!r}")
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        value = raw
    return key.strip(), value


def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    options = dict(_parse_option(item, parser) for item in args.option)
    shot: Any = None if not args.shot else (args.shot[0] if len(args.shot) == 1 else list(args.shot))

    from vaft.database import plotting

    if args.list:
        try:
            print(plotting.available_plots(shot, args.source, query=args.query, detail=args.detail))
        except ValueError as error:
            print(f"vaft plot: error: {error}", file=sys.stderr)
            return 1
        return 0
    if not args.name:
        parser.error("a plot name is required (or --list)")
    if shot is None:
        parser.error("--shot is required (or --list)")
    try:
        if args.out:
            written = plotting.render_to_file(
                args.name, shot, args.out, args.source, lazy=not args.no_lazy, **options
            )
            print(written)
        else:
            plotting.render(args.name, shot, args.source, lazy=not args.no_lazy, show=True, **options)
    except (KeyError, ValueError, NotImplementedError) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else error
        print(f"vaft plot: error: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
