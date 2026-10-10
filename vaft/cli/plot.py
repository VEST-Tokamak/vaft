"""``vaft plot``: render a canonical plot, or several composed, from shots, samples or files.

The command is a thin front on the plotting API: for database shots it names
a plot and a shot in a source, and :mod:`vaft.database.plotting` opens exactly
the IDS the plot needs and draws it; ``--sample``/``--file`` draw packaged
samples or local files instead.  ``--compose`` draws a
:class:`vaft.plot.FigureComposition` (#1467), ``--format``/``--theme``/
``--figure-options`` set the presentation (#689, #1421), and ``--request``
replays a whole :class:`vaft.plot.PlotRequest` -- the command the GUI and
``PlotRequest.to_cli()`` write.  ``--list`` prints what a shot can plot
without downloading it.  A table or text view (issue #1180) is printed to
stdout instead of opening a window, and ``--out x.txt|x.md|x.html`` writes
the matching export.  Nothing heavier than ``argparse`` is imported
before the arguments are parsed, so ``vaft plot --help`` works in a bare
install.
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
    parser.add_argument("--shot", action="append", type=int, help="database shot number (repeat for several)")
    parser.add_argument("--sample", action="append", type=int, help="packaged sample shot, offline (repeat for several)")
    parser.add_argument("--file", action="append", help="local ODS/IMAS/GEQDSK file (repeat for several)")
    parser.add_argument("--source", help="HSDS source (default: main)")
    parser.add_argument(
        "--out",
        help="write the figure here (format by extension) instead of showing it; "
             "a table/text view writes .txt, .md or .html (without --out it is printed)",
    )
    parser.add_argument("--no-lazy", action="store_true", help="stage the declared IDS instead of lazy reads")
    parser.add_argument(
        "--option", action="append", default=[], metavar="KEY=VALUE",
        help="plot option, e.g. selection=all, time_slice=4, style=normalized (repeatable)",
    )
    parser.add_argument("--format", help="presentation format: screen, single_column, double_column, slide, poster")
    parser.add_argument("--theme", help="presentation theme: technical, minimal, monochrome")
    parser.add_argument("--backend", choices=("matplotlib", "plotly"), help="drawing library (plotly writes .html)")
    parser.add_argument(
        "--figure-options", metavar="JSON",
        help='explicit figure options as JSON (or a path to one), e.g. \'{"xlim": [0.3, 0.33]}\'',
    )
    parser.add_argument(
        "--compose", metavar="JSON",
        help="draw a FigureComposition (JSON or a path) instead of one named plot",
    )
    parser.add_argument(
        "--request", metavar="JSON",
        help="replay a whole PlotRequest (JSON or a path), as PlotRequest.to_cli()/the GUI write it",
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


def _json_argument(text: str, flag: str, parser: argparse.ArgumentParser) -> Any:
    """``text`` as JSON: inline when it looks like an object, else read from the file it names."""
    import json
    from pathlib import Path

    try:
        if text.lstrip().startswith("{"):
            return json.loads(text)
        return json.loads(Path(text).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        parser.error(f"{flag} expects a JSON object or a JSON file: {error}")


def _request_from(args: argparse.Namespace, options: dict[str, Any], parser: argparse.ArgumentParser) -> Any:
    """The :class:`vaft.plot.PlotRequest` the arguments describe, or ``None`` for the shot path."""
    from vaft.plot.request import DataSource, PlotRequest

    if args.request:
        given = [flag for flag, value in (
            ("a plot name", args.name), ("--shot", args.shot), ("--sample", args.sample), ("--file", args.file),
            ("--source", args.source), ("--compose", args.compose), ("--option", args.option),
            ("--format", args.format), ("--theme", args.theme), ("--backend", args.backend),
            ("--figure-options", args.figure_options),
        ) if value]
        if given:
            parser.error(f"--request carries the whole figure; drop {', '.join(given)}")
        return PlotRequest.from_dict(_json_argument(args.request, "--request", parser))
    inputs = [(kind, values) for kind, values in (("shot", args.shot), ("sample", args.sample), ("file", args.file)) if values]
    if len(inputs) > 1:
        parser.error("give one kind of input: --shot, --sample or --file")
    if not inputs:
        parser.error("--shot, --sample or --file is required (or --list)")
    kind, values = inputs[0]
    if kind != "shot" and args.source:
        parser.error("--source names a database namespace; it applies to --shot only")
    if kind != "shot" and args.no_lazy:
        parser.error("--no-lazy applies to database shots (--shot) only")
    if kind == "shot" and not args.compose:
        return None  # the database path below opens only what the plot reads
    if bool(args.compose) == bool(args.name):
        parser.error("name one plot, or give --compose")
    return PlotRequest(
        source=DataSource(kind, tuple(values), args.source if kind == "shot" else None),
        plot=args.name,
        composition=_json_argument(args.compose, "--compose", parser) if args.compose else None,
        options=options,
        format=args.format,
        theme=args.theme,
        backend=args.backend,
        figure_options=_json_argument(args.figure_options, "--figure-options", parser) if args.figure_options else None,
    )


def _write(result: Any, out: str, figure_options: Any = None) -> str:
    """Save what a request drew to ``out``: HTML for Plotly, the extension's format otherwise.

    A table or text view (issue #1180) is text: ``.txt``, ``.md`` or ``.html``
    writes the matching export.
    """
    from vaft.plot.renderers.tables import TextView

    if isinstance(result, TextView):
        return result.save(out)
    if hasattr(result, "write_html"):
        if not out.lower().endswith((".html", ".htm")):
            raise ValueError(f"backend='plotly' writes HTML; give --out a .html path, not {out!r}")
        result.write_html(out, include_plotlyjs="cdn")
        return out
    from vaft.plot import save_figure

    return save_figure(result[0], out, figure_options=figure_options)


def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    options = dict(_parse_option(item, parser) for item in args.option)
    shot: Any = None if not args.shot else (args.shot[0] if len(args.shot) == 1 else list(args.shot))

    if not args.list and (args.request or args.compose or args.sample or args.file):
        try:
            request = _request_from(args, options, parser)
            if args.out:
                from vaft.plot.environment import use_non_interactive_backend

                use_non_interactive_backend()
                print(_write(request.render(), args.out, request.figure_options))
            else:
                request.render(show=True)
        except KeyboardInterrupt:
            print("vaft plot: interrupted", file=sys.stderr)
            return 130
        except (KeyError, TypeError, ValueError, NotImplementedError, OSError) as error:
            message = error.args[0] if isinstance(error, KeyError) and error.args else error
            print(f"vaft plot: error: {message}", file=sys.stderr)
            return 1
        return 0
    for key in ("format", "theme", "backend"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    if args.figure_options:
        options["figure_options"] = _json_argument(args.figure_options, "--figure-options", parser)

    from vaft.database import plotting

    if args.list and args.shot and len(args.shot) > 1:
        parser.error("--list describes one shot; give --shot once")
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
    except KeyboardInterrupt:
        print("vaft plot: interrupted", file=sys.stderr)
        return 130
    except (KeyError, TypeError, ValueError, NotImplementedError, OSError) as error:
        # A refused plot, an unknown source or shot, a backend that cannot draw
        # it, or an output path that cannot be written: one line, exit 1.
        message = error.args[0] if isinstance(error, KeyError) and error.args else error
        print(f"vaft plot: error: {message}", file=sys.stderr)
        return 1
    return 0


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
