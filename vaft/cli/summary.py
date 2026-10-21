"""Preset summary materialization command."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from vaft import database
from vaft.database._summary import get_summary_preset
from vaft.database.sources import known_sources


def _shot_range(value: str) -> tuple[int, int]:
    try:
        start_text, end_text = value.split(":", 1)
        start, end = int(start_text), int(end_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("shot range must have the form START:END") from exc
    if start > end:
        raise argparse.ArgumentTypeError("shot range START must not exceed END")
    return start, end


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vaft.cli summary")
    subparsers = parser.add_subparsers(dest="action", required=True)
    export = subparsers.add_parser("export", help="query and export a preset summary")
    export.add_argument("--preset", default="equilibrium_global")
    export.add_argument(
        "--shot-range",
        type=_shot_range,
        help="inclusive START:END; omit to query every available shot",
    )
    export.add_argument("--output", required=True)
    export.add_argument(
        "--source",
        default=None,
        help="named HSDS source to summarize; defaults to 'main'",
    )
    export.add_argument("--upsert", action="store_true")
    subparsers.add_parser("sources", help="list the named HSDS sources")
    monitor = subparsers.add_parser("monitor", help="collect equilibrium attempts, validation and input evidence")
    monitor.add_argument("--shot-range", required=True, type=_shot_range)
    monitor.add_argument("--mode", choices=("hsds", "server"), default="hsds")
    monitor.add_argument("--source", default=None)
    monitor.add_argument("--family", choices=("magnetic", "electron", "kinetic"), default="magnetic")
    monitor.add_argument("--filedb", help="local or mounted server FileDB root (required in server mode)")
    monitor.add_argument("--output", required=True)
    monitor.add_argument("--no-magnetics", action="store_true")
    monitor.add_argument("--vacuum", action="store_true", help="also run the PF-only vacuum benchmark; can be expensive")
    monitor.add_argument("--figures", action="store_true")
    monitor.add_argument("--batch-size", type=int, help="write bounded shot batches into separate subdirectories")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.action == "sources":
        for entry in known_sources():
            access = "read-write" if entry.writable else "read-only"
            # A sparse source holds only the shots its product was produced
            # for, so a missing shot there means nothing more than that.
            coverage = "sparse" if entry.sparse else "complete"
            print(f"{entry.name:<22} {access:<10} {coverage:<9} {entry.purpose}")
        return 0
    if args.action == "monitor":
        from vaft.database._equilibrium_monitoring import equilibrium_summary
        from pathlib import Path
        start, end = args.shot_range
        if args.batch_size is not None and args.batch_size <= 0:
            raise ValueError("batch-size must be positive")
        size = args.batch_size or (end - start + 1)
        for begin in range(start, end + 1, size):
            stop = min(end, begin + size - 1)
            output = Path(args.output) / f"shots_{begin}_{stop}" if args.batch_size else Path(args.output)
            result = equilibrium_summary(range(begin, stop + 1),
                mode=args.mode, source=args.source, family=args.family, filedb=args.filedb,
                magnetics=not args.no_magnetics, vacuum=args.vacuum)
            result.export(output)
            if args.figures:
                from vaft.plot.equilibrium_monitoring import write_monitoring_figures
                write_monitoring_figures(result, output)
        return 0
    definition = get_summary_preset(args.preset)
    frame = database.summary(args.shot_range, preset=args.preset, source=args.source)
    database.export_summary(
        frame,
        args.output,
        mode="upsert" if args.upsert else "replace",
        key_columns=definition.key_columns if args.upsert else None,
        replace_groups=definition.replace_groups if args.upsert else None,
    )
    return 0


__all__ = ["main"]
