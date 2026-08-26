"""Preset summary materialization command."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from vaft import database
from vaft.database._summary import get_summary_preset


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
    export.add_argument("--source", default="public")
    export.add_argument("--upsert", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
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
