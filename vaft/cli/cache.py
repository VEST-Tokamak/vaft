"""Inspect, prune and clear the local HSDS domain cache.

``vaft cache info`` reports the location, used size, cap and which setting the
cap came from; ``list`` shows cached domains; ``prune`` evicts least-recently
used domains down to the cap (or ``--max``) and/or those older than
``--older-than``; ``clear`` removes cached domains (all of them only with
``--yes``); ``set-limit`` persists the cap in the user config file.

The cache holds only copies of HSDS domains, so nothing here can lose data.
See :mod:`vaft.database.hsds_cache` for the cap's precedence and the layout.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import datetime as _dt
import json
import os
import sys

from vaft.database import hsds_cache


def _add_cache_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache",
        default="auto",
        help="cache base directory (default: the one cache='auto' uses)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vaft cache", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="location, used size, cap and where the cap came from")
    _add_cache_option(info)
    info.add_argument("--json", action="store_true", help="print JSON")

    listing = sub.add_parser("list", help="cached domains, least recently used first")
    _add_cache_option(listing)
    listing.add_argument("--shot", type=int, help="only this shot")
    listing.add_argument("--source", help="only this source (e.g. main, public)")

    prune = sub.add_parser("prune", help="evict least-recently-used domains down to the cap")
    _add_cache_option(prune)
    prune.add_argument("--max", dest="max_size", help="size to prune down to (default: the cap)")
    prune.add_argument("--older-than", help="also remove domains unused for this long (e.g. 30d)")

    clear = sub.add_parser("clear", help="remove cached domains")
    _add_cache_option(clear)
    clear.add_argument("--shot", type=int, help="only this shot")
    clear.add_argument("--source", help="only this source")
    clear.add_argument("--yes", action="store_true", help="confirm clearing the whole cache")

    limit = sub.add_parser("set-limit", help="persist the cap in the user config file")
    limit.add_argument("size", help="e.g. 500M, 20G; 0 disables caching, 'none' removes the cap")
    return parser


def _when(stamp: float) -> str:
    return _dt.datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M")


def _info(arguments, out) -> int:
    report = hsds_cache.info(arguments.cache)
    if arguments.json:
        print(json.dumps(report, indent=2), file=out)
        return 0
    used, cap = report["used_bytes"], report["cap_bytes"]
    print("VAFT HSDS cache", file=out)
    print(f"  location  {report['location']}", file=out)
    print(f"  used      {hsds_cache.format_size(used)}", file=out)
    print(f"  cap       {hsds_cache.format_size(cap)} (from {report['cap_source']})", file=out)
    print(
        f"  holds     {report['domains']} domain(s) of {report['shots']} shot(s)"
        + (f" from {', '.join(report['sources'])}" if report["sources"] else ""),
        file=out,
    )
    if cap is not None and used > cap:
        print("  note      over the cap; `vaft cache prune` evicts down to it", file=out)
    return 0


def _list(arguments, out) -> int:
    items = hsds_cache.entries(arguments.cache, shot=arguments.shot, source=arguments.source)
    if not items:
        print("No cached domains.", file=out)
        return 0
    print(f"{'shot':>7}  {'source':<12} {'domain':<28} {'size':>10}  last used", file=out)
    for entry in items:
        print(
            f"{entry.shot:>7}  {entry.source:<12} {entry.filename[:-3]:<28} "
            f"{hsds_cache.format_size(entry.size):>10}  {_when(entry.last_used)}",
            file=out,
        )
    return 0


def _summary(verb: str, result, out) -> None:
    print(
        f"{verb} {len(result.removed)} domain(s), {hsds_cache.format_size(result.freed_bytes)}; "
        f"{hsds_cache.format_size(result.remaining_bytes)} left.",
        file=out,
    )


def _prune(arguments, out) -> int:
    result = hsds_cache.prune(
        arguments.max_size, older_than=arguments.older_than, cache=arguments.cache
    )
    _summary("Evicted", result, out)
    return 0


def _clear(arguments, out) -> int:
    if arguments.shot is None and arguments.source is None and not arguments.yes:
        print(
            "error: this removes every cached domain; pass --yes, or narrow it "
            "with --shot/--source",
            file=sys.stderr,
        )
        return 2
    result = hsds_cache.clear(arguments.cache, shot=arguments.shot, source=arguments.source)
    _summary("Removed", result, out)
    return 0


def _set_limit(arguments, out) -> int:
    path = hsds_cache.set_cache_max(arguments.size)
    size = hsds_cache.parse_size(arguments.size)
    print(f"HSDS cache cap set to {hsds_cache.format_size(size)} in {path}.", file=out)
    override = os.environ.get(hsds_cache.CACHE_MAX_ENV, "").strip()
    if override:
        print(f"note: ${hsds_cache.CACHE_MAX_ENV}={override} overrides it in this shell.", file=out)
    return 0


_HANDLERS = {
    "info": _info,
    "list": _list,
    "prune": _prune,
    "clear": _clear,
    "set-limit": _set_limit,
}


def main(argv: Iterable[str] | None = None, *, out=None) -> int:
    out = out if out is not None else sys.stdout
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        return _HANDLERS[arguments.command](arguments, out)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


__all__ = ["main"]
