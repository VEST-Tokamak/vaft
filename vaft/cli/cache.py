"""Inspect, prune and clear VAFT's local caches.

Without ``--type`` every command works on the HSDS domain cache exactly as
before: ``vaft cache info`` reports the location, used size, cap and which
setting the cap came from; ``list`` shows cached domains; ``prune`` evicts
least-recently used domains down to the cap (or ``--max``) and/or those older
than ``--older-than``; ``clear`` removes cached domains (all of them only with
``--yes``); ``set-limit`` persists the cap in the user config file.

``--type`` selects another namespace of the shared cache root
(:mod:`vaft.storage`): ``public`` (registered public-database downloads),
``references`` (Release copies of reference samples and the EFIT baseline),
``generated`` (EFUND tables), or ``all`` for ``info``/``list``. These hold no
size cap: ``prune --type X`` needs ``--older-than``, and ``clear --type X``
needs ``--entry`` or ``--yes``; ``all`` is never accepted for removal. Only
content VAFT wrote and can fetch or generate again is ever removed; ``--dry-run``
reports first.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import datetime as _dt
import json
import os
import sys

from vaft import storage
from vaft.database import hsds_cache

#: ``--type`` value -> namespaces it covers.
TYPES = {
    "hsds": ("hsds",),
    "public": ("public",),
    "references": ("references/samples", "references/efit-baseline"),
    "generated": ("generated/efund",),
}


def _add_cache_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache",
        default="auto",
        help="cache base directory (default: the one cache='auto' uses)",
    )


def _add_type_option(parser: argparse.ArgumentParser, *, allow_all: bool) -> None:
    choices = [*TYPES, "all"] if allow_all else list(TYPES)
    parser.add_argument(
        "--type",
        choices=choices,
        default=None,
        help="cache namespace (default: hsds, unchanged behaviour)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vaft cache", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="location, used size, cap and where the cap came from")
    _add_cache_option(info)
    _add_type_option(info, allow_all=True)
    info.add_argument("--all", action="store_true", help="every namespace (same as --type all)")
    info.add_argument("--json", action="store_true", help="print JSON")

    listing = sub.add_parser("list", help="cached domains, least recently used first")
    _add_cache_option(listing)
    _add_type_option(listing, allow_all=True)
    listing.add_argument("--shot", type=int, help="only this shot")
    listing.add_argument("--source", help="only this source (e.g. main, public)")

    prune = sub.add_parser("prune", help="evict least-recently-used domains down to the cap")
    _add_cache_option(prune)
    _add_type_option(prune, allow_all=False)
    prune.add_argument("--max", dest="max_size", help="size to prune down to (default: the cap)")
    prune.add_argument("--older-than", help="also remove domains unused for this long (e.g. 30d)")
    prune.add_argument("--dry-run", action="store_true", help="report, remove nothing (not for hsds)")

    clear = sub.add_parser("clear", help="remove cached domains")
    _add_cache_option(clear)
    _add_type_option(clear, allow_all=False)
    clear.add_argument("--shot", type=int, help="only this shot")
    clear.add_argument("--source", help="only this source")
    clear.add_argument("--entry", action="append", help="only this entry or public source key (repeatable)")
    clear.add_argument("--dry-run", action="store_true", help="report, remove nothing (not for hsds)")
    clear.add_argument("--yes", action="store_true", help="confirm clearing the whole cache")

    limit = sub.add_parser("set-limit", help="persist the cap in the user config file")
    limit.add_argument("size", help="e.g. 500M, 20G; 0 disables caching, 'none' removes the cap")
    return parser


def _when(stamp: float) -> str:
    return _dt.datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M")


def _selected(arguments) -> str | None:
    """The non-default namespace type, or ``None`` for the legacy HSDS behaviour."""
    chosen = "all" if getattr(arguments, "all", False) else arguments.type
    return None if chosen in (None, "hsds") else chosen


def _override(arguments, name: str):
    """``--cache`` applies only when a single namespace is selected."""
    return None if arguments.cache == "auto" else arguments.cache


def _namespaces(kind: str) -> tuple[str, ...]:
    if kind == "all":
        return tuple(name for names in TYPES.values() for name in names)
    return TYPES[kind]


def _check_single(arguments, kind: str) -> None:
    if arguments.cache != "auto" and len(_namespaces(kind)) != 1:
        raise ValueError(f"--cache needs a single namespace; --type {kind} covers several")


def _namespace_info(name: str, override) -> dict:
    if name == "hsds":
        report = storage.namespace_report("hsds", override)
        report.update(hsds_cache.info(override if override is not None else "auto"))
        return report
    report = storage.namespace_report(name, override)
    if name == "public":
        from vaft.data.public import _fetch

        files = _fetch.cached_sources(override) if report["exists"] else []
        report.update(
            {
                "entries": len(files),
                "used_bytes": sum(item["bytes"] for item in files),
                "unverified": [item["key"] for item in files if not item["verified"]],
            }
        )
    return report


def _info_namespaces(arguments, kind: str, out) -> int:
    _check_single(arguments, kind)
    reports = [_namespace_info(name, _override(arguments, name)) for name in _namespaces(kind)]
    if arguments.json:
        print(json.dumps({"namespaces": reports}, indent=2), file=out)
        return 0
    print(f"VAFT cache root  {storage.cache_root()}", file=out)
    for report in reports:
        used = report.get("used_bytes", 0)
        state = "" if report["exists"] else "  (not created)"
        print(f"  {report['namespace']:<26} {report['location']}{state}", file=out)
        print(
            f"  {'':<26} {hsds_cache.format_size(used)} in {report.get('entries', report.get('domains', 0))} "
            f"entr(ies); eviction: {report['eviction']}",
            file=out,
        )
    return 0


def _list_namespaces(arguments, kind: str, out) -> int:
    _check_single(arguments, kind)
    if arguments.shot is not None or arguments.source is not None:
        raise ValueError("--shot/--source apply to the hsds cache only")
    rows = []
    for name in _namespaces(kind):
        override = _override(arguments, name)
        if name == "hsds":
            for entry in hsds_cache.entries(override if override is not None else "auto"):
                rows.append((name, f"{entry.shot}/{entry.source}/{entry.filename}", entry.size, entry.last_used))
        elif name == "public":
            from vaft.data.public import _fetch

            for item in _fetch.cached_sources(override):
                label = item["key"] + ("" if item["verified"] else " (checksum mismatch)")
                rows.append((name, label, item["bytes"], item["last_used"]))
        else:
            for entry in storage.entries(name, override):
                rows.append((name, entry.key, entry.size, entry.last_used))
    if not rows:
        print("No cached entries.", file=out)
        return 0
    print(f"{'namespace':<26} {'entry':<40} {'size':>10}  last used", file=out)
    for name, label, size, used in rows:
        print(f"{name:<26} {label[:40]:<40} {hsds_cache.format_size(size):>10}  {_when(used)}", file=out)
    return 0


def _remove_namespaces(arguments, kind: str, out, *, clearing: bool) -> int:
    _check_single(arguments, kind)
    if getattr(arguments, "max_size", None):
        raise ValueError(f"--max applies to the hsds cap only; --type {kind} has no cap")
    if getattr(arguments, "shot", None) is not None or getattr(arguments, "source", None) is not None:
        raise ValueError("--shot/--source apply to the hsds cache only")
    older = hsds_cache.parse_age(arguments.older_than) if getattr(arguments, "older_than", None) else None
    keys = getattr(arguments, "entry", None)
    if clearing:
        if keys is None and not arguments.yes:
            print(
                f"error: this removes every {kind} entry VAFT wrote; pass --yes, or narrow it with --entry",
                file=sys.stderr,
            )
            return 2
    elif older is None:
        print(f"error: --type {kind} has no cap; prune needs --older-than", file=sys.stderr)
        return 2
    verb = "Would remove" if arguments.dry_run else "Removed"
    for name in _namespaces(kind):
        override = _override(arguments, name)
        if name == "public":
            from vaft.data.public import _fetch

            removed, freed, kept = _fetch.remove_cached_sources(
                keys, older_than=older, cache=override, dry_run=arguments.dry_run
            )
            print(f"{verb} {len(removed)} {name} file(s), {hsds_cache.format_size(freed)}.", file=out)
            if kept:
                print(f"  kept (checksum mismatch, inspect by hand): {', '.join(kept)}", file=out)
            continue
        result = storage.remove_entries(
            name,
            keys=keys,
            older_than=older,
            everything=clearing and keys is None,
            dry_run=arguments.dry_run,
            override=override,
        )
        print(
            f"{verb} {len(result.removed)} {name} entr(ies), {hsds_cache.format_size(result.freed_bytes)}; "
            f"{hsds_cache.format_size(result.remaining_bytes)} left.",
            file=out,
        )
        if result.skipped:
            print(f"  not VAFT's, left alone: {', '.join(result.skipped)}", file=out)
    return 0


def _info(arguments, out) -> int:
    kind = _selected(arguments)
    if kind is not None:
        return _info_namespaces(arguments, kind, out)
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
    kind = _selected(arguments)
    if kind is not None:
        return _list_namespaces(arguments, kind, out)
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
    kind = _selected(arguments)
    if kind is not None:
        return _remove_namespaces(arguments, kind, out, clearing=False)
    if arguments.dry_run:
        raise ValueError("--dry-run is not available for the hsds cache")
    result = hsds_cache.prune(
        arguments.max_size, older_than=arguments.older_than, cache=arguments.cache
    )
    _summary("Evicted", result, out)
    return 0


def _clear(arguments, out) -> int:
    kind = _selected(arguments)
    if kind is not None:
        return _remove_namespaces(arguments, kind, out, clearing=True)
    if arguments.dry_run or arguments.entry:
        raise ValueError("--dry-run/--entry apply to --type public, references or generated")
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
