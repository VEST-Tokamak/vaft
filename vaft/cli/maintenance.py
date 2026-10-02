"""Maintenance of already-published HSDS shots."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Iterable


def _shots(values: Iterable[str]) -> list[int]:
    shots: list[int] = []
    for value in values:
        text = str(value).strip()
        if "-" in text and not text.startswith("-"):
            first, last = text.split("-", 1)
            shots.extend(range(int(first), int(last) + 1))
        elif text:
            shots.append(int(text))
    return sorted(dict.fromkeys(shots))


def _audit_masters(args: argparse.Namespace) -> int:
    from collections import Counter

    from vaft.database.maintenance import (
        MASTER_LINKS_MISSING,
        MASTER_MISSING,
        MASTER_UNREADABLE,
        audit_master_links,
    )

    reports = audit_master_links(_shots(args.shots), source=args.source, repair=args.apply)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump({"applied": bool(args.apply), "reports": reports}, handle, indent=2, sort_keys=True)
            handle.write("\n")
    for report in reports:
        if report["status"] in ("complete", "absent"):
            continue
        detail = report.get("error") or ", ".join(report["missing"]) or report.get("note", "")
        stubs = report.get("stubs") or []
        if stubs:
            detail = "; ".join(filter(None, [detail, "empty stubs, left unlinked: " + ", ".join(stubs)]))
        print(f"shot {report['shot']}: {report['status']}: {detail}")
    counts = Counter(report["status"] for report in reports)
    print(", ".join(f"{status} {count}" for status, count in sorted(counts.items()))
          + ("" if args.apply else "; dry run, nothing was written"))
    # A shot with IDS files and no master is unreadable data too, though
    # `--apply` cannot fix it: there is no link to copy the shape of.
    hidden = (MASTER_LINKS_MISSING, MASTER_MISSING, MASTER_UNREADABLE)
    return 1 if any(counts.get(status) for status in hidden) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vaft.cli maintenance",
        description=__doc__,
    )
    subcommands = parser.add_subparsers(dest="action", required=True)
    strip = subcommands.add_parser(
        "strip-impa",
        help="remove IMPA channels left in a baseline shot published before issue #305",
    )
    strip.add_argument("--shots", nargs="+", required=True, help="Shot numbers or FIRST-LAST ranges.")
    strip.add_argument("--source", default=None, help="Source to repair; defaults to main.")
    strip.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite the published magnetics. Without it nothing is written.",
    )
    strip.add_argument("--report", default=None, help="Write the per-shot report as JSON here.")
    audit = subcommands.add_parser(
        "audit-masters",
        help="find shots whose master.h5 does not link every IDS file stored beside it (#913)",
    )
    audit.add_argument("--shots", nargs="+", required=True, help="Shot numbers or FIRST-LAST ranges.")
    audit.add_argument("--source", default=None, help="Source to audit; defaults to main.")
    audit.add_argument(
        "--apply",
        action="store_true",
        help="Relink the missing files. Without it nothing is written.",
    )
    audit.add_argument("--report", default=None, help="Write the per-shot report as JSON here.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if args.action == "audit-masters":
        return _audit_masters(args)
    from vaft.database.maintenance import strip_impa_from_shots

    shots = _shots(args.shots)
    reports = strip_impa_from_shots(shots, source=args.source, apply=args.apply)
    payload = {"applied": bool(args.apply), "reports": reports}
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    carrying = [r for r in reports if r.get("carries_impa")]
    failed = [r for r in reports if r.get("error")]
    for report in reports:
        if report.get("error"):
            print(f"shot {report['shot']}: {report['error']}", file=sys.stderr)
        elif report.get("carries_impa"):
            verb = "removed" if report["applied"] else "would remove"
            print(f"shot {report['shot']}: {verb} {report['removed']} IMPA channels")
    print(
        f"{len(carrying)} of {len(reports)} shots carry IMPA"
        + ("" if args.apply else "; dry run, nothing was written")
    )
    return 1 if failed else 0


__all__ = ["main"]
