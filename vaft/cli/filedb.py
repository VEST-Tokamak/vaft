"""Command-line interface for canonical and legacy FileDB operations."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vaft.cli filedb", description=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser(
        "audit", help="run a read-only legacy migration audit"
    )
    audit.add_argument("legacy_root", type=Path)
    audit.add_argument("--target-root", type=Path)

    relocate = subparsers.add_parser(
        "relocate",
        help="move a canonical root onto the lineage-bearing grammar",
        description=(
            "A canonical tree written before the grammar carried "
            "family/refinement/product resolves to nothing afterwards: the "
            "resolver looks under efit/magnetic/{shot} and the data is at "
            "efit/{shot}. This renames the subtrees. Nothing is read or "
            "written -- directories are moved -- and it is a dry run printing "
            "the plan unless --apply is given."
        ),
    )
    relocate.add_argument("root", type=Path)
    relocate.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Perform the moves. Refused when any destination is already "
            "occupied: which of two subtrees is authoritative is not a question "
            "a path rewriter can answer."
        ),
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "relocate":
        from ..database.filedb import relocate_filedb_grammar

        report = relocate_filedb_grammar(args.root, apply=args.apply)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        # A plan that cannot be applied is a result, not a crash -- but the exit
        # code has to say so, or an operator scripting the dry run reads the
        # collisions as an empty plan.
        return 0 if report.safe_to_apply else 1

    from ..database.filedb import audit_legacy_filedb

    report = audit_legacy_filedb(args.legacy_root, target_root=args.target_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
