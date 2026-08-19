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
    args = parser.parse_args(list(argv) if argv is not None else None)
    from ..database.filedb import audit_legacy_filedb

    report = audit_legacy_filedb(args.legacy_root, target_root=args.target_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
