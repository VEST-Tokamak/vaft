"""Compare local ODS products using the versioned VAFT parity policy."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vaft.cli compare-ods", description=__doc__
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--tolerances", type=Path)
    parser.add_argument("--path", action="append", dest="paths")
    parser.add_argument(
        "--scope",
        choices=("union", "reference", "intersection"),
        default="union",
    )
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    from ..omas import load
    from ..omas.comparison import compare_ods, write_comparison_reports

    comparison = compare_ods(
        load(args.reference),
        load(args.candidate),
        policy=args.tolerances,
        paths=args.paths,
        scope=args.scope,
        reference_label=str(args.reference),
        candidate_label=str(args.candidate),
    )
    write_comparison_reports(
        comparison,
        json_path=args.json_report,
        markdown_path=args.markdown_report,
    )
    print(comparison.to_markdown())
    return 0 if comparison.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
