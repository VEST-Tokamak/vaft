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

    migrate = subparsers.add_parser(
        "migrate-products",
        help="bring stored stage products onto their declared container",
        description=(
            "A deployment written before #813 holds products under a name the "
            "resolver no longer asks for, and eddy products that carry the "
            "whole diagnostics product they were solved against. This rewrites "
            "them. Neither half re-runs any physics. Unlike `relocate` this "
            "opens files, so each product is written to a temporary in its own "
            "output/ directory, verified, and renamed into place -- an "
            "interrupted run leaves every product wholly old or wholly new. "
            "The superseded originals are left behind for `sweep-products`. "
            "A dry run printing the plan unless --apply is given."
        ),
    )
    migrate.add_argument("root", type=Path)
    migrate.add_argument(
        "--stage",
        action="append",
        dest="stages",
        help=(
            "Limit to this stage; repeatable. Migrate and sweep one stage at a "
            "time when disk is tight: the whole tree at once needs headroom "
            "for every new product beside every old one."
        ),
    )
    migrate.add_argument(
        "--verify-shape",
        action="store_true",
        help=(
            "Open every already-migrated product and check it holds only what "
            "its stage owns. Makes the plan exact by inspection rather than by "
            "an argument about which writers exist; costs a full read pass. "
            "Required before sweeping."
        ),
    )
    migrate.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Perform the migration. Refused when any superseded original is "
            "newer than the product that superseded it -- that means a writer "
            "on the old code is still live."
        ),
    )

    sweep = subparsers.add_parser(
        "sweep-products",
        help="delete the originals a migration superseded (irreversible)",
        description=(
            "Run only after `migrate-products --apply` reports no failures and "
            "`--verify-shape` is clean. Deletion is irreversible, so it is a "
            "separate operator step and never part of the pipeline: everything "
            "up to it can be undone by removing the new product, because the "
            "original is still there. An eddy original whose shot has no "
            "diagnostics product is refused -- it is the only local copy of "
            "that shot's magnetics."
        ),
    )
    sweep.add_argument("root", type=Path)
    sweep.add_argument("--stage", action="append", dest="stages")
    sweep.add_argument("--apply", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "migrate-products":
        from ..database.product_migration import (
            audit_product_containers,
            migrate_product_containers,
        )

        if args.apply:
            report = migrate_product_containers(
                args.root, stages=args.stages, apply=True
            )
        else:
            report = audit_product_containers(
                args.root, stages=args.stages, verify_shape=args.verify_shape
            )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        # Same contract as `relocate`: a plan that cannot be applied is a
        # result, not a crash, but the exit code has to say so or an operator
        # scripting the dry run reads the conflicts as an empty plan.
        return 0 if report.safe_to_apply else 1

    if args.command == "sweep-products":
        from ..database.product_migration import (
            audit_product_containers,
            sweep_superseded_products,
        )

        # A fresh audit, not the migration's own report: a flag written by a
        # process that may have been killed is not evidence, and a deletion
        # tool must not write. `--verify-shape` is not optional here -- it is
        # what makes "this product already holds only what its stage owns"
        # a fact about the file rather than an inference from the file name.
        report = audit_product_containers(
            args.root, stages=args.stages, verify_shape=True
        )
        result = sweep_superseded_products(report, apply=args.apply)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not result["deletion_blocked"] else 1

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
