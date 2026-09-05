#!/usr/bin/env python3
"""Run the plasma-free magnetic-response benchmark over a set of shots (issue #190).

A development/reference qualification tool, not a production stage.  The
routine per-shot QA (#139) keeps its compact channel subset and its place in
the Snakemake pipeline; this walks many carefully selected plasma-free cases
with the *full* usable channel set, so a discrepancy can be attributed to a
sensor, a coil, the wall model or a machine era rather than merely noticed.

Each case is one processed ODS -- an eddy-stage or diagnostics-stage product,
or the packaged sample.  The wall currents are re-solved from measured PF
currents alone, so a case is constructible whether or not the shot later formed
a plasma, and the routine stage's plasma-filament forcing cannot contaminate
the result.

Usage::

    python run_vacuum_benchmark.py --output benchmark.json \\
        --case 39915=/path/to/39915/eddy/omas.json.gz \\
        --case 41524=/path/to/41524/diagnostics/omas.json.gz

    # the packaged sample, for a quick check that the chain works
    python run_vacuum_benchmark.py --output benchmark.json --packaged-sample

The case list is an argument rather than a committed table: the source shots
live in the VEST database, not in this repository, and #190 asks that the
selection be reproducible rather than that it be vendored.  Record the exact
invocation alongside the output -- the manifest carries the static-model
revision each residual was measured against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from vaft.validation.vacuum_benchmark import (
    BenchmarkError,
    aggregate_benchmark,
    run_benchmark_case,
)


def _load(path: Path) -> Any:
    from omas import load_omas_json

    return load_omas_json(str(path), consistency_check=False)


def _machine_era(shot: int | None) -> str | None:
    if shot is None:
        return None
    from vaft.omas.vest_upstream import machine_era_for_shot

    try:
        return machine_era_for_shot(int(shot)).name
    except ValueError:
        return None


def _parse_case(text: str) -> tuple[int | None, Path]:
    shot, separator, path = text.partition("=")
    if not separator:
        return None, Path(shot)
    return int(shot), Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="SHOT=PATH",
        help="A processed ODS to benchmark. Repeatable. `SHOT=` is optional but "
        "makes the manifest and the cross-shot aggregation legible.",
    )
    parser.add_argument(
        "--packaged-sample",
        action="store_true",
        help="Also benchmark the packaged reference shot, which needs no database access.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Aggregate JSON to write.")
    parser.add_argument(
        "--resistance-scale",
        type=float,
        default=1.0,
        help="One global factor on every pf_passive resistance, for the #117 "
        "sensitivity study. Never fit per loop.",
    )
    parser.add_argument(
        "--n-tau",
        type=float,
        default=3.0,
        help="Wall time constants of solver history required before the validation window opens.",
    )
    parser.add_argument(
        "--per-family",
        type=int,
        default=None,
        help="Channels kept per family; the default of every usable channel is "
        "what model qualification needs.",
    )
    args = parser.parse_args(argv)

    sources: list[tuple[int | None, Any]] = []
    for entry in args.case:
        shot, path = _parse_case(entry)
        sources.append((shot, _load(path)))
    if args.packaged_sample:
        from vaft.omas.sample import sample_ods

        sources.append((39915, sample_ods()))
    if not sources:
        parser.error("no cases given; pass --case or --packaged-sample")

    cases: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for shot, ods in sources:
        try:
            cases.append(
                run_benchmark_case(
                    ods,
                    shot=shot,
                    machine_era=_machine_era(shot),
                    per_family=args.per_family,
                    resistance_scale=args.resistance_scale,
                    n_tau=args.n_tau,
                )
            )
        except BenchmarkError as error:
            # A shot that cannot support a case is reported, not skipped: which
            # shots are ineligible, and why, is part of the benchmark's result.
            rejected.append({"shot": shot, "reason": str(error)})

    payload = {
        # 2 (#409): case records and their plasma_free_evidence changed shape.
        "schema_version": 2,
        "configuration": {
            "resistance_scale": args.resistance_scale,
            "n_tau": args.n_tau,
            "per_family": args.per_family,
        },
        "rejected": rejected,
        "aggregate": aggregate_benchmark(cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )
    summary = payload["aggregate"].get("summary", {})
    print(
        f"{len(cases)} case(s), {len(rejected)} rejected -> {args.output}\n"
        f"  median improvement {summary.get('median_improvement', float('nan')):.3f}, "
        f"improved fraction {summary.get('improved_fraction', float('nan')):.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
