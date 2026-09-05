#!/usr/bin/env python3
"""Back-test the routine EFIT flux-loop ``broken`` list against automatic evidence (issue #295, step 1).

The routine configuration zero-weights six flux loops on every shot through a
hand-maintained combined-index list.  This lays the automatic evidence --
intrinsic signal quality (#189) and plasma-free agreement with the passive-wall
model (#190) -- beside that list, loop by loop and shot by shot, and records
where the two agree, where the automatic policy would reject a loop the list
keeps, and where a historical exclusion is not reproduced.

It is a report, not a gate: the exit status is 0 whatever the agreement, and
the default policy leaves model agreement report-only.  Nothing is written
into any ODS.

Usage::

    python backtest_flux_loops.py --output flux_loops.json --markdown flux_loops.md \\
        --packaged-samples

    python backtest_flux_loops.py --output flux_loops.json \\
        --case 39915=/path/to/39915/eddy/omas.json.gz \\
        --case 41524=/path/to/41524/diagnostics/omas.json.gz

The manual list defaults to ``constraints.broken`` of the routine
``config.yaml``; pass ``--manual-broken`` to test another.  As with the
vacuum benchmark, the #190 population lives in the VEST database, so the case
list is an argument and the invocation should be recorded with the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from vaft.validation.flux_loop_assessment import (
    NOT_AVAILABLE,
    REJECT_FOR_EFIT,
    FluxLoopPolicy,
    flux_loop_evidence,
)

ROUTINE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "automatic_pipeline_1_routine_data_processing"
    / "config.yaml"
)
PACKAGED_SHOTS = (39915, 41524, 41672)

AGREE_REJECT = "agree_reject"
AGREE_KEEP = "agree_keep"
FALSE_REJECTION = "false_rejection"
UNREPRODUCED_EXCLUSION = "unreproduced_exclusion"
AGREEMENTS = (AGREE_REJECT, AGREE_KEEP, FALSE_REJECTION, UNREPRODUCED_EXCLUSION, NOT_AVAILABLE)


def routine_manual_broken(config: Path = ROUTINE_CONFIG) -> list[int]:
    """The combined one-based indexes the routine pipeline hands to EFIT as broken."""
    import yaml

    with open(config, "r", encoding="utf-8") as handle:
        block = yaml.safe_load(handle).get("constraints", {})
    return sorted(int(item) for item in (block.get("broken") or []))


def agreement(state: str, manual_excluded: bool) -> str:
    """How one automatic state sits against the manual list."""
    if state == NOT_AVAILABLE:
        return NOT_AVAILABLE
    if state == REJECT_FOR_EFIT:
        return AGREE_REJECT if manual_excluded else FALSE_REJECTION
    return UNREPRODUCED_EXCLUSION if manual_excluded else AGREE_KEEP


def _load(path: Path) -> Any:
    import vaft.omas

    return vaft.omas.load(str(path))


def _parse_case(text: str) -> tuple[int | None, Path]:
    shot, separator, path = text.partition("=")
    if not separator:
        return None, Path(shot)
    return int(shot), Path(path)


def _packaged(shot: int) -> Any:
    if shot == 39915:
        from vaft.omas.sample import sample_ods

        return sample_ods()
    import vaft.data

    return _load(vaft.data.sample(shot, "omas"))


def backtest_case(ods: Any, *, shot: int | None, manual: list[int], policy: FluxLoopPolicy, benchmark: bool) -> dict[str, Any]:
    """One shot's evidence, annotated against the manual list."""
    evidence = flux_loop_evidence(ods, policy=policy, benchmark=benchmark)
    excluded = set(manual)
    rows = []
    for entry in evidence["assessments"]:
        manual_excluded = entry["combined_index_one_based"] in excluded
        rows.append({**entry, "manual_excluded": manual_excluded, "agreement": agreement(entry["state"], manual_excluded)})
    counts = {name: sum(1 for row in rows if row["agreement"] == name) for name in AGREEMENTS}
    model = evidence["model"]
    case = model.get("case") or {}
    return {
        "shot": shot,
        "window": evidence["window"],
        "nbprobe": evidence["nbprobe"],
        "model": {
            "consulted": model["consulted"],
            "available": model["available"],
            "reason": model["reason"],
            "case_type": case.get("case_type"),
            "coil_drive": case.get("coil_drive"),
            "plasma_free_evidence": case.get("plasma_free_evidence"),
            "static_model": case.get("static_model"),
            "solver": case.get("solver"),
        },
        "loops": rows,
        "summary": counts,
    }


def mapping_table(case: dict[str, Any], manual: list[int]) -> list[dict[str, Any]]:
    """Combined index -> ODS index -> field code -> name, so the list's positions are legible."""
    return [
        {
            "combined_index_one_based": row["combined_index_one_based"],
            "index": row["index"],
            "field_code": row["field_code"],
            "name": row["name"],
            "manual_excluded": row["combined_index_one_based"] in set(manual),
        }
        for row in case["loops"]
    ]


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "nan"
    return f"{number:.{digits}f}"


def markdown_report(payload: dict[str, Any]) -> str:
    """The per-loop table, one block per case, as Markdown."""
    lines = [
        "| shot | idx | combined | name | field | manual | state | usable frac (window) | events | improvement | norm. residual | correlation | wall authority | agreement |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case in payload["cases"]:
        for row in case["loops"]:
            model = row["model_agreement"] or {}
            source = row["source_validity"]
            lines.append(
                "| {shot} | {idx} | {combined} | {name} | {field} | {manual} | {state} | {frac} | {events} | {imp} | {nres} | {corr} | {auth} | {agree} |".format(
                    shot=case["shot"],
                    idx=row["index"],
                    combined=row["combined_index_one_based"],
                    name=row["name"],
                    field="-" if row["field_code"] is None else row["field_code"],
                    manual="excluded" if row["manual_excluded"] else "kept",
                    state=row["state"],
                    frac=_fmt(source["valid_fraction_in_window"]),
                    events=", ".join(source["events"]) or "-",
                    imp=_fmt(model.get("improvement")),
                    nres=_fmt(model.get("normalized_residual")),
                    corr=_fmt(model.get("correlation")),
                    auth=_fmt(model.get("wall_authority")),
                    agree=row["agreement"],
                )
            )
    lines.append("")
    lines.append("| shot | window | model | " + " | ".join(AGREEMENTS) + " |")
    lines.append("|---|---|---|" + "---|" * len(AGREEMENTS))
    for case in payload["cases"]:
        window = case["window"]
        model = case["model"]
        lines.append(
            "| {shot} | {window} | {model} | {counts} |".format(
                shot=case["shot"],
                window="-" if window is None else f"{window[0]:.4f}-{window[1]:.4f} s",
                model=("available" if model["available"] else f"unavailable: {model['reason']}") if model["consulted"] else "not consulted",
                counts=" | ".join(str(case["summary"][name]) for name in AGREEMENTS),
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", default=[], metavar="SHOT=PATH", help="A processed ODS to assess. Repeatable.")
    parser.add_argument("--packaged-sample", action="store_true", help="Also assess the packaged reference shot 39915.")
    parser.add_argument("--packaged-samples", action="store_true", help="Also assess every packaged shot (39915, 41524, 41672).")
    parser.add_argument("--output", required=True, type=Path, help="JSON report to write.")
    parser.add_argument("--markdown", type=Path, default=None, help="Also write the per-loop table as Markdown.")
    parser.add_argument(
        "--manual-broken",
        default=None,
        help="Comma-separated combined one-based indexes to test against; "
        "defaults to constraints.broken of the routine config.yaml.",
    )
    parser.add_argument("--no-benchmark", action="store_true", help="Skip the vacuum-model comparison; intrinsic quality only.")
    parser.add_argument("--min-valid-fraction", type=float, default=0.0, help="Policy: a loop must leave more than this fraction of the window usable.")
    parser.add_argument("--max-normalized-residual", type=float, default=None, help="Policy: model threshold; unset leaves the comparison report-only.")
    parser.add_argument("--min-correlation", type=float, default=None, help="Policy: model threshold; unset leaves the comparison report-only.")
    parser.add_argument("--min-wall-authority", type=float, default=0.1, help="Policy: below this the model comparison is not scored.")
    parser.add_argument("--reject-on-model-disagreement", action="store_true", help="Policy: a scored model failure rejects rather than marks suspect.")
    args = parser.parse_args(argv)

    manual = (
        sorted(int(item) for item in args.manual_broken.split(",") if item.strip())
        if args.manual_broken is not None
        else routine_manual_broken()
    )
    policy = FluxLoopPolicy(
        min_valid_fraction_in_window=args.min_valid_fraction,
        max_normalized_residual=args.max_normalized_residual,
        min_correlation=args.min_correlation,
        min_wall_authority_to_score=args.min_wall_authority,
        reject_on_model_disagreement=args.reject_on_model_disagreement,
    )

    sources: list[tuple[int | None, Any]] = []
    for entry in args.case:
        shot, path = _parse_case(entry)
        sources.append((shot, _load(path)))
    packaged = list(PACKAGED_SHOTS) if args.packaged_samples else ([39915] if args.packaged_sample else [])
    for shot in packaged:
        sources.append((shot, _packaged(shot)))
    if not sources:
        parser.error("no cases given; pass --case, --packaged-sample or --packaged-samples")

    cases = [
        backtest_case(ods, shot=shot, manual=manual, policy=policy, benchmark=not args.no_benchmark)
        for shot, ods in sources
    ]
    overall = {name: sum(case["summary"][name] for case in cases) for name in AGREEMENTS}
    payload = {
        "schema_version": 1,
        "configuration": {
            "manual_broken": manual,
            "manual_source": "argument" if args.manual_broken is not None else str(ROUTINE_CONFIG),
            "policy": policy.as_dict(),
            "benchmark": not args.no_benchmark,
            "mapping": mapping_table(cases[0], manual),
        },
        "cases": cases,
        "summary": overall,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(payload), encoding="utf-8")
    print(
        f"{len(cases)} case(s) against manual list {manual} -> {args.output}\n  "
        + ", ".join(f"{name} {overall[name]}" for name in AGREEMENTS)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
