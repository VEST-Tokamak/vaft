"""``python -m vaft.database.campaign {plan,benchmark,compare,verdict}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import compare_reports, lane_report
from .compare import compare_products, summarize
from .lanes import LaneState, disjoint, plan_lanes, read_shots, split_chunks
from .verdict import chunk_verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vaft.database.campaign", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="deal shots over lanes and split each into chunks")
    plan.add_argument("shots_file", type=Path)
    plan.add_argument("run_dir", type=Path)
    plan.add_argument("--lanes", type=int, required=True)
    plan.add_argument("--chunk", type=int, required=True)
    plan.add_argument("--exclude", type=Path, nargs="*", default=(), help="run dirs whose lanes' shots must not overlap")
    plan.add_argument("--pin", help="code SHA the run is pinned at (written to <run>/PIN_SHA for launch_lanes.sh)")

    bench = commands.add_parser("benchmark", help="cost report of one or more lanes")
    bench.add_argument("lanes", nargs="+", help="label=lane_dir or lane_dir")

    compare = commands.add_parser("compare", help="product equivalence of two FileDB roots")
    compare.add_argument("new_root", type=Path)
    compare.add_argument("ref_roots", type=Path, nargs="+")
    compare.add_argument("--shots-file", type=Path, required=True)
    compare.add_argument("--workflow-dir", type=Path, required=True)
    compare.add_argument("--out", type=Path)

    verdict = commands.add_parser("verdict", help="judge one chunk's Snakemake log")
    verdict.add_argument("log", type=Path)
    verdict.add_argument("--returncode", type=int, required=True)
    verdict.add_argument("--eligible", type=Path, required=True, help="the chunk's eligible_shots.json")

    args = parser.parse_args(argv)
    if args.command == "plan":
        shots = read_shots(args.shots_file)
        others = [LaneState(lane).shots() for run in args.exclude for lane in sorted(run.glob("lane*"))]
        overlap = disjoint(shots, *others)
        if overlap:
            print(f"{len(overlap)} shots are already in another run's lanes, e.g. {overlap[:5]}", file=sys.stderr)
            return 2
        if (args.run_dir / "shots_all.txt").exists() or any(args.run_dir.glob("lane*/chunks")):
            print(f"{args.run_dir} is already planned; its lanes' shots and chunks are fixed", file=sys.stderr)
            return 2
        args.run_dir.mkdir(parents=True, exist_ok=True)
        # What launch_lanes.sh reads to treat the run as planned (it re-deals lanes otherwise).
        (args.run_dir / "shots_all.txt").write_text("".join(f"{s}\n" for s in sorted(set(shots))), encoding="utf-8")
        if args.pin:
            (args.run_dir / "PIN_SHA").write_text(f"{args.pin}\n", encoding="utf-8")
        for index, lane_shots in enumerate(plan_lanes(shots, args.lanes)):
            lane = LaneState(args.run_dir / f"lane{index}")
            lane.ensure("chunk_size", str(args.chunk))
            (lane.path / "shots.txt").write_text("".join(f"{s}\n" for s in lane_shots), encoding="utf-8")
            split_chunks(lane_shots, args.chunk, lane.chunks_dir)
        print(json.dumps({"lanes": args.lanes, "shots": len(shots)}))
        return 0
    if args.command == "benchmark":
        reports = {}
        for item in args.lanes:
            label, _, path = item.rpartition("=")
            reports[label or path] = lane_report(path)
        print(json.dumps({"table": compare_reports(reports), "reports": reports}, indent=1, default=str))
        return 0
    if args.command == "compare":
        report = compare_products(args.new_root, args.ref_roots[0], read_shots(args.shots_file),
                                  workflow_dir=args.workflow_dir, ref_roots=args.ref_roots)
        if args.out:
            args.out.write_text(json.dumps({str(k): v for k, v in report.items()}, indent=1), encoding="utf-8")
        print(json.dumps(summarize(report), indent=1))
        return 0
    result = chunk_verdict(args.returncode, args.log.read_text(encoding="utf-8", errors="replace"),
                           has_eligible_verdict=args.eligible.exists())
    print(json.dumps({"done": result.done, "reason": result.reason, "failed_rules": list(result.failed_rules)}))
    return 0 if result.done else 1


if __name__ == "__main__":
    raise SystemExit(main())
